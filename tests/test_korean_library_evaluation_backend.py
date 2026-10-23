"""New evaluations prefer verified CUDA; resumes keep their existing backend."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from korean_openwakeword.frontend import PINNED_ASSETS


@pytest.fixture
def backend(monkeypatch):
    module = importlib.import_module("korean_openwakeword.evaluation_backend")
    monkeypatch.setattr(module, "_runtime_environment", lambda root: {})
    monkeypatch.setattr(module, "_loaded_runtime_conflict", lambda environment: None)
    return module


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    manifest = {"members": [{"path": f"model_{number}.onnx"} for number in range(3)],
                "aggregation": "ordered_float64_mean_to_float32",
                "frontend": [{"path": name, "sha256": value} for name, value in PINNED_ASSETS.items()]}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_new_auto_run_prefers_a_real_verified_cuda_backend(backend, monkeypatch, bundle):
    probes = []
    def probe(directory, *, environment):
        probes.append((directory, environment))
        return {"available": True, "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
    monkeypatch.setattr(backend, "probe_cuda_backend", probe)
    result = backend.resolve_evaluation_backend(bundle=bundle)
    assert result["requested_backend"] == "auto" and result["backend"] == "cuda"
    assert result["numerical_policy"] == backend.NUMERICAL_POLICY and len(probes) == 1


def test_auto_without_cuda_records_a_cpu_reason_but_explicit_cuda_fails(backend, monkeypatch, bundle):
    monkeypatch.setattr(backend, "probe_cuda_backend", lambda *args, **kwargs:
        {"available": False, "reason_code": "missing_cuda_dlls", "reason": "GPU DLLs are missing"})
    result = backend.resolve_evaluation_backend(bundle=bundle)
    assert result["backend"] == "cpu" and result["reason_code"] == "missing_cuda_dlls"
    assert result["numerical_policy"] is None and result["runtime_environment"] == {}
    with pytest.raises(RuntimeError, match="GPU DLLs"):
        backend.resolve_evaluation_backend("cuda", bundle=bundle)


def test_invalid_managed_runtime_is_an_explicit_reason_instead_of_a_silent_change(backend, monkeypatch, bundle):
    def invalid(root):
        raise ValueError("Changed runtime receipt")
    monkeypatch.setattr(backend, "_runtime_environment", invalid)
    result = backend.resolve_evaluation_backend(bundle=bundle)
    assert result["backend"] == "cpu" and result["reason_code"] == "cuda_runtime_configuration_failed"
    assert result["detail"] == "Changed runtime receipt"
    with pytest.raises(RuntimeError, match="Changed runtime receipt"):
        backend.resolve_evaluation_backend("cuda", bundle=bundle)


def test_successful_probe_preserves_the_managed_runtime_plan(backend, monkeypatch, bundle):
    environment = {"KOWW_GPU_RUNTIME_PATH": "managed-runtime", "KOWW_GPU_DLL_PATHS": "managed-dlls"}
    monkeypatch.setattr(backend, "_runtime_environment", lambda root: dict(environment))
    def probe(directory, *, environment):
        assert environment == {"KOWW_GPU_RUNTIME_PATH": "managed-runtime", "KOWW_GPU_DLL_PATHS": "managed-dlls"}
        return {"available": True}
    monkeypatch.setattr(backend, "probe_cuda_backend", probe)
    result = backend.resolve_evaluation_backend(bundle=bundle)
    assert result["runtime_environment"] == environment and result["backend"] == "cuda"


@pytest.mark.parametrize("requested,resume,reason", [("cpu", None, "explicit_cpu"),
    ("auto", "cpu", "resume_cpu_reference"), ("cpu", "cpu", "resume_cpu_reference")])
def test_cpu_selection_and_legacy_cpu_resumes_do_not_probe_gpu(backend, monkeypatch, requested, resume, reason):
    monkeypatch.setattr(backend, "probe_cuda_backend", lambda *args, **kwargs: pytest.fail("CPU must not probe GPU"))
    result = backend.resolve_evaluation_backend(requested, resume_backend=resume)
    assert result["backend"] == "cpu" and result["reason_code"] == reason


def test_cuda_resume_cannot_silently_become_cpu(backend, monkeypatch, bundle):
    monkeypatch.setattr(backend, "probe_cuda_backend", lambda *args, **kwargs:
        {"available": False, "reason": "unavailable"})
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.resolve_evaluation_backend("auto", bundle=bundle, resume_backend="cuda")
    monkeypatch.setattr(backend, "probe_cuda_backend", lambda *args, **kwargs: {"available": True})
    result = backend.resolve_evaluation_backend("auto", bundle=bundle, resume_backend="cuda")
    assert result["backend"] == "cuda" and result["reason_code"] == "resume_cuda"


@pytest.mark.parametrize("requested,resume", [("cuda", "cpu"), ("cpu", "cuda")])
def test_explicit_resume_backend_changes_are_rejected(backend, requested, resume):
    with pytest.raises(ValueError, match="cannot change"):
        backend.resolve_evaluation_backend(requested, resume_backend=resume)


def test_independent_final_keeps_its_frozen_cpu_protocol(backend, monkeypatch):
    monkeypatch.setattr(backend, "probe_cuda_backend", lambda *args, **kwargs: pytest.fail("Final must not probe GPU"))
    assert backend.resolve_evaluation_backend(stage="final")["reason_code"] == "final_cpu_reference"
    for requested, resume in (("cuda", None), ("auto", "cuda")):
        with pytest.raises(ValueError, match="Final"):
            backend.resolve_evaluation_backend(requested, stage="final", resume_backend=resume)


@pytest.mark.parametrize("change", ["single", "frontend"])
def test_unqualified_bundle_stays_cpu_and_cannot_be_forced_to_cuda(backend, bundle, change):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "single":
        manifest.update(members=manifest["members"][:1], aggregation="single")
    else:
        manifest["frontend"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert backend.resolve_evaluation_backend(bundle=bundle)["reason_code"] == "unsupported_gpu_bundle"
    with pytest.raises(ValueError):
        backend.resolve_evaluation_backend("cuda", bundle=bundle)


def test_invalid_selection_and_stage_fail_before_any_probe(backend):
    for options in ({"requested": "gpu"}, {"stage": "test"}, {"resume_backend": "auto"}):
        with pytest.raises(ValueError):
            backend.resolve_evaluation_backend(**options)


def test_loaded_cpu_runtime_is_not_replaced_in_a_python_process(monkeypatch, bundle, tmp_path):
    module = importlib.import_module("korean_openwakeword.evaluation_backend")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__file__=str(tmp_path / "cpu" / "__init__.py"),
        get_available_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setattr(module, "_runtime_environment", lambda root: {"KOWW_GPU_RUNTIME_PATH": str(tmp_path / "gpu")})
    result = module.resolve_evaluation_backend(bundle=bundle)
    assert result["backend"] == "cpu" and result["reason_code"] == "runtime_already_loaded"
    with pytest.raises(RuntimeError, match="ONNX Runtime"):
        module.resolve_evaluation_backend("cuda", bundle=bundle)


def test_explicit_runtime_overrides_managed_installation(monkeypatch, tmp_path):
    module = importlib.import_module("korean_openwakeword.evaluation_backend")
    managed = SimpleNamespace(gpu_runtime_environment=lambda root: pytest.fail("Explicit runtime overrides managed discovery"))
    monkeypatch.setitem(sys.modules, "korean_openwakeword.gpu_runtime_install", managed)
    monkeypatch.setenv("KOWW_GPU_RUNTIME_PATH", str(tmp_path / "chosen"))
    monkeypatch.setenv("KOWW_GPU_DLL_PATHS", str(tmp_path / "dlls"))
    assert module._runtime_environment(tmp_path) == {"KOWW_GPU_RUNTIME_PATH": str(tmp_path / "chosen"),
                                                    "KOWW_GPU_DLL_PATHS": str(tmp_path / "dlls")}


def test_managed_runtime_is_discovered_without_mutating_parent_environment(monkeypatch, tmp_path):
    module = importlib.import_module("korean_openwakeword.evaluation_backend")
    monkeypatch.delenv("KOWW_GPU_RUNTIME_PATH", raising=False)
    monkeypatch.delenv("KOWW_GPU_DLL_PATHS", raising=False)
    expected = {"KOWW_GPU_RUNTIME_PATH": str(tmp_path / "managed")}
    monkeypatch.setitem(sys.modules, "korean_openwakeword.gpu_runtime_install",
        SimpleNamespace(gpu_runtime_environment=lambda root: dict(expected)))
    assert module._runtime_environment(tmp_path) == expected
    assert "KOWW_GPU_RUNTIME_PATH" not in os.environ


def test_probe_uses_an_isolated_runtime_and_really_executes_a_worker(monkeypatch, bundle, tmp_path):
    module = importlib.import_module("korean_openwakeword.evaluation_backend")
    target = str(tmp_path / "gpu")
    def run(command, **options):
        assert command == [sys.executable, "-B", "-m", "korean_openwakeword.evaluation_backend", "--probe-cuda", str(bundle)]
        assert options["env"]["PYTHONPATH"].split(os.pathsep)[0] == target
        assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert options["timeout"] == 30 and options["capture_output"]
        return SimpleNamespace(returncode=0, stdout='unrelated initialization line\n{"available": true}\n', stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.probe_cuda_backend(bundle, environment={"KOWW_GPU_RUNTIME_PATH": target})["available"]


@pytest.mark.parametrize("failure", ["timeout", "start", "invalid_json", "invalid_result", "crash"])
def test_probe_failures_are_visible_cpu_reasons(monkeypatch, bundle, failure):
    module = importlib.import_module("korean_openwakeword.evaluation_backend")
    def run(command, **options):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if failure == "start":
            raise OSError("cannot start")
        return SimpleNamespace(returncode=1 if failure == "crash" else 0,
            stdout="not-json" if failure == "invalid_json" else '{"available": "yes"}' if failure == "invalid_result" else '{"available": true}', stderr="failure")
    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.probe_cuda_backend(bundle)
    assert result["available"] is False and result["reason_code"].startswith("cuda_probe_")


def test_gpu_runtime_environment_restores_paths_on_exception(monkeypatch):
    from korean_openwakeword.gpu_evaluation import evaluation_runtime_environment
    monkeypatch.delenv("KOWW_GPU_RUNTIME_PATH", raising=False)
    monkeypatch.setenv("PATH", "original-path")
    with pytest.raises(RuntimeError):
        with evaluation_runtime_environment({"KOWW_GPU_RUNTIME_PATH": "chosen-runtime"}):
            assert os.environ["KOWW_GPU_RUNTIME_PATH"] == "chosen-runtime"
            os.environ["PATH"] = "dlls;original-path"
            raise RuntimeError("stop")
    assert "KOWW_GPU_RUNTIME_PATH" not in os.environ and os.environ["PATH"] == "original-path"
    with pytest.raises(ValueError):
        with evaluation_runtime_environment({"HOME": "unexpected"}):
            pytest.fail("Invalid runtime environment cannot enter")


@pytest.mark.parametrize("original_flag,original_environment", [(False, None), (False, "0"), (True, "1")])
def test_gpu_runtime_prevents_inventory_bytecode_writes_and_restores_the_callers_state(monkeypatch, original_flag, original_environment):
    from korean_openwakeword.gpu_evaluation import evaluation_runtime_environment
    monkeypatch.setattr(sys, "dont_write_bytecode", original_flag)
    if original_environment is None:
        monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    else:
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", original_environment)
    with pytest.raises(RuntimeError, match="restore"):
        with evaluation_runtime_environment({}):
            assert sys.dont_write_bytecode is True
            assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
            with evaluation_runtime_environment({}):
                assert sys.dont_write_bytecode is True
            assert sys.dont_write_bytecode is True
            raise RuntimeError("restore")
    assert sys.dont_write_bytecode is original_flag
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == original_environment
