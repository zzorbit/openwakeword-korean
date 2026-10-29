"""Isolated CUDA installation safety, using tiny fake wheels and no downloads."""
import json
import importlib.util
import os
from pathlib import Path
import py_compile
import subprocess
import sys
from types import SimpleNamespace

import pytest

from korean_openwakeword import gpu_runtime_install as runtime


def _fake_payload(target):
    files = {
        "onnxruntime/__init__.py": b"# fake isolated GPU package\n",
        "onnxruntime/capi/onnxruntime_pybind11_state.pyd": b"fake pybind",
        "onnxruntime/capi/onnxruntime_providers_cuda.dll": b"fake provider",
        "nvidia/cu13/bin/x86_64/cublas64_13.dll": b"fake cublas",
        "nvidia/cu13/bin/x86_64/cublasLt64_13.dll": b"fake cublaslt",
        "nvidia/cu13/bin/x86_64/cudart64_13.dll": b"fake cudart",
        "nvidia/cudnn/bin/cudnn64_9.dll": b"fake cudnn",
    }
    for name, version in runtime.PINS.items():
        files[f"{name.replace('-', '_')}-{version}.dist-info/METADATA"] = (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n").encode()
    for name, content in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _probe(target, environment):
    return {"version": "1.31.0", "module": str(target / "onnxruntime/__init__.py"),
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_session_validated": True, "cpu_fallback_disabled": True,
            "operators": ["MatMul", "Conv"]}


@pytest.fixture
def managed(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.delenv("KOWW_CACHE_ROOT", raising=False)
    monkeypatch.setattr(runtime, "_managed_reason", lambda: None)
    # Simulated Windows runtime state must not change the host interpreter
    # platform used by sysconfig and other imported modules on Linux.
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(**vars(sys)))
    monkeypatch.setattr(runtime.sys, "prefix", str(root / ".venv"))
    monkeypatch.setattr(runtime.sys, "base_prefix", str(tmp_path / "system_python"))
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda path: SimpleNamespace(free=runtime.MINIMUM_FREE_BYTES))
    state = SimpleNamespace(root=root, pip=[], probes=[], payload_hook=None)

    def run(command, **kwargs):
        assert command[1:4] == ["-m", "pip", "install"], "Mock forbids all real processes"
        state.pip.append({"command": command, "kwargs": kwargs, "environment": dict(os.environ)})
        target = Path(command[command.index("--target") + 1])
        _fake_payload(target)
        if state.payload_hook:
            state.payload_hook(target)
        return subprocess.CompletedProcess(command, 0)

    def probe(target, environment):
        state.probes.append((target, dict(environment)))
        return _probe(target, environment)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime, "_probe_runtime", probe)
    return state


def test_fresh_setup_isolated_target_scoped_storage_and_verified_reuse(managed):
    before_environment = dict(os.environ)
    before_path = list(sys.path)
    before_ort = sys.modules.get("onnxruntime")
    receipt = runtime.setup_gpu_runtime(managed.root)
    target = Path(receipt["path"])
    assert receipt["state"] == "completed" and not receipt["reused"]
    assert receipt["versions"] == runtime.PINS
    assert target == managed.root / "cache/gpu_runtime" / runtime.RUNTIME_NAME
    assert receipt["artifacts"] and target.joinpath("receipt.json").is_file()
    assert len(managed.pip) == 1 and len(managed.probes) == 2
    command = managed.pip[0]["command"]
    assert all(option in command for option in ("--no-deps", "--no-compile", "--only-binary=:all:", "--target"))
    assert "onnxruntime==1.31.0" not in command
    assert {arg for arg in command if "==" in arg} == {name + "==" + version for name, version in runtime.PINS.items()}
    staging = Path(command[command.index("--target") + 1])
    assert staging.parent == target.parent and staging != target and not staging.exists()
    assert [path for path, _ in managed.probes] == [staging, target]
    assert receipt["cuda_session_probe"]["module"] == str(target / "onnxruntime/__init__.py")
    assert receipt["staging_cuda_session_probe"]["module"] == str(staging / "onnxruntime/__init__.py")
    assert managed.pip[0]["environment"]["PIP_CACHE_DIR"] == str(managed.root / "cache/pip")
    assert managed.pip[0]["environment"]["TEMP"] == str(managed.root / "cache/tmp")
    assert os.environ == before_environment and sys.path == before_path and sys.modules.get("onnxruntime") is before_ort
    environment = runtime.gpu_runtime_environment(managed.root)
    assert environment["KOWW_GPU_RUNTIME_PATH"] == str(target)
    assert set(environment["KOWW_GPU_DLL_PATHS"].split(os.pathsep)) == {
        str(target / "nvidia/cu13/bin/x86_64"), str(target / "nvidia/cudnn/bin")}
    saved = (target / "receipt.json").read_bytes()
    reused = runtime.setup_gpu_runtime(managed.root)
    assert reused["reused"] and reused["current_probe"]["cuda_session_validated"]
    assert len(managed.pip) == 1 and len(managed.probes) == 3
    assert (target / "receipt.json").read_bytes() == saved


@pytest.mark.parametrize("damage", ["changed_bytes", "native_bytes", "unrecorded_code", "missing_file", "receipt_version"])
def test_changed_existing_runtime_refuses_overwrite_and_preserves_bytes(managed, damage):
    receipt = runtime.setup_gpu_runtime(managed.root)
    target = Path(receipt["path"])
    path = target / "onnxruntime/__init__.py"
    if damage == "changed_bytes":
        path.write_bytes(b"changed on purpose")
    elif damage == "native_bytes":
        (target / "onnxruntime/capi/onnxruntime_providers_cuda.dll").write_bytes(b"changed native payload")
    elif damage == "unrecorded_code":
        (target / "onnxruntime/extra.py").write_text("injected = True", encoding="utf-8")
    elif damage == "missing_file":
        path.unlink()
    else:
        receipt["versions"]["onnxruntime-gpu"] = "0.0.0"
        (target / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    snapshot = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        runtime.setup_gpu_runtime(managed.root)
    assert len(managed.pip) == 1 and len(managed.probes) == 2
    assert snapshot == {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob("*") if p.is_file()}


def test_normal_import_bytecode_and_pre_rename_filenames_do_not_break_reuse(managed, monkeypatch):
    receipt = runtime.setup_gpu_runtime(managed.root)
    target = Path(receipt["path"])
    source = target / "onnxruntime/__init__.py"
    # Import the harmless fake source without loading or replacing real ORT.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    spec = importlib.util.spec_from_file_location("fake_managed_runtime", source)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    cached = Path(importlib.util.cache_from_source(str(source)))
    assert cached.is_file()
    assert runtime.setup_gpu_runtime(managed.root)["reused"]
    py_compile.compile(str(source), cfile=str(cached), dfile="previous_staging/onnxruntime/__init__.py", doraise=True)
    assert runtime.setup_gpu_runtime(managed.root)["reused"]
    assert len(managed.pip) == 1


def test_legacy_completed_staging_probe_receipt_remains_immutable_on_verified_reuse(managed):
    receipt = runtime.setup_gpu_runtime(managed.root)
    target = Path(receipt["path"])
    receipt["cuda_session_probe"] = receipt.pop("staging_cuda_session_probe")
    receipt["probe_path_scope"] = "Probe module path records the fresh staging directory before atomic publication; artifact hashes bind the published payload."
    path = target / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    before = path.read_bytes()
    reused = runtime.setup_gpu_runtime(managed.root)
    assert reused["reused"] and reused["current_probe"]["module"] == str(target / "onnxruntime/__init__.py")
    assert path.read_bytes() == before and len(managed.pip) == 1


def test_unrelated_or_changed_generated_bytecode_is_not_trusted(managed):
    receipt = runtime.setup_gpu_runtime(managed.root)
    target = Path(receipt["path"])
    source = target / "onnxruntime/__init__.py"
    cached = Path(importlib.util.cache_from_source(str(source)))
    # A generated-looking cache must represent the hash-bound source.
    unrelated = managed.root / "unrelated.py"
    unrelated.write_text("different_code = True\n", encoding="utf-8")
    cached.parent.mkdir(exist_ok=True)
    py_compile.compile(str(unrelated), cfile=str(cached), dfile=str(source), doraise=True)
    with pytest.raises(ValueError, match="unverified extra"):
        runtime.setup_gpu_runtime(managed.root)
    assert cached.exists() and len(managed.pip) == 1


def test_incomplete_existing_runtime_is_preserved(managed):
    target = runtime._target(managed.root)
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_bytes(b"unfinished previous installation")
    with pytest.raises(ValueError, match="preserved"):
        runtime.setup_gpu_runtime(managed.root)
    assert sentinel.read_bytes() == b"unfinished previous installation"
    assert not managed.pip and not managed.probes


@pytest.mark.parametrize("damage", ["missing_cuda", "wrong_version", "cpu_collision"])
def test_bad_downloaded_layout_never_publishes_canonical_target(managed, damage):
    def damage_payload(target):
        if damage == "missing_cuda":
            (target / "nvidia/cu13/bin/x86_64/cudart64_13.dll").unlink()
        elif damage == "wrong_version":
            metadata = next(target.glob("onnxruntime_gpu-*.dist-info/METADATA"))
            metadata.write_text("Name: onnxruntime-gpu\nVersion: 1.30.0\n", encoding="utf-8")
        else:
            metadata = target / "onnxruntime-1.31.0.dist-info/METADATA"
            metadata.parent.mkdir()
            metadata.write_text("Name: onnxruntime\nVersion: 1.31.0\n", encoding="utf-8")
    managed.payload_hook = damage_payload
    with pytest.raises(ValueError):
        runtime.setup_gpu_runtime(managed.root)
    target = runtime._target(managed.root)
    assert not target.exists() and not managed.probes
    assert len(list(target.parent.glob(runtime.RUNTIME_NAME + ".installing_*"))) == 1


def test_failed_native_probe_preserves_staging_without_publishing(managed, monkeypatch):
    def fail_probe(*args):
        raise ValueError("CUDA provider could not initialize")
    monkeypatch.setattr(runtime, "_probe_runtime", fail_probe)
    with pytest.raises(ValueError, match="initialize"):
        runtime.setup_gpu_runtime(managed.root)
    target = runtime._target(managed.root)
    assert not target.exists()
    staging = list(target.parent.glob(runtime.RUNTIME_NAME + ".installing_*"))
    assert len(staging) == 1 and (staging[0] / "onnxruntime/__init__.py").is_file()
    assert runtime.gpu_runtime_environment(managed.root) == {}


@pytest.mark.parametrize("failure", ["permission", "native"])
def test_failed_published_path_probe_cannot_claim_completed_runtime(managed, monkeypatch, failure):
    target = runtime._target(managed.root)
    calls = []

    def probe(path, environment):
        calls.append(path)
        if path == target:
            if failure == "permission":
                raise PermissionError(13, "Access denied", str(path / "onnxruntime/__init__.py"))
            raise ValueError("Final CUDA path could not initialize")
        return _probe(path, environment)

    monkeypatch.setattr(runtime, "_probe_runtime", probe)
    with pytest.raises(RuntimeError, match="Published GPU runtime validation failed"):
        runtime.setup_gpu_runtime(managed.root)
    assert len(calls) == 2 and calls[1] == target
    receipt = json.loads((target / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "publishing"
    assert (target / "onnxruntime/__init__.py").is_file()
    assert runtime.gpu_runtime_environment(managed.root) == {}
    incoming = list(target.parent.glob(runtime.RUNTIME_NAME + ".installing_*"))
    assert len(incoming) == (1 if sys.platform == "win32" else 0)
    before = (target / "receipt.json").read_bytes()
    with pytest.raises(ValueError, match="unverified; it was preserved"):
        runtime.setup_gpu_runtime(managed.root)
    assert len(managed.pip) == 1 and (target / "receipt.json").read_bytes() == before


def test_windows_publication_creates_fresh_verified_files_without_temp_metadata(managed, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    target = runtime._target(managed.root)
    target.parent.mkdir(parents=True)
    incoming = target.with_name(target.name + ".installing_fixture")
    incoming.mkdir()
    _fake_payload(incoming)
    original = incoming / "onnxruntime/__init__.py"
    os.utime(original, (1000000000, 1000000000))
    source_before = {p.relative_to(incoming): p.read_bytes() for p in incoming.rglob("*") if p.is_file()}
    publication = runtime._publication_copy(incoming, target, runtime._artifacts(incoming))
    copied = publication / "onnxruntime/__init__.py"
    assert copied.stat().st_mtime != original.stat().st_mtime
    assert copied.stat().st_ino != original.stat().st_ino
    assert {p.relative_to(publication): p.read_bytes() for p in publication.rglob("*") if p.is_file()} == source_before
    assert {p.relative_to(incoming): p.read_bytes() for p in incoming.rglob("*") if p.is_file()} == source_before
    assert publication.parent == target.parent and not target.exists()


def test_corrupted_publication_copy_is_not_published_and_preserves_incoming(managed, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    original_copy = runtime.shutil.copyfile

    def corrupt(source, destination):
        original_copy(source, destination)
        Path(destination).write_bytes(b"corrupted publication")
    monkeypatch.setattr(runtime.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="copy differs"):
        runtime.setup_gpu_runtime(managed.root)
    target = runtime._target(managed.root)
    assert not target.exists()
    incoming, = target.parent.glob(target.name + ".installing_*")
    assert (incoming / "onnxruntime/__init__.py").read_bytes() == b"# fake isolated GPU package\n"
    assert len(list(target.parent.glob(target.name + ".publishing_*"))) == 1


def test_windows_publication_checks_remaining_space_before_creating_copy(managed, monkeypatch):
    target = runtime._target(managed.root)
    target.parent.mkdir(parents=True)
    incoming = target.with_name(target.name + ".installing_fixture")
    incoming.mkdir()
    _fake_payload(incoming)
    artifacts = runtime._artifacts(incoming)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="publication copy"):
        runtime._publication_copy(incoming, target, artifacts)
    assert (incoming / "onnxruntime/__init__.py").is_file()
    assert not list(target.parent.glob(target.name + ".publishing_*"))


def test_successful_windows_setup_cleans_only_its_new_incoming_directory(managed, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    target = runtime._target(managed.root)
    old = target.with_name(target.name + ".installing_previous_failed_attempt")
    old.mkdir(parents=True)
    sentinel = old / "keep.txt"
    sentinel.write_bytes(b"previous failed installation")
    receipt = runtime.setup_gpu_runtime(managed.root)
    assert receipt["publication_policy"] == "fresh_files_inherit_project_cache_acl"
    assert sentinel.read_bytes() == b"previous failed installation"
    assert list(target.parent.glob(target.name + ".installing_*")) == [old]


def test_publication_cleanup_failure_warns_but_keeps_qualified_runtime(managed, monkeypatch, capsys):
    monkeypatch.setattr(runtime.sys, "platform", "win32")

    def cannot_cleanup(*args):
        raise PermissionError("Cleanup denied")
    monkeypatch.setattr(runtime, "_remove_new_staging", cannot_cleanup)
    receipt = runtime.setup_gpu_runtime(managed.root)
    target = Path(receipt["path"])
    assert receipt["state"] == "completed" and "Cleanup denied" in receipt["staging_cleanup_warning"]
    assert runtime.gpu_runtime_environment(managed.root)["KOWW_GPU_RUNTIME_PATH"] == str(target)
    assert len(list(target.parent.glob(target.name + ".installing_*"))) == 1
    assert "임시 설치 폴더" in capsys.readouterr().out


def test_insufficient_space_and_system_python_do_not_install(managed, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda path: SimpleNamespace(free=runtime.MINIMUM_FREE_BYTES - 1))
    with pytest.raises(ValueError, match="space"):
        runtime.setup_gpu_runtime(managed.root)
    assert not managed.pip and not runtime._target(managed.root).exists()
    monkeypatch.setattr(runtime.sys, "prefix", runtime.sys.base_prefix)
    with pytest.raises(ValueError, match="virtual environment"):
        runtime.setup_gpu_runtime(managed.root)
    assert not managed.pip


def test_cache_override_is_used_and_parent_environment_restored(managed, monkeypatch, tmp_path):
    shared = tmp_path / "selected_shared_cache"
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(shared))
    monkeypatch.setenv("TEMP", "previous-temp")
    receipt = runtime.setup_gpu_runtime(managed.root)
    assert Path(receipt["path"]) == shared / "gpu_runtime" / runtime.RUNTIME_NAME
    assert managed.pip[0]["environment"]["TEMP"] == str(shared / "tmp")
    assert os.environ["TEMP"] == "previous-temp" and os.environ["KOWW_CACHE_ROOT"] == str(shared)


def test_codex_git_and_base_package_storage_are_rejected(managed, monkeypatch, tmp_path):
    for root in (tmp_path / ".codex/data", tmp_path / ".git/data"):
        with pytest.raises(ValueError):
            runtime.setup_gpu_runtime(root)
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / ".codex/cache"))
    with pytest.raises(ValueError):
        runtime.setup_gpu_runtime(managed.root)
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / ".git/cache"))
    with pytest.raises(ValueError):
        runtime.setup_gpu_runtime(managed.root)
    package_dir = tmp_path / "site-packages"
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(package_dir / "cache"))
    monkeypatch.setattr(runtime.sysconfig, "get_paths", lambda: {"purelib": str(package_dir), "platlib": str(package_dir)})
    with pytest.raises(ValueError, match="site-packages"):
        runtime.setup_gpu_runtime(managed.root)
    assert not managed.pip


def test_unsupported_platform_does_not_install_or_advertise_managed_runtime(managed, monkeypatch):
    monkeypatch.setattr(runtime, "_managed_reason", lambda: "Existing Linux provider is supported; managed Windows payload is not.")
    receipt = runtime.setup_gpu_runtime(managed.root)
    assert receipt["state"] == "unsupported" and not receipt["managed"]
    assert runtime.gpu_runtime_environment(managed.root) == {} and not managed.pip


def test_locator_ignores_malformed_receipt(managed):
    target = runtime._target(managed.root)
    target.mkdir(parents=True)
    for text in ("{", "[]", '{"kind":"other"}'):
        (target / "receipt.json").write_text(text, encoding="utf-8")
        assert runtime.gpu_runtime_environment(managed.root) == {}


@pytest.mark.parametrize("failure", [None, "cpu", "module", "version", "fallback"])
def test_native_probe_is_isolated_and_requires_real_cuda_result(tmp_path, monkeypatch, failure):
    target = tmp_path / "isolated_runtime"
    target.mkdir()
    environment = {"KOWW_GPU_RUNTIME_PATH": str(target), "KOWW_GPU_DLL_PATHS": str(target / "dlls")}
    probe = _probe(target, environment)
    if failure == "cpu":
        probe["providers"] = ["CPUExecutionProvider"]
    elif failure == "module":
        probe["module"] = str(tmp_path / "base_cpu/onnxruntime/__init__.py")
    elif failure == "version":
        probe["version"] = "1.30.0"
    elif failure == "fallback":
        probe["cpu_fallback_disabled"] = False
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "a diagnostic line\n" + json.dumps(probe) + "\n", "")
    monkeypatch.setattr(runtime.subprocess, "run", run)
    if failure:
        with pytest.raises(ValueError, match="session probe"):
            runtime._probe_runtime(target, environment)
    else:
        assert runtime._probe_runtime(target, environment) == probe
    command, kwargs = calls[0]
    assert command[:4] == [sys.executable, "-I", "-B", "-c"]
    assert command[-2:] == [str(target), environment["KOWW_GPU_DLL_PATHS"]]
    assert kwargs["env"]["KOWW_GPU_RUNTIME_PATH"] == str(target) and kwargs["check"]
    assert "session.disable_cpu_ep_fallback" in command[4] and "MatMul" in command[4] and "Conv" in command[4]


def test_native_probe_surfaces_child_dll_failure(tmp_path, monkeypatch):
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="Driver does not support CUDA 13")
    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(ValueError, match="Driver does not support CUDA 13"):
        runtime._probe_runtime(tmp_path, {"KOWW_GPU_DLL_PATHS": ""})
