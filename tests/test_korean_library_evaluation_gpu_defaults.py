"""GPU-first defaults must preserve real evaluation artifacts and saved CPU work.

The fake GPU seam below deliberately computes reference CPU scores: it tests
backend routing, validation and publication without claiming CUDA qualification.
Actual CUDA numerical/streaming qualification lives in the GPU-specific tests.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import evaluation_backend, evaluation_workers, evaluate, studio_steps
from korean_openwakeword.cli import dispatch, parser
from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.evaluation_workers import EvaluationJournal
from korean_openwakeword.project import create_project
from korean_openwakeword.runtime import bundle_candidates
from test_korean_library_evaluation_preflight import _evaluation_inputs
from test_korean_library_evaluation_workers import _binding_inputs


def _selection(requested, selected, resume_backend=None, reason_code="fixture_cuda"):
    return {"requested_backend": requested, "backend": selected,
            "resume_backend": resume_backend, "runtime_environment": {"fixture": "isolated"},
            "reason_code": reason_code, "reason": "Test-only backend selection",
            "numerical_policy": "fixture" if selected == "cuda" else None,
            "probe": {"available": selected == "cuda"}}


def _reference_gpu_seam(calls):
    def score(environment, bundle, model_ids, items, **kwargs):
        consumed = []
        def recorded():
            for item in items:
                consumed.append(item[0])
                yield item
        calls.append({"environment": environment, "indices": consumed,
                      "batch_size": kwargs["batch_size"], "margin": kwargs["boundary_margin"],
                      "qualification": kwargs["qualification"]})
        kwargs["stats"]["fixture_reference_scores"] = True
        yield from evaluation_workers.score_sessions(bundle, model_ids, recorded(), 1)
    return score


def _assert_score_bits_equal(first, second):
    with np.load(first / "scores.npz") as expected, np.load(second / "scores.npz") as actual:
        for name in ("offsets", "model_ids"):
            assert np.array_equal(expected[name], actual[name])
        assert np.array_equal(expected["scores"].view(np.uint32), actual["scores"].view(np.uint32))


def test_default_auto_routes_to_selected_gpu_and_publishes_identical_reference_artifacts(monkeypatch, tmp_path):
    project, root, bundle, manifest, _ = _evaluation_inputs(tmp_path)
    resolutions, gpu_calls = [], []
    def resolve(requested="auto", **kwargs):
        resolutions.append((requested, kwargs))
        return _selection(requested, "cpu" if requested == "cpu" else "cuda", kwargs["resume_backend"])
    monkeypatch.setattr(evaluation_backend, "resolve_evaluation_backend", resolve)
    monkeypatch.setattr(evaluate, "_gpu_sessions_with_environment", _reference_gpu_seam(gpu_calls))
    baseline = evaluate.evaluate(project, bundle, root / "reports/cpu", manifest=manifest,
                                 smoke=True, workers=1, backend="cpu")
    # No backend or numerical-approval argument: exercise the public default.
    report = evaluate.evaluate(project, bundle, root / "reports/auto", manifest=manifest,
                               smoke=True, workers=1)
    assert [requested for requested, _ in resolutions] == ["cpu", "auto"]
    assert resolutions[-1][1]["resume_backend"] is None
    assert resolutions[-1][1]["project_root"] == root
    assert report["execution"]["backend"] == "cuda"
    assert report["execution"]["backend_selection"]["reason_code"] == "fixture_cuda"
    assert report["evaluation_binding"] != baseline["evaluation_binding"]
    assert report["runtime"] == baseline["runtime"]
    assert report["models"] == baseline["models"]
    assert gpu_calls == [{"environment": {"fixture": "isolated"}, "indices": [0, 1],
                          "batch_size": 128, "margin": .001,
                          "qualification": {"user_approved_numerical_differences": True,
                                            "cpu_prefix_import_sha256": None}}]
    _assert_score_bits_equal(root / "reports/cpu", root / "reports/auto")
    assert read_json(root / "reports/auto/receipt.json")["report_sha256"]
    assert not (root / "reports/auto/_evaluation_work").exists()


def test_auto_fallback_records_cpu_reason_and_keeps_historical_cpu_binding(monkeypatch, tmp_path):
    project, root, bundle, manifest, _ = _evaluation_inputs(tmp_path)
    def unavailable(requested="auto", **kwargs):
        return _selection(requested, "cpu", kwargs["resume_backend"], "cuda_runtime_unavailable")
    monkeypatch.setattr(evaluation_backend, "resolve_evaluation_backend", unavailable)
    monkeypatch.setattr(evaluate, "_gpu_sessions_with_environment", lambda *a, **kw:
                        pytest.fail("Unavailable CUDA must not enter GPU scoring"))
    report = evaluate.evaluate(project, bundle, root / "reports/fallback", manifest=manifest,
                               smoke=True, workers=1)
    execution = report["execution"]
    assert execution["backend"] == "cpu" and "gpu" not in execution
    assert execution["backend_selection"]["reason_code"] == "cuda_runtime_unavailable"
    assert execution["backend_selection"]["requested_backend"] == "auto"
    assert report["evaluation_binding"] == evaluate.evaluation_binding(project, bundle, manifest, smoke=True)


def _seed_reference_prefix(project, root, bundle, manifest, rows, backend):
    directory = root / "reports/partial"
    directory.mkdir(parents=True)
    model_ids = [row["id"] for row in bundle_candidates(read_json(bundle / "manifest.json"))]
    offsets = np.cumsum([0] + [sf.info(row["path"]).frames // 1280 for row in rows])
    binding = evaluate.evaluation_binding(project, bundle, manifest, smoke=True, backend=backend)
    index, clip, scores = next(evaluation_workers.score_sessions(
        bundle, model_ids, [(0, rows[0], Path(rows[0]["path"]))], 1))
    journal = EvaluationJournal(directory, binding, offsets, model_ids)
    try:
        journal.add(index, clip, scores)
        journal.commit()
    finally:
        journal.close()
    return directory, scores.copy()


def test_auto_resume_keeps_existing_cpu_prefix_even_when_gpu_is_available(monkeypatch, tmp_path):
    project, root, bundle, manifest, rows = _evaluation_inputs(tmp_path)
    output, prefix = _seed_reference_prefix(project, root, bundle, manifest, rows, "cpu")
    resolutions, probed = [], []
    actual_resolve = evaluation_backend.resolve_evaluation_backend
    monkeypatch.setattr(evaluation_backend, "probe_cuda_backend", lambda *a, **kw: probed.append(True))
    def resolve(requested="auto", **kwargs):
        resolutions.append((requested, kwargs["resume_backend"]))
        return actual_resolve(requested, **kwargs)
    monkeypatch.setattr(evaluation_backend, "resolve_evaluation_backend", resolve)
    monkeypatch.setattr(evaluate, "_gpu_sessions_with_environment", lambda *a, **kw:
                        pytest.fail("Saved CPU score cache must remain CPU"))
    report = evaluate.evaluate(project, bundle, output, manifest=manifest, smoke=True, workers=1, resume=True)
    assert resolutions == [("auto", "cpu")] and not probed
    assert report["execution"]["backend"] == "cpu"
    assert report["execution"]["resumed_sessions"] == 1
    assert report["execution"]["backend_selection"]["reason_code"] == "resume_cpu_reference"
    with np.load(output / "scores.npz") as published:
        assert np.array_equal(published["scores"][:len(prefix)].view(np.uint32), prefix.view(np.uint32))


def test_auto_resume_keeps_existing_cuda_contract_and_committed_score_bits(monkeypatch, tmp_path):
    project, root, bundle, manifest, rows = _evaluation_inputs(tmp_path)
    output, prefix = _seed_reference_prefix(project, root, bundle, manifest, rows, "cuda")
    resolutions, gpu_calls = [], []
    def resolve(requested="auto", **kwargs):
        resolutions.append((requested, kwargs["resume_backend"]))
        assert kwargs["resume_backend"] == "cuda"
        return _selection(requested, "cuda", "cuda", "resume_cuda")
    monkeypatch.setattr(evaluation_backend, "resolve_evaluation_backend", resolve)
    monkeypatch.setattr(evaluate, "_gpu_sessions_with_environment", _reference_gpu_seam(gpu_calls))
    report = evaluate.evaluate(project, bundle, output, manifest=manifest, smoke=True, workers=1, resume=True)
    assert resolutions == [("auto", "cuda")]
    assert report["execution"]["backend"] == "cuda" and report["execution"]["resumed_sessions"] == 1
    assert gpu_calls[0]["indices"] == [1]
    with np.load(output / "scores.npz") as published:
        assert np.array_equal(published["scores"][:len(prefix)].view(np.uint32), prefix.view(np.uint32))


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_unavailable_cuda_resume_fails_before_consuming_or_changing_checkpoint(monkeypatch, tmp_path, requested):
    project, root, bundle, manifest, _ = _evaluation_inputs(tmp_path)
    output = root / "reports/partial"
    checkpoint = output / "_evaluation_work/checkpoint.json"
    write_json(checkpoint, {"binding": evaluate.evaluation_binding(project, bundle, manifest, smoke=True, backend="cuda")})
    before = checkpoint.read_bytes()
    monkeypatch.setattr(evaluation_backend, "_runtime_environment", lambda root: {})
    monkeypatch.setattr(evaluation_backend, "_loaded_runtime_conflict", lambda environment: None)
    monkeypatch.setattr(evaluation_backend, "probe_cuda_backend", lambda *a, **kw:
                        {"available": False, "reason": "Fixture CUDA became unavailable"})
    monkeypatch.setattr(evaluate, "score_sessions", lambda *a, **kw: pytest.fail("Cannot fall back to CPU"))
    with pytest.raises(RuntimeError, match="became unavailable"):
        evaluate.evaluate(project, bundle, output, manifest=manifest, smoke=True, workers=1,
                          resume=True, backend=requested)
    assert checkpoint.read_bytes() == before and not (output / "receipt.json").exists()


@pytest.mark.parametrize("name", ["check", "trial", "production"])
def test_new_presets_select_gpu_auto_with_explicit_numerical_policy(name):
    settings = studio_steps.preset_settings(name)
    assert settings["evaluation_backend"] == "auto"
    assert settings["gpu_numerical_differences_approved"] is True
    assert settings["gpu_batch_size"] == 128 and settings["gpu_boundary_margin"] == .001


@pytest.mark.parametrize("legacy", [False, True])
def test_studio_new_settings_use_auto_but_saved_settings_without_backend_remain_cpu(monkeypatch, tmp_path, legacy):
    settings = studio_steps.preset_settings("check")
    settings.pop("evaluation_policy")
    if legacy:
        for name in ("evaluation_backend", "gpu_batch_size", "gpu_boundary_margin", "gpu_numerical_differences_approved"):
            settings.pop(name)
    before = deepcopy(settings)
    project = create_project(tmp_path / "word", "보라야", "defaults",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    _, bundle, manifest = _binding_inputs(tmp_path)
    manifest.write_text("", encoding="utf-8")
    calls = []
    def score(path, selected_bundle, output, **kwargs):
        calls.append(kwargs)
        report = {"models": [], "smoke_test_only": kwargs["smoke"]}
        write_json(output / "report.json", report)
        write_json(output / "receipt.json", {})
        write_json(output / "clips.json", [])
        return report
    monkeypatch.setattr(evaluate, "evaluate", score)
    context = {"results": {"package": {"result": {"bundle": str(bundle)}},
                           "development": {"result": {"manifest": str(manifest)}}}}
    result = studio_steps.run_stage(project, tmp_path / "workspace", settings, "evaluation", context)
    assert result["state"] == "completed" and len(calls) == 1
    assert calls[0]["backend"] == ("cpu" if legacy else "auto")
    assert calls[0]["numerical_differences_approved"] == (None if legacy else True)
    assert settings == before  # A saved workflow is never silently migrated.


def test_cli_defaults_to_auto_and_keeps_explicit_cpu_override():
    arguments = ["evaluate", "--project", "project.json", "--bundle", "bundle", "--output", "scores"]
    defaults = parser().parse_args(arguments)
    assert defaults.backend == "auto" and defaults.accept_gpu_numerical_differences is None
    assert defaults.gpu_batch_size == 128 and defaults.gpu_boundary_margin == .001
    assert parser().parse_args(arguments + ["--backend", "cpu"]).backend == "cpu"


def test_setup_gpu_cli_dispatches_only_to_the_selected_project_cache(monkeypatch, tmp_path):
    from korean_openwakeword import gpu_runtime_install
    project = tmp_path / "selected word" / "project.json"
    calls = []
    def setup(root):
        calls.append(root)
        return {"state": "completed", "path": str(root / "cache/gpu_runtime")}
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", setup)
    arguments = parser().parse_args(["setup-gpu", "--project", str(project)])
    result = dispatch(arguments)
    assert calls == [project.resolve().parent]
    assert result == {"state": "completed", "path": str(project.parent / "cache/gpu_runtime")}
    assert not project.parent.exists()  # Dispatch does not create a second workspace.


@pytest.mark.parametrize("error", [OSError("native DLL unavailable"), ValueError("receipt changed"),
                                   RuntimeError("unsupported runtime")])
def test_optional_evaluation_runtime_failure_is_reported_without_blocking_training(monkeypatch, tmp_path, error):
    from korean_openwakeword import gpu_runtime_install
    def unavailable(root):
        assert root == tmp_path
        raise error
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", unavailable)
    result = studio_steps._prepare_evaluation_runtime(tmp_path)
    assert result["state"] == "unavailable" and result["default_evaluation_backend"] == "auto"
    assert str(error) in result["reason"]


@pytest.mark.parametrize("mode,expected_setup", [("manual_cuda", True), ("legacy", False),
                                                ("explicit_cpu", False), ("optional_setup_failure", True)])
def test_manually_installed_cuda_training_environment_prepares_optional_evaluation_runtime(
        monkeypatch, tmp_path, mode, expected_setup):
    from korean_openwakeword import assets, benchmark, gpu_runtime_install
    settings = studio_steps.preset_settings("check")
    if mode == "legacy":
        settings.pop("evaluation_backend")
    project = create_project(tmp_path / "word", "보라야", "manual_cuda",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    setup_calls, frontend_calls = [], []
    def setup(root):
        setup_calls.append(root)
        if mode == "optional_setup_failure":
            raise OSError("Fixture optional evaluation tools could not download")
        return {"state": "completed", "path": str(root / "cache/gpu_runtime")}
    def frontend(destination):
        frontend_calls.append(destination)
        write_json(destination / "receipt.json", {"pinned": True})
        return {"pinned": True}
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", setup)
    monkeypatch.setattr(studio_steps, "_installed_dependencies", lambda: {"onnx": "1.20.0"})
    monkeypatch.setattr(benchmark, "doctor", lambda **kwargs:
        {"tts_check": {"ready": True}, "reference_openwakeword_ready": True, "cuda_available": True})
    monkeypatch.setattr(assets, "prepare_frontend", frontend)
    if mode == "explicit_cpu":
        monkeypatch.setattr(studio_steps, "_install_dependencies", lambda root, backend:
                            {"state": "completed", "backend": backend})
        context = {"inputs": {"dependency_backend": "cpu"}}
    else:
        monkeypatch.setattr(studio_steps, "_install_dependencies", lambda *a, **kw:
                            pytest.fail("Manually installed packages must not be installed again"))
        context = {}
    result = studio_steps.run_stage(project, tmp_path / "workspace", settings, "frontend", context)
    assert result["state"] == "completed" and len(frontend_calls) == 1
    assert setup_calls == ([project.parent] if expected_setup else [])
    installation = result["result"]["installation"]
    if expected_setup:
        expected_state = "unavailable" if mode == "optional_setup_failure" else "completed"
        assert installation["evaluation_gpu_runtime"]["state"] == expected_state
    elif mode == "legacy":
        assert installation is None
