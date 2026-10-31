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
from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.evaluation_workers import EvaluationJournal
from korean_openwakeword.project import create_project
from korean_openwakeword.runtime import bundle_candidates, sample_bundle
from test_korean_library_evaluation_preflight import _evaluation_inputs


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


def test_auto_runtime_failure_stops_before_cpu_scoring_or_result_publication(monkeypatch, tmp_path):
    project, root, bundle, manifest, _ = _evaluation_inputs(tmp_path)
    monkeypatch.setattr(evaluation_backend, "_bundle_limitation", lambda bundle: None)
    def unavailable(project_root):
        raise PermissionError("[WinError 5] runtime/onnxruntime/__init__.py")
    monkeypatch.setattr(evaluation_backend, "_runtime_environment", unavailable)
    monkeypatch.setattr(evaluate, "_gpu_sessions_with_environment", lambda *a, **kw:
                        pytest.fail("Unavailable CUDA must not enter GPU scoring"))
    monkeypatch.setattr(evaluate, "score_sessions", lambda *a, **kw:
                        pytest.fail("GPU failure must not start CPU scoring"))
    output = root / "reports/failed_gpu"
    with pytest.raises(RuntimeError, match="WinError 5.*CPU로 자동 전환하지 않습니다"):
        evaluate.evaluate(project, bundle, output, manifest=manifest, smoke=True, workers=1)
    assert not (output / "receipt.json").exists()
    assert not (output / "report.json").exists()
    assert not (output / "_evaluation_work/checkpoint.json").exists()


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


def test_cpu_prefix_is_preserved_until_cpu_resume_is_explicitly_chosen(monkeypatch, tmp_path):
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
    protected = {path: digest(path) for path in output.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="기존 CPU 평가 기록은 보존됩니다"):
        evaluate.evaluate(project, bundle, output, manifest=manifest, smoke=True, workers=1, resume=True)
    assert resolutions == [("auto", "cpu")] and not probed
    assert {path: digest(path) for path in protected} == protected
    assert not (output / "receipt.json").exists()
    report = evaluate.evaluate(project, bundle, output, manifest=manifest, smoke=True, workers=1, resume=True, backend="cpu")
    assert resolutions == [("auto", "cpu"), ("cpu", "cpu")] and not probed
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
    # This routing fixture publishes no acoustic score cache. Preserve the
    # pre-recommendation workflow while testing GPU/CPU defaults independently.
    settings.pop("runtime_recommendation_policy")
    if legacy:
        for name in ("evaluation_backend", "gpu_batch_size", "gpu_boundary_margin", "gpu_numerical_differences_approved"):
            settings.pop(name)
    before = deepcopy(settings)
    project = create_project(tmp_path / "word", "보라야", "defaults",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    bundle, manifest = sample_bundle(), tmp_path / "evaluation.jsonl"
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


@pytest.mark.parametrize("saved_kind", ["partial", "completed"])
@pytest.mark.parametrize("requested", ["auto", "cpu"])
@pytest.mark.parametrize("saved_backend", ["cpu", "cuda"])
def test_studio_reuses_only_the_selected_backend_and_preserves_other_evaluation_work(
        monkeypatch, tmp_path, saved_kind, requested, saved_backend):
    from korean_openwakeword import selection
    settings = studio_steps.preset_settings("check")
    settings.pop("evaluation_policy")
    # Checkpoint/backend reuse is tested with synthetic report-only records;
    # actual automatic recommendation uses the end-to-end stage fixture.
    settings.pop("runtime_recommendation_policy")
    settings["evaluation_backend"] = requested
    project = create_project(tmp_path / "word", "보라야", "reuse_cpu",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    bundle, manifest = sample_bundle(), tmp_path / "evaluation.jsonl"
    manifest.write_text("", encoding="utf-8")
    base = project.parent / "reports/studio_evaluation"
    saved_binding = evaluate.evaluation_binding(project, bundle, manifest, smoke=True, backend=saved_backend)
    if saved_kind == "partial":
        write_json(base / "_evaluation_work/checkpoint.json", {"schema_version": 1, "binding": saved_binding})
        (base / "_evaluation_work/scores.bin").write_bytes(b"preserve score prefix")
    else:
        write_json(base / "receipt.json", {"fixture": "completed CPU receipt"})
        write_json(base / "report.json", {"manifest_sha256": digest(manifest),
            "bundle_manifest_sha256": digest(bundle / "manifest.json"), "evaluation_binding": saved_binding,
            "execution": {"backend": saved_backend}, "models": [], "smoke_test_only": True})
        write_json(base / "clips.json", [])
    preserved = {path: digest(path) for path in base.rglob("*") if path.is_file()}
    context = {"results": {"package": {"result": {"bundle": str(bundle)}},
                           "development": {"result": {"manifest": str(manifest)}}},
               "inputs": {"review_answers": {"fixture": "preserved"}}}
    before_context, before_settings = deepcopy(context), deepcopy(settings)
    calls, reused = [], []
    def score(path, selected_bundle, output, **kwargs):
        calls.append((output, kwargs))
        report = {"models": [], "smoke_test_only": kwargs["smoke"]}
        write_json(output / "report.json", report)
        write_json(output / "receipt.json", {})
        write_json(output / "clips.json", [])
        return report
    monkeypatch.setattr(evaluate, "evaluate", score)
    monkeypatch.setattr(selection, "verified_evaluation", lambda *args: reused.append(args))
    result = studio_steps.run_stage(project, tmp_path / "workspace", settings, "evaluation", context)
    assert result["state"] == "completed" and context == before_context and settings == before_settings
    planned_backend = "cuda" if requested == "auto" else "cpu"
    if saved_backend != planned_backend:
        assert len(calls) == 1 and not reused
        output, options = calls[0]
        assert output == base.with_name("studio_evaluation_attempt2")
        assert options["backend"] == requested and options["resume"] is False
        assert {path: digest(path) for path in preserved} == preserved
    elif saved_kind == "partial":
        assert len(calls) == 1 and not reused
        assert calls[0][0] == base and calls[0][1]["backend"] == planned_backend and calls[0][1]["resume"] is True
    else:
        assert not calls and len(reused) == 1
        assert {path: digest(path) for path in preserved} == preserved


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
def test_evaluation_runtime_failure_stops_frontend_preparation(monkeypatch, tmp_path, error):
    from korean_openwakeword import gpu_runtime_install
    def unavailable(root):
        assert root == tmp_path
        raise error
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", unavailable)
    with pytest.raises(RuntimeError) as caught:
        studio_steps._prepare_evaluation_runtime(tmp_path)
    assert str(error) in str(caught.value)
    assert "CPU로 자동 전환하지 않습니다" in str(caught.value)
    assert caught.value.__cause__ is error


def test_unsupported_managed_runtime_does_not_claim_success(monkeypatch, tmp_path):
    from korean_openwakeword import gpu_runtime_install
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", lambda root:
                        {"state": "unsupported", "reason": "Configure an external GPU runtime"})
    with pytest.raises(RuntimeError, match="Configure an external GPU runtime"):
        studio_steps._prepare_evaluation_runtime(tmp_path)


def test_explicit_external_runtime_is_not_replaced_or_claimed_to_be_cuda_verified(monkeypatch, tmp_path):
    from korean_openwakeword import gpu_runtime_install
    selected = tmp_path / "chosen"
    (selected / "onnxruntime").mkdir(parents=True)
    (selected / "onnxruntime/__init__.py").write_text("# external runtime", encoding="utf-8")
    monkeypatch.setenv("KOWW_GPU_RUNTIME_PATH", str(selected))
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", lambda root:
                        pytest.fail("Do not replace an explicitly configured runtime"))
    result = studio_steps._prepare_evaluation_runtime(tmp_path)
    assert result["state"] == "configured" and result["path"] == str(selected)
    assert result["cuda_execution_verified"] is False


@pytest.mark.parametrize("mode,expected_setup", [("manual_cuda", True), ("legacy", False),
                                                ("explicit_cpu", False), ("cpu_training_gpu_evaluation", True),
                                                ("setup_failure", True)])
def test_manually_installed_cuda_training_environment_prepares_evaluation_runtime(
        monkeypatch, tmp_path, mode, expected_setup):
    from korean_openwakeword import assets, benchmark, gpu_runtime_install
    settings = studio_steps.preset_settings("check")
    if mode == "legacy":
        settings.pop("evaluation_backend")
    elif mode == "explicit_cpu":
        settings["evaluation_backend"] = "cpu"
    project = create_project(tmp_path / "word", "보라야", "manual_cuda",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    setup_calls, frontend_calls = [], []
    def setup(root):
        setup_calls.append(root)
        if mode == "setup_failure":
            raise OSError("Fixture evaluation tools could not download")
        return {"state": "completed", "path": str(root / "cache/gpu_runtime")}
    def frontend(destination):
        frontend_calls.append(destination)
        write_json(destination / "receipt.json", {"pinned": True})
        return {"pinned": True}
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", setup)
    monkeypatch.setattr(studio_steps, "_installed_dependencies", lambda: {"onnx": "1.20.0"})
    monkeypatch.setattr(benchmark, "doctor", lambda **kwargs:
        {"tts_check": {"ready": True}, "reference_openwakeword_ready": True,
         "cuda_available": mode != "cpu_training_gpu_evaluation"})
    monkeypatch.setattr(assets, "prepare_frontend", frontend)
    if mode in ("explicit_cpu", "cpu_training_gpu_evaluation"):
        monkeypatch.setattr(studio_steps, "_install_dependencies", lambda root, backend:
                            {"state": "completed", "backend": backend})
        context = {"inputs": {"dependency_backend": "cpu"}}
    else:
        monkeypatch.setattr(studio_steps, "_install_dependencies", lambda *a, **kw:
                            pytest.fail("Manually installed packages must not be installed again"))
        context = {}
    if mode == "setup_failure":
        with pytest.raises(RuntimeError, match="Fixture evaluation tools could not download"):
            studio_steps.run_stage(project, tmp_path / "workspace", settings, "frontend", context)
        assert setup_calls == [project.parent] and not frontend_calls
        return
    result = studio_steps.run_stage(project, tmp_path / "workspace", settings, "frontend", context)
    assert result["state"] == "completed" and len(frontend_calls) == 1
    assert setup_calls == ([project.parent] if expected_setup else [])
    installation = result["result"]["installation"]
    if expected_setup:
        assert installation["evaluation_gpu_runtime"]["state"] == "completed"
    elif mode == "legacy":
        assert installation is None
