"""New detection defaults and recommendations never replace historical evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from korean_openwakeword.common import digest, identity, read_json, write_json
from korean_openwakeword.microphone import MicrophoneError, MicrophoneService
from korean_openwakeword.project import create_project, default_profile, load_project
from korean_openwakeword.runtime import sample_bundle


def test_new_projects_start_at_half_threshold_without_extra_confirmation(tmp_path):
    path = create_project(tmp_path / "new", "보라야", "new_defaults",
                          positive_originals=1000, augmentation_rounds=1,
                          updates=10000, train_voice_families=20)
    runtime = load_project(path)[1]["runtime"]
    assert runtime["threshold"] == .5
    assert runtime["consecutive_frames"] == 1
    assert runtime["frame_samples"] == 1280 and runtime["sample_rate"] == 16000
    assert runtime["warmup_s"] == 2
    assert runtime["cooldown_s"] == 1 and runtime["call_postroll_s"] == .5
    assert not runtime["vad_enabled"] and not runtime["noise_suppression_enabled"]


def test_loading_existing_configuration_preserves_its_original_detection_rules(tmp_path):
    config = default_profile()
    config["runtime"].update(threshold=.9862500429153442, consecutive_frames=2)
    path = tmp_path / "existing.json"
    write_json(path, config)
    original = path.read_bytes()
    _, loaded = load_project(path)
    assert loaded["runtime"]["threshold"] == .9862500429153442
    assert loaded["runtime"]["consecutive_frames"] == 2
    assert path.read_bytes() == original
    assert default_profile()["runtime"]["threshold"] == .5
    assert default_profile()["runtime"]["consecutive_frames"] == 1


def test_packaged_chunsik_keeps_the_runtime_that_was_actually_evaluated():
    original = read_json(sample_bundle() / "manifest.json")
    assert original["runtime"]["threshold"] == .9862500429153442
    assert original["runtime"]["consecutive_frames"] == 2


class RecommendationDetector:
    """Fixed scores distinguish four trial gates without model downloads."""
    manifest = {"wakeword": "보라야"}
    keys = ["head1", "head2", "head3"]
    runtime = dict(sample_rate=16000, frame_samples=1280, warmup_s=2,
                   threshold=.9862500429153442, consecutive_frames=2,
                   cooldown_s=1)

    def __init__(self):
        self.frames = 0

    def reset(self):
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        eligible = self.frames > 25
        return {"time_s": self.frames * .08, "eligible": self.frames > 25,
                "member_scores": [.6, .7, .8] if eligible else [0., 0., 0.],
                "score": .7 if eligible else 0.}


def _process_frames(service, session_id, count=27):
    payload = np.zeros(1280, dtype="<i2").tobytes()
    return [service.process_frame(session_id, index, payload) for index in range(count)]


def recommendations():
    return {"head1": {"threshold": .55, "consecutive_frames": 1},
            "head2": {"threshold": .65, "consecutive_frames": 2},
            "head3": {"threshold": .85, "consecutive_frames": 1},
            "mean": {"threshold": .65, "consecutive_frames": 3}}


def test_mic_recommendations_compare_all_models_with_their_own_trial_rules():
    detector = RecommendationDetector()
    old_runtime = deepcopy(detector.runtime)
    choices = recommendations()
    evidence = {"status": "completed", "deployment_backend_verified": False,
                "scope": "frozen GPU evaluation scores"}
    service = MicrophoneService(detector, runtime_recommendations=evidence,
                                recommended_runtimes=choices)
    assert service.info()["default_runtime"] == {
        "threshold": .9862500429153442, "consecutive_frames": 2}
    assert service.info()["runtime_recommendations"] == evidence
    assert service.info()["recommended_runtimes"] == choices
    start = service.start({"mode": "recommended"})
    assert start["mode"] == "recommended"
    assert [m["id"] for m in start["models"]] == [*detector.keys, "mean"]
    assert set(start["runtimes"]) == {*detector.keys, "mean"}
    for key, selected in choices.items():
        assert start["runtimes"][key] == dict(old_runtime, **selected)
    frames = _process_frames(service, start["session_id"], count=28)
    assert all(not any(r["event"] for r in row["responses"]) for row in frames[:25])
    assert [r["event"] for r in frames[25]["responses"]] == [True, False, False, False]
    assert [r["event"] for r in frames[26]["responses"]] == [False, True, False, False]
    assert [r["event"] for r in frames[27]["responses"]] == [False, False, False, True]
    service.stop(start["session_id"])
    assert detector.runtime == old_runtime
    assert service.info()["recommended_runtimes"] == choices
    assert service.info()["training_or_tuning"] is False
    assert start["audio_saved"] is False


def test_recommendation_input_is_copied_and_manual_trial_does_not_replace_it():
    detector = RecommendationDetector()
    choices = recommendations()
    original = deepcopy(choices)
    service = MicrophoneService(detector, recommended_runtimes=choices)
    choices["head1"]["threshold"] = .999
    start = service.start({"threshold": .5, "consecutive_frames": 1})
    assert start["mode"] == "manual"
    assert all(v["threshold"] == .5 and v["consecutive_frames"] == 1
               for v in start["runtimes"].values())
    service.stop(start["session_id"])
    assert service.info()["recommended_runtimes"] == original
    default_start = service.start()
    assert all(v["threshold"] == .9862500429153442 and v["consecutive_frames"] == 2
               for v in default_start["runtimes"].values())


def test_mic_rejects_missing_or_ambiguous_recommendation_settings():
    detector = RecommendationDetector()
    with pytest.raises(ValueError, match="every compared model"):
        MicrophoneService(detector, recommended_runtimes={"head1": recommendations()["head1"]})
    without = MicrophoneService(detector)
    with pytest.raises(MicrophoneError, match="추천"):
        without.start({"mode": "recommended"})
    assert not without.active()
    service = MicrophoneService(detector, recommended_runtimes=recommendations())
    with pytest.raises(MicrophoneError, match="함께"):
        service.start({"mode": "recommended", "threshold": .3})
    assert not service.active()


@pytest.mark.parametrize("bad", [{"threshold": True, "consecutive_frames": 1},
    {"threshold": float("nan"), "consecutive_frames": 1},
    {"threshold": .5, "consecutive_frames": False},
    {"threshold": .5, "consecutive_frames": 0},
    {"threshold": .5, "consecutive_frames": 11}])
def test_invalid_recommendation_gates_never_start_a_microphone_stream(bad):
    choices = recommendations()
    choices["head2"] = bad
    with pytest.raises(MicrophoneError):
        MicrophoneService(RecommendationDetector(), recommended_runtimes=choices)


@pytest.mark.parametrize("extra", ["cooldown_s", "warmup_s", "sample_rate", "frame_samples"])
def test_recommendations_cannot_change_other_parts_of_the_detection_protocol(extra):
    choices = recommendations()
    choices["head1"][extra] = 0
    with pytest.raises(ValueError):
        MicrophoneService(RecommendationDetector(), recommended_runtimes=choices)


def _site_evidence(tmp_path, count):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    source = evaluation / "scored.wav"
    source.write_bytes(b"RIFF exact immutable fixture audio")
    score = {"success": True, "max_score": .9, "latency_s": .08}
    rows = [{"sample_id": f"call_{index}", "original_id": "one", "kind": "positive",
             "category": "clean", "provenance": "qwen_tts", "path": str(source),
             "sha256": digest(source), "results": {"single": score}} for index in range(count)]
    report = {"stage": "development", "wakeword": "보라야", "models": [{"id": "single"}],
              "bundle_manifest_sha256": "b" * 64, "score_cache_sha256": "c" * 64,
              "runtime": {"threshold": .9862500429153442, "consecutive_frames": 2}}
    write_json(evaluation / "clips.json", rows)
    write_json(evaluation / "report.json", report)
    write_json(evaluation / "receipt.json", {
        "bundle_sha256": report["bundle_manifest_sha256"],
        "clips_sha256": digest(evaluation / "clips.json"),
        "report_sha256": digest(evaluation / "report.json")})
    recommended = {"schema_version": 1, "method": "joint_cached_score_search_v1",
                   "stage": "development", "status": "completed", "deployment_backend_verified": False,
                   "Final_used_for_selection": False,
                   "bindings": {"source_report_sha256": digest(evaluation / "report.json"),
                                "source_clips_sha256": digest(evaluation / "clips.json"),
                                "source_bundle_manifest_sha256": report["bundle_manifest_sha256"],
                                "source_score_cache_sha256": report["score_cache_sha256"]},
                   "scope": "frozen GPU evaluation scores", "candidates": []}
    return evaluation, source, report, rows, recommended


@pytest.mark.parametrize("count", [1, 1001])
def test_site_recommendations_are_a_separate_view_of_original_scored_audio(tmp_path, count):
    from korean_openwakeword.site import export_site
    evaluation, source, report, rows, recommended = _site_evidence(tmp_path, count)
    protected = {p: digest(p) for p in evaluation.iterdir()}
    output = tmp_path / "site"
    export_site(evaluation, output, runtime_recommendations=recommended)
    payload = (output / "results-data.js").read_text(encoding="utf-8")
    exported = json.loads(payload.removeprefix("window.KOWW_RESULTS=").removesuffix(";\n"))
    assert exported["report"] == dict(report, runtime_recommendations=recommended)
    assert {p: digest(p) for p in protected} == protected
    if count == 1:
        assert exported["clips"][0]["results"] == rows[0]["results"]
    else:
        page = read_json(output / "clips/page-000000.json")
        assert page["clips"][0]["results"] == rows[0]["results"]
    assert (output / "audio" / (digest(source) + ".wav")).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("field", ["source_report_sha256", "source_clips_sha256",
                                   "source_bundle_manifest_sha256", "source_score_cache_sha256"])
def test_site_rejects_recommendations_for_other_evaluation_evidence(tmp_path, field):
    from korean_openwakeword.site import export_site
    evaluation, _, _, _, recommended = _site_evidence(tmp_path, 1)
    recommended["bindings"][field] = "f" * 64
    protected = {p: digest(p) for p in evaluation.iterdir()}
    output = tmp_path / "wrong_recommendation"
    with pytest.raises(ValueError, match="recommendation"):
        export_site(evaluation, output, runtime_recommendations=recommended)
    assert not (output / "index.html").exists()
    assert {p: digest(p) for p in protected} == protected


@pytest.mark.parametrize("automatic", [True, False])
@pytest.mark.parametrize("policy", ["joint_cached_score_search_v1", "joint_cached_hard_conditions_v2", "joint_cached_hard_conditions_v3"])
def test_studio_recommends_after_scoring_only_for_the_versioned_policy(tmp_path, monkeypatch, automatic, policy):
    from korean_openwakeword import calibration, evaluate, studio_steps as steps
    settings = steps.preset_settings("check")
    settings["runtime_recommendation_policy"] = policy
    if not automatic:
        settings.pop("runtime_recommendation_policy")
    path = create_project(tmp_path / "workflow", "춘식아", "workflow_fixture",
                          positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                          updates=settings["updates"], generic_hours=settings["generic_hours"],
                          korean_hours=settings["korean_hours"], train_voice_families=20)
    manifest = path.parent / "development.jsonl"
    manifest.write_text("", encoding="utf-8")
    order, source_bytes, artifacts = [], {}, {}

    def score(project, bundle, output, **kwargs):
        order.append("score")
        write_json(output / "report.json", {"models": [], "smoke_test_only": False})
        write_json(output / "clips.json", [])
        write_json(output / "receipt.json", {})
        source_bytes.update({p: p.read_bytes() for p in output.iterdir()})
        return read_json(output / "report.json")

    def recommend(project, bundle, evaluation, output, *, progress_callback):
        order.append("recommend")
        assert order == ["score", "recommend"]
        assert {p: p.read_bytes() for p in source_bytes} == source_bytes
        progress_callback({"done": 4, "total": 4})
        write_json(output / "recommendation.json", {"recommendations": []})
        write_json(output / "receipt.json", {})
        artifacts.update(recommendation=str(output / "recommendation.json"),
                         receipt=str(output / "receipt.json"))
        return dict(artifacts)

    monkeypatch.setattr(evaluate, "evaluate", score)
    monkeypatch.setattr(calibration, "recommend", recommend)
    context = {"results": {"package": {"result": {"bundle": str(sample_bundle())}},
                           "development": {"result": {"manifest": str(manifest)}}}}
    result = steps.run_stage(path, tmp_path, settings, "evaluation", context)
    assert result["state"] == "completed"
    assert result["result"]["Final_accessed"] is False
    assert {p: p.read_bytes() for p in source_bytes} == source_bytes
    if automatic:
        assert order == ["score", "recommend"]
        assert result["result"]["runtime_recommendations"] == artifacts
        assert set(artifacts.values()).issubset(set(map(str, result["artifacts"])))
        progress = read_json(path.parent / ".studio/progress.json")
        assert progress["phase"] == "runtime_recommendation"
        assert progress["stage"] == "evaluation" and progress["done"] == progress["total"] == 4
    else:
        assert order == ["score"]
        assert result["result"]["runtime_recommendations"] is None


def test_new_policy_does_not_add_stages_or_enable_final_access():
    from korean_openwakeword import studio_steps as steps
    for name in ("check", "trial", "production"):
        settings = steps.preset_settings(name)
        assert settings["runtime_recommendation_policy"] == "joint_cached_hard_conditions_v3"
        plan = steps.workflow_plan(settings)
        assert len(plan) == 19
        assert not any("final" in row["id"].lower() for row in plan)


@pytest.mark.parametrize("available", [None, True, False])
def test_studio_microphone_uses_original_gate_for_unavailable_recommendations_only(tmp_path, monkeypatch, available):
    import threading
    from korean_openwakeword.studio import StudioManager
    detector = RecommendationDetector()
    bundle = tmp_path / "original_bundle"
    write_json(bundle / "manifest.json", {"members": [{"path": key + ".onnx"} for key in detector.keys]})
    original_bytes = (bundle / "manifest.json").read_bytes()
    choices = recommendations()
    evidence = {"candidates": [{"id": key, "runtime": value,
                                "original_runtime": deepcopy(detector.runtime)}
                               for key, value in choices.items()]}
    if available is not None:
        evidence["candidates"][1]["recommendation_available"] = available
    state = {"project": str(tmp_path / "project.json"), "results": {
        "package": {"result": {"bundle": str(bundle)}}}}
    manager = object.__new__(StudioManager)
    manager._lock, manager._microphones = threading.RLock(), {}
    manager._load = lambda project_id: state
    manager.runtime_recommendations = lambda value: deepcopy(evidence)
    monkeypatch.setattr("korean_openwakeword.runtime.WakeWordDetector", lambda original: detector)
    before = deepcopy(evidence)
    service = manager.microphone("fixture")
    selected = service.info()["recommended_runtimes"]
    original_gate = {key: detector.runtime[key] for key in ("threshold", "consecutive_frames")}
    assert selected["head2"] == (original_gate if available is False else choices["head2"])
    assert selected["head1"] == choices["head1"]
    assert selected["head3"] == choices["head3"] and selected["mean"] == choices["mean"]
    start = service.start({"mode": "recommended"})
    assert set(start["runtimes"]) == set(choices)
    assert start["runtimes"]["head2"]["threshold"] == selected["head2"]["threshold"]
    assert evidence == before
    assert (bundle / "manifest.json").read_bytes() == original_bytes


def test_unknown_recommendation_policy_stops_before_scoring():
    from korean_openwakeword import studio_steps as steps
    settings = steps.preset_settings("check")
    settings["runtime_recommendation_policy"] = "anything_is_acceptable"
    with pytest.raises(ValueError, match="recommendation policy"):
        steps.workflow_plan(settings)


def test_completed_studio_reads_bound_recommendation_without_rehashing_large_scores(tmp_path, monkeypatch):
    from korean_openwakeword import calibration
    from korean_openwakeword.studio import StudioManager
    root = tmp_path / "completed"
    project, bundle, evaluation = root / "project.json", root / "bundle", root / "evaluation"
    write_json(project, {})
    write_json(bundle / "manifest.json", {"id": "historical"})
    write_json(evaluation / "clips.json", [])
    (evaluation / "scores.npz").write_bytes(b"already verified and frozen score archive")
    bindings = {"project_config_sha256": digest(project),
                "source_bundle_manifest_sha256": digest(bundle / "manifest.json"),
                "source_clips_sha256": digest(evaluation / "clips.json"),
                "source_score_cache_sha256": digest(evaluation / "scores.npz")}
    write_json(evaluation / "report.json", {
        "score_cache_sha256": bindings["source_score_cache_sha256"]})
    bindings["source_report_sha256"] = digest(evaluation / "report.json")
    write_json(evaluation / "receipt.json", {
        "report_sha256": bindings["source_report_sha256"],
        "bundle_sha256": bindings["source_bundle_manifest_sha256"],
        "clips_sha256": bindings["source_clips_sha256"]})
    output = root / "reports" / ("studio_runtime_recommendations_" + bindings["source_report_sha256"][:16]
                                 + "_" + identity(calibration.SEARCH_POLICY)[:12])
    recommendation = {"schema_version": 1, "method": calibration.METHOD, "stage": "development",
                      "bindings": bindings, "search_policy": calibration.SEARCH_POLICY,
                      "Final_used_for_selection": False, "candidates": []}
    write_json(output / "recommendation.json", recommendation)
    write_json(output / "receipt.json", {
        "bindings": bindings, "recommendation_sha256": digest(output / "recommendation.json")})
    state = {"project": str(project), "results": {
        "package": {"result": {"bundle": str(bundle)}},
        "evaluation": {"result": {"evaluation": str(evaluation)}}}}
    original_digest = calibration.digest

    def small_files_only(path):
        assert path not in (evaluation / "clips.json", evaluation / "scores.npz")
        return original_digest(path)

    monkeypatch.setattr(calibration, "digest", small_files_only)
    manager = object.__new__(StudioManager)
    protected = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert manager.runtime_recommendations(state) == recommendation
    assert {p: p.read_bytes() for p in protected} == protected
    receipt = read_json(evaluation / "receipt.json")
    receipt["clips_sha256"] = "f" * 64
    write_json(evaluation / "receipt.json", receipt)
    with pytest.raises(ValueError, match="inputs or receipt changed"):
        manager.runtime_recommendations(state)
