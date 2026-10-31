"""Automatic gates must preserve strict event semantics and frozen evidence."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from korean_openwakeword import calibration
from korean_openwakeword.calibration import PreparedScores, recommend, read_recommendation
from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.events import EventDetector
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.runtime import bundle_candidates, deployment_candidate, sample_bundle, verify_bundle
from korean_openwakeword.tune import iter_replay


@pytest.mark.parametrize("warmup", [0, .13, 2])
@pytest.mark.parametrize("cooldown", [0, .17, 1.03])
@pytest.mark.parametrize("count", [1, 2, 3, 6])
def test_vectorized_events_equal_scalar_detector(warmup, cooldown, count):
    rng = np.random.default_rng(305)
    lengths = [0, 3, 52, 89, 2, 45]
    offsets = np.r_[0, np.cumsum(lengths)]
    values = rng.choice(np.asarray([0, .2, .5, .98625, .98625004, 1], np.float32), offsets[-1])
    # Sustained highs across boundaries and a pre-warmup qualified run exercise
    # the eligibility edge and ensure cooldown cannot leak between sessions.
    values[40:81] = 1
    runtime = {"warmup_s": warmup, "cooldown_s": cooldown, "call_postroll_s": .5}
    clips = [{"sample_id": str(i), "kind": "negative", "category": "environment", "calls": [],
              "eligible_seconds": max(0, n * .08 - warmup)} for i, n in enumerate(lengths)]
    prepared = PreparedScores(clips, offsets, runtime)
    for threshold in [.2, .5, .98625, float(np.float32(.98625)), .999]:
        expected = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            detector = EventDetector(threshold, count, cooldown)
            for local, score in enumerate(values[start:end]):
                if detector.update(float(score), (local + 1) * .08, local * .08 + 1e-9 >= warmup)["event"]:
                    expected.append(start + local)
        actual = prepared.events(prepared.rolling_scores(values, count), threshold)
        assert actual.tolist() == expected


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("threshold", [.4, .8, .95])
def test_fast_call_matching_and_point_metrics_equal_reference(count, threshold):
    runtime = {"warmup_s": .16, "cooldown_s": .24, "call_postroll_s": .13,
               "threshold": threshold, "consecutive_frames": count}
    clips = [
        {"sample_id": "one", "kind": "positive", "category": "clean", "calls": [{"start_s": .3, "end_s": .64}], "eligible_seconds": 1.44},
        {"sample_id": "overlap", "kind": "positive", "category": "environment", "snr_db": 10,
         "calls": [{"start_s": .4, "end_s": .64}, {"start_s": .48, "end_s": .96}], "eligible_seconds": 1.44},
        {"sample_id": "reverb", "kind": "positive", "category": "clean", "reverb_condition": "short",
         "calls": [{"start_s": .4, "end_s": .56}], "eligible_seconds": 1.44},
        {"sample_id": "neg", "kind": "negative", "category": "environment", "calls": [], "eligible_seconds": 1.44},
    ]
    values = np.zeros((80, 1), np.float32)
    values[[5, 7, 11, 24, 25, 26, 29, 30, 45, 46, 47, 65, 66, 67, 71], 0] = .9
    offsets = np.arange(0, 81, 20)
    prepared = PreparedScores(clips, offsets, runtime)
    actual = prepared.summary(values[:, 0], threshold, count, {"id": "test", "kind": "single"})
    reference = list(iter_replay(clips, values, offsets, ["test"], runtime))
    calls = [call for row in reference if row["kind"] == "positive" for call in row["results"]["test"]["calls"]]
    primary = [call for row in reference if row["kind"] == "positive" and (row["category"] == "clean" or row.get("snr_db") in (20, 15, 10, 5)) for call in row["results"]["test"]["calls"]]
    assert actual["all_positive"]["successes"] == sum(call["success"] for call in calls)
    assert actual["clean"]["calls"] == 1
    latencies = [call["latency_s"] for call in primary if call["success"]]
    assert actual["primary"]["latency_p95_s"] == (float(np.quantile(latencies, .95)) if latencies else None)
    assert actual["negative"]["events"] == reference[-1]["results"]["test"]["false_activations"]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Tiny score-only fixture; it is not performance evidence for real heads."""
    project = create_project(tmp_path / "project", "춘식아", "calibration_fixture")
    # Fixture sources are read-only throughout; avoid copying the 20 MB bundle
    # for every detector/publication test.
    bundle = Path(sample_bundle())
    package = verify_bundle(bundle)
    _, config = load_project(project)
    config["runtime"] = deepcopy(package["runtime"])
    config["targets"]["development_negative_hours"] = [.0001, .001]
    write_json(project, config)
    candidates = bundle_candidates(package)
    ids = [row["id"] for row in candidates]
    clips, values, offsets = [], [], [0]
    for category, snr in [("clean", None)] + [(cat, snr) for cat in ("environment", "music", "competing_speech") for snr in (10, 5)]:
        score = np.zeros((45, len(ids)), np.float32)
        score[27:32] = [.65, .96, .01, .65]
        row = {"sample_id": f"positive{len(clips)}", "kind": "positive", "category": category, "snr_db": snr,
               "provenance": "qwen_tts", "duration_s": 3.6,
               "eligible_seconds": 3.6 - package["runtime"]["warmup_s"],
               "calls": [{"start_s": 2.0, "end_s": 2.8}], "results": {key: {} for key in ids}}
        clips.append(row); values.append(score); offsets.append(offsets[-1] + len(score))
    score = np.zeros((50, len(ids)), np.float32)
    # One brief spurious crossing is excluded by a two-frame gate.
    score[29] = [.7, .3, .2, .4]
    clips.append({"sample_id": "negative", "kind": "negative", "category": "environment", "calls": [],
                  "duration_s": 4., "eligible_seconds": 4. - package["runtime"]["warmup_s"],
                  "results": {key: {} for key in ids}})
    values.append(score); offsets.append(offsets[-1] + len(score))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    np.savez_compressed(evaluation / "scores.npz", scores=np.concatenate(values), offsets=np.asarray(offsets, np.int64), model_ids=np.asarray(ids))
    write_json(evaluation / "clips.json", clips)
    write_json(evaluation / "report.json", {"wakeword": "춘식아", "stage": "development", "sessions": len(clips),
               "runtime": package["runtime"], "candidate_definitions": candidates,
               "deployment_candidate": deployment_candidate(package), "models": [{"id": key} for key in ids],
               "score_cache_sha256": digest(evaluation / "scores.npz"), "bundle_manifest_sha256": digest(bundle / "manifest.json")})
    write_json(evaluation / "receipt.json", {"report_sha256": digest(evaluation / "report.json"), "clips_sha256": digest(evaluation / "clips.json"), "bundle_sha256": digest(bundle / "manifest.json")})
    # The event oracle above tests the full 1..6 range. A short declared grid
    # keeps these publication/resume tests focused and fast.
    monkeypatch.setattr(calibration, "COARSE_THRESHOLDS", (.5, .9))
    monkeypatch.setattr(calibration, "CONSECUTIVE_COUNTS", (1, 2))
    policy = deepcopy(calibration.SEARCH_POLICY)
    policy.update(coarse_thresholds=[.5, .9], consecutive_frames=[1, 2])
    monkeypatch.setattr(calibration, "SEARCH_POLICY", policy)
    return project, bundle, evaluation, ids


def test_recommendations_per_candidate_and_no_source_mutation(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    protected = {path: digest(path) for directory in (bundle, evaluation) for path in directory.rglob("*") if path.is_file()}
    protected[Path(project)] = digest(project)
    progress = []
    output = tmp_path / "recommendations"
    result = recommend(project, bundle, evaluation, output, progress.append)
    report = read_recommendation(project, bundle, evaluation, output)
    assert {row["id"] for row in report["candidates"]} == set(ids)
    first, second, failed, ensemble = report["candidates"]
    assert first["runtime"]["consecutive_frames"] == 2 and first["passed"]
    assert second["runtime"]["consecutive_frames"] == 1 and second["passed"]
    assert failed["status"] == "best_effort" and failed["passed"] is False
    assert ensemble["passed"]
    assert all(row["baseline"]["runtime"]["threshold"] == .5 and row["baseline"]["runtime"]["consecutive_frames"] == 1 for row in report["candidates"])
    assert result["selection_applied"] is False and result["network_rerun"] is False
    assert report["Final_used_for_selection"] is False
    assert progress[-1]["completed"] == progress[-1]["total"]
    assert all(digest(path) == sha for path, sha in protected.items())
    assert recommend(project, bundle, evaluation, output)["candidates"] == report["candidates"]


def test_interrupted_search_resumes_and_source_change_rejected(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    directory = tmp_path / "resume"
    def interrupt(progress):
        if progress["completed"] == 3:
            raise InterruptedError("test pause")
    with pytest.raises(InterruptedError):
        recommend(project, bundle, evaluation, directory, interrupt)
    old_lines = (directory / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    recommend(project, bundle, evaluation, directory)
    new_lines = (directory / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    assert new_lines[:len(old_lines)] == old_lines
    (evaluation / "clips.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs or receipt changed"):
        recommend(project, bundle, evaluation, directory)


def test_gpu_cache_is_not_mislabeled_cpu_verified(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    report = read_json(evaluation / "report.json")
    report["execution"] = {"backend": "cuda", "scoring_protocol": "cuda_batch_fp32_cpu_boundary_v1"}
    write_json(evaluation / "report.json", report)
    receipt = read_json(evaluation / "receipt.json")
    receipt["report_sha256"] = digest(evaluation / "report.json")
    write_json(evaluation / "receipt.json", receipt)
    result = recommend(project, bundle, evaluation, tmp_path / "gpu")
    assert all(row["deployment_backend_verified"] is False and row["verification_scope"] == "cached_score_protocol_only" for row in result["candidates"])


def test_final_lock_blocks_automatic_search(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    write_json(Path(project).parent / "final_lock.json", {})
    with pytest.raises(ValueError, match="Final is frozen"):
        recommend(project, bundle, evaluation, tmp_path / "blocked")


def test_no_pass_fallback_protects_clean_false_activation_and_latency_targets():
    targets = {"clean_recall": .98, "fa_per_hour": .1, "latency_p95_s": .5, "minimum_negative_hours": 20}
    def result(noise, clean, false, latency, passed=False):
        metrics = {"noise_cell_mean": noise, "clean": {"recall": clean},
                   "negative": {"fa_per_hour": false, "hours": 30}, "primary": {"latency_p95_s": latency}}
        return {"runtime": {"threshold": .5, "consecutive_frames": 1}, "metrics": metrics,
                "adoption": {"passed": passed}, "constraint_checks": calibration._constraint_checks(metrics, targets)}
    safe = result(.85, .99, 0, .2)
    noisy_always_on = result(1., 1., 100, .01)
    delayed = result(.99, .99, 0, 1.)
    clean_loss = result(.99, .97, 0, .2)
    assert max([safe, noisy_always_on, delayed, clean_loss], key=calibration._rank) is safe
    eligible = result(.95, .99, 0, .2, True)
    assert calibration._rank(eligible) > calibration._rank(safe)
    almost_safe = result(.7, .99, .2, .2)
    assert calibration._rank(almost_safe) > calibration._rank(noisy_always_on)


def test_display_loader_checks_lightweight_source_receipt(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    output = tmp_path / "recommendation"
    recommend(project, bundle, evaluation, output)
    assert read_recommendation(project, bundle, evaluation, output, verify_inputs=False)["method"] == calibration.METHOD
    receipt = read_json(evaluation / "receipt.json")
    receipt["clips_sha256"] = "0" * 64
    write_json(evaluation / "receipt.json", receipt)
    with pytest.raises(ValueError, match="inputs or receipt changed"):
        read_recommendation(project, bundle, evaluation, output, verify_inputs=False)
