"""Automatic gates must preserve strict event semantics and frozen evidence."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from korean_openwakeword import calibration
from korean_openwakeword.calibration import PreparedScores, recommend, read_recommendation
from korean_openwakeword.common import digest, iter_rows, read_json, write_json
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
    for reverb, category, snr in [("none", "clean", None)] + list(calibration.HARD_CONDITIONS):
        score = np.zeros((45, len(ids)), np.float32)
        score[27:32] = [.65, .96, .01, .65]
        row = {"sample_id": f"positive{len(clips)}", "kind": "positive", "category": category, "snr_db": snr,
               "reverb_condition": reverb,
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
    assert failed["status"] == "no_feasible_candidate" and failed["passed"] is False
    assert failed["recommendation_available"] is False and failed["protection_passed"] is False
    assert "Clean recall below target or unmeasured" in failed["recommendation_reasons"]
    assert all(row["original_runtime"] == verify_bundle(bundle)["runtime"] for row in report["candidates"])
    assert all(row["recommendation_available"] for row in (first, second, ensemble))
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


def test_resume_mid_refinement_reconstructs_the_same_search(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    complete = tmp_path / "complete"
    expected = recommend(project, bundle, evaluation, complete)
    resumed = tmp_path / "partly_refined"
    def interrupt(progress):
        if progress["completed"] == 10:
            raise InterruptedError("pause after coarse search and some refinement probes")
    with pytest.raises(InterruptedError):
        recommend(project, bundle, evaluation, resumed, interrupt)
    actual = recommend(project, bundle, evaluation, resumed)
    assert actual["candidates"] == expected["candidates"]
    assert list(iter_rows(resumed / "comparisons.jsonl")) == list(iter_rows(complete / "comparisons.jsonl"))


def test_refinement_follows_each_consecutive_counts_own_promising_interval(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    directory = tmp_path / "per_count"
    recommend(project, bundle, evaluation, directory)
    # Head 2 has a one-frame negative spike at .3. Its one-frame gate must
    # refine above that spike; two-frame confirmation can search lower values.
    entries = [row["entry"] for row in iter_rows(directory / "comparisons.jsonl") if row["candidate"] == ids[1]]
    one = {row["runtime"]["threshold"] for row in entries if row["runtime"]["consecutive_frames"] == 1}
    two = {row["runtime"]["threshold"] for row in entries if row["runtime"]["consecutive_frames"] == 2}
    assert one != two
    assert min(two) < min(one)
    assert len(one | two) > 2  # both start from the same two coarse thresholds


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
                   "hard_conditions": {"cell_mean": noise, "worst_recall": noise, "complete": True},
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


def test_hard_conditions_have_equal_weight_and_easy_or_unknown_conditions_do_not_vote():
    runtime = {"warmup_s": 0, "cooldown_s": 0, "call_postroll_s": .1}
    clips, values = [], []
    def add(condition, success):
        reverb, category, snr = condition
        clips.append({"sample_id": str(len(clips)), "kind": "positive", "category": category,
                      "snr_db": snr, "reverb_condition": reverb, "eligible_seconds": .32,
                      "calls": [{"start_s": .08, "end_s": .24}]})
        values.extend([0, float(success), 0, 0])
    for index, condition in enumerate(calibration.HARD_CONDITIONS):
        for _ in range(100 if index == 0 else 1):
            add(condition, index == 0)
    # Extra easy, diagnostic or unknown rooms do not dilute the hard objective.
    for condition in [("none", "clean", None), ("none", "environment", 20),
                      ("strong", "music", 0), ("unknown", "environment", 5)]:
        for _ in range(40):
            add(condition, True)
    prepared = PreparedScores(clips, np.arange(0, len(values) + 1, 4), runtime)
    summary = prepared.summary(np.asarray(values), .5, 1, {"id": "head", "kind": "single"})
    hard = summary["hard_conditions"]
    assert hard["complete"] and hard["expected_cells"] == hard["measured_cells"] == 20
    assert hard["cell_mean"] == pytest.approx(1 / 20)
    assert hard["worst_recall"] == 0
    assert hard["conditions"]["none:environment:10"]["calls"] == 100
    assert summary["all_positive"]["recall"] > .8  # pooled recall is a different quantity


@pytest.mark.parametrize("reverb", ["none", "unknown"])
def test_missing_reverb_coverage_is_not_a_zero_or_partial_average(reverb):
    runtime = {"warmup_s": 0, "cooldown_s": 0, "call_postroll_s": .1}
    clips = [{"sample_id": "one", "kind": "positive", "category": "environment", "snr_db": 10,
              "reverb_condition": reverb, "eligible_seconds": .32, "calls": [{"start_s": .08, "end_s": .24}]}]
    prepared = PreparedScores(clips, [0, 4], runtime)
    metrics = prepared.summary(np.asarray([0., 1., 0., 0.]), .5, 1, {"id": "head", "kind": "single"})
    hard = metrics["hard_conditions"]
    assert not hard["complete"] and hard["cell_mean"] is None and hard["worst_recall"] is None
    assert hard["measured_cells"] == (1 if reverb == "none" else 0)
    assert "strong:clean:none" in hard["missing_conditions"]
    targets = {"clean_recall": .98, "fa_per_hour": .1, "latency_p95_s": .5, "minimum_negative_hours": 20}
    assert not calibration._constraint_checks(metrics, targets)["hard_conditions"]


def test_hard_success_beats_easy_average_and_overall_goal_flag_is_not_the_objective():
    checks = dict(clean=True, false_activations=True, latency=True, negative_duration=True, hard_conditions=True)
    def item(easy, hard, passed):
        return {"runtime": {"threshold": .5, "consecutive_frames": 1}, "constraint_checks": checks,
                "adoption": {"passed": passed}, "metrics": {"noise_cell_mean": easy,
                    "hard_conditions": {"cell_mean": hard, "worst_recall": .6, "complete": True},
                    "negative": {"fa_per_hour": 0.}, "primary": {"latency_p95_s": .2}}}
    easy_winner = item(.99, .7, True)
    hard_winner = item(.9, .85, False)
    assert max([easy_winner, hard_winner], key=calibration._rank) is hard_winner


def test_feasible_recommendation_is_distinct_from_overall_adoption(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    archive = np.load(evaluation / "scores.npz", allow_pickle=False)
    scores, offsets, ids = archive["scores"].copy(), archive["offsets"], archive["model_ids"]
    # Remove one dry noisy call from head 2: Clean/FA/latency protections still
    # pass, but the existing per-noise adoption target must remain failed.
    clips = read_json(evaluation / "clips.json")
    position = next(i for i, row in enumerate(clips) if row.get("reverb_condition") == "none"
                    and row["category"] == "environment" and row.get("snr_db") == 10)
    scores[offsets[position]:offsets[position + 1], 1] = 0
    archive.close()
    np.savez_compressed(evaluation / "scores.npz", scores=scores, offsets=offsets, model_ids=ids)
    report = read_json(evaluation / "report.json")
    report["score_cache_sha256"] = digest(evaluation / "scores.npz")
    write_json(evaluation / "report.json", report)
    receipt = read_json(evaluation / "receipt.json")
    receipt["report_sha256"] = digest(evaluation / "report.json")
    write_json(evaluation / "receipt.json", receipt)
    selected = recommend(project, bundle, evaluation, tmp_path / "independent_status")["candidates"][1]
    assert selected["status"] == "recommended" and selected["recommendation_available"]
    assert selected["protection_passed"] and selected["recommendation_reasons"] == []
    assert selected["passed"] is False and selected["adoption"]["passed"] is False


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
