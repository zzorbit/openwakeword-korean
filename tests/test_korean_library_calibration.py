"""Automatic gates must preserve strict event semantics and frozen evidence."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from korean_openwakeword import calibration
from korean_openwakeword.calibration import PreparedScores, recommend, read_recommendation
from korean_openwakeword.common import digest, identity, iter_rows, read_json, write_json
from korean_openwakeword.events import EventDetector
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.runtime import bundle_candidates, deployment_candidate, sample_bundle, verify_bundle
from korean_openwakeword.tune import iter_replay


SNR_ZERO_CATEGORIES = ("environment", "music", "competing_speech")
SNR_ZERO_REVERBS = ("moderate", "strong")
EXPECTED_HARD_CONDITIONS = tuple((reverb, category, snr) for reverb in SNR_ZERO_REVERBS
                                 for category in SNR_ZERO_CATEGORIES for snr in (10, 5, 0))
PREVIOUS_HARD_CONDITIONS = tuple((reverb, category, snr) for reverb in ("none", *SNR_ZERO_REVERBS)
                                 for category in SNR_ZERO_CATEGORIES for snr in (10, 5)) + (
                                     ("moderate", "clean", None), ("strong", "clean", None))


@pytest.mark.parametrize("warmup", [0, .13, 2])
@pytest.mark.parametrize("cooldown", [0, .17, 1.03])
@pytest.mark.parametrize("count", [1, 2, 3, 6])
def test_vectorized_events_equal_scalar_detector(warmup, cooldown, count):
    rng = np.random.default_rng(305)
    lengths = [0, 3, 52, 89, 2, 45]
    offsets = np.r_[0, np.cumsum(lengths)]
    values = rng.choice(np.asarray([0, .01, .2, .5, .98625, .98625004, .99, 1], np.float32), offsets[-1])
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
    rolling = prepared.rolling_scores(values, count)
    positions, lower, upper = prepared.grid_events(values)[count]
    for rank, threshold in enumerate(calibration.COARSE_THRESHOLDS, 1):
        actual = prepared._apply_cooldown(positions[(lower < rank) & (rank <= upper)])
        np.testing.assert_array_equal(actual, prepared.events(rolling, threshold))


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
    conditions = [("none", "clean", None), *PREVIOUS_HARD_CONDITIONS]
    with_hit_frames = [(reverb, category, snr, 5) for reverb, category, snr in conditions]
    with_hit_frames += [
        ("moderate", "environment", 0, 2), ("moderate", "music", 0, 0), ("moderate", "competing_speech", 0, 1),
        ("strong", "environment", 0, 1), ("strong", "music", 0, 2), ("strong", "music", 0, 0),
        ("strong", "competing_speech", 0, 0),
    ]
    for reverb, category, snr, hit_frames in with_hit_frames:
        score = np.zeros((45, len(ids)), np.float32)
        score[27:32] = [.65, .96, .01, .65]
        score[27 + hit_frames:32, 0] = 0
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


def assert_report_snr_zero_diagnostics(evaluation, candidates):
    clips = read_json(evaluation / "clips.json")
    with np.load(evaluation / "scores.npz", allow_pickle=False) as archive:
        scores, offsets, ids = archive["scores"], archive["offsets"], archive["model_ids"].tolist()
    for candidate in candidates:
        candidate_id = candidate["id"]
        for setting in (candidate, candidate["baseline"]):
            replayed = iter_replay(clips, scores, offsets, [candidate_id], setting["runtime"],
                                   columns=[ids.index(candidate_id)])
            reverb_conditions = setting["metrics"]["reverb_conditions"]
            measured = {}
            for row in replayed:
                reverb = row.get("reverb_condition")
                if reverb in SNR_ZERO_REVERBS and row.get("snr_db") == 0:
                    measured.setdefault((reverb, row["category"]), []).extend(row["results"][candidate_id]["calls"])
            assert set(measured) == {(reverb, category) for reverb in SNR_ZERO_REVERBS for category in SNR_ZERO_CATEGORIES}
            for (reverb, category), calls in measured.items():
                actual = reverb_conditions[reverb]["conditions"][f"{category}:0"]
                successes = sum(call["success"] for call in calls)
                assert actual["calls"] == len(calls)
                assert actual["successes"] == successes
                assert actual["recall"] == successes / len(calls)


def snr_zero_recall(setting):
    return {reverb: {category: condition["recall"] for category, condition in group["conditions"].items()}
            for reverb, group in setting["metrics"]["reverb_conditions"].items()}


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
    assert_report_snr_zero_diagnostics(evaluation, report["candidates"])
    assert snr_zero_recall(first) == {
        "moderate": {"environment:0": 1., "music:0": 0., "competing_speech:0": 0.},
        "strong": {"environment:0": 0., "music:0": .5, "competing_speech:0": 0.},
    }
    assert snr_zero_recall(first["baseline"]) == {
        "moderate": {"environment:0": 1., "music:0": 0., "competing_speech:0": 1.},
        "strong": {"environment:0": 1., "music:0": .5, "competing_speech:0": 0.},
    }
    for setting, expected in ((first, 13.5 / 18), (first["baseline"], 15.5 / 18)):
        hard = setting["metrics"]["hard_conditions"]
        assert hard["measured_cells"] == hard["expected_cells"] == 18
        assert hard["cell_mean"] == pytest.approx(expected)
        assert hard["worst_recall"] == 0.


@pytest.mark.parametrize("published_strong", [False, True])
def test_report_diagnostics_replay_archived_runtimes_without_rewriting_receipt(experiment, tmp_path, monkeypatch, published_strong):
    project, bundle, evaluation, ids = experiment
    output = tmp_path / "archived_recommendation"
    recommend(project, bundle, evaluation, output)
    publication, receipt_path = output / "recommendation.json", output / "receipt.json"
    archived = read_json(publication)
    old_policy = dict(calibration.PREVIOUS_SEARCH_POLICY, coarse_thresholds=[.5, .9], consecutive_frames=[1, 2])
    monkeypatch.setattr(calibration, "PREVIOUS_SEARCH_POLICY", old_policy)
    archived.update(method=old_policy["method"], search_policy=old_policy)
    for candidate in archived["candidates"]:
        for setting in (candidate, candidate["baseline"]):
            if published_strong:
                groups = setting["metrics"]["reverb_conditions"]
                del groups["moderate"]
                groups["strong"]["conditions"] = {
                    key: {field: row[field] for field in ("calls", "successes", "recall")}
                    for key, row in groups["strong"]["conditions"].items()
                }
            else:
                del setting["metrics"]["reverb_conditions"]
    write_json(publication, archived)
    write_json(receipt_path, dict(read_json(receipt_path), recommendation_sha256=digest(publication)))
    protected = {path: path.read_bytes() for directory in (output, evaluation) for path in directory.iterdir()
                 if path.is_file()}

    source = read_recommendation(project, bundle, evaluation, output)
    original_output_files = set(output.iterdir())
    assert calibration.report_recommendation(project, bundle, evaluation, output, source, compute=False) is None
    projected = calibration.report_recommendation(project, bundle, evaluation, output, source)
    assert [row["id"] for row in projected["candidates"]] == ids
    assert_report_snr_zero_diagnostics(evaluation, projected["candidates"])
    matched_archive = deepcopy(projected)
    for candidate in matched_archive["candidates"]:
        for setting in (candidate, candidate["baseline"]):
            if published_strong:
                del setting["metrics"]["reverb_conditions"]["moderate"]
            else:
                del setting["metrics"]["reverb_conditions"]
    assert matched_archive == archived
    assert source == archived
    assert set(output.iterdir()) == original_output_files
    assert calibration.report_recommendation(project, bundle, evaluation, output, source, compute=False) == projected
    cache, = tmp_path.glob("report_diagnostics_*.json")
    saved = read_json(cache)
    assert saved["binding"]["schema_version"] == 2
    strong_only = {candidate["id"]: {mode: setting["metrics"]["reverb_conditions"]["strong"]["conditions"]
                   for mode, setting in (("chosen", candidate), ("baseline", candidate["baseline"]))}
                   for candidate in projected["candidates"]}
    write_json(cache, {"binding": dict(saved["binding"], schema_version=1), "measured": strong_only,
                       "checksum": identity(strong_only)})
    assert calibration.report_recommendation(project, bundle, evaluation, output, source, compute=False) is None
    assert calibration.report_recommendation(project, bundle, evaluation, output, source) == projected
    assert read_json(cache) == saved
    invalid = deepcopy(saved)
    invalid["measured"].pop(ids[0])
    write_json(cache, invalid)
    assert calibration.report_recommendation(project, bundle, evaluation, output, source, compute=False) is None
    assert calibration.report_recommendation(project, bundle, evaluation, output, source) == projected
    assert {path: path.read_bytes() for path in protected} == protected


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


def test_recommendation_preserves_a_receipt_published_before_lock(experiment, tmp_path, monkeypatch):
    project, bundle, evaluation, _ = experiment
    output = tmp_path / "concurrent-recommendation"
    original = calibration.writer_lock
    published = {}

    @contextmanager
    def interleaved_lock(directory):
        if not published:
            monkeypatch.setattr(calibration, "writer_lock", original)
            recommend(project, bundle, evaluation, output)
            published.update({path.name: digest(path) for path in output.iterdir()})
        with original(directory):
            yield

    monkeypatch.setattr(calibration, "writer_lock", interleaved_lock)
    result = recommend(project, bundle, evaluation, output)
    assert result["candidates"] == read_recommendation(project, bundle, evaluation, output)["candidates"]
    assert {path.name: digest(path) for path in output.iterdir()} == published


def test_resume_partway_through_fixed_grid_reconstructs_the_same_search(experiment, tmp_path):
    from korean_openwakeword.evaluate import EvaluationPaused

    project, bundle, evaluation, _ = experiment
    complete = tmp_path / "complete"
    expected = recommend(project, bundle, evaluation, complete)
    resumed = tmp_path / "partly_searched"
    completed = 0

    def update(progress):
        nonlocal completed
        completed = progress["completed"]

    with pytest.raises(EvaluationPaused):
        recommend(project, bundle, evaluation, resumed, update, pause_requested=lambda: completed == 10)
    assert len(list(iter_rows(resumed / "comparisons.jsonl"))) == 10
    actual = recommend(project, bundle, evaluation, resumed)
    assert actual["candidates"] == expected["candidates"]
    assert list(iter_rows(resumed / "comparisons.jsonl")) == list(iter_rows(complete / "comparisons.jsonl"))
    assert read_json(resumed / "recommendation.json")["search_policy"] == calibration.SEARCH_POLICY


@pytest.mark.parametrize("tail", [b'{"candidate":"unfinished",', b'{"note":"\xe3'])
def test_recommendation_repairs_only_a_torn_final_comparison(experiment, tmp_path, tail):
    project, bundle, evaluation, _ = experiment
    output = tmp_path / "torn-search"

    def interrupt(progress):
        if progress["completed"] == 3:
            raise InterruptedError("pause after a complete comparison")

    with pytest.raises(InterruptedError):
        recommend(project, bundle, evaluation, output, interrupt)
    journal = output / "comparisons.jsonl"
    committed = journal.read_bytes()
    journal.write_bytes(committed + tail)
    complete = recommend(project, bundle, evaluation, tmp_path / "complete-search")
    resumed = recommend(project, bundle, evaluation, output)
    assert resumed["candidates"] == complete["candidates"]
    assert journal.read_bytes().startswith(committed)
    assert list(iter_rows(journal)) == list(iter_rows(tmp_path / "complete-search/comparisons.jsonl"))


def test_recommendation_refuses_corrupt_completed_comparison(experiment, tmp_path):
    project, bundle, evaluation, _ = experiment
    output = tmp_path / "corrupt-search"

    def interrupt(_progress):
        raise InterruptedError("pause after a complete comparison")

    with pytest.raises(InterruptedError):
        recommend(project, bundle, evaluation, output, interrupt)
    journal = output / "comparisons.jsonl"
    corrupt = journal.read_bytes() + b"not json\n"
    journal.write_bytes(corrupt)
    with pytest.raises(ValueError):
        recommend(project, bundle, evaluation, output)
    assert journal.read_bytes() == corrupt
    assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("policy_name", ["LEGACY_SEARCH_POLICY", "PREVIOUS_SEARCH_POLICY"])
def test_unfinished_earlier_policy_cannot_resume_in_v5_directory(experiment, tmp_path, policy_name):
    project, bundle, evaluation, _ = experiment
    output = tmp_path / "interrupted_search"

    def interrupt(progress):
        if progress["completed"] == 2:
            raise InterruptedError("pause before the recommendation is published")

    with pytest.raises(InterruptedError):
        recommend(project, bundle, evaluation, output, interrupt)
    declaration = read_json(output / "search.json")
    policy = dict(getattr(calibration, policy_name), coarse_thresholds=[.5, .9], consecutive_frames=[1, 2])
    write_json(output / "search.json", dict(declaration, search_policy=policy))
    original = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="inputs or policy changed"):
        recommend(project, bundle, evaluation, output)
    assert {path: path.read_bytes() for path in original} == original
    assert not (output / "receipt.json").exists()


def test_all_counts_use_exactly_the_declared_fixed_grid_without_refinement(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    directory, progress = tmp_path / "fixed_grid", []
    selected = recommend(project, bundle, evaluation, directory, progress.append)["candidates"]
    records = list(iter_rows(directory / "comparisons.jsonl"))
    expected = {(threshold, count) for threshold in calibration.COARSE_THRESHOLDS
                for count in calibration.CONSECUTIVE_COUNTS}
    for candidate in ids:
        measured = [(row["entry"]["runtime"]["threshold"], row["entry"]["runtime"]["consecutive_frames"])
                    for row in records if row["candidate"] == candidate]
        assert len(measured) == len(set(measured)) == len(expected)
        assert set(measured) == expected
    assert all(row["tested_combinations"] == len(expected) for row in selected)
    assert progress[-1]["completed"] == progress[-1]["total"] == len(expected) * len(ids)
    assert len(progress) == len(expected) * len(ids)  # baseline does not add a duplicate comparison


def test_default_threshold_grid_has_only_one_hundredth_steps():
    expected = tuple(i / 100 for i in range(1, 100))
    assert calibration.COARSE_THRESHOLDS == expected
    assert calibration.SEARCH_POLICY["coarse_thresholds"] == list(expected)
    assert calibration.SEARCH_POLICY["threshold_step"] == .01
    assert calibration.SEARCH_POLICY["refinement"] == "none"
    assert calibration.SEARCH_POLICY["refinement_rounds"] == 0
    assert len(expected) * len(calibration.CONSECUTIVE_COUNTS) == 594


def test_v5_declares_exactly_the_eighteen_reverberant_noise_conditions():
    policy = calibration.SEARCH_POLICY
    assert policy["method"] == calibration.METHOD == "joint_cached_hard_conditions_v5"
    expected = set(EXPECTED_HARD_CONDITIONS)
    declared = policy["hard_conditions"]
    assert len(declared) == len(calibration.HARD_CONDITIONS) == 18
    assert set(calibration.HARD_CONDITIONS) == expected
    assert {(row["reverb_condition"], row["category"], row["snr_db"]) for row in declared} == expected


@pytest.mark.parametrize("false_alarms", [1, 2])
def test_recommendation_ceiling_is_distinct_from_original_adoption(experiment, tmp_path, false_alarms):
    project, bundle, evaluation, _ = experiment
    clips = read_json(evaluation / "clips.json")
    with np.load(evaluation / "scores.npz", allow_pickle=False) as archive:
        scores, offsets, ids = archive["scores"].copy(), archive["offsets"].copy(), archive["model_ids"].copy()
    # One-frame hard calls make that gate outperform the two-frame alternative.
    # The over-limit scenario leaves no strict gate by requiring one frame on
    # the quiet calls too. Both scenarios fail the frozen .1/h adoption target.
    for index, row in enumerate(clips[:-1]):
        if false_alarms == 1 and row["category"] == "clean" and row["reverb_condition"] == "none":
            continue
        scores[offsets[index]:offsets[index + 1], 0] = 0
        scores[offsets[index] + 27, 0] = .65
    warmup = verify_bundle(bundle)["runtime"]["warmup_s"]
    negative_length = int(round((3600 + warmup) / .08))
    negative = np.zeros((negative_length, len(ids)), np.float32)
    negative[29:29 + 20 * false_alarms:20, 0] = .7
    scores = np.concatenate([scores[:offsets[-2]], negative])
    offsets[-1] = len(scores)
    clips[-1].update(duration_s=negative_length * .08, eligible_seconds=3600.)
    np.savez_compressed(evaluation / "scores.npz", scores=scores, offsets=offsets, model_ids=ids)
    write_json(evaluation / "clips.json", clips)
    report = read_json(evaluation / "report.json")
    report["score_cache_sha256"] = digest(evaluation / "scores.npz")
    write_json(evaluation / "report.json", report)
    receipt = read_json(evaluation / "receipt.json")
    receipt.update(report_sha256=digest(evaluation / "report.json"), clips_sha256=digest(evaluation / "clips.json"))
    write_json(evaluation / "receipt.json", receipt)
    protected = {path: digest(path) for path in (Path(project), bundle / "manifest.json",
                 evaluation / "report.json", evaluation / "clips.json", evaluation / "scores.npz")}
    directory = tmp_path / "new_selection_ceiling"
    result = recommend(project, bundle, evaluation, directory)
    selected = result["candidates"][0]
    saved = read_recommendation(project, bundle, evaluation, directory)
    assert selected["runtime"]["consecutive_frames"] == 1
    assert selected["metrics"]["negative"]["fa_per_hour"] == float(false_alarms)
    assert selected["recommendation_available"] and calibration.recommendation_is_available(selected)
    if false_alarms == 1:
        assert selected["status"] == "recommended"
        assert selected["protection_passed"] and selected["recommendation_reasons"] == []
    else:
        assert selected["status"] == "recommended_over_false_alarm_limit"
        assert not selected["protection_passed"]
        assert selected["recommendation_reasons"] == ["FA/hour target failed or unmeasured"]
    assert selected["passed"] is False and "FA/hour target failed or unmeasured" in selected["adoption"]["reasons"]
    assert saved["recommendation_criteria"]["fa_per_hour"] == 1.
    assert saved["adoption_criteria"]["fa_per_hour"] == .1
    assert all(digest(path) == sha for path, sha in protected.items())


def test_v5_selects_the_best_gate_when_snr_zero_breaks_legacy_tie(experiment, tmp_path, monkeypatch):
    project, bundle, evaluation, _ = experiment
    clips = read_json(evaluation / "clips.json")
    with np.load(evaluation / "scores.npz", allow_pickle=False) as archive:
        scores, offsets, ids = archive["scores"], archive["offsets"], archive["model_ids"]
    # One FA/hour is eligible for recommendation. With exactly the same 10 dB
    # and 5 dB recall, the legacy policy chose the gate with zero false alarms.
    warmup = verify_bundle(bundle)["runtime"]["warmup_s"]
    negative_length = int(round((3600 + warmup) / .08))
    negative = np.zeros((negative_length, len(ids)), np.float32)
    negative[29, 0] = .7
    scores = np.concatenate([scores[:offsets[-2]], negative])
    offsets[-1] = len(scores)
    clips[-1].update(duration_s=negative_length * .08, eligible_seconds=3600.)
    np.savez_compressed(evaluation / "scores.npz", scores=scores, offsets=offsets, model_ids=ids)
    write_json(evaluation / "clips.json", clips)
    report = read_json(evaluation / "report.json")
    write_json(evaluation / "report.json", dict(report, score_cache_sha256=digest(evaluation / "scores.npz")))
    receipt = read_json(evaluation / "receipt.json")
    receipt.update(report_sha256=digest(evaluation / "report.json"), clips_sha256=digest(evaluation / "clips.json"))
    write_json(evaluation / "receipt.json", receipt)

    package = verify_bundle(bundle)
    with monkeypatch.context() as legacy:
        legacy.setattr(calibration, "HARD_CONDITIONS", PREVIOUS_HARD_CONDITIONS)
        prepared = PreparedScores(clips, offsets, package["runtime"])
        rows = []
        for count in (1, 2):
            runtime = dict(package["runtime"], threshold=.5, consecutive_frames=count)
            metrics = prepared.summary(scores[:, 0], .5, count, bundle_candidates(package)[0])
            checks = calibration._constraint_checks(metrics, calibration.recommendation_criteria(read_json(project)))
            assert all(checks.values())
            assert metrics["hard_conditions"]["measured_cells"] == 20
            assert metrics["hard_conditions"]["cell_mean"] == 1.
            rows.append({"runtime": runtime, "metrics": metrics, "constraint_checks": checks})
    assert max(rows, key=calibration._rank)["runtime"]["consecutive_frames"] == 2

    preferred = recommend(project, bundle, evaluation, tmp_path / "fresh_policy")["candidates"][0]
    assert preferred["runtime"]["consecutive_frames"] == 1
    assert preferred["metrics"]["negative"]["fa_per_hour"] == 1.
    assert preferred["metrics"]["hard_conditions"]["cell_mean"] == pytest.approx(15.5 / 18)


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
    targets = {"clean_recall": .98, "fa_per_hour": 1, "latency_p95_s": .5, "minimum_negative_hours": 20}
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
    almost_safe = result(.7, .99, 1.2, .2)
    equal_alarms_better_hard = result(.8, .99, 1.2, .2)
    assert calibration._rank(safe) > calibration._rank(equal_alarms_better_hard)
    assert calibration._rank(almost_safe) > calibration._rank(noisy_always_on)
    assert calibration._rank(almost_safe) > calibration._rank(delayed)
    assert calibration._rank(equal_alarms_better_hard) > calibration._rank(almost_safe)


def test_hard_conditions_have_equal_weight_and_easy_or_unknown_conditions_do_not_vote():
    runtime = {"warmup_s": 0, "cooldown_s": 0, "call_postroll_s": .1}
    clips, values = [], []
    def add(condition, success):
        reverb, category, snr = condition
        clips.append({"sample_id": str(len(clips)), "kind": "positive", "category": category,
                      "snr_db": snr, "reverb_condition": reverb, "eligible_seconds": .32,
                      "calls": [{"start_s": .08, "end_s": .24}]})
        values.extend([0, float(success), 0, 0])
    winning_cell = ("strong", "environment", 0)
    for condition in EXPECTED_HARD_CONDITIONS:
        for _ in range(100 if condition == winning_cell else 1):
            add(condition, condition == winning_cell)
    # Dry, reverb-only, and other SNRs do not contribute to the hard objective.
    for condition in [("none", "clean", None), ("none", "environment", 20),
                      ("none", "music", 5), ("none", "music", 0), ("moderate", "clean", None),
                      ("strong", "music", 15), ("unknown", "environment", 5)]:
        for _ in range(40):
            add(condition, True)
    prepared = PreparedScores(clips, np.arange(0, len(values) + 1, 4), runtime)
    summary = prepared.summary(np.asarray(values), .5, 1, {"id": "head", "kind": "single"})
    hard = summary["hard_conditions"]
    assert hard["complete"] and hard["expected_cells"] == hard["measured_cells"] == 18
    assert hard["cell_mean"] == pytest.approx(1 / 18)
    assert hard["worst_recall"] == 0
    assert hard["conditions"]["strong:environment:0"]["calls"] == 100
    assert summary["all_positive"]["recall"] > .8  # pooled recall is a different quantity


@pytest.mark.parametrize("reverb", ["moderate", "none", "unknown"])
def test_missing_reverb_coverage_is_not_a_zero_or_partial_average(reverb):
    runtime = {"warmup_s": 0, "cooldown_s": 0, "call_postroll_s": .1}
    clips = [{"sample_id": "one", "kind": "positive", "category": "environment", "snr_db": 10,
              "reverb_condition": reverb, "eligible_seconds": .32, "calls": [{"start_s": .08, "end_s": .24}]}]
    prepared = PreparedScores(clips, [0, 4], runtime)
    metrics = prepared.summary(np.asarray([0., 1., 0., 0.]), .5, 1, {"id": "head", "kind": "single"})
    hard = metrics["hard_conditions"]
    assert not hard["complete"] and hard["cell_mean"] is None and hard["worst_recall"] is None
    assert hard["measured_cells"] == (1 if reverb == "moderate" else 0)
    assert "strong:music:0" in hard["missing_conditions"]
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


def test_legacy_recommendation_uses_the_measured_false_alarm_exception_only():
    checks = {"clean": True, "false_activations": False, "latency": True, "negative_duration": True,
              "hard_conditions": True}
    row = {"recommendation_available": False, "constraint_checks": checks,
           "metrics": {"negative": {"fa_per_hour": 1.2}}}
    assert calibration.recommendation_is_available(row)
    assert row["recommendation_available"] is False
    for key in ("clean", "latency", "negative_duration", "hard_conditions"):
        assert not calibration.recommendation_is_available(dict(row, constraint_checks=dict(checks, **{key: False})))
        assert not calibration.recommendation_is_available(dict(row, constraint_checks={k: v for k, v in checks.items() if k != key}))
    for rate in (None, -1, float("nan"), float("inf"), True):
        assert not calibration.recommendation_is_available(dict(row, metrics={"negative": {"fa_per_hour": rate}}))
    assert not calibration.recommendation_is_available(dict(row, constraint_checks=dict(checks, additional=False)))
    assert not calibration.recommendation_is_available({"recommendation_available": False})
    assert calibration.recommendation_is_available({"recommendation_available": True})
    assert calibration.recommendation_is_available({})


@pytest.mark.parametrize("policy_name", ["LEGACY_SEARCH_POLICY", "PREVIOUS_SEARCH_POLICY"])
def test_completed_legacy_policy_is_verified_and_never_rewritten(experiment, tmp_path, monkeypatch, policy_name):
    project, bundle, evaluation, _ = experiment
    directory = tmp_path / "legacy"
    policy = dict(getattr(calibration, policy_name), coarse_thresholds=[.5, .9], consecutive_frames=[1, 2])
    monkeypatch.setattr(calibration, policy_name, policy)
    recommend(project, bundle, evaluation, directory)
    publication, receipt_path = directory / "recommendation.json", directory / "receipt.json"
    legacy, receipt = read_json(publication), read_json(receipt_path)
    legacy.update(method=policy["method"], search_policy=policy)
    write_json(publication, legacy)
    write_json(receipt_path, dict(receipt, recommendation_sha256=digest(publication)))
    original = {path: digest(path) for path in directory.iterdir() if path.is_file()}
    assert read_recommendation(project, bundle, evaluation, directory, verify_inputs=False) == legacy
    assert recommend(project, bundle, evaluation, directory)["candidates"] == legacy["candidates"]
    assert {path: digest(path) for path in original} == original

    # A matching hash by itself cannot validate a mismatched policy/method pair.
    write_json(publication, dict(legacy, method=calibration.METHOD))
    write_json(receipt_path, dict(receipt, recommendation_sha256=digest(publication)))
    with pytest.raises(ValueError, match="inputs or receipt changed"):
        read_recommendation(project, bundle, evaluation, directory, verify_inputs=False)
