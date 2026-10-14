"""Selection must deploy the exact judged heads, with no network/audio rerun."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.runtime import (WakeWordDetector, bundle_candidates,
                                        deployment_candidate, sample_bundle, verify_bundle)
from korean_openwakeword.evaluate import adoption, candidate_adoptions, summarize
from korean_openwakeword.tune import replay, tune, select_runtime
from korean_openwakeword.selection import select_candidate, verified_evaluation
from korean_openwakeword.bundle import freeze


@pytest.fixture
def experiment(tmp_path):
    """Tiny synthetic score fixture, not performance evidence for packaged heads."""
    project = create_project(tmp_path / "project", "춘식아", "selection_fixture")
    bundle = tmp_path / "source_bundle"
    shutil.copytree(sample_bundle(), bundle)
    package = verify_bundle(bundle)
    _, config = load_project(project)
    config["runtime"] = deepcopy(package["runtime"])
    config["targets"]["development_negative_hours"] = [.0001, .001]
    write_json(project, config)
    candidates = bundle_candidates(package)
    ids = [row["id"] for row in candidates]
    clips, arrays, offsets = [], [], [0]
    conditions = [("clean", None)] + [(category, snr) for category in ("environment", "music", "competing_speech") for snr in (10, 5)]
    for index, (category, snr) in enumerate(conditions):
        row = {"sample_id": f"p{index}", "original_id": "fixture", "kind": "positive",
               "category": category, "snr_db": snr, "provenance": "qwen_tts",
               "duration_s": 3.6, "eligible_seconds": 3.6 - package["runtime"]["warmup_s"],
               "calls": [{"start_s": 2.0, "end_s": 2.8}]}
        values = np.zeros((45, 4), np.float32)
        values[25:35, :3] = [.99, .999, .1]
        values[:, 3] = ((values[:, 0].astype(np.float64) + values[:, 1].astype(np.float64)
                          + values[:, 2].astype(np.float64)) / 3).astype(np.float32)
        clips.append(row); arrays.append(values); offsets.append(offsets[-1] + len(values))
    clips.append({"sample_id": "n1", "kind": "negative", "category": "environment", "provenance": "public_dataset",
                  "duration_s": 4.0, "eligible_seconds": 4.0 - package["runtime"]["warmup_s"], "calls": []})
    arrays.append(np.zeros((50, 4), np.float32)); offsets.append(offsets[-1] + 50)
    scores, offsets = np.concatenate(arrays), np.asarray(offsets, np.int64)
    clips = replay(clips, scores, offsets, ids, package["runtime"])
    summaries = summarize(clips, config, candidates)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    np.savez_compressed(evaluation / "scores.npz", scores=scores, offsets=offsets, model_ids=np.asarray(ids))
    report = {"schema_version": 1, "wakeword": "춘식아", "stage": "development", "smoke_test_only": False,
              "sessions": len(clips), "runtime": package["runtime"], "models": summaries,
              "candidate_definitions": candidates, "selected_candidate": ids[-1],
              "deployment_candidate": deployment_candidate(package),
              "candidate_adoptions": candidate_adoptions(summaries, config), "adoption": adoption(summaries, config),
              "score_cache_sha256": digest(evaluation / "scores.npz"), "bundle_manifest_sha256": digest(bundle / "manifest.json")}
    write_json(evaluation / "clips.json", clips); write_json(evaluation / "report.json", report)
    _receipt(evaluation, bundle)
    return project, bundle, evaluation, ids


def _receipt(evaluation, bundle):
    write_json(evaluation / "receipt.json", {"report_sha256": digest(evaluation / "report.json"),
                "clips_sha256": digest(evaluation / "clips.json"), "bundle_sha256": digest(bundle / "manifest.json")})


def test_each_candidate_has_independent_adoption_and_no_silent_best_choice(experiment):
    project, bundle, evaluation, ids = experiment
    report = read_json(evaluation / "report.json")
    assert report["candidate_adoptions"][ids[1]]["passed"]
    assert not report["candidate_adoptions"]["mean"]["passed"]
    assert report["adoption"]["candidate"] == "mean"
    assert report["models"][0]["kind"] == "single"
    _, config = load_project(project)
    with pytest.raises(ValueError, match="Unknown evaluated candidate"):
        adoption(report["models"], config, "not_a_model")


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_explicit_selection_is_portable_exact_and_immutable(experiment, tmp_path, monkeypatch, position):
    project, bundle, evaluation, ids = experiment
    protected = {p: digest(p) for directory in (bundle, evaluation) for p in directory.rglob("*") if p.is_file()}
    def network_forbidden(*args, **kwargs):
        raise AssertionError("Selection must not rerun inference")
    monkeypatch.setattr(WakeWordDetector, "__init__", network_forbidden)
    result = select_candidate(project, bundle, evaluation, tmp_path / "selected", ids[position])
    package = verify_bundle(result["bundle"])
    selected = read_json(Path(result["evaluation"]) / "report.json")
    expected = 3 if position == 3 else 1
    assert len(package["members"]) == expected
    assert package["aggregation"] == ("ordered_float64_mean_to_float32" if position == 3 else "single")
    assert deployment_candidate(package)["id"] == ids[position]
    assert selected["adoption"] == read_json(evaluation / "report.json")["candidate_adoptions"][ids[position]]
    assert result["runtime"] == read_json(evaluation / "report.json")["runtime"]
    assert result["network_rerun"] is False
    with np.load(evaluation / "scores.npz") as old, np.load(Path(result["evaluation"]) / "scores.npz") as new:
        expected_scores = old["scores"] if position == 3 else old["scores"][:, [position]]
        assert np.array_equal(new["scores"].view("u4"), expected_scores.view("u4"))
        assert np.array_equal(old["offsets"], new["offsets"])
    assert protected == {p: digest(p) for p in protected}
    assert digest(Path(result["bundle"]) / "THIRD_PARTY_NOTICES.md") == digest(bundle / "THIRD_PARTY_NOTICES.md")
    verified_evaluation(project, result["bundle"], result["evaluation"])
    with pytest.raises(ValueError, match="immutable"):
        select_candidate(project, bundle, evaluation, tmp_path / "selected", ids[position])


def test_select_legacy_completed_report_without_rewriting_it(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    report = read_json(evaluation / "report.json")
    for key in ("candidate_definitions", "selected_candidate", "deployment_candidate", "candidate_adoptions"):
        report.pop(key)
    write_json(evaluation / "report.json", report); _receipt(evaluation, bundle)
    original = digest(evaluation / "report.json")
    result = select_candidate(project, bundle, evaluation, tmp_path / "selected", ids[1])
    assert result["adoption"]["passed"]
    assert digest(evaluation / "report.json") == original


def test_freeze_rejects_single_judgement_with_three_deployed_heads(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    report = read_json(evaluation / "report.json")
    report["selected_candidate"] = ids[1]
    report["adoption"] = report["candidate_adoptions"][ids[1]]
    write_json(evaluation / "report.json", report); _receipt(evaluation, bundle)
    with pytest.raises(ValueError, match="deployed members/aggregation"):
        freeze(project, bundle, evaluation / "report.json")
    result = select_candidate(project, bundle, evaluation, tmp_path / "selected", ids[1])
    lock = freeze(project, result["bundle"], Path(result["evaluation"]) / "report.json")
    assert lock["selected_candidate"]["id"] == ids[1]
    assert len(lock["selected_candidate"]["members"]) == 1
    with pytest.raises(ValueError, match="frozen"):
        select_candidate(project, bundle, evaluation, tmp_path / "after_final", ids[1])


def test_tune_default_and_explicit_heads_are_actually_deployed(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    threshold = read_json(evaluation / "report.json")["runtime"]["threshold"]
    default = tune(project, bundle, evaluation, tmp_path / "default", [threshold])
    explicit = tune(project, bundle, evaluation, tmp_path / "single", [threshold], candidate=ids[1])
    assert default["candidate"] == "mean" and not default["adoption"]["passed"]
    assert len(verify_bundle(default["bundle"])["members"]) == 3
    assert explicit["candidate"] == ids[1] and explicit["adoption"]["passed"]
    assert len(verify_bundle(explicit["bundle"])["members"]) == 1
    verified_evaluation(project, explicit["bundle"], explicit["evaluation"])
    assert read_json(Path(explicit["evaluation"]) / "report.json")["models"][0]["id"] == ids[1]


@pytest.mark.parametrize("change", ["score_hash", "candidate_hash", "columns", "duration", "final_stage"])
def test_selection_rejects_invalid_evidence(experiment, tmp_path, change):
    project, bundle, evaluation, ids = experiment
    report = read_json(evaluation / "report.json")
    if change == "score_hash":
        report["score_cache_sha256"] = "0" * 64
    elif change == "candidate_hash":
        report["candidate_definitions"][1]["members"][0]["sha256"] = "0" * 64
    elif change == "columns":
        with np.load(evaluation / "scores.npz") as cache:
            np.savez_compressed(evaluation / "scores.npz", scores=cache["scores"], offsets=cache["offsets"], model_ids=np.asarray(ids[::-1]))
        report["score_cache_sha256"] = digest(evaluation / "scores.npz")
    elif change == "duration":
        clips = read_json(evaluation / "clips.json"); clips[-1]["eligible_seconds"] = 36000
        write_json(evaluation / "clips.json", clips)
    else:
        report["stage"] = "final"
    write_json(evaluation / "report.json", report); _receipt(evaluation, bundle)
    with pytest.raises(ValueError):
        select_candidate(project, bundle, evaluation, tmp_path / "invalid", ids[1])


def test_mean_named_head_has_distinct_ensemble_identity(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    package = read_json(bundle / "manifest.json")
    original = package["members"][0]["path"]
    package["members"][0]["path"] = "models/mean.onnx"
    shutil.copyfile(bundle / original, bundle / "models/mean.onnx")
    write_json(bundle / "manifest.json", package)
    candidates = bundle_candidates(verify_bundle(bundle))
    assert candidates[0]["id"] == "mean" and candidates[0]["kind"] == "single"
    assert candidates[-1]["id"] == "_mean" and candidates[-1]["kind"] == "ensemble"


def test_bundle_rejects_selected_metadata_that_disagrees_with_actual_heads(experiment):
    project, bundle, evaluation, ids = experiment
    package = read_json(bundle / "manifest.json")
    package["selected_candidate"] = bundle_candidates(package)[1]
    write_json(bundle / "manifest.json", package)
    with pytest.raises(ValueError, match="deployed members/aggregation"):
        verify_bundle(bundle)


def test_runtime_can_be_explicitly_selected_from_candidate_bundle(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    result = select_candidate(project, bundle, evaluation, tmp_path / "selected", ids[1])
    original = read_json(project)
    outcome = select_runtime(project, result["bundle"])
    assert outcome["runtime"] == original["runtime"]
    assert read_json(project) == original


def test_final_cannot_select_a_different_candidate(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    from korean_openwakeword.evaluate import evaluate
    with pytest.raises(ValueError, match="frozen deployed candidate"):
        evaluate(project, bundle, tmp_path / "forbidden_final", stage="final", candidate=ids[1])


def test_actual_single_head_scores_and_events_match_original_evaluation(experiment, tmp_path):
    project, bundle, evaluation, ids = experiment
    result = select_candidate(project, bundle, evaluation, tmp_path / "selected", ids[1])
    original = WakeWordDetector(bundle)
    single = WakeWordDetector(result["bundle"])
    from korean_openwakeword.events import EventDetector
    runtime = original.runtime
    gate = EventDetector(runtime["threshold"], runtime["consecutive_frames"], runtime["cooldown_s"])
    audio = read_json(bundle / "audio_manifest.json")[0]
    seen = 0
    with sf.SoundFile(bundle / audio["path"]) as sound:
        while True:
            frame = sound.read(1280, dtype="int16")
            if len(frame) != 1280:
                break
            score = original.score_members(frame)[1]
            actual = single.process(frame)
            decision = gate.update(float(score), (seen + 1) * .08, seen * .08 + 1e-9 >= runtime["warmup_s"])
            assert np.float32(actual["score"]).view("u4") == score.view("u4")
            assert actual["event"] == decision["event"]
            seen += 1
    assert seen > 25
