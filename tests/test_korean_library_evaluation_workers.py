"""Worker budgets and interrupted score-cache recovery must preserve results."""
from pathlib import Path
import json

import numpy as np
import pytest

from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.evaluation_workers import EvaluationJournal, worker_plan


def test_auto_cpu_workers_are_small_for_short_runs_and_memory_bounded():
    ample = 12 * 1024 ** 3
    assert worker_plan("auto", 20, logical_cpus=16, available_bytes=ample)["workers"] == 1
    assert worker_plan("auto", 100000, logical_cpus=16, available_bytes=ample)["workers"] == 8
    assert worker_plan("auto", 100000, logical_cpus=64, available_bytes=ample)["workers"] == 8
    assert worker_plan("auto", 100000, logical_cpus=2, available_bytes=ample)["workers"] == 1
    assert worker_plan("auto", 100000, logical_cpus=16, available_bytes=3 * 1024 ** 3)["workers"] == 1
    assert worker_plan("16", 3, logical_cpus=8, available_bytes=ample)["workers"] == 3


@pytest.mark.parametrize("value", [0, -1, 17, "cuda", True, 1.5])
def test_invalid_workers_are_rejected(value):
    with pytest.raises(ValueError, match="workers"):
        worker_plan(value, 100)


def _add(journal, index):
    scores = np.asarray([[index / 100, (index + 1) / 100], [index / 99, 0]], dtype=np.float32)
    journal.add(index, {"sample_id": f"slot{index}", "results": {"head": {"value": index}}}, scores)
    return scores


def test_interrupted_journal_reuses_only_committed_frames_in_original_order(tmp_path):
    offsets = np.arange(35, dtype=np.int64) * 2
    journal = EvaluationJournal(tmp_path, "same-input-hashes", offsets, ["a", "b"])
    expected = [_add(journal, index) for index in range(33)]
    journal.close()
    checkpoint = read_json(tmp_path / "_evaluation_work/checkpoint.json")
    assert checkpoint["committed_sessions"] == 32
    # Simulate an append interrupted after checkpoint publication.
    with (tmp_path / "_evaluation_work/clips.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":')
    resumed = EvaluationJournal(tmp_path, "same-input-hashes", offsets, ["a", "b"], resume=True)
    assert resumed.resumed_sessions == 32
    assert [row["sample_id"] for row in resumed.clips] == [f"slot{i}" for i in range(32)]
    _add(resumed, 32)
    expected.append(_add(resumed, 33))
    resumed.publish_scores()
    resumed.publish_clips()
    assert (tmp_path / "clips.json").read_text(encoding="utf-8") == json.dumps(resumed.clips, ensure_ascii=False, indent=2) + "\n"
    with np.load(tmp_path / "scores.npz") as cache:
        assert np.array_equal(cache["scores"].view(np.uint32), np.concatenate(expected).view(np.uint32))
        assert np.array_equal(cache["offsets"], offsets)
        assert cache["model_ids"].tolist() == ["a", "b"]
    write_json(tmp_path / "receipt.json", {"completed": True})
    resumed.discard_completed_work()
    assert not (tmp_path / "_evaluation_work").exists()


def test_resume_rejects_changed_model_or_manifest_binding(tmp_path):
    journal = EvaluationJournal(tmp_path, "original-hashes", [0, 2], ["a", "b"])
    _add(journal, 0)
    journal.commit()
    journal.close()
    with pytest.raises(ValueError, match="changed"):
        EvaluationJournal(tmp_path, "different-model-hash", [0, 2], ["a", "b"], resume=True)


def test_resume_rejects_modified_committed_scores(tmp_path):
    journal = EvaluationJournal(tmp_path, "inputs", [0, 2], ["a", "b"])
    _add(journal, 0)
    journal.commit()
    journal.close()
    frames = np.load(tmp_path / "_evaluation_work/frames.npy", mmap_mode="r+")
    frames[0, 0] = 0.99
    frames.flush()
    frames._mmap.close()
    with pytest.raises(ValueError, match="score prefix changed"):
        EvaluationJournal(tmp_path, "inputs", [0, 2], ["a", "b"], resume=True)


def test_resume_rejects_modified_committed_clip_metadata(tmp_path):
    journal = EvaluationJournal(tmp_path, "inputs", [0, 2], ["a", "b"])
    _add(journal, 0)
    journal.commit()
    journal.close()
    path = tmp_path / "_evaluation_work/clips.jsonl"
    path.write_bytes(path.read_bytes().replace(b"slot0", b"slot1"))
    with pytest.raises(ValueError, match="clip journal changed"):
        EvaluationJournal(tmp_path, "inputs", [0, 2], ["a", "b"], resume=True)


def test_checkpoint_cannot_be_discarded_before_completion(tmp_path):
    journal = EvaluationJournal(tmp_path, "inputs", [0, 0], ["a"])
    journal.add(0, {"sample_id": "empty"}, np.zeros((0, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="completion receipt"):
        journal.discard_completed_work()
    assert (tmp_path / "_evaluation_work/frames.npy").exists()


def test_reverb_summary_keeps_each_noise_cell_and_pairwise_drop_separate():
    from korean_openwakeword.evaluate import summarize
    rows = []
    for condition, clean_score, noise_score, success in ((None, .9, .8, True),
                                                        ("moderate", .7, .6, True),
                                                        ("strong", .5, .2, False)):
        for category, score in (("clean", clean_score), ("environment", noise_score)):
            call = {"success": success, "original_utterance_id": "same-source", "provenance": "qwen_tts",
                    "latency_s": .08 if success else None}
            rows.append({"id": f"{condition}-{category}", "kind": "positive", "category": category,
                         "snr_db": 10 if category == "environment" else None,
                         "reverb_condition": condition, "results": {"head": {"calls": [call],
                             "clean_score_drop": clean_score - score,
                             "dry_clean_score_drop": .9 - score}}})
    summary = summarize(rows, {}, [{"id": "head", "kind": "single"}])[0]
    assert summary["all_positive"]["calls"] == 6
    assert summary["all_positive"]["successes"] == 4
    assert summary["clean"]["calls"] == 1
    assert summary["clean"]["recall"] == 1
    assert summary["conditions"]["environment:10"]["calls"] == 1
    assert summary["noise_cell_mean"] == 1
    groups = summary["reverb_conditions"]
    assert set(groups) == {"none", "moderate", "strong"}
    assert groups["none"]["all_positive"]["calls"] == 2
    assert groups["strong"]["conditions"]["environment:10"]["recall"] == 0
    assert groups["moderate"]["clean_score_drop_from_dry_clean"]["mean"] == pytest.approx(.2)
    assert groups["strong"]["noise_score_drop_from_same_reverb_clean"]["mean"] == pytest.approx(.3)


def test_gate_replay_preserves_same_room_and_dry_clean_comparisons():
    from korean_openwakeword.common import read_json
    from korean_openwakeword.runtime import sample_bundle
    from korean_openwakeword.tune import replay
    clips, arrays = [], []
    for index, (condition, category, score) in enumerate(((None, "clean", .9), (None, "environment", .8),
                                                        ("moderate", "clean", .7), ("moderate", "environment", .6))):
        clips.append({"sample_id": f"sample{index}", "original_id": "one-original", "source_id": "one-original",
                      "kind": "positive", "provenance": "qwen_tts", "category": category,
                      "reverb_condition": condition, "calls": [{"start_s": 2, "end_s": 2.8}]})
        arrays.append(np.full((45, 1), score, np.float32))
    runtime = read_json(sample_bundle() / "manifest.json")["runtime"]
    rows = replay(clips, np.concatenate(arrays), np.arange(5) * 45, ["head"], runtime)
    assert rows[1]["results"]["head"]["clean_score_drop"] == pytest.approx(.1)
    assert rows[3]["results"]["head"]["clean_score_drop"] == pytest.approx(.1)
    assert rows[2]["results"]["head"]["dry_clean_score_drop"] == pytest.approx(.2)
    assert rows[3]["results"]["head"]["dry_clean_score_drop"] == pytest.approx(.3)


def _binding_inputs(tmp_path):
    project = tmp_path / "project.json"
    manifest = tmp_path / "evaluation.jsonl"
    bundle = tmp_path / "bundle"
    write_json(project, {"runtime_setting": 1})
    manifest.write_text("fixed audio manifest\n", encoding="utf-8")
    write_json(bundle / "manifest.json", {"members": [{"path": f"head{i}.onnx", "sha256": str(i) * 64}
                                                       for i in range(3)]})
    return project, bundle, manifest


def test_evaluation_resume_identity_binds_config_manifest_bundle_stage_selection_and_scope(tmp_path):
    from korean_openwakeword.evaluate import evaluation_binding
    project, bundle, manifest = _binding_inputs(tmp_path)
    original = evaluation_binding(project, bundle, manifest)
    assert evaluation_binding(project, bundle, manifest, candidate="mean") == original
    assert evaluation_binding(project, bundle, manifest, candidate="head0") != original
    assert evaluation_binding(project, bundle, manifest, stage="final") != original
    assert evaluation_binding(project, bundle, manifest, smoke=True) != original
    write_json(project, {"runtime_setting": 2})
    assert evaluation_binding(project, bundle, manifest) != original
    write_json(project, {"runtime_setting": 1})
    manifest.write_text("changed audio manifest\n", encoding="utf-8")
    assert evaluation_binding(project, bundle, manifest) != original
    manifest.write_text("fixed audio manifest\n", encoding="utf-8")
    package = read_json(bundle / "manifest.json")
    package["members"] = list(reversed(package["members"]))
    write_json(bundle / "manifest.json", package)
    assert evaluation_binding(project, bundle, manifest) != original


@pytest.mark.parametrize("same_binding", [True, False])
def test_studio_reuses_only_an_incomplete_evaluation_with_exact_input_binding(tmp_path, monkeypatch, same_binding):
    from korean_openwakeword import evaluate, studio_steps
    from korean_openwakeword.project import create_project
    settings = studio_steps.preset_settings("check")
    # The mocked scorer below publishes no scores.npz. Keep its historical
    # workflow policy so this test isolates checkpoint binding and reuse.
    settings.pop("runtime_recommendation_policy")
    settings["evaluation_backend"] = "cpu"
    project = create_project(tmp_path / "word", "보라야", "resume_fixture",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    _, bundle, manifest = _binding_inputs(tmp_path)
    manifest.write_text("", encoding="utf-8")
    base = project.parent / "reports/studio_evaluation"
    expected = evaluate.evaluation_binding(project, bundle, manifest, smoke=False)
    write_json(base / "_evaluation_work/checkpoint.json", {"schema_version": 1,
               "binding": expected if same_binding else "different_frozen_inputs"})
    checkpoint_bytes = (base / "_evaluation_work/checkpoint.json").read_bytes()
    calls = []
    def score(path, selected_bundle, output, **kwargs):
        calls.append((Path(output), kwargs))
        write_json(output / "report.json", {"models": [], "smoke_test_only": kwargs["smoke"]})
        write_json(output / "receipt.json", {})
        write_json(output / "clips.json", [])
        return read_json(output / "report.json")
    monkeypatch.setattr(evaluate, "evaluate", score)
    context = {"results": {"package": {"result": {"bundle": str(bundle)}},
                           "development": {"result": {"manifest": str(manifest)}}}}
    result = studio_steps.run_stage(project, tmp_path / "workspace", settings, "evaluation", context)
    output, kwargs = calls[0]
    assert output == (base if same_binding else base.with_name(base.name + "_attempt2"))
    assert kwargs["resume"] is same_binding
    assert kwargs["workers"] == "auto" and kwargs["smoke"] is False
    assert result["state"] == "completed"
    assert (base / "_evaluation_work/checkpoint.json").read_bytes() == checkpoint_bytes


def test_studio_reuses_verified_completed_evaluation_before_any_partial(tmp_path, monkeypatch):
    from korean_openwakeword import evaluate, selection, studio_steps
    from korean_openwakeword.common import digest
    from korean_openwakeword.project import create_project
    settings = studio_steps.preset_settings("check")
    # A report-only completed fixture exercises reuse without a gate search.
    settings.pop("runtime_recommendation_policy")
    settings["evaluation_backend"] = "cpu"
    project = create_project(tmp_path / "word", "보라야", "completed_fixture",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    _, bundle, manifest = _binding_inputs(tmp_path)
    manifest.write_text("", encoding="utf-8")
    base = project.parent / "reports/studio_evaluation"
    write_json(base / "report.json", {"models": [], "manifest_sha256": digest(manifest),
                                      "bundle_manifest_sha256": digest(bundle / "manifest.json"),
                                      "execution": {"backend": "cpu"},
                                      "evaluation_binding": evaluate.evaluation_binding(
                                          project, bundle, manifest, smoke=False, backend="cpu")})
    write_json(base / "receipt.json", {"completed": True})
    write_json(base / "clips.json", [])
    verified = []
    monkeypatch.setattr(selection, "verified_evaluation", lambda *args: verified.append(args))
    monkeypatch.setattr(evaluate, "evaluate", lambda *args, **kwargs: pytest.fail("Completed scores must not rerun"))
    context = {"results": {"package": {"result": {"bundle": str(bundle)}},
                           "development": {"result": {"manifest": str(manifest)}}}}
    result = studio_steps.run_stage(project, tmp_path / "workspace", settings, "evaluation", context)
    assert result["result"]["evaluation"] == str(base)
    assert len(verified) == 1
