"""Voice budget expansion preserves held-out identities and human review work."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from korean_openwakeword.common import digest, read_json, read_rows
from korean_openwakeword.project import create_project, voice_candidates
from korean_openwakeword.review import apply_review
from korean_openwakeword import studio_steps as steps


@pytest.mark.parametrize("validation_count, expected", [
    (10, "e170abe8d9a2ea9b97ca8dbbbc60b55df03b4eb952416289f82369b9ea989761"),
    (30, "cd8fe19404737fa761bcebf556533ff5dbe6e8a46afedb71e4ff560f7b89b385"),
    (100, "c8f536f5daba972143f7406db164fee0e5287ed66a2660505fbe4e2eac5789b2"),
])
def test_legacy_voice_bank_matches_frozen_pre_expansion_identity(validation_count, expected):
    payload = json.dumps(voice_candidates(validation_count), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(payload).hexdigest() == expected


def test_five_hundred_train_candidates_preserve_held_out_and_old_training_personas(tmp_path):
    legacy = voice_candidates(30)["families"]
    expanded = voice_candidates(30, train_count=500)["families"]
    assert expanded[:len(legacy)] == legacy
    assert Counter(row["split"] for row in expanded) == {"train": 500, "validation": 30, "final": 10}
    assert len({row["family_id"] for row in expanded}) == 540
    extra = expanded[len(legacy):]
    assert len({row["description"] for row in extra}) == 400
    assert not ({row["description"] for row in extra} & {row["description"] for row in legacy})
    project = create_project(tmp_path / "expanded", "보라야", "bora", validation_voice_families=30,
                             voice_policy=None)
    assert read_json(project.parent / "voices.json")["families"] == expanded


@pytest.mark.parametrize("name, train_count", [("check", 20), ("trial", 100), ("production", 500)])
def test_all_presets_select_new_training_count_and_same_common_voices(name, train_count):
    settings = steps.preset_settings(name)
    assert settings["train_references"] == train_count
    assert settings["validation_references"] == 30
    assert settings["validation_originals"] == 2000
    assert steps.workflow_plan(settings)


@pytest.mark.parametrize("count", [0, 501, True, 20.0])
def test_invalid_bank_limits_fail(count):
    with pytest.raises(ValueError, match="Training synthetic voice"):
        voice_candidates(train_count=count)


def _rows(root, train_count):
    rows = []
    audio = root / "fixture_audio"
    audio.mkdir(exist_ok=True)
    for split, count in (("train", train_count), ("validation", 30)):
        for index in range(count):
            name = f"{split}_{index}"
            path = audio / (name + ".wav")
            path.write_bytes((name + " fixed synthetic-only fixture").encode())
            rows.append({"sample_id": name, "voice_family_id": name, "split": split,
                         "path": str(path.relative_to(root)), "sha256": digest(path)})
    return rows


def _save_answers(project, queue):
    decisions = {row["sample_id"]: {"decision": "keep", "listened": True} for row in read_rows(queue)}
    apply_review(project, queue, decisions, "references")
    return read_json(project.parent / "approval.json")["reference_review_evidence"][-1]


def test_reference_growth_keeps_existing_queue_and_saved_answers_and_only_adds_sixteen(tmp_path):
    settings = steps.preset_settings()
    project = create_project(tmp_path / "reviewed", "보라야", "bora",
                             positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                             updates=settings["updates"], generic_hours=settings["generic_hours"],
                             korean_hours=settings["korean_hours"])
    root = project.parent
    workspace = root / ".studio"
    workspace.mkdir()
    original_rows = _rows(root, 4)
    base = steps._publish_reference_queues(workspace, original_rows)[0]
    answer = _save_answers(project, base)
    original_bytes = base.read_bytes()
    original_approval = (root / "approval.json").read_bytes()
    queues = steps._publish_reference_queues(workspace, _rows(root, 20))
    assert base.read_bytes() == original_bytes
    assert (root / "approval.json").read_bytes() == original_approval
    assert queues[0] == base
    assert len(queues) == 2
    assert len(read_rows(queues[1])) == 16
    context = {"results": {"references": {"queues": [str(path) for path in queues]}},
               "inputs": {"review_answers": {"references": answer}}}
    pending = steps.run_stage(project, tmp_path, steps.preset_settings(), "references_review", context)
    assert pending["state"] == "waiting_review"
    assert len(pending["reviews"]) == 1
    assert pending["reviews"][0]["count"] == 16
    descriptor = pending["reviews"][0]
    context["inputs"]["review_answers"][descriptor["id"]] = _save_answers(project, queues[1])
    complete = steps.run_stage(project, tmp_path, steps.preset_settings(), "references_review", context)
    assert complete["state"] == "completed" and complete["result"]["reviewed_queues"] == 2
    assert steps._publish_reference_queues(workspace, _rows(root, 20), queues) == queues
    assert base.read_bytes() == original_bytes


@pytest.mark.parametrize("tamper", [False, True])
def test_cli_reference_answers_survive_supplemental_summary_and_verify_archived_hash(tmp_path, tamper):
    settings = steps.preset_settings()
    project = create_project(tmp_path / "cli_reviewed", "보라야", "bora",
                             positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                             updates=settings["updates"], generic_hours=settings["generic_hours"],
                             korean_hours=settings["korean_hours"])
    root = project.parent
    workspace = root / ".studio"
    workspace.mkdir()
    base = steps._publish_reference_queues(workspace, _rows(root, 4))[0]
    original_evidence = _save_answers(project, base)
    base_hash = digest(base)
    queues = steps._publish_reference_queues(workspace, _rows(root, 20))
    _save_answers(project, queues[1])
    assert read_json(root / "review/references_results.json")["queue_sha256"] != base_hash
    context = {"results": {"references": {"queues": [str(path) for path in queues]}}}
    if tamper:
        (root / original_evidence["path"]).write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="changed bound file"):
            steps.run_stage(project, tmp_path, settings, "references_review", context)
    else:
        result = steps.run_stage(project, tmp_path, settings, "references_review", context)
        assert result["state"] == "completed" and result["result"]["reviewed_queues"] == 2
    assert digest(base) == base_hash


@pytest.mark.parametrize("change", ["missing", "metadata"])
def test_reference_growth_rejects_changes_to_previously_published_audio_metadata(tmp_path, change):
    workspace = tmp_path / ".studio"
    workspace.mkdir()
    original = _rows(tmp_path, 4)
    base = steps._publish_reference_queues(workspace, original)[0]
    original_bytes = base.read_bytes()
    rows = deepcopy(_rows(tmp_path, 20))
    if change == "missing":
        rows = rows[1:]
    else:
        rows[0]["text"] = "changed description"
    with pytest.raises(ValueError, match="기존 합성 목소리"):
        steps._publish_reference_queues(workspace, rows)
    assert base.read_bytes() == original_bytes
    assert len(list(workspace.glob("*.jsonl"))) == 1


def test_legacy_bank_cannot_claim_five_hundred_candidates_or_load_model(tmp_path, monkeypatch):
    from korean_openwakeword import tts
    settings = steps.preset_settings("production")
    project = create_project(tmp_path / "legacy", "보라야", "bora", train_voice_families=100,
                             positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                             updates=settings["updates"], generic_hours=settings["generic_hours"],
                             korean_hours=settings["korean_hours"])
    monkeypatch.setattr(tts, "_load_model", lambda *args: pytest.fail("An insufficient bank must fail before GPU loading"))
    with pytest.raises(ValueError, match="100/500"):
        steps.run_stage(project, tmp_path, settings, "references")
