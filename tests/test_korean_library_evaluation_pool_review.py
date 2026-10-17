"""Representative consent covers a frozen group without inventing listening."""
from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project
from korean_openwakeword import studio_evaluation_data as module
from korean_openwakeword.studio_data import _write_rows
from test_korean_library_studio_evaluation_data import saved_answers


def group_preparation(tmp_path, *, groups=(5,), rir_groups=()):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    root, source_rows, preview_rows, specs = project.parent, [], [], {}
    for group_no, count in enumerate(groups):
        group_id = f"group_{group_no}"
        members = []
        for index in range(count):
            sid = f"source_{group_no}_{index}"
            path = root / "data" / (sid + ".wav")
            sf.write(path, np.random.default_rng(1000 + group_no * 100 + index).normal(0, .03, 16000),
                     16000, subtype="PCM_16")
            pcm_hash, frames = module._pcm_hash(path)
            row = {"sample_id": sid, "canonical_source_id": "fixture:" + sid,
                "canonical_recording_id": "fixture:" + sid, "path": str(path), "sha256": digest(path),
                "original_sha256": digest(path), "split": "validation", "category": "environment",
                "normalized_pcm_sha256": pcm_hash, "normalized_frames": frames,
                "eval_type": "fixture_type_" + str(group_no), "duration_s": 1, "provenance": "public_dataset",
                "source_url": "https://example.test/isolated-review-fixture", "license": "isolated-test-fixture",
                "review_group_id": group_id, "evaluation_only": True, "training_admitted": False,
                "individual_human_listening": False}
            if group_no in rir_groups:
                row.update(category="rir", eval_type="rir_fixture_" + str(group_no),
                           canonical_room_id=f"fixture_room_{group_no}_{index // 4}", speech=False)
            members.append(row); source_rows.append(row)
        first = members[0]
        preview = {**first, "sample_id": "preview_" + first["sample_id"],
            "reviewed_source": {"path": first["path"], "sha256": first["sha256"]},
            "review_excerpt": {"scope": "representative_excerpt_not_full_source"}}
        preview_rows.append(preview)
        specs[group_id] = {"source_ids": [r["sample_id"] for r in members], "preview_ids": [preview["sample_id"]]}
    output = root / ".studio/materials/evaluation_v5_fixture"
    source = _write_rows(output / "evaluation_sources.jsonl", source_rows)
    queue_file = _write_rows(output / "representatives.jsonl", preview_rows)
    receipt = {"kind": "studio_evaluation_noise_preparation_v5", "recipe_version": 5,
        "wakeword": "보라야", "evaluation_only": True, "training_admitted": False,
        "intended_use": "held_out_development_noise", "state": "awaiting_human_review",
        "artifacts": {"source_manifest": source, "representative_queue": queue_file},
        "representative_groups": specs, "human_listened_count": 0}
    path = output / "preparation.json"
    write_json(path, receipt)
    prep = {**receipt, "preparation": {"path": str(path), "sha256": digest(path)}}
    queue = {"id": "fixture_v5", "evaluation_only": True, "pools": [], "preparation": str(path),
             "queue": queue_file["path"], "pool_manifest": source["path"]}
    return project, prep, queue


def test_one_genuinely_saved_representative_admits_five_sources_without_five_listening_claims(tmp_path):
    project, prep, queue = group_preparation(tmp_path)
    answers, _ = saved_answers(project, prep, queue)
    admitted = module.admit_evaluation_review(project, queue, answers, {})
    record = read_json(admitted["admission"]["path"])
    assert record["human_listened_count"] == 1 and record["kept_count"] == 5
    assert len(admitted["rows"]) == 5 and record["training_admitted"] is False
    assert all(row["individual_human_listening"] is False for row in admitted["rows"])
    assert all(row["evaluation_content_review_scope"]["scope"] == "stratified_representative_pool_not_individual_source" for row in admitted["rows"])
    assert not read_rows(project.parent / read_json(project)["manifests"]["noise"])


def test_discard_of_one_representative_holds_its_entire_group_and_keeps_other_group(tmp_path):
    project, prep, queue = group_preparation(tmp_path, groups=(5, 2))
    answers, saved = saved_answers(project, prep, queue)
    saved["decisions"]["preview_source_0_0"]["decision"] = "discard"
    write_json(answers, saved)
    admitted = module.admit_evaluation_review(project, queue, answers, {})
    record = read_json(admitted["admission"]["path"])
    assert record["kept_count"] == 2 and record["discarded_count"] == 5
    assert record["human_listened_count"] == 2
    assert {row["review_group_id"] for row in admitted["rows"]} == {"group_1"}


@pytest.mark.parametrize("change", ["missing_member", "unknown_preview", "other_group_preview"])
def test_changed_frozen_group_membership_or_unrelated_preview_cannot_approve_sources(tmp_path, change):
    project, prep, queue = group_preparation(tmp_path, groups=(5, 2))
    answers, _ = saved_answers(project, prep, queue)
    record = read_json(queue["preparation"])
    spec = record["representative_groups"]["group_0"]
    if change == "missing_member":
        spec["source_ids"].pop()
    elif change == "unknown_preview":
        spec["preview_ids"] = ["preview_unknown_source"]
    else:
        spec["preview_ids"] = ["preview_source_1_0"]
    write_json(queue["preparation"], record)
    with pytest.raises(ValueError, match="frozen source group|another group"):
        module.admit_evaluation_review(project, queue, answers, {})


def test_unheard_default_keep_is_not_genuine_representative_consent(tmp_path):
    project, prep, queue = group_preparation(tmp_path)
    answers, saved = saved_answers(project, prep, queue)
    saved["decisions"]["preview_source_0_0"]["listened"] = False
    write_json(answers, saved)
    with pytest.raises(ValueError, match="Unheard"):
        module.admit_evaluation_review(project, queue, answers, {})


def test_preview_from_unknown_original_is_rejected_even_with_rebound_queue_answers(tmp_path):
    project, prep, queue = group_preparation(tmp_path)
    previews = read_rows(queue["queue"])
    previews[0]["canonical_source_id"] = "not_a_frozen_source"
    Path(queue["queue"]).write_text("".join(json.dumps(row) + "\n" for row in previews), encoding="utf-8")
    binding = {"path": queue["queue"], "sha256": digest(queue["queue"])}
    prep["artifacts"]["representative_queue"] = binding
    record = read_json(queue["preparation"])
    record["artifacts"]["representative_queue"] = binding
    write_json(queue["preparation"], record)
    answers, _ = saved_answers(project, prep, queue)
    with pytest.raises(ValueError, match="differs from its actual source"):
        module.admit_evaluation_review(project, queue, answers, {})


def test_admit_materials_keeps_rir_filters_out_of_additive_noise_and_negative_manifests(tmp_path, monkeypatch):
    from korean_openwakeword import evaluation_materials, preparation, studio_data
    project, prep, queue = group_preparation(tmp_path, groups=(5, 40, 40), rir_groups=(1, 2))
    answers, _ = saved_answers(project, prep, queue)
    root = project.parent
    # This integration fixture supplies a tiny pre-existing Train admission;
    # it tests routing, not corpus hours or the real admission procedure.
    reserve = _write_rows(root / "review/fixture_empty_train_reserve.jsonl", [])
    receipt_path = root / "review/fixture_existing_train_admission.json"
    write_json(receipt_path, {"kind": "prepared_public_audio_admission_v1", "state": "accepted",
                             "source_manifests": {}, "manifests": {}, "evaluation_reserve": reserve})
    monkeypatch.setattr(preparation, "connect_materials", lambda *args, **kwargs: {"fixture_connection": True})
    monkeypatch.setattr(evaluation_materials, "evaluation_material_coverage", lambda rows: {"target_met": True, "shortages": []})
    result = studio_data.admit_materials(project, {"recipe_version": 5, "review_queues": [queue]},
        {queue["id"]: answers}, admissions=[{"path": str(receipt_path), "sha256": digest(receipt_path)}])
    assert result["state"] == "connected"
    noise = read_rows(result["validation_noise_manifest"]["path"])
    negative = read_rows(result["negative_validation_manifest"]["path"])
    rir = read_rows(result["validation_rir_manifest"]["path"])
    assert len(noise) == len(negative) == 5 and all(row["category"] != "rir" for row in noise + negative)
    assert len(rir) == 80 and len({row["canonical_room_id"] for row in rir}) == 20
    assert all(row["kind"] == "negative" for row in negative)
    config = read_json(project)
    assert read_rows(root / config["manifests"]["rir_validation"]) == rir
    assert not read_rows(root / config["manifests"]["rir"])
