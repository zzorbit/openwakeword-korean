"""Korean interference stays held out and is balanced alongside other sounds."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import paired
from korean_openwakeword import studio_data
from korean_openwakeword import studio_evaluation_data as evaluation
from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project


def test_v3_preserves_pinned_v2_sources_and_adds_real_korean_speech():
    old = read_json(files("korean_openwakeword") / "resources/evaluation_noise_v2.json")
    current = read_json(files("korean_openwakeword") / "resources/evaluation_noise_v3.json")
    assert current["recipe_version"] == 3
    old_by_id = {r["id"]: r for r in old["assets"]}
    current_by_id = {r["id"]: r for r in current["assets"]}
    assert len(current_by_id) == len(current["assets"]) == 40
    for key, row in old_by_id.items():
        assert current_by_id[key] == row, "Existing source pins are immutable"
    additions = [r for key, r in current_by_id.items() if key not in old_by_id]
    assert len(additions) == 6
    assert all(r["category"] == "competing_speech" for r in additions)
    assert all(r["language"] in ("Korean", "ko", "kor") for r in additions)
    assert len({r["eval_type"] for r in additions}) >= 2
    assert len({r["canonical_recording_id"] for r in additions}) == len(additions)
    assert all(r["split"] == "validation" and len(r["sha256"]) == 64 for r in additions)
    assert all(r["source_url"].startswith("https://") and r["original_source_url"].startswith("https://")
               and r["license"] and r["attribution"] and r["size_bytes"] > 0 for r in additions)
    assert current["size_bytes"] == sum(r["size_bytes"] for r in current["assets"])
    assert "F:\\" not in str(current) and "C:\\" not in str(current)


def test_v3_preview_lists_korean_assets_without_turning_them_into_training_data():
    catalog = evaluation.evaluation_catalog()
    summaries = evaluation.evaluation_preview()
    assert sum(row.get("file_count", 0) for row in summaries) == len(catalog["assets"])
    assert sum(row.get("total_bytes", 0) for row in summaries) == catalog["size_bytes"]
    for row in summaries:
        assert row["evaluation_only"] is True and row["split"] == "validation"
        if row.get("file_count"):
            assert row["individual_licenses"]
            assert all(item["source_url"].startswith("https://") for item in row["individual_licenses"])
    korean = {row["dataset_id"] for row in catalog["assets"] if row["category"] == "competing_speech"}
    assert korean
    assert all("한국어" in row["name"] for row in summaries if row["dataset_id"] in korean)
    assert all("Final" not in row["name"] for row in summaries)
    reserve = next(row for row in summaries if row["id"] == "zeroth_validation_reserve")
    assert "file_count" not in reserve and "total_bytes" not in reserve


def _fixture(tmp_path, monkeypatch, version=3, korean_seconds=1):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    paths, assets = {}, []
    specifications = [("rain", "environment", "unknown"),
                      ("music_singing", "music", "unknown"),
                      ("korean_crowd", "competing_speech", "ko")]
    for index, (subtype, category, language) in enumerate(specifications):
        path = tmp_path / (subtype + ".wav")
        seconds = korean_seconds if category == "competing_speech" else 1
        sf.write(path, np.random.default_rng(index + 800).normal(0, .03, 16000 * seconds), 16000,
                 subtype="PCM_16")
        paths[subtype] = path
        assets.append({"id": subtype, "filename": path.name, "sha256": digest(path),
            "size_bytes": path.stat().st_size, "source_url": "https://example.org/" + subtype,
            "original_source_url": "https://example.org/source/" + subtype,
            "license": "fixture only", "attribution": "test fixture", "category": category,
            "language": language, "eval_type": subtype, "dataset_id": "fixture_evaluation",
            "canonical_source_id": "fixture:" + subtype, "canonical_recording_id": "fixture:" + subtype,
            "duration_s": seconds, "media_variant": "fixture", "split": "validation"})
    catalog = {"recipe_version": version, "assets": assets,
               "size_bytes": sum(r["size_bytes"] for r in assets), "limitations": ["fixture only"]}
    monkeypatch.setattr(evaluation, "evaluation_catalog", lambda: deepcopy(catalog))
    monkeypatch.setattr(studio_data, "_download", lambda asset, directory, progress=None: paths[asset["id"]])
    prepared = evaluation.prepare_evaluation_materials(project, tmp_path / "workspace", {})
    queue = {"id": "fixture_evaluation_v3", "kind": "content", "evaluation_only": True,
             "pools": [], "preparation": prepared["preparation"]["path"],
             "queue": prepared["artifacts"]["representative_queue"]["path"],
             "pool_manifest": prepared["artifacts"]["source_manifest"]["path"]}
    return project, prepared, queue


def _answers(project, prepared, queue):
    previews = read_rows(queue["queue"])
    answer = {"wakeword": "보라야", "saved_at": "fixture_timestamp",
        "whole_pool_policy_approved": True, "evaluation_only": True,
        "intended_use": "held_out_development_noise", "pools": [],
        "consent_scope": "Held-out Development noise evaluation only; not for training",
        "source_scope": "Human dispositions entered through the local review screen",
        "source_manifest_sha256": prepared["artifacts"]["source_manifest"]["sha256"],
        "representative_queue_sha256": prepared["artifacts"]["representative_queue"]["sha256"],
        "queue_sha256": prepared["artifacts"]["representative_queue"]["sha256"],
        "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews},
        "decisions": {r["sample_id"]: {"decision": "keep", "listened": True} for r in previews}}
    path = project.parent / "review/evaluation_v3_answers.json"
    write_json(path, answer)
    return path, answer


def test_korean_content_is_unheard_after_preparation_and_validation_only_after_real_answers(tmp_path, monkeypatch):
    project, prep, queue = _fixture(tmp_path, monkeypatch)
    original_prep = Path(prep["preparation"]["path"]).read_bytes()
    rows = read_rows(queue["pool_manifest"])
    korean = next(r for r in rows if r["category"] == "competing_speech")
    assert korean["language"] == "ko"
    assert korean["evaluation_only"] is True and korean["training_admitted"] is False
    assert korean["individual_human_listening"] is False and korean["evaluation_content_approved"] is False
    assert prep["human_listened_count"] == 0
    answers, saved = _answers(project, prep, queue)
    saved["decisions"]["preview_korean_crowd"]["listened"] = False
    write_json(answers, saved)
    with pytest.raises(ValueError, match="Unheard"):
        evaluation.admit_evaluation_review(project, queue, answers, {})
    saved["decisions"]["preview_korean_crowd"]["listened"] = True
    write_json(answers, saved)
    accepted = evaluation.admit_evaluation_review(project, queue, answers, {})
    assert len(accepted["rows"]) == 3
    assert all(r["evaluation_only"] and r["split"] == "validation" and not r["training_admitted"]
               and r["individual_human_listening"] for r in accepted["rows"])
    assert Path(prep["preparation"]["path"]).read_bytes() == original_prep
    root, config = load_project(project)
    assert all(not read_rows(root / config["manifests"][pool])
               for pool in ("generic_negative", "korean_negative", "noise"))
    assert not (root / "final_lock.json").exists()


def test_korean_source_alias_cannot_cross_train_evaluation_boundary(tmp_path, monkeypatch):
    project, prep, queue = _fixture(tmp_path, monkeypatch)
    rows = read_rows(queue["pool_manifest"])
    korean = next(r for r in rows if r["category"] == "competing_speech")
    root, config = load_project(project)
    train_path = root / config["manifests"]["korean_negative"]
    # A distinct file/hash must still be rejected when the recording is the same.
    studio_data._write_rows(train_path, [{"sample_id": "train_alias", "path": "not_opened.wav",
        "sha256": "0" * 64, "split": "train", "duration_s": 1,
        "canonical_source_id": "different_publisher:alias",
        "canonical_recording_id": korean["canonical_recording_id"]}])
    with pytest.raises(ValueError, match="overlaps"):
        evaluation.assert_held_out(project, rows)


def test_balanced_selection_keeps_all_korean_subtypes_represented_without_row_count_bias():
    rows = []
    for index, (category, subtype) in enumerate([
            ("environment", "rain"), ("music", "instrumental"),
            ("competing_speech", "korean_read_speech"),
            ("competing_speech", "korean_crowd"),
            ("competing_speech", "korean_announcement")]):
        rows.append({"sample_id": str(index), "path": str(index) + ".wav", "category": category,
            "eval_type": subtype, "canonical_source_id": "source:" + str(index),
            "sha256": str(index) * 64, "language": "ko" if category == "competing_speech" else "unknown"})
    # Repeated intervals in one read-speech recording must not dominate.
    rows.extend([{**rows[2], "sample_id": "extra_" + str(i), "start_s": i, "end_s": i + 1}
                 for i in range(60)])
    pools = paired._balanced_pool(rows)
    selections = [paired._balanced_choice(pools, "competing_speech", slot, 912)[1]["eval_type"]
                  for slot in range(99)]
    assert {kind: selections.count(kind) for kind in set(selections)} == {
        "korean_read_speech": 33, "korean_crowd": 33, "korean_announcement": 33}


def _upgrade_fixture(tmp_path, monkeypatch):
    project, old, old_queue = _fixture(tmp_path, monkeypatch, version=2)
    old_catalog = evaluation.evaluation_catalog()
    answers, saved = _answers(project, old, old_queue)
    saved["decisions"]["preview_rain"]["decision"] = "discard"
    immutable_answer = answers.with_name("old_v2_answers.json")
    write_json(immutable_answer, saved)
    admission = evaluation.admit_evaluation_review(project, old_queue, immutable_answer, {})
    old_bytes = {path: Path(path).read_bytes() for path in
                 (old["preparation"]["path"], old_queue["queue"], old_queue["pool_manifest"], str(immutable_answer))}
    new_source = tmp_path / "korean_station.wav"
    sf.write(new_source, np.random.default_rng(345).normal(0, .04, 16000), 16000, subtype="PCM_16")
    new_asset = {**old_catalog["assets"][-1], "id": "korean_station", "filename": new_source.name,
                 "eval_type": "korean_transit_announcement", "sha256": digest(new_source),
                 "canonical_source_id": "fixture:korean_station", "canonical_recording_id": "fixture:korean_station",
                 "source_url": "https://example.org/korean_station"}
    catalog = {**old_catalog, "recipe_version": 3,
               "assets": old_catalog["assets"] + [new_asset],
               "size_bytes": old_catalog["size_bytes"] + new_source.stat().st_size}
    monkeypatch.setattr(evaluation, "evaluation_catalog", lambda: deepcopy(catalog))
    monkeypatch.setattr(studio_data, "_download", lambda asset, directory, progress=None: new_source
                        if asset["id"] == "korean_station" else pytest.fail("Reviewed original must be reused"))
    prepared = {"review_queues": [{"id": "existing_training_queue", "pools": ["noise"]}, old_queue]}
    result = studio_data.ensure_evaluation_materials(project, tmp_path / "workspace", prepared)
    queue = next(q for q in result["review_queues"] if q.get("evaluation_only"))
    prep = read_json(result["evaluation_preparation"]["path"])
    prep["preparation"] = result["evaluation_preparation"]
    return project, prep, queue, result, old_bytes, admission


def test_upgrade_reuses_actual_v2_decisions_only_and_preserves_discard(tmp_path, monkeypatch):
    project, prep, queue, result, old_bytes, old_admission = _upgrade_fixture(tmp_path, monkeypatch)
    assert prep["inherited_source_count"] == 3 and prep["pending_review_source_count"] == 1
    assert prep["inherited_admissions"] == [old_admission["admission"]]
    assert [r["sample_id"] for r in read_rows(queue["queue"])] == ["preview_korean_station"]
    assert len(result["review_queues"]) == 2
    assert result["review_queues"][0]["id"] == "existing_training_queue"
    assert result["review_queues"][1]["id"].startswith("materials_evaluation_v3_")
    for path, content in old_bytes.items():
        assert Path(path).read_bytes() == content
    answers, _ = _answers(project, prep, queue)
    approved = evaluation.admit_evaluation_review(project, queue, answers, result)
    assert {r["eval_type"] for r in approved["rows"]} == {
        "music_singing", "korean_crowd", "korean_transit_announcement"}
    assert "rain" not in {r["eval_type"] for r in approved["rows"]}
    assert all(not r["training_admitted"] for r in approved["rows"])


def test_new_v3_source_is_not_auto_approved_by_old_listening(tmp_path, monkeypatch):
    project, prep, queue, result, *_ = _upgrade_fixture(tmp_path, monkeypatch)
    answers, saved = _answers(project, prep, queue)
    saved["decisions"]["preview_korean_station"]["listened"] = False
    write_json(answers, saved)
    with pytest.raises(ValueError, match="Unheard"):
        evaluation.admit_evaluation_review(project, queue, answers, result)


def test_upgrade_rejects_changed_prior_listening_evidence(tmp_path, monkeypatch):
    project, prep, queue, result, *_ = _upgrade_fixture(tmp_path, monkeypatch)
    inherited = read_json(prep["inherited_admissions"][0]["path"])
    original_answer = Path(inherited["human_answers"]["path"])
    saved = read_json(original_answer)
    saved["decisions"]["preview_rain"]["decision"] = "keep"
    write_json(original_answer, saved)
    answers, _ = _answers(project, prep, queue)
    with pytest.raises(ValueError, match="Missing or changed"):
        evaluation.admit_evaluation_review(project, queue, answers, result)


def test_unreviewed_v2_candidates_do_not_reduce_v3_review_queue(tmp_path, monkeypatch):
    project, old, queue = _fixture(tmp_path, monkeypatch, version=2)
    catalog = evaluation.evaluation_catalog()
    catalog["recipe_version"] = 3
    monkeypatch.setattr(evaluation, "evaluation_catalog", lambda: deepcopy(catalog))
    result = studio_data.ensure_evaluation_materials(project, tmp_path / "workspace", {"review_queues": [queue]})
    new = read_json(result["evaluation_preparation"]["path"])
    assert new["inherited_source_count"] == 0 and new["pending_review_source_count"] == 3
    assert len(result["review_queues"]) == 1


def test_long_korean_recording_preserves_full_audio_with_honest_short_listening_scope(tmp_path, monkeypatch):
    project, prep, queue = _fixture(tmp_path, monkeypatch, korean_seconds=40)
    source = next(row for row in read_rows(queue["pool_manifest"]) if row["category"] == "competing_speech")
    preview = next(row for row in read_rows(queue["queue"]) if row["category"] == "competing_speech")
    original, rate = sf.read(source["path"], dtype="int16")
    excerpt, _ = sf.read(preview["path"], dtype="int16")
    assert len(original) == 40 * rate and len(excerpt) == 15 * rate
    start = int(preview["review_excerpt"]["start_s"] * rate)
    assert np.array_equal(excerpt, original[start:start + len(excerpt)])
    assert preview["reviewed_source"] == {"path": source["path"], "sha256": source["sha256"]}
    assert "전체를 청취한 것으로 기록하지 않습니다" in preview["review_instruction"]
    answers, _ = _answers(project, prep, queue)
    accepted = evaluation.admit_evaluation_review(project, queue, answers, {})
    korean = next(row for row in accepted["rows"] if row["category"] == "competing_speech")
    assert korean["path"] == source["path"] and korean["duration_s"] == 40
    assert korean["individual_human_listening"] is False
    assert korean["representative_human_listening"] is True
    assert korean["evaluation_content_review_scope"]["scope"] == "representative_excerpt_not_full_source"


def test_resume_checks_short_preview_integrity_without_overwriting_receipt(tmp_path, monkeypatch):
    project, prep, queue = _fixture(tmp_path, monkeypatch, korean_seconds=40)
    receipt_bytes = Path(prep["preparation"]["path"]).read_bytes()
    preview = next(row for row in read_rows(queue["queue"]) if row["category"] == "competing_speech")
    Path(preview["path"]).write_bytes(b"changed preview")
    with pytest.raises(ValueError, match="Missing or changed"):
        evaluation.prepare_evaluation_materials(project, tmp_path / "workspace", {})
    assert Path(prep["preparation"]["path"]).read_bytes() == receipt_bytes
