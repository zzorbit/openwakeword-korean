"""Held-out noise uses genuine listening evidence and never admits Train rows."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword import studio_data
from korean_openwakeword import studio_evaluation_data as module


def fixture_preparation(tmp_path, monkeypatch):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    assets, paths = [], {}
    for number, kind in enumerate(("rain", "door_wood_knock", "music_singing")):
        path = tmp_path / (kind + ".wav")
        sf.write(path, np.random.default_rng(number).normal(0, .04, 16000), 16000, subtype="PCM_16")
        asset = {"id": kind, "filename": path.name, "sha256": digest(path), "size_bytes": path.stat().st_size,
            "source_url": "https://example.org/" + kind, "license": "fixture only", "category": "music" if number == 2 else "environment",
            "eval_type": kind, "dataset_id": "fixture", "canonical_source_id": "fixture:" + kind,
            "canonical_recording_id": "fixture:" + kind, "duration_s": 1, "media_variant": "fixture"}
        assets.append(asset)
        paths[kind] = path
    monkeypatch.setattr(module, "evaluation_catalog", lambda: {"recipe_version": 2, "assets": assets,
        "size_bytes": sum(a["size_bytes"] for a in assets), "limitations": ["fixture"]})
    monkeypatch.setattr(studio_data, "_download", lambda asset, directory, progress=None: paths[asset["id"]])
    result = module.prepare_evaluation_materials(project, tmp_path / "workspace", {})
    queue = {"id": "eval_fixture", "kind": "content", "evaluation_only": True, "pools": [],
        "preparation": result["preparation"]["path"], "queue": result["artifacts"]["representative_queue"]["path"],
        "pool_manifest": result["artifacts"]["source_manifest"]["path"]}
    return project, result, queue


def saved_answers(project, prep, queue):
    previews = read_rows(queue["queue"])
    saved = {"wakeword": "보라야", "saved_at": "fixture_timestamp", "whole_pool_policy_approved": True,
        "evaluation_only": True, "intended_use": "held_out_development_noise", "pools": [],
        "consent_scope": "Held-out Development noise evaluation only; not for training",
        "source_scope": "Human dispositions entered through the local review screen",
        "source_manifest_sha256": prep["artifacts"]["source_manifest"]["sha256"],
        "representative_queue_sha256": prep["artifacts"]["representative_queue"]["sha256"],
        "queue_sha256": prep["artifacts"]["representative_queue"]["sha256"],
        "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews},
        "decisions": {r["sample_id"]: {"decision": "keep", "listened": True} for r in previews}}
    path = project.parent / "review/evaluation_answers.json"
    write_json(path, saved)
    return path, saved


def test_real_catalog_is_small_diverse_and_licensed_without_local_paths():
    catalog = module.evaluation_catalog()
    assets = catalog["assets"]
    assert len(assets) == 40 and len({a["canonical_source_id"] for a in assets}) == 40
    assert len({a["eval_type"] for a in assets if a["category"] == "environment"}) == 10
    assert len({a["eval_type"] for a in assets if a["category"] == "music"}) == 9
    assert catalog["size_bytes"] == sum(a["size_bytes"] for a in assets) < 35_000_000
    assert all(len(a["sha256"]) == 64 and a["license"] and a["attribution"] for a in assets)
    assert all(a["split"] == "validation" for a in assets)
    assert {"freesound:100355", "freesound:102956", "freesound:13613"}.isdisjoint(a["canonical_source_id"] for a in assets)
    assert len([a for a in assets if a["media_variant"] == "public_HQ_MP3_preview_not_original_WAV"]) == 4
    assert sum(r.get("file_count", 0) for r in module.evaluation_preview()) == 40
    assert "F:\\" not in str(catalog) and "C:\\" not in str(catalog)


def test_prepare_resume_and_evaluation_admission_are_immutable_and_train_free(tmp_path, monkeypatch):
    project, prep, queue = fixture_preparation(tmp_path, monkeypatch)
    original = Path(prep["preparation"]["path"]).read_bytes()
    assert prep["human_listened_count"] == 0 and not prep["training_admitted"]
    resumed = module.prepare_evaluation_materials(project, tmp_path / "workspace", {})
    assert resumed["resumed"] and Path(prep["preparation"]["path"]).read_bytes() == original
    answers, _ = saved_answers(project, prep, queue)
    result = module.admit_evaluation_review(project, queue, str(answers), {})
    receipt = read_json(result["admission"]["path"])
    assert receipt["state"] == "accepted_for_evaluation" and receipt["training_admitted"] is False
    assert all(r["split"] == "validation" and "admission" not in r and r["evaluation_admission"] for r in result["rows"])
    root, config = load_project(project)
    assert not read_rows(root / config["manifests"]["noise"])
    assert module.admit_evaluation_review(project, queue, str(answers), {})["admission"] == result["admission"]


@pytest.mark.parametrize("field,value", [("evaluation_only", False), ("intended_use", "training"),
    ("pools", ["noise"]), ("consent_scope", "Train pool consent")])
def test_train_consent_cannot_approve_evaluation_pool(tmp_path, monkeypatch, field, value):
    project, prep, queue = fixture_preparation(tmp_path, monkeypatch)
    answers, saved = saved_answers(project, prep, queue)
    saved[field] = value
    write_json(answers, saved)
    with pytest.raises(ValueError, match="Real saved"):
        module.admit_evaluation_review(project, queue, str(answers), {})


def test_unheard_keep_rejected_and_user_discards_reduce_actual_coverage(tmp_path, monkeypatch):
    project, prep, queue = fixture_preparation(tmp_path, monkeypatch)
    answers, saved = saved_answers(project, prep, queue)
    saved["decisions"]["preview_rain"]["listened"] = False
    write_json(answers, saved)
    with pytest.raises(ValueError, match="Unheard"):
        module.admit_evaluation_review(project, queue, str(answers), {})
    saved["decisions"]["preview_rain"]["decision"] = "discard"
    write_json(answers, saved)
    receipt = read_json(module.admit_evaluation_review(project, queue, str(answers), {})["admission"]["path"])
    assert receipt["kept_count"] == 2 and receipt["discarded_count"] == 1
    assert receipt["missing_eval_types"] == ["rain"]


def test_id_file_hash_and_normalized_pcm_collisions_reject_known_train(tmp_path, monkeypatch):
    project, prep, queue = fixture_preparation(tmp_path, monkeypatch)
    rows = read_rows(queue["pool_manifest"])
    root, config = load_project(project)
    train_path = root / config["manifests"]["noise"]
    row = dict(rows[0], split="train")
    studio_data._write_rows(train_path, [row])
    with pytest.raises(ValueError, match="overlaps"):
        module.assert_held_out(project, rows)
    # An unrelated container and source ID with exactly identical PCM is also blocked.
    wav = root / "data/negative/duplicate.wav"
    samples, rate = sf.read(row["path"], dtype="int16")
    sf.write(wav, samples, rate, subtype="PCM_16")
    # Deliberately different RIFF metadata, same decoded samples.
    with wav.open("ab") as stream:
        stream.write(b"different metadata")
    train_path.write_text("", encoding="utf-8")
    studio_data._write_rows(train_path, [{"path": str(wav), "sha256": digest(wav), "duration_s": 1,
        "canonical_source_id": "unrelated:source", "split": "train"}])
    with pytest.raises(ValueError, match="PCM duplicates"):
        module.assert_held_out(project, rows)


def test_pcm_audit_reuses_hash_cache_and_skips_long_recordings(tmp_path, monkeypatch):
    project, prep, queue = fixture_preparation(tmp_path, monkeypatch)
    rows = read_rows(queue["pool_manifest"])
    root, config = load_project(project)
    path = root / "data/negative/train.wav"
    sf.write(path, np.random.default_rng(900).normal(0, .02, 16000), 16000, subtype="PCM_16")
    studio_data._write_rows(root / config["manifests"]["noise"], [
        {"path": str(path), "sha256": digest(path), "duration_s": 1, "split": "train", "canonical_source_id": "train:one"},
        {"path": "deliberately_not_opened.wav", "sha256": "9" * 64, "duration_s": 3600, "split": "train", "canonical_source_id": "train:long"}])
    first = module.assert_held_out(project, rows)
    assert first["same_length_sources_pcm_read"] == 1 and first["pcm_bytes_read"] == 32000
    assert module.assert_held_out(project, rows)["pcm_cache_reused"] == 1


def test_headroom_does_not_hide_original_clipping(tmp_path, monkeypatch):
    source, target = tmp_path / "source.wav", tmp_path / "target.wav"
    sf.write(source, np.ones(16000), 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="clipped original"):
        module._convert_evaluation_audio(source, target)
    sf.write(source, np.zeros(16000), 16000, subtype="PCM_16")
    monkeypatch.setattr("korean_openwakeword.audio.read_audio", lambda path: np.asarray([0, 1.1, -.5]))
    result = module._convert_evaluation_audio(source, target)
    assert result["uniform_gain"] == pytest.approx(.95 / 1.1)
    assert result["headroom_reason"] == "resampling_overshoot_only"
    assert np.max(np.abs(sf.read(target)[0])) == pytest.approx(.95, abs=1 / 32768)


def test_v2_ensure_keeps_legacy_preparation_untouched(tmp_path, monkeypatch):
    project, prep, queue = fixture_preparation(tmp_path, monkeypatch)
    legacy = project.parent / ".studio/materials/preparation_result.json"
    write_json(legacy, {"legacy": True})
    old = legacy.read_bytes()
    result = studio_data.ensure_evaluation_materials(project, tmp_path / "workspace", {"review_queues": []})
    assert result["recipe_version"] == 2 and len(result["review_queues"]) == 1
    assert result["review_queues"][0]["evaluation_only"] and result["review_queues"][0]["pools"] == []
    assert legacy.read_bytes() == old
