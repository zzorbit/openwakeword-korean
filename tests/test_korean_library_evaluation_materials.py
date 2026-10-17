"""Real source recipes, duration accounting, and bounded archive preparation."""
import hashlib
import io
import struct
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import evaluation_materials as module
from korean_openwakeword.common import digest, read_json


def test_catalog_has_real_pins_and_all_seven_buckets():
    catalog = module.evaluation_material_catalog()
    assert catalog["target_hours_by_bucket"] == module.BUCKET_HOURS
    assert sum(module.BUCKET_HOURS.values()) == 30
    assert catalog["known_download_bytes"] == sum(a["size_bytes"] for a in catalog["archives"])
    assert len(module.evaluation_material_preview()) == 7
    for asset in catalog["archives"]:
        assert asset["source_url"].startswith("https://") and asset["size_bytes"] > 0
        assert len(asset.get("sha256", asset.get("md5", ""))) in (32, 64)
    seoul = catalog["selections"]["seoul"]
    assert len(seoul) == 40 and len({r["canonical_recording_id"] for r in seoul}) == 40
    assert sum(r["selection_duration_s"] for r in seoul if r["eval_bucket"] == "korean_conversation") == 10*3600
    assert sum(r["selection_duration_s"] for r in seoul if r["eval_bucket"] == "korean_interview") == 5*3600
    assert all(r["members"] and r["language"] == "ko" and "interview" in r["source_classification"] for r in seoul)
    music = [r for r in catalog["selections"]["musan"] if r["eval_bucket"] == "music"]
    assert len(music) >= 100 and len({r["canonical_artist_id"] for r in music}) >= 30
    assert all("ND" not in r["license_notice"] and "NoDeriv" not in r["license_notice"] for r in music)
    assert not set(catalog["excluded_supplemental_canonical_ids"]) & {r["canonical_recording_id"] for r in catalog["selections"]["fsd_eval"]}
    assert all("zero/1.0" in r["license"] or "/by/" in r["license"] for r in catalog["selections"]["fsd_eval"])
    assert len({r["eval_type"] for r in catalog["selections"]["fsd_eval"] if r["eval_bucket"] == "environment"}) >= 30
    assert len({r["eval_type"] for r in catalog["selections"]["fsd_eval"] if r["eval_bucket"] == "electronic"}) >= 10


def test_coverage_unions_intervals_and_does_not_count_diagnostic_backgrounds():
    rows = [{"eval_bucket": "music", "canonical_recording_id": "a", "source_start_s": 0, "source_end_s": 3600},
            {"eval_bucket": "music", "canonical_recording_id": "a", "source_start_s": 1800, "source_end_s": 5400},
            {"canonical_recording_id": "supplement", "duration_s": 100000}]
    result = module.evaluation_material_coverage(rows)
    assert result["actual_hours_by_bucket"]["music"] == 1.5
    assert result["actual_total_hours"] == 1.5 and not result["target_met"]
    assert result["target_hours_by_bucket"]["music"] == 3 and len(result["shortages"]) == 7
    with pytest.raises(ValueError, match="multiple"):
        module.evaluation_material_coverage([rows[0], {**rows[1], "eval_bucket": "environment"}])
    with pytest.raises(ValueError, match="finite"):
        module.evaluation_material_coverage([{**rows[0], "source_end_s": float("nan")}])
    duplicates = [{**rows[0], "original_sha256": "a"*64},
                  {**rows[0], "canonical_recording_id": "renamed", "original_sha256": "a"*64}]
    assert module.evaluation_material_coverage(duplicates)["actual_hours_by_bucket"]["music"] == 1


def test_stream_coverage_unrolls_sources_without_repeating_hours_or_seoul_parts():
    first = {"eval_bucket": "music", "canonical_recording_id": "original:a",
             "original_sha256": "a"*64, "source_start_s": 0, "source_end_s": 3600}
    second = {"eval_bucket": "music", "canonical_recording_id": "original:b",
              "original_sha256": "b"*64, "source_start_s": 0, "source_end_s": 1800}
    shard1 = {"eval_bucket": "music", "canonical_recording_id": "shard:1", "duration_s": 5400,
              "evaluation_stream_components": [first, second]}
    shard2 = {"eval_bucket": "music", "canonical_recording_id": "shard:2", "duration_s": 3600,
              "evaluation_stream_components": [dict(first)]}
    result = module.evaluation_material_coverage([shard1, {"eval_bucket": "music",
        "evaluation_stream_components": [shard2]}])
    assert result["actual_hours_by_bucket"]["music"] == 1.5
    assert result["unique_recordings_by_bucket"]["music"] == 2
    with pytest.raises(ValueError, match="multiple"):
        module.evaluation_material_coverage([shard1, {**shard2, "eval_bucket": "environment"}])
    with pytest.raises(ValueError, match="multiple"):
        module.evaluation_material_coverage([shard1, {"eval_bucket": "environment",
            "evaluation_stream_components": [{**first, "eval_bucket": "environment"}]}])
    seoul = {"eval_bucket": "korean_interview", "canonical_recording_id": "seoul:s01", "duration_s": 1800,
             "components": [{"duration_s": 100000, "canonical_recording_id": "part:1"}]}
    assert module.evaluation_material_coverage([seoul])["actual_hours_by_bucket"]["korean_interview"] == .5
    cyclic = {"eval_bucket": "music"}
    cyclic["evaluation_stream_components"] = [cyclic]
    with pytest.raises(ValueError, match="noncyclic"):
        module.evaluation_material_coverage([cyclic])


def _split_fixture(tmp_path):
    """Build the official split-ZIP64 layout without a 6 GB test artifact."""
    buffer = io.BytesIO()
    expected = {"clips/a.bin": b"a"*1234, "clips/b.bin": b"b"*2345}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in expected.items(): z.writestr(name, data)
    raw = bytearray(buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        split = z.getinfo("clips/b.bin").header_offset
        cd = z.start_dir
    old_end = raw.rfind(b"PK\x05\x06")
    pos = cd
    while pos < old_end:
        values = list(struct.unpack_from(zipfile.structCentralDir, raw, pos))
        if values[zipfile._CD_LOCAL_HEADER_OFFSET] >= split:
            values[zipfile._CD_DISK_NUMBER_START] = 1
            values[zipfile._CD_LOCAL_HEADER_OFFSET] -= split
        struct.pack_into(zipfile.structCentralDir, raw, pos, *values)
        pos += 46 + values[zipfile._CD_FILENAME_LENGTH] + values[zipfile._CD_EXTRA_FIELD_LENGTH] + values[zipfile._CD_COMMENT_LENGTH]
    end64 = struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 798, 45, 1, 1, 2, 2, old_end-cd, cd-split)
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 1, old_end-split, 2)
    footer = struct.pack("<4s4H2LH", b"PK\x05\x06", 1, 1, 2, 2, old_end-cd, cd-split, 0)
    first, second = tmp_path / "audio.z01", tmp_path / "audio.zip"
    first.write_bytes(raw[:split]); second.write_bytes(raw[split:old_end]+end64+locator+footer)
    return [first, second], expected


def test_original_split_zip64_without_original_modification(tmp_path):
    paths, expected = _split_fixture(tmp_path)
    before = [digest(p) for p in paths]
    archive, view = module._split_zip(paths)
    try:
        assert {name: archive.read(name) for name in expected} == expected
    finally:
        archive.close(); view.close()
    assert [digest(p) for p in paths] == before


def test_selected_stream_tar_checks_pin_and_refuses_escape(tmp_path):
    archive = tmp_path / "source.tar.gz"
    body = b"fixture public source"
    with tarfile.open(archive, "w:gz") as tar:
        entry = tarfile.TarInfo("corpus/selected.wav"); entry.size = len(body)
        tar.addfile(entry, io.BytesIO(body))
    selections = [{"member": "corpus/selected.wav", "sha256": hashlib.sha256(body).hexdigest()}]
    result = module._extract_selected({"id": "a", "format": "tar_gz"}, {"a": archive}, selections, tmp_path / "raw")
    assert result[selections[0]["member"]].read_bytes() == body
    assert module._extract_selected({"id": "a", "format": "tar_gz"}, {"a": archive}, selections, tmp_path / "raw") == result
    with pytest.raises(ValueError, match="inside"):
        module._extract_selected({"id": "a", "format": "tar_gz"}, {"a": archive}, [{"member": "../escape.wav"}], tmp_path / "raw")


def test_one_seoul_parent_stream_keeps_original_parts_and_exact_duration(tmp_path):
    paths = {}
    for i in range(2):
        p = tmp_path / f"part{i}.flac"
        sf.write(p, np.sin(np.arange(22050)*.01+i)*.05, 22050, subtype="PCM_16")
        paths[p.name] = p
    row = {"canonical_speaker_id": "seoul:speaker:s01", "canonical_recording_id": "seoul:s01",
        "members": [{"member": p} for p in paths], "selection_duration_s": 1.5,
        "source_url": "https://www.openslr.org/113/", "license": "CC-BY-NC-2.0"}
    joined, lineage = module._join_seoul(row, paths, tmp_path / "joined")
    assert sf.info(joined).duration == 1.5 and len(lineage) == 2
    assert sum(x["source_end_s"] for x in lineage) == 1.5
    assert all(x["original_sha256"] == digest(x["original_path"]) for x in lineage)
    same, same_lineage = module._join_seoul(row, paths, tmp_path / "joined")
    assert same == joined and same_lineage == lineage
    normalized = module._normalize(joined, tmp_path / "norm.wav", 0, 1.5)
    assert normalized["duration_s"] == 1.5 and normalized["conversion"]["noise_added"] is False
    assert sf.info(tmp_path / "norm.wav").samplerate == 16000


def test_source_identity_selection_excludes_known_speaker_artist_recording(tmp_path, monkeypatch):
    catalog = {"archives": [{"id": "a", "format": "tar_gz"}], "selections": {"a": [
        {"member": "a.wav", "canonical_recording_id": "known"},
        {"member": "b.wav", "canonical_recording_id": "other", "canonical_artist_id": "artist"},
        {"member": "c.wav", "canonical_recording_id": "new", "canonical_speaker_id": "speaker"}]}}
    captured = []
    monkeypatch.setattr(module, "_extract_selected", lambda a, p, rows, o, progress=None: captured.extend(rows) or {})
    monkeypatch.setattr(module, "_zeroth_candidates", lambda *a: [])
    result = module._candidate_rows(catalog, {}, tmp_path, "fixture", tmp_path,
        [{"canonical_recording_id": "known"}, {"canonical_artist_id": "artist"}, {"canonical_speaker_id": "speaker"}], None)
    assert result == [] and captured == []


def test_actual_small_preparation_reports_shortage_and_resumes_without_admitting(tmp_path, monkeypatch):
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    rows = []
    for index, bucket in enumerate(module.BUCKET_HOURS):
        source = tmp_path / (bucket + ".wav")
        sf.write(source, np.random.default_rng(index).normal(0, .03, 16000), 16000, subtype="PCM_16")
        rows.append({"original_path": str(source), "canonical_source_id": "fixture:"+bucket,
            "canonical_recording_id": "fixture:"+bucket, "eval_bucket": bucket,
            "eval_type": bucket, "category": "environment", "selection_duration_s": 1,
            "license": "fixture", "source_url": "https://example.org/fixture"})
    monkeypatch.setattr(module, "evaluation_material_catalog", lambda: {"archives": [], "known_download_bytes": 0,
        "working_bytes": 0, "target_hours_by_bucket": module.BUCKET_HOURS, "limitations": ["fixture"]})
    monkeypatch.setattr(module, "_known_train", lambda *a: [])
    monkeypatch.setattr(module, "_candidate_rows", lambda *a: rows)
    first = module.prepare_common_sources(project, tmp_path / "workspace")
    assert len(first["source_rows"]) == 7 and not first["target_met"] and len(first["shortages"]) == 7
    assert first["coverage"]["actual_total_hours"] == pytest.approx(7/3600)
    assert first["human_listened_count"] == 0 and first["training_admitted"] is False
    assert all(r["evaluation_content_approved"] is False and r["split"] == "validation" for r in first["source_rows"])
    again = module.prepare_common_sources(project, tmp_path / "workspace")
    assert again["resumed"] is True and again["source_rows"] == first["source_rows"]
    Path(again["source_rows"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError): module.prepare_common_sources(project, tmp_path / "workspace")
