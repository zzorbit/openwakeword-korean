"""Real source recipes, duration accounting, and bounded archive preparation."""
import hashlib
import io
import struct
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

from korean_openwakeword import evaluation_materials as module
from korean_openwakeword.common import digest, read_json, write_json


def _legacy_project(project):
    config = read_json(project)
    config.pop("common_download_policy", None)
    write_json(project, config)


@pytest.mark.parametrize("kind", ["recordings", "rirs"])
@pytest.mark.parametrize("policy_location", ["config", "argument"])
def test_retired_download_policy_stops_before_catalog_download_or_receipt_write(
        tmp_path, monkeypatch, kind, policy_location):
    from korean_openwakeword import evaluation_rir
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
    if policy_location == "config":
        config = read_json(project)
        config["common_download_policy"] = "selected_common_zip_v1"
        write_json(project, config)
    before_config = project.read_bytes()
    output = tmp_path / "existing_materials"
    output.mkdir()
    receipt = output / "receipt.json"
    receipt.write_bytes(b'{"historical":"preserve exactly"}\n')
    before_receipt = receipt.read_bytes()
    fail = lambda *a, **k: pytest.fail("Retired policy must stop before acquisition")
    monkeypatch.setattr(module, "evaluation_material_catalog", fail)
    monkeypatch.setattr(module, "_download_archive", fail)
    monkeypatch.setattr(evaluation_rir, "common_rir_catalog", fail)
    monkeypatch.setattr(evaluation_rir, "download_checked", fail)
    options = {"acquisition_policy": "selected_common_zip_v1"} if policy_location == "argument" else {}
    with pytest.raises(ValueError, match="0.16"):
        if kind == "recordings":
            module.prepare_common_sources(project, tmp_path / "new_workspace", **options)
        else:
            evaluation_rir.prepare_common_rirs(project, output, **options)
    assert project.read_bytes() == before_config
    assert receipt.read_bytes() == before_receipt
    assert not (tmp_path / "new_workspace").exists()


def test_retired_recording_receipt_is_preserved_even_without_project_policy(tmp_path, monkeypatch):
    from korean_openwakeword.common import identity
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
    catalog = {"catalog_fixture": "unchanged"}
    monkeypatch.setattr(module, "evaluation_material_catalog", lambda: catalog)
    monkeypatch.setattr(module, "_known_train", lambda *a: [])
    monkeypatch.setattr(module, "_download_archive", lambda *a, **k: pytest.fail("No download"))
    request = {"catalog_id": identity(catalog)[:16], "known_train": identity([])}
    output = tmp_path / "shared" / ("common_evaluation_30h_" + request["catalog_id"]) / ("selection_" + identity(request)[:16])
    output.mkdir(parents=True)
    receipt = output / "preparation.json"
    write_json(receipt, {"request": request, "acquisition_policy": "selected_common_zip_v1"})
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="0.16"):
        module.prepare_common_sources(project, tmp_path)
    assert receipt.read_bytes() == before


def test_retired_rir_receipt_is_preserved_even_without_project_policy(tmp_path, monkeypatch):
    from korean_openwakeword import evaluation_rir
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
    monkeypatch.setattr(evaluation_rir, "_assert_source_separation", lambda *a: None)
    monkeypatch.setattr(evaluation_rir, "download_checked", lambda *a, **k: pytest.fail("No download"))
    output = tmp_path / "rir"
    output.mkdir()
    receipt = output / "receipt.json"
    write_json(receipt, {"acquisition_policy": "selected_common_zip_v1"})
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="0.16"):
        evaluation_rir.prepare_common_rirs(project, output)
    assert receipt.read_bytes() == before


def test_retired_evaluation_admission_does_not_relabel_or_create_evidence(tmp_path):
    from korean_openwakeword.material_policy import fixed_queue_evidence
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
    queue = {"id": "evaluation", "evaluation_only": True}
    prepared = {"kind": "studio_material_preparation_v1", "project": str(project),
                "review_queues": [queue], "common_download_policy": "selected_common_zip_v1"}
    with pytest.raises(ValueError, match="0.16"):
        fixed_queue_evidence(project, prepared, queue)
    assert not (project.parent / "review/library_fixed_sources").exists()


def test_publisher_md5_download_recovers_connection_and_preserves_receipt(tmp_path, monkeypatch):
    payload, calls, progress = b"exact evaluation archive fixture", [], []
    asset = {"id": "fsd_fixture", "filename": "fixture.zip", "size_bytes": len(payload),
             "checksum_algorithm": "md5", "md5": hashlib.md5(payload).hexdigest(),
             "source_url": "https://example.org/original.zip"}
    class Response:
        def __init__(self, offset):
            self.offset = offset
            self.status_code = 206 if offset else 200
            self.headers = {"Content-Range": f"bytes {offset}-{len(payload)-1}/{len(payload)}"} if offset else {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, count):
            if self.offset == 0:
                yield payload[:4]
                raise requests.exceptions.ChunkedEncodingError("fixture disconnected")
            yield payload[self.offset:]
    def get(url, headers, **kwargs):
        calls.append(headers)
        return Response(4 if headers.get("Range") else 0)
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", get)
    monkeypatch.setattr("korean_openwakeword.downloads.time.sleep", lambda _: None)
    target = module._download_archive(asset, tmp_path, progress.append)
    assert target.read_bytes() == payload and calls[-1]["Range"] == "bytes=4-"
    assert {r["phase"] for r in progress} == {"download", "retry", "hash"}
    receipt = target.with_name(target.name+".receipt.json")
    before = receipt.read_bytes()
    assert read_json(receipt)["sha256"] == hashlib.sha256(payload).hexdigest()
    assert module._download_archive(asset, tmp_path) == target
    assert len(calls) == 2 and receipt.read_bytes() == before


def test_publisher_md5_bad_checksum_preserves_original_partial(tmp_path, monkeypatch):
    partial = tmp_path / "fixture.zip.part"; partial.write_bytes(b"changed")
    asset = {"id": "fixture", "filename": "fixture.zip", "size_bytes": 7,
             "checksum_algorithm": "md5", "md5": "0"*32, "source_url": "https://example.org/original.zip"}
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: pytest.fail("Complete partial should only be checked"))
    with pytest.raises(ValueError, match="checksum"):
        module._download_archive(asset, tmp_path)
    assert partial.read_bytes() == b"changed" and not (tmp_path / "fixture.zip").exists()


def test_common_zeroth_bootstraps_without_training_import_and_reserves_speakers(tmp_path, monkeypatch):
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    archive = tmp_path / "zeroth_fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for speaker in ("001", "002"):
            name = speaker + "_001"
            wav = io.BytesIO()
            sf.write(wav, np.sin(np.arange(16000) * .04) * .1, 16000, format="FLAC", subtype="PCM_16")
            for relative, data in ((f"train_data_01/{speaker}/{name}.flac", wav.getvalue()),
                                   (f"train_data_01/{speaker}/{speaker}.trans.txt", (name + " 보라야 시험\n").encode())):
                member = tarfile.TarInfo(relative); member.size = len(data)
                target.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(module, "_download_archive", lambda *a, **k: pytest.fail("Already verified archive is reused"))
    catalog = {"archives": [{"id": "zeroth", "license": "fixture"}]}
    known = [{"canonical_speaker_id": "zeroth:speaker:001"}]
    first = module._zeroth_candidates(project, tmp_path, known, catalog, {"zeroth": archive})
    assert len(first) == 1 and first[0]["canonical_speaker_id"] == "zeroth:speaker:002"
    assert first[0]["split"] == "validation" and first[0]["original_provider_split"] == "train"
    assert Path(first[0]["original_path"]).is_file()
    assert not list((project.parent / "data").rglob("*admission*"))
    from korean_openwakeword import studio_data
    monkeypatch.setattr(studio_data, "extract_archive", lambda *a, **k: pytest.fail("Existing index is reused"))
    assert module._zeroth_candidates(project, tmp_path, known, catalog) == first


@pytest.mark.parametrize("legacy_mode", ["all", "members", "suffixes"])
def test_common_zeroth_preserves_existing_extraction_request_and_receipt(tmp_path, monkeypatch, legacy_mode):
    from korean_openwakeword.project import create_project
    from korean_openwakeword import studio_data
    project = create_project(tmp_path / "word", "보라야", "bora")
    archive = tmp_path / "zeroth.zip"
    audio = io.BytesIO()
    sf.write(audio, np.sin(np.arange(16000) * .04) * .1, 16000, format="FLAC", subtype="PCM_16")
    selected = ["train_data_01/001/001_001.flac", "train_data_01/001/001.trans.txt", "README.txt"]
    with zipfile.ZipFile(archive, "w") as handle:
        for name, payload in zip(selected, [audio.getvalue(), "001_001 원본 전사\n".encode(), b"metadata"]):
            handle.writestr(name, payload)
        handle.writestr("nested/originals.tar.gz", b"legacy nested archive")
    directory = tmp_path / "shared" / "zeroth"
    options = {"members": selected, "max_bytes": 1024**2} if legacy_mode == "members" else {"include_suffixes": (".flac", ".txt")} if legacy_mode == "suffixes" else {}
    studio_data.extract_archive(archive, directory, **options)
    receipt = directory / "extraction_receipt.json"
    before = receipt.read_bytes()
    catalog = {"archives": [{"id": "zeroth", "license": "fixture"}]}
    result = module._zeroth_candidates(project, tmp_path, [], catalog, {"zeroth": archive})
    assert len(result) == 1 and result[0]["text"] == "원본 전사"
    assert result[0]["canonical_speaker_id"] == "zeroth:speaker:001"
    assert receipt.read_bytes() == before
    index = project.parent / ".studio" / "materials" / "zeroth_source_index.jsonl"
    before_index = index.read_bytes()
    monkeypatch.setattr(studio_data, "_extract_zeroth_archive", lambda *a, **k: pytest.fail("Existing source index is immutable"))
    monkeypatch.setattr(module, "_download_archive", lambda *a, **k: pytest.fail("Existing index needs no archive download"))
    assert module._zeroth_candidates(project, tmp_path, [], catalog) == result
    assert index.read_bytes() == before_index and receipt.read_bytes() == before


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


def test_common_sources_download_batch_preserves_catalog_order_before_selection(tmp_path, monkeypatch):
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    archives = [{"id": name, "filename": name + ".zip", "size_bytes": 1} for name in ("first", "second")]
    monkeypatch.setattr(module, "evaluation_material_catalog", lambda: {
        "archives": archives, "known_download_bytes": 2, "working_bytes": 0})
    monkeypatch.setattr(module, "_known_train", lambda *args: [])
    acquired, batches = [], []
    def acquire(asset, downloads, progress=None):
        acquired.append(asset["id"])
        assert callable(progress)
        return downloads / asset["filename"]
    monkeypatch.setattr(module, "_download_archive", acquire)
    def download_batch(items, worker, *, progress=None, max_workers=None):
        assert items == archives
        batches.append(list(items))
        results = {asset["id"]: worker(asset, lambda event: None) for asset in reversed(items)}
        return [results[asset["id"]] for asset in items]
    monkeypatch.setattr(module, "download_many", download_batch)
    class SelectionReached(Exception):
        pass
    def select(catalog, paths, *args):
        assert len(batches) == 1 and acquired == ["second", "first"]
        assert list(paths) == ["first", "second"]
        assert [path.name for path in paths.values()] == ["first.zip", "second.zip"]
        raise SelectionReached
    monkeypatch.setattr(module, "_candidate_rows", select)
    with pytest.raises(SelectionReached):
        module.prepare_common_sources(project, tmp_path / "workspace")


def test_actual_small_preparation_reports_shortage_and_resumes_without_admitting(tmp_path, monkeypatch):
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
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


def test_unified_pool_adds_only_genuine_same_catalog_context_and_keeps_legacy_separate(tmp_path, monkeypatch):
    from korean_openwakeword.project import create_project
    from korean_openwakeword.negative_streams import negative_scoring_budget
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
    # Small real WAVs exercise the exact two-second runtime exclusion without
    # downloading a corpus or claiming that these fixture files were heard.
    monkeypatch.setattr(module, "BUCKET_HOURS", {key: 3/3600 for key in module.BUCKET_HOURS})
    rows = []
    for index, bucket in enumerate(module.BUCKET_HOURS):
        source = tmp_path / (bucket + ".wav")
        sf.write(source, np.random.default_rng(index).normal(0, .03, 3 * 16000), 16000, subtype="PCM_16")
        rows.append({"original_path": str(source), "canonical_source_id": "fixture:"+bucket,
            "canonical_recording_id": "fixture:"+bucket, "eval_bucket": bucket,
            "eval_type": bucket, "category": "environment", "selection_duration_s": 3,
            "license": "fixture", "source_url": "https://example.org/fixture"})
    extra = tmp_path / "unused_same_corpus.wav"
    sf.write(extra, np.random.default_rng(100).normal(0, .03, 30 * 16000), 16000, subtype="PCM_16")
    rows.append({**rows[-1], "original_path": str(extra), "canonical_source_id": "fixture:unused",
                 "canonical_recording_id": "fixture:unused", "selection_duration_s": 30})
    monkeypatch.setattr(module, "evaluation_material_catalog", lambda: {"archives": [], "known_download_bytes": 0,
        "working_bytes": 0, "target_hours_by_bucket": module.BUCKET_HOURS, "limitations": ["fixture"]})
    monkeypatch.setattr(module, "_known_train", lambda *a: [])
    monkeypatch.setattr(module, "_candidate_rows", lambda *a: rows)
    legacy = module.prepare_common_sources(project, tmp_path / "workspace")
    strict = module.prepare_common_sources(project, tmp_path / "workspace", strict_common_pool=True)
    assert legacy["request"] != strict["request"] and len(legacy["source_rows"]) == 7
    assert len(strict["source_rows"]) == 8 and strict["target_met"]
    assert strict["source_pool_scope"] == "common_30h_only"
    assert strict["core_source_seconds"] == 21
    assert strict["initialization_margin_seconds"] > 0
    assert strict["duration_budget"]["eligible_seconds"] >= 21
    assert strict["duration_budget"]["eligible_seconds"] == negative_scoring_budget(strict["source_rows"])["eligible_seconds"]
    positive = [row for row in strict["source_rows"] if row["positive_mix_eligible"]]
    margin = [row for row in strict["source_rows"] if row["evaluation_initialization_margin"]]
    assert len(positive) == 7 and len(margin) == 1
    assert margin[0]["canonical_recording_id"] == "fixture:unused"
    assert all(row["evaluation_content_approved"] is False for row in strict["source_rows"])
    coverage = module.evaluation_material_coverage(strict["source_rows"], strict_common_pool=True)
    assert coverage["core_coverage"]["actual_total_hours"] == pytest.approx(21/3600)
    assert coverage["initialization_margin_hours"] == pytest.approx(strict["initialization_margin_seconds"]/3600)
    again = module.prepare_common_sources(project, tmp_path / "workspace", strict_common_pool=True)
    assert again["resumed"] and again["source_rows"] == strict["source_rows"]
    with pytest.raises(ValueError, match="outside its pinned"):
        module.evaluation_material_coverage(strict["source_rows"] + [{"duration_s": 100000}], strict_common_pool=True)


def test_unified_coverage_does_not_use_margin_to_fill_missing_core_quota():
    row = {"eval_bucket": "music", "canonical_recording_id": "unused:music", "source_start_s": 0,
           "source_end_s": 3 * 3600, "source_pool_scope": "common_30h_only",
           "evaluation_material_catalog_id": "fixture", "positive_mix_eligible": False,
           "evaluation_initialization_margin": True}
    coverage = module.evaluation_material_coverage([row], strict_common_pool=True)
    assert coverage["actual_hours_by_bucket"]["music"] == 3
    assert coverage["core_coverage"]["actual_hours_by_bucket"]["music"] == 0
    assert not coverage["target_met"]


def test_unified_assessment_rejects_legacy_audio_even_with_thirty_reported_hours(tmp_path):
    import json
    from korean_openwakeword.evaluation_coverage import assess_common_evaluation
    from korean_openwakeword.studio_steps import common_evaluation_policy
    path = tmp_path / "legacy_negative_fixture.jsonl"
    path.write_text(json.dumps({"sample_id": "old_diagnostic", "kind": "negative", "category": "environment",
        "canonical_recording_id": "old:193", "duration_s": 30 * 3600}) + "\n", encoding="utf-8")
    result = assess_common_evaluation(common_evaluation_policy(4),
        {"models": [{"negative": {"hours": 30}}]}, path, {"fixture": True})
    assert not result["negative_materials"]["target_met"]
    assert any("outside its pinned" in reason for reason in result["shortages"])


def test_unified_core_quotas_use_original_time_not_resampling_ceiling(tmp_path, monkeypatch):
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora")
    _legacy_project(project)
    target_s = 20.0024
    monkeypatch.setattr(module, "BUCKET_HOURS", {"environment": target_s/3600})
    rows = []
    for i in range(24):
        source = tmp_path / f"source_{i:02d}.wav"
        # Each conversion rounds up more than the source interval. Twenty
        # such normalized clips exceed the quota while original time does not.
        sf.write(source, np.random.default_rng(i).normal(0, .03, 44103), 44100, subtype="PCM_16")
        rows.append({"original_path": str(source), "canonical_source_id": f"fixture:{i}",
            "canonical_recording_id": f"fixture:{i}", "eval_bucket": "environment",
            "eval_type": "fixture_environment", "category": "environment", "selection_duration_s": 44103/44100,
            "license": "fixture", "source_url": "https://example.org/fixture"})
    monkeypatch.setattr(module, "evaluation_material_catalog", lambda: {"archives": [], "known_download_bytes": 0,
        "working_bytes": 0, "target_hours_by_bucket": module.BUCKET_HOURS, "limitations": ["fixture"]})
    monkeypatch.setattr(module, "_known_train", lambda *a: [])
    monkeypatch.setattr(module, "_candidate_rows", lambda *a: rows)
    result = module.prepare_common_sources(project, tmp_path / "workspace", strict_common_pool=True)
    core = [row for row in result["source_rows"] if row["positive_mix_eligible"]]
    assert len(core) == 21
    original_seconds = sum(row["source_end_s"] - row["source_start_s"] for row in core)
    assert original_seconds + .001 >= target_s
    assert result["core_coverage"]["target_met"] and result["target_met"]
    assert result["duration_budget"]["eligible_seconds"] >= target_s
