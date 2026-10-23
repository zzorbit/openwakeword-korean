"""Studio data planning does not download, approve or fabricate corpus scale."""
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import tarfile
import zipfile

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword import studio_data as module


def test_retired_transport_refuses_material_preparation_before_download_or_writes(tmp_path, monkeypatch):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    config = read_json(project)
    config["common_download_policy"] = "selected_common_zip_v1"
    write_json(project, config)
    before = project.read_bytes()
    monkeypatch.setattr(module, "_download", lambda *a, **k: pytest.fail("No retired request may download"))
    workspace = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="0.16"):
        module.prepare_materials(project, workspace)
    assert project.read_bytes() == before
    assert not workspace.exists()


def test_recipe_is_offline_pinned_and_reports_real_scale(monkeypatch, tmp_path):
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("preview must be offline"))
    plan = module.material_recipe(workspace=tmp_path)
    assert plan["known_download_bytes"] == sum(a["size_bytes"] for a in plan["assets"]) + plan["evaluation_download_bytes"]
    assert plan["known_download_bytes"] > 36_000_000_000
    assert plan["automatic_target_hours"] == {"generic_nominal": 2000, "korean_train": 1}
    assert plan["human_review_required"] and not plan["Final_accessed"]
    assert all(a["source_url"].startswith("https://") and a["license"] for a in plan["assets"])
    assert module.material_recipe("test")["shortages"]
    assert module.material_recipe("full")["shortages"]
    assert not plan["shortages"]
    assert plan["additional_download"]["license"].startswith("Unknown")
    assert plan["additional_download"]["files"] == 270
    for preset in ("check", "test", "full"):
        selected = module.material_recipe(preset)
        assert selected["additional_download"]["files"] == 270
        assets = {asset["id"]: asset for asset in selected["assets"]}
        assert "dns_fan" not in assets
        if preset == "full":
            assert selected["noise_material_policy"] == "reuse_negative_waveforms"
            assert set(assets) == {"generic", "zeroth"}
        else:
            assert selected["noise_material_policy"] == "official_example"
            assert assets["audioset"]["sha256"] == "9ab88550e77c8289acf71d19ae3f254fadad16f3cea305ef08a16949a928acf9"
    with pytest.raises(ValueError):
        module.material_recipe("giant")


def test_recipe_cache_availability_does_not_imply_integrity(tmp_path):
    plan = module.material_recipe(workspace=tmp_path)
    first = plan["assets"][0]
    path = Path(first["local_path"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not verified")
    assert module.material_recipe(workspace=tmp_path)["assets"][0]["availability"] == "cached_requires_integrity_check"


def test_documented_manual_cache_is_reused_without_rewriting_its_receipt(tmp_path, monkeypatch):
    directory = tmp_path / "shared/downloads"
    manual = tmp_path / "shared/source.wav"
    manual.parent.mkdir(parents=True)
    manual.write_bytes(b"verified public audio fixture")
    original = manual.with_name(manual.name + ".receipt.json")
    original.write_bytes(b"original receipt bound by an older project")
    checksum = digest(manual)
    asset = {"id": "fixture", "filename": manual.name, "sha256": checksum, "size_bytes": manual.stat().st_size,
             "checksum_algorithm": "sha256", "source_url": "https://example.org/fixture", "license": "fixture"}
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("reuse may not download"))
    assert module._download(asset, directory) == manual
    assert original.read_bytes() == b"original receipt bound by an older project"
    assert read_json(directory / "source.wav.reused.receipt.json")["reused"]


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in names:
            archive.writestr(name, value)


def test_archive_extract_exact_subset_resume_and_tamper_detection(tmp_path):
    archive = tmp_path / "source.zip"
    _zip(archive, [("folder/a.wav", b"aaa"), ("folder/b.wav", b"bbb")])
    output = tmp_path / "out"
    result = module.extract_archive(archive, output, members=["folder/a.wav"])
    assert result["bytes"] == 3 and len(result["files"]) == 1
    assert not (output / "folder/b.wav").exists()
    assert module.extract_archive(archive, output, members=["folder/a.wav"])["resumed"]
    (output / "folder/a.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.extract_archive(archive, output, members=["folder/a.wav"])


def _large_declared_zip(path):
    """Tiny ZIP64 fixture declares the publisher's 4.44 GB nested gzip size."""
    _zip(path, [("folder/a.flac", b"audio"), ("folder/a.trans.txt", b"transcript"),
                ("README.txt", b"metadata"), ("nested/train_data_01.tar.gz", b"")])
    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        start = archive.start_dir
    position = start
    while raw[position:position + 4] == b"PK\x01\x02":
        header = list(struct.unpack_from(zipfile.structCentralDir, raw, position))
        filename_end = position + 46 + header[zipfile._CD_FILENAME_LENGTH]
        name = bytes(raw[position + 46:filename_end]).decode()
        if name.endswith(".tar.gz"):
            extra = struct.pack("<HHQ", 1, 8, 4443000000)
            header[zipfile._CD_UNCOMPRESSED_SIZE] = 0xffffffff
            header[zipfile._CD_EXTRA_FIELD_LENGTH] += len(extra)
            struct.pack_into(zipfile.structCentralDir, raw, position, *header)
            raw[filename_end:filename_end] = extra
            end = raw.rfind(b"PK\x05\x06")
            footer = list(struct.unpack_from("<4s4H2LH", raw, end))
            footer[5] += len(extra)
            struct.pack_into("<4s4H2LH", raw, end, *footer)
            path.write_bytes(raw)
            return
        position = filename_end + header[zipfile._CD_EXTRA_FIELD_LENGTH] + header[zipfile._CD_COMMENT_LENGTH]
    pytest.fail("Missing large fixture member")


def test_suffix_selection_skips_large_unneeded_payloads_and_resumes_without_rewriting(tmp_path):
    archive = tmp_path / "publisher.zip"
    _large_declared_zip(archive)
    with zipfile.ZipFile(archive) as handle:
        assert handle.getinfo("nested/train_data_01.tar.gz").file_size == 4443000000
    before_archive = archive.read_bytes()
    with pytest.raises(ValueError, match="safety limit"):
        module.extract_archive(archive, tmp_path / "unfiltered")
    output = tmp_path / "selected"
    result = module.extract_archive(archive, output, include_suffixes=(".flac", ".txt"))
    assert {row["path"] for row in result["files"]} == {"folder/a.flac", "folder/a.trans.txt", "README.txt"}
    assert result["bytes"] == len(b"audiotranscriptmetadata")
    assert result["request"]["selector"] == {"version": 1, "include_suffixes": [".flac", ".txt"]}
    assert not (output / "nested/train_data_01.tar.gz").exists()
    before_receipt = (output / "extraction_receipt.json").read_bytes()
    assert module.extract_archive(archive, output, include_suffixes=(".txt", ".FLAC"))["resumed"]
    assert (output / "extraction_receipt.json").read_bytes() == before_receipt
    assert archive.read_bytes() == before_archive


def test_default_archive_request_identity_and_zeroth_legacy_member_resume_are_unchanged(tmp_path):
    archive = tmp_path / "legacy.zip"
    _zip(archive, [("a.flac", b"audio"), ("README.txt", b"metadata"), ("old.gz", b"old")])
    for mode, options in (("all", {}), ("subset", {"members": ["a.flac", "README.txt"], "max_bytes": 100})):
        output = tmp_path / mode
        result = module.extract_archive(archive, output, **options)
        assert result["request"] == {"archive": {"path": str(archive.resolve()), "sha256": digest(archive)},
            "members": sorted(options["members"]) if "members" in options else None, "max_bytes": options.get("max_bytes", 40 * 1024**3)}
        receipt = output / "extraction_receipt.json"
        before = receipt.read_bytes()
        assert module._extract_zeroth_archive(archive, output)["resumed"]
        assert receipt.read_bytes() == before
        with pytest.raises(ValueError, match="request changed"):
            module.extract_archive(archive, output, include_suffixes=(".flac", ".txt"))
        assert receipt.read_bytes() == before


@pytest.mark.parametrize("failure", ["unsafe", "symlink", "duplicate", "missing", "selected_large"])
def test_suffix_selector_preserves_full_inventory_safety_and_missing_member_checks(tmp_path, failure):
    archive = tmp_path / "unsafe.zip"
    if failure == "selected_large":
        _large_declared_zip(archive)
        with pytest.raises(ValueError, match="safety limit"):
            module.extract_archive(archive, tmp_path / "out", include_suffixes=(".gz",))
        return
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.flac", b"audio")
        if failure == "unsafe":
            handle.writestr("../skip.gz", b"bad")
        elif failure == "symlink":
            info = zipfile.ZipInfo("skip.gz"); info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            handle.writestr(info, "../outside")
        elif failure == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                handle.writestr("skip.gz", b"first"); handle.writestr("skip.gz", b"second")
    with pytest.raises(ValueError, match="Unsafe|escapes|links|Duplicate|missing"):
        module.extract_archive(archive, tmp_path / "out", include_suffixes=(".flac", ".txt"),
                               members=["absent.flac"] if failure == "missing" else None)
    assert not (tmp_path / "out/a.flac").exists()


@pytest.mark.parametrize("suffixes", [[], [""], ["flac"], ["../flac"], [".a/b"], [".a\\b"], [None], ".flac"])
def test_invalid_suffix_selector_is_rejected(tmp_path, suffixes):
    archive = tmp_path / "source.zip"
    _zip(archive, [("audio.flac", b"audio")])
    with pytest.raises(ValueError, match="suffix selector"):
        module.extract_archive(archive, tmp_path / "out", include_suffixes=suffixes)


def test_suffix_selector_still_rejects_unselected_tar_special_members(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("audio.flac"); info.size = 5
        handle.addfile(info, io.BytesIO(b"audio"))
        special = tarfile.TarInfo("unselected.gz")
        special.type = tarfile.LNKTYPE; special.linkname = "outside"
        handle.addfile(special)
    with pytest.raises(ValueError, match="links and special"):
        module.extract_archive(archive, tmp_path / "out", include_suffixes=(".flac", ".txt"))
    assert not (tmp_path / "out/audio.flac").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "CON.wav", "a./b", "file "])
def test_zip_rejects_unsafe_even_unselected_member(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    _zip(archive, [("fine.wav", b"fine"), (name, b"bad")])
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        module.extract_archive(archive, tmp_path / "out", members=["fine.wav"])
    assert not (tmp_path / "escape").exists()


def test_archive_path_rejects_windows_separator_before_zip_normalization(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        module._archive_target(tmp_path.resolve(), "a\\b")


def test_zip_rejects_symlink_and_tar_rejects_hardlink(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        stream.writestr(info, "../outside")
    with pytest.raises(ValueError, match="links"):
        module.extract_archive(archive, tmp_path / "zip")
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("link")
        info.type, info.linkname = tarfile.LNKTYPE, "outside"
        stream.addfile(info)
    with pytest.raises(ValueError, match="links"):
        module.extract_archive(archive, tmp_path / "tar")


def test_archive_missing_subset_and_budget_rejected_before_publish(tmp_path):
    archive = tmp_path / "source.zip"
    _zip(archive, [("audio.wav", b"hello")])
    with pytest.raises(ValueError, match="missing"):
        module.extract_archive(archive, tmp_path / "missing", members=["not-there"])
    with pytest.raises(ValueError, match="budget"):
        module.extract_archive(archive, tmp_path / "budget", max_bytes=4)
    assert not (tmp_path / "budget/audio.wav").exists()


def test_tar_bounded_extraction(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("nested/source.txt")
        info.size = 7
        stream.addfile(info, io.BytesIO(b"content"))
    receipt = module.extract_archive(archive, tmp_path / "out")
    assert (tmp_path / "out/nested/source.txt").read_bytes() == b"content"
    assert receipt["content_approved"] is False


def test_extraction_progress_is_separate_from_download_and_preserves_exact_files(tmp_path):
    archive = tmp_path / "source.zip"
    _zip(archive, [("a.wav", b"original audio"), ("b.wav", b"other original")])
    observed = []
    result = module.extract_archive(archive, tmp_path / "out", progress=observed.append)
    assert all(row["phase"] == "extract" for row in observed)
    assert observed[-1]["done"] == observed[-1]["total"] == result["bytes"]
    assert (tmp_path / "out/a.wav").read_bytes() == b"original audio"
    assert (tmp_path / "out/b.wav").read_bytes() == b"other original"


class _Response:
    status_code = 200
    headers = {}
    def __init__(self, content):
        self.content = content
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, size):
        yield self.content


@pytest.mark.parametrize("algorithm", ["sha1", "git_sha1"])
def test_original_upstream_checksum_download_resume_and_receipt(tmp_path, monkeypatch, algorithm):
    content = b"source audio fixture"
    prefix = f"blob {len(content)}\0".encode() if algorithm == "git_sha1" else b""
    checksum = hashlib.sha1(prefix + content).hexdigest()
    asset = {"id": "fixture", "filename": "source", "source_url": "https://example.org/original",
             "size_bytes": len(content), "checksum_algorithm": algorithm, algorithm: checksum,
             "license": "fixture only"}
    calls = []
    def get(*args, **kwargs):
        calls.append(args)
        return _Response(content)
    monkeypatch.setattr(module.requests, "get", get)
    path = module._download(asset, tmp_path)
    assert path.read_bytes() == content
    record = read_json(path.with_name("source.receipt.json"))
    assert record["verified_upstream_checksum"] == checksum and not record["content_approved"]
    module._download(asset, tmp_path)
    assert len(calls) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        module._download(asset, tmp_path)


def test_bad_upstream_checksum_preserves_partial(tmp_path, monkeypatch):
    asset = {"id": "fixture", "filename": "bad", "source_url": "https://example.org/original",
             "size_bytes": 3, "checksum_algorithm": "sha1", "sha1": "0" * 40, "license": "fixture"}
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: _Response(b"bad"))
    with pytest.raises(ValueError, match="checksum"):
        module._download(asset, tmp_path)
    assert not (tmp_path / "bad").exists() and (tmp_path / "bad.part").exists()


def _prepared_fixture(tmp_path):
    from korean_openwakeword.preparation import prepare_public_audio
    project = create_project(tmp_path / "bora", "보라야", "bora")
    directory = tmp_path / "provider"
    directory.mkdir()
    for n in range(5):
        sf.write(directory / f"source{n}.wav", np.random.default_rng(n).normal(0, .05, 32000), 16000, subtype="PCM_16")
    receipt = prepare_public_audio(project, input_directory=directory, output="data/prepared/fixture",
        dataset_id="fixture", source_url="https://example.org/fixture", license="fixture only",
        category="environment", speech=False, category_basis="provider fixture", representatives_per_stratum=2,
        validation_fraction=.2)
    queue = {"id": "materials_fixture", "kind": "content", "preparation": receipt["preparation"]["path"],
             "queue": receipt["artifacts"]["representative_queue"]["path"],
             "pool_manifest": receipt["artifacts"]["source_manifest"]["path"], "pools": ["noise"]}
    prepared = {"review_queues": [queue], "validation_noise_manifest": receipt["artifacts"]["validation_reserve"],
                "negative_validation_manifest": receipt["artifacts"]["validation_reserve"], "shortages": []}
    return project, receipt, queue, prepared


def test_admission_requires_real_listening_and_whole_pool_consent(tmp_path):
    project, receipt, queue, prepared = _prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="Saved representative"):
        module.admit_materials(project, prepared)
    answers = project.parent / "review/answers.json"
    write_json(answers, {"whole_pool_policy_approved": False, "decisions": {}})
    with pytest.raises(ValueError, match="consent"):
        module.admit_materials(project, prepared, {queue["id"]: str(answers)})
    assert not read_rows(project.parent / "data/augmentation/noise.jsonl")


def test_admission_reuses_existing_review_pipeline_and_keeps_reserve_out_of_train(tmp_path):
    project, receipt, queue, prepared = _prepared_fixture(tmp_path)
    previews = read_rows(queue["queue"])
    answers = project.parent / "review/answers.json"
    write_json(answers, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "fixture",
        "source_scope": "Human dispositions entered through the local review screen",
        "decisions": {r["sample_id"]: {"decision": "keep", "listened": True} for r in previews},
        "whole_pool_policy_approved": True,
        "source_manifest_sha256": receipt["artifacts"]["source_manifest"]["sha256"],
        "representative_queue_sha256": receipt["artifacts"]["representative_queue"]["sha256"],
        "queue_sha256": receipt["artifacts"]["representative_queue"]["sha256"],
        "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews}})
    result = module.admit_materials(project, prepared, {queue["preparation"]: {"path": str(answers), "sha256": digest(answers)}})
    assert result["state"] == "connected" and not result["Final_accessed"]
    root, config = load_project(project)
    train = read_rows(root / config["manifests"]["noise"])
    reserve = read_rows(root / config["manifests"]["noise_validation"])
    assert train and reserve
    assert {r["canonical_source_id"] for r in train}.isdisjoint(r["canonical_source_id"] for r in reserve)
    assert module.admit_materials(project, prepared, {queue["id"]: str(answers)})["connection"]["resumed"]


@pytest.mark.parametrize("preset", ["check", "test", "full"])
def test_full_preparation_recipe_uses_real_qc_and_never_auto_approves(tmp_path, monkeypatch, preset):
    """Tiny explicit fixture archives exercise all real provider/prep adapters."""
    project = create_project(tmp_path / "bora", "보라야", "bora")
    # This legacy-corpus adapter fixture does not download new evaluation media;
    # evaluation-only preparation/admission has separate tests below.
    monkeypatch.setattr(module, "ensure_evaluation_materials", lambda project, workspace, prepared, progress=None: prepared)
    source = tmp_path / "fixture_sources"
    source.mkdir()
    assets = module._official_assets()
    paths = {}
    def wav_bytes(seed, frames=32000, file_format="WAV"):
        stream = io.BytesIO()
        sf.write(stream, np.random.default_rng(seed).normal(0, .04, frames), 16000, format=file_format, subtype="PCM_16")
        return stream.getvalue()
    paths["generic"] = source / "generic.npy"
    paths["generic"].write_bytes(b"fixture precomputed array; import is explicitly mocked")
    paths["zeroth"] = source / "zeroth.tar.gz"
    with tarfile.open(paths["zeroth"], "w:gz") as archive:
        for partition, speaker in (("train_data_01", "001"), ("test_data_01", "002")):
            for name, content in [(f"{speaker}_001.flac", wav_bytes(int(speaker), file_format="FLAC")),
                                  ("transcripts.trans.txt", f"{speaker}_001 이것은 제공자 전사입니다\n".encode())]:
                info = tarfile.TarInfo(f"zeroth/{partition}/{speaker}/{name}")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    paths["audioset"] = source / "audioset.tar"
    with tarfile.open(paths["audioset"], "w") as archive:
        info = tarfile.TarInfo("audio/abcDEF12345.wav")
        content = wav_bytes(99)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    paths["audioset_segments"] = source / "segments.csv"
    paths["audioset_segments"].write_text('abcDEF12345,0,2,"/m/09x0r"\n', encoding="utf-8")
    paths["audioset_ontology"] = source / "ontology.json"
    write_json(paths["audioset_ontology"], [{"id": "/m/09x0r", "name": "Speech", "child_ids": []},
        {"id": "/m/04rlf", "name": "Music", "child_ids": []}])
    csv_stream = io.StringIO()
    import csv
    writer = csv.writer(csv_stream)
    writer.writerow(["", "artist", "track", "set", "track", "set"])
    writer.writerow(["", "id", "license", "split", "url", "subset"])
    writer.writerow(["track_id", "", "", "", "", ""])
    tracks = []
    for n in range(120):
        writer.writerow([n, n, "fixture license", "training", "https://example.org/fixture", "small"])
        # Header-based libsndfile decoding is intentional for this tiny fixture;
        # no claim is made that a WAV header is an actual FMA MP3.
        tracks.append((f"fma_small/000/{n:06d}.mp3", wav_bytes(n + 100, 1600)))
    paths["fma_metadata"] = source / "fma_metadata.zip"
    _zip(paths["fma_metadata"], [("fma_metadata/tracks.csv", csv_stream.getvalue()),
        ("fma_metadata/features.csv", "unselected feature table"),
        ("fma_metadata/echonest.csv", "unselected feature table")])
    real_extract = module.extract_archive
    metadata_extractions = []
    def extract(archive, output, *, members=None, max_bytes=40 * 1024**3, include_suffixes=None):
        if Path(archive) == paths["fma_metadata"]:
            # Regression for the real pinned tracks.csv (260,414,445 bytes):
            # a 200 MiB budget rejects it before writing. The guard remains
            # bounded, and no unrelated large metadata tables are selected.
            assert 260_414_445 <= max_bytes <= 512 * 1024**2
            assert members == ["fma_metadata/tracks.csv"]
            metadata_extractions.append(Path(output))
        return real_extract(archive, output, members=members, max_bytes=max_bytes, include_suffixes=include_suffixes)
    monkeypatch.setattr(module, "extract_archive", extract)
    paths["fma"] = source / "fma_small.zip"
    _zip(paths["fma"], tracks)
    for key in ("dns_fan", "dns_door", "electronic_validation"):
        paths[key] = source / (key + ".wav")
        paths[key].write_bytes(wav_bytes(len(paths) + 200))
    for asset in assets:
        asset.update(size_bytes=paths[asset["id"]].stat().st_size, license="fixture only")
    monkeypatch.setattr(module, "_official_assets", lambda: assets)
    requested_assets = []
    def download(asset, directory, progress=None, verification_cache=None):
        requested_assets.append(asset["id"])
        return paths[asset["id"]]
    monkeypatch.setattr(module, "_download", download)
    def mit(shared, progress=None):
        directory = shared / "mit_fixture"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "actual_filter.wav").write_bytes(wav_bytes(444, 1600))
        return directory
    monkeypatch.setattr(module, "_fetch_mit", mit)
    import korean_openwakeword.assets as generic
    def fake_import(path, output, frontend_identity, *, verification_cache=None):
        meta = {"publisher_declared_hours": 2000, "feature_path": str(path),
                "feature_sha256": assets[0]["sha256"], "canonical_frontend_assets": frontend_identity["model_sha256"]}
        write_json(Path(output) / "metadata.json", meta)
        return meta
    monkeypatch.setattr(generic, "import_generic", fake_import)
    workspace = tmp_path / "workspace"
    stale_fan = workspace / "shared/check_noise/fan_Freesound_validated_100355_0.wav"
    stale_fan.parent.mkdir(parents=True)
    stale_fan.write_bytes(wav_bytes(555))
    legacy_index = project.parent / ".studio/materials/extra_source_index.jsonl"
    legacy_index.parent.mkdir(parents=True)
    legacy_content = '{"canonical_source_id":"freesound:100355"}\n'
    legacy_index.write_text(legacy_content, encoding="utf-8")
    old_fan_hash = digest(stale_fan)
    result = module.prepare_materials(project, workspace, preset=preset)
    expected = {"zeroth", "mit_rir"} if preset == "full" else {"zeroth", "audioset", "fma", "mit_rir"}
    assert len(result["review_queues"]) == len(expected)
    assert "dns_fan" not in requested_assets
    assert {row["id"] for row in result["preparations"]} == expected
    assert result["noise_material_policy"] == ("reuse_negative_waveforms" if preset == "full" else "official_example")
    if preset == "full":
        assert set(requested_assets) == {"generic", "zeroth"}
        assert not (workspace / "shared/audioset").exists()
        assert not (workspace / "shared/fma_1h").exists()
        assert not metadata_extractions
    else:
        assert len(metadata_extractions) == 1
        assert (workspace / "shared/fma_metadata/fma_metadata/tracks.csv").is_file()
        assert not (workspace / "shared/fma_metadata/fma_metadata/features.csv").exists()
        assert not (workspace / "shared/fma_metadata/fma_metadata/echonest.csv").exists()
    assert digest(stale_fan) == old_fan_hash
    assert legacy_index.read_text(encoding="utf-8") == legacy_content
    assert result["human_listened_count"] == 0 and result["training_admitted"] is False
    assert result["generic_feature_import"]["imported"]
    assert result["shortages"]  # Tiny fixture is truthfully below the 1h quota.
    validation = read_rows(result["validation_noise_manifest"]["path"])
    assert {r["category"] for r in validation} == {"competing_speech"}
    train = [row for queue in result["review_queues"] for row in read_rows(queue["pool_manifest"])]
    assert all(row["canonical_source_id"] != "freesound:100355" for row in train)
    assert {r["canonical_source_id"] for r in train}.isdisjoint(r["canonical_source_id"] for r in validation)
    assert not read_rows(project.parent / load_project(project)[1]["manifests"]["noise"])
    assert all(not r["calls"] and r["kind"] == "negative" for r in read_rows(result["negative_validation_manifest"]["path"]))


def _additional_fixture(tmp_path, project, name, count, hours, *, noise_material_policy=None,
                        pool="generic_negative", category="environment", speech=False, language="not_applicable"):
    folder = tmp_path / name
    folder.mkdir()
    rows = []
    for i in range(count):
        path = folder / f"source{i}.wav"
        seed = sum(name.encode()) * 100 + i
        sf.write(path, np.random.default_rng(seed).normal(0, .04, 32000), 16000, subtype="PCM_16")
        rows.append({"path": path.name, "dataset_id": name, "canonical_source_id": name + ":" + str(i),
                     "category": category, "speech": speech, "language": language,
                     "category_basis": "original provider fixture category", "source_url": "https://example.org/" + name,
                     "license": "fixture only"})
    metadata = tmp_path / (name + "_provider.jsonl")
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return module.prepare_additional_materials(project, tmp_path / "workspace", {
        "directory": str(folder), "dataset": "manifest", "pool": pool,
        "hours": hours, "metadata": str(metadata)}, noise_material_policy=noise_material_policy)


def _additional_answers(project, result):
    queue = result["review_queue"]
    record = read_json(result["preparation"]["path"])
    previews = read_rows(queue["queue"])
    path = project.parent / "review" / (queue["id"] + ".json")
    write_json(path, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "fixture",
        "source_scope": "Human dispositions entered through the local review screen",
        "decisions": {r["sample_id"]: {"decision": "keep", "listened": True} for r in previews},
        "whole_pool_policy_approved": True,
        "source_manifest_sha256": record["artifacts"]["source_manifest"]["sha256"],
        "representative_queue_sha256": record["artifacts"]["representative_queue"]["sha256"],
        "queue_sha256": record["artifacts"]["representative_queue"]["sha256"],
        "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews}})
    return {queue["id"]: str(path)}


def test_additional_materials_require_provider_metadata_and_report_actual_duration(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    folder = tmp_path / "public"
    folder.mkdir()
    with pytest.raises(ValueError, match="provider metadata"):
        module.prepare_additional_materials(project, tmp_path, {"directory": str(folder), "dataset": "manifest",
            "pool": "generic_negative", "hours": 1})
    result = _additional_fixture(tmp_path, project, "extra", 5, .002)
    assert result["target_met"] and result["train_hours"] == pytest.approx(8 / 3600)
    assert not result["training_admitted"] and result["human_listened_count"] == 0
    assert result["review_queue"]["pools"] == ["generic_negative"]


def test_additional_shortage_does_not_freeze_pools_and_raw_generic_preserves_old_cache(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora", generic_hours=.003)
    original = project.parent / "features/generic_negative/old_import.txt"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"preserve legacy features")
    first = _additional_fixture(tmp_path, project, "first", 5, .002)
    prepared = {"review_queues": [first["review_queue"]], "shortages": []}
    answers = _additional_answers(project, first)
    pending = module.admit_materials(project, prepared, answers)
    assert pending["state"] == "awaiting_additional_materials" and pending["shortages"]
    root, config = load_project(project)
    assert not read_rows(root / config["manifests"]["generic_negative"])
    second = _additional_fixture(tmp_path, project, "second", 3, .001)
    prepared["review_queues"].append(second["review_queue"])
    answers.update(_additional_answers(project, second))
    result = module.admit_materials(project, prepared, answers)
    assert result["state"] == "connected" and not result["shortages"]
    root, config = load_project(project)
    assert config["pool_features"]["generic_negative"].startswith("features/generic_negative_raw_")
    assert result["generic_feature_import"]["mode"] == "raw_generic"
    assert result["admitted_unique_hours"]["generic_negative"] == pytest.approx(12 / 3600)
    assert original.read_bytes() == b"preserve legacy features"
    train = read_rows(root / config["manifests"]["generic_negative"])
    reserve = read_rows(result["negative_validation_manifest"]["path"])
    assert {r["canonical_source_id"] for r in train}.isdisjoint(r["canonical_source_id"] for r in reserve)


def test_reused_raw_negative_sources_are_reviewed_for_both_pools_and_keep_reserves_separate(tmp_path):
    policy = "reuse_negative_waveforms"
    project = create_project(tmp_path / "word", "보라야", "bora", generic_hours=.002, korean_hours=.001)
    prepared = {"review_queues": [], "noise_material_policy": policy}
    answers = {}
    for name, pool, category, speech, language in (
            ("environment", "generic_negative", "environment", False, "not_applicable"),
            ("music", "generic_negative", "music", False, "not_applicable"),
            ("speech", "korean_negative", "competing_speech", True, "ko")):
        result = _additional_fixture(tmp_path, project, name, 5, .001, noise_material_policy=policy,
            pool=pool, category=category, speech=speech, language=language)
        assert result["review_queue"]["pools"] == [pool, "noise"]
        assert not result["training_admitted"] and result["human_listened_count"] == 0
        prepared["review_queues"].append(result["review_queue"])
        answers.update(_additional_answers(project, result))
    connection = module.admit_materials(project, prepared, answers)
    assert connection["state"] == "connected" and not connection["shortages"]
    root, config = load_project(project)
    noise = read_rows(root / config["manifests"]["noise"])
    negatives = [row for pool in ("generic_negative", "korean_negative")
                 for row in read_rows(root / config["manifests"][pool])]
    assert {r["sha256"] for r in noise} == {r["sha256"] for r in negatives}
    assert {r["category"] for r in noise} == {"environment", "music", "competing_speech"}
    reserve = read_rows(connection["negative_validation_manifest"]["path"])
    assert {r["canonical_source_id"] for r in noise}.isdisjoint(r["canonical_source_id"] for r in reserve)


def test_reuse_does_not_relabel_old_single_pool_receipt_or_freeze_partial_materials(tmp_path):
    from korean_openwakeword.preparation import apply_prepared_review
    project = create_project(tmp_path / "word", "보라야", "bora", generic_hours=.001)
    old = _additional_fixture(tmp_path, project, "legacy", 5, .001)
    answers = _additional_answers(project, old)
    admission = apply_prepared_review(project, preparation=old["preparation"],
        answers=answers[old["review_queue"]["id"]], pools=["generic_negative"], connect=False)
    before = Path(admission["receipt"]["path"]).read_bytes()
    result = module.admit_materials(project, {"review_queues": [], "noise_material_policy": "reuse_negative_waveforms"},
        admissions=[admission["receipt"]])
    assert result["state"] == "awaiting_additional_materials"
    assert any("기존 단일 용도 승인" in reason for reason in result["shortages"])
    assert Path(admission["receipt"]["path"]).read_bytes() == before
    assert not read_rows(project.parent / read_json(project)["manifests"]["noise"])


def test_reuse_requires_existing_noise_categories_before_connecting(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora", generic_hours=.001)
    source = _additional_fixture(tmp_path, project, "environment", 5, .001,
                                 noise_material_policy="reuse_negative_waveforms")
    result = module.admit_materials(project, {"review_queues": [source["review_queue"]],
        "noise_material_policy": "reuse_negative_waveforms"}, _additional_answers(project, source))
    assert result["state"] == "awaiting_additional_materials"
    assert any("music" in reason and "competing_speech" in reason for reason in result["shortages"])
    assert not read_rows(project.parent / read_json(project)["manifests"]["noise"])


def test_reuse_rejects_separate_noise_source_and_feature_only_input(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    folder = tmp_path / "features"
    folder.mkdir()
    np.save(folder / "generic.npy", np.zeros((2, 16, 96), dtype=np.float32))
    with pytest.raises(ValueError, match="not a separate noise source"):
        module.prepare_additional_materials(project, tmp_path, {"pool": "noise", "directory": str(folder)},
            noise_material_policy="reuse_negative_waveforms")
    metadata = tmp_path / "provider.jsonl"
    metadata.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        module.prepare_additional_materials(project, tmp_path, {"pool": "generic_negative", "directory": str(folder),
            "dataset": "manifest", "metadata": str(metadata), "hours": 1}, noise_material_policy="reuse_negative_waveforms")
    assert not read_rows(project.parent / read_json(project)["manifests"]["noise"])
