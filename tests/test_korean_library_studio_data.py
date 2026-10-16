"""Studio data planning does not download, approve or fabricate corpus scale."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import zipfile

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword import studio_data as module


def test_recipe_is_offline_pinned_and_reports_real_scale(monkeypatch, tmp_path):
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("preview must be offline"))
    plan = module.material_recipe(workspace=tmp_path)
    assert plan["known_download_bytes"] == sum(a["size_bytes"] for a in plan["assets"])
    assert plan["known_download_bytes"] > 36_000_000_000
    assert plan["automatic_target_hours"] == {"generic_nominal": 2000, "korean_train": 1}
    assert plan["human_review_required"] and not plan["Final_accessed"]
    assert all(a["source_url"].startswith("https://") and a["license"] for a in plan["assets"])
    assert module.material_recipe("test")["shortages"]
    assert module.material_recipe("full")["shortages"]
    assert not plan["shortages"]
    assert plan["additional_download"]["license"].startswith("Unknown")
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


def test_full_preparation_recipe_uses_real_qc_and_never_auto_approves(tmp_path, monkeypatch):
    """Tiny explicit fixture archives exercise all real provider/prep adapters."""
    project = create_project(tmp_path / "bora", "보라야", "bora")
    source = tmp_path / "fixture_sources"
    source.mkdir()
    assets = module._official_assets()
    paths = {}
    def wav_bytes(seed, frames=32000):
        stream = io.BytesIO()
        sf.write(stream, np.random.default_rng(seed).normal(0, .04, frames), 16000, format="WAV", subtype="PCM_16")
        return stream.getvalue()
    paths["generic"] = source / "generic.npy"
    paths["generic"].write_bytes(b"fixture precomputed array; import is explicitly mocked")
    paths["zeroth"] = source / "zeroth.tar.gz"
    with tarfile.open(paths["zeroth"], "w:gz") as archive:
        for partition, speaker in (("train_data_01", "001"), ("test_data_01", "002")):
            for name, content in [(f"{speaker}_001.wav", wav_bytes(int(speaker))),
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
    _zip(paths["fma_metadata"], [("fma_metadata/tracks.csv", csv_stream.getvalue())])
    paths["fma"] = source / "fma_small.zip"
    _zip(paths["fma"], tracks)
    for key in ("dns_fan", "dns_door", "electronic_validation"):
        paths[key] = source / (key + ".wav")
        paths[key].write_bytes(wav_bytes(len(paths) + 200))
    for asset in assets:
        asset.update(size_bytes=paths[asset["id"]].stat().st_size, license="fixture only")
    monkeypatch.setattr(module, "_official_assets", lambda: assets)
    monkeypatch.setattr(module, "_download", lambda asset, directory, progress=None: paths[asset["id"]])
    def mit(shared, progress=None):
        directory = shared / "mit_fixture"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "actual_filter.wav").write_bytes(wav_bytes(444, 1600))
        return directory
    monkeypatch.setattr(module, "_fetch_mit", mit)
    import korean_openwakeword.assets as generic
    def fake_import(path, output, frontend_identity):
        meta = {"publisher_declared_hours": 2000, "feature_path": str(path),
                "feature_sha256": assets[0]["sha256"], "canonical_frontend_assets": frontend_identity["model_sha256"]}
        write_json(Path(output) / "metadata.json", meta)
        return meta
    monkeypatch.setattr(generic, "import_generic", fake_import)
    result = module.prepare_materials(project, tmp_path / "workspace")
    assert len(result["review_queues"]) == 5
    assert result["human_listened_count"] == 0 and result["training_admitted"] is False
    assert result["generic_feature_import"]["imported"]
    assert result["shortages"]  # Tiny fixture is truthfully below the 1h quota.
    validation = read_rows(result["validation_noise_manifest"]["path"])
    assert {r["category"] for r in validation} == {"environment", "music", "competing_speech"}
    train = [row for queue in result["review_queues"] for row in read_rows(queue["pool_manifest"])]
    assert {r["canonical_source_id"] for r in train}.isdisjoint(r["canonical_source_id"] for r in validation)
    assert not read_rows(project.parent / load_project(project)[1]["manifests"]["noise"])
    assert all(not r["calls"] and r["kind"] == "negative" for r in read_rows(result["negative_validation_manifest"]["path"]))


def _additional_fixture(tmp_path, project, name, count, hours):
    folder = tmp_path / name
    folder.mkdir()
    rows = []
    for i in range(count):
        path = folder / f"source{i}.wav"
        seed = sum(name.encode()) * 100 + i
        sf.write(path, np.random.default_rng(seed).normal(0, .04, 32000), 16000, subtype="PCM_16")
        rows.append({"path": path.name, "dataset_id": name, "canonical_source_id": name + ":" + str(i),
                     "category": "environment", "speech": False, "language": "not_applicable",
                     "category_basis": "original provider fixture category", "source_url": "https://example.org/" + name,
                     "license": "fixture only"})
    metadata = tmp_path / (name + "_provider.jsonl")
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return module.prepare_additional_materials(project, tmp_path / "workspace", {
        "directory": str(folder), "dataset": "manifest", "pool": "generic_negative",
        "hours": hours, "metadata": str(metadata)})


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
