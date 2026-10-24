"""Same official-example originals over bounded selected ZIP transfers."""
from copy import deepcopy
import csv
import hashlib
import io
import re
import zipfile

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import official_music as module
from korean_openwakeword import studio_data
from korean_openwakeword.common import digest, read_json
from korean_openwakeword.range_zip import RangeZipArchive


def test_frozen_inventory_and_transfer_budget_are_the_same_120_original_tracks():
    catalog, preview = module.official_music_catalog(), module.official_music_preview()
    rows = catalog["archives"]["music"]["members"]
    assert len(rows) == 120 and rows == sorted(rows, key=lambda r: int(r["name"].split("/")[-1][:-4]))
    assert sum(r["size_bytes"] for r in rows) == 103251934
    assert preview["selected_payload_bytes"] == 120544916
    assert preview["replaced_whole_archive_bytes"] == 8038007316
    assert preview["transfer_upper_bytes"] < 122000000
    assert all(len(row["sha256"]) == 64 for source in catalog["archives"].values() for row in source["members"])


class Response:
    def __init__(self, status, headers, payload=b""):
        self.status_code, self.headers, self.payload = status, headers, payload
    def raise_for_status(self): pass
    def close(self): pass
    def iter_content(self, chunk_size):
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset:offset+chunk_size]


class Session:
    def __init__(self, data): self.data, self.requests = data, []
    def head(self, url, **kwargs):
        return Response(200, {"Content-Length": str(len(self.data[url])), "ETag": '"fixture"'})
    def get(self, url, *, headers, **kwargs):
        assert headers["If-Match"] == '"fixture"'
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        self.requests.append((url, start, end))
        return Response(206, {"Content-Range": f"bytes {start}-{end}/{len(self.data[url])}",
                              "ETag": '"fixture"', "Content-Length": str(end-start+1)}, self.data[url][start:end+1])


def fixture_catalog(tmp_path, monkeypatch, frames=160):
    original_audio = {}
    for number in range(120):
        buffer = io.BytesIO()
        sf.write(buffer, np.random.default_rng(number+1).normal(0, .05, frames), 16000, format="WAV", subtype="PCM_16")
        original_audio[f"fma_small/000/{number:06d}.mp3"] = buffer.getvalue()
    metadata = io.StringIO(newline="")
    writer = csv.writer(metadata)
    writer.writerow(["", "track", "set", "artist"]); writer.writerow(["", "license", "subset", "id"])
    writer.writerow(["track_id", "", "", ""])
    for number in range(120): writer.writerow([number, "fixture license", "small", number])
    sources = {"music": original_audio, "metadata": {"fma_metadata/tracks.csv": metadata.getvalue().encode()}}
    catalog, data, legacy = {"schema_version": 1, "archives": {}}, {}, {}
    for key, members in sources.items():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in members.items(): archive.writestr(name, payload)
            archive.writestr("unselected.bin", b"unselected"*100000)
        payload, url = buffer.getvalue(), f"https://example.org/{key}.zip"
        data[url] = payload
        local = tmp_path / (key+".zip"); local.write_bytes(payload); legacy[key] = local
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            with local.open("rb") as stream:
                size = zipfile._EndRecData(stream)[zipfile._ECD_SIZE]
            source = {"source_url": url, "size_bytes": len(payload), "etag": '"fixture"',
                      "index_bytes": size, "index_sha256": hashlib.sha256(payload[archive.start_dir:archive.start_dir+size]).hexdigest(),
                      "members": []}
            for name, original in members.items():
                info = archive.getinfo(name)
                source["members"].append({"name": name, "size_bytes": len(original), "compressed_bytes": info.compress_size,
                                         "sha256": hashlib.sha256(original).hexdigest(), "crc32": f"{info.CRC:08x}"})
            catalog["archives"][key] = source
    session = Session(data)
    monkeypatch.setattr(module, "official_music_catalog", lambda: deepcopy(catalog))
    monkeypatch.setattr(module, "RangeZipArchive", lambda *args, **kwargs: RangeZipArchive(*args, session=session, **kwargs))
    return catalog, original_audio, session, legacy


@pytest.mark.parametrize("aggregate", [False, True])
def test_selective_inputs_match_legacy_selection_original_bytes_and_pcm(tmp_path, monkeypatch, aggregate):
    catalog, originals, session, legacy = fixture_catalog(tmp_path, monkeypatch)
    observed = []
    result = module.prepare_official_music(tmp_path / "selected", observed.append, aggregate_progress=aggregate)
    legacy_dir = tmp_path / "legacy"
    studio_data._fma_selection(result["metadata"], legacy["music"], legacy_dir)
    for name, expected in originals.items():
        selected = result["music_directory"] / name
        old = legacy_dir / name
        assert selected.read_bytes() == old.read_bytes() == expected
        actual_pcm, rate = sf.read(selected, dtype="int16")
        old_pcm, old_rate = sf.read(old, dtype="int16")
        np.testing.assert_array_equal(actual_pcm, old_pcm)
        assert rate == old_rate == 16000
    assert any(row["phase"] == "download" for row in observed)
    assert sum(end-start+1 for _, start, end in session.requests) < sum(len(p.read_bytes()) for p in legacy.values()) / 4
    receipt = read_json(result["receipt"])
    assert receipt["original_bytes_preserved"] and not receipt["audio_reencoded"]
    assert not receipt["whole_archive_sha1_verified_this_run"]
    if aggregate:
        expected_total = sum(row["compressed_bytes"] for source in catalog["archives"].values() for row in source["members"])
        assert all(row["aggregate_done"] and row["total"] == expected_total for row in observed)
        assert all(0 <= row["done"] <= expected_total for row in observed)
        assert observed[-1]["done"] == expected_total
        assert observed[-1]["network_bytes"] == result["observed_transfer_bytes_this_run"]
    previous_calls = len(session.requests)
    resumed = module.prepare_official_music(tmp_path / "selected")
    assert len(session.requests) == previous_calls and resumed["observed_transfer_bytes_this_run"] == 0


def test_cached_legacy_files_are_verified_and_receipts_are_not_rewritten(tmp_path, monkeypatch):
    catalog, originals, session, _ = fixture_catalog(tmp_path, monkeypatch)
    shared = tmp_path / "shared"
    for key, source in catalog["archives"].items():
        for row in source["members"]:
            path = module._destination(shared, key, row); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(originals[row["name"]] if key == "music" else b"")
    # Obtain the exact metadata bytes from the fixture ZIP, as a historical extraction would.
    with zipfile.ZipFile(io.BytesIO(session.data[catalog["archives"]["metadata"]["source_url"]])) as archive:
        module._destination(shared, "metadata", catalog["archives"]["metadata"]["members"][0]).write_bytes(archive.read("fma_metadata/tracks.csv"))
    old_receipt = shared / "fma_1h/extraction_receipt.json"; old_receipt.write_bytes(b"historical frozen receipt")
    module.prepare_official_music(shared)
    assert not session.requests and old_receipt.read_bytes() == b"historical frozen receipt"
    changed = shared / "fma_1h" / next(iter(originals)); changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed"):
        module.prepare_official_music(shared)
    assert changed.read_bytes() == b"changed" and not session.requests


def test_interrupted_selection_resumes_only_missing_members(tmp_path, monkeypatch):
    _, originals, _, _ = fixture_catalog(tmp_path, monkeypatch)
    original_extract, counter = RangeZipArchive.extract, []
    def interrupt(self, name, destination, **kwargs):
        if name.startswith("fma_small/"):
            counter.append(name)
            if len(counter) == 3: raise KeyboardInterrupt("fixture process interrupted")
        return original_extract(self, name, destination, **kwargs)
    monkeypatch.setattr(RangeZipArchive, "extract", interrupt)
    with pytest.raises(KeyboardInterrupt): module.prepare_official_music(tmp_path / "shared")
    finished = tmp_path / "shared/fma_1h" / next(iter(originals))
    before = (digest(finished), finished.stat().st_mtime_ns)
    monkeypatch.setattr(RangeZipArchive, "extract", original_extract)
    module.prepare_official_music(tmp_path / "shared")
    assert (digest(finished), finished.stat().st_mtime_ns) == before


def test_new_policy_is_explicit_and_old_recipes_keep_complete_archive_contract(tmp_path):
    legacy = studio_data.material_recipe("check")
    selected = studio_data.material_recipe("check", official_noise_acquisition_policy=module.POLICY)
    assert "official_noise_acquisition_policy" not in legacy
    assert selected["known_download_bytes"] < legacy["known_download_bytes"] - 7900000000
    assert selected["automatic_target_hours"] == legacy["automatic_target_hours"]
    assert selected["official_music_selection"]["music_tracks"] == 120
    for row in selected["assets"]:
        if row["id"] in ("fma", "fma_metadata"):
            assert row["download_mode"] == "selected_members" and row["selected_download_bytes"] < row["size_bytes"]
    from korean_openwakeword.material_policy import _catalog_pins
    old_pins, new_pins = _catalog_pins(), _catalog_pins(module.POLICY)
    assert "official_fma_small_v1.json" not in old_pins["resources"]
    assert new_pins["resources"].pop("official_fma_small_v1.json")
    assert old_pins == new_pins  # Old completed admission bindings stay valid.


@pytest.mark.parametrize("policy", [None, module.POLICY])
def test_worker_keeps_saved_official_acquisition_policy(tmp_path, monkeypatch, policy):
    from korean_openwakeword import studio_steps
    from korean_openwakeword.project import create_project
    project = create_project(tmp_path / "word", "보라야", "bora", positive_originals=1000,
                             augmentation_rounds=1, updates=10000, generic_hours=2000, korean_hours=1)
    settings, captured = studio_steps.preset_settings("check"), []
    settings.pop("official_noise_acquisition_policy", None)
    if policy: settings["official_noise_acquisition_policy"] = policy
    monkeypatch.setattr(studio_data, "prepare_materials", lambda *a, **k: captured.append(k) or {"review_queues": []})
    studio_steps.run_stage(project, tmp_path, settings, "data_download")
    assert captured[0].get("official_noise_acquisition_policy") == policy
    assert ("official_noise_acquisition_policy" in captured[0]) is (policy is not None)


def test_selected_music_follows_real_technical_preparation_and_fixed_admission(tmp_path, monkeypatch):
    from korean_openwakeword.common import identity
    from korean_openwakeword.material_policy import fixed_queue_evidence, validate_library_policy_evidence
    from korean_openwakeword.preparation import create_source_index, prepare_public_audio
    from korean_openwakeword.project import create_project
    catalog, _, _, _ = fixture_catalog(tmp_path, monkeypatch, frames=32000)
    workspace = tmp_path / "workspace"
    selected = module.prepare_official_music(workspace / "shared")
    project = create_project(tmp_path / "word", "보라야", "bora")
    stage = project.parent / ".studio/materials"; stage.mkdir(parents=True)
    index = stage / "fma_source_index.jsonl"
    create_source_index(selected["music_directory"], output=index, dataset="fma", metadata=selected["metadata"])
    preparation = prepare_public_audio(project, input_directory=selected["music_directory"], source_index=index,
                                       output=stage / "fma", validation_fraction=0)
    queue = {"id": "materials_fma", "kind": "content", "pools": ["noise"],
             "queue": preparation["artifacts"]["representative_queue"]["path"],
             "pool_manifest": preparation["artifacts"]["source_manifest"]["path"],
             "preparation": preparation["preparation"]["path"]}
    prepared = {"kind": "studio_material_preparation_v1", "project": str(project), "workspace": str(workspace),
                "review_queues": [queue], "preparations": [{"id": "fma", "pools": ["noise"], "preparation": preparation["preparation"]}],
                "official_noise_acquisition_policy": module.POLICY,
                "official_music_preparation": {"catalog_sha256": identity(catalog),
                    "receipt": {"path": selected["receipt"], "sha256": digest(selected["receipt"])}}}
    evidence = fixed_queue_evidence(project, prepared, queue)
    artifact = {"state": "accepted", "wakeword": "보라야", "type": "library_fixed_catalog_technical_validation",
                "policy_id": "library_fixed_sources_v1", "content_review": "library_fixed_sources_technical_validation",
                "human_listening": False, "human_listened_count": 0, "human_answers": None,
                "library_evidence": evidence, "source_manifest": preparation["artifacts"]["source_manifest"]}
    assert validate_library_policy_evidence(project.parent, artifact, "noise")["scope"] == "train"
    assert not studio_data.material_review_queues(project, prepared, policy="library_fixed_sources_v1")
    selected["metadata"].write_text("changed provider metadata", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or changed"):
        validate_library_policy_evidence(project.parent, artifact, "noise")
