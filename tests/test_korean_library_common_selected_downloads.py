"""Selected common ZIPs preserve source bytes, admission and legacy evidence."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import evaluation_materials as materials, evaluation_rir as rirs
from korean_openwakeword import material_policy, selected_zip, studio_evaluation_data
from korean_openwakeword.common import digest, identity, read_json, write_json
from korean_openwakeword.project import create_project
from scripts.build_common_selected_catalog import pin_member, plan


POLICY = materials.COMMON_DOWNLOAD_POLICY


def binding(path):
    return {"path": str(path), "sha256": digest(path)}


def wav(values):
    stream = io.BytesIO()
    sf.write(stream, values, 16000, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def zip_group(tmp_path, key, originals):
    path = tmp_path / (key + ".zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, payload in originals.items():
            target.writestr(name, payload)
    volume = {"id": key, "filename": path.name, "size_bytes": path.stat().st_size,
              "source_url": "https://fixture.test/" + path.name,
              "sha256": digest(path), "checksum_algorithm": "sha256"}
    with zipfile.ZipFile(path) as source:
        members = [pin_member(source, name, [volume]) for name in originals]
    group = {"id": key, "policy": POLICY, "volumes": [volume], "members": members}
    group["plan"] = plan(members, [volume])
    return path, group


def test_frozen_transport_matches_all_original_candidates_and_original_catalogs():
    transport = materials.common_selected_download_catalog()
    common, rir = materials.evaluation_material_catalog(), rirs.common_rir_catalog()
    assert transport["groups"]["fsd_eval"]["source_catalog_sha256"] == identity(common)
    assert transport["groups"]["common_rir"]["source_catalog_sha256"] == identity(rir)
    assert [r["name"] for r in transport["groups"]["fsd_eval"]["members"]] == [
        r["member"] for r in common["selections"]["fsd_eval"]]
    assert len(transport["groups"]["fsd_eval"]["members"]) == 1861
    assert len(transport["groups"]["common_rir"]["members"]) == 80
    assert [r["sha256"] for r in transport["groups"]["common_rir"]["members"]] == [
        r["sha256"] for r in rir["assets"]]
    for group in transport["groups"].values():
        assert selected_zip.selected_transfer_bytes(group) == group["plan"]["selected_download_bytes"]
        assert sum(group["plan"]["download_bytes_by_volume"].values()) == group["plan"]["selected_download_bytes"]


def test_selected_catalog_pin_does_not_change_legacy_evidence():
    old = material_policy._catalog_pins()
    new = material_policy._catalog_pins(common_download_policy=POLICY)
    assert "common_selected_downloads_v1.json" not in old["resources"]
    assert set(new["resources"]) - set(old["resources"]) == {"common_selected_downloads_v1.json"}
    assert all(new["resources"][k] == v for k, v in old["resources"].items())
    assert {k: v for k, v in new.items() if k != "resources"} == {
        k: v for k, v in old.items() if k != "resources"}


@pytest.fixture
def sources(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    project = create_project(workspace / "projects/bora", "보라야", "bora")
    config = read_json(project)
    config["runtime"]["warmup_s"] = 0
    write_json(project, config)
    audio = wav(.1 * np.sin(np.arange(16000) / 23))
    fsd_path, fsd = zip_group(tmp_path, "fsd_eval", {
        "FSD50K.eval_audio/first.wav": audio,
        "FSD50K.eval_audio/second.wav": wav(.1 * np.sin(np.arange(16000) / 31))})
    entries = [{"member": row["name"], "dataset_id": "fixture_fsd",
                "canonical_source_id": "fixture:fsd:" + str(i),
                "canonical_recording_id": "fixture:recording:" + str(i),
                "source_url": "https://fixture.test/fsd", "license": "fixture",
                "category": "environment", "eval_bucket": "environment",
                "eval_type": "fixture_environment", "selection_duration_s": 1}
               for i, row in enumerate(fsd["members"])]
    originals, assets = {}, []
    for index, rt in enumerate((.35, .7)):
        payload = wav(.8 * np.exp(-np.arange(16000) / 16000 * np.log(1000) / rt))
        name = "RIRS_NOISES/fixture/room_%d.wav" % index
        originals[name] = payload
        estimate = rirs.estimate_rt60_t20(sf.read(io.BytesIO(payload), dtype="float64")[0], 16000)
        assets.append({"id": "fixture_rir_%d" % index, "archive_member": name,
            "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
            "canonical_source_id": "fixture:rir:%d" % index,
            "canonical_room_id": "fixture:room:%d" % index,
            "reverb_group": "moderate" if index == 0 else "strong",
            "acoustic_source_type": "simulated", "rt60_s": estimate["rt60_s"]})
    rir_path, rir_group = zip_group(tmp_path, "common_rir", originals)
    catalog = {"id": "fixture_common", "schema_version": 1,
        "archives": fsd["volumes"], "selections": {"fsd_eval": entries},
        "working_bytes": 0, "limitations": ["isolated fixture"]}
    rir_catalog = {"id": "fixture_rir", "assets": assets, "archive": rir_group["volumes"][0],
        "source_url": "https://fixture.test/rir", "license": "fixture", "scope": "fixture only"}
    transport = {"schema_version": 1, "policy": POLICY,
                 "groups": {"fsd_eval": fsd, "common_rir": rir_group}}
    for group, source in ((fsd, catalog), (rir_group, rir_catalog)):
        group["source_catalog_sha256"] = identity(source)
    monkeypatch.setattr(materials, "BUCKET_HOURS", {"environment": .4 / 3600})
    monkeypatch.setattr(materials, "evaluation_material_catalog", lambda: catalog)
    monkeypatch.setattr(materials, "common_selected_download_catalog", lambda: transport)
    monkeypatch.setattr(rirs, "common_rir_catalog", lambda: rir_catalog)
    monkeypatch.setattr(materials, "_known_train", lambda *a: [])
    monkeypatch.setattr(materials, "_zeroth_candidates", lambda *a: [])
    monkeypatch.setattr(materials, "_download_archive", lambda *a: pytest.fail("Selected audio must not fetch whole ZIP"))
    monkeypatch.setattr(studio_evaluation_data, "prepare_evaluation_materials", lambda *a, **k: pytest.fail("Common recipe must not load legacy supplements"))
    monkeypatch.setattr(selected_zip.requests, "Session", lambda: pytest.fail("Fixture must use original local ranges"))
    acquire = selected_zip.acquire_selected
    calls = []
    paths = {"fsd_eval": fsd_path, "common_rir": rir_path}
    def local(group, destination, state_dir, **kwargs):
        calls.append(group["id"])
        return acquire(group, destination, state_dir,
            **{**kwargs, "local_volumes": {group["id"]: paths[group["id"]]}})
    monkeypatch.setattr(selected_zip, "acquire_selected", local)
    return project, workspace, transport, rir_path, calls


def test_selected_common_preparation_uses_all_candidates_and_resumes(sources):
    project, workspace, transport, _, calls = sources
    first = materials.prepare_common_sources(project, workspace, strict_common_pool=True)
    assert first["target_met"] is True and first["acquisition_policy"] == POLICY
    assert first["coverage"]["actual_total_hours"] == pytest.approx(.4 / 3600)
    assert first["human_listened_count"] == 0 and first["training_admitted"] is False
    stage = workspace / "shared" / ("common_evaluation_30h_" + identity(materials.evaluation_material_catalog())[:16])
    original = stage / "originals/fsd_eval"
    assert all(digest(original / row["name"]) == row["sha256"]
               for row in transport["groups"]["fsd_eval"]["members"])
    source = first["source_rows"][0]
    expected = original / transport["groups"]["fsd_eval"]["members"][0]["name"]
    assert source["original_path"] == str(expected) and source["original_sha256"] == digest(expected)
    again = materials.prepare_common_sources(project, workspace, strict_common_pool=True)
    assert again["resumed"] is True and again["source_rows"] == first["source_rows"]
    assert calls == ["fsd_eval"]


def test_selected_rir_and_legacy_whole_archive_produce_identical_bytes(sources):
    project, workspace, _, archive, calls = sources
    cache = workspace / "shared/downloads/common_rir"
    selected = rirs.prepare_common_rirs(project, project.parent / "reports/selected", cache_directory=cache)
    legacy = rirs.prepare_common_rirs(project, project.parent / "reports/legacy", archive=archive)
    assert selected["source_rows"] and len(selected["source_rows"]) == 2
    assert [(r["sha256"], Path(r["path"]).read_bytes()) for r in selected["source_rows"]] == [
        (r["sha256"], Path(r["path"]).read_bytes()) for r in legacy["source_rows"]]
    assert "archive" not in selected and selected["archive_reference"]["whole_archive_verified"] is False
    assert "acquisition_policy" not in legacy and legacy["archive"]["sha256"] == digest(archive)
    assert rirs.prepare_common_rirs(project, project.parent / "reports/selected", cache_directory=cache) == selected
    before = (project.parent / "reports/legacy/receipt.json").read_bytes()
    assert rirs.prepare_common_rirs(project, project.parent / "reports/legacy", cache_directory=cache) == legacy
    assert (project.parent / "reports/legacy/receipt.json").read_bytes() == before
    assert calls == ["common_rir"]


def test_selected_fixed_evaluation_reaches_admission_without_whole_archives(sources):
    project, workspace, _, _, calls = sources
    prepared = {"kind": "studio_material_preparation_v1", "project": str(project),
        "workspace": str(workspace), "recipe_version": 6, "common_download_policy": POLICY,
        "evaluation_materials_policy": {"evaluation_materials_recipe": 6}}
    result = studio_evaluation_data.prepare_common_evaluation_materials(project, workspace, prepared)
    prep = result["preparation"]
    source = result["artifacts"]["source_manifest"]
    preview = result["artifacts"]["representative_queue"]
    queue = {"id": "materials_evaluation_v6_" + result["catalog_id"], "kind": "content", "pools": [],
        "evaluation_only": True, "queue": preview["path"], "pool_manifest": source["path"],
        "preparation": prep["path"]}
    prepared.update(review_queues=[queue], evaluation_preparation=prep)
    evidence = material_policy.fixed_queue_evidence(project, prepared, queue)
    admitted = studio_evaluation_data.admit_fixed_evaluation(project, queue, prepared, evidence)
    assert len(admitted["rows"]) == 3 and calls == ["fsd_eval", "common_rir"]
    assert all(row["individual_human_listening"] is False for row in admitted["rows"])
    artifact = read_json(admitted["admission"]["path"])
    assert artifact["human_answers"] is None and artifact["human_listened_count"] == 0
    record = read_json(project.parent / evidence["path"])
    assert record["recipe"]["common_download_policy"] == POLICY
    assert "common_selected_downloads_v1.json" in record["catalog_pins"]["resources"]
    assert not list((workspace / "shared/downloads").rglob("*.receipt.json"))
    # Changing an acquired inventory cannot be laundered by updating its receipt hash.
    acquisition = next((workspace / "shared/selected_common_zip_v1/fsd_eval").glob("receipt_*.json"))
    changed = read_json(acquisition)
    changed["files"] = list(reversed(changed["files"]))
    write_json(acquisition, changed)
    with pytest.raises(ValueError):
        studio_evaluation_data.admit_fixed_evaluation(project, queue, prepared, evidence)
    write_json(project.parent / "final_lock.json", {"fixture": "locked"})
    with pytest.raises(ValueError, match="Final is frozen"):
        studio_evaluation_data.admit_fixed_evaluation(project, queue, prepared, evidence)


def test_new_policy_reuses_completed_legacy_common_receipt_without_rewriting(sources):
    project, workspace, _, _, calls = sources
    selected = materials.prepare_common_sources(project, workspace, strict_common_pool=True)
    request = {k: v for k, v in selected["request"].items()
               if k not in ("acquisition_policy", "transport_catalog_sha256")}
    legacy = {k: v for k, v in selected.items()
              if k not in ("acquisition_policy", "transport_catalog_sha256", "selected_zip_acquisitions", "resumed")}
    legacy["request"] = request
    stage = workspace / "shared" / ("common_evaluation_30h_" + selected["catalog_id"])
    receipt = stage / ("selection_" + identity(request)[:16]) / "preparation.json"
    write_json(receipt, legacy)
    before = receipt.read_bytes()
    reused = materials.prepare_common_sources(project, workspace, strict_common_pool=True)
    assert reused == {**legacy, "resumed": True}
    assert receipt.read_bytes() == before and calls == ["fsd_eval"]


@pytest.mark.parametrize("change", ["order", "scope", "destination", "request"])
def test_selected_lineage_rejects_rebound_forgery(sources, change):
    project, workspace, transport, _, _ = sources
    key = "fsd_eval"
    state = workspace / "shared/selected_common_zip_v1" / key
    destination = workspace / "shared/originals"
    acquired = selected_zip.acquire_selected(transport["groups"][key], destination, state)
    proof = read_json(acquired["receipt"]["path"])
    if change == "order":
        proof["files"].reverse()
    elif change == "scope":
        proof["whole_archive_verified"] = True
    elif change == "destination":
        proof["files"][0]["path"] = str(project.parent / "elsewhere.wav")
    else:
        proof["request"]["group_sha256"] = "0" * 64
    write_json(acquired["receipt"]["path"], proof)
    with pytest.raises(ValueError, match="selected common"):
        material_policy._selected_common_lineage(project.parent, workspace, key,
            binding(Path(acquired["receipt"]["path"])), destination, {})
