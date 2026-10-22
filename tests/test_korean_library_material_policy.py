"""Fixed-source admission needs pinned preparation evidence, never a label."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, resolved, write_json
from korean_openwakeword.material_policy import fixed_queue_evidence, validate_library_policy_evidence
from korean_openwakeword.preparation import create_source_index, prepare_public_audio
from korean_openwakeword.project import create_project


def _bound(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def _rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def custom_material(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    directory = tmp_path / "user_audio"
    directory.mkdir()
    sf.write(directory / "user.wav", np.random.default_rng(7).normal(0, .05, 48000),
             16000, subtype="PCM_16")
    preparation = prepare_public_audio(project, input_directory=directory,
        output=project.parent / "prepared" / "user", dataset_id="user_added",
        source_url="https://example.org/user-audio", license="fixture permission",
        category="environment", category_basis="User-supplied fixture declaration",
        speech=False, validation_fraction=0)
    queue = {"id": "materials_user_added", "kind": "content", "pools": ["noise"],
             "queue": preparation["artifacts"]["representative_queue"]["path"],
             "pool_manifest": preparation["artifacts"]["source_manifest"]["path"],
             "preparation": preparation["preparation"]["path"]}
    prepared = {"kind": "studio_material_preparation_v1", "project": str(project),
                "workspace": str(tmp_path), "review_queues": [queue],
                "preparations": [{"id": "user_added", "pools": ["noise"],
                                  "preparation": preparation["preparation"]}]}
    return project, prepared, queue


def test_custom_material_stays_in_human_review(custom_material):
    project, prepared, queue = custom_material
    assert fixed_queue_evidence(project, prepared, queue) is None
    assert not (project.parent / "review" / "library_fixed_sources").exists()


def test_non_studio_preparation_does_not_receive_fixed_source_evidence(custom_material):
    project, prepared, queue = custom_material
    prepared["kind"] = "user_material_preparation"
    assert fixed_queue_evidence(project, prepared, queue) is None


@pytest.mark.parametrize("source_key,pools", [
    ("zeroth", ["korean_negative", "noise"]), ("audioset", ["noise"]),
    ("fma", ["noise"]), ("mit_rir", ["rir"]),
])
def test_official_queue_name_cannot_admit_user_audio(custom_material, source_key, pools):
    project, prepared, queue = custom_material
    queue.update(id="materials_" + source_key, pools=pools)
    prepared["preparations"][0].update(id=source_key, pools=pools)
    try:
        evidence = fixed_queue_evidence(project, prepared, queue)
    except ValueError:
        return
    assert evidence is None


@pytest.mark.parametrize("component", ["audioset_unbalanced", "mls", "fma_full", "yodas2"])
def test_automatic_component_marker_cannot_admit_user_audio(custom_material, component):
    project, prepared, queue = custom_material
    queue.update(id="materials_automatic_fake", automatic_component=component)
    prepared["preparations"][0]["id"] = "automatic_fake"
    prepared["material_acquisition_policy"] = "automatic_v3"
    try:
        evidence = fixed_queue_evidence(project, prepared, queue)
    except ValueError:
        return
    assert evidence is None


def test_policy_marker_without_bound_evidence_is_not_an_admission(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    artifact = {"state": "accepted", "wakeword": "보라야",
                "type": "library_fixed_catalog_technical_validation",
                "policy_id": "library_fixed_sources_v1",
                "content_review": "library_fixed_sources_technical_validation",
                "human_listening": False, "human_listened_count": 0, "human_answers": None}
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, artifact, "noise")


@pytest.fixture
def fixed_zeroth(tmp_path):
    from korean_openwakeword.studio_data import _official_assets

    workspace = tmp_path / "workspace"
    project = create_project(workspace / "projects" / "bora", "보라야", "bora")
    shared = workspace / "shared"
    raw = shared / "zeroth" / "train_data_01" / "001" / "001_0001.flac"
    raw.parent.mkdir(parents=True)
    sf.write(raw, np.random.default_rng(13).normal(0, .05, 48000), 16000, subtype="PCM_16")
    transcript = raw.with_name("001.trans.txt")
    transcript.write_text("001_0001 오늘 날씨가 좋습니다\n", encoding="utf-8")
    stage = project.parent / ".studio" / "materials"
    stage.mkdir(parents=True)
    index = stage / "zeroth_source_index.jsonl"
    create_source_index(shared / "zeroth", output=index, dataset="zeroth", dataset_id="zeroth",
                        source_url="https://www.openslr.org/40/", license="CC-BY-4.0")
    preparation = prepare_public_audio(project, input_directory=shared / "zeroth",
        output=stage / "zeroth", source_index=index, validation_fraction=0, target_hours=None)
    pin = next(item for item in _official_assets() if item["id"] == "zeroth")
    archive = shared / "downloads" / pin["filename"]
    # The helper checks the existing pinned download/extraction evidence, not a
    # second full read of a multi-GB archive. No synthetic archive is downloaded.
    write_json(archive.with_name(archive.name + ".receipt.json"),
               {"path": str(archive), "sha256": pin["sha256"], "bytes": pin["size_bytes"],
                "source_url": pin["source_url"]})
    write_json(shared / "zeroth" / "extraction_receipt.json", {
        "request": {"archive": {"path": str(archive), "sha256": pin["sha256"]}},
        "files": [{"path": path.relative_to(shared / "zeroth").as_posix(),
                   "sha256": digest(path), "bytes": path.stat().st_size}
                  for path in (raw, transcript)]})
    pools = ["korean_negative", "noise"]
    queue = {"id": "materials_zeroth", "kind": "content", "pools": pools,
             "queue": preparation["artifacts"]["representative_queue"]["path"],
             "pool_manifest": preparation["artifacts"]["source_manifest"]["path"],
             "preparation": preparation["preparation"]["path"]}
    prepared = {"kind": "studio_material_preparation_v1", "project": str(project),
                "workspace": str(workspace), "review_queues": [queue],
                "preparations": [{"id": "zeroth", "pools": pools,
                                  "preparation": preparation["preparation"]}]}
    return project, prepared, queue, preparation


def _artifact(evidence, preparation):
    return {"state": "accepted", "wakeword": "보라야",
            "type": "library_fixed_catalog_technical_validation",
            "policy_id": "library_fixed_sources_v1",
            "content_review": "library_fixed_sources_technical_validation",
            "human_listening": False, "human_listened_count": 0, "human_answers": None,
            "library_evidence": evidence,
            "source_manifest": preparation["artifacts"]["source_manifest"]}


def test_fixed_zeroth_connects_without_inventing_human_answers(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    artifact = _artifact(evidence, preparation)
    validated = validate_library_policy_evidence(project.parent, artifact, "noise")
    assert validated["scope"] == "train"
    assert set(validated["allowed_pools"]) == {"korean_negative", "noise"}
    assert validated["human_listening"] is False
    assert validated["source_manifest"] == preparation["artifacts"]["source_manifest"]
    assert evidence == fixed_queue_evidence(project, prepared, queue)
    assert not list((project.parent / "review").glob("**/*answers*"))


@pytest.mark.parametrize("role", ["positive", "hard_negative", "rir", "generic_negative", "evaluation", "final"])
def test_fixed_evidence_is_restricted_to_its_allowed_train_pools(fixed_zeroth, role):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), role)


@pytest.mark.parametrize("mutation", [
    {"human_listening": True}, {"human_listened_count": 1},
    {"human_answers": {"path": "made-up-human-review.json", "sha256": "0" * 64}},
    {"state": "accepted_for_evaluation"}, {"wakeword": "다른호출어"},
])
def test_fixed_policy_never_accepts_false_listening_or_another_scope(fixed_zeroth, mutation):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    artifact = _artifact(evidence, preparation)
    artifact.update(mutation)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, artifact, "noise")


def test_changed_library_evidence_is_rejected_by_its_binding(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    path = resolved(project.parent, evidence["path"])
    contents = read_json(path)
    contents["allowed_pools"].append("rir")
    write_json(path, contents)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), "noise")


def test_changed_preparation_is_rejected_after_evidence_publication(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    path = Path(preparation["preparation"]["path"])
    contents = read_json(path)
    contents["technical_passed_files"] = 999
    write_json(path, contents)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), "noise")


def test_source_manifest_binding_cannot_be_swapped(fixed_zeroth, tmp_path):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    artifact = _artifact(evidence, preparation)
    replacement = tmp_path / "unrelated.jsonl"
    _rows(replacement, [{"path": "user-audio.wav", "sha256": "0" * 64}])
    artifact["source_manifest"] = _bound(replacement)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, artifact, "noise")


def test_catalog_pin_change_invalidates_previously_generated_evidence(fixed_zeroth, monkeypatch):
    from korean_openwakeword import studio_data

    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    changed_assets = copy.deepcopy(studio_data._official_assets())
    next(item for item in changed_assets if item["id"] == "zeroth")["sha256"] = "0" * 64
    monkeypatch.setattr(studio_data, "_official_assets", lambda: changed_assets)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), "noise")


def test_original_extraction_hash_is_bound_to_the_prepared_sources(fixed_zeroth):
    project, prepared, queue, _ = fixed_zeroth
    receipt = Path(prepared["workspace"]) / "shared" / "zeroth" / "extraction_receipt.json"
    contents = read_json(receipt)
    contents["files"][0]["sha256"] = "0" * 64
    write_json(receipt, contents)
    with pytest.raises(ValueError):
        fixed_queue_evidence(project, prepared, queue)


def test_shared_download_receipt_refresh_preserves_existing_evidence(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    receipt_path = next((Path(prepared["workspace"]) / "shared" / "downloads").glob("*.receipt.json"))
    receipt = read_json(receipt_path)
    receipt.update(completed_at="2026-10-21T01:00:00+00:00", reused=False)
    write_json(receipt_path, receipt)
    evidence = fixed_queue_evidence(project, prepared, queue)
    receipt.update(completed_at="2026-10-22T01:00:00+00:00", reused=True)
    write_json(receipt_path, receipt)
    assert validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), "noise")["scope"] == "train"
    assert fixed_queue_evidence(project, prepared, queue) == evidence


@pytest.mark.parametrize("field", ["sha256", "bytes", "source_url"])
def test_shared_download_receipt_immutable_fields_cannot_change(fixed_zeroth, field):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    receipt_path = next((Path(prepared["workspace"]) / "shared" / "downloads").glob("*.receipt.json"))
    receipt = read_json(receipt_path)
    receipt[field] = {"sha256": "0" * 64, "bytes": receipt["bytes"] + 1,
                      "source_url": "https://example.org/unpinned-source"}[field]
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), "noise")
    with pytest.raises(ValueError):
        fixed_queue_evidence(project, prepared, queue)


def test_existing_preparation_uses_its_recorded_storage_workspace(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    workspace = prepared.pop("workspace")
    write_json(project.parent / ".studio" / "workflow.json", {
        "project": str(project), "storage_workspace": workspace,
    })
    evidence = fixed_queue_evidence(project, prepared, queue)
    assert validate_library_policy_evidence(project.parent, _artifact(evidence, preparation), "noise")["scope"] == "train"


def test_current_evaluation_catalog_id_alone_cannot_admit_user_audio(custom_material):
    from korean_openwakeword.evaluation_materials import evaluation_material_catalog
    from korean_openwakeword.evaluation_rir import common_rir_catalog

    project, prepared, custom_queue = custom_material
    catalog_id = identity([evaluation_material_catalog(), common_rir_catalog(),
                           {"evaluation_source_scope": "common_30h_only"}])[:16]
    output = project.parent / ".studio" / "materials" / ("evaluation_v6_" + catalog_id)
    sources, queue_file = output / "evaluation_sources.jsonl", output / "representatives.jsonl"
    source = json.loads(Path(custom_queue["pool_manifest"]).read_text(encoding="utf-8").splitlines()[0])
    source.update(split="validation", evaluation_only=True, training_admitted=False,
                  eval_bucket="environment", evaluation_source_scope="common_30h_only",
                  evaluation_material_catalog_id=identity(evaluation_material_catalog())[:16])
    _rows(sources, [source])
    _rows(queue_file, [])
    receipt_path = output / "preparation.json"
    write_json(receipt_path, {"kind": "studio_evaluation_noise_preparation_v6", "recipe_version": 6,
        "catalog_id": catalog_id, "wakeword": "보라야", "state": "awaiting_human_review",
        "evaluation_only": True, "training_admitted": False, "intended_use": "held_out_development_noise",
        "evaluation_source_scope": "common_30h_only", "Final_accessed": False,
        "artifacts": {"source_manifest": _bound(sources), "representative_queue": _bound(queue_file)}})
    queue = {"id": "materials_evaluation_v6_" + catalog_id, "kind": "content",
             "evaluation_only": True, "intended_use": "held_out_development_noise", "pools": [],
             "queue": str(queue_file), "pool_manifest": str(sources), "preparation": str(receipt_path)}
    prepared.update(recipe_version=6, evaluation_source_scope="common_30h_only",
                    evaluation_preparation=_bound(receipt_path), review_queues=[queue])
    try:
        evidence = fixed_queue_evidence(project, prepared, queue)
    except ValueError:
        return
    assert evidence is None
