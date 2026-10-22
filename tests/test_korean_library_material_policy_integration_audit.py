"""Exercise real fixed-source evidence through Train admission consumers."""
from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json
from korean_openwakeword.data import validate_admission
from korean_openwakeword.material_policy import fixed_queue_evidence
from korean_openwakeword.negative_streams import pack_negative_streams
from korean_openwakeword.preparation import apply_prepared_review
from korean_openwakeword.project import create_project, load_project
from test_korean_library_material_policy import fixed_zeroth


def test_real_fixed_source_evidence_admits_connects_and_resumes_without_human_answers(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    root = project.parent
    # A preserved legacy artifact must never be repurposed as technical evidence.
    legacy = Path(preparation["preparation"]["path"]).parent / "admitted/evidence/admission.json"
    write_json(legacy, {"human_answers": {"path": "old-listening.json", "sha256": "f" * 64}})
    legacy_sha = digest(legacy)
    evidence = fixed_queue_evidence(project, prepared, queue)
    result = apply_prepared_review(project, preparation=preparation["preparation"],
        pools=queue["pools"], library_evidence=evidence, connect=True)
    assert result["connected"] is True
    artifact = read_json(root / result["admission"]["path"])
    assert artifact["human_answers"] is None and artifact["human_listening"] is False
    assert artifact["human_listened_count"] == 0
    assert artifact["source_manifest"] == preparation["artifacts"]["source_manifest"]
    assert digest(legacy) == legacy_sha
    _, config = load_project(project)
    for role in queue["pools"]:
        rows = read_rows(root / config["manifests"][role])
        assert rows and all(r["individual_human_listening"] is False for r in rows)
        validate_admission(root, iter(rows), config, role)
    resumed = apply_prepared_review(project, preparation=preparation["preparation"],
        pools=queue["pools"], library_evidence=evidence, connect=True)
    assert resumed["resumed"] is True and resumed["connection"]["resumed"] is True
    assert digest(legacy) == legacy_sha


def test_fixed_source_admission_respects_final_lock_before_creating_receipts(fixed_zeroth):
    project, prepared, queue, preparation = fixed_zeroth
    evidence = fixed_queue_evidence(project, prepared, queue)
    write_json(project.parent / "final_lock.json", {"fixture": "locked"})
    with pytest.raises(ValueError, match="Final is frozen"):
        apply_prepared_review(project, preparation=preparation["preparation"],
            pools=queue["pools"], library_evidence=evidence, connect=True)
    assert not (Path(preparation["preparation"]["path"]).parent / "admitted_library_fixed_sources_v1").exists()


def test_real_evaluation_policy_admission_reaches_stream_packing_without_listening(tmp_path, monkeypatch):
    from korean_openwakeword import evaluation_materials, evaluation_rir, studio_evaluation_data

    def binding(path):
        return {"path": str(path), "sha256": digest(path)}

    workspace = tmp_path / "workspace"
    project = create_project(workspace / "projects/bora", "보라야", "bora")
    root = project.parent
    archive = {"id": "fixture", "filename": "fixture.tar", "size_bytes": 32,
        "source_url": "https://example.test/pinned-fixture.tar", "sha256": "a" * 64,
        "checksum_algorithm": "sha256"}
    pin = {"canonical_source_id": "fixture:environment", "canonical_recording_id": "fixture:recording",
        "dataset_id": "fixture", "source_url": "https://example.test/environment", "license": "fixture",
        "category": "environment", "eval_bucket": "environment", "eval_type": "fixture_environment"}
    catalog = {"selections": {"fixture": [pin]}, "archives": [archive]}
    rir_archive = {**archive, "filename": "rir.tar", "source_url": "https://example.test/pinned-rir.tar"}
    rir_catalog = {"assets": [], "archive": rir_archive, "source_url": "https://example.test/rir"}
    monkeypatch.setattr(evaluation_materials, "evaluation_material_catalog", lambda: catalog)
    monkeypatch.setattr(evaluation_rir, "common_rir_catalog", lambda: rir_catalog)
    common = workspace / "shared" / ("common_evaluation_30h_" + identity(catalog)[:16]) / "selection_fixture"
    audio = common / "audio/environment.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, .1 * np.sin(np.arange(6400) / 23), 16000, subtype="PCM_16")
    pcm, frames = studio_evaluation_data._pcm_hash(audio)
    source = {**pin, "sample_id": "fixture_environment", "path": str(audio), "sha256": digest(audio),
        "original_path": str(audio), "original_sha256": digest(audio), "duration_s": .4,
        "normalized_pcm_sha256": pcm, "normalized_frames": frames, "split": "validation",
        "provenance": "public_dataset", "technical_pass": True, "evaluation_only": True,
        "training_admitted": False, "evaluation_source_scope": "common_30h_only",
        "evaluation_material_catalog_id": identity(catalog)[:16], "individual_human_listening": False}
    write_json(common / "preparation.json", {"kind": "common_evaluation_materials_v2",
        "source_pool_scope": "common_30h_only", "catalog_id": identity(catalog)[:16],
        "target_met": True, "source_rows": [source]})
    catalog_id = identity([catalog, rir_catalog, {"evaluation_source_scope": "common_30h_only"}])[:16]
    stage = root / ".studio/materials" / ("evaluation_v6_" + catalog_id)
    stage.mkdir(parents=True)
    sources, previews = stage / "evaluation_sources.jsonl", stage / "representatives.jsonl"
    sources.write_text(json.dumps(source) + "\n", encoding="utf-8")
    previews.write_text("", encoding="utf-8")
    preparation = stage / "preparation.json"
    write_json(preparation, {"kind": "studio_evaluation_noise_preparation_v6", "recipe_version": 6,
        "catalog_id": catalog_id, "wakeword": "보라야", "evaluation_only": True,
        "training_admitted": False, "intended_use": "held_out_development_noise",
        "evaluation_source_scope": "common_30h_only",
        "artifacts": {"source_manifest": binding(sources), "representative_queue": binding(previews)}})
    write_json(stage / "rir/receipt.json", {"kind": "common_rir_preparation_v1",
        "catalog_sha256": identity(rir_catalog)})
    for asset, directory in ((archive, workspace / "shared/downloads"),
                              (rir_archive, workspace / "shared/downloads/common_rir")):
        target = directory / asset["filename"]
        write_json(target.with_name(target.name + ".receipt.json"), {
            "path": str(target), "sha256": asset["sha256"], "bytes": asset["size_bytes"],
            "source_url": asset["source_url"]})
    queue = {"id": "materials_evaluation_v6_" + catalog_id, "kind": "content", "pools": [],
        "evaluation_only": True, "queue": str(previews), "pool_manifest": str(sources),
        "preparation": str(preparation)}
    prepared = {"kind": "studio_material_preparation_v1", "project": str(project),
        "workspace": str(workspace), "recipe_version": 6, "review_queues": [queue],
        "evaluation_preparation": binding(preparation)}
    evidence = fixed_queue_evidence(project, prepared, queue)
    admitted = studio_evaluation_data.admit_fixed_evaluation(project, queue, prepared, evidence)
    assert all(r["individual_human_listening"] is False and r["representative_human_listening"] is False
               for r in admitted["rows"])
    packed = pack_negative_streams(project, admitted["rows"], root / "reports/fixture_streams", seconds=1)
    assert packed["stream_count"] == 1 and packed["packed_seconds"] == pytest.approx(.4)
    artifact = read_json(admitted["admission"]["path"])
    assert artifact["human_listening"] is False and artifact["human_listened_count"] == 0
    assert artifact["human_answers"] is None
    write_json(root / "final_lock.json", {"fixture": "locked"})
    with pytest.raises(ValueError, match="Final is frozen"):
        studio_evaluation_data.admit_fixed_evaluation(project, queue, prepared, evidence)
