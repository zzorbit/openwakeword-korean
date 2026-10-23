"""Strict provenance and byte preservation for explicit mixed CPU/CUDA runs."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, write_json
from korean_openwakeword.evaluate import evaluation_binding, validate_cpu_prefix_import


def _bound(path):
    return {"path": str(path), "sha256": digest(path)}


def _line(row):
    return (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


@pytest.fixture
def imported(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    project = root / "project.json"
    write_json(project, {"wakeword": "보라야", "frozen": True})
    bundle = root / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"schema_version": 1, "wakeword": "보라야", "aggregation": "single",
        "members": [{"path": "models/member.onnx", "sha256": "1" * 64}]})
    sources = []
    for index in range(2):
        path = root / f"audio{index}.wav"
        sf.write(path, np.full(32 * 1280, .05 * (index + 1)), 16000, subtype="PCM_16")
        sources.append({"sample_id": f"session_{index}", "kind": "positive", "category": "clean", "split": "validation",
            "source_id": f"source_{index}", "path": str(path), "sha256": digest(path),
            "calls": [{"call_id": f"call_{index}", "start_s": 2.4, "end_s": 2.5}]})
    manifest = root / "evaluation.jsonl"
    manifest.write_bytes(b"".join(_line(row) for row in sources))
    cpu_output, output = root / "reports/cpu", root / "reports/gpu"
    cpu_work, target_work = cpu_output / "_evaluation_work", output / "_evaluation_work"
    cpu_work.mkdir(parents=True)
    target_work.mkdir(parents=True)
    scores = np.arange(64, dtype=np.float32).reshape(-1, 1) / 1000
    np.save(cpu_work / "frames.npy", scores, allow_pickle=False)
    scored_rows = [{**row, "duration_s": 2.56,
                   "results": {"member": {"success": False, "max_score": .031 + index * .032}}}
                  for index, row in enumerate(sources)]
    prefix_raw = _line(scored_rows[0])
    (cpu_work / "clips.jsonl").write_bytes(prefix_raw + _line(scored_rows[1]))
    old_binding = evaluation_binding(project, bundle, manifest)
    canonical = {"source_binding": old_binding, "committed_sessions": 1,
        "frame_prefix_sha256": hashlib.sha256(scores[:32].tobytes()).hexdigest(),
        "journal_prefix_sha256": hashlib.sha256(prefix_raw).hexdigest()}
    checkpoint = {"schema_version": 1, "binding": old_binding, "shape": [64, 1], "committed_sessions": 1,
        "journal_bytes": len(prefix_raw), "frame_prefix_sha256": canonical["frame_prefix_sha256"],
        "journal_prefix_sha256": canonical["journal_prefix_sha256"]}
    write_json(cpu_work / "checkpoint.json", checkpoint)
    new_binding = evaluation_binding(project, bundle, manifest, backend="cuda", gpu_batch_size=128,
                                    boundary_margin=.001, cpu_prefix=canonical)
    shutil.copyfile(cpu_work / "frames.npy", target_work / "frames.npy")
    (target_work / "clips.jsonl").write_bytes(prefix_raw)
    write_json(target_work / "checkpoint.json", {**checkpoint, "binding": new_binding})
    old_code = {"__init__.py": b'__version__ = "old_cpu_fixture"\n',
                "evaluate.py": b"# Fixture producer: journal created after completed preflight.\n"}
    implementation = identity({name: hashlib.sha256(value).hexdigest() for name, value in old_code.items()})
    job = root / ".studio/jobs/old_task"
    job.mkdir(parents=True)
    request = {"task_id": "old_task", "stage": "evaluation", "project": str(project), "config_sha256": digest(project),
        "implementation_id": implementation,
        "context": {"results": {"package": {"result": {"bundle": str(bundle)}},
                                "development": {"result": {"manifest": str(manifest)}}}}}
    write_json(job / "request.json", request)
    write_json(job / "owner.json", {"task_id": "old_task", "request_sha256": digest(job / "request.json"),
                                   "pid": 12345, "creation_identity": "fixture"})
    backup = root / "backup"
    backup.mkdir()
    full_frames = backup / "frames_full.npy"
    shutil.copyfile(cpu_work / "frames.npy", full_frames)
    archive_path = backup / "cpu_prefix.zip"
    buffer = io.BytesIO()
    np.save(buffer, scores[:32], allow_pickle=False)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("checkpoint.json", json.dumps(checkpoint))
        archive.writestr("clips_prefix.jsonl", prefix_raw)
        archive.writestr("frames_prefix.npy", buffer.getvalue())
        for name, value in old_code.items():
            archive.writestr("code_before/" + name, value)
    receipt = {"schema_version": 1, "kind": "explicit_cpu_score_prefix_import_v1", "prefix": canonical,
        "project": str(project), "project_config_sha256": digest(project),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"), "evaluation_manifest": _bound(manifest),
        "source_evaluation_directory": str(cpu_output), "source_work_directory": str(cpu_work),
        "source_checkpoint": _bound(cpu_work / "checkpoint.json"), "source_frames": _bound(cpu_work / "frames.npy"),
        "source_journal": _bound(cpu_work / "clips.jsonl"), "source_request": _bound(job / "request.json"),
        "source_owner": _bound(job / "owner.json"), "source_implementation_id": implementation,
        "source_scoring_protocol": "reference_cpu_session_v1", "target_evaluation_directory": str(output),
        "target_backend": "cuda", "target_binding": new_binding, "gpu_batch_size": 128, "boundary_margin": .001,
        "user_approved_numerical_differences": True, "cpu_prefix_backup": _bound(archive_path),
        "full_score_backup": _bound(full_frames),
        "old_preflight_completion_proof": {"source_checkpoint_binding": old_binding,
            "source_evaluate_module_sha256": hashlib.sha256(old_code["evaluate.py"]).hexdigest(),
            "source_request": _bound(job / "request.json")},
        "cloned_checkpoint_at_import": _bound(target_work / "checkpoint.json"),
        "cloned_frames_at_import": _bound(target_work / "frames.npy"),
        "cloned_journal_at_import": _bound(target_work / "clips.jsonl")}
    import_path = output / "cpu_prefix_import.json"
    write_json(import_path, receipt)
    return {"project": project, "bundle": bundle, "manifest": manifest, "output": output, "import_path": import_path,
            "receipt": receipt, "sources": sources, "cpu_work": cpu_work, "target_work": target_work,
            "scores": scores, "scored_rows": scored_rows}


def _validate(fixture, **overrides):
    arguments = {"project": fixture["project"], "bundle": fixture["bundle"], "manifest": fixture["manifest"],
        "output": fixture["output"], "stage": "development", "candidate": None, "smoke": False, "backend": "cuda",
        "gpu_batch_size": 128, "boundary_margin": .001, "import_path": fixture["import_path"], "sources": fixture["sources"]}
    arguments.update(overrides)
    return validate_cpu_prefix_import(**arguments)


def test_valid_cpu_prefix_import_is_exact_and_read_only(imported):
    protected = {path: digest(path) for path in imported["project"].parent.rglob("*") if path.is_file()}
    assert _validate(imported) == imported["receipt"]
    assert {path: digest(path) for path in protected} == protected
    assert _validate(imported, sources=None)["prefix"] == imported["receipt"]["prefix"]


@pytest.mark.parametrize("name", ["frames.npy", "clips.jsonl"])
def test_changed_original_cpu_work_is_rejected_even_outside_committed_prefix(imported, name):
    path = imported["cpu_work"] / name
    path.write_bytes(path.read_bytes() + b"changed tail")
    with pytest.raises(ValueError):
        _validate(imported)


def test_changed_target_cpu_score_bit_is_rejected(imported):
    values = np.load(imported["target_work"] / "frames.npy", mmap_mode="r+")
    values[0, 0] = np.nextafter(values[0, 0], np.float32(1))
    values.flush()
    values._mmap.close()
    with pytest.raises(ValueError, match="float32 score bits changed"):
        _validate(imported)


def test_changed_target_cpu_clip_metadata_is_rejected(imported):
    path = imported["target_work"] / "clips.jsonl"
    row = json.loads(path.read_bytes())
    row["sample_id"] = "wrong-session"
    path.write_bytes(_line(row))
    with pytest.raises(ValueError, match="clip metadata"):
        _validate(imported)


def test_imported_prefix_must_match_manifest_order(imported):
    with pytest.raises(ValueError, match="input order"):
        _validate(imported, sources=list(reversed(imported["sources"])))


def test_gpu_progress_can_change_target_tail_and_checkpoint_after_import(imported):
    work = imported["target_work"]
    values = np.load(work / "frames.npy", mmap_mode="r+")
    values[32:] = .987
    values.flush()
    frames_hash = hashlib.sha256(values.tobytes()).hexdigest()
    values._mmap.close()
    gpu_row = {**imported["scored_rows"][1], "results": {"member": {"success": True, "max_score": .987}}}
    with (work / "clips.jsonl").open("ab") as stream:
        stream.write(_line(gpu_row))
    current = read_json(work / "checkpoint.json")
    current.update(committed_sessions=2, journal_bytes=(work / "clips.jsonl").stat().st_size,
                   frame_prefix_sha256=frames_hash, journal_prefix_sha256=digest(work / "clips.jsonl"))
    write_json(work / "checkpoint.json", current)
    for field in ("cloned_frames_at_import", "cloned_journal_at_import", "cloned_checkpoint_at_import"):
        assert digest(imported["receipt"][field]["path"]) != imported["receipt"][field]["sha256"]
    assert _validate(imported)["prefix"]["committed_sessions"] == 1


@pytest.mark.parametrize("field,value", [("user_approved_numerical_differences", False), ("gpu_batch_size", 64),
                                        ("boundary_margin", .002), ("source_scoring_protocol", "other_protocol")])
def test_unapproved_or_changed_policy_is_rejected(imported, field, value):
    receipt = read_json(imported["import_path"])
    receipt[field] = value
    write_json(imported["import_path"], receipt)
    with pytest.raises(ValueError):
        _validate(imported)


@pytest.mark.parametrize("override", [{"backend": "cpu"}, {"stage": "final"}, {"smoke": True}])
def test_import_cannot_bypass_other_evaluation_modes(imported, override):
    with pytest.raises(ValueError, match="Development CUDA"):
        _validate(imported, **override)


def test_prefix_identity_rejects_extra_fields(imported):
    receipt = read_json(imported["import_path"])
    receipt["prefix"]["untrusted_extra"] = True
    write_json(imported["import_path"], receipt)
    with pytest.raises(ValueError, match="exactly four"):
        _validate(imported)


def test_archived_producer_code_must_match_bound_cpu_implementation(imported):
    receipt = read_json(imported["import_path"])
    archive_path = Path(receipt["cpu_prefix_backup"]["path"])
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["code_before/evaluate.py"] += b"# altered producer\n"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    receipt["cpu_prefix_backup"] = _bound(archive_path)
    write_json(imported["import_path"], receipt)
    with pytest.raises(ValueError, match="archived CPU implementation"):
        _validate(imported)
