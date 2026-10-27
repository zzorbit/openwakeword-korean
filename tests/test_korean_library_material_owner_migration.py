"""Public-material provenance fix migrates only its audited release pair."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json, write_json
from test_korean_library_studio import manager, _paused_after_first_stage


def material_owner_pair(manager, monkeypatch):
    view = _paused_after_first_stage(manager)
    state = manager._load(view["id"])
    current = deepcopy(manager._execution_contract)
    previous = deepcopy(state["execution_contract"])
    previous["files"]["material_policy.py"] = "0" * 64
    previous["computation_sha256"] = identity(previous["files"])
    previous["stages"]["features"] = "1" * 64
    error = {"stage": "features", "type": "ValueError",
             "message": "Fixed library material policy: policy evidence/catalog/preparation/source lineage changed"}
    state.update(index=next(i for i, step in enumerate(state["plan"]) if step["id"] == "features"),
                 status="error", error=error, job=None, execution_contract=previous)
    answer = Path(state["project"]).parent / ".studio/answers/preserved-hard.json"
    write_json(answer, {"decisions": {"heard-representative": {"decision": "keep", "listened": True}}})
    state["inputs"]["review_answers"] = {"hard": {"path": str(answer), "sha256": digest(answer)}}
    write_json(manager._path(view["id"]), state)
    manager._implementation_id = "audited-material-owner-release"
    manager._execution_contract = current
    audit = {"source_stages_sha256": identity(previous["stages"]),
             "target_computation_sha256": current["computation_sha256"],
             "target_stages_sha256": identity(current["stages"]),
             "changed_files": ["material_policy.py"],
             "evidence_sha256": identity({"fixed_public_source_owner_validation": "fixture"})}
    monkeypatch.setattr(studio, "_AUDITED_ACQUISITION_UPDATES", {previous["computation_sha256"]: audit})
    return state, previous, current, audit, answer


def test_material_owner_update_preserves_error_review_history_and_existing_files(manager, monkeypatch):
    state, previous, current, audit, answer = material_owner_pair(manager, monkeypatch)
    workflow = manager._path(state["id"])
    before = workflow.read_bytes()
    protected = {path: path.read_bytes() for path in (Path(state["project"]), answer)}
    migration = manager._migration_view(state)
    assert migration["state"] == "available"
    assert migration["audited_acquisition_update"]["changed_files"] == ["material_policy.py"]
    assert "원래 준비 기록" in migration["reason"]
    assert "다운로드 순서" not in migration["reason"]
    assert workflow.read_bytes() == before
    manager.migrate(state["id"])
    after = manager._load(state["id"])
    for key in ("index", "status", "error", "inputs", "history", "settings", "plan", "job"):
        assert after[key] == state[key]
    assert after["job"] is None and len(manager.workers.items) == 1
    assert all(path.read_bytes() == value for path, value in protected.items())
    receipt = read_json(after["migrations"][-1]["path"])
    assert Path(receipt["previous_workflow"]["path"]).read_bytes() == before
    assert digest(after["migrations"][-1]["path"]) == after["migrations"][-1]["sha256"]
    proof = receipt["audited_acquisition_update"]
    assert proof["source_computation_sha256"] == previous["computation_sha256"]
    assert proof["target_computation_sha256"] == current["computation_sha256"]
    assert proof["evidence_sha256"] == audit["evidence_sha256"]


@pytest.mark.parametrize("tamper", ["source_files", "source_stages", "target_files", "target_stages", "proof"])
def test_material_owner_update_requires_exact_release_and_proof(manager, monkeypatch, tamper):
    state, previous, current, audit, _ = material_owner_pair(manager, monkeypatch)
    if tamper == "source_files":
        previous["files"]["audio.py"] = "2" * 64
    elif tamper == "source_stages":
        previous["stages"]["training"] = "2" * 64
    elif tamper == "target_files":
        current["files"]["train.py"] = "2" * 64
        current["computation_sha256"] = identity(current["files"])
    elif tamper == "target_stages":
        current["stages"]["evaluation"] = "2" * 64
    else:
        audit["evidence_sha256"] = "unfinished"
    write_json(manager._path(state["id"]), state)
    before = manager._path(state["id"]).read_bytes()
    assert manager._migration_view(state)["state"] == "blocked"
    with pytest.raises(ValueError):
        manager.migrate(state["id"])
    assert manager._path(state["id"]).read_bytes() == before


@pytest.mark.parametrize("extra", ["features.py", "train.py", "resources/public_datasets.json"])
def test_material_owner_audit_cannot_admit_extra_computation_or_catalog_change(manager, monkeypatch, extra):
    state, _, current, audit, _ = material_owner_pair(manager, monkeypatch)
    current["files"][extra] = "3" * 64
    current["computation_sha256"] = identity(current["files"])
    audit["target_computation_sha256"] = current["computation_sha256"]
    audit["changed_files"].append(extra)
    assert manager._migration_view(state)["state"] == "blocked"


def test_unpinned_material_owner_release_cannot_authorize_migration(manager, monkeypatch):
    state, previous, _, _, _ = material_owner_pair(manager, monkeypatch)
    monkeypatch.setattr(studio, "_AUDITED_ACQUISITION_UPDATES", {previous["computation_sha256"]: {}})
    assert manager._migration_view(state)["state"] == "blocked"
