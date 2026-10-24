"""Exact acquisition release updates and owned interrupted-download recovery."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json, write_json
from test_korean_library_studio import create, manager, _paused_after_first_stage


def audited_pair(manager, monkeypatch):
    view = _paused_after_first_stage(manager)
    state = manager._load(view["id"])
    previous = deepcopy(state["execution_contract"])
    previous["files"].pop("material_downloads.py", None)
    previous["computation_sha256"] = identity(previous["files"])
    previous["stages"]["data_download"] = "1" * 64
    previous["stages"]["development"] = "2" * 64
    state["execution_contract"] = previous
    write_json(manager._path(view["id"]), state)
    current = deepcopy(previous)
    current["files"]["downloads.py"] = "3" * 64
    current["files"]["material_downloads.py"] = "4" * 64
    current["computation_sha256"] = identity(current["files"])
    current["stages"]["data_download"] = "5" * 64
    current["stages"]["development"] = "6" * 64
    manager._implementation_id = "audited-acquisition-release"
    manager._execution_contract = current
    audit = {"source_stages_sha256": identity(previous["stages"]),
             "target_computation_sha256": current["computation_sha256"],
             "target_stages_sha256": identity(current["stages"]),
             "changed_files": ["downloads.py", "material_downloads.py"],
             "evidence_sha256": identity({"pinned_source_and_manifest_equivalence": "fixture"})}
    monkeypatch.setattr(studio, "_AUDITED_ACQUISITION_UPDATES", {previous["computation_sha256"]: audit})
    return view, state, previous, current, audit


def test_audited_acquisition_update_is_explicit_and_preserves_downloads_and_history(manager, monkeypatch):
    view, state, previous, current, audit = audited_pair(manager, monkeypatch)
    partial = manager.workspace / "shared/downloads/example.zip.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"preserved incomplete original bytes")
    workflow = manager._path(view["id"])
    before = workflow.read_bytes()
    config = Path(state["project"]).read_bytes()
    migration = manager._migration_view(state)
    assert migration["state"] == "available"
    assert migration["audited_acquisition_update"]["evidence_sha256"] == audit["evidence_sha256"]
    assert workflow.read_bytes() == before
    with pytest.raises(ValueError, match="업데이트 적용 후"):
        manager.start(view["id"])
    assert workflow.read_bytes() == before
    manager.migrate(view["id"])
    after = manager._load(view["id"])
    receipt = read_json(after["migrations"][-1]["path"])
    assert receipt["audited_acquisition_update"]["source_computation_sha256"] == previous["computation_sha256"]
    assert receipt["audited_acquisition_update"]["target_computation_sha256"] == current["computation_sha256"]
    assert digest(after["migrations"][-1]["path"]) == after["migrations"][-1]["sha256"]
    assert Path(receipt["previous_workflow"]["path"]).read_bytes() == before
    assert Path(state["project"]).read_bytes() == config
    assert partial.read_bytes() == b"preserved incomplete original bytes"
    assert after["history"] == state["history"] and after["inputs"] == state["inputs"]
    assert len(manager.workers.items) == 1  # Migration never launches work.
    manager.start(view["id"])
    assert len(manager.workers.items) == 2


@pytest.mark.parametrize("key", ["target_computation_sha256", "target_stages_sha256", "source_stages_sha256", "evidence_sha256"])
@pytest.mark.parametrize("value", [None, "", "unfinished"])
def test_unfinished_or_malformed_audit_never_authorizes_migration(manager, monkeypatch, key, value):
    _, state, _, _, audit = audited_pair(manager, monkeypatch)
    audit[key] = value
    assert manager._migration_view(state)["state"] == "blocked"


@pytest.mark.parametrize("tamper", ["source_files", "source_stages", "target_files", "target_stages", "changed_files", "evidence"])
def test_exact_source_target_and_proof_are_required(manager, monkeypatch, tamper):
    view, state, previous, current, audit = audited_pair(manager, monkeypatch)
    if tamper == "source_files":
        previous["files"]["audio.py"] = "a" * 64
    elif tamper == "source_stages":
        previous["stages"]["training"] = "b" * 64
    elif tamper == "target_files":
        current["files"]["train.py"] = "c" * 64
        current["computation_sha256"] = identity(current["files"])
    elif tamper == "target_stages":
        current["stages"]["training"] = "d" * 64
    elif tamper == "changed_files":
        audit["changed_files"].append("audio.py")
    else:
        audit.pop("evidence_sha256")
    write_json(manager._path(view["id"]), state)
    before = manager._path(view["id"]).read_bytes()
    assert manager._migration_view(state)["state"] == "blocked"
    with pytest.raises(ValueError):
        manager.migrate(view["id"])
    assert manager._path(view["id"]).read_bytes() == before


@pytest.mark.parametrize("name", ["resources/public_datasets.json", "train.py", "features.py", "audio.py"])
def test_even_an_exact_pair_cannot_allow_changed_catalogs_or_computation(manager, monkeypatch, name):
    _, state, _, current, audit = audited_pair(manager, monkeypatch)
    current["files"][name] = "9" * 64
    current["computation_sha256"] = identity(current["files"])
    audit["target_computation_sha256"] = current["computation_sha256"]
    audit["changed_files"].append(name)
    assert manager._migration_view(state)["state"] == "blocked"


@pytest.mark.parametrize("tamper", ["settings", "plan", "project", "history"])
def test_acquisition_audit_cannot_bypass_frozen_input_or_history_checks(manager, monkeypatch, tamper):
    view, state, _, _, _ = audited_pair(manager, monkeypatch)
    if tamper == "settings":
        state["settings"]["updates"] += 1
    elif tamper == "plan":
        state["plan"][1]["title"] = "changed plan"
    elif tamper == "project":
        config = read_json(state["project"])
        config["training"]["updates"] += 1
        write_json(state["project"], config)
    else:
        Path(state["history"][0]["result"]).write_bytes(b"changed history")
    write_json(manager._path(view["id"]), state)
    before = manager._path(view["id"]).read_bytes()
    with pytest.raises(ValueError):
        manager.migrate(view["id"])
    assert manager._path(view["id"]).read_bytes() == before


def test_acquisition_audit_cannot_rebind_a_live_worker(manager, monkeypatch):
    view, _, _, _, _ = audited_pair(manager, monkeypatch)
    state = manager._load(view["id"])
    live = manager.workers.items[-1]
    live.returncode = None
    manager.workers.identities[live.pid] = state["job"]["creation_identity"] if state.get("job") else "birth-" + str(live.pid)
    state["job"] = deepcopy(state["last_job"])
    write_json(manager._path(view["id"]), state)
    before = manager._path(view["id"]).read_bytes()
    assert manager._migration_view(state)["state"] == "blocked"
    with pytest.raises(ValueError, match="현재 작업"):
        manager.migrate(view["id"])
    assert manager._path(view["id"]).read_bytes() == before


def test_nested_common_rir_lock_recovery_preserves_partial_and_archives_owned_lock(manager):
    view = create(manager)
    manager.start(view["id"])
    state = manager._load(view["id"])
    shared = manager.workspace / "shared/downloads/common_rir"
    partial = shared / "rirs.zip.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"resume these bytes")
    lock = shared / ".rirs.zip.download/.writer.lock"
    record = {"pid": state["job"]["pid"], "marker": "owned nested archive lock"}
    write_json(lock, record)
    manager.workers.items[-1].returncode = 9
    manager.workers.identities.clear()
    manager._recover_dead_locks(state)
    assert not lock.exists()
    assert partial.read_bytes() == b"resume these bytes"
    archived = list((Path(state["project"]).parent / ".studio/recovered_locks").glob("*.json"))
    assert any(read_json(path)["record"] == record for path in archived)


@pytest.mark.parametrize("kind", ["different_owner", "recycled_pid"])
def test_nested_common_rir_recovery_never_steals_unknown_or_live_pid(manager, kind):
    view = create(manager)
    manager.start(view["id"])
    state = manager._load(view["id"])
    pid = state["job"]["pid"]
    manager.workers.items[-1].returncode = 9
    manager.workers.identities.clear()
    lock = manager.workspace / "shared/downloads/common_rir/.rirs.zip.download/.writer.lock"
    write_json(lock, {"pid": 77777 if kind == "different_owner" else pid})
    if kind == "recycled_pid":
        manager.workers.identities[pid] = "another-process-birth"
    contents = lock.read_bytes()
    with pytest.raises(ValueError, match="다른 작업|살아|재사용"):
        manager._recover_dead_locks(state)
    assert lock.read_bytes() == contents
