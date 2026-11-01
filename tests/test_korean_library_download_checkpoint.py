"""Explicit web download checkpoints never replace computational/review evidence."""
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json
from test_korean_library_studio import create, manager  # noqa: F401


DOWNLOAD_STAGES = ("frontend", "data_download", "data_review", "data_connect")


def finish_stages(manager, project_id, count):
    manager.start(project_id)
    for stage in DOWNLOAD_STAGES[:count]:
        worker = manager.workers.items[-1]
        assert read_json(worker.request_path)["stage"] == stage
        worker.finish()
        manager.tick()
    return manager._load(project_id)


def test_new_web_model_waits_for_explicit_download_start_without_changing_computational_settings(manager):
    project = create(manager, download_checkpoint=True)
    state = manager._load(project["id"])
    assert project["download_checkpoint"] == {"policy": "explicit_download_v1", "state": "awaiting_start"}
    assert state["status"] == "ready" and state["index"] == 0
    assert "download_checkpoint" not in state["settings"]
    assert state["execution_contract"]["settings_sha256"] == identity(state["settings"])
    assert state["execution_contract"]["plan_sha256"] == identity(state["plan"])
    before = Path(state["project"]).read_bytes()
    for _ in range(3):
        manager.tick()
    assert manager.workers.calls == []
    assert manager.start(project["id"])["download_checkpoint"]["state"] == "running"
    assert len(manager.workers.calls) == 1
    assert Path(state["project"]).read_bytes() == before


def test_actual_four_stage_receipts_pause_before_voices_until_the_user_continues(manager):
    project = create(manager, download_checkpoint=True)
    root = Path(project["directory"])
    protected = {str(path): digest(path) for path in (root / "project.json", root / "approval.json", root / "voices.json")}
    state = finish_stages(manager, project["id"], 4)
    assert state["status"] == "paused" and state["index"] == 4 and state["job"] is None
    assert state["auto_continue"] is False
    assert state["download_checkpoint"]["state"] == "awaiting_next"
    assert manager._download_group_complete(state) is True
    assert [row["stage"] for row in state["history"]] == list(DOWNLOAD_STAGES)
    for _ in range(3):
        manager.tick()
    assert len(manager.workers.calls) == 4
    resumed = manager.start(project["id"])
    assert resumed["download_checkpoint"]["state"] == "released" and resumed["status"] == "running"
    assert read_json(manager.workers.items[-1].request_path)["stage"] == "references"
    assert len(manager.workers.calls) == 5
    assert {path: digest(path) for path in protected} == protected


def test_old_models_keep_automatic_flow_and_are_not_opted_in_by_recreation(manager):
    project = create(manager)
    path = manager._path(project["id"])
    before = path.read_bytes()
    reused = create(manager, download_checkpoint=True)
    assert reused["id"] == project["id"] and "download_checkpoint" not in reused
    assert path.read_bytes() == before
    state = finish_stages(manager, project["id"], 4)
    assert state["status"] == "running" and "download_checkpoint" not in state
    assert read_json(manager.workers.items[-1].request_path)["stage"] == "references"


@pytest.mark.parametrize("gate,count", [("waiting_review", 2), ("waiting_input", 3)])
def test_download_review_and_shortage_gates_cannot_be_replaced_by_a_next_checkpoint(manager, gate, count):
    project = create(manager, download_checkpoint=True)
    finish_stages(manager, project["id"], count)
    state = manager._load(project["id"])
    output = {"state": gate}
    if gate == "waiting_review":
        queue = Path(state["project"]).parent / ".studio/material-review.jsonl"
        queue.write_text("", encoding="utf-8")
        output["reviews"] = [{"id": "materials", "kind": "content", "queue": str(queue), "queue_sha256": digest(queue)}]
    else:
        output.update(input_kind="material_admissions", shortages=["required audio is missing"])
    manager.workers.items[-1].finish(output)
    manager.tick()
    state = manager._load(project["id"])
    calls = len(manager.workers.calls)
    assert state["status"] == gate and state["download_checkpoint"]["state"] == "running"
    assert manager._download_group_complete(state) is False
    assert manager.start(project["id"])["status"] == gate
    assert len(manager.workers.calls) == calls


@pytest.mark.parametrize("pretend_results", [False, True])
def test_forged_completed_marker_without_four_real_worker_receipts_is_rejected(manager, pretend_results):
    project = create(manager, download_checkpoint=True)
    state = manager._load(project["id"])
    state.update(status="paused", index=4 if pretend_results else 0)
    state["download_checkpoint"]["state"] = "awaiting_next"
    if pretend_results:
        state["results"] = {stage: {"state": "completed"} for stage in DOWNLOAD_STAGES}
    manager._save(state)
    before = manager._path(project["id"]).read_bytes()
    with pytest.raises(ValueError, match="모두 완료"):
        manager.start(project["id"])
    assert manager.workers.calls == []
    assert manager._path(project["id"]).read_bytes() == before


def test_checkpoint_survives_recovery_when_final_download_worker_completed_while_server_was_closed(manager):
    project = create(manager, download_checkpoint=True)
    finish_stages(manager, project["id"], 3)
    manager.workers.items[-1].finish()
    recovered = studio.StudioManager(manager.workspace, popen=manager.workers)
    try:
        state = recovered._load(project["id"])
        assert state["status"] == "paused" and state["download_checkpoint"]["state"] == "awaiting_next"
        recovered.tick()
        assert len(manager.workers.calls) == 4
        before = recovered._path(project["id"]).read_bytes()
        recovered.snapshot()
        assert recovered._path(project["id"]).read_bytes() == before
    finally:
        recovered.close()


def test_changed_completed_download_receipt_cannot_release_the_checkpoint(manager):
    project = create(manager, download_checkpoint=True)
    state = finish_stages(manager, project["id"], 4)
    Path(state["history"][-1]["result"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="기록이 변경"):
        manager.start(project["id"])
    assert len(manager.workers.calls) == 4


@pytest.mark.parametrize("value", [1, "true", None, {}])
def test_checkpoint_creation_flag_requires_an_actual_boolean(manager, value):
    with pytest.raises(ValueError, match="참 또는 거짓"):
        create(manager, download_checkpoint=value)
    assert manager.registry["projects"] == {}
