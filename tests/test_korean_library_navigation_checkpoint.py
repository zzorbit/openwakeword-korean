"""Opt-in story navigation pauses real work, never computational/review gates."""
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, read_json, write_json
from test_korean_library_studio import create, manager  # noqa: F401


GROUPS = (
    ("downloads", ("frontend", "data_download", "data_review", "data_connect")),
    ("voices", ("references", "references_review")),
    ("pronunciation", ("pilot", "pilot_review")), ("positive-generation", ("positives",)),
    ("phrases", ("hard_phrases",)), ("hard-generation", ("hard_generation",)),
    ("hard-review", ("hard_review", "hard_connect")), ("features", ("features",)),
    ("training", ("training", "package")), ("evaluation", ("development", "evaluation")),
    ("results", ("results",)),
)


def finish_group(manager, project_id, group, *, finish_last=True):
    view = manager.start(project_id)
    assert view["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": group[0], "state": "running"}
    for number, stage in enumerate(group[1]):
        worker = manager.workers.items[-1]
        assert read_json(worker.request_path)["stage"] == stage
        if number == len(group[1]) - 1 and not finish_last:
            break
        worker.finish()
        manager.tick()
    return manager._load(project_id)


def test_each_story_group_holds_for_next_and_keeps_computation_and_approval_unchanged(manager):
    project = create(manager, navigation_checkpoints=True)
    state = manager._load(project["id"])
    assert project["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "downloads", "state": "awaiting_start"}
    assert "download_checkpoint" not in state and "navigation_checkpoint" not in state["settings"]
    root = Path(state["project"]).parent
    protected = {str(path): digest(path) for path in (root / "project.json", root / "voices.json", root / "approval.json")}
    contract = state["execution_contract"]
    manager.tick()
    assert manager.workers.calls == []
    finished = 0
    for group in GROUPS:
        state = finish_group(manager, project["id"], group)
        finished += len(group[1])
        assert state["index"] == finished
        assert state["navigation_checkpoint"]["step"] == group[0]
        assert state["execution_contract"] == contract
        assert len(manager.workers.calls) == finished
        if group[0] == "results":
            assert state["status"] == "completed" and state["navigation_checkpoint"]["state"] == "completed"
        else:
            assert state["status"] == "paused" and state["navigation_checkpoint"]["state"] == "awaiting_next"
            assert state["auto_continue"] is False
        before = manager._path(project["id"]).read_bytes()
        manager.tick(); manager.snapshot(); manager.tick()
        assert manager._path(project["id"]).read_bytes() == before
        assert len(manager.workers.calls) == finished
    assert manager.start(project["id"])["status"] == "completed"
    assert len(manager.workers.calls) == 19
    assert {path: digest(path) for path in protected} == protected
    for worker in manager.workers.items:
        request = read_json(worker.request_path)
        assert "navigation_checkpoint" not in request and "navigation_checkpoint" not in request["context"]


def test_pausing_inside_voice_preparation_resumes_same_group_then_waits_before_pilot(manager):
    project = create(manager, navigation_checkpoints=True)
    finish_group(manager, project["id"], GROUPS[0])
    manager.start(project["id"])
    manager.pause(project["id"])
    manager.workers.items[-1].finish()
    manager.tick()
    state = manager._load(project["id"])
    assert state["index"] == 5 and state["status"] == "paused"
    assert state["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "voices", "state": "running"}
    manager.start(project["id"])
    assert read_json(manager.workers.items[-1].request_path)["stage"] == "references_review"
    manager.workers.items[-1].finish(); manager.tick()
    state = manager._load(project["id"])
    assert state["index"] == 6 and state["navigation_checkpoint"]["state"] == "awaiting_next"
    assert len(manager.workers.calls) == 6


def test_failed_voice_generation_retries_before_the_completed_voice_checkpoint(manager):
    project = create(manager, navigation_checkpoints=True)
    finish_group(manager, project["id"], GROUPS[0])
    manager.start(project["id"])
    manager.workers.items[-1].finish(error={"message": "temporary TTS failure"}); manager.tick()
    state = manager._load(project["id"])
    assert state["status"] == "error" and state["index"] == 4 and state["navigation_checkpoint"]["state"] == "running"
    manager.start(project["id"])
    assert read_json(manager.workers.items[-1].request_path)["stage"] == "references"
    manager.workers.items[-1].finish(); manager.tick()
    manager.workers.items[-1].finish(); manager.tick()
    state = manager._load(project["id"])
    assert state["status"] == "paused" and state["index"] == 6 and state["navigation_checkpoint"]["state"] == "awaiting_next"


def test_voice_completion_while_server_closed_recovers_without_starting_pilot(manager):
    project = create(manager, navigation_checkpoints=True)
    finish_group(manager, project["id"], GROUPS[0])
    finish_group(manager, project["id"], GROUPS[1], finish_last=False)
    manager.workers.items[-1].finish()
    recovered = studio.StudioManager(manager.workspace, popen=manager.workers)
    try:
        state = recovered._load(project["id"])
        assert state["index"] == 6 and state["status"] == "paused"
        assert state["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "voices", "state": "awaiting_next"}
        recovered.tick(); recovered.snapshot()
        assert len(manager.workers.calls) == 6
    finally:
        recovered.close()


@pytest.mark.parametrize("step,gate", [("downloads", "waiting_input"), ("pronunciation", "waiting_review"), ("phrases", "waiting_input")])
def test_story_next_never_bypasses_review_input_or_shortage_gates(manager, step, gate):
    project = create(manager, navigation_checkpoints=True)
    for group in GROUPS:
        if group[0] == step:
            finish_group(manager, project["id"], group, finish_last=False)
            break
        finish_group(manager, project["id"], group)
    state = manager._load(project["id"])
    output = {"state": gate}
    if gate == "waiting_review":
        queue = Path(state["project"]).parent / ".studio/gated-review.jsonl"
        queue.write_text("", encoding="utf-8")
        output["reviews"] = [{"id": "pilot", "kind": "pilot", "queue": str(queue), "queue_sha256": digest(queue)}]
    else:
        output.update(input_kind="material_admissions" if step == "downloads" else "hard_phrases", shortages=["required input"])
    manager.workers.items[-1].finish(output); manager.tick()
    calls = len(manager.workers.calls)
    view = manager.start(project["id"])
    assert view["status"] == gate and view["navigation_checkpoint"]["state"] == "running"
    assert len(manager.workers.calls) == calls


def test_a_display_only_completed_marker_without_bound_receipts_cannot_start_pilot(manager):
    project = create(manager, navigation_checkpoints=True)
    state = manager._load(project["id"])
    state.update(index=6, status="paused")
    state["navigation_checkpoint"].update(step="voices", state="awaiting_next")
    state["results"] = {row["id"]: {"state": "completed"} for row in state["plan"][:6]}
    manager._save(state)
    with pytest.raises(ValueError, match="모두 완료"):
        manager.start(project["id"])
    assert manager.workers.calls == []


def test_config_and_receipt_tampering_cannot_release_story_navigation(manager):
    project = create(manager, navigation_checkpoints=True)
    state = finish_group(manager, project["id"], GROUPS[0])
    config_path = Path(state["project"])
    original = config_path.read_bytes()
    config = read_json(config_path); config["wakeword"] = "다른아"; write_json(config_path, config)
    with pytest.raises(ValueError):
        manager.start(project["id"])
    config_path.write_bytes(original)
    Path(state["history"][-1]["result"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="기록이 변경"):
        manager.start(project["id"])
    assert len(manager.workers.calls) == 4


def test_existing_records_and_download_v1_are_not_converted_to_the_new_policy(manager):
    old = create(manager)
    before = manager._path(old["id"]).read_bytes()
    same = create(manager, navigation_checkpoints=True)
    assert same["id"] == old["id"] and "navigation_checkpoint" not in same
    assert manager._path(old["id"]).read_bytes() == before
    v1 = create(manager, wakeword="다음아", download_checkpoint=True)
    before = manager._path(v1["id"]).read_bytes()
    same = create(manager, wakeword="다음아", navigation_checkpoints=True)
    assert "navigation_checkpoint" not in same and same["download_checkpoint"]["policy"] == "explicit_download_v1"
    assert manager._path(v1["id"]).read_bytes() == before


@pytest.mark.parametrize("payload", [{"navigation_checkpoints": 1}, {"navigation_checkpoints": "true"}, {"navigation_checkpoints": None}, {"navigation_checkpoints": True, "download_checkpoint": True}])
def test_navigation_creation_requires_an_unambiguous_boolean_opt_in(manager, payload):
    with pytest.raises(ValueError):
        create(manager, **payload)
    assert manager.registry["projects"] == {}
