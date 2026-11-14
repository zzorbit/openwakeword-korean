"""Opt-in story navigation pauses real work, never computational/review gates."""
import json
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, read_json, write_json
from test_korean_library_studio import create, http_server, manager  # noqa: F401


GROUPS = (
    ("downloads", ("frontend", "data_download", "data_review", "data_connect")),
    ("voices", ("references", "references_review")),
    ("call-pilot-generation", ("pilot",)), ("call-pilot-review", ("pilot_review",)),
    ("positive-generation", ("positives",)), ("pronunciation", ("positive_review",)),
    ("phrases", ("hard_phrases",)), ("hard-generation", ("hard_generation",)),
    ("hard-review", ("hard_review", "hard_connect")), ("features", ("features",)),
    ("training", ("training", "package")), ("evaluation", ("development", "evaluation", "results")),
)


def finish_report(worker):
    root = Path(read_json(worker.request_path)["project"]).parent
    site = root / "reports" / "studio_results"
    site.mkdir(parents=True, exist_ok=True)
    index = site / "index.html"
    index.write_text("<h1>Prepared test report</h1>", encoding="utf-8")
    (site / "results-data.js").write_text("window.KOWW_RESULTS = {};", encoding="utf-8")
    provenance = site / "provenance.json"
    write_json(provenance, {})
    worker.finish({"state": "completed", "result": {"site": str(index)}, "artifacts": [str(provenance)]},
                  artifact_bindings=[{"path": str(provenance), "sha256": digest(provenance)}])


def finish_group(manager, project_id, group, *, finish_last=True):
    view = manager.start(project_id)
    assert view["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": group[0], "state": "running"}
    for number, stage in enumerate(group[1]):
        worker = manager.workers.items[-1]
        assert read_json(worker.request_path)["stage"] == stage
        if number == len(group[1]) - 1 and not finish_last:
            break
        if stage == "results":
            finish_report(worker)
        else:
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
        assert state["status"] == ("completed" if group[0] == "evaluation" else "paused")
        assert state["navigation_checkpoint"]["state"] == "awaiting_next"
        assert state["auto_continue"] is False
        before = manager._path(project["id"]).read_bytes()
        manager.tick(); manager.snapshot(); manager.tick()
        assert manager._path(project["id"]).read_bytes() == before
        assert len(manager.workers.calls) == finished
    opened = manager.start(project["id"])
    assert opened["status"] == "completed"
    assert opened["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "evaluation", "state": "completed"}
    assert len(manager.workers.calls) == 20
    assert {path: digest(path) for path in protected} == protected
    for worker in manager.workers.items:
        request = read_json(worker.request_path)
        assert "navigation_checkpoint" not in request and "navigation_checkpoint" not in request["context"]


def test_report_preparation_failure_resumes_only_report_and_next_opens_the_finished_site(manager, http_server):
    project_id = create(manager, navigation_checkpoints=True)["id"]
    for group in GROUPS[:-1]:
        finish_group(manager, project_id, group)
    state = finish_group(manager, project_id, GROUPS[-1], finish_last=False)
    evaluation_history = state["history"][-1]
    assert evaluation_history["stage"] == "evaluation"
    assert state["navigation_checkpoint"]["state"] == "running"
    assert state["job"]["stage"] == "results"
    assert "results" not in manager.describe(state)
    assert http_server(f"/results/{project_id}/")[0] == 400

    manager.workers.items[-1].finish(error={"message": "temporary report export failure"})
    manager.tick()
    failed = manager._load(project_id)
    assert failed["status"] == "error" and failed["history"][-2] == evaluation_history
    assert failed["navigation_checkpoint"]["step"] == "evaluation"
    resumed = manager.start(project_id)
    assert resumed["status"] == "running" and resumed["stage_id"] == "results"
    assert read_json(manager.workers.items[-1].request_path)["stage"] == "results"
    finish_report(manager.workers.items[-1])
    manager.tick()

    ready = manager.describe(manager._load(project_id))
    assert ready["status"] == "completed" and ready["navigation_checkpoint"]["state"] == "awaiting_next"
    completed_at = manager._load(project_id)["completed_at"]
    calls = len(manager.workers.calls)
    status, body, _ = http_server(f"/api/projects/{project_id}/start", "POST", {}, headers={"X-Koww-Token": manager.token})
    assert status == 200
    opened = json.loads(body)
    assert opened["navigation_checkpoint"]["state"] == "completed"
    assert opened["results"]["url"] == ready["results"]["url"]
    assert http_server(opened["results"]["url"])[0] == 200
    assert http_server(opened["results"]["url"] + "results-data.js")[0] == 200
    assert manager._load(project_id)["completed_at"] == completed_at
    assert len(manager.workers.calls) == calls


@pytest.mark.parametrize("step,phase", [("evaluation", "awaiting_next"), ("results", "running")])
def test_unfinished_legacy_results_checkpoint_migrates_without_rerunning_evaluation(manager, monkeypatch, step, phase):
    project_id = create(manager, navigation_checkpoints=True)["id"]
    for group in GROUPS[:-1]:
        finish_group(manager, project_id, group)
    finish_group(manager, project_id, ("evaluation", ("development", "evaluation")), finish_last=False)
    manager.pause(project_id)
    manager.workers.items[-1].finish()
    manager.tick()
    old = manager._load(project_id)
    old["navigation_checkpoint"].update(step=step, state=phase)
    manager._save(old)
    monkeypatch.setattr(studio, "_implementation", lambda: "merged-evaluation-report")
    manager._implementation_id = studio._implementation()
    worker_count = len(manager.workers.calls)

    migrated = manager.migrate(project_id)
    assert migrated["status"] == "paused" and migrated["stage_id"] == "results"
    assert migrated["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "evaluation", "state": "running"}
    new = manager._load(project_id)
    assert new["history"] == old["history"] and new["results"] == old["results"]
    assert new["auto_continue"] is False and len(manager.workers.calls) == worker_count
    assert manager.start(project_id)["stage_id"] == "results"
    assert len(manager.workers.calls) == worker_count + 1
    assert read_json(manager.workers.items[-1].request_path)["stage"] == "results"


def test_existing_completed_report_remains_unchanged_and_tampered_new_receipt_cannot_release(manager):
    project_id = create(manager, navigation_checkpoints=True)["id"]
    for group in GROUPS:
        state = finish_group(manager, project_id, group)
    result = Path(state["history"][-1]["result"])
    original = result.read_bytes()
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="기록이 변경"):
        manager.start(project_id)
    result.write_bytes(original)
    site = Path(state["results"]["results"]["result"]["site"]).parent
    boot = site / "results-data.js"
    boot_content = boot.read_bytes()
    boot.unlink()
    with pytest.raises(ValueError, match="결과 파일"):
        manager.start(project_id)
    boot.write_bytes(boot_content)
    provenance = site / "provenance.json"
    provenance_content = provenance.read_bytes()
    write_json(provenance, {"changed": True})
    with pytest.raises(ValueError, match="provenance.json"):
        manager.start(project_id)
    provenance.write_bytes(provenance_content)
    state["navigation_checkpoint"].update(step="results", state="completed")
    manager._save(state)
    before = manager._path(project_id).read_bytes()
    assert manager.migrate(project_id)["navigation_checkpoint"]["step"] == "results"
    assert manager.start(project_id)["navigation_checkpoint"]["state"] == "completed"
    assert manager._path(project_id).read_bytes() == before
    assert len(manager.workers.calls) == len(state["plan"])


def test_recover_completed_legacy_report_waits_for_next_without_new_worker(manager):
    project_id = create(manager, navigation_checkpoints=True)["id"]
    for group in GROUPS[:-1]:
        finish_group(manager, project_id, group)
    old = finish_group(manager, project_id, GROUPS[-1], finish_last=False)
    old["navigation_checkpoint"]["step"] = "results"
    manager._save(old)
    finish_report(manager.workers.items[-1])
    calls = len(manager.workers.calls)

    recovered = studio.StudioManager(manager.workspace, popen=manager.workers)
    try:
        ready = recovered.describe(recovered._load(project_id))
        assert ready["status"] == "completed" and ready["results"]["url"]
        assert ready["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "evaluation", "state": "awaiting_next"}
        assert recovered.start(project_id)["navigation_checkpoint"]["state"] == "completed"
        assert len(manager.workers.calls) == calls
    finally:
        recovered.close()


def test_a_report_display_marker_without_bound_receipts_cannot_open_the_report(manager):
    project_id = create(manager, navigation_checkpoints=True)["id"]
    state = manager._load(project_id)
    state.update(index=len(state["plan"]), status="completed")
    state["navigation_checkpoint"].update(step="evaluation", state="awaiting_next")
    state["results"] = {row["id"]: {"state": "completed"} for row in state["plan"]}
    manager._save(state)
    with pytest.raises(ValueError, match="학습 결과 준비 기록"):
        manager.start(project_id)
    assert manager._load(project_id)["navigation_checkpoint"]["state"] == "awaiting_next"
    assert not manager.workers.calls


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


@pytest.mark.parametrize("step,gate", [("downloads", "waiting_input"), ("call-pilot-review", "waiting_review"),
                                     ("pronunciation", "waiting_review"), ("phrases", "waiting_input")])
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
        review_kind = "positive" if step == "pronunciation" else "pilot"
        output["reviews"] = [{"id": review_kind, "kind": review_kind, "queue": str(queue), "queue_sha256": digest(queue)}]
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
