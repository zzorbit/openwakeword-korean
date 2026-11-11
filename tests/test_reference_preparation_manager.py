"""Isolated preparation orchestration tests: fake TTS/child, no user data/GPU."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio, studio_worker
from korean_openwakeword.common import bound_file, digest, identity, read_json, write_json
from test_korean_library_studio import create, http_server, manager
from test_reference_candidate_manager import review, send


@pytest.fixture
def preparation(review):
    service, project_id, root = review
    initial = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=initial["active_group"])
    assert service.finish_candidates() == 0
    initial = service.reference_preview_queue(project_id)
    for row in initial["items"][:2]:
        send(service, project_id, "hear", row)
        send(service, project_id, "toggle", row)
    state = service._load(project_id)
    state["reference_preparation_policy"] = "initial_candidate_batches_v1"
    service._save(state)
    def finish():
        child = service.workers.items[-1]
        code = studio_worker.execute(child.request_path, reference_preparation=True,
                                    load_model=lambda *args: service.fake_model)
        child.returncode = code
        service.workers.identities.pop(child.pid, None)
        service.tick()
        return code
    service.finish_preparation = finish
    return service, project_id, root, initial


def test_preparation_get_counts_candidates_not_original_bank_and_never_starts_gpu(preparation, http_server):
    service, project_id, root, initial = preparation
    before = (root / ".studio/reference-candidates/state.json").read_bytes()
    view = service.describe(service._load(project_id))["reference_preparation"]
    assert view["required"] and not view["ready"] and view["status"] == "pending"
    assert view["rows"][0]["done"] == 5 and view["rows"][1]["done"] == 0
    assert [row["total"] for row in view["rows"]] == [50, 50]
    assert view["can_start"] and view["operation"] == "helper"
    status, content, _ = http_server(f"/reference-preview/queue?project={project_id}")
    assert status == 200 and not json.loads(content)["editable"]
    assert (root / ".studio/reference-candidates/state.json").read_bytes() == before
    assert not service.workers.items and service.fake_model.counter == 5


def test_helper_prepares_only_missing_groups_preserves_existing_choices_and_main_history(preparation):
    service, project_id, root, initial = preparation
    old = service._load(project_id)
    old_ledger = read_json(root / ".studio/reference-candidates/state.json")
    preserved = {key: deepcopy(old_ledger[key]) for key in ("selected", "heard", "approved", "current_batch")}
    before_waves = {row["path"]: (root / row["path"]).read_bytes() for row in old_ledger["candidates"].values()}
    view = service.prepare_start(project_id)
    assert view["reference_preparation"]["status"] == "running"
    for action in ("hear", "toggle"):
        with pytest.raises(ValueError, match="변경"):
            send(service, project_id, action, initial["items"][0])
    with pytest.raises(ValueError, match="변경"):
        send(service, project_id, "more", group=initial["active_group"])
    assert service.finish_preparation() == 0
    state = service._load(project_id)
    ready = service.describe(state)["reference_preparation"]
    assert ready["ready"] and ready["done_groups"] == 20
    assert [row["done"] for row in ready["rows"]] == [50, 50]
    assert state["index"] == old["index"] and state.get("history", []) == old.get("history", [])
    assert state["results"] == old["results"] and state["settings"] == old["settings"]
    assert state["inputs"]["reference_candidate_preparation"]
    assert bound_file(root, state["inputs"]["reference_candidate_preparation"]).is_file()
    uncertain = {**state, "reference_preparation_job": {**state["reference_preparation_job"], "status": "interrupted"}}
    assert not service._reference_preparation_view(uncertain)["ready"], "a core receipt alone cannot replace a verified helper completion"
    after = read_json(root / ".studio/reference-candidates/state.json")
    for key in ("selected", "heard", "approved"):
        assert after[key] == preserved[key]
    for key, value in preserved["current_batch"].items():
        assert after["current_batch"][key] == value
    assert all((root / path).read_bytes() == data for path, data in before_waves.items())
    assert service.fake_model.counter == 100 and len(after["candidates"]) == 100
    historical = service._reference_preparation_view({**state, "status": "completed", "index": len(state["plan"])})
    assert historical["policy"] == "initial_candidate_batches_v1" and not historical["required"]
    assert historical["ready"] and [row["done"] for row in historical["rows"]] == [50, 50]
    verification_wait = {**state, "reference_preparation_job": None,
        "settings": {**state["settings"], "reference_preparation_policy": "initial_candidate_batches_v1"},
        "index": next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "references_review"), "status": "paused"}
    pending_verification = service._reference_preparation_view(verification_wait)
    assert not pending_verification["ready"] and pending_verification["status"] == "paused"
    assert pending_verification["can_resume"], "pausing after the last batch must leave the final verification resumable"
    count = len(service.workers.items)
    service.prepare_resume(project_id)
    assert len(service.workers.items) == count  # Completed preparation is reused, not rerun.


def test_helper_pause_finishes_current_five_then_resumes_preserved_plan(preparation, monkeypatch):
    service, project_id, root, initial = preparation
    generate = service.fake_model.generate_voice_design
    requested = []
    def pause_once(**kwargs):
        result = generate(**kwargs)
        if service.fake_model.counter == 7 and not requested:
            requested.append(True)
            service.prepare_pause(project_id)
        return result
    monkeypatch.setattr(service.fake_model, "generate_voice_design", pause_once)
    service.prepare_start(project_id)
    assert service.finish_preparation() == 0
    paused = service.describe(service._load(project_id))["reference_preparation"]
    assert paused["status"] == "paused" and paused["can_resume"] and not paused["ready"]
    assert service.fake_model.counter == 10 and paused["done_groups"] == 2
    ledger = read_json(root / ".studio/reference-candidates/state.json")
    plans = deepcopy(ledger["batches"])
    service.prepare_resume(project_id)
    assert service.finish_preparation() == 0
    assert service.describe(service._load(project_id))["reference_preparation"]["ready"]
    after = read_json(root / ".studio/reference-candidates/state.json")
    assert all(after["batches"][key][:len(value)] == value for key, value in plans.items())
    assert service.fake_model.counter == 100


def test_helper_job_blocks_other_model_and_routes_require_existing_token(preparation, http_server):
    service, project_id, root, initial = preparation
    route = f"/api/projects/{project_id}/prepare_start"
    assert http_server(route, "POST", {})[0] == 403
    status, body, _ = http_server(route, "POST", {}, headers={"X-Koww-Token": service.token})
    assert status == 200 and json.loads(body)["reference_preparation"]["status"] == "running"
    other = create(service, wakeword="다음아")
    with pytest.raises(ValueError, match="기준 목소리"):
        service.start(other["id"])
    state = service._load(project_id)
    assert state["job"] is None and service._reference_review_job_alive(state)


def test_interrupted_helper_releases_no_unrelated_lock_and_is_resumable(preparation):
    service, project_id, root, initial = preparation
    service.prepare_start(project_id)
    child = service.workers.items[-1]
    child.returncode = 1
    service.workers.identities.pop(child.pid, None)
    service.tick()
    state = service._load(project_id)
    view = service.describe(state)["reference_preparation"]
    assert view["status"] == "interrupted" and view["can_resume"]
    lock = root / ".studio/reference-candidates/preparation/lock/.writer.lock"
    lock.parent.mkdir(parents=True)
    write_json(lock, {"pid": 1234567})
    with pytest.raises(ValueError, match="다른 작업"):
        service.prepare_resume(project_id)
    assert lock.exists()


@pytest.mark.parametrize("owner", ["known_dead", "unknown", "live"])
def test_deep_audio_lock_recovery_requires_registered_plan_and_exact_dead_owner(preparation, owner):
    service, project_id, root, initial = preparation
    service.prepare_start(project_id)
    child = service.workers.items[-1]
    state = service._load(project_id)
    ledger_path = root / ".studio/reference-candidates/state.json"
    saved = read_json(ledger_path)
    from korean_openwakeword import reference_candidates as core
    group = saved["groups"][1]
    plan = core._plan_candidate_batch(state["project"], saved, group)
    core._register_candidate_batch(saved, plan, group, initial=True)
    write_json(ledger_path, saved)
    output = root / ".studio/reference-candidates/jobs" / ("initial_" + plan["batch_id"]) / "audio"
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(root / ".studio/reference-candidates/preparation/state.json",
               {"status": "running", "current": {"group": group["key"], "batch_id": plan["batch_id"],
                                                  "output": output.relative_to(root).as_posix()}})
    pid = child.pid if owner != "unknown" else 9999999
    write_json(output / ".writer.lock", {"pid": pid, "started_at": "fixture"})
    child.returncode = 1
    service.workers.identities.pop(child.pid, None)
    service.tick()
    if owner == "live":
        service.workers.identities[pid] = "reused-live-pid"
    if owner != "known_dead":
        with pytest.raises(ValueError, match="잠금"):
            service.prepare_resume(project_id)
        assert (output / ".writer.lock").exists()
    else:
        service.prepare_resume(project_id)
        assert not (output / ".writer.lock").exists()
        assert list(output.glob("recovered-lock-*.json"))
        assert service.finish_preparation() == 0
        assert service.describe(service._load(project_id))["reference_preparation"]["ready"]


@pytest.mark.parametrize("with_evidence", [True, False])
def test_cross_implementation_old_manual_plan_needs_audited_resume_envelope(preparation, monkeypatch, with_evidence):
    service, project_id, root, initial = preparation
    ledger_path = root / ".studio/reference-candidates/state.json"
    saved = read_json(ledger_path)
    first = saved["candidates"][initial["items"][0]["id"]]
    directory = root / first["path"]
    directory = directory.parent.parent
    old_request = directory / "request.json"
    old_bytes = old_request.read_bytes()
    request = read_json(old_request)
    saved["job"] = {"task_id": request["task_id"], "directory": str(directory), "status": "failed",
                    "resumable": True, "group": initial["active_group"], "request_sha256": digest(old_request)}
    write_json(ledger_path, saved)
    service._implementation_id = "explicitly-migrated-new-runtime"
    service.migrate(project_id)
    if not with_evidence:
        state = service._load(project_id)
        state["migrations"] = []
        service._save(state)
    called = []
    import korean_openwakeword.reference_candidates as core
    def prepare(*args, **kwargs):
        called.append(True)
        return {"status": "paused"}
    monkeypatch.setattr(core, "prepare_initial_candidate_batches", prepare)
    service.prepare_start(project_id)
    child = service.workers.items[-1]
    code = studio_worker.execute(child.request_path, reference_preparation=True,
                                load_model=lambda *args: pytest.fail("Envelope validation must not load a model"))
    result = read_json(child.request_path.parent / "result.json")
    assert old_request.read_bytes() == old_bytes
    envelope = child.request_path.parent / "audited_resume.json"
    if with_evidence:
        assert code == 0 and called and envelope.exists()
        proof = read_json(envelope)
        assert proof["old_request"]["sha256"] == digest(old_request)
        assert proof["new_request_sha256"] == digest(child.request_path)
    else:
        assert code == 1 and not called and not envelope.exists()
        assert "이관 근거" in result["error"]


def test_interrupted_helper_can_recover_its_lock_inside_proved_old_manual_output(preparation):
    service, project_id, root, initial = preparation
    service.prepare_start(project_id)
    child = service.workers.items[-1]
    state = service._load(project_id)
    ledger_path = root / ".studio/reference-candidates/state.json"
    saved = read_json(ledger_path)
    from korean_openwakeword import reference_candidates as core
    group = saved["groups"][1]
    plan = core._plan_candidate_batch(state["project"], saved, group)
    core._register_candidate_batch(saved, plan, group, initial=True)
    directory = root / ".studio/reference-candidates/jobs/old_manual_partial"
    output = directory / "audio"
    output.mkdir(parents=True)
    request_path = directory / "request.json"
    write_json(request_path, {"task_id": "old_manual_partial", "kind": "reference_candidate_batch_v2", "plan": plan,
        "project": state["project"], "config_sha256": state["config_sha256"], "implementation_id": service._implementation_id})
    saved["job"] = {"task_id": "old_manual_partial", "directory": str(directory), "pid": 61000,
                    "request_sha256": digest(request_path), "status": "failed", "resumable": True, "group": group["key"]}
    write_json(ledger_path, saved)
    write_json(output / "plan.json", plan)
    write_json(root / ".studio/reference-candidates/preparation/state.json",
        {"status": "running", "current": {"group": group["key"], "batch_id": plan["batch_id"],
        "output": output.relative_to(root).as_posix()}})
    write_json(output / ".writer.lock", {"pid": child.pid})  # The interrupted helper owned the reused output.
    child.returncode = 1
    service.workers.identities.pop(child.pid, None)
    service.tick()
    service.prepare_resume(project_id)
    assert not (output / ".writer.lock").exists()
    assert list(output.glob("recovered-lock-*.json"))


def test_analysis_entry_is_blocked_while_preparation_worker_owns_gpu(preparation, http_server, monkeypatch):
    service, project_id, root, initial = preparation
    service.prepare_start(project_id)
    calls = []
    monkeypatch.setattr(service.analysis, "launch", lambda *args: calls.append(args))
    status, body, _ = http_server(f"/analysis/{project_id}/api/compare", "POST", {},
                                 headers={"X-Koww-Token": service.token})
    assert status == 400 and "목소리 후보" in json.loads(body)["error"]
    assert calls == [] and service.fake_model.counter == 5


def test_active_analysis_blocks_helper_training_and_manual_generation_entries(preparation, monkeypatch):
    service, project_id, root, initial = preparation
    monkeypatch.setattr(service.analysis, "busy", lambda project_id: True)
    with pytest.raises(ValueError, match="다른 생성"):
        service.prepare_start(project_id)
    other = create(service, wakeword="비교중아")
    with pytest.raises(ValueError, match="비교·선택"):
        service.start(other["id"])
    state = service._load(project_id)
    state.pop("reference_preparation_policy")  # Exercise the existing manual-generation entry too.
    service._save(state)
    with pytest.raises(ValueError, match="다른 생성"):
        send(service, project_id, "more", group=initial["active_group"])
    assert service.workers.items == [] and service.fake_model.counter == 5
