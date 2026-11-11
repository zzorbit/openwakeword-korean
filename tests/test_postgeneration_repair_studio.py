"""Selective post-review orchestration; actual journals, fake acoustic model/worker."""
import copy
import http.client
import json
from pathlib import Path
import threading

import pytest

from korean_openwakeword import pilot, postgeneration_repair as repair, studio, studio_steps
from korean_openwakeword.common import bound_file, digest, identity, read_json, read_rows, snapshot_json, write_json
from test_korean_library_studio import create, manager  # noqa: F401
from test_postgeneration_repair import fixture as acoustic_fixture


def waiting_post(manager, tmp_path, monkeypatch):
    project, config, banks, sources, queue, generated, targets, report, model, loads = acoustic_fixture(tmp_path, monkeypatch)
    initial = create(manager)
    state = manager._load(initial["id"])
    manager.registry["projects"][state["id"]] = str(project)
    write_json(manager.registry_path, manager.registry)
    root = project.parent
    state.update(project=str(project), config_sha256=digest(project), wakeword=config["wakeword"], index=1,
        status="waiting_review", auto_continue=False, job=None,
        plan=[{"id": "positives", "label": "본생성", "kind": "task"},
              {"id": "positive_review", "label": "본검수", "kind": "review"},
              {"id": "hard_phrases", "label": "유사어", "kind": "input"}])
    state["settings"].update(targets, train_references=2, validation_references=2,
        pilot_review_policy=pilot.PREPOST_PILOT_POLICY, reference_selection_policy=None,
        reference_preparation_policy=None, evaluation_policy=None)
    config["targets"].update(positive_originals=targets["originals"], validation_originals=targets["validation_originals"])
    state["settings"].update(rounds=config["augmentation_rounds"], updates=config["training"]["updates"],
        generic_hours=config["targets"]["generic_hours"], korean_hours=config["targets"]["korean_hours"])
    write_json(project, config)
    state["config_sha256"] = digest(project)
    descriptor = {"id": "positive", "kind": "positive", "queue": generated["queue"], "queue_sha256": generated["queue_binding"]["sha256"]}
    state["results"] = {"positives": {"state": "completed", "result": generated},
                        "positive_review": {"state": "waiting_review", "reviews": [descriptor]}}
    artifacts = [generated["queue_binding"], *generated["preview_sources"].values()]
    receipt = root / ".studio/original_generation_result.json"
    write_json(receipt, {"result": state["results"]["positives"], "artifact_bindings": artifacts})
    state["history"] = [{"stage": "positives", "result": str(receipt), "sha256": digest(receipt)}]
    manager._save(state)
    # Concurrent source editing by other implementation owners is outside this
    # worker fixture. Individual guard tests below explicitly supply failures.
    monkeypatch.setattr(studio, "_implementation", lambda: manager._implementation_id)
    return state, descriptor, queue, generated, report, model, loads


def rejection_payload(root, descriptor, report):
    return {"queue_sha256": descriptor["queue_sha256"], "decisions": read_json(bound_file(root, report))["decisions"]}


def finish_repair_worker(manager, project_id):
    worker = manager.workers.items[-1]
    request = read_json(worker.request_path)
    output = studio_steps.run_stage(request["project"], request["workspace"], request["settings"], request["stage"], request["context"])
    worker.finish(output)
    manager.tick()
    return output


def test_rejection_starts_same_stage_preserves_generation_and_carries_only_true_old_answers(manager, tmp_path, monkeypatch):
    state, descriptor, rows, generated, report, model, loads = waiting_post(manager, tmp_path, monkeypatch)
    root = Path(state["project"]).parent
    original = {path: digest(path) for path in [Path(generated["queue"]), *(root / item["path"] for item in generated["preview_sources"].values()), root / "project.json", root / "approval.json"]}
    old_history, old_generation = copy.deepcopy(state["history"]), copy.deepcopy(state["results"]["positives"])
    result = manager.save_review(state["id"], "positive", rejection_payload(root, descriptor, report))
    assert result["workflow_action"] == "repair" and result["state"] == "postgeneration_repair_started"
    assert result["review_finalized"] and result["policy_approved"] is False
    live = manager._load(state["id"])
    assert live["index"] == 1 and live["job"]["stage"] == "positive_review"
    assert live["history"] == old_history and live["results"]["positives"] == old_generation
    assert "positive_review" not in live["results"]
    assert len(live["inputs"]["postgeneration_repair_rounds"]) == 1
    assert manager.describe(live)["postgeneration_repair"]["source_queue_sha256"] == descriptor["queue_sha256"]
    with pytest.raises(ValueError, match="현재 검수 목록"):
        manager.save_review(state["id"], "positive", rejection_payload(root, descriptor, report))
    output = finish_repair_worker(manager, state["id"])
    assert output["state"] == "waiting_review" and model.calls == 3 and len(loads) == 1
    current = manager._load(state["id"])
    new_descriptor = current["results"]["positive_review"]["reviews"][0]
    assert new_descriptor["queue_sha256"] != descriptor["queue_sha256"]
    metadata = manager._postgeneration_repair_review(current, new_descriptor)
    assert len(metadata["carried_decisions"]) == 3 and len(metadata["pending_ids"]) == 1
    current_rows = read_rows(new_descriptor["queue"])
    old = {row["sample_id"]: row["sha256"] for row in rows}
    assert all(old[sample] == next(row["sha256"] for row in current_rows if row["sample_id"] == sample) for sample in metadata["carried_decisions"])
    assert original == {path: digest(path) for path in original}
    # Read-only status/preview/history must not repeatedly validate whole WAV pools.
    monkeypatch.setattr(repair, "resolve_postgeneration_repair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full resolver on presentation poll")))
    assert manager.describe(current)["postgeneration_repair"]["state"] == "waiting_review"
    _, page, total = manager.positive_preview_rows(state["id"], 0, 100)
    assert total == 12 and len(page) == 12
    assert page[0][2]["sample_id"] == current_rows[0]["sample_id"]


def test_prior_saved_negative_report_recovery_never_rewrites_human_answers(manager, tmp_path, monkeypatch):
    state, descriptor, _, generated, report, _, _ = waiting_post(manager, tmp_path, monkeypatch)
    root = Path(state["project"]).parent
    human = read_json(bound_file(root, report))
    archive = snapshot_json(root, human, "positive_review")
    write_json(root / "review/positive_results.json", human)
    before = {path: digest(path) for path in [root / "review/positive_results.json", bound_file(root, archive)]}
    result = manager.retry_postgeneration(state["id"], {"queue_sha256": descriptor["queue_sha256"]})
    assert result["state"] == "postgeneration_repair_started"
    state_after = manager._load(state["id"])
    saved_round = state_after["inputs"]["postgeneration_repair_rounds"][0]
    assert saved_round["source_queue_sha256"] == descriptor["queue_sha256"]
    assert before == {path: digest(path) for path in before}
    count = len(manager.workers.items)
    assert manager.retry_postgeneration(state["id"], {"queue_sha256": descriptor["queue_sha256"]})["state"] == "postgeneration_repair_started"
    assert len(manager.workers.items) == count
    assert len(manager._load(state["id"])["inputs"]["postgeneration_repair_rounds"]) == 1
    with pytest.raises(ValueError, match="목록이 바뀌"):
        manager.retry_postgeneration(state["id"], {"queue_sha256": "stale"})


def test_source_guard_rejects_before_writing_post_report(manager, tmp_path, monkeypatch):
    state, descriptor, _, _, report, _, _ = waiting_post(manager, tmp_path, monkeypatch)
    root = Path(state["project"]).parent
    monkeypatch.setattr(manager, "_guard_config", lambda _state: (_ for _ in ()).throw(ValueError("stale implementation")))
    with pytest.raises(ValueError, match="stale implementation"):
        manager.save_review(state["id"], "positive", rejection_payload(root, descriptor, report))
    assert not (root / "review/positive_results.json").exists()
    assert not manager.workers.items


def test_failed_launch_preserves_request_and_retries_same_plan(manager, tmp_path, monkeypatch):
    state, descriptor, _, _, report, _, _ = waiting_post(manager, tmp_path, monkeypatch)
    root = Path(state["project"]).parent
    real_start = manager.start
    monkeypatch.setattr(manager, "start", lambda _project: (_ for _ in ()).throw(OSError("fixture launch failed")))
    result = manager.save_review(state["id"], "positive", rejection_payload(root, descriptor, report))
    assert result["state"] == "postgeneration_repair_pending"
    paused = manager._load(state["id"])
    request = copy.deepcopy(paused["inputs"]["postgeneration_repair_rounds"])
    assert paused["status"] == "paused" and not manager.workers.items
    monkeypatch.setattr(manager, "start", real_start)
    assert manager.retry_postgeneration(state["id"])["state"] == "postgeneration_repair_started"
    assert manager._load(state["id"])["inputs"]["postgeneration_repair_rounds"] == request


def test_carried_answers_cannot_be_changed_and_final_admission_receives_effective_corpus(manager, tmp_path, monkeypatch):
    state, descriptor, _, generated, report, _, _ = waiting_post(manager, tmp_path, monkeypatch)
    root = Path(state["project"]).parent
    manager.save_review(state["id"], "positive", rejection_payload(root, descriptor, report))
    finish_repair_worker(manager, state["id"])
    current = manager._load(state["id"])
    descriptor = current["results"]["positive_review"]["reviews"][0]
    rows = read_rows(descriptor["queue"])
    decisions = {row["sample_id"]: {"decision": "keep", "listened": True} for row in rows}
    carried = manager._postgeneration_repair_review(current, descriptor)["carried_decisions"]
    tampered = copy.deepcopy(decisions); tampered[next(iter(carried))]["decision"] = "discard"
    saved_before = digest(root / "review/positive_results.json")
    with pytest.raises(ValueError, match="기존 적합 판정"):
        manager.save_review(state["id"], "positive", {"queue_sha256": descriptor["queue_sha256"], "decisions": tampered})
    assert digest(root / "review/positive_results.json") == saved_before
    result = manager.save_review(state["id"], "positive", {"queue_sha256": descriptor["queue_sha256"], "decisions": decisions})
    assert result["policy_approved"] is True
    calls = []
    monkeypatch.setattr(pilot, "admit_postgeneration_positives", lambda project, corpus: calls.append(corpus) or [])
    manager.start(state["id"])
    finish_repair_worker(manager, state["id"])
    assert len(calls) == 1 and calls[0]["queue_binding"]["sha256"] == descriptor["queue_sha256"]
    assert calls[0]["preview_sources"] != generated["preview_sources"]
    final = manager._load(state["id"])
    monkeypatch.setattr(repair, "resolve_postgeneration_repair", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full resolver on historical view")))
    _, answer, history = manager.history_review_rows(state["id"], "pronunciation")
    assert len(history) == 4 and set(answer["decisions"]) == {row["sample_id"] for row in history}


def test_retry_http_action_and_repair_queue_expose_hash_bound_retained_answers(manager, tmp_path, monkeypatch):
    state, descriptor, _, _, report, _, _ = waiting_post(manager, tmp_path, monkeypatch)
    root = Path(state["project"]).parent
    human = read_json(bound_file(root, report)); snapshot_json(root, human, "positive_review")
    write_json(root / "review/positive_results.json", human)
    server = studio.make_studio_server(manager, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
        connection.request("POST", f"/api/projects/{state['id']}/retry_postgeneration", json.dumps({"queue_sha256": descriptor["queue_sha256"]}),
            {"Content-Type": "application/json", "X-Koww-Token": manager.token})
        response = connection.getresponse(); payload = json.loads(response.read())
        assert response.status == 200, payload
        assert payload["state"] == "postgeneration_repair_started"
        finish_repair_worker(manager, state["id"])
        current = manager._load(state["id"])
        descriptor = current["results"]["positive_review"]["reviews"][0]
        connection.request("GET", f"/review/queue?project={state['id']}&queue=positive&rev={descriptor['queue_sha256']}")
        response = connection.getresponse(); payload = json.loads(response.read())
        assert response.status == 200, payload
        assert len(payload["items"]) == 4 and len(payload["retained_decisions"]) == 3 and len(payload["focus_sample_ids"]) == 1
        by_id = {row["id"]: row for row in payload["items"]}
        assert all(answer == {"decision": "keep", "listened": True, "audio_sha256": by_id[sample]["audio_sha256"]} for sample, answer in payload["retained_decisions"].items())
        assert set(payload["focus_sample_ids"]).isdisjoint(payload["retained_decisions"])
        connection.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
