"""Playback-start drafts are partial, revisioned, and finalized only on Next."""
import http.client
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from korean_openwakeword import pilot, review, studio
from korean_openwakeword.common import bound_file, digest, read_json, read_rows, write_json
from test_korean_library_studio import create, manager, reference_queue  # noqa: F401


def preflight_queue(manager, *, phase="pre"):
    project = create(manager)
    project_id = project["id"]
    audio, queue = reference_queue(manager, project_id, kind="pilot" if phase == "pre" else "positive")
    state = manager._load(project_id)
    state["settings"]["pilot_review_policy"] = pilot.PREPOST_PILOT_POLICY
    rows = read_rows(queue)
    for row in rows:
        reference = {"sample_id": "reference-1", "voice_family_id": row["voice_family_id"],
                     "path": row["path"], "sha256": row["sha256"], "provenance": "qwen_synthetic_reference",
                     "human_reference_used": False, "split": "train", "gender": "man", "age_group": "child"}
        row.update(pronunciation_policy=pilot.PREPOST_PILOT_POLICY, text=state["wakeword"],
                   provenance="qwen_tts", human_reference_used=False, qc={"passed": True},
                   split="train", review_group="학습용", voice_number=1,
                   voice_reference_sha256=reference["sha256"], preflight_reference=reference)
        if phase == "pre": row["preflight_round"] = 0
    queue.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    descriptor = {"id": "pilot" if phase == "pre" else "positive", "kind": "pilot" if phase == "pre" else "positive",
                  "queue": str(queue), "queue_sha256": digest(queue)}
    stage_id = "pilot_review" if phase == "pre" else "positive_review"
    state.update(plan=[{"id": "pilot", "label": "샘플", "kind": "task"},
                       {"id": stage_id, "label": "검수", "kind": "review"}], index=1,
                 results={stage_id: {"state": "waiting_review", "reviews": [descriptor]}}, status="waiting_review")
    manager._save(state)
    return project_id, descriptor, rows


def autosave(manager, project_id, descriptor, row, *, revision=0, decision="keep", started=True, completed=False, **changes):
    payload = {"action": "autosave", "queue_sha256": descriptor["queue_sha256"],
               "sample_id": row["sample_id"], "audio_sha256": row["sha256"], "decision": decision,
               "playback_started": started, "playback_completed": completed, "revision": revision, **changes}
    return manager.save_review(project_id, descriptor["id"], payload)


def finalize(manager, project_id, descriptor, revision):
    return manager.save_review(project_id, descriptor["id"], {"action": "finalize",
        "queue_sha256": descriptor["queue_sha256"], "revision": revision})


def test_start_autosaves_keep_but_never_fakes_completed_listening_or_finalizes(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    state_before = manager._path(project_id).read_bytes()
    approval = Path(manager._load(project_id)["project"]).parent / "approval.json"
    approval_before = approval.read_bytes()
    result = autosave(manager, project_id, descriptor, rows[0])
    assert result["complete"] is True and result["draft_revision"] == 1
    assert result["queue_sha256"] == descriptor["queue_sha256"]
    assert result["draft_decisions"][rows[0]["sample_id"]] == {
        "decision": "keep", "playback_started": True, "playback_completed": False, "listened": False}
    assert manager._path(project_id).read_bytes() == state_before
    assert approval.read_bytes() == approval_before
    assert manager._load(project_id)["status"] == "waiting_review"
    assert manager.preflight_draft(project_id, descriptor["id"]) == result


@pytest.mark.parametrize("bad", [{"started": False}, {"revision": 2}, {"queue_sha256": "other"},
                                  {"audio_sha256": "other"}, {"sample_id": "other"},
                                  {"started": False, "completed": True}, {"started": 1}])
def test_unheard_keep_stale_revision_or_changed_binding_is_rejected_without_write(manager, bad):
    project_id, descriptor, rows = preflight_queue(manager)
    root = Path(manager._load(project_id)["project"]).parent
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError): autosave(manager, project_id, descriptor, rows[0], **bad)
    assert before == {str(path): digest(path) for path in root.rglob("*") if path.is_file()}


def test_unheard_discard_can_clear_and_started_facts_cannot_be_erased(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    result = autosave(manager, project_id, descriptor, rows[0], decision="discard", started=False)
    assert result["complete"] and result["draft_decisions"][rows[0]["sample_id"]]["listened"] is False
    result = autosave(manager, project_id, descriptor, rows[0], revision=1, decision="clear", started=False)
    assert result["draft_decisions"] == {} and result["draft_revision"] == 2
    with pytest.raises(ValueError, match="아직 판정"): finalize(manager, project_id, descriptor, 2)
    result = autosave(manager, project_id, descriptor, rows[0], revision=2, completed=True)
    assert result["draft_decisions"][rows[0]["sample_id"]]["listened"] is True
    with pytest.raises(ValueError, match="기록은 지울"): autosave(manager, project_id, descriptor, rows[0], revision=3, decision="clear", started=False)
    result = autosave(manager, project_id, descriptor, rows[0], revision=3, decision="discard", started=False)
    assert result["draft_decisions"][rows[0]["sample_id"]]["playback_completed"] is True


def test_next_finalizes_started_keep_report_and_preflight_accepts_only_explicit_policy(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    autosave(manager, project_id, descriptor, rows[0])
    result = finalize(manager, project_id, descriptor, 1)
    assert result["policy_approved"] is True and result["workflow_action"] == "resume"
    state = manager._load(project_id)
    root = Path(state["project"]).parent
    report = read_json(bound_file(root, result["report"]))
    assert report["autosave_policy"] == review.PLAY_STARTED_POLICY
    assert report["playback_started_count"] == 1 and report["playback_completed_count"] == 0
    assert report["decisions"][rows[0]["sample_id"]]["listened"] is False
    banks = {"train": [rows[0]["preflight_reference"]], "validation": []}
    approved, pending, _, _ = pilot._preflight_state(root, read_json(state["project"]), banks, state["inputs"]["call_pilot_reviews"])
    assert set(approved) == {rows[0]["voice_family_id"]} and not pending
    report.pop("autosave_policy")
    # The old unmarked format still requires completed listening.
    from korean_openwakeword.common import snapshot_json
    altered = snapshot_json(root, report, "unmarked_test")
    evidence = [{"queue": state["inputs"]["call_pilot_reviews"][0]["queue"], "report": altered}]
    with pytest.raises(ValueError): pilot._preflight_state(root, read_json(state["project"]), banks, evidence)


def test_next_with_unheard_discard_uses_existing_retry_path(manager, monkeypatch):
    project_id, descriptor, rows = preflight_queue(manager)
    autosave(manager, project_id, descriptor, rows[0], decision="discard", started=False)
    starts = []
    monkeypatch.setattr(manager, "start", lambda project: starts.append(project))
    result = finalize(manager, project_id, descriptor, 1)
    assert result["workflow_action"] == "retry" and result["state"] == "pilot_retry_started"
    assert starts == [project_id]
    state = manager._load(project_id)
    report = read_json(bound_file(Path(state["project"]).parent, result["report"]))
    assert report["playback_started_count"] == report["playback_completed_count"] == 0
    assert len(state["inputs"]["call_pilot_reviews"]) == 1
    root = Path(state["project"]).parent
    banks = {"train": [rows[0]["preflight_reference"]], "validation": []}
    approved, pending, _, _ = pilot._preflight_state(root, read_json(state["project"]), banks, state["inputs"]["call_pilot_reviews"])
    assert approved == {} and pending == {rows[0]["voice_family_id"]}


def test_post_review_and_old_api_stay_strict(manager):
    project_id, descriptor, rows = preflight_queue(manager, phase="post")
    with pytest.raises(ValueError, match="사전 검수에서만"):
        autosave(manager, project_id, descriptor, rows[0])
    with pytest.raises(ValueError):
        manager.save_review(project_id, descriptor["id"], {"queue_sha256": descriptor["queue_sha256"],
            "decisions": {rows[0]["sample_id"]: {"decision": "keep", "listened": False}}})


def test_concurrent_same_revision_has_one_writer_and_one_stale_rejection(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    def save(_):
        try:
            return autosave(manager, project_id, descriptor, rows[0])["draft_revision"]
        except ValueError as error:
            return str(error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, range(2)))
    assert results.count(1) == 1
    assert sum(isinstance(value, str) and "판정이 바뀌" in value for value in results) == 1


def test_old_preflight_save_keeps_strict_listened_requirement_and_report_format(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    with pytest.raises(ValueError):
        manager.save_review(project_id, descriptor["id"], {"queue_sha256": descriptor["queue_sha256"],
            "decisions": {rows[0]["sample_id"]: {"decision": "keep", "listened": False}}})
    result = manager.save_review(project_id, descriptor["id"], {"queue_sha256": descriptor["queue_sha256"],
            "decisions": {rows[0]["sample_id"]: {"decision": "keep", "listened": True}}})
    state = manager._load(project_id); root = Path(state["project"]).parent
    report = read_json(bound_file(root, result["report"]))
    assert "autosave_policy" not in report
    banks = {"train": [rows[0]["preflight_reference"]], "validation": []}
    assert not pilot._preflight_state(root, read_json(state["project"]), banks, state["inputs"]["call_pilot_reviews"])[1]


def test_completed_history_preserves_started_keep_and_age_gender_without_fake_completion(manager, monkeypatch):
    project_id, descriptor, rows = preflight_queue(manager)
    autosave(manager, project_id, descriptor, rows[0])
    finalize(manager, project_id, descriptor, 1)
    state = manager._load(project_id)
    state["settings"].update(train_references=1, validation_references=0)
    manager._save(state)
    monkeypatch.setattr(manager, "_history_result", lambda state, stage: {} if stage == "pilot_review" else None)
    _, answer, saved_rows = manager._history_call_pilot_rows(state)
    assert len(saved_rows) == 1 and answer["autosave_policy"] == review.PLAY_STARTED_POLICY
    assert answer["decisions"][rows[0]["sample_id"]]["listened"] is False
    views = manager._completed_stage_views(state)
    history = views["call-pilot-review"]["review"]
    assert history["saved"] and history["autosave_policy"] == review.PLAY_STARTED_POLICY
    assert history["items"][0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    assert history["items"][0]["playback_started"] is True and history["items"][0]["playback_completed"] is False


def test_http_queue_exposes_bound_draft_and_age_gender_labels(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    autosave(manager, project_id, descriptor, rows[0])
    server = studio.make_studio_server(manager, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", f"/review/queue?project={project_id}&queue={descriptor['id']}&rev={descriptor['queue_sha256']}", headers={"X-KOWW-Token": manager.token})
        response = connection.getresponse(); data = json.loads(response.read())
        assert response.status == 200, data
        assert data["autosave_policy"] == review.PLAY_STARTED_POLICY and data["draft_revision"] == 1
        assert data["items"][0]["audio_sha256"] == rows[0]["sha256"]
        assert data["items"][0]["gender"] == "남성" and data["items"][0]["age_group"] == "어린이"
        assert data["items"][0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_draft_with_changed_provenance_is_rejected_before_further_save(manager):
    project_id, descriptor, rows = preflight_queue(manager)
    autosave(manager, project_id, descriptor, rows[0])
    state = manager._load(project_id)
    path = manager._preflight_draft_path(state, descriptor)
    draft = read_json(path); draft["source_scope"] = "imported guesses"
    write_json(path, draft)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="입력 출처"):
        autosave(manager, project_id, descriptor, rows[0], revision=1)
    assert path.read_bytes() == before
