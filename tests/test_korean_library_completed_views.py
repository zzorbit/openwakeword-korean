"""Completed story views use saved evidence without restarting or reapproving work."""
import json
from pathlib import Path
import wave

import pytest

from korean_openwakeword import studio, studio_data, studio_steps
from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.pilot import PREPOST_PILOT_POLICY, TRAIN_PILOT_POLICY
from test_korean_library_studio import create, http_server, manager  # noqa: F401


def complete(manager, project_id, results):
    state = manager._load(project_id)
    state.update(status="completed", index=len(state["plan"]), job=None)
    state["results"] = {key: {"state": "completed", "result": value} for key, value in results.items()}
    manager._save(state)
    return state


def saved_review(manager, *, hard=False, legacy=False):
    project = create(manager)
    state = manager._load(project["id"])
    assert state["settings"]["pilot_review_policy"] == PREPOST_PILOT_POLICY
    if legacy:
        assert not hard
        state["settings"]["pilot_review_policy"] = TRAIN_PILOT_POLICY
        state["plan"] = studio_steps.workflow_plan(state["settings"])
        manager._save(state)
    generation = "hard_generation" if hard else "pilot" if legacy else "positives"
    review_stage = "hard_review" if hard else "pilot_review" if legacy else "positive_review"
    queue_id = "hard" if hard else "pilot" if legacy else "positive"
    state = complete(manager, project["id"], {generation: {}, review_stage: {"reviewed_queues": 1}})
    root = Path(state["project"]).parent
    audio = root / "data/history.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 1600)
    row = {"sample_id": "heard-1", "text": "보라냐" if hard else "보라야", "path": "data/history.wav",
           "sha256": digest(audio), "voice_family_id": "voice-1"}
    queue = root / ".studio/history-queue.jsonl"
    queue.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    answer = {"kind": ("content" if hard else queue_id) + "_listening_review",
              "wakeword": state["wakeword"], "saved_at": "2026-11-01T00:00:00Z", "queue_sha256": digest(queue),
              "representative_queue_sha256": digest(queue),
              "source_scope": "Human dispositions entered through the local review screen",
              "queued_audio_sha256": {row["sample_id"]: row["sha256"]},
              "decisions": {row["sample_id"]: {"decision": "keep", "listened": True}}}
    if hard:
        selection = root / ".studio/hard_representative_selection_v1.json"
        write_json(selection, {"policy": "one_per_phrase_v1", "source_manifest": {"sha256": "f" * 64},
                              "representative_queue": {"path": ".studio/history-queue.jsonl", "sha256": digest(queue)}})
        answer.update(synthetic_review_policy="one_per_phrase_v1", representative_selection_sha256=digest(selection),
                      source_manifest_sha256="f" * 64)
        # A different whole pool must never replace the actually heard queue.
        state["results"]["hard_generation"]["result"]["review"] = {"queue": str(root / "whole-pool.jsonl")}
    else:
        state["results"][generation]["result"].update(queue=str(queue),
            queue_binding={"path": str(queue), "sha256": digest(queue)})
    answer_path = root / ".studio/history-answer.json"
    write_json(answer_path, answer)
    state["inputs"]["review_answers"] = {queue_id: {"path": str(answer_path), "sha256": digest(answer_path)}}
    if not hard:
        state["inputs"]["review_results"] = {queue_id: {"state": "saved", "kept": 1, "total": 1,
            "policy_approved": True}}
    manager._save(state)
    return state, audio, queue, answer_path


def test_completed_counts_come_from_saved_results_not_current_progress(manager):
    project = create(manager)
    state = complete(manager, project["id"], {
        "references": {"outputs": [{"generated": 20}, {"generated": 30}]}, "references_review": {},
        "positives": {"train_originals": 1000, "validation_originals": 2000,
                      "outputs": [{"technical_pass": 998}, {"technical_pass": 1999}]},
        "hard_phrases": {"phrases": ["보라냐", "돌아와"]},
        "hard_generation": {"technical_pass": 40, "target_originals": 40},
        "features": {"pools": {"positive": {"feature_rows": 1000}}, "positive_expected_rows": 1000},
        "training": {"members": [{"updates": 10000}, {"updates": 10000}, {"updates": 9999}]},
        "development": {"audio_files": 48000, "negative_recordings": 137},
        "evaluation": {"report": {"sessions": 48137}}})
    write_json(Path(state["project"]).parent / ".studio/progress.json",
               {"done": 1, "total": 100, "stage": "evaluation"})
    before = digest(manager._path(state["id"]))
    views = manager.describe(state)["completed_stage_views"]
    assert views["voices"]["rows"][0]["done"] == 20
    assert views["positive-generation"]["rows"][0]["done"] == 998
    assert views["positive-generation"]["rows"][1]["done"] == 1999
    assert views["phrases"]["phrases"] == ["보라냐", "돌아와"]
    assert views["hard-generation"]["rows"][0]["done"] == 40
    assert views["features"]["rows"][0]["done"] == 1000
    assert views["training"]["rows"][2]["done"] == 9999
    assert views["evaluation"]["rows"][0] == {"id": "evaluation", "label": "성능 평가 음원", "done": 48137,
                                               "total": 48137, "unit": "개"}
    assert digest(manager._path(state["id"])) == before


def test_pending_failed_missing_and_target_only_results_do_not_invent_completion(manager):
    project = create(manager)
    state = complete(manager, project["id"], {"positives": {"train_originals": 1000, "validation_originals": 2000}})
    views = manager._completed_stage_views(state)
    assert views["positive-generation"]["rows"][0]["done"] is None
    assert views["positive-generation"]["rows"][0]["total"] == 1000
    assert "training" not in views
    state["results"]["positives"]["state"] = "failed"
    assert "positive-generation" not in manager._completed_stage_views(state)
    state["results"]["positives"]["state"] = "completed"
    state["index"] = next(i for i, row in enumerate(state["plan"]) if row["id"] == "positives")
    assert "positive-generation" not in manager._completed_stage_views(state)


def test_small_metadata_cache_reuses_reads_and_rechecks_changed_binding(manager, monkeypatch):
    project = create(manager)
    state = manager._load(project["id"])
    path = Path(state["project"]).parent / ".studio/small.json"
    write_json(path, {"count": 1})
    expected = digest(path)
    original = studio.read_json
    opened = []
    monkeypatch.setattr(studio, "read_json", lambda p: (opened.append(str(p)), original(p))[1])
    assert manager._history_read(state, path, sha256=expected) == {"count": 1}
    assert manager._history_read(state, path, sha256=expected) == {"count": 1}
    assert len(opened) == 1
    write_json(path, {"count": 200})
    with pytest.raises(ValueError, match="변경"):
        manager._history_read(state, path, sha256=expected)
    assert manager._history_read(state, path) == {"count": 200}


def test_download_history_uses_verified_receipt_bytes_without_opening_archives(manager, monkeypatch):
    project = create(manager)
    state = complete(manager, project["id"], {"data_download": {"recipe_version": 6}})
    asset = {"id": "generic", "filename": "large.npy", "size_bytes": 10000000000, "sha256": "a" * 64,
             "source_url": "https://example.test/large.npy", "checksum_algorithm": "sha256"}
    monkeypatch.setattr(studio_data, "_official_assets", lambda: [asset])
    shared = manager.workspace / "shared"
    download = shared / "downloads/large.npy"
    write_json(download.with_name(download.name + ".receipt.json"), {"path": str(download), "bytes": asset["size_bytes"],
        "sha256": asset["sha256"], "source_url": asset["source_url"], "reused": True, "completed_at": "2026-11-01"})
    original_digest = studio.digest
    def small_only(path):
        assert Path(path) != download
        return original_digest(path)
    monkeypatch.setattr(studio, "digest", small_only)
    files = manager._completed_stage_views(state)["downloads"]["progress"]["files"]
    verified = next(row for row in files if row["id"] == "base:generic")
    assert verified["done"] == verified["total"] == asset["size_bytes"]
    assert verified["state"] == "complete" and verified["network_bytes"] == 0
    assert verified["bytes_per_second"] is None
    assert any(row["state"] == "unknown" and row["done"] is None for row in files)


@pytest.mark.parametrize("hard,legacy", [(False, False), (False, True), (True, False)])
def test_completed_review_replays_exact_saved_decisions_without_save_route(manager, http_server, hard, legacy):
    state, audio, queue, answer = saved_review(manager, hard=hard, legacy=legacy)
    step = "hard-review" if hard else "pronunciation"
    paths = [manager._path(state["id"]), audio, queue, answer]
    before = {str(path): digest(path) for path in paths}
    view = manager.describe(state)["completed_stage_views"][step]
    assert view["review"]["saved"] is True and len(view["review"]["items"]) == 1
    item = view["review"]["items"][0]
    assert item["decision"] == "keep" and item["listened"] is True
    status, content, headers = http_server(item["audio"])
    assert status == 200 and content == audio.read_bytes()
    assert headers["Content-Type"].startswith("audio/")
    queue_id = "hard" if hard else "pilot" if legacy else "positive"
    assert http_server(f"/review/queue?project={state['id']}&queue={queue_id}")[0] == 400
    assert http_server(item["audio"], method="POST", payload={}, headers={"X-Koww-Token": manager.token})[0] in (400, 404)
    assert {str(path): digest(path) for path in paths} == before
    assert manager._load(state["id"])["inputs"]["review_answers"] == state["inputs"]["review_answers"]


def test_changed_history_queue_and_answers_never_appear_as_saved_review(manager):
    state, _, queue, answer = saved_review(manager)
    assert manager._completed_stage_views(state)["pronunciation"]["review"]["saved"] is True
    queue.write_text(queue.read_text(encoding="utf-8").replace("보라야", "다른말"), encoding="utf-8")
    assert manager._completed_stage_views(state)["pronunciation"]["review"] == {"items": [], "saved": False}
    assert read_json(answer)["decisions"]["heard-1"]["decision"] == "keep"


def test_history_audio_refuses_changed_audio_or_paths_outside_model(manager, http_server):
    state, audio, _, _ = saved_review(manager)
    url = manager._completed_stage_views(state)["pronunciation"]["review"]["items"][0]["audio"]
    audio.write_bytes(b"changed audio")
    assert http_server(url)[0] == 400
    other = manager.workspace / "outside-model.wav"
    other.write_bytes(b"outside")
    monkey_rows = [{"path": str(other), "sha256": digest(other)}]
    manager.history_review_rows = lambda project_id, step: (state, {}, monkey_rows)
    assert http_server(url)[0] == 400


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionAbortedError, ConnectionResetError])
def test_cancelled_history_audio_does_not_attempt_an_error_response(manager, error):
    state, _, _, _ = saved_review(manager)
    url = manager._completed_stage_views(state)["pronunciation"]["review"]["items"][0]["audio"]
    server = studio.make_studio_server(manager, port=0)
    try:
        handler = object.__new__(server.RequestHandlerClass)
        handler.path = url
        handler.guard = lambda: None
        responses = []
        handler.respond = lambda *args, **kwargs: responses.append((args, kwargs))
        def cancelled(path):
            raise error("browser cancelled the audio request")
        handler.send_file = cancelled
        handler.do_GET()
        assert responses == []
    finally:
        server.server_close()
