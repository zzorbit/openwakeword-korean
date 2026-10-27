"""One-per-phrase listening never changes the generated source pool."""
from copy import deepcopy
import json
from pathlib import Path
import wave

import pytest

from korean_openwakeword import studio, studio_steps as steps
from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json, write_immutable_rows
from test_korean_library_studio import create, http_server, manager  # noqa: F401


def generated_hard(manager, phrase_count=6, voice_count=20):
    view = create(manager)
    state = manager._load(view["id"])
    root = Path(state["project"]).parent
    bank = read_json(root / "voices.json")["families"]
    voices = [r for r in bank if r["split"] == "train"][:voice_count]
    phrases = ["다" + chr(0xac00 + index) for index in range(phrase_count)]
    rows = []
    for voice in voices:
        for phrase in phrases:
            slot = len(rows)
            path = root / "data/generated/hard" / f"hard_{slot}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes((slot + 1).to_bytes(2, "little") * 1600)
            rows.append({"sample_id": f"hard_{slot}", "slot": slot, "path": path.relative_to(root).as_posix(),
                "sha256": digest(path), "technical_pass": True, "text": phrase, "voice_family_id": voice["family_id"],
                "dataset_id": "qwen_fixture", "split": "train", "provenance": "qwen_tts", "human_reference_used": False})
    write_immutable_rows(root / "data/generated/hard/candidates.jsonl", rows)
    workspace = root / ".studio"
    old = steps._hard_queue(state["project"], workspace)
    state.update(index=next(i for i, item in enumerate(state["plan"]) if item["id"] == "hard_review"),
                 status="waiting_review", auto_continue=False)
    state["results"]["hard_generation"] = {"state": "completed", "result": {
        "full_target_reached": True, "target_originals": len(rows), "technical_pass": len(rows), "review": old}}
    descriptor = steps._hard_queue(state["project"], workspace, "one_per_phrase_v1")
    state["results"]["hard_review"] = {"state": "waiting_review", "reviews": [descriptor]}
    manager._save(state)
    return state, old, descriptor


def test_six_representatives_keep_all_120_sources_and_frozen_legacy_queue(manager):
    state, old, descriptor = generated_hard(manager)
    root = Path(state["project"]).parent
    frozen = {Path(old[key]): digest(old[key]) for key in ("queue", "pool_manifest")}
    previews = read_rows(descriptor["queue"])
    assert old["count"] == 120 and descriptor["count"] == 6
    assert len(read_rows(descriptor["pool_manifest"])) == 120
    assert len({row["text"] for row in previews}) == len({row["voice_family_id"] for row in previews}) == 6
    bank = {voice["family_id"]: voice for voice in read_json(root / "voices.json")["families"]}
    assert sorted(bank[row["voice_family_id"]]["gender"] for row in previews) == ["man"] * 3 + ["woman"] * 3
    assert len({bank[row["voice_family_id"]]["age_group"] for row in previews}) == 5
    assert steps._hard_queue(state["project"], root / ".studio", "one_per_phrase_v1") == descriptor
    assert all(digest(path) == sha for path, sha in frozen.items())
    selection = read_json(root / descriptor["representative_selection"]["path"])
    assert len(selection["representatives"]) == 6
    assert selection["source_manifest"]["sha256"] == old["pool_sha256"]


def test_optional_full_player_is_read_only_and_audio_hash_bound(manager, http_server):
    state, _, descriptor = generated_hard(manager, 2, 3)
    root = Path(state["project"]).parent
    protected = {path: digest(path) for path in (manager._path(state["id"]), root / "approval.json", Path(descriptor["queue"]))}
    status, content, _ = http_server(f"/hard-preview/queue?project={state['id']}")
    payload = json.loads(content)
    assert status == 200 and payload["count"] == len(payload["items"]) == 6
    assert len(json.loads(http_server(f"/review/queue?project={state['id']}&queue=hard")[1])["items"]) == 2
    status, wav, _ = http_server(payload["items"][0]["audio"])
    assert status == 200 and wav[:4] == b"RIFF"
    assert all(digest(path) == sha for path, sha in protected.items())
    assert not (root / ".studio/answers").exists() and not manager.workers.items
    path = root / read_rows(descriptor["pool_manifest"])[0]["path"]
    path.write_bytes(b"changed audio")
    assert http_server(payload["items"][0]["audio"])[0] == 400


def test_saved_six_answers_do_not_claim_the_other_114_were_heard(manager, http_server):
    state, _, descriptor = generated_hard(manager)
    root = Path(state["project"]).parent
    payload = json.loads(http_server(f"/review/queue?project={state['id']}&queue=hard")[1])
    assert "나머지 자동 검사 통과 음원" in payload["policy_text"]
    decisions = {row["sample_id"]: {"decision": "keep", "listened": True} for row in read_rows(descriptor["queue"])}
    manager.save_review(state["id"], "hard", {"decisions": decisions, "policy_approved": True})
    saved = manager._load(state["id"])
    evidence = read_json(saved["inputs"]["review_answers"]["hard"]["path"])
    assert len(evidence["decisions"]) == 6 and evidence["synthetic_review_policy"] == "one_per_phrase_v1"
    assert evidence["representative_selection_sha256"] == descriptor["representative_selection"]["sha256"]
    assert len(read_rows(root / "data/generated/hard/candidates.jsonl")) == 120
    assert not manager.workers.items


def old_release(manager, monkeypatch):
    state, old, _ = generated_hard(manager)
    state["settings"].pop("hard_review_policy")
    state["results"]["hard_review"] = {"state": "waiting_review", "reviews": [old]}
    state["execution_contract"]["settings_sha256"] = identity(state["settings"])
    manager._save(state)
    manager._implementation_id = "new-review-release"
    manager._execution_contract = deepcopy(manager._execution_contract)
    manager._execution_contract["stages"]["hard_review"] = "new-sampled-stage"
    monkeypatch.setattr(studio, "_audited_hard_review_update", lambda a, b: {"kind": "fixture_audit"})
    return state, old


def test_update_preserves_old_queue_and_history_without_starting_work(manager, monkeypatch):
    state, old = old_release(manager, monkeypatch)
    root = Path(state["project"]).parent
    protected = {path: digest(path) for path in (root / "project.json", root / "approval.json", Path(old["queue"]), Path(old["pool_manifest"]))}
    inputs, history = deepcopy(state["inputs"]), deepcopy(state.get("history", []))
    manager.migrate(state["id"])
    after = manager._load(state["id"])
    assert after["status"] == "waiting_review" and after["results"]["hard_review"]["reviews"][0]["count"] == 6
    assert after["inputs"] == inputs and after.get("history", []) == history
    assert all(digest(path) == sha for path, sha in protected.items()) and not manager.workers.items


@pytest.mark.parametrize("answer_kind", ["keep", "discard", "review", "detached"])
def test_saved_old_answers_are_not_bypassed_by_shrinking_the_queue(manager, monkeypatch, answer_kind):
    state, old = old_release(manager, monkeypatch)
    root = Path(state["project"]).parent
    if answer_kind == "detached":
        path = root / ".studio/answers/old.json"
        path.parent.mkdir(exist_ok=True)
        write_json(path, {"queue_sha256": old["queue_sha256"], "decisions": {"fixture": {"decision": "review"}}})
    else:
        state["inputs"]["review_answers"] = {"hard": {"path": "preserved.json", "decision": answer_kind}}
        manager._save(state)
    manager.migrate(state["id"])
    after = manager._load(state["id"])
    assert "hard_review_policy" not in after["settings"]
    assert after["results"]["hard_review"]["reviews"][0] == old
    assert after["inputs"] == state["inputs"] and not manager.workers.items
