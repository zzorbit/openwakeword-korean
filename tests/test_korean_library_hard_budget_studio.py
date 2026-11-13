"""Phrase budgets integrate with Studio without rewriting completed evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio, studio_steps
from korean_openwakeword.common import append_row, digest, identity, read_json
from test_korean_library_studio import create, http_server, manager  # noqa: F401


@pytest.mark.parametrize("preset,per_phrase", [("check", 20), ("trial", 100), ("production", 500)])
def test_each_phrase_gets_the_entire_declared_voice_bank(preset, per_phrase):
    settings = studio_steps.preset_settings(preset)
    assert settings["hard_samples_per_phrase"] == settings["train_references"] == per_phrase
    assert "hard_originals" not in settings
    assert studio_steps.hard_generation_budget(settings, ["보리야", "돌아와"]) == {
        "phrase_count": 2, "samples_per_phrase": per_phrase, "total": 2 * per_phrase}


def test_legacy_total_is_not_reinterpreted_as_a_per_phrase_count():
    settings = studio_steps.preset_settings()
    settings.pop("hard_generation_policy")
    settings.pop("hard_samples_per_phrase")
    settings["hard_originals"] = 100
    assert studio_steps.workflow_plan(settings)
    assert studio_steps.hard_generation_budget(settings, ["보리야", "돌아와"])["total"] == 100


def legacy_waiting_state(manager):
    view = create(manager)
    state = manager._load(view["id"])
    state["settings"].pop("hard_generation_policy")
    state["settings"].pop("hard_samples_per_phrase")
    state["settings"]["hard_originals"] = 100
    state.update(status="waiting_input", auto_continue=True,
                 index=next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "hard_phrases"))
    state["results"]["hard_phrases"] = {"state": "waiting_input", "input_kind": "hard_phrases"}
    state["inputs"]["review_answers"] = {"pilot": {"preserved": "human answer"}}
    state.setdefault("history", [])
    state["execution_contract"]["settings_sha256"] = identity(state["settings"])
    manager._save(state)
    manager._implementation_id = "hard-per-phrase-release"
    manager._execution_contract = deepcopy(manager._execution_contract)
    manager._execution_contract["stages"]["hard_generation"] = "new-hard-stage"
    manager._execution_contract["stages"]["hard_review"] = "new-hard-review-stage"
    return state


def test_audited_budget_update_keeps_audio_answers_and_waiting_input(manager, monkeypatch):
    state = legacy_waiting_state(manager)
    root = Path(state["project"]).parent
    audio = root / "data" / "existing_call.wav"
    audio.write_bytes(b"previous positive audio")
    before = {path: digest(path) for path in (root / "project.json", root / "voices.json", root / "approval.json", audio)}
    inputs, history = deepcopy(state["inputs"]), deepcopy(state["history"])
    monkeypatch.setattr(studio, "_audited_hard_per_phrase_update", lambda old, new: {"kind": "fixture_audit"})
    assert manager._migration_view(state)["state"] == "available"
    view = manager.migrate(state["id"])
    after = manager._load(state["id"])
    assert view["input"]["samples_per_phrase"] == 20
    assert after["settings"]["hard_generation_policy"] == "per_phrase_v1"
    assert "hard_originals" not in after["settings"]
    assert after["status"] == "waiting_input" and after["inputs"] == inputs and after["history"] == history
    assert not manager.workers.items
    assert all(digest(path) == sha for path, sha in before.items())
    receipt = read_json(after["migrations"][-1]["path"])
    assert receipt["audited_hard_per_phrase_update"]["kind"] == "fixture_audit"
    assert read_json(receipt["previous_workflow"]["path"])["settings"]["hard_originals"] == 100


@pytest.mark.parametrize("artifact", ["contract", "audio", "past_stage", "final", "old_voice_count"])
def test_update_never_changes_started_or_frozen_hard_generation(manager, monkeypatch, artifact):
    state = legacy_waiting_state(manager)
    root = Path(state["project"]).parent
    if artifact in ("contract", "audio"):
        path = root / "data/generated/hard" / ("contract.json" if artifact == "contract" else "hard.wav")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"preserved")
    elif artifact == "past_stage":
        state["index"] += 2
    elif artifact == "final":
        (root / "final_lock.json").write_bytes(b"preserved")
    else:
        state["settings"]["train_references"] = 4
        state["execution_contract"]["settings_sha256"] = identity(state["settings"])
    monkeypatch.setattr(studio, "_audited_hard_per_phrase_update", lambda old, new: {"kind": "fixture_audit"})
    assert manager._migration_view(state)["state"] == "blocked"
    assert not manager.workers.items


def test_release_audit_rejects_any_unlisted_computational_change(monkeypatch):
    previous = {"schema_version": 1, "files": {name: "old" for name in ("hard_tts.py", "pool_review.py", "studio_steps.py")},
                "stages": {"hard_generation": "old"}}
    current = {"schema_version": 1, "files": {name: "new" for name in previous["files"]},
               "stages": {"hard_generation": "new"}}
    for contract in (previous, current):
        contract["computation_sha256"] = identity(contract["files"])
    audit = {"target_computation_sha256": current["computation_sha256"],
             "source_stages_sha256": identity(previous["stages"]),
             "target_stages_sha256": identity(current["stages"]), "evidence_sha256": "a" * 64}
    monkeypatch.setattr(studio, "_AUDITED_HARD_PER_PHRASE_UPDATES", {previous["computation_sha256"]: audit})
    assert studio._audited_hard_per_phrase_update(previous, current)
    current["files"]["tts.py"] = "unreviewed"
    current["computation_sha256"] = audit["target_computation_sha256"] = identity(current["files"])
    assert studio._audited_hard_per_phrase_update(previous, current) is None


def stopped_hard_generation(manager, *, status="paused", legacy=False, stage="hard_generation"):
    project = create(manager, navigation_checkpoints=True)
    state = manager._load(project["id"])
    state["settings"]["hard_review_policy"] = "one_per_phrase_v1" if legacy else studio_steps.HARD_SAMPLE_REVIEW_POLICY
    state["plan"] = studio_steps.workflow_plan(state["settings"])
    state["execution_contract"].update(settings_sha256=identity(state["settings"]), plan_sha256=identity(state["plan"]))
    state.update(status=status, index=next(i for i, row in enumerate(state["plan"]) if row["id"] == stage))
    state["navigation_checkpoint"].update(step="hard-review" if stage == "hard_pilot" else "hard-generation", state="running")
    phrases = ["보리야", "돌아와"]
    state["inputs"].update(hard_phrases=phrases, hard_phrases_approved=True,
                          review_answers={"hard": {"path": "previous answer"}, "positive": {"path": "positive answer"}})
    state["results"]["hard_phrases"] = {"state": "completed", "result": {"phrases": phrases}}
    if status == "waiting_input":
        state["results"][stage] = {"state": "waiting_input", "input_kind": "generation_policy"}
    manager._save(state)
    return state


@pytest.mark.parametrize("stage", ["hard_pilot", "hard_generation"])
def test_hard_pause_stops_at_the_saved_candidate_and_resumes_same_scope(manager, stage):
    state = stopped_hard_generation(manager, stage=stage)
    project_id = state["id"]
    manager.start(project_id)
    active = manager._load(project_id)
    marker = Path(active["job"]["directory"]) / "pause.json"
    assert manager.pause(project_id)["status"] == "pausing"
    assert read_json(marker)["task_id"] == active["job"]["task_id"]
    root = Path(state["project"]).parent
    journal = root / "data/generated" / ("hard_pilot" if stage == "hard_pilot" else "hard") / "candidates.jsonl"
    journal.parent.mkdir(parents=True)
    append_row(journal, {"technical_pass": True})
    result_path = manager.workers.items[0].finish({"state": "paused"})
    manager.tick()
    view = manager.describe(manager._load(project_id))
    assert view["stage_id"] == stage and view["status"] == "paused"
    assert view["progress"]["done"] == 1 and view["can_edit_hard_phrases"]
    assert manager._load(project_id)["history"][-1]["sha256"] == digest(result_path)
    manager.start(project_id)
    assert len(manager.workers.items) == 2
    resumed = manager._load(project_id)
    assert resumed["job"]["stage"] == stage and not (Path(resumed["job"]["directory"]) / "pause.json").exists()
    assert digest(journal)


@pytest.mark.parametrize("status", ["paused", "waiting_input", "error", "interrupted"])
def test_unchanged_hard_phrase_edit_restores_original_scope_without_launch(manager, status):
    state = stopped_hard_generation(manager, status=status)
    project_id = state["id"]
    assert manager.describe(state)["can_edit_hard_phrases"]
    opened = manager.edit_hard_phrases(project_id)
    assert opened["status"] == "waiting_input" and opened["stage_id"] == "hard_phrases"
    assert opened["input"]["editing_paused_generation"] and opened["input"]["phrases"] == state["inputs"]["hard_phrases"]
    assert opened["navigation_checkpoint"] == {"policy": "explicit_story_v1", "step": "phrases", "state": "running"}
    assert manager.edit_hard_phrases(project_id)["input"] == opened["input"]
    with pytest.raises(ValueError, match="유사어"):
        manager.submit_input(project_id, {"admissions": ["irrelevant"]})
    restored = manager.submit_input(project_id, {"phrases": opened["input"]["phrases"]})
    assert restored["status"] == status and restored["stage_id"] == "hard_generation"
    assert restored["navigation_checkpoint"] == state["navigation_checkpoint"]
    after = manager._load(project_id)
    assert "hard_phrase_edit" not in after and "hard_phrase_history" not in after
    assert after["results"] == state["results"] and after["inputs"] == state["inputs"]
    assert not manager.workers.items


def test_hard_phrase_http_edit_preserves_old_audio_and_restarts_only_revised_scope(manager, http_server):
    import json

    state = stopped_hard_generation(manager, legacy=True, stage="hard_review")
    root = Path(state["project"]).parent
    audio = root / "data/generated/hard/previous.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"old clip")
    old_sha256 = digest(audio)
    state["last_job"] = {"stage": "hard_generation"}
    state["navigation_checkpoint"]["state"] = "awaiting_next"
    state["results"]["hard_generation"] = {"state": "completed", "result": {"technical_pass": 40}}
    manager._save(state)
    prefix = f"/api/projects/{state['id']}"
    headers = {"X-Koww-Token": manager.token}
    assert http_server(prefix + "/edit_hard_phrases", "POST", {})[0] == 403
    code, content, _ = http_server(prefix + "/edit_hard_phrases", "POST", {}, headers=headers)
    assert code == 200 and json.loads(content)["input"]["editing_paused_generation"]
    assert not manager.workers.items
    updated = ["보리야", "기다려"]
    code, content, _ = http_server(prefix + "/input", "POST", {"phrases": updated}, headers=headers)
    assert code == 200 and json.loads(content)["status"] == "running"
    request = read_json(manager.workers.items[0].request_path)
    assert request["stage"] == "hard_phrases"
    assert request["context"]["inputs"]["hard_phrases"] == updated
    assert len(request["context"]["inputs"]["hard_phrase_revision"]) == 32
    assert request["context"]["inputs"]["review_answers"] == {"positive": {"path": "positive answer"}}
    assert "hard_generation" not in request["context"]["results"]
    archived = manager._load(state["id"])["hard_phrase_history"][-1]
    assert archived["review_answers"] == {"path": "previous answer"}
    assert archived["results"]["hard_generation"] == state["results"]["hard_generation"]
    assert digest(audio) == old_sha256


@pytest.mark.parametrize("blocked", ["running", "pausing", "admitted", "final", "features"])
def test_hard_phrase_edit_rejects_running_or_admitted_projects(manager, blocked):
    state = stopped_hard_generation(manager, stage="features" if blocked == "features" else "hard_generation")
    root = Path(state["project"]).parent
    if blocked in ("running", "pausing"):
        state["status"] = blocked
    elif blocked == "admitted":
        from korean_openwakeword.project import load_project
        _, config = load_project(state["project"])
        (root / config["manifests"]["hard_negative"]).write_bytes(b"admitted source")
    elif blocked == "final":
        (root / "final_lock.json").write_bytes(b"locked final")
    manager._save(state)
    assert not manager.describe(state).get("can_edit_hard_phrases")
    with pytest.raises(ValueError, match="일시정지"):
        manager.edit_hard_phrases(state["id"])
