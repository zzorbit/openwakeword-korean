"""Phrase budgets integrate with Studio without rewriting completed evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio, studio_steps
from korean_openwakeword.common import digest, identity, read_json
from test_korean_library_studio import create, manager  # noqa: F401


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
