"""Current two-stage voice pilots preserve coverage, legacy runs, and human gates."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import pilot, studio, studio_steps as steps
from korean_openwakeword.common import digest, identity, now, read_json, read_rows, write_immutable_rows, write_json
from korean_openwakeword.project import create_project
from test_korean_library_studio import create, manager  # noqa: F401


def make_project(tmp_path, preset="check"):
    settings = steps.preset_settings(preset)
    project = create_project(tmp_path / preset, "보라야", "bora", positive_originals=settings["originals"],
        augmentation_rounds=settings["rounds"], updates=settings["updates"],
        generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"],
        train_voice_families=settings["train_references"], validation_voice_families=settings["validation_references"])
    return project, settings


def run(project, settings, stage, context=None):
    return steps.run_stage(project, project.parent.parent, settings, stage, context)


def pilot_rows(project, count, split="train"):
    root = project.parent
    config = read_json(project)
    families = [row for row in read_json(root / "voices.json")["families"] if row["split"] == split]
    rows = []
    for index in range(count):
        path = root / "data/pilot" / split / f"fixture_{index}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((f"synthetic {split} pilot fixture " + str(index)).encode())
        rows.append({"sample_id": f"pilot_{split}_{index}", "slot": index, "path": str(path.relative_to(root)),
                     "sha256": digest(path), "voice_family_id": families[index % len(families)]["family_id"],
                     "split": split, "qc": {"passed": True}, "generation_method": "clone",
                     "tts_settings_id": identity(config["tts"]), "text": "보라야"})
    return rows


@pytest.mark.parametrize("preset,expected", [("check", 20), ("trial", 100), ("production", 500)])
def test_default_pronunciation_pilot_covers_both_voice_banks_before_and_after_generation(preset, expected):
    settings = steps.preset_settings(preset)
    assert settings["pilot_review_policy"] == pilot.PREPOST_PILOT_POLICY
    assert settings["train_references"] == expected
    plan = {stage["id"]: stage for stage in steps.workflow_plan(settings)}
    assert plan["pilot"]["kind"] == "task"
    assert plan["pilot_review"]["kind"] == "review"
    assert plan["positive_review"]["kind"] == "review" and len(plan) == 20
    assert list(plan).index("pilot_review") < list(plan).index("positives") < list(plan).index("positive_review")
    assert settings["train_references"] + settings["validation_references"] == expected + 30
    assert settings["validation_references"] == 30 and settings["final_locked"] is True


def test_legacy_pilot_keeps_frozen_twenty_four_generation_and_review(tmp_path, monkeypatch):
    from korean_openwakeword import tts_runtime
    project, settings = make_project(tmp_path)
    settings.pop("pilot_review_policy")
    rows = pilot_rows(project, 24)
    queue = project.parent / "data/pilot/train/candidates.jsonl"
    write_immutable_rows(queue, rows)
    calls = []

    def generate(*args):
        calls.append(args)
        return {"full_target_reached": True, "technical_pass": 24, "review_queue": str(queue)}

    monkeypatch.setattr(tts_runtime, "generate_with_runtime", generate)
    before = (project.parent / "approval.json").read_bytes()
    result = run(project, settings, "pilot")
    assert calls == [(project, "train", 24, True, "clone")]
    assert len(read_rows(result["result"]["queue"])) == 24
    assert run(project, settings, "pilot_review")["state"] == "waiting_review"
    assert (project.parent / "approval.json").read_bytes() == before


@pytest.mark.parametrize("preset,expected", [("check", 20), ("trial", 100), ("production", 500)])
def test_current_pilot_stage_prepares_one_call_per_voice_in_both_banks_and_keeps_human_review(tmp_path, monkeypatch, preset, expected):
    from korean_openwakeword import pilot, tts_runtime
    project, settings = make_project(tmp_path, preset)
    rows = pilot_rows(project, expected) + pilot_rows(project, 30, "validation")
    queue = project.parent / "data/call_pronunciation/preflight_fixture.jsonl"
    queue.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_rows(queue, rows)
    before = {path: digest(path) for path in (project, project.parent / "voices.json", project.parent / "approval.json", queue)}
    calls = []

    def prepare(path, supplied_settings, *, prior_rounds):
        calls.append((path, supplied_settings, prior_rounds))
        return {"target": expected + 30, "technical_pass": expected + 30, "full_target_reached": True,
                "review_queue": str(queue), "split_counts": {
                    "train": {"target": expected, "technical_pass": expected},
                    "validation": {"target": 30, "technical_pass": 30}},
                "bindings": {"review_queue": {"path": str(queue.relative_to(project.parent)), "sha256": digest(queue)}}}

    monkeypatch.setattr(pilot, "prepare_prepost_pilot", prepare)
    monkeypatch.setattr(pilot, "prepare_pilot", lambda *args, **kwargs: pytest.fail("Current policy cannot use the legacy helper"))
    monkeypatch.setattr(tts_runtime, "generate_with_runtime", lambda *args: pytest.fail("Current stage delegates to the two-stage helper"))
    result = run(project, settings, "pilot")
    assert result["state"] == "completed" and calls == [(project, settings, [])]
    selected = read_rows(result["result"]["queue"])
    assert len(selected) == len({row["voice_family_id"] for row in selected}) == expected + 30
    assert sum(row["split"] == "train" for row in selected) == expected
    assert sum(row["split"] == "validation" for row in selected) == 30
    assert all(digest(path) == sha for path, sha in before.items())
    pending = run(project, settings, "pilot_review", {"results": {"pilot": result}})
    assert pending["state"] == "waiting_review" and pending["reviews"][0]["count"] == expected + 30
    assert read_json(project.parent / "approval.json")["generation_policy_approved"] is False


def test_incomplete_voice_coverage_cannot_publish_or_approve_a_pilot(tmp_path, monkeypatch):
    from korean_openwakeword import pilot
    project, settings = make_project(tmp_path)
    monkeypatch.setattr(pilot, "prepare_prepost_pilot", lambda *args, **kwargs: {
        "full_target_reached": False, "technical_pass": 49, "target": 50, "missing_reference_families": ["missing"]})
    before = (project.parent / "approval.json").read_bytes()
    result = run(project, settings, "pilot")
    assert result["state"] == "waiting_input"
    assert result["result"]["target"] == 50 and result["result"]["technical_pass"] == 49
    assert "queue" not in result["result"]
    assert not (project.parent / ".studio/pilot_queue.jsonl").exists()
    assert (project.parent / "approval.json").read_bytes() == before


def test_explicit_pilot_retry_reuses_passed_clips_and_spends_one_round_per_failed_bank(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    from korean_openwakeword import tts, tts_runtime
    from test_call_generation_recovery import RecoveryModel, legacy_run
    from test_korean_library_tts_journal import cpu_seed

    project, _, train, _, base = legacy_run(tmp_path, count=3, references_count=3,
                                            scope_kind="preflight", revision=identity([]))
    root = project.parent
    validation = []
    for number in range(2):
        audio = root / f"eval_{number}.wav"
        sf.write(audio, .11 * np.sin(np.arange(8000) / (25 + number)), 16000, subtype="FLOAT")
        validation.append(dict(train[0], voice_family_id=f"eval_{number}", split="validation", text="평가 기준 음성입니다.",
                               path=str(audio.relative_to(root)), sha256=digest(audio)))

    class SelectiveModel(RecoveryModel):
        fail_validation = False

        def generate_voice_clone(self, **options):
            if self.fail_validation and options["voice_clone_prompt"]["ref_text"] == validation[0]["text"]:
                self.calls += 1
                return [np.zeros(9600)], 16000
            return super().generate_voice_clone(**options)

    model = SelectiveModel(fail=True)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    monkeypatch.setattr(tts_runtime, "CloneRuntime", lambda *args: model)
    monkeypatch.setattr(pilot, "_preflight_banks", lambda *_: {"train": train, "validation": validation})
    settings = {"train_references": 3, "validation_references": 2}
    first = pilot.prepare_prepost_pilot(project, settings)
    assert first["technical_pass"] == 2 and first["target"] == 5
    base_journal = base / "candidates.jsonl"
    original_wavs = {row["voice_family_id"]: row["sha256"] for row in read_rows(base_journal) if row["qc"]["passed"]}
    base_bytes = {path: path.read_bytes() for path in base.iterdir() if path.is_file()}
    approval = (root / "approval.json").read_bytes()

    retry = pilot.prepare_prepost_pilot(project, settings, quality_retry_rounds=1)
    assert retry["technical_pass"] == 2 and "review_queue" not in retry
    assert retry["exhausted_slots"][0]["label"] == first["exhausted_slots"][0]["label"]
    assert retry["exhausted_slots"][0]["slot"] == 2
    contracts = list((root / "data/call_pronunciation").glob("*/train/contract.json"))
    retry_path, = (path for path in contracts if path.parent != base)
    retry_scope = read_json(retry_path)["staged_scope"]
    assert retry_scope["quality_retry_round"] == 1 and retry_scope["references"] == [train[2]]
    assert retry_scope["excluded_sha256"] == sorted(original_wavs.values())
    retry_bytes = (retry_path.parent / "candidates.jsonl").read_bytes()
    calls = model.calls
    assert pilot.prepare_prepost_pilot(project, settings, quality_retry_rounds=1) == retry
    assert model.calls == calls

    model.fail = False
    model.fail_validation = True
    evaluation_failed = pilot.prepare_prepost_pilot(project, settings, quality_retry_rounds=2)
    assert evaluation_failed["technical_pass"] == 3
    assert {row["split"] for row in evaluation_failed["exhausted_slots"]} == {"validation"}
    validation_scopes = [read_json(path)["staged_scope"] for path in (root / "data/call_pronunciation").glob("*/validation/contract.json")]
    assert len(validation_scopes) == 1 and "quality_retry_round" not in validation_scopes[0]

    model.fail_validation = False
    complete = pilot.prepare_prepost_pilot(project, settings, quality_retry_rounds=3)
    assert complete["full_target_reached"] is True and not complete["exhausted_slots"]
    queue = read_rows(complete["review_queue"])
    assert [row["preflight_reference"] for row in queue] == train + validation
    assert [row["voice_number"] for row in queue] == [1, 2, 3, 1, 2]
    assert {row["voice_family_id"]: row["sha256"] for row in queue[:2]} == original_wavs
    assert len({row["sha256"] for row in queue}) == 5
    assert all(path.read_bytes() == content for path, content in base_bytes.items())
    assert (retry_path.parent / "candidates.jsonl").read_bytes() == retry_bytes
    assert (root / "approval.json").read_bytes() == approval
    calls = model.calls
    assert pilot.prepare_prepost_pilot(project, settings, quality_retry_rounds=3) == complete
    assert model.calls == calls


@pytest.mark.parametrize("preset,expected", [("check", 20), ("trial", 100), ("production", 500)])
def test_legacy_per_training_voice_progress_does_not_count_duplicate_candidates_as_extra_voices(manager, preset, expected):
    view = create(manager, preset=preset)
    manager.start(view["id"])
    state = manager._load(view["id"])
    # This journal layout and family-based retry count belong to the saved v1 workflow.
    state["settings"]["pilot_review_policy"] = pilot.TRAIN_PILOT_POLICY
    state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot")
    state["job"]["stage"] = "pilot"
    manager._save(state)
    root = Path(state["project"]).parent
    rows = pilot_rows(Path(state["project"]), 3)
    # A retry/second rendition of one voice must not imply a fourth reviewed voice.
    duplicate = dict(rows[0], sample_id="another_attempt", slot=expected + 1)
    write_immutable_rows(root / "data/pilot/train/candidates.jsonl", rows + [duplicate])
    before = {path: path.read_bytes() for path in (root / "approval.json", manager._path(view["id"]))}
    progress = manager.describe(state)["progress"]
    assert progress["total"] == expected
    assert progress["done"] == 3
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(manager.workers.items) == 1


@pytest.mark.parametrize("preset,expected", [("check", 20), ("trial", 100), ("production", 500)])
def test_current_pilot_progress_counts_both_banks_only_in_the_current_preflight_revision(manager, preset, expected):
    view = create(manager, preset=preset)
    manager.start(view["id"])
    state = manager._load(view["id"])
    state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot")
    state["job"]["stage"] = "pilot"
    manager._save(state)
    root = Path(state["project"]).parent
    revision = identity([])
    for name, kind, scope_revision, count in (
            ("current", "preflight", revision, 3),
            ("previous", "preflight", "old-review-revision", 8),
            ("production", "production", revision, 9)):
        for split in ("train", "validation"):
            folder = root / "data/call_pronunciation" / name / split
            write_json(folder / "contract.json", {"staged_scope": {"kind": kind, "revision": scope_revision}})
            rows = pilot_rows(Path(state["project"]), count, split)
            # Failed attempts never increase successful sample coverage.
            write_immutable_rows(folder / "candidates.jsonl", rows + [{**rows[0], "qc": {"passed": False}}])
    before = {path: path.read_bytes() for path in (root / "approval.json", manager._path(view["id"]))}
    progress = manager.describe(state)["progress"]
    assert progress["total"] == expected + 30 and progress["done"] == 6
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(manager.workers.items) == 1


def test_saved_legacy_pilot_progress_remains_twenty_four(manager):
    view = create(manager)
    manager.start(view["id"])
    state = manager._load(view["id"])
    state["settings"].pop("pilot_review_policy")
    state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot")
    state["job"]["stage"] = "pilot"
    manager._save(state)
    progress = manager.describe(state)["progress"]
    assert progress["total"] == 24 and progress["done"] == 0


def test_unknown_pilot_review_policy_fails_before_generation(tmp_path, monkeypatch):
    from korean_openwakeword import tts_runtime
    project, settings = make_project(tmp_path)
    settings["pilot_review_policy"] = "unreviewed-policy"
    monkeypatch.setattr(tts_runtime, "generate_with_runtime", lambda *args: pytest.fail("Invalid policy cannot generate"))
    with pytest.raises(ValueError, match="[Pp]ilot|파일럿|발음"):
        steps.workflow_plan(settings)
    with pytest.raises(ValueError, match="[Pp]ilot|파일럿|발음"):
        run(project, settings, "pilot")


def pending_legacy_pilot(manager):
    view = create(manager)
    state = manager._load(view["id"])
    root, project = Path(state["project"]).parent, Path(state["project"])
    state["settings"].pop("pilot_review_policy")
    state["plan"] = steps.workflow_plan(state["settings"])
    state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot_review")
    rows = pilot_rows(project, 24)
    queue = root / ".studio/pilot_queue.jsonl"
    write_immutable_rows(queue, rows)
    journal = root / "data/pilot/train/candidates.jsonl"
    write_immutable_rows(journal, rows)
    answers = root / ".studio/answers/legacy-pilot.json"
    write_json(answers, {"queue_sha256": digest(queue), "decisions": {
        rows[0]["sample_id"]: {"decision": "keep", "listened": True}}})
    receipt = root / ".studio/jobs/legacy-pilot-result.json"
    write_json(receipt, {"started_at": now(), "completed_at": now(), "artifact_bindings": [
        {"path": str(queue), "sha256": digest(queue)}, {"path": str(journal), "sha256": digest(journal)}]})
    state.update(status="waiting_review", auto_continue=True,
        results={"pilot": {"state": "completed", "result": {"queue": str(queue), "technical_pass": 24}},
                 "pilot_review": {"state": "waiting_review", "reviews": []}},
        inputs={"review_answers": {"pilot": {"path": str(answers), "sha256": digest(answers)}}},
        history=[{"stage": "pilot", "result": str(receipt), "sha256": digest(receipt)}])
    state["execution_contract"].update(settings_sha256=identity(state["settings"]), plan_sha256=identity(state["plan"]))
    manager._save(state)
    return state, queue


def mock_audited_pilot_release(manager, monkeypatch):
    manager._implementation_id = "one-pilot-per-training-voice-release"
    manager._execution_contract = deepcopy(manager._execution_contract)
    for stage in ("pilot", "pilot_review", "results"):
        manager._execution_contract["stages"][stage] = "audited-new-" + stage
    monkeypatch.setattr(studio, "_audited_pilot_update", lambda old, new: {"kind": "fixture_pilot_audit"})


def test_pilot_migration_preserves_old_twenty_four_answers_history_and_approval_then_prepares_immutable_twenty(manager, monkeypatch):
    from korean_openwakeword import pilot
    state, old_queue = pending_legacy_pilot(manager)
    root, project = Path(state["project"]).parent, Path(state["project"])
    protected = [project, root / "voices.json", root / "approval.json", old_queue,
                 root / "data/pilot/train/candidates.jsonl", root / ".studio/answers/legacy-pilot.json",
                 root / ".studio/jobs/legacy-pilot-result.json"] + list((root / "data/pilot/train").glob("*.wav"))
    before = {path: digest(path) for path in protected}
    old_state = deepcopy(state)
    mock_audited_pilot_release(manager, monkeypatch)
    assert manager.describe(state)["migration"]["state"] == "available"
    manager.migrate(state["id"])
    after = manager._load(state["id"])
    assert after["settings"]["pilot_review_policy"] == "one_per_training_voice_v1"
    assert after["plan"][after["index"]]["id"] == "pilot"
    assert after["status"] == "paused" and after["auto_continue"] is False
    assert after["inputs"] == old_state["inputs"] and after["history"] == old_state["history"]
    assert after["results"] == old_state["results"]
    assert all(digest(path) == sha for path, sha in before.items())
    assert not manager.workers.items
    migration = read_json(after["migrations"][-1]["path"])
    assert migration["audited_pilot_update"]["kind"] == "fixture_pilot_audit"
    selected = read_rows(old_queue)[:20]
    new_queue = root / "data/pilot/train/review_one_per_voice_fixture.jsonl"
    write_immutable_rows(new_queue, selected)
    monkeypatch.setattr(pilot, "prepare_pilot", lambda *args, **kwargs: {
        "full_target_reached": True, "technical_pass": 20, "review_queue": str(new_queue),
        "selected_reference_count": 20, "reused_existing_count": 20})
    result = run(project, after["settings"], "pilot", {"results": after["results"], "inputs": after["inputs"]})
    assert Path(result["result"]["queue"]) == new_queue and new_queue != old_queue
    assert len(read_rows(old_queue)) == 24 and len(read_rows(new_queue)) == 20
    pending = run(project, after["settings"], "pilot_review", {"results": {"pilot": result}, "inputs": after["inputs"]})
    assert pending["state"] == "waiting_review"
    assert all(digest(path) == sha for path, sha in before.items())
    manager.migrate(state["id"])
    assert len(manager._load(state["id"])["migrations"]) == 1


@pytest.mark.parametrize("freeze", ["bulk_calls", "trained", "final", "approved", "past_pilot", "live_worker"])
def test_pilot_migration_blocks_live_work_bulk_calls_approved_or_locked_runs(manager, monkeypatch, freeze):
    state, _ = pending_legacy_pilot(manager)
    root = Path(state["project"]).parent
    if freeze in ("bulk_calls", "trained"):
        path = root / ("data/generated/train" if freeze == "bulk_calls" else "runs") / "already-created.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve")
    elif freeze == "final":
        write_json(root / "final_lock.json", {"locked": True})
    elif freeze == "approved":
        approval = read_json(root / "approval.json")
        approval["generation_policy_approved"] = True
        write_json(root / "approval.json", approval)
    elif freeze == "past_pilot":
        state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "positives")
    else:
        state["job"] = {"pid": 123, "creation_identity": "live"}
        monkeypatch.setattr(manager, "_job_alive", lambda job: bool(job))
    assert manager._can_update_pilot(state) is False
    mock_audited_pilot_release(manager, monkeypatch)
    manager._save(state)
    before = manager._path(state["id"]).read_bytes()
    assert manager._migration_view(state)["state"] == "blocked"
    with pytest.raises(ValueError):
        manager.migrate(state["id"])
    assert manager._path(state["id"]).read_bytes() == before
    assert not manager.workers.items


def test_pilot_release_audit_requires_exact_generation_preserving_file_and_stage_pins(monkeypatch):
    previous = {"schema_version": 1, "files": {"studio_steps.py": "old", "tts.py": "same"},
                "stages": {"pilot": "old", "pilot_review": "old"}}
    current = {"schema_version": 1, "files": {"studio_steps.py": "new", "pilot.py": "new", "tts.py": "same"},
               "stages": {"pilot": "new", "pilot_review": "new"}}
    for contract in (previous, current):
        contract["computation_sha256"] = identity(contract["files"])
    audit = {"source_stages_sha256": identity(previous["stages"]), "target_stages_sha256": identity(current["stages"]),
             "target_computation_sha256": current["computation_sha256"], "evidence_sha256": "b" * 64}
    monkeypatch.setattr(studio, "_AUDITED_PILOT_UPDATES", {previous["computation_sha256"]: audit})
    assert studio._audited_pilot_update(previous, current)["changed_files"] == ["pilot.py", "studio_steps.py"]
    source = deepcopy(previous)
    source["files"]["tts.py"] = "incorrect source file"
    assert studio._audited_pilot_update(source, current) is None
    changed = deepcopy(current)
    changed["files"]["tts.py"] = "unreviewed generation change"
    changed["computation_sha256"] = identity(changed["files"])
    audit["target_computation_sha256"] = changed["computation_sha256"]
    assert studio._audited_pilot_update(previous, changed) is None
    audit["target_computation_sha256"] = current["computation_sha256"]
    changed = deepcopy(current)
    changed["stages"]["pilot_review"] = "unreviewed stage"
    assert studio._audited_pilot_update(previous, changed) is None
    audit["evidence_sha256"] = "unfinished"
    assert studio._audited_pilot_update(previous, current) is None
