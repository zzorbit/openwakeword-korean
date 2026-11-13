"""Prepared clone inputs recover only failed slots under a bounded audit trail."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import tts
from korean_openwakeword.common import append_row, digest, identity, read_json, read_rows, write_json
from korean_openwakeword.pilot import _passing_slots
from test_korean_library_tts_journal import approved_project, cpu_seed


def legacy_run(tmp_path, *, count=50, scoped=True, pilot=True, unknown=False,
               references_count=2, scope_kind=None, revision="original_round"):
    project = approved_project(tmp_path / "word", "clone", count=count)
    root, config = project.parent, read_json(project)
    references = read_rows(root / "data/references/train/references.jsonl")
    for index in range(2, references_count):
        extra = dict(references[0], voice_family_id=f"voice_{index}")
        audio = root / "data/references/train" / f"voice_{index}.wav"
        sf.write(audio, .1 * np.sin(np.arange(8000) / (20 + index)), 16000, subtype="FLOAT")
        extra.update(path=str(audio.relative_to(root)), sha256=digest(audio))
        references.append(extra)
    reference = references[-1]
    path = root / reference["path"]
    sf.write(path, np.concatenate([.1 * np.sin(np.arange(16000) / 20), np.zeros(32000)]), 16000, subtype="FLOAT")
    reference["sha256"] = digest(path)
    (root / "data/references/train/references.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in references), encoding="utf-8")
    scope = {"kind": scope_kind or ("preflight" if count == 50 else "production"), "revision": revision,
             "references": references, "excluded_sha256": []} if scoped else None
    output, scope_id = tts.call_generation_location(root, "train", scope) if scoped else (
        root / "data" / ("pilot" if pilot else "generated") / "train", "")
    output.mkdir(parents=True, exist_ok=True)
    current = tts.call_generation_contract(config, "train", references, pilot, "clone", scope)
    contract = {k: v for k, v in current.items() if k != "clone_input_policy"}
    contract["implementation_sha256"] = "unknown" if unknown else tts.LEGACY_CALL_IMPLEMENTATION
    write_json(output / "contract.json", contract)
    for slot in range(count):
        for attempt in range(3 if slot == count - 1 else 1):
            sample_id = f"train_{slot:06d}_attempt{attempt}"
            if scope_id:
                sample_id = scope_id + "_" + sample_id
            path = output / (sample_id + ".wav")
            passed = slot != count - 1
            wave = .1 * np.sin(np.arange(9600) / (20 + slot)) if passed else np.zeros(9600)
            sf.write(path, wave, 16000, subtype="PCM_16")
            row = {"sample_id": sample_id, "slot": slot, "source_id": "qwen:fixture:" + sample_id,
                "path": str(path.relative_to(root)), "sha256": digest(path), "split": "train",
                "voice_family_id": references[slot % len(references)]["voice_family_id"],
                "text": config["wakeword"], "provenance": "qwen_tts", "human_reference_used": False,
                "qc": {"passed": passed, "reasons": [] if passed else ["near_silence"]},
                "axes": {key: "fixture" for key in ("voice_family", "style", "speed", "intonation",
                                                    "vocal_effort", "call_length", "text_profile")},
                "generation_contract_id": identity(contract), "generation_method": "clone",
                "tts_settings_id": identity(config["tts"]), "individual_human_listening": False}
            if scope:
                row["voice_reference_sha256"] = references[slot % len(references)]["sha256"]
            append_row(output / "candidates.jsonl", row)
    return project, config, references, scope, output


class RecoveryModel:
    def __init__(self, *, fail=False, interrupt=False):
        self.prompts, self.calls = [], 0
        self.fail, self.interrupt = fail, interrupt

    def create_voice_clone_prompt(self, **options):
        self.prompts.append(options)
        assert sf.info(options["ref_audio"]).duration < 1.3
        return options

    def generate_voice_clone(self, **options):
        self.calls += 1
        if self.interrupt:
            raise RuntimeError("interrupted recovery")
        if self.fail:
            wave = np.zeros(9600)
        else:
            wave = .13 * np.sin(np.arange(9600) / 22) + np.random.normal(0, .002, 9600)
        return [wave], 16000


def test_production_keeps_legacy_attempts_and_runs_both_banks_before_recovery(tmp_path, monkeypatch):
    from korean_openwakeword import clone_reference, pilot

    project = approved_project(tmp_path / "word", "clone", count=2)
    root, config = project.parent, read_json(project)
    train = read_rows(root / "data/references/train/references.jsonl")
    legacy = train[0]
    source = root / legacy["path"]
    tone = .11 * np.sin(np.arange(8000) / 23)
    sf.write(source, np.concatenate([tone, np.full(5 * 16000, .00001), tone]), 16000, subtype="FLOAT")
    legacy["sha256"] = digest(source)
    validation = root / "validation.wav"
    sf.write(validation, .15 * np.sin(np.arange(9600) / 28), 16000, subtype="FLOAT")
    banks = {"train": train, "validation": [dict(train[1], voice_family_id="eval", split="validation",
        path=validation.name, sha256=digest(validation))]}
    revision = "existing-review"
    scope = {"kind": "production", "revision": revision, "references": train, "excluded_sha256": []}
    train_dir, _ = tts.call_generation_location(root, "train", scope)

    # Emulate a pre-update worker with six recorded failures and no validation.
    old_prepare = clone_reference.prepare_clone_reference
    monkeypatch.setattr(clone_reference, "prepare_clone_reference", lambda root, row, **_: old_prepare(root, row))
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    old = RecoveryModel(fail=True)
    old.create_voice_clone_prompt = lambda **kw: kw
    result = tts.generate(project, "train", 1, True, "clone", model=old, staged_scope=scope)
    assert result["exhausted_slots"][0]["attempts"] == 6
    journal = train_dir / "candidates.jsonl"
    historical = journal.read_bytes()
    contract = (train_dir / "contract.json").read_bytes()
    monkeypatch.setattr(clone_reference, "prepare_clone_reference", old_prepare)

    events = []

    class Updated(RecoveryModel):
        def create_voice_clone_prompt(self, **kw):
            duration = sf.info(kw["ref_audio"]).duration
            assert duration < 2
            return kw

        def generate_voice_clone(self, **kw):
            events.append(Path(kw["voice_clone_prompt"]["ref_audio"]).stem)
            return super().generate_voice_clone(**kw)

    scopes, directories, summaries = pilot._generate_production_banks(
        project, root, banks, {"train": 2, "validation": 2}, revision, set(), Updated())
    assert events == ["voice_1", "validation", "validation", "reference"]
    assert directories["train"] == train_dir and scopes["train"] == scope
    assert all(item["full_target_reached"] for item in summaries.values())
    assert journal.read_bytes().startswith(historical)
    assert (train_dir / "contract.json").read_bytes() == contract
    rows, passing = _passing_slots(root, config, "train", train, train_dir, scope)
    assert len(rows) == 8 and passing[0]["attempt"] == 6
    assert passing[0]["retry_phase"] == "conditioned_recovery"
    assert passing[0]["clone_reference_preparation"]["policy"] == clone_reference.INTERNAL_SILENCE_POLICY

    events.clear()
    pilot._generate_production_banks(project, root, banks, {"train": 2, "validation": 2}, revision, set(), Updated())
    assert events == []


def test_invalid_model_outputs_consume_budget_without_fake_waveforms(tmp_path, monkeypatch):
    project, config, refs, scope, output = legacy_run(tmp_path, count=6)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    class Invalid(RecoveryModel):
        def generate_voice_clone(self, **options):
            self.calls += 1
            return [np.array([np.nan])], 16000
    model = Invalid()
    result = tts.generate(project, "train", 6, True, "clone", model=model, staged_scope=scope)
    assert model.calls == 3 and result["exhausted_slots"][0]["attempts"] == 6
    rows = read_rows(output / "candidates.jsonl")
    assert all(row["path"] is None and row["sha256"] is None and row["qc"]["passed"] is False
               for row in rows[-3:])
    assert not list(output.glob("*attempt[345].wav"))
    reviewable = read_rows(result["review_queue"])
    assert len(reviewable) == 8 and all(row.get("path") and row.get("sha256") for row in reviewable)
    assert len([row for row in reviewable if row["qc"]["passed"] is False]) == 3
    assert result["attempt_journal"] == str(output / "candidates.jsonl")
    assert len(_passing_slots(project.parent, config, "train", refs, output, scope)[1]) == 5
    monkeypatch.setattr(tts, "_load_model", lambda *args: pytest.fail("invalid-output exhaustion loaded a model"))
    assert tts.generate(project, "train", 6, True, "clone", staged_scope=scope)["exhausted_slots"]


@pytest.mark.parametrize("scoped,pilot,count", [(True, True, 50), (True, True, 100), (False, True, 6), (False, False, 6)])
def test_legacy_passes_and_contract_remain_exact_only_failed_slot_recovers(tmp_path, monkeypatch, scoped, pilot, count):
    project, config, refs, scope, output = legacy_run(tmp_path, count=count, scoped=scoped, pilot=pilot,
                                                    references_count=50 if count == 50 else 2)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    original_contract = (output / "contract.json").read_bytes()
    write_json(output / "summary.json", {"historical": "preserve exact old summary"})
    original_summary = (output / "summary.json").read_bytes()
    original_journal = (output / "candidates.jsonl").read_bytes()
    original_rows = read_rows(output / "candidates.jsonl")
    waves = {row["path"]: digest(project.parent / row["path"]) for row in original_rows}
    source = (project.parent / refs[-1]["path"]).read_bytes()
    model = RecoveryModel()
    result = tts.generate(project, "train", count, pilot, "clone", model=model, staged_scope=scope)
    assert result["technical_pass"] == count and result["full_target_reached"]
    assert result["exhausted_slots"] == [] and result["human_listened"] == 0
    assert model.calls == len(model.prompts) == 1
    assert (output / "contract.json").read_bytes() == original_contract
    assert (output / "summary.json").read_bytes() == original_summary
    assert len(list(output.glob("summary_recovery_*.json"))) == 1
    assert (output / "candidates.jsonl").read_bytes().startswith(original_journal)
    assert (project.parent / refs[-1]["path"]).read_bytes() == source
    assert all(digest(project.parent / path) == sha for path, sha in waves.items())
    rows = read_rows(output / "candidates.jsonl")
    latest = rows[-1]
    assert latest["slot"] == count - 1 and latest["attempt"] == 3 and latest["retry_phase"] == "recovery"
    assert latest["generation_contract_id"] != original_rows[0]["generation_contract_id"]
    assert latest["generation_update"]["sha256"] == digest(output / "generation_update.json")
    assert latest["clone_reference_preparation"]["trimmed"] is True
    assert latest["individual_human_listening"] is False
    if pilot:
        validated, passing = _passing_slots(project.parent, config, "train", refs, output, scope)
        assert len(validated) == count + 3 and len(passing) == count
    before = (output / "candidates.jsonl").read_bytes()
    monkeypatch.setattr(tts, "_load_model", lambda *args: pytest.fail("completed run loaded a model"))
    assert tts.generate(project, "train", count, pilot, "clone", staged_scope=scope)["full_target_reached"]
    assert (output / "candidates.jsonl").read_bytes() == before


def test_six_failures_are_terminal_and_resume_does_not_load_or_retry(tmp_path, monkeypatch):
    project, _, _, scope, output = legacy_run(tmp_path, count=6)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    model = RecoveryModel(fail=True)
    result = tts.generate(project, "train", 6, True, "clone", model=model, staged_scope=scope)
    assert model.calls == 3 and result["technical_pass"] == 5
    exhausted = result["exhausted_slots"]
    assert len(exhausted) == 1 and exhausted[0]["slot"] == 5
    assert exhausted[0]["attempts"] == exhausted[0]["maximum_attempts"] == 6
    assert exhausted[0]["automatic_retry_available"] is False
    assert len(exhausted[0]["failures"]) == 6
    before = {str(p): digest(p) for p in output.iterdir() if p.is_file()}
    monkeypatch.setattr(tts, "_load_model", lambda *args: pytest.fail("exhausted run loaded a model"))
    monkeypatch.setattr(tts.clone_reference, "prepare_clone_reference", lambda *args: pytest.fail("exhausted run derived input"))
    resumed = tts.generate(project, "train", 6, True, "clone", staged_scope=scope)
    assert resumed["exhausted_slots"] == exhausted
    assert all(digest(p) == sha for p, sha in before.items())


def test_interrupted_recovery_resumes_same_seed_and_prepared_input(tmp_path, monkeypatch):
    project, _, _, scope, output = legacy_run(tmp_path, count=6)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    with pytest.raises(RuntimeError, match="interrupted"):
        tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(interrupt=True), staged_scope=scope)
    receipt = (output / "generation_update.json").read_bytes()
    assert len(read_rows(output / "candidates.jsonl")) == 8
    assert tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(), staged_scope=scope)["full_target_reached"]
    assert (output / "generation_update.json").read_bytes() == receipt
    row = read_rows(output / "candidates.jsonl")[-1]
    scope_id = identity(scope)[:16]
    assert row["seed"] == int(identity([read_json(project)["tts"]["seed"], "train", 5, 3, scope_id])[:8], 16)


@pytest.mark.parametrize("tamper", ["unknown_contract", "settings", "old_prefix", "new_policy", "prepared", "source"])
def test_recovery_fails_closed_on_unknown_or_tampered_identity(tmp_path, monkeypatch, tamper):
    project, _, refs, scope, output = legacy_run(tmp_path, count=6, unknown=tamper == "unknown_contract")
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    if tamper == "unknown_contract":
        with pytest.raises(ValueError, match="changed"):
            tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(), staged_scope=scope)
        assert not (output / "generation_update.json").exists()
        return
    tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(), staged_scope=scope)
    if tamper == "settings":
        config = read_json(project); config["tts"]["seed"] += 1; write_json(project, config)
    elif tamper == "source":
        (project.parent / refs[-1]["path"]).write_bytes(b"changed")
    else:
        rows = read_rows(output / "candidates.jsonl")
        if tamper == "old_prefix": rows[0]["qc"]["reasons"] = ["changed"]
        elif tamper == "new_policy": rows[-1]["clone_input_policy"] = "other"
        elif tamper == "prepared": rows[-1]["clone_reference_preparation"]["end_frame"] -= 1
        (output / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(tts, "_load_model", lambda *args: pytest.fail("tampered run loaded a model"))
    with pytest.raises(ValueError):
        tts.generate(project, "train", 6, True, "clone", staged_scope=scope)


def test_final_boundary_blocks_before_reference_preparation(tmp_path, monkeypatch):
    project = approved_project(tmp_path / "word", "clone")
    monkeypatch.setattr(tts.clone_reference, "prepare_clone_reference", lambda *args: pytest.fail("Final read"))
    with pytest.raises(ValueError, match="Final generation stays locked"):
        tts.generate(project, "final", pilot=True)


def test_failure_message_is_readable_and_old_sparse_summaries_stay_supported():
    from korean_openwakeword.studio_steps import _generation_complete
    row = {"voice_family_id": "internal_voice_id", "label": "공통 평가용 · 남성 · 중년 · 1번", "attempts": 6,
           "failures": [{"qc": {"reasons": ["duration_outlier", "near_silence"]}}]}
    result = _generation_complete({"full_target_reached": False, "exhausted_slots": [row]})
    message = result["message"]
    assert "공통 평가용 · 남성 · 중년 · 1번" in message and "거의 무음" in message
    assert "internal_voice_id" not in message and "duration_outlier" not in message
    assert result["exhausted_slots"] == [row]
    assert _generation_complete({"full_target_reached": False, "exhausted_slots": [{}]})
    assert _generation_complete({"full_target_reached": False})


def test_completed_legacy_bank_reuses_every_artifact_without_preparation_or_receipt(tmp_path, monkeypatch):
    project, config, refs, scope, output = legacy_run(tmp_path, count=6)
    rows = read_rows(output / "candidates.jsonl")[:-2]
    row = rows[-1]
    path = project.parent / row["path"]
    sf.write(path, .1 * np.sin(np.arange(9600) / 32), 16000, subtype="PCM_16")
    row.update(sha256=digest(path), qc={"passed": True, "reasons": []})
    (output / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "summary.json", {"original": True})
    before = {str(path): digest(path) for path in project.parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(tts, "_load_model", lambda *args: pytest.fail("complete historical bank loaded model"))
    monkeypatch.setattr(tts.clone_reference, "prepare_clone_reference", lambda *args: pytest.fail("complete historical bank prepared input"))
    result = tts.generate(project, "train", 6, True, "clone", staged_scope=scope)
    assert result["full_target_reached"] and len(_passing_slots(project.parent, config, "train", refs, output, scope)[1]) == 6
    after = {str(path): digest(path) for path in project.parent.rglob("*") if path.is_file()}
    assert before == after


def test_validation_hashes_generator_once_for_whole_journal(tmp_path, monkeypatch):
    project, config, refs, scope, output = legacy_run(tmp_path, count=6)
    expected = tts.call_generation_contract(config, "train", refs, True, "clone", scope)
    rows, context = tts.call_journal_context(project.parent, output, expected)
    calls = []
    original = tts.digest
    def measured(path):
        calls.append(Path(path).resolve())
        return original(path)
    monkeypatch.setattr(tts, "digest", measured)
    tts.validate_call_rows(project.parent, config, "train", refs, rows, context, scope=scope)
    assert calls.count(Path(tts.__file__).resolve()) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_current_prepost_workflow_recovers_or_reports_actual_failure_without_human_approval(tmp_path, monkeypatch, fail):
    from korean_openwakeword import pilot
    from korean_openwakeword.studio_steps import _generation_complete
    project, config, refs, scope, output = legacy_run(tmp_path, count=3, references_count=3,
        scope_kind="preflight", revision=identity([]))
    root = project.parent
    validation = []
    for number in range(2):
        audio = root / f"validation_{number}.wav"
        sf.write(audio, .1 * np.sin(np.arange(8000) / (25 + number)), 16000, subtype="FLOAT")
        validation.append(dict(refs[0], voice_family_id=f"eval_{number}", split="validation",
                               path=str(audio.relative_to(root)), sha256=digest(audio)))
    model = RecoveryModel(fail=fail)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    monkeypatch.setattr(pilot, "_preflight_banks", lambda *args: {"train": refs, "validation": validation})
    monkeypatch.setattr("korean_openwakeword.tts_runtime.CloneRuntime", lambda *args: model)
    before_approval = (root / "approval.json").read_bytes()
    original_journal = (output / "candidates.jsonl").read_bytes()
    result = pilot.prepare_prepost_pilot(project, {"train_references": 3, "validation_references": 2})
    assert result["full_target_reached"] is not fail
    assert (root / "approval.json").read_bytes() == before_approval
    assert (output / "candidates.jsonl").read_bytes().startswith(original_journal)
    if fail:
        assert result["technical_pass"] == 2 and result["target"] == 5
        failures = result["exhausted_slots"]
        assert [(row["split"], row["slot"]) for row in failures] == [("train", 2), ("validation", 0), ("validation", 1)]
        assert all(row["attempts"] == 6 and "near_silence" in row["failures"][-1]["qc"]["reasons"] for row in failures)
        assert _generation_complete(result)["exhausted_slots"] == result["exhausted_slots"]
        assert "review_queue" not in result
        calls = model.calls
        assert pilot.prepare_prepost_pilot(project, {"train_references": 3, "validation_references": 2})["exhausted_slots"]
        assert model.calls == calls
    else:
        queue = read_rows(result["review_queue"])
        assert len(queue) == 5 and all(row["qc"]["passed"] is True for row in queue)
        assert all(row.get("individual_human_listening") is False for row in queue)
        assert {row["review_group"] for row in queue} == {"학습용", "공통 평가용"}
        assert _generation_complete(result) is None
