"""New projects begin with metadata slots and publish only explicitly selected WAVs."""
from copy import deepcopy
import random

import pytest

from korean_openwakeword import reference_candidates as candidates, tts
from korean_openwakeword.common import digest, identity, read_json, read_rows, snapshot_json, write_json
from korean_openwakeword.pilot import _alternate_reference, _preflight_banks
from korean_openwakeword.reference_qc import admitted_reference_rows
from korean_openwakeword.reference_voice_review import overlay_reference_banks
from test_korean_library_reference_candidates import waveform


@pytest.fixture
def planned(tmp_path, monkeypatch):
    root = tmp_path / "new-project"
    root.mkdir()
    project = root / "project.json"
    config = {"tts": {"design_model": "fixture-model", "reference_text": "오늘은 좋은 날입니다.",
                      "generation": {"temperature": .8}}}
    write_json(project, config)
    write_json(root / "approval.json", {"state": "pending", "generation_policy_approved": False})
    families = []
    for split, count in (("train", 2), ("validation", 3)):
        for gender in candidates.GENDERS:
            for age in candidates.AGES:
                for number in range(count):
                    families.append({"family_id": f"{split}_{gender}_{age.replace(' ', '_')}_{number}",
                        "split": split, "gender": gender, "age_group": age, "description": f"Original {gender} {age}"})
    write_json(root / "voices.json", {"families": families})
    monkeypatch.setattr(candidates, "load_project", lambda project: (root, config))
    monkeypatch.setattr(tts, "_seed", lambda seed: None)
    settings = {"train_references": 20, "validation_references": 30,
                "reference_selection_policy": candidates.PLANNED_CANDIDATE_POLICY}
    slots = candidates.source_slots(project, settings)
    saved = candidates.create_candidate_review(slots)
    calls = []
    class Model:
        def generate_voice_design(self, **kwargs):
            calls.append(kwargs)
            return [waveform(500 + len(calls))], 16000
    model = Model()
    return root, project, settings, slots, saved, calls, model


def fill(planned, *, listen=True):
    root, project, settings, slots, saved, calls, model = planned
    for number, group in enumerate(saved["groups"]):
        plan = candidates.plan_candidate_batch(project, saved, group["key"], rng=random.Random(number + 1))
        assert plan["originals"] == {} and plan["slot_ids"] == saved["slot_ids"]
        candidates.register_candidate_batch(saved, plan)
        rows = candidates.generate_candidate_batch(project, candidates.candidate_jobs(root) / plan["batch_id"] / "audio",
            plan["batch_id"], plan, [], load_model=lambda *args: model)
        candidates.ingest_candidate_batch(root, saved, plan, rows)
        for index, row in enumerate(rows[:group["target"]]):
            if listen is True or listen == "mixed" and index == 0:
                candidates.candidate_action(root, saved, "hear", candidate_id=row["candidate_id"], sha256=row["sha256"])
            candidates.candidate_action(root, saved, "toggle", candidate_id=row["candidate_id"], sha256=row["sha256"])
        candidates.candidate_action(root, saved, "approve")


def test_metadata_initialization_has_no_waveforms_models_or_admission_claims(planned):
    root, project, settings, slots, saved, calls, model = planned
    before = (root / "approval.json").read_bytes()
    descriptor = candidates.prepare_candidate_slots(project, settings)
    assert descriptor["kind"] == candidates.SLOT_PLAN_KIND and descriptor["generated"] == 0
    assert descriptor["slot_count"] == 50 and descriptor["group_count"] == 20
    assert candidates.verify_candidate_slots(project, settings, descriptor["slot_plan"]) == descriptor
    assert not calls and not list(root.rglob("*.wav"))
    assert not (root / "data/references").exists()
    assert all("path" not in slot and "sha256" not in slot for slot in slots)
    assert saved["kind"] == candidates.PLANNED_REVIEW_KIND and saved["originals"] == {}
    assert len(saved["slot_ids"]) == 50
    assert (root / "approval.json").read_bytes() == before
    with pytest.raises(ValueError, match="모든"):
        candidates.freeze_candidate_review(root, saved, slots)
    with (root / "voices.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="계획"):
        candidates.verify_candidate_slots(project, settings, descriptor["slot_plan"])


def test_pilot_replacement_replays_exact_candidate_prompt_and_preserves_legacy_format(planned):
    root, project, settings, slots, saved, calls, model = planned
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    rows = candidates.generate_candidate_batch(project, candidates.candidate_jobs(root) / plan["batch_id"] / "audio",
        "replacement-test", plan, [], load_model=lambda *args: model)
    config = read_json(project)
    config["tts"]["seed"] = 73129
    original = candidates._canonical_reference(slots[0], rows[0])
    replacement = _alternate_reference(root, config, original, "round2", lambda: model, {original["sha256"]})
    assert calls[-1]["instruct"] == original["design_instruction"]
    assert calls[-1]["language"] == "Korean" and replacement["sha256"] != original["sha256"]
    legacy = {key: value for key, value in original.items() if key not in ("candidate_id", "design_instruction")}
    legacy.update(voice_family_id="legacy_voice", description="A warm adult voice.", style="neutral", speed="natural", intonation="level")
    _alternate_reference(root, config, legacy, "round2", lambda: model, {legacy["sha256"]})
    assert calls[-1]["instruct"] == tts._reference_instruction(legacy)


def test_selected_exact_wavs_publish_after_all_qc_and_pilot_reads_the_canonical_bank(planned, monkeypatch):
    root, project, settings, slots, saved, calls, model = planned
    bank_before = (root / "voices.json").read_bytes()
    approval_before = (root / "approval.json").read_bytes()
    fill(planned)
    assert len(calls) == 100 and len(saved["candidates"]) == 100
    assert not (root / "data/references").exists()
    # A damaged chosen WAV cannot partially admit even a valid first partition.
    cid = saved["selected"][saved["groups"][-1]["key"]][-1]
    damaged = root / saved["candidates"][cid]["path"]
    original_bytes = damaged.read_bytes()
    damaged.write_bytes(original_bytes + b"tampered")
    with pytest.raises(ValueError, match="changed bound file"):
        candidates.freeze_candidate_review(root, saved, slots)
    assert not (root / "data/references").exists()
    assert (root / "approval.json").read_bytes() == approval_before
    damaged.write_bytes(original_bytes)
    actual_check = candidates._check_wave
    with monkeypatch.context() as patched:
        def failed_final_check(path):
            reasons, metrics = actual_check(path)
            return (["near_silence"], metrics) if path == damaged else (reasons, metrics)
        patched.setattr(candidates, "_check_wave", failed_final_check)
        with pytest.raises(ValueError, match="기술 검사"):
            candidates.freeze_candidate_review(root, saved, slots)
    assert not (root / "data/references").exists()
    assert (root / "approval.json").read_bytes() == approval_before
    manifest = candidates.freeze_candidate_review(root, saved, slots)
    assert manifest["kind"] == candidates.PLANNED_REVIEW_KIND
    assert "originals" not in manifest and len(manifest["references"]) == 50
    assert (root / "voices.json").read_bytes() == bank_before
    journals = {split: read_rows(root / "data/references" / split / "references.jsonl") for split in ("train", "validation")}
    assert [len(journals[split]) for split in ("train", "validation")] == [20, 30]
    for rows in journals.values():
        for row in rows:
            candidate = saved["candidates"][row["candidate_id"]]
            assert row["path"] == candidate["path"] and row["sha256"] == candidate["sha256"]
            assert row["sample_id"] == row["voice_family_id"]
            assert row["description"] == candidate["design_instruction"]
            assert "replaces_reference_sha256" not in row
            assert digest(root / row["path"]) == row["sha256"]
    admission = read_json(root / "approval.json")
    assert len(admitted_reference_rows(root, admission)) == 50
    assert admission["generation_policy_approved"] is False
    journal_bytes = {split: (root / "data/references" / split / "references.jsonl").read_bytes() for split in journals}
    again = candidates.freeze_candidate_review(root, saved, slots)
    assert again["technical_admission"] == manifest["technical_admission"]
    assert journal_bytes == {split: (root / "data/references" / split / "references.jsonl").read_bytes() for split in journals}
    binding = snapshot_json(root, manifest, "planned-candidate-review")
    slot_banks = {split: [slot for slot in slots if slot["split"] == split] for split in journals}
    assert overlay_reference_banks(root, slot_banks, binding) == journals
    assert overlay_reference_banks(root, journals, binding) == journals
    missing_policy = deepcopy(manifest)
    missing_policy.pop("listening_policy")
    with pytest.raises(ValueError, match="식별자"):
        candidates.overlay_planned_banks(root, slot_banks, missing_policy)
    # Reconstruct the previous, unmarked all-listened document shape. Its
    # original identity and receipt remain readable under the new validator.
    legacy = deepcopy(missing_policy)
    body_keys = ("source_binding", "slot_ids", "groups", "approved_groups", "selections", "heard", "references", "source_scope")
    legacy["selection_id"] = identity({key: legacy[key] for key in body_keys})
    legacy_receipt = read_json(root / manifest["technical_admission"]["path"])
    legacy_receipt["selection_id"] = legacy["selection_id"]
    legacy_path = root / "review/legacy-admission.json"
    write_json(legacy_path, legacy_receipt)
    legacy["technical_admission"] = {"path": legacy_path.relative_to(root).as_posix(), "sha256": digest(legacy_path)}
    assert candidates.overlay_planned_banks(root, slot_banks, legacy) == journals
    removed = next(iter(legacy["heard"]))
    legacy["heard"].pop(removed)
    legacy["selection_id"] = identity({key: legacy[key] for key in body_keys})
    with pytest.raises(ValueError, match="청취 기록"):
        candidates.overlay_planned_banks(root, slot_banks, legacy)
    monkeypatch.setenv("KOWW_STUDIO_REFERENCE_REVIEW", __import__("json").dumps(binding))
    assert _preflight_banks(root, settings) == journals
    altered = deepcopy(manifest)
    altered["groups"][0]["families"], altered["groups"][1]["families"] = altered["groups"][1]["families"], altered["groups"][0]["families"]
    with pytest.raises(ValueError, match="슬롯 및 분류"):
        candidates.overlay_planned_banks(root, slot_banks, altered)
    # Existing nonempty canonical rows can never be overwritten with a new choice.
    first_journal = root / "data/references/train/references.jsonl"
    with first_journal.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(FileExistsError, match="immutable"):
        candidates.freeze_candidate_review(root, saved, slots)
    with pytest.raises(ValueError, match="공개 은행"):
        overlay_reference_banks(root, slot_banks, binding)


@pytest.mark.parametrize("listen", [False, "mixed"])
def test_optional_listening_publishes_honest_zero_or_partial_counts_after_full_technical_checks(planned, monkeypatch, listen):
    root, project, settings, slots, saved, calls, model = planned
    fill(planned, listen=listen)
    expected_heard = deepcopy(saved["heard"])
    assert len(expected_heard) == (20 if listen else 0)
    original_approval = (root / "approval.json").read_bytes()
    original_bank = (root / "voices.json").read_bytes()
    chosen_id = saved["selected"][saved["groups"][-1]["key"]][-1]
    chosen = root / saved["candidates"][chosen_id]["path"]
    actual_check = candidates._check_wave
    with monkeypatch.context() as patched:
        def fail_chosen(path):
            reasons, metrics = actual_check(path)
            return (["near_silence"], metrics) if path == chosen else (reasons, metrics)
        patched.setattr(candidates, "_check_wave", fail_chosen)
        with pytest.raises(ValueError, match="기술 검사"):
            candidates.freeze_candidate_review(root, saved, slots)
    assert not (root / "data/references").exists()
    assert (root / "approval.json").read_bytes() == original_approval
    manifest = candidates.freeze_candidate_review(root, saved, slots)
    assert manifest["listening_policy"] == candidates.OPTIONAL_LISTENING_POLICY
    assert manifest["heard"] == expected_heard == saved["heard"]
    receipt = read_json(root / manifest["technical_admission"]["path"])
    assert receipt["human_listened_count"] == len(expected_heard)
    assert len(receipt["checked_audio_sha256"]) == len(admitted_reference_rows(root, read_json(root / "approval.json"))) == 50
    assert (root / "voices.json").read_bytes() == original_bank
    banks = {split: [slot for slot in slots if slot["split"] == split] for split in ("train", "validation")}
    published = candidates.overlay_planned_banks(root, banks, manifest)
    assert [len(published[split]) for split in banks] == [20, 30]
    binding = snapshot_json(root, manifest, "optional-listening-review")
    monkeypatch.setenv("KOWW_STUDIO_REFERENCE_REVIEW", __import__("json").dumps(binding))
    assert _preflight_banks(root, settings) == published
    for forged_count in (50, False):
        bad_receipt = {**receipt, "human_listened_count": forged_count}
        target = root / f"review/forged-count-{forged_count}.json"
        write_json(target, bad_receipt)
        changed = {**manifest, "technical_admission": {"path": target.relative_to(root).as_posix(), "sha256": digest(target)}}
        with pytest.raises(ValueError, match="기술 검사 승인"):
            candidates.overlay_planned_banks(root, banks, changed)
    changed = deepcopy(manifest)
    changed["heard"][chosen_id] = "0" * 64
    with pytest.raises(ValueError, match="청취 기록"):
        candidates.overlay_planned_banks(root, banks, changed)
    with pytest.raises(ValueError, match="청취 정책"):
        candidates.overlay_planned_banks(root, banks, {**manifest, "listening_policy": "unrecognized"})
    chosen.write_bytes(chosen.read_bytes() + b"changed after admission")
    with pytest.raises(ValueError, match="changed bound file"):
        candidates.overlay_planned_banks(root, banks, manifest)
