"""Repair uses real hash-bound files and the existing clone/QC journal code.

Only the acoustic model and preflight fixture bank are mocked. No GPU, model
download, user disposition, or production project is involved.
"""
from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import postgeneration_repair as repair, tts
from korean_openwakeword.common import digest, identity, read_json, read_rows, snapshot_json, write_immutable_rows, write_json
from korean_openwakeword.pilot import PREPOST_PILOT_POLICY, REVIEW_GROUPS
from korean_openwakeword.project import create_project


class AcousticModel:
    def __init__(self):
        self.calls = 0; self.fail = False; self.interrupt_at = None

    def create_voice_clone_prompt(self, **options):
        return options

    def generate_voice_clone(self, **options):
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise RuntimeError("fixture interruption")
        wave = np.zeros(9600) if self.fail else .12 * np.sin(np.arange(9600) / 21) + np.random.normal(0, .002, 9600)
        return [wave], 16000


def fixture(tmp_path, monkeypatch, *, counts=(2, 2), targets=(5, 7), bad=("train", 0)):
    project = create_project(tmp_path / "word", "가영아", "gayoung", train_voice_families=counts[0], validation_voice_families=counts[1])
    root, config = project.parent, read_json(project)
    banks, rows, queue = {}, {}, []
    for split, count, target in zip(REVIEW_GROUPS, counts, targets):
        banks[split] = []
        for index in range(count):
            family = f"{split}-voice-{index}"
            path = root / "fixture/references" / f"{family}.wav"; path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, .1 * np.sin(np.arange(9600) / (20 + index + (7 if split == "validation" else 0))), 16000, subtype="PCM_16")
            banks[split].append({"sample_id": family, "voice_family_id": family, "split": split,
                "path": path.relative_to(root).as_posix(), "sha256": digest(path), "text": config["tts"]["reference_text"],
                "provenance": "qwen_synthetic_reference", "human_reference_used": False})
        rows[split] = []
        for slot in range(target):
            reference = banks[split][slot % count]
            path = root / "fixture/calls" / f"{split}-{slot}.wav"; path.parent.mkdir(parents=True, exist_ok=True)
            # These are existing source fixtures, not newly admitted outputs.
            noise = np.random.default_rng(slot + (10000 if split == "validation" else 0)).normal(0, .08, 256)
            sf.write(path, noise, 16000, subtype="PCM_16")
            row = {"sample_id": path.stem, "slot": slot, "path": path.relative_to(root).as_posix(), "sha256": digest(path),
                "split": split, "voice_family_id": reference["voice_family_id"], "voice_reference_sha256": reference["sha256"],
                "text": config["wakeword"], "tts_settings_id": identity(config["tts"]), "provenance": "qwen_tts",
                "human_reference_used": False, "generation_method": "clone", "qc": {"passed": True}, "individual_human_listening": False}
            rows[split].append(row)
            if slot < count:
                queue.append({**row, "review_group": REVIEW_GROUPS[split], "voice_number": slot + 1, "pronunciation_policy": PREPOST_PILOT_POLICY})
    monkeypatch.setattr(repair, "_reference_banks", lambda _root, _counts, _binding: banks)
    monkeypatch.setattr(repair, "_preflight_state", lambda *_: ({row["voice_family_id"]: row for bank in banks.values() for row in bank}, set(), {"preflight-fixture-hash"}, {row["sha256"] for bank in banks.values() for row in bank}))
    model, loads = AcousticModel(), []
    monkeypatch.setattr(repair, "_runtime", lambda *_: loads.append(True) or model)
    monkeypatch.setattr(tts, "_seed", lambda seed: (random.seed(seed), np.random.seed(seed)))
    bindings = {}
    for split, values in rows.items():
        path = root / "fixture" / f"sources-{split}.jsonl"; write_immutable_rows(path, values); bindings[split] = repair._binding(root, path)
    path = root / "fixture/review.jsonl"; write_immutable_rows(path, queue)
    generated = {"review_kind": "positive", "full_target_reached": True, "queue": str(path), "queue_binding": repair._binding(root, path),
                 "preview_sources": bindings, "preflight_reviews": [{"fixture": True}], "train_originals": targets[0], "validation_originals": targets[1], "outputs": []}
    settings = {"originals": targets[0], "validation_originals": targets[1]}
    report = review_report(root, config, generated, queue, {banks[bad[0]][bad[1]]["voice_family_id"]})
    return project, config, banks, rows, queue, generated, settings, report, model, loads


def review_report(root, config, generated, queue, bad):
    report = {"kind": "positive_listening_review", "wakeword": config["wakeword"], "queue_sha256": generated["queue_binding"]["sha256"],
        "queued_audio_sha256": {row["sample_id"]: row["sha256"] for row in queue}, "saved_at": "fixture-real-disposition",
        "source_scope": "Human dispositions entered through the local review screen",
        "decisions": {row["sample_id"]: {"decision": "discard" if row["voice_family_id"] in bad else "keep", "listened": True} for row in queue}}
    return snapshot_json(root, report, "fixture_post_review")


def hashes(root):
    return {path: digest(path) for path in root.rglob("*") if path.is_file() and path.suffix != ".lock"}


def test_real_journal_selective_repair_preserves_good_sources_and_true_review(tmp_path, monkeypatch):
    project, config, banks, rows, queue, original, settings, report, model, loads = fixture(tmp_path, monkeypatch)
    before = hashes(project.parent)
    request = repair.prepare_postgeneration_repair(project, original, report)
    assert request["target"] == 3 and request["pending_review_count"] == 1 and request["carried_review_count"] == 3
    result = repair.generate_postgeneration_repair(project, settings, request)
    assert len(loads) == 1 and model.calls == 3
    assert all(digest(path) == sha for path, sha in before.items())
    effective = read_rows(project.parent / result["preview_sources"]["train"]["path"])
    assert [row for row in effective if row["voice_family_id"] != banks["train"][0]["voice_family_id"]] == rows["train"][1::2]
    assert read_rows(project.parent / result["preview_sources"]["validation"]["path"]) == rows["validation"]
    new_queue = read_rows(project.parent / result["queue_binding"]["path"])
    assert new_queue[1:] == queue[1:]
    state = repair.postgeneration_repair_review(project, result)
    assert state["pending_ids"] == [new_queue[0]["sample_id"]]
    assert state["carried_decisions"] == {row["sample_id"]: {"decision": "keep", "listened": True} for row in queue[1:]}
    assert new_queue[0]["individual_human_listening"] is False
    assert repair.resolve_postgeneration_repair(project, original, [{"request": request}]) == result
    after = hashes(project.parent)
    assert repair.generate_postgeneration_repair(project, settings, request) == result
    assert len(loads) == 1 and model.calls == 3 and hashes(project.parent) == after


def test_interruption_resumes_same_scope_without_regenerating_completed_calls(tmp_path, monkeypatch):
    project, _, _, _, _, original, settings, report, model, loads = fixture(tmp_path, monkeypatch)
    request = repair.prepare_postgeneration_repair(project, original, report); model.interrupt_at = 3
    with pytest.raises(RuntimeError, match="fixture interruption"):
        repair.generate_postgeneration_repair(project, settings, request)
    assert repair.postgeneration_repair_progress(project, request)["done"] == 2
    assert repair.resolve_postgeneration_repair(project, original, [{"request": request}]) == original
    completed = {path: digest(path) for path in project.parent.glob("data/call_pronunciation/*/train/*.wav")}
    model.interrupt_at = None
    result = repair.generate_postgeneration_repair(project, settings, request)
    assert result["full_target_reached"] and model.calls == 4 and len(loads) == 2
    assert all(digest(path) == sha for path, sha in completed.items())


def test_exhausted_qc_keeps_old_corpus_and_reuses_existing_bounded_failures(tmp_path, monkeypatch):
    project, _, _, _, _, original, settings, report, model, _ = fixture(tmp_path, monkeypatch)
    request = repair.prepare_postgeneration_repair(project, original, report); before = hashes(project.parent); model.fail = True
    result = repair.generate_postgeneration_repair(project, settings, request)
    assert result["full_target_reached"] is False and result["technical_pass"] == 0 and model.calls == 18
    assert result["exhausted_slots"] and all(slot["maximum_attempts"] == 6 for slot in result["exhausted_slots"])
    assert all(digest(path) == sha for path, sha in before.items())
    assert repair.resolve_postgeneration_repair(project, original, [{"request": request}]) == original
    repair.generate_postgeneration_repair(project, settings, request)
    assert model.calls == 18


def test_multiple_rounds_keep_exact_fixed_reference_and_exclude_old_replacements(tmp_path, monkeypatch):
    project, config, banks, _, _, original, settings, report, model, _ = fixture(tmp_path, monkeypatch)
    first = repair.prepare_postgeneration_repair(project, original, report)
    result = repair.generate_postgeneration_repair(project, settings, first)
    queue = read_rows(project.parent / result["queue_binding"]["path"])
    report2 = review_report(project.parent, config, result, queue, {banks["train"][0]["voice_family_id"]})
    second = repair.prepare_postgeneration_repair(project, result, report2)
    assert second["round"] == 2 and second["plan"] != first["plan"]
    result2 = repair.generate_postgeneration_repair(project, settings, second)
    queue2 = read_rows(project.parent / result2["queue_binding"]["path"])
    assert queue2[1:] == queue[1:] and queue2[0]["sha256"] != queue[0]["sha256"]
    assert queue2[0]["voice_reference_sha256"] == queue[0]["voice_reference_sha256"]
    assert repair.resolve_postgeneration_repair(project, original, [{"request": first}, {"request": second}]) == result2
    assert model.calls == 6


@pytest.mark.parametrize("tamper", ["wave", "queue", "report", "plan"])
def test_changed_bound_inputs_block_without_model_calls(tmp_path, monkeypatch, tamper):
    project, _, _, rows, _, original, settings, report, model, _ = fixture(tmp_path, monkeypatch)
    request = repair.prepare_postgeneration_repair(project, original, report)
    path = {"wave": project.parent / rows["train"][1]["path"], "queue": Path(original["queue"]),
            "report": project.parent / report["path"], "plan": project.parent / request["plan"]["path"]}[tamper]
    if tamper == "report":
        path = project.parent / read_json(project.parent / request["plan"]["path"])["parent_report"]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        repair.generate_postgeneration_repair(project, settings, request)
    assert model.calls == 0


def test_two_rejected_families_share_one_runtime_and_keep_uneven_family_quotas(tmp_path, monkeypatch):
    project, config, banks, _, queue, original, settings, _, model, loads = fixture(tmp_path, monkeypatch)
    report = review_report(project.parent, config, original, queue, {banks["train"][0]["voice_family_id"], banks["validation"][0]["voice_family_id"]})
    request = repair.prepare_postgeneration_repair(project, original, report)
    assert [item["count"] for item in request["affected_families"]] == [3, 4]
    result = repair.generate_postgeneration_repair(project, settings, request)
    assert model.calls == 7 and len(loads) == 1
    state = repair.postgeneration_repair_review(project, result)
    assert len(state["pending_ids"]) == 2 and len(state["carried_decisions"]) == 2


def test_plan_cannot_be_rebound_to_change_quotas_or_exclusions(tmp_path, monkeypatch):
    project, _, _, _, _, original, settings, report, model, _ = fixture(tmp_path, monkeypatch)
    request = repair.prepare_postgeneration_repair(project, original, report)
    plan = read_json(project.parent / request["plan"]["path"])
    plan["families"][0]["count"] += 1
    altered = repair._request(snapshot_json(project.parent, plan, "fixture_rebound_plan"), plan)
    with pytest.raises(ValueError, match="exact quotas"):
        repair.generate_postgeneration_repair(project, settings, altered)
    assert model.calls == 0


def test_selected_reference_binding_is_explicit_and_frozen_without_global_env(tmp_path, monkeypatch):
    original_builder = repair._reference_banks
    project, _, banks, _, _, original, settings, report, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(repair, "_reference_banks", original_builder)
    monkeypatch.setattr(repair, "_human_answers", lambda *_: {})
    monkeypatch.setattr(repair, "_admitted_references", lambda _root, _approval, _answers, split, _count: banks[split])
    selected = snapshot_json(project.parent, {"fixture": "selected reference overlay"}, "fixture_reference_selection")
    seen = []
    monkeypatch.setattr("korean_openwakeword.reference_voice_review.overlay_reference_banks", lambda _root, raw, binding: seen.append(binding) or raw)
    monkeypatch.setenv("KOWW_STUDIO_REFERENCE_REVIEW", "invalid global binding must be ignored")
    request = repair.prepare_postgeneration_repair(project, original, report, reference_review=selected)
    assert seen == [selected] and read_json(project.parent / request["plan"]["path"])["reference_review"] == selected
    result = repair.generate_postgeneration_repair(project, settings, request, reference_review=selected)
    assert repair.postgeneration_repair_review(project, result, reference_review=selected)["pending_ids"]
    with pytest.raises(ValueError, match="frozen project"):
        repair.generate_postgeneration_repair(project, settings, request, reference_review={"path": "different", "sha256": "different"})


def test_repaired_all_keep_review_uses_existing_admission_and_resumes_partial_publication(tmp_path, monkeypatch):
    from korean_openwakeword import pilot
    from korean_openwakeword.review import apply_review
    project, _, banks, _, _, original, settings, report, _, _ = fixture(tmp_path, monkeypatch)
    request = repair.prepare_postgeneration_repair(project, original, report)
    generated = repair.generate_postgeneration_repair(project, settings, request)
    monkeypatch.setattr(pilot, "_preflight_banks", lambda *_: banks)
    monkeypatch.setattr(pilot, "_preflight_state", repair._preflight_state)
    queue = read_rows(Path(generated["queue"]))
    answers = {row["sample_id"]: {"decision": "keep", "listened": True} for row in queue}
    # Actual review admission remains the existing full-corpus API. The core
    # itself never invents these new listening facts.
    applied = apply_review(project, generated["queue"], answers, "positive")
    assert applied["policy_approved"] is True
    real_write = pilot.write_immutable_rows
    calls = []
    def interrupt_second(path, rows, **options):
        calls.append(Path(path))
        if len(calls) == 2:
            raise RuntimeError("fixture publication interruption")
        return real_write(path, rows, **options)
    monkeypatch.setattr(pilot, "write_immutable_rows", interrupt_second)
    with pytest.raises(RuntimeError, match="publication interruption"):
        pilot.admit_postgeneration_positives(project, generated)
    first_hash = digest(calls[0])
    monkeypatch.setattr(pilot, "write_immutable_rows", real_write)
    published = pilot.admit_postgeneration_positives(project, generated)
    assert digest(calls[0]) == first_hash
    assert [len(read_rows(path)) for path in published] == [5, 7]
    assert sum(row["individual_human_listening"] for path in published for row in read_rows(path)) == 4
    assert read_json(project.parent / "approval.json")["generation_policy_approved"] is True
    assert pilot.admit_postgeneration_positives(project, generated) == published


def test_twenty_thirty_reference_fixture_repairs_fifty_calls_and_preserves_2950(tmp_path, monkeypatch):
    project, _, banks, rows, queue, original, settings, report, model, loads = fixture(tmp_path, monkeypatch, counts=(20, 30), targets=(1000, 2000), bad=("train", 5))
    request = repair.prepare_postgeneration_repair(project, original, report)
    assert request["target"] == 50 and request["carried_review_count"] == 49
    result = repair.generate_postgeneration_repair(project, settings, request)
    assert model.calls == 50 and len(loads) == 1
    effective = {split: read_rows(project.parent / result["preview_sources"][split]["path"]) for split in REVIEW_GROUPS}
    family = banks["train"][5]["voice_family_id"]
    kept = sum(old == new for split in REVIEW_GROUPS for old, new in zip(rows[split], effective[split]))
    assert kept == 2950
    new_queue = read_rows(Path(result["queue"]))
    assert sum(a == b for a, b in zip(queue, new_queue)) == 49
    state = repair.postgeneration_repair_review(project, result)
    assert len(state["carried_decisions"]) == 49 and len(state["pending_ids"]) == 1
    assert all(row["voice_family_id"] == family for row in effective["train"] if row not in rows["train"])
