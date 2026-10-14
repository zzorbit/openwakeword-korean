"""Candidate generation must stay reproducible, gated and separate from admission."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import hard_tts
from korean_openwakeword.common import append_row, digest, identity, read_json, read_rows, write_json
from korean_openwakeword.data import validate_hard_negative
from korean_openwakeword.hard_phrase_policy import protected_call_terms, validate_noncall_text
from korean_openwakeword.project import create_project


PHRASES = ["보리", "보리야", "보람아", "돌아와"]


def approved_project(tmp_path):
    project = create_project(tmp_path / "bora", "보라야", "bora", positive_originals=1000)
    root = project.parent
    config = read_json(project)
    references = []
    for number in range(4):
        path = root / "data" / "references" / "train" / f"voice{number}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, .1 * np.sin(np.arange(16000) * (.04 + number * .01)), 16000, subtype="PCM_16")
        row = {"voice_family_id": f"voice{number}", "split": "train", "path": path.relative_to(root).as_posix(),
            "sha256": digest(path), "text": "합성 참조 문장", "status": "accepted",
            "human_reference_used": False, "provenance": "qwen_synthetic_reference"}
        append_row(path.parent / "references.jsonl", row)
        references.append(row)
    write_json(root / "approval.json", {"state": "accepted", "wakeword": "보라야", "human_reference_used": False,
        "generation_policy_approved": True, "generation_method": "clone", "tts_settings_id": identity(config["tts"]),
        "accepted_reference_families": [r["voice_family_id"] for r in references]})
    return project


class FakeModel:
    def __init__(self, fail_at=None, duplicate=False, long_first=False):
        self.calls = 0
        self.prompts = 0
        self.fail_at = fail_at
        self.duplicate = duplicate
        self.long_first = long_first

    def create_voice_clone_prompt(self, **kwargs):
        self.prompts += 1
        return kwargs["ref_audio"]

    def generate_voice_clone(self, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        count = 40000 if self.long_first and self.calls == 1 else 16000
        frequency = .045 if self.duplicate else .025 + self.calls * .002
        return [.1 * np.sin(np.arange(count) * frequency)], 16000


@pytest.fixture(autouse=True)
def no_seed_gpu(monkeypatch):
    monkeypatch.setattr(hard_tts.tts, "_seed", lambda seed: None)


def test_balanced_100_candidate_contract_and_resume_never_loads_model(tmp_path, monkeypatch):
    project = approved_project(tmp_path)
    model = FakeModel()
    result = hard_tts.generate_hard(project, PHRASES, 100, model)
    assert result["full_target_reached"] and result["technical_pass"] == result["attempts"] == 100
    assert set(result["phrase_distribution"].values()) == {25}
    assert set(result["voice_distribution"].values()) == {25}
    assert len(result["combination_distribution"]) == 16
    assert set(result["combination_distribution"].values()) == {6, 7}
    assert result["policy_admitted"] == result["individual_human_listened"] == result["independent_STT_runs"] == 0
    assert not result["actual_training_manifest_published"]
    assert model.prompts == 4
    root = project.parent
    contract = read_json(root / "data/generated/hard/contract.json")
    assert contract["kind"] == "qwen_hard_negative_generation_contract_v1"
    assert contract["human_reference_used"] is False
    assert contract["phrase_policy"] == {
        "kind": "exclude_complete_call_and_vocative_base_v1", "protected_terms": ["보라야", "보라"]}
    assert digest(contract["implementation"]["path"]) == contract["implementation"]["sha256"]
    rows = read_rows(root / "data/generated/hard/candidates.jsonl")
    assert all(r["generation_contract_id"] == identity(contract) and "admission" not in r for r in rows)
    assert all(r["technical_pass"] and r["end_s"] - r["start_s"] <= 1.84 for r in rows)
    assert len({r["seed"] for r in rows}) == 100
    assert not read_rows(root / read_json(project)["manifests"]["hard_negative"])
    monkeypatch.setattr(hard_tts.tts, "_load_model", lambda *args: pytest.fail("Completed resume loaded a model"))
    resumed = hard_tts.generate_hard(project, PHRASES, 100)
    assert resumed["attempts"] == 100
    assert not (root / "data/generated/hard/.writer.lock").exists()


def test_approval_and_phrase_limits_are_checked_before_model_loading(tmp_path, monkeypatch):
    project = create_project(tmp_path / "unapproved", "보라야", "bora")
    monkeypatch.setattr(hard_tts.tts, "_load_model", lambda *args: pytest.fail("Weights loaded before validation"))
    with pytest.raises(ValueError, match="Approve"):
        hard_tts.generate_hard(project, PHRASES)
    for count in (0, True, 1.5, hard_tts.MAX_ORIGINALS + 1):
        with pytest.raises(ValueError, match="count"):
            hard_tts.generate_hard(project, PHRASES, count)
    for phrases in (["보라야"], ["보라야켜줘"], ["보라", "보라"], ["보라 야"], []):
        with pytest.raises(ValueError, match="phrase|Hangul"):
            hard_tts.generate_hard(project, phrases)


def test_qc_failure_retries_same_phrase_voice_then_accepts(tmp_path):
    project = approved_project(tmp_path)
    model = FakeModel(long_first=True)
    result = hard_tts.generate_hard(project, PHRASES, 1, model)
    assert result["attempts"] == 2 and result["technical_pass"] == 1
    rows = read_rows(project.parent / "data/generated/hard/candidates.jsonl")
    assert rows[0]["text"] == rows[1]["text"] == "보리"
    assert rows[0]["voice_family_id"] == rows[1]["voice_family_id"]
    assert rows[0]["technical_pass"] is False and rows[1]["technical_pass"] is True
    assert "does_not_fit_full_call_feature_window" in rows[0]["qc"]["reasons"]
    assert rows[0]["seed"] != rows[1]["seed"]


def test_duplicate_outputs_are_quarantined_and_exhaustion_is_not_completion(tmp_path):
    project = approved_project(tmp_path)
    result = hard_tts.generate_hard(project, PHRASES, 2, FakeModel(duplicate=True))
    assert result["technical_pass"] == 1 and result["attempts"] == 4
    assert result["exhausted_slots"] == [1] and not result["full_target_reached"]
    rows = read_rows(project.parent / "data/generated/hard/candidates.jsonl")
    assert all("exact_duplicate" in r["qc"]["reasons"] for r in rows[1:])
    assert result["policy_admitted"] == 0


def test_interrupted_resume_preserves_accepted_files_and_contract(tmp_path):
    project = approved_project(tmp_path)
    with pytest.raises(RuntimeError, match="interruption"):
        hard_tts.generate_hard(project, PHRASES, 4, FakeModel(fail_at=3))
    path = project.parent / "data/generated/hard/candidates.jsonl"
    before = read_rows(path)
    hashes = {r["path"]: r["sha256"] for r in before}
    resumed_model = FakeModel()
    resumed_model.calls = 2
    result = hard_tts.generate_hard(project, PHRASES, 4, resumed_model)
    assert result["technical_pass"] == result["attempts"] == 4
    assert read_rows(path)[:2] == before
    assert all(digest(project.parent / p) == h for p, h in hashes.items())


def test_changed_audio_reference_or_settings_refuses_resume(tmp_path):
    project = approved_project(tmp_path)
    hard_tts.generate_hard(project, PHRASES, 1, FakeModel())
    with pytest.raises(ValueError, match="changed"):
        hard_tts.generate_hard(project, ["다리", "보리야", "보람아", "돌아와"], 1, FakeModel())
    with pytest.raises(ValueError, match="changed"):
        hard_tts.generate_hard(project, PHRASES, 2, FakeModel())
    row = read_rows(project.parent / "data/generated/hard/candidates.jsonl")[0]
    (project.parent / row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed bound file"):
        hard_tts.generate_hard(project, PHRASES, 1, FakeModel())


def test_human_reference_rejected_before_loading(tmp_path, monkeypatch):
    project = approved_project(tmp_path)
    path = project.parent / "data/references/train/references.jsonl"
    rows = read_rows(path)
    rows[0]["human_reference_used"] = True
    import json
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(hard_tts.tts, "_load_model", lambda *args: pytest.fail("Human reference loaded"))
    with pytest.raises(ValueError, match="Qwen synthetic"):
        hard_tts.generate_hard(project, PHRASES, 1)


@pytest.mark.parametrize("text", [
    "보라", "보라가", "보라를", "보라는", "보라야", "보라야불켜줘",
    "보라야 불켜줘", "보 라 를", "예쁜보라", "보라색",
])
def test_direct_call_name_rejected_before_initial_or_resume_loading(tmp_path, monkeypatch, text):
    project = approved_project(tmp_path)
    # An existing output must not make a prohibited phrase resumable or be
    # silently overwritten by the new phrase policy.
    output = project.parent / "data/generated/hard"
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "contract.json"
    marker.write_text('{"historical_contract": true}\n', encoding="utf-8")
    before = marker.read_bytes()
    monkeypatch.setattr(hard_tts.tts, "_load_model", lambda *args: pytest.fail("Forbidden call-name loaded Qwen"))
    with pytest.raises(ValueError, match="complete call-name"):
        hard_tts.generate_hard(project, [text], 1)
    assert marker.read_bytes() == before
    assert not (output / "candidates.jsonl").exists()
    assert not (output / ".writer.lock").exists()


@pytest.mark.parametrize("wakeword,text", [
    ("보라야", "보리야"), ("보라야", "보람아"), ("보라야", "돌아와"),
    ("춘식아", "춘천"), ("도아", "도우미"), ("도아", "도착"),
])
def test_shared_syllables_do_not_ban_distinct_noncall_words(wakeword, text):
    assert hard_tts._phrases([text], wakeword) == [text]


@pytest.mark.parametrize("text", ["보라", "보라가", "보라를", "보라는", "보 라 를", "보라야 불켜줘"])
def test_imported_hard_negative_cannot_bypass_direct_name_guard(text):
    row = {"text": text, "speech": True, "provenance": "public_dataset",
           "source_url": "https://example.org/hard", "license": "fixture"}
    with pytest.raises(ValueError, match="complete call-name"):
        validate_hard_negative(row, {"wakeword": "보라야"})


def test_vocative_base_policy_is_explicit_and_does_not_guess_single_syllable_names():
    assert protected_call_terms("춘식아") == ["춘식아", "춘식"]
    assert protected_call_terms("보라야") == ["보라야", "보라"]
    assert protected_call_terms("도아") == ["도아"]
    with pytest.raises(ValueError, match="complete call-name"):
        validate_noncall_text("춘식이를불러", "춘식아")
