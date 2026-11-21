"""Hard generation keeps each phrase and voice bound to its durable candidate."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import hard_tts, tts
from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json
from test_call_voice_fill import write_torn_batch
from test_korean_library_hard_tts import PHRASES, approved_project
from test_korean_library_tts_journal import SeededQwen, cpu_seed


class MixedVoiceModel:
    def __init__(self, *, available=True, decline_batch=False, reject_one=False):
        self.available = available
        self.decline_batch = decline_batch
        self.reject_one = reject_one
        self.batches = []
        self.singles = []

    def mixed_voice_batch_available(self):
        return self.available

    def create_voice_clone_prompt(self, *, ref_audio, **kwargs):
        return [Path(ref_audio).stem]

    def generate_short_clone_mixed_voice_batch(self, *, batch_seed, voice_clone_prompts, text, language, **kwargs):
        voices = tuple(prompt[0] for prompt in voice_clone_prompts)
        assert language == "Korean" and len(voices) == len(set(voices)) == 4
        self.batches.append((text, voices, batch_seed))
        if self.decline_batch:
            return None
        wavs = [SeededQwen.waveform((batch_seed + index) % 2**32) for index in range(4)]
        endings = [{"termination": "eos", "generated_tokens": 10, "token_limit": 51} for _ in wavs]
        if self.reject_one:
            endings[1].update(termination="token_limit", generated_tokens=51)
            self.reject_one = False
        return wavs, 16000, 51, endings

    def generate_short_clone(self, *, retry, termination, voice_clone_prompt, text, **kwargs):
        seed = int(np.random.get_state()[1][0])
        self.singles.append((text, voice_clone_prompt[0], seed, retry))
        termination.update(termination="eos", generated_tokens=10, token_limit=51)
        return [SeededQwen.waveform(seed)], 16000, 51


@pytest.fixture
def hard_project(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    return approved_project(tmp_path)


def test_mixed_batches_use_distinct_voices_per_phrase_and_retry_only_failed_item(hard_project):
    model = MixedVoiceModel(reject_one=True)
    phrases = PHRASES[:2]
    result = hard_tts.generate_hard(hard_project, phrases, model=model, samples_per_phrase=6)
    rows = read_rows(result["candidates"])

    assert result["full_target_reached"] and result["technical_pass"] == 12 and result["attempts"] == 13
    assert result["phrase_distribution"] == dict.fromkeys(phrases, 6)
    assert [(text, voices) for text, voices, _ in model.batches] == [
        (phrase, ("voice0", "voice1", "voice2", "voice3")) for phrase in phrases
    ]
    batched = [row for row in rows if "cuda_batch" in row]
    assert {row["slot"] for row in batched} == set(range(8))
    for row in batched:
        slot = row["slot"]
        assert row["text"] == phrases[slot % 2] and row["voice_family_id"] == f"voice{slot // 2}"
        expected_slots = list(range(slot % 2, 8, 2))
        assert row["cuda_batch"]["slots"] == expected_slots
        seed = row["cuda_batch"]["seed"]
        waveform = SeededQwen.waveform((seed + expected_slots.index(slot)) % 2**32)
        _, qc, converted = tts.check_short_clone_audio(waveform, 16000, 51, row["batch_termination"])
        expected_pcm, _ = tts.encode_candidate_pcm16(converted, qc)
        saved_pcm, _ = sf.read(hard_project.parent / row["path"], dtype="int16")
        np.testing.assert_array_equal(saved_pcm, expected_pcm)
    failed, = [row for row in rows if not row["technical_pass"]]
    retried, = [row for row in rows if row["attempt"] == 1]
    assert failed["slot"] == retried["slot"] == 2
    assert failed["qc"]["reasons"] == ["generation_length_limit"]
    assert failed["qc"]["generation_termination"] == "token_limit"
    assert retried["technical_pass"] and retried["qc"]["generation_termination"] == "eos"
    assert [entry for entry in model.singles if entry[-1]] == [(phrases[0], "voice1", retried["seed"], True)]
    assert len(model.singles) == 5
    assert all(digest(hard_project.parent / row["path"]) == row["sha256"] for row in rows)


@pytest.mark.parametrize("available", [False, True])
def test_no_mixed_batch_uses_scalar_with_the_original_seeds(hard_project, available):
    model = MixedVoiceModel(available=available, decline_batch=True)
    result = hard_tts.generate_hard(hard_project, PHRASES[:1], model=model, samples_per_phrase=4)
    rows = read_rows(result["candidates"])

    assert result["full_target_reached"] and result["attempts"] == 4
    assert len(model.batches) == int(available)
    assert [(row["voice_family_id"], row["seed"]) for row in rows] == [
        (voice, seed) for _, voice, seed, _ in model.singles
    ]
    assert all("cuda_batch" not in row for row in rows)


def test_pause_preserves_the_published_mixed_batch_and_resumes_at_next_group(hard_project):
    model = MixedVoiceModel()
    phrases = PHRASES[:2]
    directory = hard_project.parent / "data/generated/hard"
    with pytest.raises(hard_tts.HardGenerationPaused):
        hard_tts.generate_hard(hard_project, phrases, model=model, samples_per_phrase=4,
                               pause_requested=lambda: bool(model.batches))
    journal = directory / "candidates.jsonl"
    saved = read_rows(journal)
    assert len(saved) == 4 and all(row["text"] == phrases[0] for row in saved)
    frozen_contract = digest(directory / "contract.json")
    assert len(model.batches) == 1 and model.singles == []

    resumed = MixedVoiceModel()
    result = hard_tts.generate_hard(hard_project, phrases, model=resumed, samples_per_phrase=4)

    assert result["full_target_reached"] and result["attempts"] == 8
    assert read_rows(journal)[:4] == saved and digest(directory / "contract.json") == frozen_contract
    assert [text for text, _, _ in resumed.batches] == [phrases[1]] and resumed.singles == []


def test_torn_mixed_batch_journal_recovers_without_resynthesizing(hard_project, monkeypatch):
    model = MixedVoiceModel(reject_one=True)
    directory = hard_project.parent / "data/generated/hard"
    original_finish = tts._finish_call_batch

    def interrupt(output, intent):
        write_torn_batch(output, intent)
        raise KeyboardInterrupt

    monkeypatch.setattr(tts, "_finish_call_batch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        hard_tts.generate_hard(hard_project, PHRASES[:1], model=model, samples_per_phrase=4)
    pending = directory / "pending_cuda_batch.json"
    staged = read_json(pending)["rows"]
    assert len(model.batches) == 1 and len(staged) == 4

    monkeypatch.setattr(tts, "_finish_call_batch", original_finish)
    resumed = MixedVoiceModel()
    result = hard_tts.generate_hard(hard_project, PHRASES[:1], model=resumed, samples_per_phrase=4)

    assert result["full_target_reached"] and result["attempts"] == 5 and not pending.exists()
    rows = read_rows(result["candidates"])
    assert rows[:4] == staged
    assert [(row["slot"], row["attempt"]) for row in rows] == [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)]
    assert resumed.batches == [] and len(resumed.singles) == 1 and resumed.singles[0][-1]


@pytest.mark.parametrize("old_implementation,old_qc", [
    ("bd75d7e028246ac3da21722510b5250691461f968a14776d8511720439109350",
     "35e797140ce4bc13235a0d9e259b129d63b7bbd66aac23fe2b1f8f624d9808bf"),
    ("bbe435b58a3e21e4f6d5e24d3c6f809c7d0c3d5df02610b4343e35021cc39c37",
     "4b80f8a5561ba98454b8e6fd8b0449301b6bf3852d6eb78b8f3af043b52bb488"),
])
def test_previous_release_receipt_and_audio_survive_resume(hard_project, old_implementation, old_qc):
    model = MixedVoiceModel(available=False)
    root = hard_project.parent
    directory = root / "data/generated/hard"
    with pytest.raises(hard_tts.HardGenerationPaused):
        hard_tts.generate_hard(hard_project, PHRASES[:1], model=model, samples_per_phrase=4,
                               pause_requested=lambda: bool(model.singles))
    contract_path = directory / "contract.json"
    contract = read_json(contract_path)
    contract.pop("hard_batch_policy", None)
    contract["implementation"]["sha256"] = old_implementation
    contract["technical_qc_implementation"]["sha256"] = old_qc
    write_json(contract_path, contract)
    journal = directory / "candidates.jsonl"
    first, = read_rows(journal)
    first["generation_contract"] = {"path": contract_path.relative_to(root).as_posix(), "sha256": digest(contract_path)}
    first["generation_contract_id"] = identity(contract)
    journal.write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")
    frozen_contract = contract_path.read_bytes()

    resumed = MixedVoiceModel()
    result = hard_tts.generate_hard(hard_project, PHRASES[:1], model=resumed, samples_per_phrase=4)

    assert result["full_target_reached"] and result["attempts"] == 4
    assert contract_path.read_bytes() == frozen_contract and read_rows(journal)[0] == first
    assert digest(root / first["path"]) == first["sha256"]
    assert resumed.batches == [] and len(resumed.singles) == 3
