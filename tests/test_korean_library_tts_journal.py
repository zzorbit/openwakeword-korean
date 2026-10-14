"""Bulk TTS bookkeeping preserves accepted PCM and deterministic retries."""
from collections import Counter
import random
import shutil

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import tts
from korean_openwakeword.common import append_row, digest, identity, read_json, read_rows, write_json
from korean_openwakeword.project import create_project


def approved_project(path, method, count=6):
    project = create_project(path, "보라야", "bora", positive_originals=count)
    config = read_json(project)
    approval = read_json(path / "approval.json")
    approval.update(state="accepted", generation_policy_approved=True,
                    generation_method=method, tts_settings_id=identity(config["tts"]),
                    accepted_reference_families=["voice_0", "voice_1"])
    write_json(path / "approval.json", approval)
    if method == "clone":
        directory = path / "data" / "references" / "train"
        directory.mkdir(parents=True)
        for number in range(2):
            audio = directory / f"voice_{number}.wav"
            sf.write(audio, np.sin(np.arange(8000) / (20 + number)) * .1, 16000, subtype="FLOAT")
            append_row(directory / "references.jsonl", {
                "voice_family_id": f"voice_{number}", "path": str(audio.relative_to(path)),
                "sha256": digest(audio), "text": "합성 참조 음성입니다.",
                "provenance": "qwen_synthetic_reference", "human_reference_used": False,
                "split": "train", "status": "accepted",
            })
    return project


class SeededQwen:
    """Deterministic Qwen stand-in with one bad duration and one duplicate."""
    def __init__(self, config, interrupt_at=None):
        self.config = config
        self.interrupt_at = interrupt_at
        self.seeds = []
        self.prompts = []

    def seed(self, slot, attempt):
        return int(identity([self.config["tts"]["seed"], "train", slot, attempt])[:8], 16)

    def create_voice_clone_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        return [object()]

    @staticmethod
    def waveform(seed, frames=9600):
        rng = np.random.RandomState(seed)
        return .15 * np.sin(np.arange(frames) / 20) + rng.normal(0, .002, frames)

    def generate_voice_design(self, text, language, **kwargs):
        assert text == "보라야" and language == "Korean"
        seed = int(np.random.get_state()[1][0])
        self.seeds.append(seed)
        if len(self.seeds) == self.interrupt_at:
            raise RuntimeError("simulated interruption before candidate persistence")
        frames = 144000 if seed == self.seed(0, 0) else 9600
        wave = .15 * np.sin(np.arange(frames) / 20) + np.random.normal(0, .002, frames)
        if seed == self.seed(1, 0):
            wave = self.waveform(self.seed(0, 1))
        return [wave], 16000

    generate_voice_clone = generate_voice_design


def cpu_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def stable_rows(path):
    rows = read_rows(path)
    return [{key: value for key, value in row.items() if key not in ("created_at", "generation_seconds")}
            for row in rows]


@pytest.mark.parametrize("method", ["clone", "design"])
def test_interrupted_retries_match_uninterrupted_pcm_manifest_and_rng(tmp_path, monkeypatch, method):
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    projects = [approved_project(tmp_path / name, method) for name in ("full", "resumed")]
    if method == "clone":
        # FLOAT WAVs include a PEAK timestamp. Reuse the same bound reference
        # bytes so this compares interrupted work with identical inputs.
        queue = "data/references/train/references.jsonl"
        for reference in read_rows(projects[0].parent / queue):
            source = projects[0].parent / reference["path"]
            target = projects[1].parent / reference["path"]
            shutil.copyfile(source, target)
            assert digest(source) == digest(target) == reference["sha256"]
        shutil.copyfile(projects[0].parent / queue, projects[1].parent / queue)
    config = read_json(projects[0])
    full = SeededQwen(config)
    expected = tts.generate(projects[0], method=method, model=full)
    full_rng = np.random.get_state()
    interrupted = SeededQwen(config, interrupt_at=5)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        tts.generate(projects[1], method=method, model=interrupted)
    retry = SeededQwen(config)
    actual = tts.generate(projects[1], method=method, model=retry)
    resumed_rng = np.random.get_state()
    assert full.seeds == interrupted.seeds[:4] + retry.seeds
    assert interrupted.seeds[-1] == retry.seeds[0]
    assert full_rng[0] == resumed_rng[0] and np.array_equal(full_rng[1], resumed_rng[1])
    assert full_rng[2:] == resumed_rng[2:]
    queues = [project.parent / "data/generated/train/candidates.jsonl" for project in projects]
    rows = stable_rows(queues[0])
    assert rows == stable_rows(queues[1])
    assert rows[0]["qc"]["reasons"] == ["duration_outlier"]
    assert rows[2]["qc"]["reasons"] == ["exact_duplicate"]
    assert [r["sample_id"] for r in rows[:4]] == ["train_000000_attempt0", "train_000000_attempt1",
                                                "train_000001_attempt0", "train_000001_attempt1"]
    for row in rows:
        pcm = [sf.read(project.parent / row["path"], dtype="int16")[0] for project in projects]
        assert np.array_equal(*pcm)
    manifest = config["manifests"]["positive_train"]
    assert stable_rows(projects[0].parent / manifest) == stable_rows(projects[1].parent / manifest)
    assert {k: v for k, v in expected.items() if k != "review_queue"} == {
        k: v for k, v in actual.items() if k != "review_queue"}
    assert expected["attempts"] == 8 and expected["technical_pass"] == 6
    assert expected["policy_admitted"] == 6 and expected["human_listened"] == 0
    assert expected["independent_STT_runs"] == 0
    if method == "clone":
        assert len(full.prompts) == 2 and len(interrupted.prompts) == 2 and len(retry.prompts) == 2


def test_bulk_hashes_immutable_settings_once_without_changing_contract_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    project = approved_project(tmp_path / "word", "design", count=20)
    model = SeededQwen(read_json(project))
    calls = Counter()
    original_identity = tts.identity

    def measured_identity(value):
        if isinstance(value, dict):
            calls["contract" if "references" in value else "tts_settings"] += 1
        return original_identity(value)

    monkeypatch.setattr(tts, "identity", measured_identity)
    tts.generate(project, method="design", model=model)
    queue = project.parent / "data/generated/train/candidates.jsonl"
    rows = read_rows(queue)
    assert len(rows) == 22
    assert calls == {"tts_settings": 1, "contract": 1}
    contract = read_json(queue.parent / "contract.json")
    assert all(r["generation_contract_id"] == identity(contract) for r in rows)
    assert all(r["tts_settings_id"] == identity(read_json(project)["tts"]) for r in rows)


def test_completed_candidates_are_reused_and_changed_contract_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    project = approved_project(tmp_path / "word", "design", count=2)
    model = SeededQwen(read_json(project))
    first = tts.generate(project, method="design", model=model)
    queue = project.parent / "data/generated/train/candidates.jsonl"
    before = queue.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Completed candidates must not load or rerun TTS")

    monkeypatch.setattr(tts, "_load_model", forbidden)
    monkeypatch.setattr(tts, "_seed", forbidden)
    again = tts.generate(project, method="design")
    assert again == first and queue.read_bytes() == before
    contract = read_json(queue.parent / "contract.json")
    contract["tts"]["seed"] += 1
    write_json(queue.parent / "contract.json", contract)
    with pytest.raises(ValueError, match="settings changed"):
        tts.generate(project, method="design")
    assert queue.read_bytes() == before
