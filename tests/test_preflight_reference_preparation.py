"""Replacement reference generation preserves raw output before edge cleanup."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import pilot, tts
from korean_openwakeword.common import digest, identity, read_json, write_json
from test_korean_library_pilot import reference_fixture
from test_korean_library_tts_journal import cpu_seed


class Design:
    def __init__(self, *, silent=False): self.calls, self.silent = [], silent
    def generate_voice_design(self, **options):
        self.calls.append(options)
        voice = np.zeros(16000) if self.silent else .13 * np.sin(np.arange(16000) / 18)
        return [np.concatenate([voice, np.zeros(12 * 16000)])], 16000


def test_replacement_trims_new_edge_silence_and_cached_record_verifies_without_generation(tmp_path, monkeypatch):
    project, refs = reference_fixture(tmp_path, 1)
    root, config, original = project.parent, read_json(project), refs[0]
    before_source = digest(root / original["path"])
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    model = Design()
    replacement = pilot._alternate_reference(root, config, original, "round2", lambda: model, {original["sha256"]})
    assert len(model.calls) == 1 and replacement["replaces_reference_sha256"] == original["sha256"]
    preparation = replacement["reference_preparation"]
    assert preparation["source_duration_s"] == 13 and preparation["prepared_duration_s"] < 1.3
    assert preparation["trimmed"] and sf.info(root / replacement["path"]).duration < 1.3
    assert digest(root / original["path"]) == before_source
    assert (root / preparation["raw_receipt"]["path"]).exists()
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    assert pilot._alternate_reference(root, config, original, "round2", lambda: pytest.fail("cached model load"), {original["sha256"]}) == replacement
    assert before == {str(path): digest(path) for path in root.rglob("*") if path.is_file()}


def test_rejected_raw_attempts_remain_immutable_and_resume_without_model_reload(tmp_path, monkeypatch):
    project, refs = reference_fixture(tmp_path, 1)
    root, config, original = project.parent, read_json(project), refs[0]
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    model = Design(silent=True)
    with pytest.raises(ValueError, match="기술 검사를 통과하지"):
        pilot._alternate_reference(root, config, original, "round2", lambda: model, {original["sha256"]})
    assert len(model.calls) == 3
    raw = root / "data/call_pronunciation/references/raw"
    assert len(list(raw.glob("*.wav"))) == len(list(raw.glob("*.raw.json"))) == 3
    before = {str(path): digest(path) for path in raw.iterdir()}
    with pytest.raises(ValueError, match="기술 검사를 통과하지"):
        pilot._alternate_reference(root, config, original, "round2", lambda: pytest.fail("exhausted raw loaded model"), {original["sha256"]})
    assert before == {str(path): digest(path) for path in raw.iterdir()}


def test_unreceipted_orphan_is_preserved_and_new_distinct_generation_is_recorded(tmp_path, monkeypatch):
    project, refs = reference_fixture(tmp_path, 1)
    root, config, original = project.parent, read_json(project), refs[0]
    token = identity(["round2", original["voice_family_id"], original["sha256"]])[:16]
    orphan = root / f"data/call_pronunciation/references/raw/{token}_attempt0.wav"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"interrupted incomplete output")
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    model = Design()
    replacement = pilot._alternate_reference(root, config, original, "round2", lambda: model, {original["sha256"]})
    assert orphan.read_bytes() == b"interrupted incomplete output"
    assert "_recovery1.wav" in replacement["reference_preparation"]["source"]["path"]
    assert len(model.calls) == 1


def test_cached_replacement_metadata_tamper_blocks_without_regeneration(tmp_path, monkeypatch):
    project, refs = reference_fixture(tmp_path, 1)
    root, config, original = project.parent, read_json(project), refs[0]
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    replacement = pilot._alternate_reference(root, config, original, "round2", lambda: Design(), {original["sha256"]})
    token = identity(["round2", original["voice_family_id"], original["sha256"]])[:16]
    record = root / f"data/call_pronunciation/references/{token}.json"
    replacement["reference_preparation"]["end_frame"] -= 1
    write_json(record, replacement)
    with pytest.raises(ValueError):
        pilot._alternate_reference(root, config, original, "round2", lambda: pytest.fail("tamper loaded model"), {original["sha256"]})
