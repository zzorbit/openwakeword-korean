"""The preceding recovered call bank stays readable and append-only."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import tts
from korean_openwakeword.call_generation_history import PREVIOUS_TTS, PREVIOUS_REFERENCE
from korean_openwakeword.common import digest, read_json, read_rows, write_json
from test_call_generation_recovery import legacy_run, RecoveryModel
from test_korean_library_tts_journal import approved_project, cpu_seed


def previous_release(monkeypatch):
    original = tts.digest
    def pinned(path):
        path = Path(path)
        if path == Path(tts.__file__):
            return PREVIOUS_TTS
        if path == Path(tts.clone_reference.__file__):
            return PREVIOUS_REFERENCE
        return original(path)
    monkeypatch.setattr(tts, "digest", pinned)


def test_completed_previous_recovery_is_read_only_and_never_loads_model(tmp_path, monkeypatch):
    project, config, refs, scope, output = legacy_run(tmp_path, count=6)
    original_summary = b'{"technical_pass":5,"target":6,"old_evidence":true}\n'
    (output / "summary.json").write_bytes(original_summary)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    with monkeypatch.context() as old:
        previous_release(old)
        assert tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(), staged_scope=scope)["technical_pass"] == 6
    before = {p: digest(p) for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(tts, "_load_model", lambda *a: pytest.fail("loaded model for a complete old bank"))
    assert tts.generate(project, "train", 6, True, "clone", staged_scope=scope)["technical_pass"] == 6
    assert {p: digest(p) for p in output.rglob("*") if p.is_file()} == before
    assert (output / "summary.json").read_bytes() == original_summary
    from korean_openwakeword.pilot import _passing_slots
    assert len(_passing_slots(project.parent, config, "train", refs, output, scope)[1]) == 6


def test_partial_previous_recovery_uses_new_receipt_and_preserves_old_bytes(tmp_path, monkeypatch):
    project, config, refs, scope, output = legacy_run(tmp_path, count=6)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    class Interrupted(RecoveryModel):
        def generate_voice_clone(self, **options):
            if self.calls:
                raise RuntimeError("interrupted previous release")
            return super().generate_voice_clone(**options)
    with monkeypatch.context() as old:
        previous_release(old)
        with pytest.raises(RuntimeError, match="interrupted previous"):
            tts.generate(project, "train", 6, True, "clone", model=Interrupted(fail=True), staged_scope=scope)
    contract, update = output / "contract.json", output / "generation_update.json"
    saved = {p: p.read_bytes() for p in (contract, update, output / "candidates.jsonl")}
    result = tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(), staged_scope=scope)
    assert result["technical_pass"] == 6
    assert contract.read_bytes() == saved[contract] and update.read_bytes() == saved[update]
    assert (output / "candidates.jsonl").read_bytes().startswith(saved[output / "candidates.jsonl"])
    receipts = list((output / "generation_updates").glob("*.json"))
    assert len(receipts) == 1
    new = read_rows(output / "candidates.jsonl")[-1]
    assert new["attempt"] == 4 and new["generation_update"]["sha256"] == digest(receipts[0])
    assert tts.generate(project, "train", 6, True, "clone", staged_scope=scope)["technical_pass"] == 6
    changed = deepcopy(read_json(receipts[0]))
    changed["previous_journal_prefix_sha256"] = "0" * 64
    write_json(receipts[0], changed)
    with pytest.raises(ValueError, match="journal prefix changed"):
        tts.generate(project, "train", 6, True, "clone", staged_scope=scope)


def test_unknown_previous_recovery_target_is_rejected(tmp_path, monkeypatch):
    project, _, _, scope, output = legacy_run(tmp_path, count=6)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    with monkeypatch.context() as old:
        previous_release(old)
        tts.generate(project, "train", 6, True, "clone", model=RecoveryModel(), staged_scope=scope)
    path = output / "generation_update.json"
    changed = read_json(path)
    changed["target_contract"]["implementation_sha256"] = "0" * 64
    write_json(path, changed)
    with pytest.raises(ValueError, match="receipt or journal prefix changed"):
        tts.generate(project, "train", 6, True, "clone", staged_scope=scope)


def test_previous_current_contract_appends_without_inventing_an_old_recovery(tmp_path, monkeypatch):
    project = approved_project(tmp_path / "word", "clone", count=6)
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    class Interrupted(RecoveryModel):
        def generate_voice_clone(self, **options):
            if self.calls:
                raise RuntimeError("interrupted fresh previous bank")
            return super().generate_voice_clone(**options)
    with monkeypatch.context() as old:
        previous_release(old)
        with pytest.raises(RuntimeError, match="fresh previous"):
            tts.generate(project, "train", 6, True, "clone", model=Interrupted())
    output = project.parent / "data/pilot/train"
    contract, journal = output / "contract.json", output / "candidates.jsonl"
    contract_bytes, prefix = contract.read_bytes(), journal.read_bytes()
    assert not (output / "generation_update.json").exists()
    assert tts.generate(project, "train", 6, True, "clone", model=RecoveryModel())["technical_pass"] == 6
    assert contract.read_bytes() == contract_bytes and journal.read_bytes().startswith(prefix)
    assert not (output / "generation_update.json").exists()
    assert len(list((output / "generation_updates").glob("*.json"))) == 1
