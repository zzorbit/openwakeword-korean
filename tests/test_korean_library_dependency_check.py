"""A matching package version must be checked before loading TTS binaries."""
import importlib.metadata
import pytest

from korean_openwakeword.benchmark import doctor
from korean_openwakeword.cli import main


def test_tts_version_mismatch_is_actionable_and_fails_cli(monkeypatch, capsys):
    original = importlib.metadata.version
    def installed(name):
        return {"torch": "2.8.0+cu128", "torchaudio": "2.11.0", "qwen-tts": "0.1.1"}.get(name) or original(name)
    monkeypatch.setattr(importlib.metadata, "version", installed)
    report = doctor(include_tts=True)
    assert report["tts_check"]["ready"] is False
    assert any("torchaudio==2.8.0" in reason for reason in report["tts_check"]["reasons"])
    assert report["tts_check"]["model_download_or_generation"] is False
    with pytest.raises(SystemExit) as result:
        main(["check", "--tts"])
    assert result.value.code == 1
    assert "torchaudio==2.8.0" in capsys.readouterr().err


def test_base_check_does_not_require_tts_extras(monkeypatch):
    original = importlib.metadata.version
    def installed(name):
        if name in ("qwen-tts", "torchaudio"):
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(importlib.metadata, "version", installed)
    report = doctor()
    assert report["versions"]["qwen-tts"] is None
    assert "tts_check" not in report
