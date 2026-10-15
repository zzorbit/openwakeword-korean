"""A matching package version must be checked before loading TTS binaries."""
import importlib.metadata
import subprocess
import sys
import pytest

from korean_openwakeword import benchmark
from korean_openwakeword.benchmark import doctor
from korean_openwakeword.cli import main


def test_tts_version_mismatch_is_actionable_and_fails_cli(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "_check_tts_imports", lambda: pytest.fail("Do not import mismatched binaries"))
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
    monkeypatch.setattr(benchmark, "_check_tts_imports", lambda: pytest.fail("Base check must not import TTS"))
    original = importlib.metadata.version
    def installed(name):
        if name in ("qwen-tts", "torchaudio"):
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(importlib.metadata, "version", installed)
    report = doctor()
    assert report["versions"]["qwen-tts"] is None
    assert "tts_check" not in report


OPTIONAL_OUTPUT = (
    "\n********\nWarning: flash-attn is not installed. Will only run the manual PyTorch version. "
    "Please install flash-attn for faster inference.\n********\n"
    "SoX could not be found!\n\nIf you do not have SoX, proceed here:\n"
    "http://sox.sourceforge.net/\nIf you do, double-check your path variables.\n"
)


@pytest.mark.parametrize("shell_warning", [
    "'sox' is not recognized as an internal or external command,\noperable program or batch file.\n",
    "'sox'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는\n배치 파일이 아닙니다.\n",
    "/bin/sh: 1: sox: not found\n",
])
def test_optional_import_noise_is_explained_without_hiding_other_warnings(monkeypatch, capsys, shell_warning):
    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 120
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        return subprocess.CompletedProcess(command, 0, OPTIONAL_OUTPUT.encode(),
            (shell_warning + "Unexpected dependency warning\n").encode())
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    reasons, warnings, notes = benchmark._check_tts_imports()
    assert reasons == []
    assert warnings == "Unexpected dependency warning"
    assert len(notes) == 2 and "12Hz" in notes[0] and "SDPA" in notes[1]
    assert capsys.readouterr() == ("", "")


def test_failed_import_preserves_actual_error_and_optional_logs(monkeypatch):
    failure = OPTIONAL_OUTPUT + "Traceback: OSError: missing torchaudio DLL"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
        subprocess.CompletedProcess(a[0], 1, b"", failure.encode()))
    reasons, warnings, notes = benchmark._check_tts_imports()
    assert "missing torchaudio DLL" in reasons[0]
    assert "SoX could not be found!" in reasons[0]  # no filtering on failure
    assert warnings == "" and notes == []


@pytest.mark.parametrize("error,expected", [
    (subprocess.TimeoutExpired("python", 120), "timed out"),
    (OSError("permission denied"), "permission denied"),
])
def test_unavailable_import_probe_fails(monkeypatch, error, expected):
    def run(*args, **kwargs):
        raise error
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    reasons, warnings, notes = benchmark._check_tts_imports()
    assert expected in reasons[0]
    assert warnings == "" and notes == []


def test_matching_versions_report_actual_probe_failure(monkeypatch):
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name:
        {"torch": "2.8.0+cu128", "torchaudio": "2.8.0+cu128", "qwen-tts": "0.1.1"}.get(name) or original(name))
    monkeypatch.setattr(benchmark, "_check_tts_imports", lambda: (["actual binary failure"], "", []))
    assert doctor(include_tts=True)["tts_check"]["ready"] is False
    with pytest.raises(SystemExit) as result:
        main(["check", "--tts"])
    assert result.value.code == 1


def test_real_child_native_stderr_does_not_leak(tmp_path, monkeypatch, capfd):
    (tmp_path / "torchaudio.py").write_text("", encoding="utf-8")
    stub = (
        "import os\n"
        f"print({OPTIONAL_OUTPUT!r})\n"
        "os.write(2, b\"'sox' is not recognized as an internal or external command,\\noperable program or batch file.\\n\")\n"
        "class Qwen3TTSModel: pass\n"
    )
    (tmp_path / "qwen_tts.py").write_text(stub, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    reasons, warnings, notes = benchmark._check_tts_imports()
    assert reasons == [] and warnings == "" and len(notes) == 2
    assert capfd.readouterr() == ("", "")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CMD uses the active ANSI code page")
def test_korean_cmd_encoding_is_handled(monkeypatch):
    korean = "'sox'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는\n배치 파일이 아닙니다.\n"
    try:
        native_bytes = korean.encode("mbcs")
    except UnicodeEncodeError:
        pytest.skip("The active Windows code page does not encode Korean CMD messages")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k:
        subprocess.CompletedProcess(a[0], 0, OPTIONAL_OUTPUT.encode(), native_bytes))
    reasons, warnings, notes = benchmark._check_tts_imports()
    assert reasons == [] and warnings == "" and len(notes) == 2
