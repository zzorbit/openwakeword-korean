import os
from pathlib import Path
import sys
import tempfile
import types
import subprocess

import pytest

from korean_openwakeword.storage import (
    local_tts_snapshot, project_storage, storage_environment, storage_paths,
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell 5.1 encoding regression")
def test_storage_script_is_readable_in_windows_powershell_51(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "Use-ProjectStorage.ps1"
    environment = dict(os.environ, KOWW_TEST_STORAGE_SCRIPT=str(script), KOWW_TEST_STORAGE_ROOT=str(tmp_path))
    shell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(shell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        ". $env:KOWW_TEST_STORAGE_SCRIPT -ProjectRoot $env:KOWW_TEST_STORAGE_ROOT"],
        env=environment, capture_output=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.startswith("프로젝트 캐시: " + str(tmp_path / "storage/runtime_cache"))


def test_caches_and_previously_cached_temp_are_project_local_then_restored(tmp_path, monkeypatch):
    monkeypatch.delenv("KOWW_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HF_HOME", "previous_user_cache")
    monkeypatch.setattr(tempfile, "tempdir", "previous_system_temp")
    constants = types.SimpleNamespace(HF_XET_CACHE="previous_xet", HF_HUB_CACHE="previous_hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    root = tmp_path / "project"
    with pytest.raises(RuntimeError):
        with storage_environment(root) as paths:
            assert paths["root"] == root / "cache"
            assert Path(tempfile.gettempdir()) == paths["temp"]
            assert os.environ["HF_HOME"] == str(paths["hf_home"])
            assert os.environ["HF_HUB_CACHE"] == str(paths["hf_hub"])
            assert constants.HF_XET_CACHE == str(paths["hf_xet"])
            assert all(p.is_dir() and p.is_relative_to(root) for p in paths.values())
            raise RuntimeError("intentional loader failure")
    assert os.environ["HF_HOME"] == "previous_user_cache"
    assert tempfile.tempdir == "previous_system_temp"
    assert constants.HF_XET_CACHE == "previous_xet"
    assert constants.HF_HUB_CACHE == "previous_hub"


def test_explicit_shared_cache_and_codex_rejection(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(shared))
    assert storage_paths(tmp_path / "word")["root"] == shared
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / ".codex" / "cache" / "wakeword"))
    with pytest.raises(ValueError, match=".codex"):
        storage_paths(tmp_path)


def test_remote_snapshot_uses_explicit_cache_and_pinned_revision(tmp_path, monkeypatch):
    calls = []
    snapshot = tmp_path / "cache" / "huggingface" / "hub" / "snapshot"
    monkeypatch.delenv("KOWW_CACHE_ROOT", raising=False)
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    assert local_tts_snapshot("Qwen/model", {"model_revisions": {"Qwen/model": "abc123"}}, tmp_path) == str(snapshot)
    assert calls == [{"repo_id": "Qwen/model", "revision": "abc123", "cache_dir": str(tmp_path / "cache/huggingface/hub")}]
    with pytest.raises(ValueError, match="pinned"):
        local_tts_snapshot("Qwen/model", {}, tmp_path)
    local = tmp_path / "existing_model"
    local.mkdir()
    assert local_tts_snapshot(str(local), {}, tmp_path) == str(local)
    assert len(calls) == 1  # local assets never cause a download


def test_project_api_enters_storage_before_loading_dependencies(tmp_path, monkeypatch):
    monkeypatch.delenv("KOWW_CACHE_ROOT", raising=False)
    @project_storage
    def actual(project, *, failure=False):
        assert Path(os.environ["TORCH_HOME"]) == Path(project).parent / "cache/torch"
        return tempfile.gettempdir()
    result = actual(tmp_path / "word/project.json")
    assert Path(result) == tmp_path / "word/cache/tmp"


def test_qwen_subloaders_receive_a_local_pinned_model_not_a_remote_repo(tmp_path, monkeypatch):
    from korean_openwakeword.tts import _load_model
    monkeypatch.delenv("KOWW_CACHE_ROOT", raising=False)
    snapshot = tmp_path / "pinned_snapshot"
    snapshot.mkdir()
    calls = []
    def download(**kwargs):
        assert kwargs["revision"] == "fixed_revision"
        assert Path(kwargs["cache_dir"]).is_relative_to(tmp_path)
        return str(snapshot)
    def load(model_path, **kwargs):
        assert Path(model_path).is_dir()
        assert Path(os.environ["HF_XET_CACHE"]).is_relative_to(tmp_path)
        calls.append((model_path, kwargs))
        return "loaded"
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=load)))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32="float32", bfloat16="bfloat16", cuda=types.SimpleNamespace(is_available=lambda: False)))
    assert _load_model("Qwen/model", {"device": "auto", "attention": "sdpa", "model_revisions": {"Qwen/model": "fixed_revision"}}, tmp_path) == "loaded"
    assert calls == [(str(snapshot), {"device_map": "cpu", "dtype": "float32", "attn_implementation": "sdpa"})]
