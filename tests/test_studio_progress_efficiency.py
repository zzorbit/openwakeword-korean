"""Fresh source contracts and live checkpoint counters without corpus reads."""
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import write_json
from test_korean_library_studio_evaluation_progress import _fixture


def test_contract_graph_cache_rechecks_hashes_and_does_not_share_mutable_results(monkeypatch):
    studio._EXECUTION_CONTRACT_CACHE.clear()
    original_parse, original_digest = studio.ast.parse, studio.digest
    parsed, hashed = [], []
    monkeypatch.setattr(studio.ast, "parse", lambda *args, **kwargs: (parsed.append(True), original_parse(*args, **kwargs))[1])
    monkeypatch.setattr(studio, "digest", lambda path: (hashed.append(str(path)), original_digest(path))[1])
    first = studio._execution_contract()
    assert parsed
    parses, reads = len(parsed), len(hashed)
    first["files"]["train.py"] = "not-a-real-contract"
    second = studio._execution_contract()
    assert len(parsed) == parses and len(hashed) > reads
    assert second["files"]["train.py"] != "not-a-real-contract"


def test_changed_source_digest_cannot_reuse_a_previous_contract(monkeypatch):
    before = studio._execution_contract()
    original = studio.digest
    monkeypatch.setattr(studio, "digest", lambda path: "f" * 64 if Path(path).name == "train.py" else original(path))
    after = studio._execution_contract()
    assert after["computation_sha256"] != before["computation_sha256"]
    assert after["stages"]["training"] != before["stages"]["training"]


def _active_checkpoint(tmp_path, monkeypatch, count=48032):
    manager, state, _ = _fixture(tmp_path, b"Evaluate development: 48000/48137 (CUDA batch=128, CPU rechecked=443)\n")
    state["job"]["started_at"] = "2026-01-01T00:00:00+00:00"
    manager._worker_owner = lambda job: {"pid": 123, "creation_identity": "current-birth"}
    monkeypatch.setattr(studio, "process_identity", lambda pid: "current-birth" if pid == 123 else None)
    folder = Path(state["project"]).parent / "reports/studio_evaluation"
    write_json(folder / ".writer.lock", {"pid": 123, "started_at": "2026-01-01T00:00:01+00:00"})
    write_json(folder / "_evaluation_work/checkpoint.json", {"schema_version": 1, "binding": "a" * 64, "committed_sessions": count})
    return manager, state, folder


def test_evaluation_uses_current_workers_durable_progress_between_log_updates(tmp_path, monkeypatch):
    manager, state, _ = _active_checkpoint(tmp_path, monkeypatch)
    actual = manager.describe(state)
    assert actual["progress"]["done"] == 48032 and actual["progress"]["total"] == 48137
    assert actual["progress"]["source"] == "current_task_checkpoint"
    assert actual["progress"]["phase"] == "scoring"


@pytest.mark.parametrize("change", ["different_pid", "recycled_pid", "old_lock", "too_large", "behind_log"])
def test_unrelated_or_invalid_checkpoint_never_advances_current_task(tmp_path, monkeypatch, change):
    manager, state, folder = _active_checkpoint(tmp_path, monkeypatch)
    if change == "different_pid":
        write_json(folder / ".writer.lock", {"pid": 456, "started_at": "2026-01-01T00:00:01+00:00"})
    elif change == "recycled_pid":
        monkeypatch.setattr(studio, "process_identity", lambda pid: "different-birth")
    elif change == "old_lock":
        write_json(folder / ".writer.lock", {"pid": 123, "started_at": "2025-01-01T00:00:01+00:00"})
    else:
        write_json(folder / "_evaluation_work/checkpoint.json", {"schema_version": 1, "binding": "a" * 64,
            "committed_sessions": 48138 if change == "too_large" else 47000})
    assert manager.describe(state)["progress"]["done"] == 48000
