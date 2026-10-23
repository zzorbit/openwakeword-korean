"""Evaluation polling reads only the current task's bounded stdout tail."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio


def _fixture(tmp_path, content=b"", *, task="active-evaluation"):
    root = tmp_path / "word"
    directory = root / ".studio/jobs" / task
    directory.mkdir(parents=True)
    log = directory / "output.log"
    log.write_bytes(content)
    manager = object.__new__(studio.StudioManager)
    manager._implementation_id = "fixture-implementation"
    manager._timing_receipts = {}
    manager._rates = {}
    manager._job_alive = lambda job: True
    manager.review_descriptors = lambda state: []
    state = {"id": "word", "wakeword": "보라야", "preset": "check", "status": "running",
        "implementation_id": manager._implementation_id,
        "project": str(root / "project.json"), "index": 1,
        "plan": [{"id": "training", "label": "학습", "kind": "task"},
                 {"id": "evaluation", "label": "성능 평가", "kind": "task"},
                 {"id": "results", "label": "결과", "kind": "task"}],
        "settings": {}, "results": {}, "next_action": "작업을 진행하고 있습니다.",
        "job": {"task_id": task, "stage": "evaluation", "directory": str(directory), "log": str(log)}}
    return manager, state, log


def test_cuda_evaluation_shows_its_session_gauge_separately_from_workflow_steps(tmp_path):
    manager, state, _ = _fixture(tmp_path, b"Evaluate development: 94752/112137 (CUDA batch=128, CPU rechecked=292)\n")
    before = deepcopy(state)
    actual = manager.describe(state)
    assert actual["progress"] == {"done": 94752, "total": 112137, "unit": "개 음원",
        "label": "성능 평가 · GPU", "phase": "scoring", "source": "current_task_log",
        "task_id": "active-evaluation", "backend": "cuda", "batch_size": 128, "cpu_rechecked_sessions": 292}
    assert actual["stage_progress"] == {"done": 1, "total": 3, "unit": "완료 단계"}
    assert actual["progress_note"] == "음원 수 기준이며, 긴 비호출 음원은 더 오래 걸립니다."
    assert actual["eta_seconds"] is None and manager._rates == {}
    assert state == before


def test_old_cpu_lines_and_latest_valid_count_are_supported(tmp_path):
    manager, state, _ = _fixture(tmp_path, b"Evaluate development: 100/1000 (CPU workers=4)\r\n"
        b"Evaluate development: 200/1000 (CPU workers=8)\r\n"
        b"Evaluate development: 2001/1000 (CPU workers=8)\r\n"
        b"Evaluate development: 900/0 (CPU workers=8)\r\n"
        b"Evaluate development: 300/1000 (CPU wor")
    result = manager.describe(state)
    assert result["progress"]["done"] == 200 and result["progress"]["total"] == 1000
    assert result["progress"]["backend"] == "cpu" and result["progress"]["workers"] == 8


@pytest.mark.parametrize("content", [b"", b"Checking model and audio hashes...\n", b"Traceback: input validation failed\n"])
def test_preflight_has_a_validation_label_without_inventing_zero_percent(tmp_path, content):
    manager, state, _ = _fixture(tmp_path, content)
    result = manager.describe(state)
    assert result["progress"]["done"] is None and result["progress"]["total"] is None
    assert result["progress"]["phase"] == "validation"
    assert result["progress"]["label"] == "평가 음원 검증 중"
    assert result["next_action"] == "평가 음원의 무결성과 형식을 확인하고 있습니다."
    assert result["eta_seconds"] is None


def test_complete_session_count_still_means_result_publication_not_finished_workflow(tmp_path):
    manager, state, _ = _fixture(tmp_path, b"Evaluate development: 112137/112137 (CUDA batch=128, CPU rechecked=1044)\n")
    result = manager.describe(state)
    assert result["progress"]["done"] == result["progress"]["total"]
    assert result["progress"]["phase"] == "finalizing"
    assert result["next_action"] == "채점을 마쳤으며 결과를 정리하고 있습니다."
    assert result["status"] == "running" and result["stage_progress"]["done"] == 1
    assert "results" not in result


def test_old_attempt_log_is_never_used_for_the_current_task(tmp_path):
    manager, state, current = _fixture(tmp_path)
    prior = current.parent.parent / "older-evaluation" / "output.log"
    prior.parent.mkdir()
    prior.write_bytes(b"Evaluate development: 1000/1000 (CPU workers=4)\n")
    state["last_job"] = {"task_id": "older-evaluation", "stage": "evaluation", "log": str(prior), "directory": str(prior.parent)}
    assert manager.describe(state)["progress"]["done"] is None
    # A current task's descriptor cannot relabel another task's log as its own.
    state["job"]["log"] = str(prior)
    assert manager.describe(state)["progress"]["done"] is None


def test_periodic_view_reads_at_most_one_24k_log_tail_and_no_large_artifacts(monkeypatch, tmp_path):
    tail = b"Evaluate development: 5100/112137 (CPU workers=4)\n"
    manager, state, log = _fixture(tmp_path, b"old output\n" * 200000 + tail)
    original_open = Path.open
    reads = []
    class Reader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def seek(self, *args):
            return self.handle.seek(*args)
        def read(self, size=-1):
            assert 0 < size <= 24000
            result = self.handle.read(size)
            reads.append(len(result))
            return result
    def guarded_open(path, *args, **kwargs):
        assert path == log, "Polling must not open a manifest, score array, WAV or prior task log"
        return Reader(original_open(path, *args, **kwargs))
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(studio, "read_json", lambda *args: pytest.fail("No evaluation JSON may be read for progress"))
    monkeypatch.setattr(studio, "read_rows", lambda *args: pytest.fail("No audio manifest may be scanned for progress"))
    monkeypatch.setattr(studio, "digest", lambda *args: pytest.fail("Polling must not hash artifacts"))
    monkeypatch.setattr(Path, "glob", lambda *args: pytest.fail("Polling must not search evaluation attempts"))
    for _ in range(3):
        assert manager.describe(state)["progress"]["done"] == 5100
    assert reads == [24000, 24000, 24000]


def test_log_api_uses_the_same_current_task_tail_and_preserves_last_job_fallback(tmp_path):
    manager, state, log = _fixture(tmp_path, b"current output\n")
    manager._load = lambda project_id: state
    assert manager.log("word") == "current output\n"
    state["last_job"], state["job"] = state["job"], None
    assert manager.log("word") == "current output\n"
    log.unlink()
    assert manager.log("word") == "아직 실행 로그가 없습니다."
