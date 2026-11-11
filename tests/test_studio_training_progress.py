"""Preparation progress is task-owned and never invents learning updates."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio, studio_worker
from test_korean_library_studio_evaluation_progress import _fixture


@pytest.mark.parametrize("phase", ["feature_validation", "device_selection", "model_initialization", "gpu_cache_loading"])
def test_training_preparation_is_separate_from_optimizer_updates(tmp_path, phase):
    manager, state, _ = _fixture(tmp_path)
    manager._progress_snapshots = {}
    state.update(index=0)
    state["job"]["stage"] = "training"
    state["settings"]["updates"] = 10
    progress = {"kind": "training_preparation_v1", "task_id": state["job"]["task_id"],
                "phase": phase, "label": "학습 입력 준비", "done": 12, "total": 24, "unit": "bytes"}
    studio.write_json(Path(state["job"]["directory"]) / "progress.json", progress)
    before = deepcopy(state)
    view = manager.describe(state)
    assert view["training_preparation"] == progress
    assert view["progress"] == progress
    assert view["stage_progress"] == {"done": 0, "total": 3, "unit": "완료 단계"}
    assert state == before


@pytest.mark.parametrize("changed", ["task_id", "kind", "phase"])
def test_old_or_finished_preparation_cannot_replace_current_learning_progress(tmp_path, changed):
    manager, state, _ = _fixture(tmp_path)
    manager._progress_snapshots = {}
    state.update(index=0)
    state["job"]["stage"] = "training"
    state["settings"]["updates"] = 10
    progress = {"kind": "training_preparation_v1", "task_id": state["job"]["task_id"],
                "phase": "gpu_cache_loading", "label": "GPU 적재", "done": 10, "total": 20, "unit": "bytes"}
    progress[changed] = "training" if changed == "phase" else "old"
    studio.write_json(Path(state["job"]["directory"]) / "progress.json", progress)
    root = Path(state["project"]).parent
    studio.write_json(root / "runs/studio_run/member1_seed1/progress.json", {"completed_updates": 8})
    view = manager.describe(state)
    assert "training_preparation" not in view
    assert view["progress"]["done"] == 8 and view["progress"]["total"] == 30


@pytest.mark.parametrize("phase", ["feature_cache_verification", "gpu_cache_loading", "training_finalizing"])
def test_current_worker_training_telemetry_replaces_updates_only_during_preparation_or_finalizing(tmp_path, monkeypatch, phase):
    manager, state, _ = _fixture(tmp_path, task="training-worker")
    state["index"] = 0
    state["settings"]["updates"] = 10
    job = state["job"]
    job.update(stage="training", request_sha256="a" * 64, started_at="2026-01-01T00:00:00+00:00")
    directory = Path(job["directory"])
    root = Path(state["project"]).parent
    studio.write_json(root / "runs/studio_run/member1_seed1/progress.json", {"completed_updates": 8})
    owner = {"task_id": job["task_id"], "request_sha256": job["request_sha256"],
             "pid": 123, "creation_identity": "training-birth"}
    studio.write_json(directory / "owner.json", owner)
    monkeypatch.setattr(studio, "process_identity", lambda pid: owner["creation_identity"] if pid == owner["pid"] else None)
    progress = {**owner, "kind": "studio_task_progress_v1", "stage": "training", "phase": phase,
                "updated_at": "2026-01-01T00:00:01+00:00", "label": "학습 입력 검사",
                "done": 1024, "total": 4096, "unit": "bytes", "eta_eligible": False}
    studio.write_json(directory / "progress.json", progress)
    view = manager.describe(state)
    assert view["training_preparation"] == progress
    assert view["progress"] == progress
    for change in ({"creation_identity": "old-birth"}, {"phase": "training"}):
        studio.write_json(directory / "progress.json", {**progress, **change})
        view = manager.describe(state)
        assert "training_preparation" not in view
        assert view["progress"]["done"] == 8 and view["progress"]["total"] == 30


def test_worker_throttles_chunk_progress_but_publishes_phase_and_completion(monkeypatch, tmp_path):
    clock = iter([0, .1, .5, 1.1, 1.2, 1.3])
    monkeypatch.setattr(studio_worker.time, "monotonic", lambda: next(clock))
    written = []
    monkeypatch.setattr(studio_worker, "write_json", lambda path, row: written.append((path, row)))
    report = studio_worker._task_progress_writer({"task_id": "current", "stage": "training"}, tmp_path, "sha")
    for value in (0, 10, 20, 90, 100):
        report({"phase": "gpu_cache_loading", "label": "GPU 적재", "unit": "bytes", "done": value, "total": 100})
    report({"phase": "training", "label": "모델 학습", "done": 0, "total": 10, "unit": "학습 업데이트"})
    assert [row["done"] for _, row in written] == [0, 90, 100, 0]
    assert all(path == tmp_path / "progress.json" for path, _ in written)
    assert all(row["kind"] == "studio_task_progress_v1" and row["task_id"] == "current" for _, row in written)
