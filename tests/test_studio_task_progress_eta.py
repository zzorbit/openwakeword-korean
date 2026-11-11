"""Task-owned display telemetry and stable processing ETA, without real work."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio
from test_korean_library_studio_evaluation_progress import _fixture


def _owned(tmp_path, monkeypatch, *, stage="development", content=b""):
    manager, state, _ = _fixture(tmp_path, content)
    state["plan"][1]["id"] = state["job"]["stage"] = stage
    state["job"].update(request_sha256="a" * 64, started_at="2026-01-01T00:00:00+00:00")
    directory = Path(state["job"]["directory"])
    studio.write_json(directory / "owner.json", {
        "task_id": state["job"]["task_id"], "request_sha256": "a" * 64,
        "pid": 123, "creation_identity": "current-birth"})
    monkeypatch.setattr(studio, "process_identity", lambda pid: "current-birth" if pid == 123 else None)
    record = {"kind": "studio_task_progress_v1", "task_id": state["job"]["task_id"],
        "request_sha256": "a" * 64, "stage": stage, "pid": 123,
        "creation_identity": "current-birth", "updated_at": "2026-01-01T00:00:01+00:00",
        "phase": "preparation_generation", "label": "평가 자료 생성", "done": 17,
        "total": 48000, "unit": "개 음원", "eta_eligible": True}
    studio.write_json(directory / "progress.json", record)
    return manager, state, record, directory / "progress.json"


def test_development_progress_is_current_worker_owned_and_not_a_completion_receipt(tmp_path, monkeypatch):
    manager, state, record, path = _owned(tmp_path, monkeypatch)
    before = deepcopy(state)
    view = manager.describe(state)
    assert view["progress"] == record
    assert view["stage_progress"]["done"] == 1 and view["status"] == "running"
    record.update(done=48000, phase="preparation_finalizing", eta_eligible=False)
    studio.write_json(path, record)
    view = manager.describe(state)
    assert view["progress"]["done"] == 48000
    assert view["progress"]["phase"] == "preparation_finalizing"
    assert view["status"] == "running" and view["eta_seconds"] is None
    assert state == before


@pytest.mark.parametrize("field,value", [("task_id", "old-task"), ("request_sha256", "b" * 64),
    ("stage", "features"), ("pid", 456), ("creation_identity", "old-birth"),
    ("kind", "old-kind"), ("updated_at", "2025-12-31T23:59:59+00:00")])
def test_unrelated_progress_cannot_replace_current_stage_counter(tmp_path, monkeypatch, field, value):
    manager, state, record, path = _owned(tmp_path, monkeypatch)
    record[field] = value
    studio.write_json(path, record)
    assert manager.describe(state)["progress"]["unit"] == "완료 단계"


def test_recycled_or_gone_owner_cannot_publish_activity(tmp_path, monkeypatch):
    manager, state, _, _ = _owned(tmp_path, monkeypatch)
    for birth in ("recycled-birth", None):
        monkeypatch.setattr(studio, "process_identity", lambda pid: birth)
        assert manager.describe(state)["progress"]["unit"] == "완료 단계"


def test_other_task_directory_and_large_record_are_not_read(tmp_path, monkeypatch):
    manager, state, _, path = _owned(tmp_path, monkeypatch)
    path.write_text(" " * 32769, encoding="utf-8")
    assert manager.describe(state)["progress"]["unit"] == "완료 단계"
    state["job"]["directory"] = str(path.parent.parent / "another-task")
    assert manager.describe(state)["progress"]["unit"] == "완료 단계"


@pytest.mark.parametrize("operation", ["stat", "read"])
def test_sharing_denial_keeps_only_the_same_tasks_last_owned_progress(tmp_path, monkeypatch, operation):
    manager, state, record, path = _owned(tmp_path, monkeypatch)
    assert manager.describe(state)["progress"] == record
    original_stat, original_read = Path.stat, studio.read_json

    def stat(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("progress snapshot is being replaced")
        return original_stat(candidate, *args, **kwargs)

    def read(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("progress snapshot is being replaced")
        return original_read(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat if operation == "stat" else original_stat)
    monkeypatch.setattr(studio, "read_json", read if operation == "read" else original_read)
    view = manager.describe(state)
    assert view["progress"] == record and view["progress_pending"] is True
    assert view["error"] is None
    # A replacement task cannot inherit the previous task's cached display.
    state["job"]["task_id"] = "replacement"
    assert manager.describe(state)["progress"]["unit"] == "완료 단계"


def test_features_preparation_overrides_a_previous_feature_batch_log(tmp_path, monkeypatch):
    manager, state, record, _ = _owned(tmp_path, monkeypatch, stage="features")
    def unexpected_log_read(job):
        pytest.fail("Owned feature progress already provides the current view")

    monkeypatch.setattr(studio, "_job_log_tail", unexpected_log_read)
    assert manager.describe(state)["progress"] == record


def test_features_falls_back_to_current_log_without_owned_telemetry(tmp_path):
    manager, state, _ = _fixture(tmp_path, b"positive R1: 100/1000\n")
    state["plan"][1]["id"] = state["job"]["stage"] = "features"
    progress = manager.describe(state)["progress"]
    assert progress["done"] == 100 and progress["total"] == 1000


@pytest.mark.parametrize("value", [b"null", b"[1]", b"{\x00"])
def test_malformed_optional_progress_does_not_break_status(tmp_path, monkeypatch, value):
    manager, state, _, path = _owned(tmp_path, monkeypatch)
    path.write_bytes(value)
    (Path(state["project"]).parent / ".studio/task_progress.json").write_bytes(value)
    view = manager.describe(state)
    assert view["progress"]["unit"] == "완료 단계" and view["error"] is None


def test_scoped_download_caption_retains_file_rows_and_the_current_hash_event(tmp_path, monkeypatch):
    manager, state, record, path = _owned(tmp_path, monkeypatch, stage="data_download")
    manager._live_download_files = lambda state, data, **kwargs: data["files"]
    record.update(phase="extract_verify", label="기존 추출본 확인", done=4, total=100,
                  updated_at="2026-01-01T00:00:03+00:00", eta_eligible=False)
    studio.write_json(path, record)
    files = [{"id": "asset", "state": "verifying", "done": 1000, "total": 1000}]
    event = {"phase": "hash", "done": 30, "total": 100}
    studio.write_json(Path(state["project"]).parent / ".studio/progress.json", {
        "stage": "data_download", "phase": "download_many", "label": "자료 전송",
        "updated_at": "2026-01-01T00:00:02+00:00", "files": files,
        "current_asset": "asset", "current_event": event})
    view = manager.describe(state)
    assert view["progress"]["phase"] == "extract_verify"
    assert view["progress"]["current_event"] == event
    assert view["progress"]["files"] == files and view["download_files"] == files
    assert view["eta_seconds"] is None


@pytest.mark.parametrize("committed", [0, 48000])
def test_long_audio_activity_works_before_first_or_resumed_completion_without_inflating_counts(tmp_path, monkeypatch, committed):
    manager, state, record, path = _owned(tmp_path, monkeypatch, stage="evaluation",
        content=f"Evaluate development: {committed}/48137 (CUDA batch=128, CPU rechecked=1)\n".encode())
    activity = {"phase": "gpu_scoring", "label": "긴 비호출 음원 계산", "done": 20,
                "total": 10000, "unit": "frames", "audio_seconds": 1.6}
    record.update(phase="scoring_activity", done=None, total=None, activity=activity, eta_eligible=False)
    studio.write_json(path, record)
    view = manager.describe(state)
    assert view["progress"]["done"] == committed and view["progress"]["phase"] == "scoring"
    assert view["progress"]["activity"] == activity
    assert view["eta_seconds"] is None
    Path(state["job"]["log"]).write_text("Evaluate development: 48137/48137 (CUDA batch=128, CPU rechecked=1)\n")
    view = manager.describe(state)
    assert view["progress"]["phase"] == "finalizing" and "activity" not in view["progress"]


def test_old_recommendation_does_not_replace_new_task_validation(tmp_path):
    manager, state, _ = _fixture(tmp_path)
    state["job"]["started_at"] = "2026-01-01T00:00:00+00:00"
    studio.write_json(Path(state["project"]).parent / ".studio/progress.json", {
        "stage": "evaluation", "phase": "runtime_recommendation", "updated_at": "2025-12-31T23:59:59+00:00",
        "done": 2376, "total": 2376, "unit": "조합"})
    assert manager.describe(state)["progress"]["phase"] == "validation"


def test_initial_candidate_progress_and_eta_share_the_new_100_candidate_counter(tmp_path, monkeypatch):
    manager, state, _ = _fixture(tmp_path)
    state["plan"][1]["id"] = state["job"]["stage"] = "references"
    state["settings"]["reference_preparation_policy"] = "initial_candidate_batches_v1"
    state["reference_voice_review_policy"] = "candidate_pool_v2"
    preparation = {"rows": [{"done": 26, "total": 50}, {"done": 0, "total": 50}],
                   "generated_candidates": 16, "reused_candidates": 10, "ready": False}
    manager._reference_preparation_view = lambda state: deepcopy(preparation)
    before = deepcopy(state)
    view = manager.describe(state)
    assert view["progress"]["done"] == 26 and view["progress"]["total"] == 100
    assert view["progress"]["rate_done"] == 16
    assert view["counts"] == {"technical_passed": 26}
    assert state == before


def _eta_manager(monkeypatch):
    manager = object.__new__(studio.StudioManager)
    manager._rates = {}
    clock = [0.0]
    monkeypatch.setattr(studio.time, "monotonic", lambda: clock[0])
    state = {"id": "word", "status": "running"}
    job = {"task_id": "first", "stage": "positives"}
    return manager, state, job, clock


def test_eta_excludes_long_startup_and_waits_for_three_stable_intervals(monkeypatch):
    manager, state, job, clock = _eta_manager(monkeypatch)
    progress = {"done": 0, "total": 100, "unit": "개", "phase": "tts_generation"}
    assert manager._progress_eta(state, job, progress) is None
    clock[0] = 1000
    progress["done"] = 1
    assert manager._progress_eta(state, job, progress) is None
    for stamp, count in ((1005, 2), (1010, 3)):
        clock[0], progress["done"] = stamp, count
        assert manager._progress_eta(state, job, progress) is None
    clock[0], progress["done"] = 1015, 4
    assert manager._progress_eta(state, job, progress) == pytest.approx(480)


def test_eta_qualification_and_task_phase_pause_or_counter_reset_clear_the_old_window(monkeypatch):
    manager, state, job, clock = _eta_manager(monkeypatch)
    progress = {"done": 100, "total": 30000, "unit": "학습 업데이트", "phase": "training"}
    for stamp, count in ((0, 100), (5, 1100), (10, 2100), (15, 3100)):
        clock[0], progress["done"] = stamp, count
        estimate = manager._progress_eta(state, job, progress)
    assert estimate is not None
    for change in ("qualification", "task", "phase", "pause", "counter"):
        altered, changed_job, changed_state = deepcopy(progress), dict(job), dict(state)
        if change == "qualification":
            altered.update(phase="tts_qualification", eta_eligible=False)
        elif change == "task":
            changed_job["task_id"] = "second"
        elif change == "phase":
            altered["phase"] = "another_pool"
        elif change == "pause":
            changed_state["status"] = "paused"
        else:
            altered["done"] = 2
        assert manager._progress_eta(changed_state, changed_job, altered) is None
        manager._rates.clear()
        for stamp, count in ((20, 100), (25, 1100), (30, 2100), (35, 3100)):
            clock[0], progress["done"] = stamp, count
            manager._progress_eta(state, job, progress)


def test_reused_candidates_are_not_new_speed_samples_and_complete_has_no_false_eta(monkeypatch):
    manager, state, job, clock = _eta_manager(monkeypatch)
    progress = {"done": 50, "rate_done": 0, "total": 100, "unit": "개", "phase": "reference_candidates_generation"}
    assert manager._progress_eta(state, job, progress) is None
    for stamp, generated in ((1000, 1), (1005, 2), (1010, 3), (1015, 4)):
        clock[0] = stamp
        progress.update(done=50 + generated, rate_done=generated)
        estimate = manager._progress_eta(state, job, progress)
    assert estimate == pytest.approx(230)
    progress.update(done=100, rate_done=50)
    assert manager._progress_eta(state, job, progress) is None


def test_speed_changes_stalls_and_nonfinite_counters_do_not_show_a_precise_eta(monkeypatch):
    manager, state, job, clock = _eta_manager(monkeypatch)
    progress = {"done": 1, "total": 100, "unit": "개", "phase": "generation"}
    for stamp, count in ((0, 1), (5, 2), (10, 3), (15, 30)):
        clock[0], progress["done"] = stamp, count
        assert manager._progress_eta(state, job, progress) is None
    clock[0] = 50
    assert manager._progress_eta(state, job, progress) is None
    for bad in (float("nan"), float("inf"), True, -1):
        progress["done"] = bad
        assert manager._progress_eta(state, job, progress) is None
