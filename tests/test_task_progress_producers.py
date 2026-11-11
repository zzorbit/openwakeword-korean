"""Owned activity telemetry never becomes a durable completion count."""
from types import SimpleNamespace

import numpy as np
import pytest

from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.task_progress import activity, emit, generation_attempt, task_progress


def test_task_scope_is_optional_nested_and_restored_after_error():
    outer, inner = [], []
    emit("idle", "not installed")
    with task_progress(outer.append):
        emit("outer", "outer")
        with pytest.raises(RuntimeError), task_progress(inner.append):
            activity("gpu_frames", "working", 32, 100, audio_seconds=2.56)
            raise RuntimeError("fixture")
        emit("outer_again", "outer")
    emit("idle", "not installed")
    assert [r["phase"] for r in outer] == ["outer", "outer_again"]
    assert inner[0]["done"] is None and inner[0]["total"] is None
    assert inner[0]["activity"]["done"] == 32


def test_cached_extraction_reports_verified_files_without_rewriting_receipt(tmp_path):
    import zipfile
    from korean_openwakeword.common import digest
    from korean_openwakeword.studio_data import extract_archive
    archive, output = tmp_path / "source.zip", tmp_path / "extracted"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("one.wav", b"one")
        stream.writestr("two.wav", b"two")
    extract_archive(archive, output)
    before = digest(output / "extraction_receipt.json")
    records = []
    result = extract_archive(archive, output, progress=records.append)
    verification = [r for r in records if r.get("phase") == "extract_verify"]
    assert verification[0]["done"] == 0 and verification[-1]["done"] == verification[-1]["total"] == 2
    assert result["resumed"] is True and digest(output / "extraction_receipt.json") == before
    (output / "two.wav").write_bytes(b"changed")
    records.clear()
    with pytest.raises(ValueError, match="Published extracted files changed"):
        extract_archive(archive, output, progress=records.append)
    assert not any(r.get("phase") == "extract_verify" and r["done"] == r["total"] for r in records)


def test_writer_owns_request_and_throttles_activity_without_completing_it(monkeypatch, tmp_path):
    from korean_openwakeword import studio_worker
    write_json(tmp_path / "owner.json", {"creation_identity": "owned-process"})
    clock = iter([0., .1, .2, 1.2, 1.3])
    monkeypatch.setattr(studio_worker.time, "monotonic", lambda: next(clock))
    report = studio_worker._task_progress_writer({"task_id": "task", "stage": "evaluation"}, tmp_path, "request-hash")
    with task_progress(report):
        activity("gpu_frames", "working", 0, 100)
        activity("gpu_frames", "working", 5, 100)
        assert read_json(tmp_path / "progress.json")["activity"]["done"] == 0
        activity("gpu_frames", "working", 100, 100)
        assert read_json(tmp_path / "progress.json")["activity"]["done"] == 0
        activity("gpu_frames", "working", 100, 100)
        emit("preparation_finalizing", "final validation", 100, 100, "files")
    saved = read_json(tmp_path / "progress.json")
    assert saved["phase"] == "preparation_finalizing"
    assert saved["kind"] == "studio_task_progress_v1"
    assert saved["task_id"] == "task" and saved["request_sha256"] == "request-hash"
    assert saved["creation_identity"] == "owned-process" and saved["stage"] == "evaluation"
    assert "completed" not in saved


def test_display_write_denial_keeps_work_running_and_recovers_latest_hint(monkeypatch, tmp_path):
    from korean_openwakeword import studio_worker
    original_write = studio_worker.write_json
    writes = []
    def intermittently_locked(path, record):
        writes.append(record["done"])
        if len(writes) == 1:
            raise PermissionError("temporary Windows sharing denial")
        return original_write(path, record)
    monkeypatch.setattr(studio_worker, "write_json", intermittently_locked)
    clock = iter([0., .1, 1.1])
    monkeypatch.setattr(studio_worker.time, "monotonic", lambda: next(clock))
    report = studio_worker._task_progress_writer({"task_id": "task", "stage": "development"}, tmp_path, "sha")
    processed = []
    with task_progress(report):
        for count in (0, 1, 2):
            processed.append(count)
            emit("preparation_generation", "writing actual audio", count, 10, "files")
    assert processed == [0, 1, 2]
    assert writes == [0, 2]  # No retry storm while the hint is throttled.
    assert read_json(tmp_path / "progress.json")["done"] == 2


def test_generation_rate_counts_durable_attempts_not_polling_or_gpu_qualification(monkeypatch, tmp_path):
    from korean_openwakeword import studio, studio_worker
    clock = [0.]
    monkeypatch.setattr(studio_worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(studio_worker.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(studio_worker.time, "time", lambda: 1_800_000_000 + clock[0])
    job = {"task_id": "tts", "stage": "positives"}
    report = studio_worker._task_progress_writer(job, tmp_path, "sha")
    manager = object.__new__(studio.StudioManager)
    manager._rates = {}
    state = {"id": "project", "status": "running"}

    def eta(done):
        return manager._progress_eta(state, job, {
            **read_json(tmp_path / "progress.json"), "done": done, "total": 100, "unit": "개"})

    def committed(seconds, passed, *, qualification=False):
        with generation_attempt() as sample:
            if qualification:
                emit("tts_qualification", "GPU check", eta_eligible=False)
            clock[0] += seconds
        sample.committed(passed)

    with task_progress(report):
        emit("tts_model_loading", "Load", eta_eligible=False)
        clock[0] += 75
        emit("tts_generation", "Generate", eta_eligible=True)
        assert eta(40) is None  # Reused outputs did not produce a speed sample.
        committed(30, True, qualification=True)
        assert eta(41) is None
        for duration, passed in ((2, True), (4, False), (2, True), (2, True)):
            emit("tts_prompt_preparation", "Prompt", eta_eligible=False)
            assert eta(41) is None
            clock[0] += 3
            committed(duration, passed)
        assert eta(44) == pytest.approx(56 / (3 / 10))
        clock[0] += 21  # Sparse or interleaved status reads do not change the rate.
        assert eta(44) == pytest.approx(56 / (3 / 10))
        emit("tts_finalizing", "Save", eta_eligible=False)
        assert eta(44) is None


def test_gpu_long_frame_activity_keeps_identical_ordered_scores():
    from korean_openwakeword.gpu_evaluation import _GpuBatchScorer
    class Session:
        def __init__(self, role, value=0.):
            self.role, self.value = role, value
        def get_inputs(self):
            return [SimpleNamespace(name="input")]
        def run(self, _, inputs):
            batch = len(inputs["input"])
            return [np.full((batch, 1, 32 if self.role == "mel" else 96 if self.role == "embedding" else 1),
                            self.value, dtype=np.float32)]
    def scorer():
        value = _GpuBatchScorer.__new__(_GpuBatchScorer)
        value.initial_features = np.zeros((120, 96), np.float32)
        value.mel, value.embedding = Session("mel"), Session("embedding")
        value.heads = [Session("head", score) for score in (.1, .2, .3)]
        return value
    pcms = [np.zeros(751 * 1280, np.int16), np.ones(750 * 1280, np.int16)]
    expected = scorer().score_batch(pcms)
    signals = []
    with task_progress(signals.append):
        actual = scorer().score_batch(pcms)
    assert all(np.array_equal(a.view("u4"), b.view("u4")) for a, b in zip(actual, expected))
    activity_rows = [r["activity"] for r in signals]
    assert activity_rows[0]["done"] == 0 and activity_rows[-1]["done"] == 1501
    assert activity_rows[-1]["audio_seconds"] == pytest.approx(1501 * .08)
    assert all(r["done"] is None and r["total"] is None for r in signals)
    assert [r["done"] for r in activity_rows] == sorted(r["done"] for r in activity_rows)
