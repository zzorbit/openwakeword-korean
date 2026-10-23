"""Bounded file concurrency preserves the pinned downloader and caller progress."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import pytest

from korean_openwakeword.downloads import VerificationCache, download_checked, download_many, verify_checksum


def assets(count, size=10):
    return [{"id": str(index), "size_bytes": size} for index in range(count)]


def test_real_concurrency_is_bounded_and_results_keep_input_order(monkeypatch):
    monkeypatch.delenv("KOWW_DOWNLOAD_WORKERS", raising=False)
    barrier, lock = threading.Barrier(3), threading.Lock()
    active, peak, threads, observed = 0, 0, set(), []
    caller = threading.get_ident()

    def worker(item, report):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        report({"phase": "download", "done": 10, "total": 10, "unit": "bytes"})
        with lock:
            active -= 1
        return "result-" + item["id"]

    def progress(record):
        assert threading.get_ident() == caller
        observed.append(record)

    assert download_many(assets(6), worker, progress=progress) == [f"result-{i}" for i in range(6)]
    assert peak == 3 and caller not in threads
    assert observed[-1]["completed"] and observed[-1]["verified_count"] == 6
    assert observed[-1]["active_count"] == 0 and observed[-1]["done"] == 60
    assert all(record["eta_seconds"] is None and "bytes_per_second" not in record for record in observed)


@pytest.mark.parametrize("value", ["0", "5", "two", "", "3.0", "-1"])
def test_invalid_environment_is_rejected_before_start(monkeypatch, value):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", value)
    with pytest.raises(ValueError, match="KOWW_DOWNLOAD_WORKERS"):
        download_many(assets(2), lambda *_: pytest.fail("Invalid override must not start a worker"))


@pytest.mark.parametrize("value", [0, 5, True, 2.5, "2"])
def test_invalid_explicit_worker_limit(value):
    with pytest.raises(ValueError, match="workers"):
        download_many(assets(1), lambda *_: None, max_workers=value)


def test_environment_can_select_serial_and_explicit_limit_overrides_it(monkeypatch):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "1")
    started = []
    records = []
    assert download_many(assets(4), lambda item, _: started.append(item["id"]), progress=records.append) == [None] * 4
    assert started == ["0", "1", "2", "3"]
    assert all(record["active_count"] <= 1 for record in records)
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "invalid")
    assert download_many([], lambda *_: None, max_workers=2) == []


def test_failure_stops_new_submissions_and_waits_for_active_files(tmp_path):
    barrier, release = threading.Barrier(3), threading.Event()
    started, records = [], []

    def worker(item, report):
        index = item["id"]
        started.append(index)
        barrier.wait(timeout=5)
        if index == "0":
            (tmp_path / "0.part").write_bytes(b"resume me")
            raise ValueError("fixture transfer failed")
        assert release.wait(timeout=5)
        (tmp_path / index).write_bytes(b"verified fixture")
        return index

    def progress(record):
        records.append(record)
        if record["active_count"] == 2 and record["verified_count"] == 0:
            release.set()

    with pytest.raises(ValueError, match="fixture transfer failed"):
        download_many(assets(8), worker, progress=progress, max_workers=3)
    assert sorted(started) == ["0", "1", "2"]
    assert (tmp_path / "0.part").read_bytes() == b"resume me"
    assert (tmp_path / "1").is_file() and (tmp_path / "2").is_file()
    assert not any(record["completed"] for record in records)
    assert records[-1]["verified_count"] == 2


def test_progress_callback_error_propagates_after_active_writes_finish(tmp_path):
    ready, release, finished = threading.Event(), threading.Event(), threading.Event()

    def worker(item, report):
        ready.set()
        report({"phase": "download", "done": 1, "total": 10, "unit": "bytes"})
        assert release.wait(timeout=5)
        (tmp_path / "preserved").write_bytes(b"original")
        finished.set()

    class ProgressFailure(Exception):
        pass

    def progress(record):
        if record["current_event"] is not None:
            assert ready.is_set()
            release.set()
            raise ProgressFailure("fixture progress failed")

    with pytest.raises(ProgressFailure, match="fixture progress failed"):
        download_many(assets(1), worker, progress=progress)
    assert finished.is_set() and (tmp_path / "preserved").read_bytes() == b"original"


def test_aggregate_bytes_do_not_jump_between_files_or_reset_during_hash():
    records = []

    def worker(item, report):
        report({"phase": "download", "done": 4, "total": 10, "unit": "bytes"})
        report({"phase": "download", "done": 10, "total": 10, "unit": "bytes"})
        report({"phase": "hash", "done": 0, "total": 10, "unit": "bytes"})
        report({"phase": "hash", "done": 10, "total": 10, "unit": "bytes"})

    download_many(assets(3), worker, progress=records.append, max_workers=1)
    counts = [record["done"] for record in records]
    assert counts == sorted(counts) and counts[-1] == 30
    assert all(record["total"] == 30 and record["unit"] == "bytes" for record in records)
    hash_starts = [record for record in records if record["current_event"] and
                   record["current_event"]["phase"] == "hash" and record["current_event"]["done"] == 0]
    assert [record["done"] for record in hash_starts] == [10, 20, 30]


def test_unknown_sizes_report_verified_files_and_do_not_invent_byte_totals():
    records = []
    assert download_many([{"id": "unknown"}], lambda *_: "verified", progress=records.append) == ["verified"]
    assert records[0]["done"] == 0 and records[-1]["done"] == 1
    assert all(record["unit"] == "개" and record["total"] == 1 for record in records)


def test_duplicate_ids_fail_before_worker_starts():
    with pytest.raises(ValueError, match="distinct"):
        download_many([{"id": "same"}, {"id": "same"}], lambda *_: pytest.fail("Duplicate ID"))


def test_resume_and_corruption_keep_existing_per_file_guards(tmp_path, monkeypatch):
    payload, calls, barrier = b"the pinned bytes", [], threading.Barrier(2)
    (tmp_path / "good.part").write_bytes(payload[:4])
    items = [{"id": "good", "size_bytes": len(payload)}, {"id": "bad", "size_bytes": len(payload)}]

    class Response:
        def __init__(self, data, status, headers):
            self.data, self.status_code, self.headers = data, status, headers
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, _):
            yield self.data

    def get(url, *, headers, **kwargs):
        calls.append((url, headers))
        barrier.wait(timeout=5)
        if url.endswith("good"):
            assert headers["Range"] == "bytes=4-"
            return Response(payload[4:], 206, {"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})
        return Response(b"x" * len(payload), 200, {})

    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", get)

    def worker(item, report):
        return download_checked("https://example.org/" + item["id"], tmp_path / item["id"],
                                hashlib.sha256(payload).hexdigest(), item["size_bytes"], progress=report)

    with pytest.raises(ValueError, match="altered"):
        download_many(items, worker, max_workers=2)
    assert (tmp_path / "good").read_bytes() == payload
    assert (tmp_path / "good.receipt.json").is_file()
    assert (tmp_path / "bad.part").read_bytes() == b"x" * len(payload)
    assert not (tmp_path / "bad").exists() and len(calls) == 2


def test_verification_cache_can_be_shared_by_parallel_assets(tmp_path):
    cache, barrier = VerificationCache(), threading.Barrier(4)

    def verify(index):
        path = tmp_path / str(index)
        path.write_bytes(str(index).encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        barrier.wait(timeout=5)
        verify_checksum(path, "sha256", sha, verification_cache=cache)
        events = []
        verify_checksum(path, "sha256", sha, verification_cache=cache, progress=events.append)
        return events[-1]["verification_reused"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(verify, range(4)))
