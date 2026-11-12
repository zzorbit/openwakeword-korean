"""Adaptive scheduling must not mistake local verification for network work."""
from collections import Counter, deque
import hashlib
import threading

import pytest
import requests

from korean_openwakeword.downloads import _DownloadAdmission, download_checked, download_many, download_partial


class Response:
    def __init__(self, chunks, status=200, headers=None):
        self.chunks, self.status_code, self.headers = chunks, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        yield from self.chunks


@pytest.mark.parametrize("setting", [None, "auto"])
def test_default_and_explicit_auto_report_the_selected_mode(monkeypatch, setting):
    monkeypatch.delenv("KOWW_DOWNLOAD_MAX_WORKERS", raising=False)
    if setting is None:
        monkeypatch.delenv("KOWW_DOWNLOAD_WORKERS", raising=False)
    else:
        monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", setting)
    records, caller = [], threading.get_ident()

    def progress(record):
        assert threading.get_ident() == caller
        records.append(record)

    items = [{"id": str(index), "size_bytes": 100} for index in range(8)]
    assert download_many(items, lambda item, _: item["id"], progress=progress) == [str(i) for i in range(8)]
    assert records and all(record["download_mode"] == "auto" for record in records)
    assert all(1 <= record["target_workers"] <= record["max_workers"] <= 16 for record in records)
    assert all(record["network_bytes"] == 0 for record in records)
    assert all(record["eta_seconds"] is None and "bytes_per_second" not in record for record in records)


@pytest.mark.parametrize("kind", ["environment", "argument"])
def test_manual_workers_above_the_previous_limit_are_fixed(monkeypatch, kind):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "5" if kind == "environment" else "invalid")
    records, barrier = [], threading.Barrier(5)
    lock = threading.Lock()
    active, peak = 0, 0

    def worker(item, report):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return item["id"]

    kwargs = {} if kind == "environment" else {"max_workers": 5}
    assert download_many([{"id": str(i)} for i in range(5)], worker,
                         progress=records.append, **kwargs) == [str(i) for i in range(5)]
    assert peak == 5
    assert all(record["download_mode"] == "fixed" and record["target_workers"] == 5 for record in records)


def test_auto_starts_one_file_per_host_before_admitting_more_of_any_host(monkeypatch):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "auto")
    monkeypatch.setenv("KOWW_DOWNLOAD_MAX_WORKERS", "3")
    items = [
        {"id": "a1", "group": "a", "source_url": "https://one.example/a1"},
        {"id": "a2", "group": "a", "source_url": "https://one.example/a2"},
        {"id": "a3", "group": "a", "source_url": "https://one.example/a3"},
        {"id": "b1", "group": "b", "source_url": "https://one.example/b1"},
        {"id": "c1", "group": "c", "source_url": "https://two.example/c1"},
        {"id": "d1", "group": "d", "source_url": "https://three.example/d1"},
    ]
    first_wave, lock = threading.Barrier(3), threading.Lock()
    started = []

    def worker(item, _report):
        with lock:
            started.append(item["id"])
            in_first_wave = len(started) <= 3
        if in_first_wave:
            first_wave.wait(timeout=5)
        return item["id"]

    assert download_many(items, worker) == [item["id"] for item in items]
    assert set(started[:3]) == {"a1", "c1", "d1"}


def test_retry_does_not_hold_open_slots_for_unrelated_origins(monkeypatch):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "auto")
    monkeypatch.setenv("KOWW_DOWNLOAD_MAX_WORKERS", "3")
    items = [
        {"id": "retry0", "group": "first", "source_url": "https://retry.example/0"},
        {"id": "retry1", "group": "second", "source_url": "https://retry.example/1"},
        {"id": "healthy0", "group": "healthy", "source_url": "https://healthy.example/0"},
        {"id": "healthy1", "group": "healthy", "source_url": "https://healthy.example/1"},
    ]
    first_wave = threading.Barrier(2)
    retry_observed, replacement_started, retry_finished = (threading.Event() for _ in range(3))
    records = []

    def worker(item, report):
        name = item["id"]
        if name in {"retry0", "healthy0"}:
            first_wave.wait(timeout=5)
        if name == "retry0":
            report({"phase": "retry"})
            assert replacement_started.wait(timeout=5), "An unrelated origin should keep filling available slots"
            retry_finished.set()
        elif name == "healthy0":
            assert retry_observed.wait(timeout=5)
        elif name == "healthy1":
            replacement_started.set()
        else:
            assert retry_finished.is_set(), "Do not admit another file on the same origin during a retry"
        return name

    def progress(record):
        records.append(record)
        if (record.get("current_event") or {}).get("phase") == "retry":
            retry_observed.set()

    assert download_many(items, worker, progress=progress) == [item["id"] for item in items]
    assert replacement_started.is_set()
    assert all(record["active_count"] <= 3 for record in records)
    assert records[-1]["verified_count"] == len(items)


def observe_windows(admission, waiting, active, files, now, received, window_bytes):
    for count in window_bytes:
        now += 3
        received += count
        admission.observe(now, received, files, waiting, active)
    return now, received


@pytest.mark.parametrize(("baseline", "trial", "keep"), [
    pytest.param((60_000,) * 3, (60_000,) * 3, False, id="unchanged"),
    pytest.param((60_000,) * 3, (63_000,) * 3, True, id="sustained-five-percent"),
    pytest.param((59_500, 60_000, 60_500), (61_200, 61_000, 61_400), True, id="smaller-gain-beyond-noise"),
    pytest.param((60_000,) * 3, (90_000, 60_000, 60_000), False, id="transient-spike"),
    pytest.param((60_000,) * 3, (63_000, 63_000, 51_000), False, id="lower-aggregate-rate"),
    pytest.param((57_000, 60_000, 63_000), (62_000,) * 3, False, id="gain-within-observed-noise"),
])
def test_probe_uses_whole_batch_throughput_and_only_adds_one_file(baseline, trial, keep):
    hosts = ["first", "first", "first", "second", "second"]
    admission = _DownloadAdmission(hosts, 4)
    active = Counter({"first": 1, "second": 1})
    waiting = {"first": deque([(2, 1), (4, 2)]), "second": deque([(3, 4)])}

    admission.observe(0, 0, {0, 3}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0, 3}, 0, 0, baseline[:2])
    assert (admission.limit("first"), admission.limit("second")) == (1, 1)
    now, received = observe_windows(admission, waiting, active, {0, 3}, now, received, baseline[2:])
    assert (admission.limit("first"), admission.limit("second")) == (2, 1)

    waiting["first"].popleft()
    active["first"] += 1
    admission.observe(now, received, {0, 1, 3}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0, 1, 3}, now, received, trial[:2])
    assert admission.reason == "probe"
    now, received = observe_windows(admission, waiting, active, {0, 1, 3}, now, received, trial[2:])
    assert admission.limit("first") == (2 if keep else 1)
    assert admission.reason == ("gain" if keep else "no_gain")
    assert admission.limit("second") == 1

    if keep:
        admission.retry("second", now)
        assert (admission.limit("first"), admission.limit("second")) == (2, 1)
    else:
        admission.observe(now + 3, received + 60_000, {0, 1, 3}, waiting, active)
        assert (admission.limit("first"), admission.limit("second")) == (1, 1)


def test_unproductive_third_file_falls_back_to_two_after_a_successful_trial():
    admission = _DownloadAdmission(["source"] * 4, 4)
    active = Counter({"source": 1})
    waiting = {"source": deque([(1, 1), (2, 2), (3, 3)])}

    admission.observe(0, 0, {0}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0}, 0, 0, (60_000,) * 3)
    assert admission.limit("source") == 2

    waiting["source"].popleft()
    active["source"] = 2
    admission.observe(now, received, {0, 1}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0, 1}, now, received, (90_000,) * 3)
    assert admission.limit("source") == 2
    now, received = observe_windows(admission, waiting, active, {0, 1}, now, received, (90_000,))
    assert admission.limit("source") == 3

    waiting["source"].popleft()
    active["source"] = 3
    admission.observe(now, received, {0, 1, 2}, waiting, active)
    observe_windows(admission, waiting, active, {0, 1, 2}, now, received, (90_000,) * 3)
    assert admission.limit("source") == 2


def test_completion_on_another_host_invalidates_in_progress_throughput_comparison():
    hosts = ["first", "first", "first", "second"]
    admission = _DownloadAdmission(hosts, 4)
    active = Counter({"first": 1, "second": 1})
    waiting = {"first": deque([(2, 1), (3, 2)])}
    admission.observe(0, 0, {0, 3}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0, 3}, 0, 0, (60_000,) * 3)
    assert admission.limit("first") == 2

    waiting["first"].popleft()
    active["first"] += 1
    admission.observe(now, received, {0, 1, 3}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0, 1, 3}, now, received, (63_000,) * 2)
    admission.end_transfer(3)
    active["second"] -= 1
    admission.observe(now + 3, received + 63_000, {0, 1}, waiting, active)
    assert admission.limit("first") == 1
    assert admission.reason == "inconclusive"


def test_probe_expires_when_the_added_file_never_transfers():
    admission = _DownloadAdmission(["source"] * 3, 3)
    active = Counter({"source": 1})
    waiting = {"source": deque([(1, 1), (2, 2)])}
    admission.observe(0, 0, {0}, waiting, active)
    now, received = observe_windows(admission, waiting, active, {0}, 0, 0, (60_000,) * 3)
    assert admission.limit("source") == 2

    waiting["source"].popleft()
    active["source"] = 2
    observe_windows(admission, waiting, active, {0}, now, received, (60_000,) * 5)
    assert admission.limit("source") == 1
    assert admission.reason == "inconclusive"


def test_scheduler_changes_admission_without_interrupting_an_active_file(monkeypatch):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "auto")
    monkeypatch.delenv("KOWW_DOWNLOAD_MAX_WORKERS", raising=False)
    second_started, release_first, first_finished = (threading.Event() for _ in range(3))
    records = []

    class ScriptedController:
        def __init__(self, hosts, max_workers):
            self.admission_limit, self.reason = 1, "fixture initial"

        def limit(self, host):
            return self.admission_limit

        def observe(self, now, total, network_files, waiting, active_by_host):
            if total >= 2:
                self.admission_limit, self.reason = 1, "fixture reduction"
            elif total >= 1:
                self.admission_limit, self.reason = 2, "fixture growth"

        def end_transfer(self, index):
            pass

    monkeypatch.setattr("korean_openwakeword.downloads._DownloadAdmission", ScriptedController)

    def worker(item, report):
        if item["id"] == "first":
            report({"phase": "download", "network_bytes": 1, "network_active": True})
            assert second_started.wait(timeout=5), "Increased capacity must admit the second file"
            assert release_first.wait(timeout=5)
            first_finished.set()
        elif item["id"] == "second":
            second_started.set()
            report({"phase": "download", "network_bytes": 1, "network_active": True})
        else:
            assert first_finished.is_set(), "A lower limit must delay new files without cancelling existing work"
        return item["id"]

    def progress(record):
        records.append(record)
        if record["target_workers"] == 1 and record["verified_count"] == 1:
            assert not first_finished.is_set()
            release_first.set()

    names = ["first", "second", "third"]
    items = [{"id": name, "source_url": f"https://source.example/{name}"} for name in names]
    assert download_many(items, worker, progress=progress) == names
    assert any(record["target_workers"] == 2 for record in records)
    assert first_finished.is_set() and records[-1]["completed"]


@pytest.mark.parametrize("value", ["0", "33", "two", "", "3.0", "-1"])
def test_invalid_auto_ceiling_fails_before_any_worker_starts(monkeypatch, value):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "auto")
    monkeypatch.setenv("KOWW_DOWNLOAD_MAX_WORKERS", value)
    with pytest.raises(ValueError, match="KOWW_DOWNLOAD_MAX_WORKERS"):
        download_many([{"id": "asset"}], lambda *_: pytest.fail("Invalid policy must not start a worker"))


def test_resumed_bytes_and_hash_reads_are_excluded_from_network_totals(tmp_path, monkeypatch):
    payload = b"an immutable original dataset payload"
    output = tmp_path / "asset"
    output.with_name("asset.part").write_bytes(payload[:9])
    events = []

    def get(url, *, headers, **kwargs):
        assert headers["Range"] == "bytes=9-"
        return Response([payload[9:15], payload[15:]], 206,
                        {"Content-Range": f"bytes 9-{len(payload)-1}/{len(payload)}"})

    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", get)
    download_checked("https://example.org/asset", output, hashlib.sha256(payload).hexdigest(),
                     len(payload), progress=events.append)
    assert output.read_bytes() == payload
    network = [event for event in events if "network_bytes" in event]
    assert network and network[-1]["network_bytes"] == len(payload) - 9
    assert [event["network_bytes"] for event in network] == sorted(event["network_bytes"] for event in network)
    assert network[-1]["network_active"] is False
    assert any(event["network_active"] for event in network)
    first_hash = next(index for index, event in enumerate(events) if event["phase"] == "hash")
    assert next(event for event in reversed(events[:first_hash]) if "network_active" in event)["network_active"] is False
    assert all(event["read_seconds"] >= 0 and event["write_seconds"] >= 0 for event in network)


def test_retry_keeps_fresh_transfer_counters_without_counting_existing_prefix(tmp_path, monkeypatch):
    payload = b"a pinned complete original"
    partial = tmp_path / "asset.part"
    partial.write_bytes(payload[:4])
    events, calls = [], []

    class Interrupted(Response):
        def iter_content(self, _):
            yield payload[4:9]
            raise requests.exceptions.ChunkedEncodingError("fixture network interruption")

    def get(url, *, headers, **kwargs):
        calls.append(headers)
        offset = 4 if len(calls) == 1 else 9
        assert headers["Range"] == f"bytes={offset}-"
        metadata = {"Content-Range": f"bytes {offset}-{len(payload)-1}/{len(payload)}"}
        return Interrupted([], 206, metadata) if len(calls) == 1 else Response([payload[9:]], 206, metadata)

    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", get)
    monkeypatch.setattr("korean_openwakeword.downloads.time.sleep", lambda _: None)
    download_partial("https://example.org/asset", partial, len(payload), progress=events.append)
    assert partial.read_bytes() == payload and len(calls) == 2
    network = [event for event in events if "network_bytes" in event]
    assert network[-1]["network_bytes"] == len(payload) - 4
    assert [event["network_bytes"] for event in network] == sorted(event["network_bytes"] for event in network)
    assert network[-1]["network_active"] is False
    assert any(event["phase"] == "retry" for event in events)


def test_aggregate_network_totals_ignore_cached_files_and_hash_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "auto")
    monkeypatch.delenv("KOWW_DOWNLOAD_MAX_WORKERS", raising=False)
    payload, records = b"same pinned original bytes", []
    (tmp_path / "cached").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        assert url.endswith("fresh")
        return Response([payload])

    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", get)

    def worker(item, report):
        return download_checked("https://example.org/" + item["id"], tmp_path / item["id"],
                                sha, len(payload), progress=report)

    items = [{"id": name, "size_bytes": len(payload)} for name in ("cached", "fresh")]
    result = download_many(items, worker, progress=records.append)
    assert [record["reused"] for record in result] == [True, False]
    assert calls == ["https://example.org/fresh"]
    assert records[-1]["done"] == 2 * len(payload)
    assert records[-1]["network_bytes"] == len(payload)
    assert records[-1]["active_network"] == 0
    assert all(record["network_bytes"] <= len(payload) for record in records)
    assert all(record["eta_seconds"] is None and "bytes_per_second" not in record for record in records)


def test_complete_partial_has_no_network_telemetry(tmp_path, monkeypatch):
    payload, events = b"already fully persisted", []
    output = tmp_path / "asset"
    output.with_name("asset.part").write_bytes(payload)
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *_args, **_kwargs: pytest.fail("No HTTP is needed"))
    download_checked("https://example.org/asset", output, hashlib.sha256(payload).hexdigest(),
                     len(payload), progress=events.append)
    assert output.read_bytes() == payload
    assert not any(event.get("network_active") or event.get("network_bytes", 0) for event in events)
