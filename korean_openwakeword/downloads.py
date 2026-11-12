"""Checksum-pinned public assets with resumable downloads and no silent approval."""
from collections import Counter, deque
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from statistics import median
from urllib.parse import urlsplit
import hashlib
import math
import os
import queue
import re
import shutil
import threading
import time
import requests

from .common import now, read_json, write_json, writer_lock


def _stat_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_identity(path):
    return _stat_identity(Path(path).stat())


def _verification_path_key(resolved_path):
    # Windows can preserve the extended prefix on one spelling of the same file.
    # Normalize only drive and UNC forms. File identity is still checked separately.
    key = str(resolved_path)
    if os.name == "nt" and key.startswith("\\\\?\\"):
        if key[4:8].lower() == "unc\\":
            return "\\\\" + key[8:]
        if key[5:7] == ":\\":
            return key[4:]
    return key


class VerificationCache:
    """Hash evidence owned by one preparation call, never a persistent trust cache."""
    def __init__(self):
        self._verified = {}
        self._lock = threading.Lock()

    def matches(self, path, algorithm, expected):
        path = Path(path).resolve()
        checksum = self._checksum(path, algorithm, _file_identity(path))
        return checksum is not None and checksum == expected

    def _checksum(self, path, algorithm, observed_identity):
        with self._lock:
            checksum, identity = self._verified.get((_verification_path_key(path), algorithm), (None, None))
        return checksum if identity == observed_identity else None

    def verified_sha256(self, path, observed_identity):
        return self._checksum(Path(path).resolve(), "sha256", observed_identity)

    def record(self, path, algorithm, expected, observed_identity):
        path = Path(path).resolve()
        if _file_identity(path) != observed_identity:
            raise ValueError("File changed during checksum verification")
        with self._lock:
            self._verified[(_verification_path_key(path), algorithm)] = (expected, observed_identity)


def _download_host(item):
    url = item.get("source_url")
    request = item.get("request")
    if not url and isinstance(request, (list, tuple)) and request:
        url = request[0]
    if isinstance(url, str):
        try:
            return urlsplit(url).hostname
        except ValueError:
            pass
    return None


def _dataset_order(items):
    """Give each dataset a turn before admitting more files from the same dataset."""
    groups = {}
    for index, item in enumerate(items):
        group = item.get("group")
        key = group if isinstance(group, str) and group else item["id"]
        groups.setdefault(key, deque()).append(index)
    remaining = deque(groups.values())
    while remaining:
        group = remaining.popleft()
        yield group.popleft()
        if group:
            remaining.append(group)


@dataclass(frozen=True)
class _ThroughputSample:
    started: float
    network_bytes: float
    files: frozenset[int]


@dataclass(frozen=True)
class _ThroughputWindow:
    seconds: float
    network_bytes: float

    @property
    def rate(self):
        return self.network_bytes / self.seconds


@dataclass(frozen=True)
class _HostProbe:
    host: str
    index: int
    started: float
    baseline: tuple[_ThroughputWindow, ...]
    baseline_files: frozenset[int]


def _sustained_throughput_gain(baseline, trial):
    baseline_rates = [window.rate for window in baseline]
    trial_rates = [window.rate for window in trial]
    baseline_rate, trial_rate = median(baseline_rates), median(trial_rates)
    noise = max(median(abs(rate - baseline_rate) for rate in baseline_rates),
                median(abs(rate - trial_rate) for rate in trial_rates))
    if math.isclose(trial_rate, baseline_rate) or trial_rate - baseline_rate <= noise:
        return False
    baseline_average = sum(window.network_bytes for window in baseline) / sum(window.seconds for window in baseline)
    trial_average = sum(window.network_bytes for window in trial) / sum(window.seconds for window in trial)
    return trial_average > baseline_average and not math.isclose(trial_average, baseline_average)


class _DownloadAdmission:
    """Start one file per host and compare batch throughput for each extra file."""

    def __init__(self, hosts, max_workers, *, window_seconds=3.0):
        self._hosts = hosts
        self._limits = dict.fromkeys(hosts, 1)
        self._probe_order = deque(host for host in self._limits if host is not None)
        self._max_workers = max_workers
        self._window_seconds = window_seconds
        self._retry_until = {}
        self._rejected = set()
        self._remaining_hosts = frozenset()
        self._sample = None
        self._windows = deque(maxlen=3)
        self._probe = None
        self.reason = "initial"

    def limit(self, host):
        probing = self._probe is not None and self._probe.host == host
        return self._limits[host] + int(probing)

    def _reset_measurement(self):
        self._sample = None
        self._windows.clear()

    def _finish_probe(self, reason):
        self._probe = None
        self._reset_measurement()
        self.reason = reason

    def retry(self, host, now):
        self._limits[host] = max(1, self._limits[host] // 2)
        self._retry_until[host] = now + 15.0
        self._rejected.discard(host)
        self._finish_probe("retry")

    def end_transfer(self, index):
        probe = self._probe
        if probe and (index == probe.index or index in probe.baseline_files):
            self._finish_probe("inconclusive")
        elif self._sample and index in self._sample.files:
            self._reset_measurement()

    def _measure(self, now, network_bytes, network_files, *, minimum_bytes=0):
        sample = self._sample
        if sample is None or sample.files != network_files:
            self._windows.clear()
            self._sample = _ThroughputSample(now, network_bytes, network_files)
            return None
        elapsed = now - sample.started
        received = network_bytes - sample.network_bytes
        if elapsed >= self._window_seconds and received >= minimum_bytes:
            self._sample = _ThroughputSample(now, network_bytes, network_files)
            self._windows.append(_ThroughputWindow(elapsed, received))
            if len(self._windows) == self._windows.maxlen:
                return tuple(self._windows)
        return None

    def _observe_probe(self, now, network_bytes, network_files):
        probe = self._probe
        trial_files = probe.baseline_files | {probe.index}
        if not probe.baseline_files <= network_files <= trial_files:
            self._finish_probe("inconclusive")
            return
        trial = self._measure(now, network_bytes, network_files) if network_files == trial_files else None
        if trial is None:
            if now - probe.started >= 5 * self._window_seconds:
                self._finish_probe("inconclusive")
            return
        if _sustained_throughput_gain(probe.baseline, trial):
            self._limits[probe.host] += 1
            self._finish_probe("gain")
            self._sample = _ThroughputSample(now, network_bytes, network_files)
            self._windows.extend(trial)
        else:
            self._rejected.add(probe.host)
            self._finish_probe("no_gain")

    def observe(self, now, network_bytes, network_files, waiting, active_by_host):
        network_files = frozenset(network_files)
        if self._probe:
            self._observe_probe(now, network_bytes, network_files)
            return

        # A provider finishing changes the capacity available to everyone else.
        remaining = frozenset(waiting) | {host for host, active in active_by_host.items() if active}
        if remaining != self._remaining_hosts:
            self._remaining_hosts = remaining
            self._rejected.clear()
            self._reset_measurement()

        # Drain any extra file still running after a rejected trial or retry.
        if (sum(active_by_host.values()) >= self._max_workers
                or any(active > self._limits[host] for host, active in active_by_host.items())):
            self._reset_measurement()
            return

        network_by_host = Counter(self._hosts[index] for index in network_files)
        candidates = {
            host for host in waiting if host is not None and host not in self._rejected
            and now >= self._retry_until.get(host, -math.inf)
            and active_by_host[host] == network_by_host[host] == self._limits[host]
        }
        if not candidates:
            self._reset_measurement()
            return

        baseline = self._measure(now, network_bytes, network_files, minimum_bytes=16 * 1024)
        if baseline is None:
            return
        host = next(host for host in self._probe_order if host in candidates)
        self._probe_order.rotate(-self._probe_order.index(host) - 1)
        self._probe = _HostProbe(host, waiting[host][0][1], now, baseline, network_files)
        self._reset_measurement()
        self.reason = "probe"


def download_many(items, worker, *, progress=None, max_workers=None):
    """Run independent whole-file downloads with bounded concurrency.

    Automatic admission starts one file per source host. It tries one additional
    file at a time and retains the higher limit when total network throughput improves.
    Explicit worker counts retain fixed concurrency and input admission order.
    Results are returned in input order in both modes.

    ``worker(item, report)`` retains the existing per-file lock, resume and hash
    checks. Only this caller invokes ``progress``; workers enqueue their events.
    The caller must reserve disk space for the combined pending downloads.
    """
    automatic = False
    if max_workers is None:
        setting = os.environ.get("KOWW_DOWNLOAD_WORKERS", "auto")
        automatic = setting == "auto"
        if automatic:
            setting = os.environ.get("KOWW_DOWNLOAD_MAX_WORKERS", "16")
        if not re.fullmatch(r"(?:[1-9]|[12][0-9]|3[0-2])", setting):
            name = "KOWW_DOWNLOAD_MAX_WORKERS" if automatic else "KOWW_DOWNLOAD_WORKERS"
            raise ValueError(name + " must be an integer from 1 to 32 (WORKERS also accepts auto)")
        max_workers = int(setting)
    if type(max_workers) is not int or not 1 <= max_workers <= 32:
        raise ValueError("Download workers must be an integer from 1 to 32")
    items = list(items)
    identifiers = [item["id"] for item in items]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(items):
        raise ValueError("Parallel download items require distinct nonempty IDs")
    sizes = [item.get("size_bytes") for item in items]
    if any(size is not None and (type(size) is not int or size <= 0) for size in sizes):
        raise ValueError("Pinned asset size must be a positive integer")
    byte_total = sum(sizes) if all(size is not None for size in sizes) else None
    available, results = [0] * len(items), [None] * len(items)
    received = [0.0] * len(items)
    networking = [False] * len(items)
    transferring = set()
    downloaded = [False] * len(items)
    file_views = [{"id": item["id"], "label": item.get("label", item["id"]),
                   "group": item.get("group", item["id"]),
                   "group_label": item.get("group_label", item.get("label", item["id"])),
                   "state": "waiting", "done": 0, "total": size,
                   "network_bytes": 0, "bytes_per_second": None, "eta_seconds": None}
                  for item, size in zip(items, sizes)]
    events, failed = queue.SimpleQueue(), threading.Event()
    pending, verified, first_error = {}, 0, None
    last_emit, last_signature = -float("inf"), None

    hosts = [_download_host(item) if automatic else None for item in items]
    waiting = {}
    order = _dataset_order(items) if automatic else range(len(items))
    for priority, index in enumerate(order):
        waiting.setdefault(hosts[index], deque()).append((priority, index))
    active_by_host = Counter()
    admission = _DownloadAdmission(hosts, max_workers) if automatic else None

    def target():
        if admission is None:
            return max_workers
        remaining_capacity = sum(admission.limit(host) for host in active_by_host.keys() | waiting.keys()
                                 if host in waiting or active_by_host[host])
        return max(1, min(max_workers, remaining_capacity))

    def next_ready():
        # Give a host its first file before adding work to an already served host.
        eligible = ((host, entries[0]) for host, entries in waiting.items()
                    if admission is None or active_by_host[host] < admission.limit(host))
        chosen = min(eligible, key=lambda entry: (
            entry[0] in active_by_host, active_by_host[entry[0]] > 0, entry[1][0]), default=None)
        if chosen is None:
            return None
        host, (_, index) = chosen
        waiting[host].popleft()
        if not waiting[host]:
            del waiting[host]
        return index

    def emit(index=None, event=None, *, completed=False):
        nonlocal last_emit, last_signature
        if progress:
            timestamp = time.monotonic()
            reason = admission.reason if admission is not None else "fixed"
            signature = (target(), verified, len(pending), completed, (event or {}).get("phase"), reason)
            if timestamp - last_emit < .25 and signature == last_signature:
                return
            last_emit, last_signature = timestamp, signature
            mode = "자동" if automatic else "고정"
            progress({"phase": "download_many", "label": f"자료 다운로드 · {mode} {target()}개 · {verified}/{len(items)}개 확인",
                      "done": sum(available) if byte_total is not None else verified,
                      "total": byte_total if byte_total is not None else len(items),
                      "unit": "bytes" if byte_total is not None else "개",
                      "verified_count": verified, "asset_count": len(items), "active_count": len(pending),
                      "max_workers": max_workers, "completed": completed, "eta_seconds": None,
                      "download_mode": "auto" if automatic else "fixed", "target_workers": target(),
                      "adjustment_reason": reason,
                      "network_bytes": sum(received), "active_network": sum(networking),
                      "files": [dict(view) for view in file_views],
                      "current_asset": identifiers[index] if index is not None else None,
                      "current_event": event})

    def drain():
        while True:
            try:
                index, event = events.get_nowait()
            except queue.Empty:
                return
            # Hashing starts after the local bytes are present. Its zero-based
            # counter must not reset the download total or pretend to be a rate.
            if event.get("unit") == "bytes":
                count = (event.get("total") if event.get("phase") == "hash" and not event.get("aggregate_done")
                         else event.get("done"))
                if isinstance(count, (int, float)) and not isinstance(count, bool) and count >= 0:
                    available[index] = max(available[index], min(count, sizes[index]) if sizes[index] else count)
            previous_bytes = received[index]
            value = event.get("network_bytes")
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0):
                received[index] = max(received[index], value)
            was_networking = networking[index]
            if isinstance(event.get("network_active"), bool):
                networking[index] = event["network_active"]
            elif event.get("phase") in ("hash", "retry"):
                networking[index] = False
            if not networking[index]:
                transferring.discard(index)
            elif received[index] > previous_bytes:
                # A request can announce itself before the first bytes arrive.
                transferring.add(index)
            view = file_views[index]
            phase = event.get("phase")
            view["state"] = {"download": "downloading", "hash": "verifying",
                             "retry": "retrying"}.get(phase, view["state"])
            if phase == "download":
                downloaded[index] = True
            if event.get("failed"):
                view["state"] = "failed"
            view.update(done=available[index], network_bytes=received[index],
                        bytes_per_second=None, eta_seconds=None)
            if sizes[index] is None and event.get("unit") == "bytes":
                amount = event.get("total")
                if (isinstance(amount, (int, float)) and not isinstance(amount, bool)
                        and math.isfinite(amount) and amount >= 0):
                    view["total"] = amount
            if type(event.get("attempt")) is int:
                view["attempt"] = event["attempt"]
            if view["state"] == "downloading" and received[index] > 0:
                for key in ("bytes_per_second", "eta_seconds"):
                    value = event.get(key)
                    if (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value >= 0):
                        view[key] = value
            if admission is not None:
                if phase == "retry":
                    admission.retry(hosts[index], time.monotonic())
                elif was_networking and not networking[index]:
                    admission.end_transfer(index)
            emit(index, event)

    def run(index):
        try:
            def report(event):
                events.put((index, {"observed_at": time.monotonic(), **event}))
            return worker(items[index], report)
        except BaseException:
            failed.set()
            raise

    emit()
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="koww-download") as executor:
        try:
            while waiting or pending:
                while waiting and len(pending) < max_workers and not failed.is_set():
                    index = next_ready()
                    if index is None:
                        break
                    file_views[index]["state"] = "downloading"
                    pending[executor.submit(run, index)] = index
                    active_by_host[hosts[index]] += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=.05, return_when=FIRST_COMPLETED)
                drain()
                for future in done:
                    index = pending.pop(future)
                    host = hosts[index]
                    active_by_host[host] -= 1
                    networking[index] = False
                    transferring.discard(index)
                    if admission is not None:
                        admission.end_transfer(index)
                    try:
                        results[index] = future.result()
                    except BaseException as error:
                        file_views[index].update(state="failed", bytes_per_second=None,
                                                 eta_seconds=None, error=type(error).__name__)
                        if first_error is None:
                            first_error = error
                    else:
                        verified += 1
                        if sizes[index] is not None:
                            available[index] = sizes[index]
                        result = results[index]
                        reused = (not received[index] and
                                  (result.get("reused") is True if isinstance(result, dict) and "reused" in result
                                   else not downloaded[index]))
                        file_views[index].update(state="reused" if reused else "complete",
                                                 done=available[index], bytes_per_second=None, eta_seconds=None)
                reason = admission.reason if admission is not None else None
                if admission is not None and not failed.is_set():
                    admission.observe(time.monotonic(), sum(received), transferring, waiting, active_by_host)
                if done or (admission is not None and admission.reason != reason):
                    emit()
            drain()
        except BaseException:
            # Stop scheduling and let already running writes finish safely. No
            # queue callback runs here, so the original callback error survives.
            failed.set()
            raise
    if first_error is not None:
        raise first_error
    emit(completed=True)
    return results


def verify_checksum(path, algorithm, expected, *, progress=None, verification_cache=None, return_sha256=False):
    """Verify unchanged bytes, optionally reusing evidence from this call only."""
    path = Path(path).resolve()
    before = _file_identity(path)
    if (verification_cache is not None and verification_cache.matches(path, algorithm, expected)
            and (not return_sha256 or algorithm == "sha256")):
        if progress:
            progress({"phase": "hash", "done": before[2], "total": before[2], "unit": "bytes",
                      "verification_reused": True, "eta_seconds": 0})
        return expected
    if algorithm not in ("sha256", "sha1", "git_sha1", "md5"):
        raise ValueError("Unsupported pinned checksum algorithm")
    checksum = hashlib.sha256() if algorithm == "sha256" else hashlib.md5() if algorithm == "md5" else hashlib.sha1()
    sha256 = checksum if algorithm == "sha256" else hashlib.sha256() if return_sha256 else None
    if algorithm == "git_sha1":
        checksum.update(f"blob {before[2]}\0".encode())
    done, started, emitted = 0, time.monotonic(), -1.0
    if progress:
        progress({"phase": "hash", "done": 0, "total": before[2], "unit": "bytes"})
    with path.open("rb") as stream:
        opened = _stat_identity(os.fstat(stream.fileno()))
        # Windows may report different timestamp precision for a path and a handle.
        if opened[:3] != before[:3]:
            raise ValueError("File changed during checksum verification")
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
            if sha256 is not None and sha256 is not checksum:
                sha256.update(chunk)
            done += len(chunk)
            elapsed = time.monotonic() - started
            if progress and (elapsed - emitted >= .5 or done == before[2]):
                rate = done / elapsed if elapsed > 0 else None
                progress({"phase": "hash", "done": done, "total": before[2], "unit": "bytes",
                          "bytes_per_second": rate,
                          "eta_seconds": max(0, (before[2] - done) / rate) if rate else None})
                emitted = elapsed
        if _stat_identity(os.fstat(stream.fileno())) != opened:
            raise ValueError("File changed during checksum verification")
    if _file_identity(path) != before:
        raise ValueError("File changed during checksum verification")
    if checksum.hexdigest() != expected:
        raise ValueError("Incomplete/altered asset checksum; preserve file for diagnosis or resume")
    if verification_cache is not None:
        verification_cache.record(path, algorithm, expected, before)
        if sha256 is not None:
            verification_cache.record(path, "sha256", sha256.hexdigest(), before)
    return sha256.hexdigest() if return_sha256 else expected


def _transient_network_error(error):
    if isinstance(error, requests.exceptions.SSLError):
        return False
    if isinstance(error, requests.exceptions.HTTPError):
        return getattr(error.response, "status_code", None) in (408, 429, 500, 502, 503, 504)
    return isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                              requests.exceptions.ChunkedEncodingError))


def download_partial(url, partial, size_bytes, *, progress=None, retries=2):
    """Bound transient retries; reopen at the bytes actually persisted on disk."""
    partial = Path(partial)
    if type(retries) is not int or not 0 <= retries <= 5:
        raise ValueError("Transient download retries must be between zero and five")
    if size_bytes is not None and (type(size_bytes) is not int or size_bytes <= 0):
        raise ValueError("Pinned asset size must be a positive integer")
    network_bytes, read_seconds, write_seconds = 0, 0.0, 0.0
    def report(record, active):
        if progress:
            progress({**record, "network_bytes": network_bytes, "read_seconds": read_seconds,
                      "write_seconds": write_seconds, "network_active": active,
                      "observed_at": time.monotonic()})
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if size_bytes is not None and offset > size_bytes:
            raise ValueError("Partial file exceeds the pinned asset size")
        if size_bytes is not None and offset == size_bytes:
            return
        required = max(0, (size_bytes or 0) - offset)
        if required and shutil.disk_usage(partial.parent).free < required:
            raise ValueError("Insufficient disk space for this download")
        headers = {"Accept-Encoding": "identity", **({"Range": f"bytes={offset}-"} if offset else {})}
        try:
            report({"phase": "download", "done": offset, "total": size_bytes, "unit": "bytes",
                    "attempt": attempt + 1}, True)
            request_started = time.monotonic()
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                read_seconds += time.monotonic() - request_started
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
                    if (not match or int(match[1]) != offset or int(match[2]) < offset
                            or size_bytes is not None and (match[3] != str(size_bytes) or int(match[2]) >= size_bytes)):
                        raise ValueError("Server returned a mismatched resume range; partial file preserved")
                elif response.status_code == 200:
                    offset = 0  # Ignored Range: restart safely, never append a full response.
                    if size_bytes is not None and shutil.disk_usage(partial.parent).free < size_bytes:
                        raise ValueError("Insufficient disk space to restart the pinned download")
                else:
                    raise ValueError("Unexpected partial download response")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Encoded download response cannot preserve pinned byte ranges")
                started, initial, emitted = time.monotonic(), offset, -1.0
                report({"phase": "download", "done": offset, "total": size_bytes, "unit": "bytes",
                        "attempt": attempt + 1}, True)
                with partial.open("ab" if offset else "wb") as handle:
                    chunks = iter(response.iter_content(64 * 1024))
                    while True:
                        read_started = time.monotonic()
                        try:
                            chunk = next(chunks)
                        except StopIteration:
                            read_seconds += time.monotonic() - read_started
                            break
                        read_seconds += time.monotonic() - read_started
                        if not chunk:
                            continue
                        network_bytes += len(chunk)
                        if size_bytes is not None and offset + len(chunk) > size_bytes:
                            raise ValueError("Download response exceeds the pinned asset size; partial preserved")
                        write_started = time.monotonic()
                        handle.write(chunk)
                        write_seconds += time.monotonic() - write_started
                        offset += len(chunk)
                        elapsed = time.monotonic() - started
                        if progress and (elapsed - emitted >= .5 or offset == size_bytes):
                            rate = (offset - initial) / elapsed if elapsed > 0 else None
                            report({"phase": "download", "done": offset, "total": size_bytes, "unit": "bytes",
                                    "attempt": attempt + 1, "bytes_per_second": rate,
                                    "eta_seconds": max(0, (size_bytes - offset) / rate) if rate and size_bytes else None}, True)
                            emitted = elapsed
                    write_started = time.monotonic()
                    handle.flush()
                    write_seconds += time.monotonic() - write_started
                if size_bytes is not None and offset != size_bytes:
                    raise requests.exceptions.ChunkedEncodingError("Incomplete pinned download response")
            report({"phase": "download", "done": offset, "total": size_bytes, "unit": "bytes",
                    "attempt": attempt + 1}, False)
            return
        except requests.exceptions.RequestException as error:
            if not _transient_network_error(error) or attempt == retries:
                report({"phase": "download", "done": partial.stat().st_size if partial.exists() else 0,
                        "total": size_bytes, "unit": "bytes", "attempt": attempt + 1, "failed": True}, False)
                raise
            report({"phase": "retry", "done": partial.stat().st_size if partial.exists() else 0,
                    "total": size_bytes, "unit": "bytes", "attempt": attempt + 2,
                    "retry_limit": retries + 1, "retry_reason": type(error).__name__,
                    "http_status": getattr(getattr(error, "response", None), "status_code", None)}, False)
            time.sleep(min(.5 * 2**attempt, 2.0))


def dataset_catalog():
    return read_json(files("korean_openwakeword") / "resources/public_datasets.json")


def verified_asset(url, output, algorithm, expected, size_bytes=None, *, progress=None, verification_cache=None):
    """Fetch/verify one pinned asset; the caller owns its lock and receipt."""
    length = 64 if algorithm == "sha256" else 40
    if (algorithm not in ("sha256", "sha1", "git_sha1")
            or not isinstance(expected, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", expected)
            or not isinstance(url, str) or not url.startswith("https://")):
        raise ValueError("A supported pinned checksum and HTTPS source are required")
    if size_bytes is not None and (type(size_bytes) is not int or size_bytes <= 0):
        raise ValueError("Pinned asset size must be a positive integer")
    output = Path(output)
    if output.exists():
        if not output.is_file() or size_bytes is not None and output.stat().st_size != size_bytes:
            raise ValueError("Existing download differs from the pinned size or checksum")
        sha256 = verify_checksum(output, algorithm, expected, progress=progress,
                                 verification_cache=verification_cache, return_sha256=True)
        return sha256, True
    partial = output.with_name(output.name + ".part")
    download_partial(url, partial, size_bytes, progress=progress)
    if size_bytes is not None and partial.stat().st_size != size_bytes:
        raise ValueError("Incomplete/altered asset; preserve .part for diagnosis or resume")
    verified_identity = _file_identity(partial)
    sha256 = verify_checksum(partial, algorithm, expected, progress=progress, return_sha256=True)
    partial.replace(output)
    if verification_cache is not None:
        published_identity = _file_identity(output)
        if published_identity[:4] != verified_identity[:4]:
            raise ValueError("File identity changed after verified download publication")
        verification_cache.record(output, algorithm, expected, published_identity)
        verification_cache.record(output, "sha256", sha256, published_identity)
    return sha256, False


def download_checked(url, output, sha256, size_bytes=None, *, progress=None, verification_cache=None):
    if (not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or not isinstance(url, str) or not url.startswith("https://")):
        raise ValueError("A pinned SHA256 and HTTPS source are required")
    if size_bytes is not None and (type(size_bytes) is not int or size_bytes <= 0):
        raise ValueError("Pinned asset size must be a positive integer")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A separate lock for each asset permits unrelated downloads in parallel.
    with writer_lock(output.parent / ("." + output.name + ".download")):
        try:
            _, reused = verified_asset(url, output, "sha256", sha256, size_bytes,
                                       progress=progress, verification_cache=verification_cache)
        except ValueError as error:
            if output.exists():
                raise ValueError("Existing download differs from the pinned size or checksum") from error
            raise
        record = {"path": str(output), "sha256": sha256, "bytes": output.stat().st_size,
                  "source_url": url, "completed_at": now(), "reused": reused,
                  "content_approved": False}
        write_json(output.with_name(output.name + ".receipt.json"), record)
        return record


def download_dataset(dataset, output):
    catalog = dataset_catalog()
    if dataset not in catalog["assets"]:
        raise ValueError("Unknown dataset ID; use koww datasets to inspect the catalog")
    asset = catalog["assets"][dataset]
    record = download_checked(asset["source_url"], output, asset["sha256"], asset["size_bytes"])
    record.update(dataset=dataset, license=asset["license"], upstream_revision=asset.get("upstream_revision"),
                  role=asset["role"], note="Download integrity is not content approval or training/Final admission.")
    output = Path(output).resolve()
    write_json(output.with_name(output.name + ".receipt.json"), record)
    return record
