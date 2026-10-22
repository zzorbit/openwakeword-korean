"""Atomic JSON publication retries transient Windows sharing failures only."""
import ctypes
import errno
import json
import os
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import pytest

from korean_openwakeword import common


RETRY_DELAYS = [.01, .02, .04, .08, .16, .25, .25]


def windows_error(code):
    error = PermissionError(errno.EACCES, "fixture Windows sharing failure")
    error.winerror = code
    return error


def json_bytes(value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return text.replace("\n", os.linesep).encode("utf-8")


@pytest.mark.parametrize("winerror", [5, 32, 33])
@pytest.mark.parametrize("failure_count", [2, 7])
def test_windows_sharing_failures_retry_atomic_publication_without_changing_json(tmp_path, monkeypatch, winerror, failure_count):
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old", "done": 7})
    target.write_bytes(before)
    value = {"label": "자료 다운로드", "done": 19, "total": 29, "approved": False}
    expected = json_bytes(value)
    real_replace = common.os.replace
    calls, delays = [], []

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        assert Path(destination) == target
        assert target.read_bytes() == before
        assert Path(source).read_bytes() == expected
        if len(calls) <= failure_count:
            raise windows_error(winerror)
        real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    common.write_json(target, value)
    assert len(calls) == failure_count + 1
    assert delays == RETRY_DELAYS[:failure_count]
    assert len({source for source, _ in calls}) == 1
    assert target.read_bytes() == expected
    assert common.read_json(target) == value
    assert not list(tmp_path.glob("progress.json.*.tmp"))


def test_retry_exhaustion_preserves_previous_destination_and_complete_temporary_json(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old"})
    target.write_bytes(before)
    value = {"state": "new", "label": "파일 확인"}
    calls, delays = [], []
    failure = windows_error(5)

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        assert target.read_bytes() == before
        raise failure

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        common.write_json(target, value)
    assert raised.value is failure
    assert len(calls) == 8
    assert delays == RETRY_DELAYS
    assert sum(delays) == pytest.approx(.81)
    assert target.read_bytes() == before
    temporary = list(tmp_path.glob("progress.json.*.tmp"))
    assert len(temporary) == 1
    assert temporary[0].read_bytes() == json_bytes(value)
    assert all(source == temporary[0] and destination == target for source, destination in calls)


@pytest.mark.parametrize("failure", [
    PermissionError(errno.EACCES, "ordinary permission denial"),
    OSError(errno.ENOSPC, "disk full"),
    OSError(errno.EIO, "unrelated I/O failure"),
    windows_error(2),
    windows_error(3),
    windows_error(87),
])
def test_unrelated_os_and_permission_failures_are_not_retried(tmp_path, monkeypatch, failure):
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old"})
    target.write_bytes(before)
    calls, delays = [], []

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        raise failure

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(OSError) as raised:
        common.write_json(target, {"state": "new"})
    assert raised.value is failure
    assert len(calls) == 1
    assert delays == []
    assert target.read_bytes() == before


@pytest.mark.parametrize("value,exception", [
    ({"bad": object()}, TypeError),
    ({"bad": float("nan")}, ValueError),
    ({"bad": float("inf")}, ValueError),
])
def test_serialization_failure_never_attempts_to_replace_the_destination(tmp_path, monkeypatch, value, exception):
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old"})
    target.write_bytes(before)
    monkeypatch.setattr(common.os, "replace", lambda *args: pytest.fail("Invalid JSON must never be published"))
    monkeypatch.setattr(common.time, "sleep", lambda *args: pytest.fail("Serialization failures must not retry"))
    with pytest.raises(exception):
        common.write_json(target, value)
    assert target.read_bytes() == before


@pytest.mark.parametrize("winerror", [5, 32, 33])
@pytest.mark.parametrize("failure_count", [2, 7])
def test_windows_read_sharing_failures_retry_without_changing_json(tmp_path, monkeypatch, winerror, failure_count):
    target = tmp_path / "progress.json"
    value = {"label": "자료 확인", "done": 23}
    before = json_bytes(value)
    target.write_bytes(before)
    real_read = common.Path.read_text
    calls, delays = [], []

    def read_text(path, *args, **kwargs):
        if path == target:
            calls.append(path)
            if len(calls) <= failure_count:
                raise windows_error(winerror)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(common.Path, "read_text", read_text)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    assert common.read_json(target) == value
    assert len(calls) == failure_count + 1
    assert delays == RETRY_DELAYS[:failure_count]
    assert target.read_bytes() == before


def test_read_sharing_retry_exhaustion_preserves_file_and_original_error(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old"})
    target.write_bytes(before)
    calls, delays = [], []
    failure = windows_error(32)

    def read_text(path, *args, **kwargs):
        calls.append(path)
        raise failure

    monkeypatch.setattr(common.Path, "read_text", read_text)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        common.read_json(target)
    assert raised.value is failure
    assert len(calls) == 8
    assert delays == RETRY_DELAYS
    assert target.read_bytes() == before


@pytest.mark.parametrize("failure", [
    FileNotFoundError(errno.ENOENT, "file is absent"),
    OSError(errno.EIO, "unrelated I/O failure"),
    OSError(errno.EINVAL, "invalid argument"),
])
def test_unrelated_read_errors_do_not_retry(tmp_path, monkeypatch, failure):
    target = tmp_path / "progress.json"
    calls, delays = [], []

    def read_text(path, *args, **kwargs):
        calls.append(path)
        raise failure

    monkeypatch.setattr(common.Path, "read_text", read_text)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(OSError) as raised:
        common.read_json(target)
    assert raised.value is failure
    assert len(calls) == 1
    assert delays == []


@pytest.mark.parametrize("platform_name", ["nt", "posix"])
def test_errno_only_read_access_denial_retries_on_windows_but_not_other_platforms(tmp_path, monkeypatch, platform_name):
    target = tmp_path / "progress.json"
    target.write_bytes(json_bytes({"state": "complete"}))
    real_read = common.Path.read_text
    failure = PermissionError(errno.EACCES, "Windows CRT may omit winerror")
    calls, delays = [], []

    def read_text(path, *args, **kwargs):
        calls.append(path)
        if len(calls) <= 2:
            raise failure
        return real_read(path, *args, **kwargs)

    # Replace this module's platform view, not global os.name used by pathlib.
    monkeypatch.setattr(common, "os", SimpleNamespace(name=platform_name))
    monkeypatch.setattr(common.Path, "read_text", read_text)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    if platform_name == "nt":
        assert common.read_json(target) == {"state": "complete"}
        assert len(calls) == 3
        assert delays == RETRY_DELAYS[:2]
    else:
        with pytest.raises(PermissionError) as raised:
            common.read_json(target)
        assert raised.value is failure
        assert len(calls) == 1
        assert delays == []


def test_persistent_errno_only_windows_read_denial_still_fails_after_eight_attempts(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old"})
    target.write_bytes(before)
    failure = PermissionError(errno.EACCES, "persistent access denial")
    calls, delays = [], []

    def read_text(path, *args, **kwargs):
        calls.append(path)
        raise failure

    monkeypatch.setattr(common, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(common.Path, "read_text", read_text)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        common.read_json(target)
    assert raised.value is failure
    assert len(calls) == 8
    assert delays == RETRY_DELAYS
    assert target.read_bytes() == before


def test_invalid_json_parse_is_not_a_retryable_file_sharing_failure(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    target.write_text('{"partial":', encoding="utf-8")
    real_read = common.Path.read_text
    calls, delays = [], []

    def read_text(path, *args, **kwargs):
        calls.append(path)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(common.Path, "read_text", read_text)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(json.JSONDecodeError):
        common.read_json(target)
    assert len(calls) == 1
    assert delays == []


@pytest.mark.skipif(sys.platform != "win32", reason="Requires actual Windows file-sharing semantics")
def test_real_windows_reader_without_delete_sharing_retries_then_publishes(tmp_path, monkeypatch):
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    target = tmp_path / "progress.json"
    before = json_bytes({"state": "old"})
    target.write_bytes(before)
    value = {"state": "new", "label": "다운로드 진행"}
    # Share reads/writes, deliberately omit FILE_SHARE_DELETE (0x4).
    handle = create_file(str(target), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    calls, delays, failures = [], [], []
    real_replace = common.os.replace

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        try:
            return real_replace(source, destination)
        except OSError as error:
            failures.append(error.winerror)
            raise

    def release_reader(delay):
        nonlocal handle
        delays.append(delay)
        assert target.read_bytes() == before
        assert handle is not None
        assert close_handle(handle)
        handle = None

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", release_reader)
    try:
        common.write_json(target, value)
    finally:
        if handle is not None:
            close_handle(handle)
    assert len(calls) == 2
    assert delays == [.01]
    assert len(failures) == 1 and failures[0] in (5, 32, 33)
    assert target.read_bytes() == json_bytes(value)
    assert not list(tmp_path.glob("progress.json.*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises concurrent Windows JSON readers and atomic replacement")
def test_concurrent_windows_json_readers_only_observe_complete_publications(tmp_path):
    target = tmp_path / "progress.json"
    text = "보라야" * 2048
    common.write_json(target, {"sequence": 0, "mirror": 0, "text": text})
    stop, ready = threading.Event(), threading.Event()
    failures, observed = queue.Queue(), queue.Queue()

    def read_progress():
        try:
            while not stop.is_set():
                value = common.read_json(target)
                assert value["sequence"] == value["mirror"]
                assert value["text"] == text
                observed.put(value["sequence"])
                ready.set()
                stop.wait(.0005)
        except BaseException as error:
            failures.put(error)
            ready.set()

    reader = threading.Thread(target=read_progress, daemon=True)
    reader.start()
    try:
        assert ready.wait(3), "JSON reader did not start"
        for sequence in range(1, 41):
            common.write_json(target, {"sequence": sequence, "mirror": sequence, "text": text})
    finally:
        stop.set()
        reader.join(3)
    assert not reader.is_alive()
    assert failures.empty(), [(repr(error), error.__dict__) for error in failures.queue]
    assert not observed.empty()
    assert common.read_json(target) == {"sequence": 40, "mirror": 40, "text": text}
    assert not list(tmp_path.glob("progress.json.*.tmp"))
