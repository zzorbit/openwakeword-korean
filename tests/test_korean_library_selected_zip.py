"""Exact-member transport, persistent resume and rejection of altered sources."""
import hashlib
import io
from pathlib import Path
import struct
import zipfile

import pytest
import requests

from korean_openwakeword import selected_zip as module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(split=False):
    original = {"audio/first.wav": bytes(range(256))*30,
                "audio/second.wav": bytes(reversed(range(256)))*42}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in original.items():
            archive.writestr(name, data)
    payload = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        rows = []
        for info in archive.infolist():
            start = info.header_offset
            h = struct.unpack_from("<4s5H3I2H", payload, start)
            header_bytes = 30+h[9]+h[10]
            data = start+header_bytes
            rows.append({"name": info.filename, "size_bytes": info.file_size,
                "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}",
                "sha256": sha(original[info.filename]), "compression_method": info.compress_type,
                "flags": info.flag_bits, "header_offset": start, "data_offset": data,
                "header_bytes": header_bytes, "header_sha256": sha(payload[start:data]),
                "compressed_sha256": sha(payload[data:data+info.compress_size])})
    cut = rows[0]["data_offset"]+7 if split else len(payload)
    parts = [payload[:cut], payload[cut:]] if split else [payload]
    volumes = [{"id": f"vol{i}", "source_url": f"https://fixture.test/{i}", "size_bytes": len(data)}
               for i, data in enumerate(parts)]
    return {"id": "fixture", "volumes": volumes, "members": rows}, dict(zip([v["source_url"] for v in volumes], parts)), original


class Response:
    def __init__(self, status, headers, body, fail=False):
        self.status_code, self.headers, self.body, self.fail = status, headers, body, fail
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError("fixture error", response=self)
            raise error

    def iter_content(self, size):
        if self.fail:
            yield self.body[:7]
            raise requests.exceptions.ConnectionError("connection interrupted")
        yield self.body

    def close(self):
        self.closed = True


class Network:
    def __init__(self, bodies, *, status=206, wrong_range=False, fail=0, corrupt=False):
        self.bodies, self.status, self.wrong_range = bodies, status, wrong_range
        self.fail, self.corrupt = fail, corrupt
        self.requests, self.responses, self.sessions = [], [], []

    def session(self):
        owner = self
        class Session:
            def __init__(self):
                self.closed = False
            def get(self, url, *, headers, **kwargs):
                owner.requests.append((url, dict(headers)))
                start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
                body = owner.bodies[url][start:end+1]
                if owner.corrupt:
                    body = bytes([body[0]^1])+body[1:]
                response = Response(owner.status, {"Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{end}/{len(owner.bodies[url])}" if not owner.wrong_range else "bytes 1-2/3",
                    "ETag": '"pinned-source"'}, body, owner.fail > 0)
                owner.fail -= 1
                owner.responses.append(response)
                return response
            def close(self):
                self.closed = True
        value = Session(); self.sessions.append(value)
        return value


def acquire(tmp_path, group, network, **kwargs):
    return module.acquire_selected(group, tmp_path/"originals", tmp_path/"transaction",
                                   session_factory=network.session, **kwargs)


@pytest.mark.parametrize("split", [False, True])
def test_selected_originals_exact_and_completed_reuse_needs_no_network(tmp_path, split):
    group, bodies, original = fixture(split)
    network = Network(bodies)
    result = acquire(tmp_path, group, network)
    assert [r["name"] for r in result["files"]] == list(original)
    assert all(Path(r["path"]).read_bytes() == original[r["name"]] for r in result["files"])
    assert result["whole_archive_verified"] is False and result["audio_reencoded"] is False
    assert all(r.closed for r in network.responses) and all(s.closed for s in network.sessions)
    assert not list((tmp_path/"transaction/ranges").glob("*.part"))
    count = len(network.requests)
    again = acquire(tmp_path, group, network)
    assert again["reused_members"] == 2 and again["observed_transfer_bytes_this_run"] == 0
    assert len(network.requests) == count


def test_interrupted_range_retries_from_persisted_offset(tmp_path, monkeypatch):
    group, bodies, _ = fixture()
    network = Network(bodies, fail=1)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = acquire(tmp_path, group, network, workers=1)
    assert len(network.requests) == 2
    assert network.requests[0][1]["Range"].startswith("bytes=0-")
    assert network.requests[1][1]["Range"].startswith("bytes=7-")
    assert network.requests[1][1]["If-Match"] == '"pinned-source"'
    assert result["observed_transfer_bytes_this_run"] == module.selected_transfer_bytes(group)


def test_exhausted_connection_keeps_partial_and_next_run_continues(tmp_path, monkeypatch):
    group, bodies, _ = fixture()
    network = Network(bodies, fail=3)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(requests.exceptions.ConnectionError):
        acquire(tmp_path, group, network, workers=1)
    assert next((tmp_path/"transaction/ranges").glob("*.part")).stat().st_size == 21
    count = len(network.requests)
    result = acquire(tmp_path, group, network, workers=1)
    assert network.requests[count][1]["Range"].startswith("bytes=21-")
    assert len(result["files"]) == 2


@pytest.mark.parametrize("status,wrong", [(200, False), (206, True), (403, False)])
def test_full_response_wrong_range_and_auth_failure_never_publish(tmp_path, status, wrong):
    group, bodies, _ = fixture()
    network = Network(bodies, status=status, wrong_range=wrong)
    with pytest.raises((ValueError, requests.exceptions.HTTPError)):
        acquire(tmp_path, group, network)
    assert not list((tmp_path/"originals").rglob("*.wav"))
    assert not list((tmp_path/"transaction").glob("receipt_*.json"))
    assert all(r.closed for r in network.responses)


def test_corrupted_compressed_bytes_are_not_admitted(tmp_path):
    group, bodies, _ = fixture()
    network = Network(bodies, corrupt=True)
    with pytest.raises(ValueError, match="compressed bytes"):
        acquire(tmp_path, group, network)
    assert not list((tmp_path/"originals").rglob("*.wav"))
    assert list((tmp_path/"transaction/ranges").glob("*.part"))


@pytest.mark.parametrize("pin", ["sha256", "crc32", "header_sha256", "compressed_sha256"])
def test_each_independent_original_pin_is_enforced(tmp_path, pin):
    group, bodies, _ = fixture()
    group["members"][0][pin] = "0"*len(group["members"][0][pin])
    with pytest.raises(ValueError):
        acquire(tmp_path, group, Network(bodies))
    assert not (tmp_path/"originals/audio/first.wav").exists()


def test_existing_changed_original_is_preserved_without_network(tmp_path):
    group, bodies, _ = fixture()
    network = Network(bodies)
    acquire(tmp_path, group, network)
    output = tmp_path/"originals/audio/first.wav"
    output.write_bytes(b"altered")
    count = len(network.requests)
    with pytest.raises(ValueError, match="Existing selected original changed"):
        acquire(tmp_path, group, network)
    assert output.read_bytes() == b"altered" and len(network.requests) == count


def test_local_volume_reuse_verifies_same_members_without_transfer(tmp_path):
    group, bodies, original = fixture(True)
    local = {}
    for volume in group["volumes"]:
        path = tmp_path/(volume["id"]+".zip")
        path.write_bytes(bodies[volume["source_url"]]);local[volume["id"]] = path
    network = Network(bodies)
    result = acquire(tmp_path, group, network, local_volumes=local)
    assert not network.requests and result["observed_transfer_bytes_this_run"] == 0
    assert result["local_read_bytes_this_run"] == module.selected_transfer_bytes(group)
    assert all((tmp_path/"originals"/name).read_bytes() == data for name,data in original.items())


@pytest.mark.parametrize("name", ["../escape.wav", "/escape.wav", "a\\b.wav", "C:/escape.wav", "a/../b.wav"])
def test_unsafe_member_names_are_rejected_before_transfer(tmp_path, name):
    group, bodies, _ = fixture()
    group["members"][0]["name"] = name
    network = Network(bodies)
    with pytest.raises(ValueError):
        acquire(tmp_path, group, network)
    assert not network.requests


def test_request_changes_do_not_rebind_old_transaction(tmp_path):
    group, bodies, _ = fixture()
    network = Network(bodies)
    result = acquire(tmp_path, group, network)
    receipt = Path(result["receipt"]["path"]).read_bytes()
    group["members"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="another immutable request"):
        acquire(tmp_path, group, network)
    assert Path(result["receipt"]["path"]).read_bytes() == receipt


def test_parallel_chunks_use_one_stable_fresh_nested_destination(tmp_path):
    group, bodies, original = fixture()
    # Separate the original member spans without modifying either member.
    row = group["members"][1]
    cut, gap = row["header_offset"], 300 * 1024
    url = group["volumes"][0]["source_url"]
    bodies[url] = bodies[url][:cut] + b"\x00"*gap + bodies[url][cut:]
    group["volumes"][0]["size_bytes"] += gap
    row["header_offset"] += gap; row["data_offset"] += gap
    nested = tmp_path/"new"/"nested"/"project"
    network = Network(bodies)
    result = acquire(nested, group, network, workers=4)
    assert len(network.requests) == 2
    assert all(Path(r["path"]).read_bytes() == original[r["name"]] for r in result["files"])


def test_resume_speed_uses_newly_transferred_bytes_only(tmp_path, monkeypatch):
    group, bodies, _ = fixture()
    _, chunks = module._plan(group)
    start, end = chunks[0]["start"], chunks[0]["end"]
    partial = tmp_path/"transaction/ranges"/(module.identity(["vol0",start,end])+".part")
    partial.parent.mkdir(parents=True)
    body = next(iter(bodies.values()))[start:end]
    partial.write_bytes(body[:-1])
    clock = iter(range(10000))
    monkeypatch.setattr(module.time,"monotonic",lambda: next(clock))
    progress=[]
    result = acquire(tmp_path, group, Network(bodies), workers=1, progress=progress.append)
    assert result["observed_transfer_bytes_this_run"] == 1
    assert progress[-1]["done"] == progress[-1]["total"]
    assert 0 < progress[-1]["bytes_per_second"] <= 1
