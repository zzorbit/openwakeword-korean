"""Real ZIP bytes over a fake HTTP server exercise byte limits and resumption."""
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zipfile

import numpy as np
import pytest
import requests

from korean_openwakeword.common import digest
from korean_openwakeword.range_zip import RangeZipArchive


URL = "https://example.org/original.zip"


class Response:
    def __init__(self, *, status=206, headers=None, body=b"", fail_after=None):
        self.status_code, self.headers, self.body = status, headers or {}, body
        self.fail_after, self.iterated, self.closed = fail_after, False, False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        self.iterated = True
        if self.fail_after is not None:
            yield self.body[:self.fail_after]
            raise requests.ConnectionError("fixture interrupted transport")
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset+chunk_size]

    def close(self): self.closed = True


class Server:
    def __init__(self, data):
        self.data, self.etag = data, '"frozen-fixture"'
        self.calls, self.responses = [], []
        self.change, self.failures, self.interrupt_at = None, 0, None

    def head(self, url, **kwargs):
        assert url == URL
        self.calls.append(("HEAD", kwargs))
        return Response(status=200, headers={"Content-Length": str(len(self.data)), "ETag": self.etag})

    def get(self, url, *, headers, **kwargs):
        assert url == URL and kwargs["stream"] is True
        assert headers["Accept-Encoding"] == "identity" and headers["If-Match"] == self.etag
        assert "If-Range" not in headers
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        self.calls.append(("GET", start, end))
        if self.interrupt_at is not None and len([c for c in self.calls if c[0] == "GET"]) >= self.interrupt_at:
            raise KeyboardInterrupt("fixture process interruption")
        response_headers = {"Content-Length": str(end-start+1), "ETag": self.etag,
                            "Content-Range": f"bytes {start}-{end}/{len(self.data)}"}
        status, body = 206, self.data[start:end+1]
        if self.change == "full": status, body = 200, self.data
        elif self.change == "etag": response_headers["ETag"] = '"other-version"'
        elif self.change == "range": response_headers["Content-Range"] = f"bytes {start+1}-{end}/{len(self.data)}"
        elif self.change == "encoding": response_headers["Content-Encoding"] = "gzip"
        elif self.change == "precondition": status = 412
        elif self.change == "overlong": body += b"unexpected"
        elif self.change == "short": body = body[:-1]
        fail_after = None
        if self.failures:
            self.failures -= 1
            fail_after = max(1, len(body)//2)
        response = Response(status=status, headers=response_headers, body=body, fail_after=fail_after)
        self.responses.append(response)
        return response


def archive_bytes(*, compression=zipfile.ZIP_DEFLATED, force_zip64=False, names=None):
    chosen = np.random.default_rng(11).integers(0, 256, 200000, dtype=np.uint8).tobytes()
    unused = np.random.default_rng(12).integers(0, 256, 2000000, dtype=np.uint8).tobytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        if force_zip64:
            with archive.open("folder/chosen.mp3", "w", force_zip64=True) as file:
                file.write(chosen)
        else:
            archive.writestr((names or ["folder/chosen.mp3"])[0], chosen)
        archive.writestr((names or [None, "folder/unused.mp3"])[1], unused)
    return stream.getvalue(), chosen


def open_archive(data, server=None, **kwargs):
    server = server or Server(data)
    return RangeZipArchive(URL, len(data), session=server, max_download_bytes=8 * 1024**2, **kwargs), server


def test_only_selected_original_bytes_are_transferred_and_receipt_is_reused(tmp_path):
    data, original = archive_bytes()
    archive, server = open_archive(data)
    target = tmp_path / "chosen.mp3"
    receipt = archive.extract("folder/chosen.mp3", target, chunk_bytes=65536)
    assert target.read_bytes() == original
    assert receipt["sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["crc32_verified"] and receipt["original_bytes_preserved"]
    assert receipt["audio_reencoded"] is False and receipt["whole_archive_sha1_verified"] is False
    assert archive.downloaded_bytes < len(data) / 4
    before = target.with_name(target.name+".range.json").read_bytes()
    calls = len(server.calls)
    assert archive.extract("folder/chosen.mp3", target) == receipt
    assert len(server.calls) == calls
    assert target.with_name(target.name+".range.json").read_bytes() == before
    assert not list(tmp_path.glob("*.part"))
    assert all(response.closed for response in server.responses)


def test_central_directory_hash_covers_actual_index_bytes():
    data, _ = archive_bytes()
    archive, _ = open_archive(data)
    with zipfile.ZipFile(io.BytesIO(data)) as local:
        start = local.start_dir
    central = data[start:start+archive.index_bytes]
    assert hashlib.sha256(central).hexdigest() == archive.index_sha256
    assert archive.binding()["etag"] == '"frozen-fixture"'


def test_original_bzip2_metadata_is_read_and_extracted_without_reencoding(tmp_path):
    original = (b"track_id,artist,title,duration\n160,artist,track,70\n" * 10000)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_BZIP2) as local:
        local.writestr("fma_metadata/tracks.csv", original)
        local.writestr("fma_metadata/unselected.txt", b"unused" * 100000)
    archive, server = open_archive(stream.getvalue())
    info = archive._entry("fma_metadata/tracks.csv")
    assert info.extract_version == 46 and info.compress_type == zipfile.ZIP_BZIP2
    assert archive.read(info) == original
    destination = tmp_path / "tracks.csv"
    receipt = archive.extract(info, destination)
    assert destination.read_bytes() == original
    assert receipt["request"]["compression"] == zipfile.ZIP_BZIP2
    assert receipt["crc32_verified"] and receipt["original_bytes_preserved"]
    assert receipt["sha256"] == hashlib.sha256(original).hexdigest()
    assert all(response.closed for response in server.responses)


@pytest.mark.parametrize("change", ["full", "etag", "range", "encoding", "precondition"])
def test_wrong_range_or_source_is_rejected_before_reading_body(change):
    data, _ = archive_bytes()
    server = Server(data)
    server.change = change
    with pytest.raises(ValueError, match="frozen ZIP byte range"):
        open_archive(data, server)
    assert len(server.responses) == 1
    assert not server.responses[0].iterated and server.responses[0].closed


def test_expected_etag_and_size_must_match_before_ranges():
    data, _ = archive_bytes()
    server = Server(data)
    with pytest.raises(ValueError, match="ETag"):
        open_archive(data, server, expected_etag='"old"')
    assert all(call[0] == "HEAD" for call in server.calls)
    with pytest.raises(ValueError, match="size"):
        RangeZipArchive(URL, len(data)+1, session=server, max_download_bytes=1000000)


@pytest.mark.parametrize("etag", ['W/"weak"', "not-quoted", None])
def test_weak_or_missing_etag_is_not_a_source_binding(etag):
    data, _ = archive_bytes()
    server = Server(data)
    server.etag = etag
    with pytest.raises(ValueError, match="strong archive ETag"):
        open_archive(data, server)


def test_transport_retry_counts_failed_bytes_and_is_bounded():
    data, original = archive_bytes()
    server = Server(data)
    server.failures = 1
    archive, _ = open_archive(data, server, retries=1)
    assert archive.read("folder/chosen.mp3") == original
    assert any(event["state"] == "transport_error" for event in archive.requests)
    assert archive.downloaded_bytes == sum(event["bytes"] for event in archive.requests)
    server = Server(data)
    server.failures = 3
    with pytest.raises(ValueError, match="bounded retries"):
        open_archive(data, server, retries=1)
    assert len(server.responses) == 2


def test_short_and_overlong_ranges_never_become_verified_bytes():
    data, _ = archive_bytes()
    for change in ("short", "overlong"):
        server = Server(data)
        server.change = change
        with pytest.raises((ValueError, requests.RequestException)):
            open_archive(data, server, retries=0)


def test_memory_and_transfer_limits_refuse_reads_before_unbounded_body():
    data, _ = archive_bytes()
    server = Server(data)
    with pytest.raises(ValueError, match="transfer budget"):
        RangeZipArchive(URL, len(data), session=server, max_download_bytes=1)
    assert not server.responses
    archive, server = open_archive(data)
    before = len(server.calls)
    archive._reader.seek(0)
    # Reads are clamped to remaining bytes. Use a smaller explicit read cap to
    # exercise the limit without manufacturing an unnecessarily huge ZIP.
    archive._reader.max_read_bytes = 16
    with pytest.raises(ValueError, match="memory limit"):
        archive._reader.read(100)
    assert len(server.calls) == before


def test_interrupted_member_resumes_only_unfinished_compressed_ranges(tmp_path):
    data, original = archive_bytes()
    archive, server = open_archive(data)
    target = tmp_path / "chosen.mp3"
    def interrupt(event):
        if event["done"] >= 65536:
            raise KeyboardInterrupt("fixture stopped after committed chunk")
    with pytest.raises(KeyboardInterrupt):
        archive.extract("folder/chosen.mp3", target, chunk_bytes=65536, progress=interrupt)
    state_path = target.with_name(target.name+".range-part.json")
    state = json.loads(state_path.read_text())
    offset = state["compressed_bytes"]
    assert offset == 65536 and not target.exists()
    assert target.with_name(target.name+".compressed.part").stat().st_size == offset
    resumed, new_server = open_archive(data, expected_etag=archive.etag)
    payload_start = resumed._compressed_start(resumed._entry("folder/chosen.mp3"))
    before = len(new_server.calls)
    resumed.extract("folder/chosen.mp3", target, chunk_bytes=65536)
    payload_calls = [call for call in new_server.calls[before:] if call[0] == "GET" and call[2]-call[1] > 512]
    assert payload_calls and min(call[1] for call in payload_calls) >= payload_start+offset
    assert target.read_bytes() == original and not state_path.exists()


def test_changed_partial_source_and_corrupted_checkpoint_are_refused(tmp_path):
    data, _ = archive_bytes()
    archive, _ = open_archive(data)
    target = tmp_path / "chosen.mp3"
    with pytest.raises(KeyboardInterrupt):
        archive.extract("folder/chosen.mp3", target, chunk_bytes=4096,
                        progress=lambda event: (_ for _ in ()).throw(KeyboardInterrupt()))
    cache = target.with_name(target.name+".compressed.part")
    cache.write_bytes(b"x"+cache.read_bytes()[1:])
    with pytest.raises(ValueError, match="Committed compressed"):
        archive.extract("folder/chosen.mp3", target)
    assert not target.exists()


def test_zip64_local_header_and_zip64_archive_index_are_supported(tmp_path):
    data, original = archive_bytes(force_zip64=True)
    # Upgrade the end records to real ZIP64 inventory/offset fields while the
    # small payload itself remains suitable for a fast offline fixture.
    eocd = list(struct.unpack("<4s4H2IH", data[-22:]))
    central_size, central_offset = eocd[5:7]
    zip64_offset = len(data)-22
    zip64 = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                        eocd[3], eocd[4], central_size, central_offset)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, zip64_offset, 1)
    eocd[3:7] = [0xffff, 0xffff, 0xffffffff, 0xffffffff]
    data = data[:-22]+zip64+locator+struct.pack("<4s4H2IH", *eocd)
    archive, _ = open_archive(data)
    target = tmp_path / "original.mp3"
    archive.extract("folder/chosen.mp3", target)
    assert target.read_bytes() == original


def test_crc_failure_never_publishes_or_approves_member(tmp_path):
    data, _ = archive_bytes(compression=zipfile.ZIP_STORED)
    with zipfile.ZipFile(io.BytesIO(data)) as local:
        member = local.getinfo("folder/chosen.mp3")
    local_name, local_extra = struct.unpack_from("<HH", data, member.header_offset+26)
    start = member.header_offset+30+local_name+local_extra
    damaged = bytearray(data)
    damaged[start+100] ^= 1
    archive, _ = open_archive(bytes(damaged))
    target = tmp_path / "original.mp3"
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        archive.extract("folder/chosen.mp3", target)
    assert not target.exists() and not target.with_name(target.name+".range.json").exists()
    assert target.with_name(target.name+".compressed.part").exists()


@pytest.mark.parametrize("name", ["../escape.mp3", "/absolute.mp3", "folder\\escape.mp3", "C:drive.mp3"])
def test_unsafe_archive_paths_are_refused(name):
    data, _ = archive_bytes(names=[name, "folder/unused.mp3"])
    if "\\" in name:
        # Windows' ZIP writer normalizes backslashes; emulate an actual unsafe
        # publisher central/local filename without changing byte lengths.
        data = data.replace(name.replace("\\", "/").encode(), name.encode())
    with pytest.raises(ValueError, match="unsafe"):
        open_archive(data)


def test_completed_receipt_is_immutable_and_changed_audio_refused(tmp_path):
    data, _ = archive_bytes()
    archive, _ = open_archive(data)
    target = tmp_path / "original.mp3"
    archive.extract("folder/chosen.mp3", target)
    original_receipt = target.with_name(target.name+".range.json").read_bytes()
    target.write_bytes(b"x"+target.read_bytes()[1:])
    with pytest.raises(ValueError, match="Completed ZIP member"):
        archive.extract("folder/chosen.mp3", target)
    assert target.with_name(target.name+".range.json").read_bytes() == original_receipt


def test_owned_uncommitted_compressed_tail_can_be_safely_recovered(tmp_path):
    data, original = archive_bytes()
    archive, _ = open_archive(data)
    target = tmp_path / "original.mp3"
    with pytest.raises(KeyboardInterrupt):
        archive.extract("folder/chosen.mp3", target, chunk_bytes=4096,
                        progress=lambda event: (_ for _ in ()).throw(KeyboardInterrupt()))
    cache = target.with_name(target.name+".compressed.part")
    with cache.open("ab") as stream:
        stream.write(b"uncommitted bytes from interrupted write")
    archive.extract("folder/chosen.mp3", target)
    assert target.read_bytes() == original


def test_publish_gap_recovers_only_a_hash_bound_crc_verified_output(tmp_path, monkeypatch):
    from korean_openwakeword import range_zip as module
    data, original = archive_bytes()
    archive, _ = open_archive(data)
    target = tmp_path / "original.mp3"
    real_write = module.write_json
    def interrupt(path, value):
        if str(path).endswith(".range.json"):
            raise KeyboardInterrupt("fixture stopped after atomic member publication")
        return real_write(path, value)
    monkeypatch.setattr(module, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        archive.extract("folder/chosen.mp3", target)
    assert target.read_bytes() == original
    monkeypatch.setattr(module, "write_json", real_write)
    receipt = archive.extract("folder/chosen.mp3", target)
    assert receipt["sha256"] == digest(target) and receipt["crc32_verified"]
    assert not target.with_name(target.name+".range-part.json").exists()


def test_unowned_existing_destination_is_preserved(tmp_path):
    data, _ = archive_bytes()
    archive, _ = open_archive(data)
    target = tmp_path / "original.mp3"
    target.write_bytes(b"user file")
    with pytest.raises(ValueError, match="preserved"):
        archive.extract("folder/chosen.mp3", target)
    assert target.read_bytes() == b"user file"


def test_head_transport_retries_are_bounded_without_getting_full_content():
    data, _ = archive_bytes()
    server = Server(data)
    original_head = server.head
    calls = 0
    def temporary_failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.Timeout("fixture HEAD timeout")
        return original_head(*args, **kwargs)
    server.head = temporary_failure
    archive, _ = open_archive(data, server, retries=1)
    assert calls == 2 and archive.entries
    server = Server(data)
    server.head = lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout())
    with pytest.raises(ValueError, match="HEAD request failed"):
        open_archive(data, server, retries=1)
    assert not server.responses


def test_changed_index_cannot_reuse_the_same_partial_even_with_same_etag(tmp_path):
    data, _ = archive_bytes()
    archive, _ = open_archive(data)
    target = tmp_path / "original.mp3"
    with pytest.raises(KeyboardInterrupt):
        archive.extract("folder/chosen.mp3", target, chunk_bytes=4096,
                        progress=lambda event: (_ for _ in ()).throw(KeyboardInterrupt()))
    altered = bytearray(data)
    with zipfile.ZipFile(io.BytesIO(data)) as local:
        altered[local.start_dir+16] ^= 1  # Same URL/size/ETag; changed central CRC.
    changed, _ = open_archive(bytes(altered))
    with pytest.raises(ValueError, match="different source bytes"):
        changed.extract("folder/chosen.mp3", target)
    assert not target.exists()


def test_cleanup_interruption_reuses_completed_receipt_without_rewriting_it(tmp_path, monkeypatch):
    data, original = archive_bytes()
    archive, server = open_archive(data)
    target = tmp_path / "original.mp3"
    real_unlink = Path.unlink
    def interrupt(path, *args, **kwargs):
        if path.name.endswith(".compressed.part"):
            raise KeyboardInterrupt("fixture stopped before cache cleanup")
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(KeyboardInterrupt):
        archive.extract("folder/chosen.mp3", target)
    assert target.read_bytes() == original
    receipt_path = target.with_name(target.name+".range.json")
    before = receipt_path.read_bytes()
    calls = len(server.calls)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    archive.extract("folder/chosen.mp3", target)
    assert len(server.calls) == calls and receipt_path.read_bytes() == before
    assert not target.with_name(target.name+".compressed.part").exists()
