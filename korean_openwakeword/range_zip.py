"""Bounded, ETag-pinned HTTP ZIP reads without a whole-archive fallback.

Extracted members retain their original bytes. CRC32 detects ZIP corruption;
local SHA256 receipts bind subsequent reuse, not a publisher's full ZIP hash.
The caller owns the output directory and its single-writer process lock.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import time
from urllib.parse import urlparse
import zipfile

import requests

from .common import digest, now, read_json, write_json


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or ":" in name or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))):
        raise ValueError("ZIP member path is unsafe")
    return name


class _RangeReader(io.RawIOBase):
    def __init__(self, url, size, *, expected_etag, session, max_download_bytes,
                 max_read_bytes, retries, timeout):
        self.url, self.size, self.pos = url, size, 0
        self.session, self.max_download_bytes = session, max_download_bytes
        self.max_read_bytes, self.retries, self.timeout = max_read_bytes, retries, timeout
        self.downloaded_bytes, self.requests, self.read_history = 0, [], []
        self.cache = None
        for attempt in range(retries + 1):
            response = None
            try:
                response = self.session.head(url, headers={"Accept-Encoding": "identity"},
                                             timeout=timeout, allow_redirects=False)
                response.raise_for_status()
                if (response.status_code != 200 or int(response.headers.get("Content-Length", -1)) != size
                        or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError("Archive size or content encoding differs from its frozen source")
                etag = response.headers.get("ETag")
                if (not isinstance(etag, str) or not re.fullmatch(r'"[^"\r\n]+"', etag)
                        or expected_etag is not None and etag != expected_etag):
                    raise ValueError("A matching strong archive ETag is required")
                self.etag = etag
                break
            except requests.exceptions.RequestException as error:
                if attempt == retries:
                    raise ValueError("ZIP source HEAD request failed after bounded retries") from error
                time.sleep(min(.1 * (attempt + 1), .3))
            finally:
                if response is not None:
                    response.close()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if type(offset) is not int or whence not in (0, 1, 2):
            raise ValueError("Invalid bounded archive seek")
        position = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if not 0 <= position <= self.size:
            raise ValueError("Archive seek leaves the frozen file")
        self.pos = position
        return position

    def read(self, size=-1):
        if size is not None and type(size) is not int:
            raise ValueError("Archive read size must be an integer")
        size = min(self.size - self.pos, self.size - self.pos if size is None or size < 0 else size)
        if size == 0:
            return b""
        if size > self.max_read_bytes:
            raise ValueError("Archive read exceeds its bounded memory limit")
        start, end = self.pos, self.pos + size - 1
        if self.cache is not None and self.cache[0] <= start and end < self.cache[0] + len(self.cache[1]):
            offset = start - self.cache[0]
            payload = self.cache[1][offset:offset+size]
        else:
            payload = self._fetch(start, end)
            self.cache = (start, payload)
        self.pos += size
        self.read_history.append({"start": start, "bytes": size,
                                  "sha256": hashlib.sha256(payload).hexdigest()})
        return payload

    def _fetch(self, start, end):
        size = end - start + 1
        for attempt in range(self.retries + 1):
            if self.downloaded_bytes + size > self.max_download_bytes:
                raise ValueError("HTTP ZIP transfer budget exceeded; whole download is forbidden")
            headers = {"Range": f"bytes={start}-{end}", "If-Match": self.etag,
                       "Accept-Encoding": "identity"}
            received, response = 0, None
            event = {"url": self.url, "start": start, "end": end, "attempt": attempt + 1}
            try:
                response = self.session.get(self.url, headers=headers, stream=True,
                                            timeout=self.timeout, allow_redirects=False)
                event["status"] = response.status_code
                if (response.status_code != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                        or response.headers.get("Content-Encoding", "identity") != "identity"
                        or response.headers.get("ETag") != self.etag
                        or response.headers.get("Content-Length") is not None
                        and int(response.headers["Content-Length"]) != size):
                    raise ValueError("Server did not honor the frozen ZIP byte range; no full fallback")
                parts = []
                for block in response.iter_content(128 * 1024):
                    received += len(block)
                    self.downloaded_bytes += len(block)
                    if received > size or self.downloaded_bytes > self.max_download_bytes:
                        raise ValueError("Server response exceeds its selected range or transfer budget")
                    parts.append(block)
                if received != size:
                    raise requests.exceptions.ChunkedEncodingError("Incomplete ZIP byte range")
                event["state"] = "verified_range"
                return b"".join(parts)
            except requests.exceptions.RequestException as error:
                event["state"] = "transport_error"
                if attempt == self.retries:
                    # zipfile treats OSError subclasses as an invalid ZIP;
                    # retain a meaningful bounded-transport error instead.
                    raise ValueError("Selected ZIP range transfer failed after bounded retries") from error
                time.sleep(min(.1 * (attempt + 1), .3))
            finally:
                event["bytes"] = received
                self.requests.append(event)
                if response is not None:
                    response.close()


class _MemberZip(io.RawIOBase):
    """A small virtual ZIP around already downloaded original compressed bytes."""
    def __init__(self, compressed, info):
        filename = info.filename.encode("utf-8")
        # FMA's original tracks.csv is BZIP2-compressed (method 12), which
        # requires ZIP extraction version 4.6. The wrapper preserves the
        # compressed payload and lets Python's bounded ZipExtFile decode it.
        flags, dos_date = 0x800, 33
        version = max(20, info.extract_version,
                      46 if info.compress_type == zipfile.ZIP_BZIP2 else 20)
        self.header = struct.pack("<4s5H3I2H", b"PK\x03\x04", version, flags,
            info.compress_type, 0, dos_date, info.CRC, info.compress_size, info.file_size,
            len(filename), 0) + filename
        central = struct.pack("<4s6H3I5H2I", b"PK\x01\x02", version, version, flags,
            info.compress_type, 0, dos_date, info.CRC, info.compress_size, info.file_size,
            len(filename), 0, 0, 0, 0, 0, 0) + filename
        self.tail = central + struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1,
            len(central), len(self.header) + info.compress_size, 0)
        self.payload = Path(compressed).open("rb")
        self.payload_size, self.pos = info.compress_size, 0
        self.size = len(self.header) + self.payload_size + len(self.tail)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if whence not in (0, 1, 2) or not 0 <= position <= self.size:
            raise ValueError("Invalid member ZIP seek")
        self.pos = position
        return position

    def read(self, size=-1):
        size = min(self.size-self.pos, self.size-self.pos if size < 0 else size)
        pieces = []
        while size:
            if self.pos < len(self.header):
                block = self.header[self.pos:self.pos+size]
            elif self.pos < len(self.header) + self.payload_size:
                offset = self.pos - len(self.header)
                self.payload.seek(offset)
                block = self.payload.read(min(size, self.payload_size-offset))
            else:
                offset = self.pos - len(self.header) - self.payload_size
                block = self.tail[offset:offset+size]
            if not block:
                raise ValueError("Incomplete compressed member cache")
            pieces.append(block)
            size -= len(block)
            self.pos += len(block)
        return b"".join(pieces)

    def close(self):
        self.payload.close()
        super().close()


class RangeZipArchive:
    """Read ZIP64 indexes and selected original files under a transfer cap.

    HTTPS, a frozen byte length and a strong ETag are mandatory. A server that
    returns a full response instead of a selected range is rejected unread.
    ``extract`` resumes compressed-byte checkpoints, then validates full CRC32.
    """
    def __init__(self, url, expected_size, *, expected_etag=None, session=None,
                 max_download_bytes, max_read_bytes=32 * 1024**2,
                 max_index_bytes=32 * 1024**2, max_member_bytes=512 * 1024**2,
                 retries=2, timeout=(20, 60), max_entries=1000000):
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("A public HTTPS archive URL is required")
        for name, value in (("expected_size", expected_size), ("max_download_bytes", max_download_bytes),
            ("max_read_bytes", max_read_bytes), ("max_index_bytes", max_index_bytes),
            ("max_member_bytes", max_member_bytes), ("max_entries", max_entries)):
            _positive(value, name)
        if type(retries) is not int or not 0 <= retries <= 5:
            raise ValueError("ZIP transport retries must be between zero and five")
        self.url, self.size_bytes = url, expected_size
        self.max_member_bytes = min(max_member_bytes, 2**32 - 1)
        self._owns_session = session is None
        self._session = session if session is not None else requests.Session()
        self._reader = _RangeReader(url, expected_size, expected_etag=expected_etag,
            session=self._session, max_download_bytes=max_download_bytes,
            max_read_bytes=min(max_read_bytes, max_index_bytes), retries=retries, timeout=timeout)
        self._zip = zipfile.ZipFile(self._reader)
        self.entries = self._zip.infolist()
        if len(self.entries) > max_entries or len({info.filename for info in self.entries}) != len(self.entries):
            raise ValueError("ZIP entry inventory is oversized or ambiguous")
        for info in self.entries:
            _safe_name(info.filename)
            _safe_name(info.orig_filename)
        offsets = sorted(info.header_offset for info in self.entries)
        if len(set(offsets)) != len(offsets) or any(not 0 <= offset < self._zip.start_dir for offset in offsets):
            raise ValueError("ZIP local member offsets are invalid or ambiguous")
        self._member_ends = dict(zip(offsets, offsets[1:]+[self._zip.start_dir]))
        index_reads = [row for row in self._reader.read_history if row["start"] == self._zip.start_dir]
        if not index_reads:
            raise ValueError("ZIP central directory bytes were not observed")
        index = max(index_reads, key=lambda row: row["bytes"])
        if index["bytes"] > max_index_bytes:
            raise ValueError("ZIP central directory exceeds its bounded index size")
        self.index_sha256, self.index_bytes = index["sha256"], index["bytes"]
        self.etag = self._reader.etag

    @property
    def downloaded_bytes(self): return self._reader.downloaded_bytes

    @property
    def requests(self): return self._reader.requests

    def binding(self):
        return {"url": self.url, "size_bytes": self.size_bytes, "etag": self.etag,
                "index_sha256": self.index_sha256, "index_bytes": self.index_bytes}

    def close(self):
        self._zip.close()
        self._reader.close()
        if self._owns_session:
            self._session.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _entry(self, name):
        name = name.filename if isinstance(name, zipfile.ZipInfo) else name
        _safe_name(name)
        info = self._zip.getinfo(name)
        if (info.is_dir() or info.flag_bits & (1 | 0x40)
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                                              zipfile.ZIP_BZIP2)
                or info.external_attr >> 16 & 0o170000 == 0o120000
                or not 0 < info.file_size <= self.max_member_bytes
                or not 0 < info.compress_size <= self.max_member_bytes
                or not 0 <= info.header_offset < self._zip.start_dir):
            raise ValueError("Selected ZIP member is empty, unsupported, unsafe or oversized")
        return info

    def read(self, name, *, max_bytes=256 * 1024**2):
        info = self._entry(name)
        _positive(max_bytes, "max_bytes")
        if info.file_size > max_bytes:
            raise ValueError("Selected ZIP member exceeds its read limit")
        with self._zip.open(info) as stream:
            pieces, received = [], 0
            while block := stream.read(min(1024**2, max_bytes-received+1)):
                received += len(block)
                if received > info.file_size or received > max_bytes:
                    raise ValueError("ZIP member expanded beyond its declared size")
                pieces.append(block)
        if received != info.file_size:
            raise ValueError("ZIP member size mismatch")
        return b"".join(pieces)

    def _compressed_start(self, info):
        self._reader.seek(info.header_offset)
        # Typical local headers fit in one small request. Three independent
        # header/name/extra HTTP calls would multiply latency for many tracks.
        initial = self._reader.read(min(512, self._zip.start_dir-info.header_offset))
        header = initial[:30]
        if len(header) != 30:
            raise ValueError("Incomplete local ZIP header")
        signature, version, flags, method, _, _, crc, compressed, size, name_size, extra_size = struct.unpack("<4s5H3I2H", header)
        if signature != b"PK\x03\x04" or flags != info.flag_bits or method != info.compress_type:
            raise ValueError("Local ZIP header differs from the frozen index")
        count = 30+name_size+extra_size
        if len(initial) < count:
            self._reader.seek(info.header_offset+len(initial))
            initial += self._reader.read(count-len(initial))
        raw_name, extra = initial[30:30+name_size], initial[30+name_size:count]
        if raw_name.decode("utf-8" if flags & 0x800 else "cp437") != info.orig_filename:
            raise ValueError("Local ZIP member name differs from the index")
        if not flags & 8:
            if compressed == 0xffffffff or size == 0xffffffff:
                offset, zip64 = 0, None
                while offset + 4 <= len(extra):
                    tag, count = struct.unpack_from("<HH", extra, offset)
                    offset += 4
                    if offset+count > len(extra):
                        raise ValueError("Invalid local ZIP64 extra data")
                    if tag == 1:
                        zip64 = extra[offset:offset+count]
                    offset += count
                if zip64 is None:
                    raise ValueError("Missing local ZIP64 member sizes")
                offset = 0
                for key in ("size", "compressed"):
                    if (size if key == "size" else compressed) == 0xffffffff:
                        if offset+8 > len(zip64):
                            raise ValueError("Incomplete local ZIP64 member sizes")
                        value = struct.unpack_from("<Q", zip64, offset)[0]
                        offset += 8
                        if key == "size": size = value
                        else: compressed = value
            if (crc, compressed, size) != (info.CRC, info.compress_size, info.file_size):
                raise ValueError("Local ZIP sizes or CRC differ from the index")
        start = info.header_offset+30+name_size+extra_size
        if start+info.compress_size > self._member_ends[info.header_offset]:
            raise ValueError("ZIP member overlaps another member or its central directory")
        return start

    def extract(self, name, destination, *, resume=True, chunk_bytes=4 * 1024**2, progress=None):
        info = self._entry(name)
        _positive(chunk_bytes, "chunk_bytes")
        chunk_bytes = min(chunk_bytes, self._reader.max_read_bytes)
        destination = Path(destination).absolute()
        if any(part.lower() == ".codex" for part in destination.parts):
            raise ValueError("Codex state is not an audio download directory")
        compressed = destination.with_name(destination.name+".compressed.part")
        checkpoint = destination.with_name(destination.name+".range-part.json")
        partial = destination.with_name(destination.name+".part")
        receipt_path = destination.with_name(destination.name+".range.json")
        for path in (destination, compressed, checkpoint, partial, receipt_path):
            if path.is_symlink() or path.exists() and not path.is_file():
                raise ValueError("ZIP destination or state path is not a regular owned file")
        request = {"archive": self.binding(), "member": info.filename, "header_offset": info.header_offset,
                   "compressed_bytes": info.compress_size, "bytes": info.file_size,
                   "crc32": f"{info.CRC:08x}", "compression": info.compress_type}
        if receipt_path.exists():
            saved = read_json(receipt_path)
            if (saved.get("kind") != "range_zip_member_v1" or saved.get("request") != request
                    or saved.get("path") != str(destination) or not destination.is_file()
                    or destination.stat().st_size != info.file_size or digest(destination) != saved.get("sha256")):
                raise ValueError("Completed ZIP member or its receipt changed")
            if checkpoint.exists():
                committed = read_json(checkpoint)
                if committed.get("kind") != "range_zip_partial_v1" or committed.get("request") != request:
                    raise ValueError("Completed member has unrelated partial state")
                compressed.unlink(missing_ok=True)
                checkpoint.unlink()
            return saved
        if destination.exists():
            if not checkpoint.is_file():
                raise ValueError("Existing member without its completed receipt is preserved")
            committed = read_json(checkpoint)
            if (committed.get("kind") != "range_zip_partial_v1" or committed.get("request") != request
                    or committed.get("crc32_verified") is not True
                    or committed.get("output_bytes") != info.file_size
                    or destination.stat().st_size != info.file_size
                    or digest(destination) != committed.get("output_sha256")):
                raise ValueError("Unreceipted member does not match its committed verified output")
            result = self._receipt(request, destination, committed["output_sha256"])
            write_json(receipt_path, result)
            compressed.unlink(missing_ok=True)
            checkpoint.unlink()
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not resume and (checkpoint.exists() or compressed.exists() or partial.exists()):
            raise ValueError("Existing partial ZIP member needs explicit resume")
        if checkpoint.exists():
            state = read_json(checkpoint)
            if state.get("kind") != "range_zip_partial_v1" or state.get("request") != request:
                raise ValueError("Partial ZIP member belongs to different source bytes")
            offset = state.get("compressed_bytes")
            if type(offset) is not int or not 0 <= offset <= info.compress_size:
                raise ValueError("Invalid compressed-member checkpoint")
        else:
            if compressed.exists() or partial.exists():
                raise ValueError("Unowned partial ZIP files are preserved")
            offset = 0
            state = {"kind": "range_zip_partial_v1", "request": request,
                     "compressed_bytes": 0, "compressed_sha256": hashlib.sha256(b"").hexdigest()}
            write_json(checkpoint, state)
        hasher = hashlib.sha256()
        if compressed.exists():
            if compressed.stat().st_size < offset:
                raise ValueError("Partial ZIP cache is shorter than its committed checkpoint")
            with compressed.open("rb") as stream:
                remaining = offset
                while remaining:
                    block = stream.read(min(1024**2, remaining))
                    hasher.update(block)
                    remaining -= len(block)
            if hasher.hexdigest() != state.get("compressed_sha256"):
                raise ValueError("Committed compressed ZIP member changed")
            if compressed.stat().st_size != offset:
                with compressed.open("r+b") as stream:
                    stream.truncate(offset)  # Only uncommitted owned partial bytes.
        elif offset:
            raise ValueError("Committed compressed ZIP member is missing")
        if shutil.disk_usage(destination.parent).free < info.compress_size-offset+info.file_size+65536:
            raise ValueError("Insufficient selected storage for the original ZIP member")
        start = self._compressed_start(info)
        with compressed.open("ab") as stream:
            while offset < info.compress_size:
                self._reader.seek(start+offset)
                block = self._reader.read(min(chunk_bytes, info.compress_size-offset))
                stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
                hasher.update(block)
                offset += len(block)
                state.update(compressed_bytes=offset, compressed_sha256=hasher.hexdigest())
                write_json(checkpoint, state)
                if progress:
                    progress({"member": info.filename, "done": offset, "total": info.compress_size, "unit": "bytes"})
        actual_hash, count = hashlib.sha256(), 0
        try:
            with _MemberZip(compressed, info) as member_zip, zipfile.ZipFile(member_zip) as archive:
                with archive.open(info.filename) as original, partial.open("wb") as output:
                    while block := original.read(1024**2):
                        count += len(block)
                        if count > info.file_size:
                            raise ValueError("ZIP member expanded beyond its declared size")
                        actual_hash.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            if count != info.file_size:
                raise ValueError("ZIP member size mismatch")
            state.update(output_sha256=actual_hash.hexdigest(), output_bytes=count, crc32_verified=True)
            write_json(checkpoint, state)
            os.replace(partial, destination)
            result = self._receipt(request, destination, actual_hash.hexdigest())
            write_json(receipt_path, result)
            compressed.unlink()
            checkpoint.unlink()
            return result
        except (OSError, ValueError, zipfile.BadZipFile):
            partial.unlink(missing_ok=True)
            raise

    def _receipt(self, request, destination, sha256):
        return {"kind": "range_zip_member_v1", "request": request, "path": str(destination),
            "bytes": request["bytes"], "sha256": sha256, "crc32": request["crc32"],
            "crc32_verified": True, "original_bytes_preserved": True, "audio_reencoded": False,
            "whole_archive_sha1_verified": False, "content_approved": False, "completed_at": now()}
