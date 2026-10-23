"""Pinned ZIP members over resumable bounded ranges, including split volumes.

The catalog pins original headers, compressed payloads and decoded files. A
successful selective transaction never claims to verify a complete archive.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import hashlib
import os
import re
import shutil
import struct
import threading
import time
import uuid
import zlib

import requests

from .common import digest, identity, now, read_json, write_json, writer_lock, _retry_windows_sharing
from .downloads import _transient_network_error


POLICY = "selected_common_zip_v1"
_MAX_CHUNK = 16 * 1024**2
_GAP = 256 * 1024


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _publish_original(temporary, output):
    """Publish without replacing a file created by another writer.

    Windows rename rejects existing destinations. POSIX rename replaces them,
    so use an exclusive hard-link publish there and retain the temporary on
    unsupported filesystems or collisions instead of weakening the contract.
    """
    if os.name == "nt":
        _retry_windows_sharing(lambda: os.rename(temporary, output))
    else:
        os.link(temporary, output)
        temporary.unlink()


def _integer(value, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("Selected ZIP sizes and offsets must be bounded integers")
    return value


def _name(name):
    if not isinstance(name, str):
        raise ValueError("Selected ZIP member name must be a string")
    parts = PurePosixPath(name)
    if (not name or parts.is_absolute() or "\\" in name or ":" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Selected ZIP member leaves its original directory")
    return parts


def _output(root, name):
    parts = _name(name)
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Selected ZIP destination escapes its directory")
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError("Selected ZIP paths cannot be symbolic links")
    return path


def _plan(group):
    volumes, position = [], 0
    for row in group["volumes"]:
        size = _integer(row["size_bytes"], True)
        if not isinstance(row["source_url"], str) or not row["source_url"].startswith("https://"):
            raise ValueError("Selected ZIP volumes require HTTPS sources")
        volumes.append({**row, "start": position, "end": position + size})
        position += size
    if not volumes or len({v["id"] for v in volumes}) != len(volumes):
        raise ValueError("Selected ZIP volumes must be distinct")
    names, chunks = set(), []
    for row in sorted(group["members"], key=lambda row: row["header_offset"]):
        if row["name"] in names:
            raise ValueError("Duplicate selected ZIP member")
        names.add(row["name"])
        _name(row["name"])
        start = _integer(row["header_offset"])
        header = _integer(row["header_bytes"], True)
        compressed = _integer(row["compressed_bytes"], True)
        size = _integer(row["size_bytes"], True)
        end = start + header + compressed
        if (row["data_offset"] != start + header or header < 30 or end > position
                or end-start > _MAX_CHUNK or size > 512 * 1024**2
                or row["compression_method"] not in (0, 8) or row.get("flags", 0) & 1):
            raise ValueError("Unsupported or unbounded selected ZIP member")
        for key in ("sha256", "header_sha256", "compressed_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("Selected ZIP member requires complete SHA256 pins")
        if not re.fullmatch(r"[0-9a-f]{8}", row["crc32"]):
            raise ValueError("Selected ZIP CRC pin is invalid")
        if chunks and start < chunks[-1]["members"][-1]["data_offset"] + chunks[-1]["members"][-1]["compressed_bytes"]:
            raise ValueError("Selected ZIP members overlap")
        if chunks and start - chunks[-1]["end"] <= _GAP and end-chunks[-1]["start"] <= _MAX_CHUNK:
            chunks[-1]["end"] = end
            chunks[-1]["members"].append(row)
        else:
            chunks.append({"start": start, "end": end, "members": [row]})
    if not names:
        raise ValueError("Selected ZIP inventory is empty")
    return volumes, chunks


def selected_transfer_bytes(group):
    """Exact requested bodies before retries; the frozen index is bundled."""
    _, chunks = _plan(group)
    return sum(c["end"]-c["start"] for c in chunks)


def _decode(row, span):
    header, compressed = span[:row["header_bytes"]], span[row["header_bytes"]:]
    if _hash(header) != row["header_sha256"] or _hash(compressed) != row["compressed_sha256"]:
        raise ValueError("Selected ZIP original compressed bytes changed; source preserved")
    fields = struct.unpack_from("<4s5H3I2H", header)
    name_bytes = header[30:30+fields[9]]
    name = name_bytes.decode("utf-8" if fields[2] & 0x800 else "cp437")
    if (fields[0] != b"PK\x03\x04" or name != row["name"] or fields[3] != row["compression_method"]
            or fields[2] != row["flags"] or len(header) != 30+fields[9]+fields[10]):
        raise ValueError("Selected ZIP local header differs from its immutable pin")
    if row["compression_method"] == 0:
        payload = compressed
    else:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(compressed, row["size_bytes"]+1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Selected ZIP compressed member is incomplete or oversized")
    if (len(payload) != row["size_bytes"] or f"{zlib.crc32(payload):08x}" != row["crc32"]
            or _hash(payload) != row["sha256"]):
        raise ValueError("Selected ZIP original file failed its size/CRC/SHA256 checks")
    return payload


def acquire_selected(group, destination, state_dir, *, progress=None, workers=4,
                     local_volumes=None, session_factory=None):
    """Publish only verified original members; failed partials remain resumable."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Selected ZIP workers must be between one and four")
    destination, state_dir = Path(destination).resolve(), Path(state_dir).resolve()
    if any(part.lower() == ".codex" for p in (destination, state_dir) for part in p.parts):
        raise ValueError("Codex state is not a selected ZIP cache")
    volumes, chunks = _plan(group)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    request = {"policy": POLICY, "group_sha256": identity(group), "destination": str(destination)}
    state_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(state_dir):
        path = state_dir / "request.json"
        if path.exists() and read_json(path) != request:
            raise ValueError("Selected ZIP cache belongs to another immutable request")
        if not path.exists():
            write_json(path, request)
        existing = set()
        for row in group["members"]:
            output = _output(destination, row["name"])
            if output.exists():
                if not output.is_file() or output.stat().st_size != row["size_bytes"] or digest(output) != row["sha256"]:
                    raise ValueError("Existing selected original changed; file preserved")
                existing.add(row["name"])
        pending = [c for c in chunks if any(r["name"] not in existing for r in c["members"])]
        required = sum(r["size_bytes"] for c in pending for r in c["members"] if r["name"] not in existing)
        required += workers * _MAX_CHUNK
        if pending and shutil.disk_usage(state_dir).free < required:
            raise ValueError("Insufficient space for selected originals and bounded transfer scratch")
        lock, local = threading.RLock(), threading.local()
        sessions, etags, completed = [], {}, {}
        totals = {"network": 0, "local": 0}
        total = sum(c["end"]-c["start"] for c in pending)
        started = time.monotonic()
        last_emit = [started-1]
        session_factory = session_factory or requests.Session

        def emit(phase="download", **extra):
            if progress:
                with lock:
                    done = sum(completed.values())
                    instant = time.monotonic()
                    if phase in ("download", "hash") and instant-last_emit[0] < .5 and done != total:
                        return
                    last_emit[0] = instant
                    elapsed = instant-started
                    transferred = totals["network"] + totals["local"]
                    rate = transferred/elapsed if transferred and elapsed else None
                    progress({"label": "필요한 평가 자료 다운로드: " + group.get("id", "ZIP"),
                        "phase": phase, "done": done, "total": total, "unit": "bytes",
                        "bytes_per_second": rate, "eta_seconds": (total-done)/rate if rate else None, **extra})

        def session():
            if not hasattr(local, "session"):
                local.session = session_factory()
                with lock:
                    sessions.append(local.session)
            return local.session

        def segment(volume, start, end, key):
            length = end-start
            cached = (local_volumes or {}).get(volume["id"])
            if cached is not None and Path(cached).is_file():
                cached = Path(cached)
                if cached.is_symlink() or cached.stat().st_size != volume["size_bytes"]:
                    raise ValueError("Cached original ZIP volume changed; source preserved")
                with cached.open("rb") as stream:
                    stream.seek(start); data = stream.read(length)
                if len(data) != length:
                    raise ValueError("Cached ZIP range is truncated")
                with lock:
                    totals["local"] += length; completed[key] = length
                emit("hash", label="기존 평가 원본 확인: " + group.get("id", "ZIP"))
                return data, None
            partial = state_dir / "ranges" / (identity([volume["id"], start, end]) + ".part")
            partial.parent.mkdir(parents=True, exist_ok=True)
            if partial.is_symlink() or partial.exists() and (not partial.is_file() or partial.stat().st_size > length):
                raise ValueError("Selected range partial changed; file preserved")
            for attempt in range(3):
                offset = partial.stat().st_size if partial.exists() else 0
                with lock:
                    completed[key] = offset
                if offset == length:
                    return partial.read_bytes(), partial
                response = None
                try:
                    headers = {"Range": f"bytes={start+offset}-{end-1}", "Accept-Encoding": "identity"}
                    with lock:
                        etag = etags.get(volume["id"])
                    if etag:
                        headers["If-Match"] = etag
                    response = session().get(volume["source_url"], headers=headers, stream=True,
                                             timeout=(30, 120), allow_redirects=False)
                    response.raise_for_status()
                    if (response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start+offset}-{end-1}/{volume['size_bytes']}"
                            or response.headers.get("Content-Encoding", "identity") != "identity"
                            or int(response.headers.get("Content-Length", -1)) != length-offset):
                        raise ValueError("Server ignored or altered the selected range; whole download is forbidden")
                    observed = response.headers.get("ETag")
                    if observed and re.fullmatch(r'"[^"\r\n]+"', observed):
                        with lock:
                            if volume["id"] in etags and etags[volume["id"]] != observed:
                                raise ValueError("Selected ZIP source changed during transfer")
                            etags[volume["id"]] = observed
                    with partial.open("ab") as output:
                        for block in response.iter_content(128 * 1024):
                            if offset+len(block) > length:
                                raise ValueError("Selected range exceeds its exact bound")
                            output.write(block); offset += len(block)
                            with lock:
                                totals["network"] += len(block); completed[key] = offset
                            emit()
                        output.flush(); os.fsync(output.fileno())
                    if offset != length:
                        raise requests.exceptions.ChunkedEncodingError("Selected range was interrupted")
                    return partial.read_bytes(), partial
                except requests.exceptions.RequestException as error:
                    if not _transient_network_error(error) or attempt == 2:
                        raise
                    emit("retry", attempt=attempt+2, retry_limit=3, retry_reason=type(error).__name__)
                    time.sleep(min(.5 * 2**attempt, 2))
                finally:
                    if response is not None:
                        response.close()

        def transfer(chunk):
            fragments, partials = [], []
            for volume in volumes:
                start, end = max(chunk["start"], volume["start"]), min(chunk["end"], volume["end"])
                if start < end:
                    data, partial = segment(volume, start-volume["start"], end-volume["start"], str([start, end]))
                    fragments.append(data)
                    if partial is not None:
                        partials.append(partial)
            data = b"".join(fragments)
            for row in chunk["members"]:
                offset = row["header_offset"]-chunk["start"]
                payload = _decode(row, data[offset:offset+row["header_bytes"]+row["compressed_bytes"]])
                output = _output(destination, row["name"])
                if row["name"] in existing:
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".part")
                with temporary.open("xb") as stream:
                    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                if output.exists():
                    raise ValueError("Selected original appeared concurrently; both files preserved")
                _publish_original(temporary, output)
            commit = state_dir / "chunks" / (identity([chunk["start"], chunk["end"]]) + ".json")
            write_json(commit, {"request": request, "start": chunk["start"], "end": chunk["end"],
                "files": [{"path": str(_output(destination, r["name"])), "sha256": r["sha256"],
                           "size_bytes": r["size_bytes"]} for r in chunk["members"]]})
            # Only newly owned compressed scratch is retired, after originals
            # and their durable commit exist. Original archives are untouched.
            for partial in partials:
                partial.unlink()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(transfer, chunk) for chunk in pending]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        finally:
            for value in sessions:
                value.close()
        files = [{**r, "path": str(_output(destination, r["name"]))} for r in group["members"]]
        record = {"kind": POLICY, "request": request, "group_sha256": identity(group), "files": files,
            "observed_transfer_bytes_this_run": totals["network"], "local_read_bytes_this_run": totals["local"],
            "whole_archive_verified": False, "original_bytes_preserved": True,
            "audio_reencoded": False, "content_approved": False,
            "reused_members": len(existing), "completed_at": now()}
        path = state_dir / ("receipt_" + identity(record) + ".json")
        write_json(path, record)
        return {**record, "receipt": {"path": str(path), "sha256": digest(path)}}
