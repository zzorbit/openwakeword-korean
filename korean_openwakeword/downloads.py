"""Checksum-pinned public assets with resumable downloads and no silent approval."""
from importlib.resources import files
from pathlib import Path
import hashlib
import re
import shutil
import time
import requests

from .common import now, read_json, write_json, writer_lock


def _file_identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class VerificationCache:
    """Hash evidence owned by one preparation call, never a persistent trust cache."""
    def __init__(self):
        self._verified = {}

    def matches(self, path, algorithm, expected):
        path = Path(path).resolve()
        return self._verified.get((str(path), algorithm, expected)) == _file_identity(path)

    def record(self, path, algorithm, expected, observed_identity):
        path = Path(path).resolve()
        if _file_identity(path) != observed_identity:
            raise ValueError("File changed during checksum verification")
        self._verified[(str(path), algorithm, expected)] = observed_identity


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
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
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
                if progress:
                    progress({"phase": "download", "done": offset, "total": size_bytes, "unit": "bytes",
                              "attempt": attempt + 1})
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(8 * 1024**2):
                        if not chunk:
                            continue
                        if size_bytes is not None and offset + len(chunk) > size_bytes:
                            raise ValueError("Download response exceeds the pinned asset size; partial preserved")
                        handle.write(chunk)
                        offset += len(chunk)
                        elapsed = time.monotonic() - started
                        if progress and (elapsed - emitted >= .5 or offset == size_bytes):
                            rate = (offset - initial) / elapsed if elapsed > 0 else None
                            progress({"phase": "download", "done": offset, "total": size_bytes, "unit": "bytes",
                                      "attempt": attempt + 1, "bytes_per_second": rate,
                                      "eta_seconds": max(0, (size_bytes - offset) / rate) if rate and size_bytes else None})
                            emitted = elapsed
                if size_bytes is not None and offset != size_bytes:
                    raise requests.exceptions.ChunkedEncodingError("Incomplete pinned download response")
            return
        except requests.exceptions.RequestException as error:
            if not _transient_network_error(error) or attempt == retries:
                raise
            if progress:
                progress({"phase": "retry", "done": partial.stat().st_size if partial.exists() else 0,
                          "total": size_bytes, "unit": "bytes", "attempt": attempt + 2,
                          "retry_limit": retries + 1, "retry_reason": type(error).__name__})
            time.sleep(min(.5 * 2**attempt, 2.0))


def dataset_catalog():
    return read_json(files("korean_openwakeword") / "resources/public_datasets.json")


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
        if output.exists():
            if not output.is_file() or size_bytes is not None and output.stat().st_size != size_bytes:
                raise ValueError("Existing download differs from the pinned size or checksum")
            try:
                verify_checksum(output, "sha256", sha256, progress=progress, verification_cache=verification_cache)
            except ValueError as error:
                raise ValueError("Existing download differs from the pinned size or checksum") from error
            reused = True
        else:
            reused = False
            partial = output.with_name(output.name + ".part")
            download_partial(url, partial, size_bytes, progress=progress)
            if size_bytes is not None and partial.stat().st_size != size_bytes:
                raise ValueError("Incomplete/altered asset; preserve .part for diagnosis or resume")
            verified_identity = _file_identity(partial)
            verify_checksum(partial, "sha256", sha256, progress=progress)
            partial.replace(output)
            if verification_cache is not None:
                if _file_identity(output)[:4] != verified_identity[:4]:
                    raise ValueError("File identity changed after verified download publication")
                verification_cache.record(output, "sha256", sha256, _file_identity(output))
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
