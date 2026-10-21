"""Checksum-pinned public assets with resumable downloads and no silent approval."""
from importlib.resources import files
from pathlib import Path
import re
import shutil
import requests

from .common import digest, now, read_json, write_json, writer_lock


def dataset_catalog():
    return read_json(files("korean_openwakeword") / "resources/public_datasets.json")


def download_checked(url, output, sha256, size_bytes=None):
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
            if (not output.is_file() or size_bytes is not None and output.stat().st_size != size_bytes
                    or digest(output) != sha256):
                raise ValueError("Existing download differs from the pinned size or checksum")
            reused = True
        else:
            reused = False
            partial = output.with_name(output.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            if size_bytes is not None and offset > size_bytes:
                raise ValueError("Partial file exceeds the pinned asset size")
            complete_partial = size_bytes is not None and offset == size_bytes
            if not complete_partial:
                required = max(0, (size_bytes or 0) - offset)
                if required and shutil.disk_usage(output.parent).free < required:
                    raise ValueError("Insufficient disk space for this download")
                headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"} if offset else {"Accept-Encoding": "identity"}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset:
                            raise ValueError("Server returned a mismatched resume range; partial file preserved")
                        if size_bytes is not None and match[3] != str(size_bytes):
                            raise ValueError("Upstream download size differs from the pin")
                    elif response.status_code == 200:
                        offset = 0  # Server ignored Range: safely restart, never append.
                    elif response.status_code != 206 or offset == 0:
                        raise ValueError("Unexpected partial download response")
                    with partial.open("ab" if offset else "wb") as handle:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            if not chunk:
                                continue
                            if size_bytes is not None and offset + len(chunk) > size_bytes:
                                raise ValueError("Download response exceeds the pinned asset size; partial preserved")
                            handle.write(chunk)
                            offset += len(chunk)
            if (size_bytes is not None and partial.stat().st_size != size_bytes) or digest(partial) != sha256:
                raise ValueError("Incomplete/altered asset; preserve .part for diagnosis or resume")
            partial.replace(output)
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
