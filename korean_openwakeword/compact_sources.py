"""Keep original archive members once, retiring only owned temporary archives.

This is a new cache layout. Historical downloads and their receipts are never
migrated or deleted. A pinned archive is downloaded and verified, every member
is safely extracted and SHA256-verified, and durable preservation evidence is
committed before the owned archive can be removed. Reuse validates retained
original bytes without re-downloading the retired archive.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import uuid

from .common import digest, now, read_json, writer_lock


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def _durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _directory(path):
    path = Path(path).expanduser().absolute()
    if any(part.lower() in (".codex", ".git") for part in path.parts):
        raise ValueError("Compact originals require the selected data workspace")
    if any(_is_link(part) for part in (path, *path.parents)):
        raise ValueError("Compact cache directories cannot traverse links")
    resolved = path.resolve()
    if any(part.lower() in (".codex", ".git") for part in resolved.parts):
        raise ValueError("Resolved compact originals require the selected data workspace")
    return resolved


def _is_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _inside(root, relative):
    from .studio_data import _archive_target
    # Inspect the lexical path too: _archive_target resolves paths, which can
    # hide a pre-existing symlink/junction pointing elsewhere inside the root.
    lexical = root.joinpath(*PurePosixPath(relative).parts)
    if any(_is_link(part) for part in (lexical, *lexical.parents) if part.is_relative_to(root)):
        raise ValueError("Compact retained source cannot traverse a link")
    return _archive_target(root, relative)


def _asset_file(asset, directory, progress=None):
    """Use existing pinned/resumable downloaders; preserve completed bytes."""
    from .commercial_sources import _verify_cached_original
    from .downloads import download_checked
    target = _inside(directory, asset["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    acquisition = "existing"
    if target.exists():
        if not target.is_file() or target.stat().st_size != asset["size_bytes"]:
            raise ValueError("Existing compact asset has changed size")
        checksum = _verify_cached_original(target, asset)
    elif asset["checksum_algorithm"] == "sha256":
        record = download_checked(asset["source_url"], target, asset["sha256"], asset["size_bytes"])
        checksum = record["sha256"]
        acquisition = "downloaded_here"
    else:
        from .studio_data import _download
        downloaded = _download({**asset, "id": asset.get("id", asset["path"])}, target.parent, progress)
        acquisition = "downloaded_here"
        if downloaded.resolve() != target:
            # Ancestor caches belong to another policy. Preserve them and keep
            # a verified local metadata copy; never claim ownership of them.
            try:
                os.link(downloaded, target)
            except OSError:
                shutil.copyfile(downloaded, target)
            acquisition = "external_reuse"
        checksum = _verify_cached_original(target, asset)
    return {"source_path": asset["path"], "path": str(target), "sha256": checksum,
            "bytes": asset["size_bytes"], "checksum_algorithm": asset["checksum_algorithm"],
            "publisher_checksum": asset[asset["checksum_algorithm"]],
            "source_url": asset["source_url"], "upstream_revision": asset["upstream_revision"],
            "acquisition": acquisition}


def _file_identity(path):
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino}


def _sync_file(path):
    # Windows CRT _commit requires a writable descriptor. No bytes are changed.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _verify_download_claim(directory, index, asset, request_binding, part):
    path = directory / "downloads" / f"part_{index:04d}.json"
    if part.get("download") != _binding(path):
        raise ValueError("Compact archive acquisition binding changed")
    claim = read_json(path)
    if (claim.get("kind") != "compact_archive_acquisition_v1" or claim.get("asset") != asset
            or claim.get("request") != request_binding or claim.get("archive") != part.get("archive")
            or type(claim.get("retire_eligible")) is not bool
            or part.get("retire_archive") is not claim["retire_eligible"]):
        raise ValueError("Compact archive acquisition evidence changed")
    if claim["retire_eligible"] and claim["archive"].get("acquisition") != "downloaded_here":
        raise ValueError("Only an explicitly acquired temporary archive can be retired")
    return claim


def _verify_part(directory, index, asset, request_binding, part):
    extraction_path = directory / "audio" / f"part_{index:04d}" / "extraction_receipt.json"
    if (part.get("kind") != "compact_source_part_v1" or part.get("asset") != asset
            or part.get("request") != request_binding
            or part.get("extraction") != _binding(extraction_path)):
        raise ValueError("Compact original extraction binding changed")
    claim = _verify_download_claim(directory, index, asset, request_binding, part)
    extraction = read_json(extraction_path)
    archive = _inside(directory / "originals", asset["path"])
    if (extraction.get("request", {}).get("archive") != {
            "path": str(archive), "sha256": part["archive"]["sha256"]}
            or extraction["request"].get("members") is not None
            or part["archive"].get("path") != str(archive)
            or part["archive"].get("source_path") != asset["path"]
            or part["archive"].get("bytes") != asset["size_bytes"]
            or part.get("raw_bytes") != extraction.get("bytes")):
        raise ValueError("Compact extraction no longer binds its complete original archive")
    return extraction_path, extraction, claim


def _verify_raw(directory, extraction, max_bytes, *, sync=False):
    """Check the complete original-member manifest, not a sampled subset."""
    files = extraction.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Compact extraction needs a nonempty original-file manifest")
    names, size = set(), 0
    for row in files:
        name = row.get("path")
        if not isinstance(name, str):
            raise ValueError("Compact original member path is missing")
        if name in names:
            raise ValueError("Compact extraction repeats original members")
        names.add(name)
        target = _inside(directory, name)
        if (not target.is_file() or type(row.get("bytes")) is not int or row["bytes"] < 0
                or target.stat().st_size != row["bytes"] or digest(target) != row.get("sha256")):
            raise ValueError("Retained compact original is missing or changed")
        if sync:
            _sync_file(target)
        size += row["bytes"]
    if size != extraction.get("bytes") or size > max_bytes:
        raise ValueError("Retained compact originals exceed the storage bound")
    return size


def _retire_archive(archive, part, request, originals):
    """Only the verified, exclusively owned, explicitly named temp is retired."""
    if not part["retire_archive"] or not archive.exists():
        return
    owned = request["owned_archives"].get(part["asset"]["path"])
    if owned is not True or archive != _inside(originals, part["asset"]["path"]):
        raise ValueError("Compact archive retirement has no ownership proof")
    if archive.is_symlink() or archive.stat().st_nlink != 1:
        raise ValueError("Owned compact archive became shared; original is preserved")
    claim = read_json(part["download"]["path"])
    if _file_identity(archive) != claim["file_identity"]:
        raise ValueError("Owned compact archive was replaced; original is preserved")
    if archive.stat().st_size != part["archive"]["bytes"] or digest(archive) != part["archive"]["sha256"]:
        raise ValueError("Compact archive changed before retirement; original is preserved")
    # This function is called only after its durable part receipt and complete
    # raw-file verification. Removing this one temporary is recoverable from
    # retained originals, provider pins and preservation evidence.
    archive.unlink()


def _result(directory, receipt, *, resumed):
    receipt_binding = _binding(directory / "compact_receipt.json")
    parts = receipt["parts"]
    extractions = [read_json(part["path"])["extraction"] for part in parts]
    downloads = [read_json(part["path"])["download"] for part in parts]
    return {"metadata_directory": str(directory / "originals"), "audio_directory": str(directory / "audio"),
            "compact_receipt": receipt_binding,
            "receipts": [receipt_binding, *parts, *extractions, *downloads],
            **{key: receipt[key] for key in ("reserved_download_bytes", "original_storage_bytes",
                "retired_archive_bytes", "temporary_archive_peak_bytes")},
            "artifacts": [receipt_binding["path"], receipt["source_resolution"]["path"],
                          *[b["path"] for b in parts + extractions + downloads]], "resumed": resumed}


def verify_compact_source(directory):
    """Read-only full verification of a completed compact cache, no network."""
    from .commercial_sources import _validate_resolution, _verify_cached_original
    directory = _directory(directory)
    receipt_path = directory / "compact_receipt.json"
    receipt = read_json(receipt_path)
    request_path = directory / "compact_request.json"
    if (receipt.get("kind") != "compact_source_preservation_v1"
            or receipt.get("request") != _binding(request_path)):
        raise ValueError("Compact completion request binding changed")
    request = read_json(request_path)
    resolution_path = directory / "originals/source_resolution.json"
    if receipt.get("source_resolution") != _binding(resolution_path):
        raise ValueError("Compact provider resolution changed")
    resolution = read_json(resolution_path); _validate_resolution(resolution)
    if (request.get("kind") != "compact_source_request_v1" or request.get("directory") != str(directory)
            or request.get("source_id") != resolution["source_id"]
            or request.get("resolution_sha256") != resolution["resolution_sha256"]
            or set(request.get("owned_archives", {})) != {a["path"] for a in resolution["assets"]}
            or any(type(owned) is not bool for owned in request["owned_archives"].values())):
        raise ValueError("Compact cache belongs to another directory or source")
    originals = directory / "originals"
    metadata = receipt.get("metadata", [])
    if len(metadata) != len(resolution["metadata_assets"]):
        raise ValueError("Compact metadata coverage changed")
    retained, retired = 0, 0
    for asset, record in zip(resolution["metadata_assets"], metadata):
        path = _inside(originals, asset["path"])
        if (record.get("source_path") != asset["path"] or record.get("path") != str(path)
                or record.get("bytes") != asset["size_bytes"] or not path.is_file()
                or path.stat().st_size != asset["size_bytes"]
                or _verify_cached_original(path, asset) != record.get("sha256")):
            raise ValueError("Retained compact metadata is missing or changed")
        retained += asset["size_bytes"]
    parts = receipt.get("parts", [])
    if len(parts) != len(resolution["assets"]):
        raise ValueError("Compact original archive coverage changed")
    for index, (asset, binding) in enumerate(zip(resolution["assets"], parts)):
        part_path = directory / "parts" / f"part_{index:04d}.json"
        if binding != _binding(part_path):
            raise ValueError("Compact original preservation record changed")
        part = read_json(part_path)
        extraction_path, extraction, _ = _verify_part(directory, index, asset, receipt["request"], part)
        retained += _verify_raw(extraction_path.parent, extraction, 1024**5)
        archive = _inside(originals, asset["path"])
        if part["retire_archive"]:
            if request.get("owned_archives", {}).get(asset["path"]) is not True or archive.exists():
                raise ValueError("Retired compact archive reappeared or ownership changed")
            retired += asset["size_bytes"]
        else:
            if (not archive.is_file() or archive.stat().st_size != asset["size_bytes"]
                    or _verify_cached_original(archive, asset) != part["archive"]["sha256"]):
                raise ValueError("Preserved external archive is missing or changed")
            retained += asset["size_bytes"]
    if (retained != receipt.get("original_storage_bytes") or retired != receipt.get("retired_archive_bytes")
            or resolution["total_bytes"] != receipt.get("reserved_download_bytes")):
        raise ValueError("Compact retained storage accounting changed")
    return _result(directory, receipt, resumed=True)


def prepare_compact_source(resolution, directory, *, max_download_bytes,
                           max_extract_bytes, progress=None):
    """Return retained metadata/raw originals and stable preservation bindings.

    ``max_extract_bytes`` limits permanent retained original bytes, excluding
    one temporary archive. Current physical free space is also checked by the
    download and extraction APIs. Runtime limits are checked but not frozen in
    the source identity, so another project can reuse this exact cache.
    """
    from .commercial_sources import _validate_resolution
    from .studio_data import extract_archive
    _validate_resolution(resolution)
    if resolution["source_id"] not in ("audioset_unbalanced", "mls", "yodas2"):
        raise ValueError("Compact original archives support AudioSet, MLS and YODAS2")
    for value in (max_download_bytes, max_extract_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("Compact cache budgets need positive integer bytes")
    if resolution["total_bytes"] > max_download_bytes:
        raise ValueError("Selected compact source exceeds its download byte cap")
    if sum(a["size_bytes"] for a in resolution["metadata_assets"]) >= max_extract_bytes:
        raise ValueError("Compact provider metadata exceeds the retained-byte cap")
    directory = _directory(directory)
    originals, audio = directory / "originals", directory / "audio"
    originals.mkdir(parents=True, exist_ok=True); audio.mkdir(exist_ok=True)
    directory = _directory(directory)
    request_path, receipt_path = directory / "compact_request.json", directory / "compact_receipt.json"
    with writer_lock(directory):
        if receipt_path.exists():
            checked = verify_compact_source(directory)
            saved = read_json(originals / "source_resolution.json")
            if saved["resolution_sha256"] != resolution["resolution_sha256"]:
                raise ValueError("Completed compact cache belongs to another source")
            if checked["original_storage_bytes"] > max_extract_bytes:
                raise ValueError("Compact retained originals exceed the current storage bound")
            return checked
        body = {"kind": "compact_source_request_v1", "resolution_sha256": resolution["resolution_sha256"],
                "directory": str(directory), "source_id": resolution["source_id"]}
        if request_path.exists():
            request = read_json(request_path)
            if any(request.get(key) != value for key, value in body.items()):
                raise ValueError("Compact cache request changed")
            if set(request.get("owned_archives", {})) != {a["path"] for a in resolution["assets"]}:
                raise ValueError("Compact archive ownership coverage changed")
            if any(type(owned) is not bool for owned in request["owned_archives"].values()):
                raise ValueError("Compact archive ownership values changed")
        else:
            # Pre-existing files or partials have unknown owners and therefore
            # cannot be retired by this policy, even inside its dedicated root.
            request = {**body, "owned_archives": {asset["path"]:
                not _inside(originals, asset["path"]).exists()
                and not _inside(originals, asset["path"] + ".part").exists()
                for asset in resolution["assets"]}, "created_at": now()}
            _durable_json(request_path, request)
        resolution_path = originals / "source_resolution.json"
        if resolution_path.exists():
            saved = read_json(resolution_path); _validate_resolution(saved)
            if saved["resolution_sha256"] != resolution["resolution_sha256"]:
                raise ValueError("Compact metadata belongs to another provider selection")
        else:
            _durable_json(resolution_path, resolution)
        previous = read_json(receipt_path) if receipt_path.exists() else None
        metadata, parts, bindings = [], [], []
        retained = 0
        for asset in resolution["metadata_assets"]:
            record = _asset_file(asset, originals, progress)
            _sync_file(Path(record["path"]))
            metadata.append(record); retained += record["bytes"]
        if retained > max_extract_bytes:
            raise ValueError("Compact provider metadata exceeds the retained-byte cap")
        for index, asset in enumerate(resolution["assets"]):
            archive = _inside(originals, asset["path"])
            destination = audio / f"part_{index:04d}"
            part_path = directory / "parts" / f"part_{index:04d}.json"
            download_path = directory / "downloads" / f"part_{index:04d}.json"
            extraction_path = destination / "extraction_receipt.json"
            if part_path.exists():
                part = read_json(part_path)
                extraction_path, extraction, _ = _verify_part(directory, index, asset, _binding(request_path), part)
                raw_bytes = _verify_raw(destination, extraction, max_extract_bytes - retained, sync=True)
            else:
                if download_path.exists():
                    claim = read_json(download_path)
                    if (claim.get("kind") != "compact_archive_acquisition_v1" or claim.get("asset") != asset
                            or claim.get("request") != _binding(request_path)):
                        raise ValueError("Compact archive acquisition request changed")
                    archive_record = claim["archive"]
                    if not archive.is_file():
                        raise ValueError("Compact acquired archive is missing; preservation cannot continue")
                    checked = _asset_file(asset, originals, progress)
                    if any(checked[key] != archive_record[key] for key in ("path", "sha256", "bytes")):
                        raise ValueError("Compact acquired archive changed")
                    if claim["retire_eligible"] and _file_identity(archive) != claim["file_identity"]:
                        raise ValueError("Compact acquired archive was replaced; original is preserved")
                else:
                    archive_record = _asset_file(asset, originals, progress)
                    retire = (request["owned_archives"][asset["path"]]
                              and archive_record["acquisition"] == "downloaded_here"
                              and archive.stat().st_nlink == 1)
                    claim = {"kind": "compact_archive_acquisition_v1", "asset": asset,
                        "archive": archive_record, "request": _binding(request_path),
                        "retire_eligible": retire, "file_identity": _file_identity(archive), "acquired_at": now()}
                    _durable_json(download_path, claim)
                if extraction_path.exists():
                    # A crash may happen after safe extraction but before the
                    # compact part commit. Reuse its original fixed allowance.
                    allowance = read_json(extraction_path)["request"]["max_bytes"]
                else:
                    allowance = max_extract_bytes - retained - (0 if claim["retire_eligible"] else asset["size_bytes"])
                if allowance <= 0:
                    raise ValueError("Compact retained originals exceed the storage byte cap")
                extraction = extract_archive(archive, destination, max_bytes=allowance)
                raw_bytes = _verify_raw(destination, extraction, max_extract_bytes - retained, sync=True)
                retire = claim["retire_eligible"]
                part = {"kind": "compact_source_part_v1", "asset": asset, "archive": archive_record,
                        "request": _binding(request_path), "extraction": _binding(extraction_path),
                        "download": _binding(download_path),
                        "retire_archive": retire, "raw_bytes": raw_bytes,
                        "retention": "raw_originals_verified_archive_temporary" if retire else "preexisting_or_shared_archive_preserved",
                        "content_approved": False, "committed_at": now()}
                _durable_json(part_path, part)
            _sync_file(extraction_path)
            retained += raw_bytes
            if not part["retire_archive"]:
                if (not archive.is_file() or archive.stat().st_size != asset["size_bytes"]
                        or digest(archive) != part["archive"]["sha256"]):
                    raise ValueError("Preserved existing archive is missing or changed")
                retained += asset["size_bytes"]
            if retained > max_extract_bytes:
                raise ValueError("Compact retained originals exceed the storage byte cap")
            _retire_archive(archive, part, request, originals)
            parts.append(_binding(part_path)); bindings.append(_binding(extraction_path))
            if progress:
                progress({"label": "원본 검증·중복 묶음 정리", "done": index + 1,
                          "total": len(resolution["assets"]), "unit": "archives"})
        receipt = {"kind": "compact_source_preservation_v1", "request": _binding(request_path),
            "source_resolution": _binding(resolution_path), "metadata": metadata, "parts": parts,
            "original_storage_bytes": retained, "reserved_download_bytes": resolution["total_bytes"],
            "retired_archive_bytes": sum(a["size_bytes"] for a, b in zip(resolution["assets"], parts)
                                          if read_json(b["path"])["retire_archive"]),
            "temporary_archive_peak_bytes": max((a["size_bytes"] for a in resolution["assets"]), default=0),
            "original_bytes_preserved": True, "audio_reencoded": False, "content_approved": False,
            "training_admitted": False, "Final_accessed": False,
            "created_at": previous["created_at"] if previous else now()}
        if previous is not None and previous != receipt:
            raise ValueError("Completed compact source evidence changed")
        if previous is None:
            _durable_json(receipt_path, receipt)
        return _result(directory, receipt, resumed=previous is not None)
