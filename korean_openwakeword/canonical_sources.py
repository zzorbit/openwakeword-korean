"""Commit canonical training audio before retiring newly acquired raw audio.

This transaction is used only by the new canonical cache namespace. Historical
caches, shared hardlinks and external files are never retired. Provider pins,
indexes and acquisition receipts remain available after raw audio retirement.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from .common import digest, now, read_json, writer_lock
from .compact_sources import (_binding, _directory, _durable_json, _file_identity,
                              _inside, _is_link, _sync_file)


_REQUEST = "canonical_request.json"
_ACQUISITION = "canonical_acquisition.json"
_PENDING = "canonical_pending.json"
_RECEIPT = "canonical_receipt.json"
_INTERNAL = {_REQUEST, _ACQUISITION, _PENDING, _RECEIPT, ".canonical_lock/.writer.lock"}
_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm"}


def _files(directory):
    """Snapshot lexical names and reject links instead of following them."""
    result = {}
    for base, directories, names in os.walk(directory, followlinks=False):
        for name in directories + names:
            path = Path(base) / name
            if _is_link(path):
                raise ValueError("Canonical cache files cannot traverse links")
        for name in names:
            path = Path(base) / name
            relative = path.relative_to(directory).as_posix()
            if relative not in _INTERNAL:
                result[relative] = path
    return result


def _record(path):
    path = Path(path).absolute()
    if not path.is_file() or _is_link(path):
        raise ValueError("Canonical preservation needs an ordinary file")
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def _managed(directory, path):
    path = Path(path).absolute()
    try:
        relative = path.relative_to(directory).as_posix()
    except ValueError:
        raise ValueError("Canonical retained files must stay inside their cache") from None
    return _inside(directory, relative)


def _check_record(record, *, directory=None, sync=False):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Canonical preservation file binding is missing")
    path = (_managed(directory, record["path"]) if directory is not None
            else Path(record["path"]).absolute())
    if (not path.is_file() or _is_link(path)
            or type(record.get("bytes")) is not int or record["bytes"] < 0
            or path.stat().st_size != record["bytes"]
            or digest(path) != record.get("sha256")):
        raise ValueError("Canonical retained file is missing or changed")
    if sync:
        _sync_file(path)
    return path


def _source_root(prepared):
    source = prepared.get("source", {})
    value = source.get("directory") or prepared.get("audio_directory")
    return Path(value).absolute() if value else None


def _claim_acquisition(directory, prepared, request, initial):
    """Freeze proof before conversion; no ownership is inferred on reuse."""
    root = _source_root(prepared)
    claims = []
    for relative, path in sorted(_files(directory).items()):
        record = _record(path)
        audio = root is not None and path.is_relative_to(root) and path.suffix.lower() in _AUDIO
        claims.append({**record, "relative_path": relative,
            "file_identity": _file_identity(path),
            "retire_eligible": bool(audio and relative not in initial
                                    and path.stat().st_nlink == 1)})
    body = {"kind": "canonical_acquisition_v1", "request": request,
            "prepared": prepared, "files": claims, "created_at": now()}
    _durable_json(directory / _ACQUISITION, body)
    return body


def _check_acquisition(directory, acquisition, request, *, allow_retired=False, verify_contents=True):
    if (acquisition.get("kind") != "canonical_acquisition_v1"
            or acquisition.get("request") != request
            or not isinstance(acquisition.get("prepared"), dict)
            or not isinstance(acquisition.get("files"), list)):
        raise ValueError("Canonical acquisition evidence changed")
    names = set()
    for claim in acquisition["files"]:
        path = _managed(directory, claim.get("path", ""))
        if (claim.get("relative_path") != path.relative_to(directory).as_posix()
                or claim["relative_path"] in names
                or type(claim.get("retire_eligible")) is not bool):
            raise ValueError("Canonical acquisition member coverage changed")
        names.add(claim["relative_path"])
        if allow_retired and claim["retire_eligible"] and not path.exists():
            continue
        if verify_contents:
            _check_record(claim, directory=directory)
        elif (not path.is_file() or type(claim.get("bytes")) is not int
              or claim["bytes"] < 0 or path.stat().st_size != claim["bytes"]):
            raise ValueError("Canonical acquired input is missing or changed")
    return acquisition


def _bound_records(directory, values):
    records = {}
    for value in values:
        if isinstance(value, (str, Path)):
            value = {"path": str(value)}
        path = _managed(directory, value["path"])
        actual = _record(path)
        if any(key in value and value[key] != actual[key] for key in ("sha256", "bytes")):
            raise ValueError("Canonical callback returned a changed file binding")
        existing = records.get(str(path))
        if existing and existing != actual:
            raise ValueError("Canonical callback repeats inconsistent bindings")
        records[str(path)] = actual
    return list(records.values())


def _conversion_records(directory, conversion, acquisition):
    if not isinstance(conversion.get("prepared"), dict):
        raise ValueError("Canonical conversion did not return prepared sources")
    mappings = conversion.get("converted", [])
    if not isinstance(mappings, list):
        raise ValueError("Canonical conversion mappings are missing")
    claims = {claim["path"]: claim for claim in acquisition["files"]}
    originals, canonical, checked = set(), set(), []
    for mapping in mappings:
        original, output = mapping.get("original", {}), mapping.get("canonical", {})
        original_path = _managed(directory, original.get("path", ""))
        output_path = _managed(directory, output.get("path", ""))
        if (str(original_path) in originals or original_path == output_path
                or str(original_path) not in claims):
            raise ValueError("Canonical conversion original is missing or repeated")
        original_actual = _record(original_path)
        output_actual = _record(output_path)
        if (original_actual != {k: original.get(k) for k in ("path", "sha256", "bytes")}
                or output_actual != {k: output.get(k) for k in ("path", "sha256", "bytes")}
                or any(original_actual[k] != claims[str(original_path)][k]
                       for k in ("path", "sha256", "bytes"))):
            raise ValueError("Canonical conversion changed its acquisition input")
        originals.add(str(original_path)); canonical.add(str(output_path))
        checked.append({"original": original_actual, "canonical": output_actual})
    # A retained canonical file can never also be an input scheduled for removal.
    if originals & canonical:
        raise ValueError("Canonical conversion replaces another retained input")
    bindings = _bound_records(directory, [*(conversion.get("bindings") or []),
        *[mapping["canonical"] for mapping in checked],
        *[claim for claim in acquisition["files"]
          if claim["path"] not in originals or not claim["retire_eligible"]]])
    return checked, bindings


def _verify_pending(directory, pending, request, acquisition, *, verify=None, sync=False):
    if (pending.get("kind") != "canonical_preservation_pending_v1"
            or pending.get("request") != request
            or pending.get("acquisition") != _binding(directory / _ACQUISITION)
            or not isinstance(pending.get("prepared"), dict)):
        raise ValueError("Canonical preservation commit changed")
    # Retained originals are content-checked through bindings below. Converted
    # newly owned originals are content-checked immediately before unlink.
    # Hashing the acquisition list again here would reread the entire raw corpus
    # at every commit/reuse while adding no independent preservation evidence.
    _check_acquisition(directory, acquisition, request, allow_retired=True, verify_contents=False)
    claims = {claim["path"]: claim for claim in acquisition["files"]}
    retained = {record["path"]: record for record in pending.get("bindings", [])}
    if len(retained) != len(pending.get("bindings", [])):
        raise ValueError("Canonical preservation repeats retained bindings")
    for record in retained.values():
        _check_record(record, directory=directory, sync=sync)
    originals, outputs = set(), set()
    for mapping in pending.get("converted", []):
        original, output = mapping["original"], mapping["canonical"]
        claim = claims.get(original["path"])
        if (not claim or original["path"] in originals
                or any(claim[k] != original[k] for k in ("path", "sha256", "bytes"))
                or retained.get(output["path"]) != output):
            raise ValueError("Canonical conversion preservation evidence changed")
        originals.add(original["path"]); outputs.add(output["path"])
    if originals & outputs:
        raise ValueError("Canonical retained output is marked for retirement")
    expected = {claim["path"] for claim in acquisition["files"]
                if claim["path"] not in originals or not claim["retire_eligible"]}
    if not expected.issubset(retained):
        raise ValueError("Canonical preservation omitted a retained source")
    if verify is not None:
        verify(pending["prepared"])
    return sum(record["bytes"] for record in retained.values())


def _retire_original(directory, mapping, claim):
    path = _managed(directory, mapping["original"]["path"])
    if not claim["retire_eligible"] or not path.exists():
        return
    if (path.stat().st_nlink != 1 or _is_link(path)
            or _file_identity(path) != claim["file_identity"]):
        raise ValueError("Owned raw audio became shared or replaced; preserved")
    _check_record(mapping["original"], directory=directory)
    # The complete canonical payload and metadata have a durable pending commit.
    # Only this exact newly acquired ordinary file is retired, never its folder.
    path.unlink()


def _result(directory, receipt, *, resumed):
    result = copy.deepcopy(receipt["prepared"])
    binding = _binding(directory / _RECEIPT)
    receipts = result.get("receipts", [])
    by_path = {record["path"]: record for record in receipts}
    # The transaction receipt binds the complete canonical payload manifest.
    # Repeating every waveform in outer source/material receipts would create
    # huge duplicate indexes and duplicate verification work for large corpora.
    by_path[binding["path"]] = binding
    artifacts = [*result.get("artifacts", []),
                 *(str(directory / name) for name in (_REQUEST, _ACQUISITION, _PENDING, _RECEIPT))]
    if result.get("canonical_index"):
        artifacts.append(result["canonical_index"]["path"])
    result.update({"canonical_receipt": binding, "receipts": list(by_path.values()),
        "artifacts": list(dict.fromkeys(artifacts)),
        "canonical_directory": str(directory), "storage_policy": "canonical_flac_only_v1",
        **{key: receipt[key] for key in ("original_storage_bytes", "canonical_storage_bytes",
             "retired_original_bytes", "acquired_original_peak_bytes")}, "resumed": resumed})
    return result


def _request(directory, request):
    expected = {"kind": "canonical_source_request_v1", "directory": str(directory),
                "request": request}
    path = directory / _REQUEST
    if path.exists():
        saved = read_json(path)
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Canonical cache belongs to another request")
    else:
        saved = {**expected, "preexisting_files": sorted(_files(directory)), "created_at": now()}
        _durable_json(path, saved)
    if not isinstance(saved.get("preexisting_files"), list):
        raise ValueError("Canonical initial ownership evidence changed")
    return saved, _binding(path)


def verify_canonical_source(directory, *, verify=None):
    """Verify a completed canonical cache locally, without original audio/network."""
    directory = _directory(directory)
    request_path = directory / _REQUEST
    request = read_json(request_path)
    if (request.get("kind") != "canonical_source_request_v1"
            or request.get("directory") != str(directory)):
        raise ValueError("Canonical request directory changed")
    binding = _binding(request_path)
    acquisition = read_json(directory / _ACQUISITION)
    if (acquisition.get("kind") != "canonical_acquisition_v1"
            or acquisition.get("request") != binding):
        raise ValueError("Canonical acquisition request binding changed")
    receipt = read_json(directory / _RECEIPT)
    pending = read_json(directory / _PENDING)
    if (receipt.get("kind") != "canonical_source_preservation_v1"
            or receipt.get("pending") != _binding(directory / _PENDING)
            or any(receipt.get(key) != pending.get(key)
                   for key in ("request", "acquisition", "prepared", "bindings", "converted"))):
        raise ValueError("Canonical completion no longer binds its preservation commit")
    retained = _verify_pending(directory, pending, binding, acquisition, verify=verify)
    claims = {claim["path"]: claim for claim in acquisition["files"]}
    retired = 0
    for mapping in pending["converted"]:
        claim = claims[mapping["original"]["path"]]
        if claim["retire_eligible"]:
            if _managed(directory, claim["path"]).exists():
                raise ValueError("Retired canonical input reappeared")
            retired += claim["bytes"]
    canonical = {mapping["canonical"]["path"]: mapping["canonical"]["bytes"]
                 for mapping in pending["converted"]}
    if (receipt.get("original_storage_bytes") != retained
            or receipt.get("canonical_storage_bytes") != sum(canonical.values())
            or receipt.get("retired_original_bytes") != retired
            or receipt.get("acquired_original_peak_bytes") != sum(c["bytes"] for c in acquisition["files"])):
        raise ValueError("Canonical storage accounting changed")
    return _result(directory, receipt, resumed=True)


def stage_canonical_source(directory, *, request, acquire):
    """Durably claim one acquisition before a separate conversion consumer.

    This small transaction boundary permits a bounded producer to acquire the
    next independent source while the consumer canonicalizes the current one.
    It uses the same ownership rules as ``prepare_canonical_source``. A file
    already present when staging starts is never newly claimed for retirement.
    Completed sources are verified locally without requesting retired inputs.
    """
    directory = _directory(directory); directory.mkdir(parents=True, exist_ok=True)
    directory = _directory(directory)
    with writer_lock(directory / ".canonical_lock"):
        initial, request_binding = _request(directory, request)
        if (directory / _RECEIPT).exists():
            return verify_canonical_source(directory)
        acquisition_path = directory / _ACQUISITION
        if acquisition_path.exists():
            acquisition = read_json(acquisition_path)
            if not (directory / _PENDING).exists():
                _check_acquisition(directory, acquisition, request_binding)
            else:
                # A crash can occur after durable preservation and during raw
                # retirement. The consumer validates and completes that commit.
                _verify_pending(directory, read_json(directory / _PENDING),
                                request_binding, acquisition)
        else:
            before_acquire = set(_files(directory))
            prepared = acquire()
            if not isinstance(prepared, dict):
                raise ValueError("Canonical acquisition did not return prepared sources")
            acquisition = _claim_acquisition(directory, prepared, request_binding,
                set(initial["preexisting_files"]) | before_acquire)
        return copy.deepcopy(acquisition["prepared"])


def prepare_canonical_source(directory, *, request, acquire, canonicalize,
                             verify=None, max_retained_bytes=None, progress=None):
    """Acquire/index, canonicalize, commit, and retire only new private raw audio.

    ``acquire()`` returns a normal source-backend result. ``canonicalize`` takes
    that result and an output directory, returning ``prepared``, original/output
    ``converted`` bindings and optional retained ``bindings``. It must prove the
    canonical decoded samples match the existing training input before returning.
    Invalid inputs are omitted from ``converted`` and remain available to QC.
    A verified pending commit can resume retirement after a crash without another
    download or decode. The optional verifier receives the prepared result.
    """
    if max_retained_bytes is not None and (type(max_retained_bytes) is not int or max_retained_bytes <= 0):
        raise ValueError("Canonical retained-byte bound must be a positive integer")
    directory = _directory(directory); directory.mkdir(parents=True, exist_ok=True)
    directory = _directory(directory)
    with writer_lock(directory / ".canonical_lock"):
        initial, request_binding = _request(directory, request)
        if (directory / _RECEIPT).exists():
            result = verify_canonical_source(directory, verify=verify)
            if max_retained_bytes is not None and result["original_storage_bytes"] > max_retained_bytes:
                raise ValueError("Canonical retained sources exceed current storage bound")
            return result
        acquisition_path = directory / _ACQUISITION
        pending_path = directory / _PENDING
        if acquisition_path.exists():
            acquisition = read_json(acquisition_path)
            # Pending verification below permits only its explicitly mapped
            # retired originals to be absent after an interrupted retirement.
            if not pending_path.exists():
                _check_acquisition(directory, acquisition, request_binding)
        else:
            # A previous acquisition may have stopped before it could commit
            # ownership evidence. Any file already present on this invocation
            # is conservatively preserved, even in this new cache namespace.
            before_acquire = set(_files(directory))
            prepared = acquire()
            if not isinstance(prepared, dict):
                raise ValueError("Canonical acquisition did not return prepared sources")
            acquisition = _claim_acquisition(directory, prepared, request_binding,
                set(initial["preexisting_files"]) | before_acquire)
        if pending_path.exists():
            pending = read_json(pending_path)
        else:
            conversion = canonicalize(copy.deepcopy(acquisition["prepared"]), directory / "canonical")
            mappings, bindings = _conversion_records(directory, conversion, acquisition)
            pending = {"kind": "canonical_preservation_pending_v1", "request": request_binding,
                "acquisition": _binding(acquisition_path), "prepared": conversion["prepared"],
                "converted": mappings, "bindings": bindings, "created_at": now()}
            # All bytes are flushed and checked before the durable commit permits
            # retirement; current capacity limits cannot silently discard inputs.
            retained = _verify_pending(directory, pending, request_binding, acquisition,
                                       verify=verify, sync=True)
            if max_retained_bytes is not None and retained > max_retained_bytes:
                raise ValueError("Canonical retained sources exceed current storage bound")
            _durable_json(pending_path, pending)
        retained = _verify_pending(directory, pending, request_binding, acquisition, verify=verify)
        if max_retained_bytes is not None and retained > max_retained_bytes:
            raise ValueError("Canonical retained sources exceed current storage bound")
        claims = {claim["path"]: claim for claim in acquisition["files"]}
        for mapping in pending["converted"]:
            _retire_original(directory, mapping, claims[mapping["original"]["path"]])
        canonical = {mapping["canonical"]["path"]: mapping["canonical"]["bytes"]
                     for mapping in pending["converted"]}
        receipt = {**pending, "kind": "canonical_source_preservation_v1",
            "pending": _binding(pending_path), "original_storage_bytes": retained,
            "canonical_storage_bytes": sum(canonical.values()),
            "retired_original_bytes": sum(mapping["original"]["bytes"] for mapping in pending["converted"]
                if claims[mapping["original"]["path"]]["retire_eligible"]),
            "acquired_original_peak_bytes": sum(c["bytes"] for c in acquisition["files"]),
            "completed_at": now()}
        _durable_json(directory / _RECEIPT, receipt)
        result = verify_canonical_source(directory, verify=verify)
        result["resumed"] = False
        if progress:
            progress({"stage": "canonical_sources_committed", "retained_bytes": retained,
                      "retired_original_bytes": receipt["retired_original_bytes"]})
        return result
