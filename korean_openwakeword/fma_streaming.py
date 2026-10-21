"""Bounded FMA acquisition with the unchanged CPU canonical audio conversion.

The standalone original-MP3 downloader is intentionally unchanged. This policy
uses its frozen selection plan in a separate namespace. Only one downloader
owns the range archive; the consumer verifies/canonicalizes the current song
while at most one next song is acquired. Each song has its own durable
canonical transaction, so a restart never needs a retired MP3 again.
"""
from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import re
import shutil
import threading

import soundfile as sf

from .canonical_audio import canonicalize_prepared_source
from .canonical_sources import (prepare_canonical_source, stage_canonical_source,
                                verify_canonical_source)
from .common import digest, now, read_json, read_rows, writer_lock
from .compact_sources import _binding, _directory, _durable_json, _inside
from .fma_download import (_FMA_FILES, _FULL_URL, _publish_rows, _remaining_bytes, _same_or_write,
                           download_fma)

POLICY = "per_track_flac_prefetch_v1"
_RECEIPT = "streaming_receipt.json"
_REQUEST = "streaming_request.json"


def _checked_binding(directory, record):
    path = Path(record["path"]).absolute()
    if not path.is_relative_to(directory):
        raise ValueError("FMA streaming receipt escapes its cache")
    path = _inside(directory, path.relative_to(directory).as_posix())
    if not path.is_file() or digest(path) != record.get("sha256"):
        raise ValueError("FMA streaming retained file is missing or changed")
    return path


def _track_request(plan, row):
    return {"policy": POLICY, "plan": plan, "track": row}


def _metadata_records(metadata):
    """Read exact provider row strings once, rather than once per song."""
    with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        upper, lower, first = next(reader), next(reader), next(reader)
        columns = list(zip(upper, lower))
        if len(upper) != len(lower):
            raise ValueError("FMA tracks.csv header widths differ")
        def position(key, required=False):
            if columns.count(key) > 1 or required and key not in columns:
                raise ValueError(f"FMA metadata needs one {key} column")
            return columns.index(key) if key in columns else None
        license_column = position(("track", "license"), True)
        artist_column = position(("artist", "id"), True)
        partition_column = position(("set", "split"))
        result = {}
        from itertools import chain
        for row in chain([] if first[0] == "track_id" else [first], reader):
            if not row or len(row) != len(columns) or not row[0].isdigit() or int(row[0]) in result:
                raise ValueError("FMA provider track metadata changed")
            result[int(row[0])] = {"license": row[license_column],
                "artist_id": row[artist_column].strip(),
                "partition": row[partition_column] if partition_column is not None else "unknown"}
    return result


def _provider_row(row, target, audio, metadata, provider):
    # This is the FMA provider adapter's row representation, including its
    # original recording/artist IDs. No transcript or vocal language is inferred.
    track = row["track_id"]
    return {"path": target.relative_to(audio).as_posix(), "sha256": digest(target),
        "track_id": track, "canonical_source_id": f"fma:track:{track}",
        "canonical_recording_id": f"fma:track:{track}", "dataset_id": "fma",
        "provider_metadata_bindings": [metadata], "license": provider["license"],
        "source_url": "https://github.com/mdeff/fma", "category": "music",
        "speech": None, "speech_presence": "unknown", "language": "unknown",
        "canonical_artist_id": "fma:artist:" + provider["artist_id"] if provider["artist_id"] else None,
        "original_dataset_partition": provider["partition"],
        "category_basis": "FMA original music corpus and per-track metadata; vocal presence/language not inferred"}


def _aggregate_rows(directory, parts):
    rows = []
    for part in parts:
        root = Path(part["source"]["directory"])
        for supplied in part["rows"]:
            row = copy.deepcopy(supplied)
            path = (root / row["path"]).absolute()
            if not path.is_relative_to(directory):
                raise ValueError("FMA canonical row escapes its cache")
            row["path"] = path.relative_to(directory).as_posix()
            rows.append(row)
    # Legacy whole-batch FMA indexing sorts the ORIGINAL archive paths. Keep
    # that order, not playlist order or order of completion, for later sampling.
    return sorted(rows, key=lambda row: int(row["track_id"]))


def _metadata_bytes(directory):
    selection = directory / "selection"
    return sum(path.stat().st_size for path in selection.rglob("*") if path.is_file()) + sum(
        (directory / name).stat().st_size for name in (_REQUEST, "excluded_tracks.json", "streaming_index.jsonl", "streaming_peak.json")
        if (directory / name).is_file())


def _result(directory, receipt, resumed):
    prepared = copy.deepcopy(receipt["prepared"])
    binding = _binding(directory / _RECEIPT)
    prepared.update(canonical_receipt=binding, canonical_directory=str(directory),
        fma_acquisition_policy=POLICY, storage_policy="canonical_flac_only_v1",
        resumed=resumed, downloaded_bytes_this_run=0,
        download_byte_scope="Transfer bytes in the completing invocation, matching the original selective downloader; not a lifetime retry-traffic cap",
        raw_mp3_peak_scope="High-water bytes of source MP3 paths, including preserved QC rejects and staged songs; compressed partials and conversion workspace are separate",
        **receipt["accounting"])
    prepared["receipts"] = [*prepared.get("receipts", []), binding]
    prepared["artifacts"] = list(dict.fromkeys([*prepared.get("artifacts", []),
        str(directory / _REQUEST), str(directory / _RECEIPT)]))
    return prepared


def verify_fma_streaming_source(directory):
    """Validate completed per-song preservation locally, without any network."""
    directory = _directory(directory)
    receipt = read_json(directory / _RECEIPT)
    request = read_json(directory / _REQUEST)
    if (receipt.get("kind") != "fma_streaming_source_v1" or request.get("policy") != POLICY
            or receipt.get("request") != _binding(directory / _REQUEST)):
        raise ValueError("FMA streaming request or completion changed")
    plan = read_json(_checked_binding(directory, receipt["plan"]))
    _checked_binding(directory, plan["metadata"])
    selected = plan["selection"]["selected"] + plan["selection"].get("reserve", [])
    records, parts, seconds = receipt["tracks"], [], 0.0
    if not records or len(records) > len(selected):
        raise ValueError("FMA streaming playlist coverage changed")
    for index, record in enumerate(records):
        row = selected[index]
        if record["track_id"] != row["track_id"] or seconds + 1e-6 >= request["target_hours"] * 3600:
            raise ValueError("FMA streaming actual-duration selection changed")
        track_directory = directory / "tracks" / f"{row['track_id']:06d}"
        if read_json(track_directory / "canonical_request.json")["request"] != _track_request(receipt["plan"], row):
            raise ValueError("FMA per-song transaction request changed")
        _checked_binding(directory, record["receipt"])
        if record["receipt"] != _binding(track_directory / "canonical_receipt.json"):
            raise ValueError("FMA per-song completion binding changed")
        part = verify_canonical_source(track_directory)
        if part["fma_track"]["track_id"] != row["track_id"]:
            raise ValueError("FMA per-song source identity changed")
        seconds += part["fma_track"]["audio_header_duration_s"]
        parts.append(part)
    if seconds + 1e-6 < request["target_hours"] * 3600:
        raise ValueError("FMA streaming actual header duration is below target")
    rows = _aggregate_rows(directory, parts)
    index_path = _checked_binding(directory, receipt["index"])
    if rows != read_rows(index_path) or rows != receipt["prepared"]["rows"]:
        raise ValueError("FMA streaming canonical index changed")
    expected = {"canonical_storage_bytes": sum(p["canonical_storage_bytes"] for p in parts),
        "retired_original_bytes": sum(p["retired_original_bytes"] for p in parts),
        "original_storage_bytes": sum(p["original_storage_bytes"] for p in parts) + _metadata_bytes(directory)}
    if any(receipt["accounting"].get(key) != value for key, value in expected.items()):
        raise ValueError("FMA streaming storage accounting changed")
    if receipt.get("actual_header_hours") != seconds / 3600:
        raise ValueError("FMA streaming header duration changed")
    peak = read_json(_checked_binding(directory, receipt["mp3_peak"]))
    if peak.get("raw_mp3_peak_bytes") != receipt["accounting"].get("acquired_original_peak_bytes"):
        raise ValueError("FMA streaming raw MP3 peak evidence changed")
    return _result(directory, receipt, True)


def prepare_fma_streaming_source(directory, *, target_hours, max_download_bytes,
        max_extract_bytes, seed=20261021, excluded_track_ids=(), progress=None,
        archive_factory=None):
    """Acquire the fixed playlist with one-song prefetch and bounded raw storage.

    Budgets are hard bounds, not permission to change selection or decode. A
    complete cache is verified without opening an archive. Existing MP3s and
    shared hardlinks are preserved by the canonical transaction ownership proof.
    """
    if type(target_hours) not in (int, float) or not math.isfinite(target_hours) or target_hours <= 0:
        raise ValueError("FMA streaming hours must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("FMA streaming seed must be a nonnegative integer")
    if any(type(value) is not int or value <= 0 for value in (max_download_bytes, max_extract_bytes)):
        raise ValueError("FMA streaming requires positive byte bounds")
    if any(type(track) is not int or track <= 0 for track in excluded_track_ids):
        raise ValueError("FMA excluded track IDs must be positive integers")
    directory = _directory(directory); directory.mkdir(parents=True, exist_ok=True)
    request = {"kind": "fma_streaming_request_v1", "policy": POLICY,
        "target_hours": float(target_hours), "seed": seed,
        "excluded_track_ids": sorted(set(excluded_track_ids)), "prefetch_songs": 1}
    with writer_lock(directory / ".streaming_lock"):
        if not (directory / _REQUEST).exists() and any(path.name != ".streaming_lock" for path in directory.iterdir()):
            raise ValueError("FMA streaming requires a new policy namespace")
        _same_or_write(directory / _REQUEST, request)
        if (directory / _RECEIPT).exists():
            result = verify_fma_streaming_source(directory)
            if result["original_storage_bytes"] > max_extract_bytes:
                raise ValueError("FMA streaming retained sources exceed current storage bound")
            return result
        exclusions = directory / "excluded_tracks.json"
        _same_or_write(exclusions, {"track_ids": request["excluded_track_ids"]})
        preview = download_fma(directory / "selection", hours=target_hours, seed=seed,
            max_download_bytes=max_download_bytes, exclusions=exclusions, plan_only=True,
            archive_factory=archive_factory, progress=progress)
        plan_binding = preview["plan"]
        plan = read_json(plan_binding["path"])
        rows = plan["selection"]["selected"] + plan["selection"].get("reserve", [])
        metadata = _binding(preview["metadata_csv"])
        provider_records = _metadata_records(metadata["path"])
        if _metadata_bytes(directory) >= max_extract_bytes:
            raise ValueError("FMA streaming metadata exceeds the storage cap")
        preview_bytes = preview["metadata_transfer_bytes_this_run"]
        if preview_bytes >= max_download_bytes:
            raise ValueError("FMA streaming metadata exhausted the transfer budget")
        from .range_zip import RangeZipArchive
        archive = (archive_factory or RangeZipArchive)(_FULL_URL,
            expected_size=_FMA_FILES["fma_full.zip"][0],
            max_download_bytes=max_download_bytes - preview_bytes,
            expected_etag=plan["sources"]["audio"]["etag"])
        try:
            if archive.binding() != plan["sources"]["audio"]:
                raise ValueError("FMA streaming pinned audio archive changed")
            remaining = sum(_remaining_bytes([row], directory / "tracks" / f"{row['track_id']:06d}" / "audio")
                for row in plan["selection"]["selected"]
                if not (directory / "tracks" / f"{row['track_id']:06d}" / "canonical_receipt.json").exists())
            if archive.downloaded_bytes + remaining > max_download_bytes - preview_bytes:
                raise ValueError("Selected songs exceed remaining byte budget; no new songs requested")
            largest = max(row["original_bytes"] + row["compressed_bytes"] for row in rows)
            if shutil.disk_usage(directory).free < 3 * largest + 1024**2:
                raise ValueError("Insufficient space for bounded FMA originals and conversion")
            callback_lock = threading.Lock()
            def emit(record):
                if progress:
                    with callback_lock:
                        progress(record)
            raw_lock = threading.Lock()
            raw_ready = {int(path.stem): path.stat().st_size
                for path in (directory / "tracks").glob("*/audio/fma_full/*/*.mp3")}
            peak_path = directory / "streaming_peak.json"
            peak_request = _binding(directory / _REQUEST)
            prior_peak = read_json(peak_path) if peak_path.exists() else {}
            if prior_peak and (prior_peak.get("request") != peak_request
                    or type(prior_peak.get("raw_mp3_peak_bytes")) is not int
                    or prior_peak["raw_mp3_peak_bytes"] < 0):
                raise ValueError("FMA streaming raw peak journal changed")
            raw_peak = max(prior_peak.get("raw_mp3_peak_bytes", 0), sum(raw_ready.values()))
            def save_peak():
                _durable_json(peak_path, {"kind": "fma_streaming_mp3_peak_v1",
                    "request": peak_request, "raw_mp3_peak_bytes": raw_peak,
                    "scope": "High-water bytes of source MP3 paths simultaneously present; compressed partials and conversion workspace are separate"})
            save_peak()
            def stage(row):
                nonlocal raw_peak
                track_directory = directory / "tracks" / f"{row['track_id']:06d}"
                def acquire():
                    member = row["archive_member"]
                    if not re.fullmatch(r"fma_full/\d{3}/\d{6}\.mp3", member):
                        raise ValueError("FMA streaming member path changed")
                    audio = track_directory / "audio"
                    target = _inside(audio, member)
                    transfer = archive.extract(member, target)
                    header = sf.info(target)
                    if header.frames <= 0 or header.samplerate <= 0 or header.channels <= 0:
                        raise ValueError("FMA source has an invalid audio header")
                    track = {"track_id": row["track_id"], "archive_member": member,
                        "path": str(target), "sha256": transfer["sha256"],
                        "bytes": target.stat().st_size, "crc32": row["crc32"],
                        "provider_duration_s": row["provider_duration_s"],
                        "audio_header_duration_s": header.frames / header.samplerate,
                        "license": row["license"], "artist_id": row["artist_id"],
                        "artist_name": row["artist_name"], "original_source_url": row["original_source_url"],
                        "plan": plan_binding, "source": archive.binding(),
                        "audio_reencoded": False, "content_approved": False, "training_admitted": False,
                        "verification_scope": "Original ZIP entry CRC32, local SHA256, size and audio header; not whole ZIP SHA1 or human review"}
                    receipt = track_directory / "song_receipt.json"
                    _same_or_write(receipt, track)
                    index = track_directory / "provider_index.jsonl"
                    provider_row = _provider_row(row, target, audio, metadata, provider_records[row["track_id"]])
                    _publish_rows(index, [provider_row])
                    return {"source": {"directory": str(audio), "adapter": "fma",
                        "metadata": metadata["path"], "source_url": "https://github.com/mdeff/fma"},
                        "rows": [provider_row], "fma_track": track, "receipts": [_binding(receipt)],
                        "reserved_download_bytes": 0, "original_storage_bytes": track["bytes"],
                        "artifacts": [str(index), str(receipt)]}
                part = stage_canonical_source(track_directory,
                    request=_track_request(plan_binding, row), acquire=acquire)
                raw = track_directory / "audio" / row["archive_member"]
                with raw_lock:
                    if raw.exists():
                        raw_ready[row["track_id"]] = raw.stat().st_size
                    measured = sum(raw_ready.values())
                    if measured > raw_peak:
                        raw_peak = measured
                        save_peak()
                return part
            parts, track_records, seconds, retained = [], [], 0.0, _metadata_bytes(directory)
            # Executor shutdown joins an in-flight download even when conversion
            # raises. Its acquisition stays durable and is reused on restart.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fma-range") as executor:
                future = executor.submit(stage, rows[0])
                for position, row in enumerate(rows):
                    acquired = future.result()
                    seconds += acquired["fma_track"]["audio_header_duration_s"]
                    more = seconds + 1e-6 < target_hours * 3600
                    next_row = rows[position + 1] if more and position + 1 < len(rows) else None
                    # Reserve room for the exact current decode plus one next
                    # original. Refuse capacity changes; never drop a song.
                    current_pcm = math.ceil(acquired["fma_track"]["audio_header_duration_s"] * 16000) * 4
                    bounded_work = acquired["fma_track"]["bytes"] + current_pcm + 1024**2
                    if next_row:
                        bounded_work += next_row["original_bytes"] + next_row["compressed_bytes"]
                    if retained + bounded_work > max_extract_bytes:
                        raise ValueError("FMA streaming bounded workspace exceeds the storage cap")
                    if shutil.disk_usage(directory).free < bounded_work:
                        raise ValueError("Insufficient space for the next bounded FMA conversion")
                    future = executor.submit(stage, next_row) if next_row else None
                    track_directory = directory / "tracks" / f"{row['track_id']:06d}"
                    def no_acquire():
                        raise ValueError("FMA staged acquisition is missing")
                    part = prepare_canonical_source(track_directory,
                        request=_track_request(plan_binding, row), acquire=no_acquire,
                        canonicalize=canonicalize_prepared_source,
                        max_retained_bytes=max_extract_bytes - retained)
                    parts.append(part); retained += part["original_storage_bytes"]
                    track_records.append({"track_id": row["track_id"], "receipt": part["canonical_receipt"]})
                    with raw_lock:
                        raw = track_directory / "audio" / row["archive_member"]
                        if not raw.exists():
                            raw_ready.pop(row["track_id"], None)
                    emit({"stage": "fma_streaming_song_committed", "done": len(parts),
                        "total": len(rows), "actual_header_hours": seconds / 3600,
                        "target_hours": target_hours, "downloaded_bytes_this_run": preview_bytes + archive.downloaded_bytes,
                        "raw_mp3_peak_bytes": raw_peak})
                    if not more:
                        break
                    if next_row is None:
                        raise ValueError("FMA fixed reserve exhausted below actual header target")
            final_rows = _aggregate_rows(directory, parts)
            index = directory / "streaming_index.jsonl"
            _publish_rows(index, final_rows)
            accounting = {"original_storage_bytes": sum(p["original_storage_bytes"] for p in parts) + _metadata_bytes(directory),
                "canonical_storage_bytes": sum(p["canonical_storage_bytes"] for p in parts),
                "retired_original_bytes": sum(p["retired_original_bytes"] for p in parts),
                "acquired_original_peak_bytes": raw_peak,
                "reserved_download_bytes": preview_bytes + archive.downloaded_bytes}
            if accounting["original_storage_bytes"] > max_extract_bytes:
                raise ValueError("FMA streaming retained sources exceed the storage cap")
            prepared = {"source": {"directory": str(directory), "adapter": "manifest",
                "metadata": str(index), "source_url": "https://github.com/mdeff/fma",
                "canonical_storage_policy": "canonical_flac_only_v1"}, "rows": final_rows,
                "receipts": [plan_binding, *[record["receipt"] for record in track_records]],
                "artifacts": [str(index), *[artifact for p in parts for artifact in p["artifacts"]]],
                "canonical_index": _binding(index),
                "canonical_audio_bytes": accounting["canonical_storage_bytes"],
                "canonical_pcm16_bytes": sum(p["canonical_pcm16_bytes"] for p in parts)}
            receipt = {"kind": "fma_streaming_source_v1", "request": _binding(directory / _REQUEST),
                "plan": plan_binding, "index": _binding(index), "tracks": track_records,
                "mp3_peak": _binding(peak_path),
                "prepared": prepared, "accounting": accounting,
                "actual_header_hours": seconds / 3600, "prefetch_songs": 1,
                "whole_archive_sha1_verified": False, "content_approved": False,
                "training_admitted": False, "completed_at": now()}
            _durable_json(directory / _RECEIPT, receipt)
            result = verify_fma_streaming_source(directory)
            result["resumed"] = False
            result["downloaded_bytes_this_run"] = preview_bytes + archive.downloaded_bytes
            return result
        finally:
            archive.close()
