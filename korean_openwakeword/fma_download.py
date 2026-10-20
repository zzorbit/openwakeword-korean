"""Select full-length original FMA tracks without fetching the whole ZIP.

The frozen selection is based on publisher durations. Downloaded hours use
actual audio headers. Neither transfer checks nor source metadata grant content
approval, and partial transfer cannot verify the upstream whole-archive SHA1.
"""
from contextlib import contextmanager, ExitStack
from importlib.resources import files
import math
import os
from pathlib import Path
import re
import shutil

import soundfile as sf

from .commercial_sources import _FMA_FILES
from .common import digest, now, read_json, write_json

_FULL_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
_CSV_MEMBER = "fma_metadata/tracks.csv"
_METADATA_BUDGET = 64 * 1024**2


@contextmanager
def _lease(directory):
    """OS-released ownership also permits resume after a killed process."""
    lock = (directory / "download.lock").open("a+b")
    if lock.seek(0, 2) == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock.close()
        raise RuntimeError("A selective FMA download already owns this output directory") from error
    try:
        yield
    finally:
        lock.seek(0)
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def _protected_sources():
    """Conservative declared aliases only; no claim of complete independence."""
    catalog = read_json(files("korean_openwakeword") / "resources/evaluation_materials_v1.json")
    tracks, artists, urls = set(), set(), set()
    def visit(value):
        if isinstance(value, dict):
            source = str(value.get("canonical_source_id", ""))
            match = re.fullmatch(r"fma:track:(\d+)", source)
            if match:
                tracks.add(int(match[1]))
            url = value.get("original_source_url", "")
            if isinstance(url, str) and "freemusicarchive.org/" in url:
                urls.add(url)
                artist = value.get("canonical_artist_id", "")
                if isinstance(artist, str) and artist.startswith("musan:artist:"):
                    artists.add(artist.split(":", 2)[2])
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(catalog)
    return {"track_ids": sorted(tracks), "artist_ids": [], "artist_names": sorted(artists),
            "url_prefixes": sorted(urls), "catalog_sha256": digest(Path(str(files("korean_openwakeword") / "resources/evaluation_materials_v1.json"))),
            "scope": "Known FMA aliases in common evaluation only; unresolved aliases and near-duplicates remain unverified"}


def _exclusions(path):
    result = _protected_sources()
    if path is None:
        return result
    supplied = read_json(path)
    if not isinstance(supplied, dict) or set(supplied) - {"track_ids", "artist_ids", "artist_names", "url_prefixes"}:
        raise ValueError("Exclusions need track_ids, artist_ids, artist_names and/or url_prefixes")
    for key in ("track_ids", "artist_ids", "artist_names", "url_prefixes"):
        values = supplied.get(key, [])
        if not isinstance(values, list):
            raise ValueError("Exclusion values must be lists")
        if key in ("track_ids", "artist_ids"):
            if any(isinstance(v, bool) or not str(v).isdigit() for v in values):
                raise ValueError("Excluded track and artist IDs must be nonnegative integers")
            values = [int(v) for v in values]
        elif any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("Excluded names/URLs must be nonempty strings")
        result[key] = sorted(set(result[key]) | set(values))
    result["additional_file_sha256"] = digest(path)
    return result


def _same_or_write(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Frozen selective download settings or source changed; choose a new output directory")
    else:
        write_json(path, value)


def _publish_rows(path, rows):
    import json
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                            prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        if path.exists():
            if digest(path) != digest(temporary):
                raise ValueError("Completed selective track manifest changed")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remaining_bytes(rows, directory):
    needed = 0
    for row in rows:
        target = directory / row["archive_member"]
        if target.is_file() and target.stat().st_size == row["original_bytes"]:
            continue  # The transport verifies its SHA/CRC before reuse.
        partial = target.with_name(target.name + ".compressed.part")
        checkpoint = target.with_name(target.name + ".range-part.json")
        have = 0
        if partial.is_file() and checkpoint.is_file():
            state = read_json(checkpoint)
            request = state.get("request", {})
            committed = state.get("compressed_bytes")
            if (state.get("kind") == "range_zip_partial_v1"
                    and request.get("member") == row["archive_member"]
                    and request.get("compressed_bytes") == row["compressed_bytes"]
                    and request.get("crc32") == row["crc32"]
                    and type(committed) is int
                    and 0 <= committed <= min(partial.stat().st_size, row["compressed_bytes"])):
                have = committed  # Uncommitted tail bytes are discarded by the transport.
        needed += max(0, row["compressed_bytes"] - min(have, row["compressed_bytes"])) + 65536
    return needed


def download_fma(output, *, hours=1000, seed=20261021, max_download_bytes=None,
                 plan_only=False, license_policy="allow-derivatives", exclusions=None,
                 progress=None, archive_factory=None):
    """Freeze a playlist, transfer exact MP3s, then verify/resume each track.

    plan_only fetches ZIP directories and tracks.csv (not original songs).
    A 5% provider-duration reserve is frozen but only used if actual durations
    leave a shortfall. Stop at the first complete track meeting actual hours.
    """
    from .fma_selection import select_fma_tracks
    from .range_zip import RangeZipArchive
    if (isinstance(hours, bool) or not isinstance(hours, (float, int)) or not math.isfinite(hours)
            or hours <= 0 or type(seed) is not int or seed < 0 or type(plan_only) is not bool):
        raise ValueError("Hours must be positive and finite; seed must be nonnegative")
    if license_policy not in ("allow-derivatives", "commercial-only"):
        raise ValueError("Choose allow-derivatives or commercial-only")
    if max_download_bytes is not None and (type(max_download_bytes) is not int or max_download_bytes <= 0):
        raise ValueError("A positive integer maximum download byte budget is required")
    if not plan_only and max_download_bytes is None:
        raise ValueError("Actual song downloads require an explicit maximum byte budget")
    output = Path(output).expanduser().resolve()
    if any(p.lower() in (".codex", ".git") for p in output.parts):
        raise ValueError("Select a data folder outside Codex/Git state")
    output.mkdir(parents=True, exist_ok=True)
    protected = _exclusions(exclusions)
    request = {"kind": "fma_selective_request_v1", "hours": float(hours), "seed": seed,
        "license_policy": license_policy, "exclusions": protected,
        "full_archive_url": _FULL_URL, "full_archive_size": _FMA_FILES["fma_full.zip"][0],
        "metadata_archive_url": _META_URL, "metadata_archive_size": _FMA_FILES["fma_metadata.zip"][0]}
    factory = archive_factory or RangeZipArchive
    with _lease(output), ExitStack() as resources:
        request_path = output / "request.json"
        if not request_path.exists() and any(p.name != "download.lock" for p in output.iterdir()):
            raise ValueError("Use a dedicated empty output folder or resume its existing request")
        _same_or_write(request_path, request)
        prior = read_json(output / "selection_plan.json") if (output / "selection_plan.json").is_file() else None
        meta_budget = min(max_download_bytes or _METADATA_BUDGET, _METADATA_BUDGET)
        metadata = factory(_META_URL, _FMA_FILES["fma_metadata.zip"][0], max_download_bytes=meta_budget,
            expected_etag=prior["sources"]["metadata"]["etag"] if prior else None)
        resources.callback(metadata.close)
        csv_path = output / "metadata" / "tracks.csv"
        csv_path.parent.mkdir(exist_ok=True)
        metadata.extract(_CSV_MEMBER, csv_path)
        full_budget = (max_download_bytes or _METADATA_BUDGET) - metadata.downloaded_bytes
        if full_budget <= 0:
            raise ValueError("Download byte budget exhausted by source metadata")
        archive = factory(_FULL_URL, _FMA_FILES["fma_full.zip"][0], max_download_bytes=full_budget,
            expected_etag=prior["sources"]["audio"]["etag"] if prior else None)
        resources.callback(archive.close)
        selected = select_fma_tracks(csv_path, archive.entries, target_hours=hours, seed=seed,
            license_policy=license_policy, excluded_track_ids=protected["track_ids"],
            excluded_artist_ids=protected["artist_ids"], excluded_artist_names=protected["artist_names"],
            original_url_prefixes=protected["url_prefixes"])
        if not selected["target_met"]:
            write_json(output / "shortfall.json", selected)
            raise ValueError("Eligible FMA originals do not meet the selected hour target; no song download started")
        plan = {"kind": "fma_selective_plan_v1", "request": request,
            "sources": {"metadata": metadata.binding(), "audio": archive.binding()},
            "metadata": _binding(csv_path), "selection": selected,
            "whole_archive_sha1_verified": False,
            "whole_archive_sha1_reference": _FMA_FILES["fma_full.zip"][1],
            "content_approved": False, "training_admitted": False}
        _same_or_write(output / "selection_plan.json", plan)
        plan_binding = _binding(output / "selection_plan.json")
        selected_rows = selected["selected"]
        reserve_rows = selected.get("reserve", [])
        main_bytes = sum(row["compressed_bytes"] for row in selected_rows)
        reserve_bytes = sum(row["compressed_bytes"] for row in reserve_rows)
        summary = {"mode": "selective_original_mp3", "plan": plan_binding,
            "selected_tracks": len(selected_rows), "reserve_tracks": len(reserve_rows),
            "selected_provider_hours": sum(row["provider_duration_s"] for row in selected_rows)/3600,
            "estimated_song_transfer_bytes": main_bytes,
            "estimated_song_transfer_gib": main_bytes/1024**3,
            "reserve_song_payload_bytes": reserve_bytes,
            "estimated_max_song_transfer_bytes": sum(row["compressed_bytes"] + 65536
                                                       for row in selected_rows + reserve_rows),
            "transfer_estimate_scope": "Main song payload; fixed reserves and header overhead are separate. Only required reserves are fetched; retries remain under the explicit per-run cap.",
            "metadata_transfer_bytes_this_run": metadata.downloaded_bytes + archive.downloaded_bytes,
            "audio_directory": str(output / "audio"), "metadata_csv": str(csv_path),
            "content_approved": False, "training_admitted": False, "whole_archive_sha1_verified": False}
        if plan_only:
            return {**summary, "state": "planned", "audio_download_started": False}
        audio_root = output / "audio"
        audio_root.mkdir(exist_ok=True)
        receipts_root = output / "song_receipts"
        receipts_root.mkdir(exist_ok=True)
        remaining = _remaining_bytes(selected_rows, audio_root)
        if archive.downloaded_bytes + remaining > full_budget:
            raise ValueError(f"Selected songs exceed remaining byte budget ({remaining} bytes); plan preserved, no new songs requested")
        original_remaining = sum(row["original_bytes"] for row in selected_rows
            if not (audio_root / row["archive_member"]).is_file())
        largest = max((row["original_bytes"] + row["compressed_bytes"] for row in selected_rows + reserve_rows), default=0)
        if shutil.disk_usage(output).free < original_remaining + 2*largest:
            raise ValueError("Insufficient selected-drive space for original MP3s and resumable per-track temporary data")
        completed, actual_seconds = [], 0.0
        receipt_path = output / "download_receipt.json"
        old_completed = read_json(receipt_path) if receipt_path.exists() else None
        for index, row in enumerate(selected_rows + reserve_rows):
            if actual_seconds + 1e-6 >= hours * 3600:
                break
            member = row["archive_member"]
            if not re.fullmatch(r"fma_full/\d{3}/\d{6}\.mp3", member):
                raise ValueError("Selected original path escapes the FMA full audio namespace")
            target = (audio_root / member).resolve()
            if not target.is_relative_to(audio_root):
                raise ValueError("Selected original path escapes audio output")
            if progress:
                progress({"label": "FMA 원곡 선택 다운로드·검증", "done": actual_seconds/3600,
                          "total": hours, "unit": "hours", "track_id": row["track_id"]})
            transfer = archive.extract(member, target)
            header = sf.info(target)
            if header.frames < 1 or header.samplerate < 1 or header.channels < 1:
                raise ValueError("Original FMA track has an unusable audio header")
            track = {"track_id": row["track_id"], "archive_member": member,
                "path": str(target), "sha256": transfer["sha256"], "bytes": target.stat().st_size,
                "crc32": row["crc32"], "provider_duration_s": row["provider_duration_s"],
                "audio_header_duration_s": header.frames/header.samplerate,
                "license": row["license"], "artist_id": row.get("artist_id"),
                "artist_name": row.get("artist_name"), "original_source_url": row.get("original_source_url"),
                "plan": plan_binding, "source": archive.binding(), "audio_reencoded": False,
                "content_approved": False, "training_admitted": False,
                "verification_scope": "Original ZIP entry CRC32, local SHA256, size and audio header; not whole ZIP SHA1 or human review"}
            _same_or_write(receipts_root / (str(row["track_id"]) + ".json"), track)
            completed.append(track)
            actual_seconds += track["audio_header_duration_s"]
            write_json(output / "download_progress.json", {"state": "downloading", "tracks": len(completed),
                "actual_header_hours": actual_seconds/3600, "target_hours": hours,
                "downloaded_bytes_this_run": metadata.downloaded_bytes + archive.downloaded_bytes,
                "content_approved": False})
        if actual_seconds + 1e-6 < hours * 3600:
            raise ValueError("Downloaded originals and fixed reserves are short of actual target hours; preserve results and choose a supplementary plan")
        manifest = output / "downloaded_tracks.jsonl"
        _publish_rows(manifest, completed)
        receipt = {"kind": "fma_selective_download_v1", "state": "downloaded", "plan": plan_binding,
            "manifest": _binding(manifest), "metadata": _binding(csv_path),
            "audio_directory": str(audio_root), "track_count": len(completed), "target_hours": hours,
            "actual_header_hours": actual_seconds/3600,
            "provider_hours": sum(t["provider_duration_s"] for t in completed)/3600,
            "original_mp3_bytes": sum(t["bytes"] for t in completed), "audio_reencoded": False,
            "content_approved": False, "training_admitted": False,
            "whole_archive_sha1_verified": False, "created_at": old_completed.get("created_at") if old_completed else now()}
        _same_or_write(receipt_path, receipt)
        write_json(output / "download_progress.json", {"state": "verified_reuse" if old_completed else "downloaded",
            "tracks": len(completed), "actual_header_hours": actual_seconds/3600, "target_hours": hours,
            "downloaded_bytes_this_run": metadata.downloaded_bytes + archive.downloaded_bytes,
            "content_approved": False})
        return {**summary, **receipt, "receipt": _binding(receipt_path), "reused": old_completed is not None,
            "downloaded_bytes_this_run": metadata.downloaded_bytes + archive.downloaded_bytes}
