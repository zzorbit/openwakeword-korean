"""Select the unchanged official-example music without fetching its full ZIP.

Pins were derived from the SHA1-verified original FMA small and metadata
archives. The 120 MP3s, ordering and full tracks.csv are identical to the
historical extraction. Existing whole archives and extraction receipts remain
untouched; old workflows select their original acquisition policy.
"""
from importlib.resources import files
from pathlib import Path
import re
import time

from .common import identity, now, read_json, write_json, writer_lock
from .downloads import verify_checksum
from .range_zip import RangeZipArchive


POLICY = "selective_fma_small_v1"


def official_music_catalog():
    catalog = read_json(files("korean_openwakeword") / "resources/official_fma_small_v1.json")
    music, metadata = catalog["archives"]["music"], catalog["archives"]["metadata"]
    members = music["members"]
    if (catalog.get("schema_version") != 1 or len(members) != 120
            or len({r["name"] for r in members}) != 120
            or any(not re.fullmatch(r"fma_small/\d{3}/\d{6}\.mp3", r["name"]) for r in members)
            or metadata["members"][0]["name"] != "fma_metadata/tracks.csv"
            or len(metadata["members"]) != 1):
        raise ValueError("Invalid frozen official-example music inventory")
    return catalog


def official_music_preview():
    catalog = official_music_catalog()
    payload = sum(row["compressed_bytes"] for source in catalog["archives"].values() for row in source["members"])
    # Exact selected compressed bytes + exact indexes + bounded local headers
    # and end records. Retransmissions are separate, not claimed as saved bytes.
    index = sum(source["index_bytes"] for source in catalog["archives"].values())
    return {"policy": POLICY, "music_tracks": 120, "selected_payload_bytes": payload,
            "index_bytes": index, "transfer_upper_bytes": payload + index + 121 * 1024 + 2 * 65536,
            "retained_original_bytes": sum(row["size_bytes"] for source in catalog["archives"].values() for row in source["members"]),
            "replaced_whole_archive_bytes": sum(source["size_bytes"] for source in catalog["archives"].values()),
            "scope": "Unchanged 120 original MP3s and tracks.csv; range payload/index/header budget before retries; no whole-ZIP fallback"}


def _destination(shared, key, member):
    # Same audio and metadata paths as the original complete-ZIP extraction.
    if key == "music":
        return shared / "fma_1h" / member["name"]
    return shared / "fma_metadata" / member["name"]


def _member_matches(path, row, progress):
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError("Selected music destination is not an owned regular file")
    if not path.exists():
        return False
    if path.stat().st_size != row["size_bytes"]:
        raise ValueError("Existing official-example member size changed; file preserved")
    def report(record):
        if progress:
            progress({"label": "자료 무결성 검사: " + Path(row["name"]).name, **record})
    verify_checksum(path, "sha256", row["sha256"], progress=report if progress else None)
    return True


def prepare_official_music(shared, progress=None, *, aggregate_progress=False):
    """Fetch only missing pinned members, preserving originals and resume state."""
    shared = Path(shared).resolve()
    if any(part.lower() == ".codex" for part in shared.parts):
        raise ValueError("Codex state is not an audio download directory")
    catalog = official_music_catalog()
    total_selected = sum(row["compressed_bytes"] for source in catalog["archives"].values() for row in source["members"])
    completed_selected = 0
    state_dir = shared / "official_music_selective_v1"
    request = {"policy": POLICY, "catalog_sha256": identity(catalog)}
    state_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(state_dir):
        request_path = state_dir / "request.json"
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError("Frozen official-example music request changed")
        if not request_path.exists():
            write_json(request_path, request)
        bindings, observed_transfer, selected = {}, 0, []
        def aggregate(record, done, *, network_bytes, active, member=None):
            if progress:
                progress({**record, "aggregate_done": True, "done": done, "total": total_selected,
                    "unit": "bytes", "network_bytes": network_bytes, "network_active": active,
                    "bytes_per_second": None, "eta_seconds": None, "current_file": member})
        # Metadata first retains the same provider data available to indexing.
        for key in ("metadata", "music"):
            source = catalog["archives"][key]
            missing = []
            for row in source["members"]:
                path = _destination(shared, key, row)
                def cached_progress(record):
                    aggregate(record, completed_selected, network_bytes=observed_transfer,
                              active=False, member=row["name"])
                if not _member_matches(path, row, cached_progress if aggregate_progress else progress):
                    missing.append(row)
                else:
                    completed_selected += row["compressed_bytes"]
                    if aggregate_progress:
                        aggregate({"phase": "hash", "label": "기존 음악 자료 확인", "verification_reused": True},
                                  completed_selected, network_bytes=observed_transfer, active=False, member=row["name"])
                selected.append({"path": str(path), **row})
            if not missing:
                continue
            total = sum(r["compressed_bytes"] for r in missing)
            # Three attempts of each selected byte are bounded; a full archive
            # response is rejected before reading its body by RangeZipArchive.
            cap = 3 * (total + source["index_bytes"] + len(missing) * 1024 + 2 * 65536)
            with RangeZipArchive(source["source_url"], source["size_bytes"], expected_etag=source["etag"],
                                 max_download_bytes=cap, max_member_bytes=512*1024**2) as archive:
                if (archive.index_sha256 != source["index_sha256"] or archive.index_bytes != source["index_bytes"]):
                    raise ValueError("Official-example ZIP index differs from pinned original")
                bindings[key] = archive.binding()
                inventory = {entry.filename: entry for entry in archive.entries}
                done, started = 0, time.monotonic()
                for row in missing:
                    info = inventory.get(row["name"])
                    if (info is None or info.file_size != row["size_bytes"] or info.compress_size != row["compressed_bytes"]
                            or f"{info.CRC:08x}" != row["crc32"]):
                        raise ValueError("Selected official-example member differs from its pin")
                    def report(record):
                        if progress:
                            if aggregate_progress:
                                aggregate({"label": "필요한 음악 자료 다운로드" if key == "music" else "음악 자료 목록 다운로드",
                                           "phase": "download"}, completed_selected + record["done"],
                                          network_bytes=observed_transfer + archive.downloaded_bytes,
                                          active=record["done"] < record["total"], member=row["name"])
                                return
                            completed = done + record["done"]
                            elapsed = time.monotonic() - started
                            rate = completed / elapsed if elapsed > 0 else None
                            progress({"label": "필요한 음악 자료 다운로드" if key == "music" else "음악 자료 목록 다운로드",
                                      "phase": "download", "done": completed, "total": total, "unit": "bytes",
                                      "bytes_per_second": rate,
                                      "eta_seconds": max(0, (total-completed)/rate) if rate else None})
                    destination = _destination(shared, key, row)
                    if aggregate_progress:
                        aggregate({"phase": "download", "label": "음악 자료 연결·이어받기"}, completed_selected,
                                  network_bytes=observed_transfer + archive.downloaded_bytes, active=True, member=row["name"])
                    result = archive.extract(row["name"], destination, progress=report if progress else None)
                    if result["sha256"] != row["sha256"]:
                        raise ValueError("Selected official-example original differs from frozen SHA256")
                    done += row["compressed_bytes"]
                    completed_selected += row["compressed_bytes"]
                    if aggregate_progress:
                        aggregate({"phase": "hash", "label": "음악 원본 무결성 확인"}, completed_selected,
                                  network_bytes=observed_transfer + archive.downloaded_bytes, active=False, member=row["name"])
                observed_transfer += archive.downloaded_bytes
        result = {"kind": POLICY, "request": request, "files": selected, "archive_bindings": bindings,
                  "observed_transfer_bytes_this_run": observed_transfer,
                  "original_bytes_preserved": True, "audio_reencoded": False,
                  "whole_archive_sha1_verified_this_run": False, "content_approved": False, "completed_at": now()}
        # History is append-only in a unique record. The request remains fixed.
        receipt_path = state_dir / ("receipt_" + identity(result) + ".json")
        write_json(receipt_path, result)
        return {"metadata": shared / "fma_metadata/fma_metadata/tracks.csv",
                "music_directory": shared / "fma_1h", "receipt": str(receipt_path), **result}
