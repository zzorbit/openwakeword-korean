"""Pinned, evaluation-only 30-hour recordings shared by every training budget.

Preparation creates technical candidates, never listening decisions or training
admissions. Durations are unions of original recording intervals, not looped or
summed model exposure. All large artifacts stay in the selected workspace.
"""
from collections import Counter, defaultdict
from importlib.resources import files
import hashlib
import io
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile
import zipfile

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

from .common import bound_file, digest, identity, now, read_json, read_rows, write_json, writer_lock

BUCKET_HOURS = {"korean_conversation": 10, "korean_interview": 5, "korean_read": 3,
                "other_speech": 4, "music": 3, "environment": 4, "electronic": 1}
BUCKET_LABELS = {"korean_conversation": "한국어 자연발화 대화", "korean_interview": "한국어 인터뷰",
                 "korean_read": "한국어 낭독", "other_speech": "영어 등 다른 언어의 말소리",
                 "music": "음악·보컬", "environment": "환경·생활음", "electronic": "전자음·알람·경보"}


def evaluation_material_catalog():
    catalog = read_json(files("korean_openwakeword") / "resources/evaluation_materials_v1.json")
    if catalog.get("schema_version") != 1 or catalog.get("target_hours_by_bucket") != BUCKET_HOURS:
        raise ValueError("Unsupported common evaluation material recipe")
    return catalog


def evaluation_material_preview():
    catalog = evaluation_material_catalog()
    archives = {row["id"]: row for row in catalog["archives"]}
    return [{"id": "common_30h_" + key, "dataset_id": key, "name": BUCKET_LABELS[key],
             "role": "held_out_evaluation_material", "split": "validation", "evaluation_only": True,
             "target_hours": hours, "target_hours_not_prepared_hours": True,
             "description": catalog["buckets"][key]["description"],
             "source_url": catalog["buckets"][key]["source_url"],
             "license": catalog["buckets"][key]["license"],
             "total_bytes": sum(archives[a]["size_bytes"] for a in catalog["buckets"][key]["archives"]),
             "shared_archive_bytes_not_additive": True}
            for key, hours in BUCKET_HOURS.items()]


def evaluation_material_coverage(rows):
    """Count each original interval once, and refuse cross-bucket reuse."""
    def original_rows(items, parents=(), inherited_bucket=None):
        for row in items:
            if not isinstance(row, dict) or id(row) in parents or len(parents) >= 64:
                raise ValueError("Evaluation stream lineage must contain bounded, noncyclic source rows")
            bucket = row.get("eval_bucket")
            if inherited_bucket in BUCKET_HOURS and bucket != inherited_bucket:
                raise ValueError("One original recording cannot fill multiple evaluation buckets")
            if "evaluation_stream_components" in row:
                components = row["evaluation_stream_components"]
                if not isinstance(components, list) or not components:
                    raise ValueError("Evaluation streams need nonempty original source components")
                yield from original_rows(components, (*parents, id(row)), bucket)
            else:
                # Seoul's ordinary ``components`` describe its FLAC parts, not
                # extra evaluation exposure. Only stream wrappers are unrolled.
                yield row

    grouped = defaultdict(list)
    original_aliases = {}
    for row in original_rows(rows):
        bucket = row.get("eval_bucket")
        if bucket not in BUCKET_HOURS:
            continue  # The old diagnostic catalog is supplementary, not quota.
        source = row.get("canonical_recording_id") or row.get("canonical_source_id")
        if not source:
            raise ValueError("Common evaluation recordings need original source identity")
        original_hash = row.get("original_sha256")
        if original_hash:
            source = original_aliases.setdefault(original_hash, source)
        start = float(row.get("source_start_s", 0))
        finish = float(row.get("source_end_s", start + float(row.get("duration_s", 0))))
        if not (math.isfinite(start) and math.isfinite(finish) and 0 <= start < finish):
            raise ValueError("Common evaluation source intervals must be finite and nonempty")
        grouped[source].append((start, finish, bucket))
    totals = Counter()
    counts = Counter()
    for source, intervals in grouped.items():
        if len({item[2] for item in intervals}) != 1:
            raise ValueError("One original recording cannot fill multiple evaluation buckets")
        bucket = intervals[0][2]
        merged = []
        for start, finish, _ in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(finish, merged[-1][1])
            else:
                merged.append([start, finish])
        totals[bucket] += sum(b - a for a, b in merged)
        counts[bucket] += 1
    actual = {key: totals[key] / 3600 for key in BUCKET_HOURS}
    shortages = [f"{BUCKET_LABELS[key]}: 실제 {actual[key]:.3f}시간 / 필요 {hours}시간"
                 for key, hours in BUCKET_HOURS.items() if totals[key] + .001 < hours * 3600]
    return {"target_hours_by_bucket": BUCKET_HOURS.copy(), "actual_hours_by_bucket": actual,
            "unique_recordings_by_bucket": dict(counts), "actual_total_hours": sum(actual.values()),
            "target_total_hours": 30, "target_met": not shortages, "shortages": shortages,
            "duration_scope": "unique original recording intervals; no loops or model-count multiplication"}


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Archive member must stay inside its original-source directory")
    return path


def _download_archive(asset, downloads, progress=None):
    """Official SHA256 pins, or the original publisher's MD5 plus local SHA256."""
    from .studio_data import _download, _emit
    if asset.get("checksum_algorithm") != "md5":
        return _download(asset, downloads, progress)
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / asset["filename"]
    expected, size = asset["md5"], asset["size_bytes"]
    if not re.fullmatch(r"[0-9a-f]{32}", expected) or not asset["source_url"].startswith("https://"):
        raise ValueError("Publisher checksum and HTTPS are required")
    def sums(path):
        md5, sha = hashlib.md5(), hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                md5.update(block); sha.update(block)
        return md5.hexdigest(), sha.hexdigest()
    with writer_lock(downloads / ("." + target.name + ".download")):
        reused = target.is_file()
        verified = None
        if not reused:
            partial = target.with_name(target.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size or shutil.disk_usage(downloads).free < size - offset:
                raise ValueError("Partial archive size or selected-workspace free space is invalid")
            if offset < size:
                headers = {"Accept-Encoding": "identity", **({"Range": f"bytes={offset}-"} if offset else {})}
                with requests.get(asset["source_url"], headers=headers, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if response.headers.get("Content-Range", "").split("/", 1) != [f"bytes {offset}-{size-1}", str(size)]:
                            raise ValueError("Publisher resume range differs from the pinned archive")
                    elif response.status_code == 200:
                        offset = 0
                    else:
                        raise ValueError("Unexpected partial publisher response")
                    with partial.open("ab" if offset else "wb") as stream:
                        for block in response.iter_content(4 * 1024**2):
                            stream.write(block)
                            _emit(progress, "공통 평가 원본 다운로드 · " + asset["id"], stream.tell(), size, "bytes")
            verified = sums(partial) if partial.stat().st_size == size else None
            if verified is None or verified[0] != expected:
                raise ValueError("Publisher archive checksum failed; partial source preserved")
            partial.replace(target)
        md5, sha = verified or sums(target)
        if target.stat().st_size != size or md5 != expected:
            raise ValueError("Existing publisher archive changed")
        record = {"path": str(target), "bytes": size,
            "publisher_md5": md5, "sha256": sha, "source_url": asset["source_url"],
            "reused": reused, "content_approved": False, "completed_at": now()}
        receipt_path = target.with_name(target.name + ".receipt.json")
        if receipt_path.exists():
            previous = read_json(receipt_path)
            if any(previous.get(key) != record[key] for key in ("path", "bytes", "publisher_md5", "sha256", "source_url")):
                raise ValueError("Existing publisher archive receipt differs; original evidence preserved")
        else:
            write_json(receipt_path, record)
    return target


class _Volumes(io.RawIOBase):
    """Seek across pinned ZIP volumes without copying their multi-GB contents."""
    def __init__(self, paths):
        self.streams = [Path(path).open("rb") for path in paths]
        self.offsets = [0]
        for path in paths:
            self.offsets.append(self.offsets[-1] + Path(path).stat().st_size)
        self.position = 0
        self.patch = b""
        self.patch_start = self.offsets[-1]
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.offsets[-1] + offset
        if target < 0: raise ValueError("Negative archive offset")
        self.position = target
        return target
    def read(self, size=-1):
        end = self.offsets[-1] if size < 0 else min(self.offsets[-1], self.position + size)
        result = []
        while self.position < end:
            if self.position >= self.patch_start:
                stop = end; start = self.position - self.patch_start
                result.append(self.patch[start:start + stop - self.position]); self.position = stop
                continue
            disk = next(i for i, finish in enumerate(self.offsets[1:]) if self.position < finish)
            stop = min(end, self.offsets[disk+1], self.patch_start)
            stream = self.streams[disk]; stream.seek(self.position - self.offsets[disk])
            part = stream.read(stop - self.position)
            if not part: raise EOFError("Truncated pinned ZIP volume")
            self.position += len(part); result.append(part)
        return b"".join(result)
    def close(self):
        for stream in self.streams: stream.close()
        super().close()


def _split_zip(paths):
    """Expose the original split ZIP with corrected logical footer offsets.

    Neither original volume is modified. ZipFile still checks the selected
    member's CRC and uncompressed size. Source IDs and original terms are kept.
    """
    view = _Volumes(paths)
    view.seek(-min(65557, view.offsets[-1]), 2)
    tail_start = view.tell(); tail = bytearray(view.read())
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        view.close(); raise ValueError("Split ZIP footer is missing")
    fields = list(struct.unpack_from("<4s4H2LH", tail, eocd))
    cd_disk = fields[2]; locator = tail.rfind(b"PK\x06\x07", 0, eocd)
    if locator >= 0:
        end64 = tail.rfind(b"PK\x06\x06", 0, locator)
        if end64 < 0:
            view.close(); raise ValueError("ZIP64 footer is not in the bounded archive tail")
        z64 = list(struct.unpack_from("<4sQ2H2L4Q", tail, end64))
        cd_disk = z64[5]
        z64[4] = z64[5] = 0; z64[9] += view.offsets[cd_disk]
        struct.pack_into("<4sQ2H2L4Q", tail, end64, *z64)
        struct.pack_into("<4sLQL", tail, locator, b"PK\x06\x07", 0, tail_start + end64, 1)
        fields[1] = fields[2] = 0; fields[6] = 0xFFFFFFFF
    else:
        absolute = fields[6] + view.offsets[cd_disk]
        if absolute > 0xFFFFFFFF:
            view.close(); raise ValueError("Large split ZIP needs its publisher ZIP64 footer")
        fields[1] = fields[2] = 0; fields[6] = absolute
    struct.pack_into("<4s4H2LH", tail, eocd, *fields)
    view.patch_start = tail_start; view.patch = bytes(tail)
    archive = zipfile.ZipFile(view)
    for entry in archive.infolist():
        if not 0 <= entry.volume < len(paths):
            archive.close(); view.close(); raise ValueError("ZIP member volume is invalid")
        entry.header_offset += view.offsets[entry.volume]
    finish = archive.start_dir
    for entry in sorted(archive.infolist(), key=lambda item: item.header_offset, reverse=True):
        entry._end_offset = finish
        finish = entry.header_offset
    return archive, view


def _copy_member(stream, output, size, expected=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    h = hashlib.sha256(); total = 0
    with temporary.open("wb") as destination:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            total += len(block)
            if total > size: raise ValueError("Archive member exceeds its declared size")
            h.update(block); destination.write(block)
    if total != size or expected and h.hexdigest() != expected:
        raise ValueError("Original archive member differs from its pin")
    if output.exists() and digest(output) != h.hexdigest():
        raise ValueError("Existing immutable evaluation original differs")
    os.replace(temporary, output)
    return h.hexdigest()


def _extract_selected(asset, paths, members, output, progress=None):
    """One sequential tar pass, or bounded ZIP-member reads; never extractall."""
    wanted = {row["member"]: row for row in members}
    from .studio_data import _emit
    result = {}
    for name, row in wanted.items():
        path = output.joinpath(*_safe_member(name).parts)
        if path.is_file() and row.get("sha256") and digest(path) == row["sha256"]:
            result[name] = path
    missing = set(wanted) - set(result)
    if not missing: return result
    archive_path = paths[asset["id"]]
    if asset["format"] == "tar_nested_zip":
        nested = output / asset["nested_member"]
        if not nested.exists() or digest(nested) != asset["nested_sha256"]:
            with tarfile.open(archive_path, mode="r|gz", bufsize=4 * 1024**2) as archive:
                for entry in archive:
                    _safe_member(entry.name)
                    if entry.name == asset["nested_member"] and entry.isfile():
                        with archive.extractfile(entry) as stream:
                            _copy_member(stream, nested, entry.size, asset["nested_sha256"])
                        break
                else: raise ValueError("Pinned nested Seoul archive is missing")
        handle, view = zipfile.ZipFile(nested), None
    elif asset["format"] == "split_zip":
        handle, view = _split_zip([paths[key] for key in asset["volume_ids"]])
    else:
        with tarfile.open(archive_path, mode="r|gz", bufsize=4 * 1024**2) as archive:
            for entry in archive:
                _safe_member(entry.name)
                if entry.name not in missing: continue
                if not entry.isfile() or entry.size > 512 * 1024**2:
                    raise ValueError("Selected public source must be a bounded regular member")
                destination = output.joinpath(*PurePosixPath(entry.name).parts)
                with archive.extractfile(entry) as stream:
                    _copy_member(stream, destination, entry.size, wanted[entry.name].get("sha256"))
                result[entry.name] = destination
                missing.remove(entry.name)
                _emit(progress, "공통 평가 원본 추출 · " + asset["id"], len(result), len(wanted), "sources")
                if not missing: break
        if missing: raise ValueError("Pinned public archive is missing selected recordings")
        return result
    try:
        for name in sorted(missing):
            entry = handle.getinfo(name)
            if entry.is_dir() or entry.file_size > 512 * 1024**2 or entry.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError("Selected ZIP member must be a bounded regular source")
            destination = output.joinpath(*PurePosixPath(name).parts)
            with handle.open(entry) as stream:
                _copy_member(stream, destination, entry.file_size, wanted[name].get("sha256"))
            result[name] = destination
            _emit(progress, "공통 평가 원본 추출 · " + asset["id"], len(result), len(wanted), "sources")
    finally:
        handle.close()
        if view is not None: view.close()
    return result


def _normalize(source, destination, start, seconds):
    """Preserve level, record the contiguous source interval and any headroom."""
    from math import gcd
    from .audio import write_pcm16
    from .pool_review import _technical_check
    info = sf.info(source)
    begin = round(start * info.samplerate); count = min(round(seconds * info.samplerate), info.frames - begin)
    if count <= 0: raise ValueError("Common evaluation source interval is empty")
    with sf.SoundFile(source) as stream:
        stream.seek(begin); original = stream.read(count, dtype="float64", always_2d=True)
    if not np.isfinite(original).all(): raise ValueError("Nonfinite original evaluation source")
    clipped = float(np.mean(np.abs(original) >= 32760 / 32768))
    if clipped >= .01: raise ValueError("Excessively clipped original common evaluation source")
    mono = original.mean(axis=1)
    if info.samplerate != 16000:
        divisor = gcd(info.samplerate, 16000)
        mono = resample_poly(mono, 16000 // divisor, info.samplerate // divisor)
    peak = float(np.max(np.abs(mono), initial=0)); gain = .95 / peak if peak > 1 else 1.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.wav")
    write_pcm16(temporary, mono * gain)
    technical = _technical_check(temporary, "environment")
    if destination.exists() and digest(destination) != digest(temporary):
        raise ValueError("Immutable normalized common evaluation source changed")
    os.replace(temporary, destination)
    return {"duration_s": len(mono) / 16000, "normalized_frames": len(mono),
        "source_start_s": begin / info.samplerate, "source_end_s": (begin + count) / info.samplerate,
        "technical_qc": technical, "conversion": {"method": "contiguous_interval_mono_polyphase_16khz_PCM16",
            "original_clipped_fraction": clipped, "uniform_gain": gain, "silence_trimmed": False,
            "noise_added": False, "original_sample_rate": info.samplerate}}


def _known_train(project, prepared):
    from .studio_evaluation_data import _known_train
    return list(_known_train(project, prepared or {}))


def prepare_common_sources(project, workspace, prepared=None, progress=None):
    """Prepare actual, fixed 30-hour technical candidates; resume by hashes.

    Review and admission are owned by the surrounding Studio review workflow.
    Existing frozen experiments are rejected by that caller before this API.
    """
    from .project import load_project
    from .studio_data import _emit, _write_rows
    root, config = load_project(project)
    workspace = Path(workspace).resolve()
    if any(part.lower() == ".codex" for part in workspace.parts):
        raise ValueError("Common evaluation data must stay in the selected project workspace")
    catalog = evaluation_material_catalog(); catalog_id = identity(catalog)[:16]
    stage = workspace / "shared" / ("common_evaluation_30h_" + catalog_id)
    stage.mkdir(parents=True, exist_ok=True)
    known = _known_train(project, prepared)
    request = {"catalog_id": catalog_id, "known_train": identity(known)}
    output = stage / ("selection_" + identity(request)[:16])
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "preparation.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("request") != request: raise ValueError("Common evaluation selection changed")
        for row in receipt["source_rows"]: bound_file(root, row)
        return {**receipt, "resumed": True}
    required = catalog["known_download_bytes"] + catalog.get("working_bytes", 12 * 1024**3)
    downloads = workspace / "shared" / "downloads"
    cached = sum(a["size_bytes"] for a in catalog["archives"] if (downloads / a["filename"]).is_file())
    if shutil.disk_usage(stage).free < required - cached:
        raise ValueError("Selected workspace lacks space for pinned originals and common evaluation WAVs")
    paths = {a["id"]: _download_archive(a, downloads, progress) for a in catalog["archives"]}
    rows, failures = [], []
    with writer_lock(output):
        candidates = _candidate_rows(catalog, paths, stage, project, workspace, known, progress)
        used = set()
        seconds = Counter()
        for candidate in candidates:
            bucket = candidate["eval_bucket"]
            if seconds[bucket] + .001 >= BUCKET_HOURS[bucket] * 3600: continue
            original = Path(candidate["original_path"])
            source = candidate["canonical_recording_id"]
            # A single parent recording stays in one purpose bucket. Parts of
            # the same interview may contribute disjoint contiguous intervals.
            ownership = next((b for s, b in used if s == source), None)
            if ownership and ownership != bucket: continue
            count = min(candidate["selection_duration_s"], BUCKET_HOURS[bucket] * 3600 - seconds[bucket])
            token = identity({"candidate": candidate, "seconds": count})[:24]
            check = output / "checks" / (token + ".json")
            if check.exists():
                row = read_json(check); bound_file(root, row)
                if digest(original) != row["original_sha256"]: raise ValueError("Original common evaluation recording changed")
            else:
                destination = output / "audio" / (token + ".wav")
                try:
                    normalized = _normalize(original, destination, candidate.get("selection_start_s", 0), count)
                except ValueError as exc:
                    failures.append({"source": source, "bucket": bucket, "reason": str(exc)})
                    continue
                from .studio_evaluation_data import _pcm_hash
                pcm, frames = _pcm_hash(destination)
                row = {**candidate, **normalized, "path": str(destination), "sha256": digest(destination),
                    "original_sha256": digest(original), "normalized_pcm_sha256": pcm,
                    "normalized_frames": frames, "sample_id": "common30_" + token, "split": "validation",
                    "technical_pass": True, "provenance": "public_dataset", "training_admitted": False,
                    "evaluation_only": True, "evaluation_content_approved": False,
                    "individual_human_listening": False, "pcm_hash_scope": "exact_full_source_16khz_mono_pcm16"}
                write_json(check, row)
            rows.append(row); seconds[bucket] += row["duration_s"]; used.add((source, bucket))
            _emit(progress, "공통 평가 원본 준비 · " + BUCKET_LABELS[bucket], seconds[bucket], BUCKET_HOURS[bucket] * 3600, "seconds")
        coverage = evaluation_material_coverage(rows)
        receipt = {"kind": "common_evaluation_materials_v1", "request": request, "catalog_id": catalog_id,
            "source_rows": rows, "coverage": coverage, "target_met": coverage["target_met"],
            "shortages": coverage["shortages"], "source_count": len(rows), "failures": failures,
            "human_listened_count": 0, "training_admitted": False, "evaluation_only": True,
            "Final_accessed": False, "created_at": now(), "limitations": catalog["limitations"]}
        _write_rows(output / "sources.jsonl", rows); write_json(receipt_path, receipt)
        return {**receipt, "resumed": False}


def _candidate_rows(catalog, paths, stage, project, workspace, known, progress):
    """Resolve source metadata from the pinned catalog and original providers."""
    # The catalog contains only true recording metadata; selection ordering and
    # quota allocation never relabel a read utterance as natural conversation.
    assets = {row["id"]: row for row in catalog["archives"]}
    from .source_identity import partition_keys
    forbidden = {item for row in known for item in partition_keys(row)}
    result = []
    for archive_id, selections in catalog["selections"].items():
        selections = [row for row in selections if not forbidden.intersection(partition_keys(row))]
        extraction = [part for row in selections for part in row.get("members", [row])]
        from .studio_data import _emit
        _emit(progress, "공통 평가 원본 추출 · " + archive_id, 0, len(extraction), "sources")
        original_paths = _extract_selected(assets[archive_id], paths, extraction, stage / "originals" / archive_id, progress)
        for row in selections:
            if row.get("members"):
                source, components = _join_seoul(row, original_paths, stage / "originals" / archive_id / "joined")
                result.append({**row, "original_path": str(source), "components": components,
                    "source_construction": "provider-ordered consecutive interview parts; first contiguous 20 or 30 minutes"})
            else:
                result.append({**row, "original_path": str(original_paths[row["member"]])})
    result.extend(_zeroth_candidates(project, workspace, known, catalog))
    return result


def _join_seoul(row, paths, output):
    """Keep one parent interview per evaluation stream, with its FLAC lineage."""
    output.mkdir(parents=True, exist_ok=True)
    destination = output / (row["canonical_speaker_id"].rsplit(":", 1)[-1] + ".wav")
    receipt_path = destination.with_suffix(".json")
    requested = [{"path": str(paths[m["member"]]), "sha256": digest(paths[m["member"]])} for m in row["members"]]
    request = {"parts": requested, "duration_s": row["selection_duration_s"]}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["request"] != request or digest(destination) != receipt["sha256"]:
            raise ValueError("Joined Seoul original or its provider parts changed")
        return destination, receipt["components"]
    temporary = destination.with_name(destination.stem + ".part.wav")
    first = sf.info(requested[0]["path"]); remaining = round(row["selection_duration_s"] * first.samplerate)
    components = []
    with sf.SoundFile(temporary, "w", samplerate=first.samplerate, channels=first.channels, subtype="PCM_16") as target:
        for source in requested:
            info = sf.info(source["path"])
            if info.samplerate != first.samplerate or info.channels != first.channels:
                raise ValueError("Consecutive Seoul provider parts have inconsistent audio formats")
            count = min(info.frames, remaining)
            if count <= 0: break
            with sf.SoundFile(source["path"]) as stream:
                for block in stream.blocks(blocksize=65536, frames=count, dtype="int16"):
                    target.write(block)
            components.append({**source, "role": "original_interview_part", "original_path": source["path"],
                "original_sha256": source["sha256"], "canonical_recording_id": row["canonical_recording_id"],
                "canonical_source_id": row["canonical_recording_id"] + ":part:" + Path(source["path"]).stem,
                "canonical_speaker_id": row["canonical_speaker_id"], "source_start_s": 0,
                "source_end_s": count / first.samplerate, "split": "validation", "provenance": "public_dataset",
                "source_url": row["source_url"], "license": row["license"]})
            remaining -= count
    if remaining: raise ValueError("Original Seoul interview is shorter than the pinned interval")
    if destination.exists() and digest(destination) != digest(temporary):
        raise ValueError("Existing joined Seoul original differs")
    os.replace(temporary, destination)
    write_json(receipt_path, {"request": request, "sha256": digest(destination), "components": components})
    return destination, components


def _zeroth_candidates(project, workspace, known, catalog):
    """Reuse the original Zeroth index; reserve whole speakers absent in Train."""
    from .project import load_project
    from .preparation import create_source_index
    root, _ = load_project(project)
    directory = workspace / "shared" / "zeroth"
    index = root / ".studio" / "materials" / "zeroth_source_index.jsonl"
    if not index.exists():
        raise ValueError("Prepare the pinned Zeroth training source/index before common evaluation")
    excluded = {r.get("canonical_speaker_id") for r in known if r.get("canonical_speaker_id")}
    asset = next(r for r in catalog["archives"] if r["id"] == "zeroth")
    grouped = defaultdict(list)
    for row in read_rows(index):
        speaker = row.get("canonical_speaker_id")
        if speaker in excluded or not speaker: continue
        source = directory / row["path"]
        if not source.resolve().is_relative_to(directory.resolve()): raise ValueError("Zeroth path escapes selected corpus")
        grouped[speaker].append((row, source))
    # Interleave original speakers and retain their provider-declared partition.
    positions = Counter(); result = []
    while True:
        changed = False
        for speaker in sorted(grouped):
            position = positions[speaker]
            if position >= len(grouped[speaker]): continue
            row, source = grouped[speaker][position]; positions[speaker] += 1; changed = True
            result.append({**row, "original_provider_split": row.get("split"), "split": "validation",
                "original_path": str(source), "eval_bucket": "korean_read", "eval_type": "korean_read_speech",
                "eval_type_label": "한국어 낭독", "selection_start_s": 0, "selection_duration_s": 180,
                "license": asset["license"], "source_url": "https://www.openslr.org/40/",
                "reservation_scope": "whole provider speaker absent from this project's raw Train inputs"})
        if not changed: break
    return result
