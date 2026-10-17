"""Pack short held-out recordings into exact, provenance-bound PCM streams.

No source is cut between streams. No gaps, fades, looping, normalization or
resampling are introduced. The persisted WAV is the playback/scoring input.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import re
import shutil

import soundfile as sf

from .common import bound_file, digest, identity, read_json, resolved, write_json, writer_lock
from .data import coverage_interval, source_key
from .project import load_project
from .source_identity import component_lineage
from .storage import project_storage


def _inventory(root, rows, wakeword):
    buckets, passthrough, hashes, intervals = defaultdict(lambda: defaultdict(list)), [], {}, defaultdict(list)
    admissions, seen_audio, recording_buckets = {}, set(), {}
    for row in rows:
        if (row.get("split") != "validation" or row.get("category") == "rir" or row.get("calls")
                or row.get("kind") not in (None, "negative")):
            raise ValueError("Continuous evaluation streams accept held-out non-call recordings, never RIRs or calls")
        if row.get("provenance") != "public_dataset" or not row.get("source_url") or not row.get("license"):
            raise ValueError("Evaluation stream originals need public provenance, URL and license")
        bucket = row.get("eval_bucket")
        if not bucket:
            passthrough.append({**row, "kind": "negative", "calls": []})
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", bucket):
            raise ValueError("Evaluation bucket needs a safe stable ASCII identifier")
        if not source_key(row) or not row.get("sha256"):
            raise ValueError("Evaluation stream originals need immutable source identity and SHA256")
        admission = row.get("evaluation_admission")
        if row.get("evaluation_content_approved") is not True or not isinstance(admission, dict):
            raise ValueError("Pack only genuinely reviewed and admitted common evaluation sources")
        admission_key = admission["path"], admission["sha256"]
        if admission_key not in admissions:
            record = read_json(bound_file(root, admission))
            if (record.get("state") != "accepted_for_evaluation" or record.get("evaluation_only") is not True
                    or record.get("training_admitted") is not False or record.get("wakeword") != wakeword):
                raise ValueError("Common evaluation stream admission has an invalid scope")
            admissions[admission_key] = set(record.get("accepted_source_sha256", []))
        if row["sha256"] not in admissions[admission_key]:
            raise ValueError("Common evaluation admission does not cover this exact original audio")
        path = resolved(root, row["path"])
        key = (str(path), row["sha256"])
        if key not in hashes:
            bound_file(root, row)
            info = sf.info(path)
            if (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16"):
                raise ValueError("Evaluation stream originals must already be mono 16kHz PCM16; no conversion is implicit")
            hashes[key] = (info.frames, path.stat().st_size, path.stat().st_mtime_ns)
        total, _, _ = hashes[key]
        start = round(float(row.get("start_s", 0)) * 16000)
        end = round(float(row.get("end_s", total / 16000)) * 16000)
        if not 0 <= start < end <= total:
            raise ValueError("Evaluation stream original interval is empty or outside its WAV")
        exact = row.get("normalized_pcm_sha256", row["sha256"]), start, end
        if exact in seen_audio:
            raise ValueError("Repeated original PCM intervals cannot be copied into evaluation streams")
        seen_audio.add(exact)
        parent, interval = coverage_interval(row, start / 16000, end / 16000)
        intervals[parent].append((interval[0], interval[1], bucket))
        recording = row.get("canonical_recording_id", row.get("parent_source_id", parent))
        if recording in recording_buckets and recording_buckets[recording] != bucket:
            raise ValueError("One original recording cannot appear in multiple evaluation buckets")
        recording_buckets[recording] = bucket
        buckets[bucket][recording].append({"row": row, "path": path, "start": start,
                                       "frames": end - start, "file_signature": hashes[key][1:]})
    for parent, values in intervals.items():
        if len({value[2] for value in values}) > 1:
            raise ValueError("One original recording cannot appear in multiple evaluation buckets")
        previous = -math.inf
        for start, end, _ in sorted(values):
            if start < previous - 1 / 32000:
                raise ValueError("Repeated or overlapping original recording intervals cannot inflate evaluation streams")
            previous = end
    passthrough.sort(key=identity)
    return buckets, passthrough


def _plans(buckets, seconds):
    plans = []
    for bucket, recordings in sorted(buckets.items()):
        pending, frames = [], 0
        for parent, values in sorted(recordings.items()):
            values = sorted(values, key=lambda value: (value["start"], str(source_key(value["row"])), value["row"]["sha256"]))
            categories = {value["row"]["category"] for value in values}
            if len(categories) != 1:
                raise ValueError("One evaluation recording cannot have conflicting acoustic categories")
            size = sum(value["frames"] for value in values)
            if pending and frames + size > seconds * 16000:
                plans.append((bucket, pending)); pending, frames = [], 0
            pending.extend(values); frames += size
            if frames >= seconds * 16000:
                plans.append((bucket, pending)); pending, frames = [], 0
        if pending:
            plans.append((bucket, pending))
    return plans


def _render(root, output, recipe_sha, number, bucket, sources):
    components, offset = [], 0
    categories = {item["row"]["category"] for item in sources}
    if len(categories) != 1:
        raise ValueError("A continuous evaluation bucket must retain one acoustic category")
    for item in sources:
        count = item["frames"]
        components.append({**item["row"], "stream_start_s": offset / 16000,
            "stream_end_s": (offset + count) / 16000, "original_read_start_frame": item["start"],
            "original_read_frames": count})
        offset += count
    key = identity([recipe_sha, bucket, components])
    sid = f"negative_stream_{number:04d}_{key[:16]}"
    path, receipt = output / "audio" / (sid + ".wav"), output / "streams" / (sid + ".json")
    if receipt.is_file():
        row = read_json(receipt)
        if row.get("stream_recipe_sha256") != recipe_sha or row.get("evaluation_stream_components") != components:
            raise ValueError("Completed evaluation stream provenance differs from its immutable recipe")
        bound_file(root, row)
        return row
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    pcm_hash, written = hashlib.sha256(), 0
    with sf.SoundFile(partial, "w", samplerate=16000, channels=1, subtype="PCM_16", format="WAV") as destination:
        for item in sources:
            before = item["path"].stat()
            if (before.st_size, before.st_mtime_ns) != item["file_signature"]:
                raise ValueError("An original evaluation source changed during stream construction")
            with sf.SoundFile(item["path"]) as source:
                source.seek(item["start"])
                remaining = item["frames"]
                while remaining:
                    block = source.read(min(65536, remaining), dtype="int16")
                    if not len(block):
                        raise ValueError("Original evaluation audio ended before its declared interval")
                    destination.write(block)
                    pcm_hash.update(block.astype("<i2", copy=False).tobytes())
                    written += len(block); remaining -= len(block)
            after = item["path"].stat()
            if (after.st_size, after.st_mtime_ns) != item["file_signature"]:
                raise ValueError("An original evaluation source changed during stream construction")
    verified, actual = hashlib.sha256(), 0
    with sf.SoundFile(partial) as stream:
        if (stream.samplerate, stream.channels, stream.subtype) != (16000, 1, "PCM_16"):
            raise ValueError("Persisted evaluation stream changed PCM format")
        for block in stream.blocks(blocksize=65536, dtype="int16"):
            verified.update(block.astype("<i2", copy=False).tobytes()); actual += len(block)
    if actual != written or written != offset or verified.hexdigest() != pcm_hash.hexdigest():
        raise ValueError("Persisted evaluation stream differs from the exact original PCM sequence")
    checksum = digest(partial)
    if path.exists():
        if digest(path) != checksum:
            raise ValueError("Existing immutable evaluation stream differs; partial reconstruction was preserved")
        partial.unlink()
    else:
        partial.replace(path)
    urls = sorted({item["row"]["source_url"] for item in sources})
    row = {"sample_id": sid, "source_id": "evaluation_stream:" + key,
        "canonical_source_id": "evaluation_stream:" + key, "canonical_recording_id": "evaluation_stream:" + key,
        "path": str(path), "sha256": checksum, "split": "validation", "kind": "negative", "calls": [],
        "category": next(iter(categories)), "eval_bucket": bucket, "eval_type": "continuous_" + bucket,
        "provenance": "public_dataset", "source_url": urls[0], "original_source_urls": urls,
        "source_url_scope": "Original source pointers; the stream itself is deterministic PCM concatenation",
        "license": "Original per-component licenses apply; see evaluation_stream_components",
        "start_s": 0, "end_s": actual / 16000, "duration_s": actual / 16000,
        "normalized_frames": actual, "normalized_pcm_sha256": verified.hexdigest(),
        "pcm_hash_scope": "exact_full_source_16khz_mono_pcm16", "stream_recipe_sha256": recipe_sha,
        "evaluation_stream_components": components, "components": [component_lineage(item["row"], "background") for item in sources],
        "stream_construction": "whole_original_intervals_in_fixed_order_without_gaps_fades_repeat_or_gain",
        "evaluation_only": True, "training_admitted": False, "individual_human_listening": False,
        "individual_stream_listening_claimed": False, "source_recordings": len(sources)}
    write_json(receipt, row)
    return row


@project_storage
def pack_negative_streams(project, rows, output, seconds=600):
    """Return exact longer Negative WAVs plus unchanged supplemental recordings.

    Only rows with eval_bucket are packed. A >seconds original is one standalone
    stream; it is never cut. This reduces repeated frontend warmup losses but
    does not count warmup time as measured False Activation exposure.
    """
    root, config = load_project(project)
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Evaluation stream target seconds must be positive and finite")
    rows = list(rows)
    if not rows:
        raise ValueError("No held-out Negative source recordings are connected")
    buckets, passthrough = _inventory(root, rows, config["wakeword"])
    plans = _plans(buckets, seconds)
    recipe = {"kind": "negative_stream_recipe_v1", "seconds": seconds,
        "source_rows_sha256": identity(sorted(rows, key=lambda row: identity(row))),
        "ordering": "eval_bucket_then_canonical_parent_then_original_interval_v1",
        "transform": "exact_pcm16_concat_no_padding_no_repeat_no_resampling_no_gain",
        "original_split_policy": "never_split_one_recording_across_streams"}
    output = Path(output).resolve()
    if any(part.lower() == ".codex" for part in output.parts):
        raise ValueError("Bulk evaluation streams must not be stored inside .codex")
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output):
        recipe_path = output / "recipe.json"
        if recipe_path.exists() and read_json(recipe_path) != recipe:
            raise ValueError("Continuous evaluation streams use a different immutable input recipe")
        if not recipe_path.exists():
            if any(path.name != ".writer.lock" for path in output.iterdir()):
                raise ValueError("Use a dedicated empty evaluation stream directory")
            write_json(recipe_path, recipe)
        needed = sum(sum(item["frames"] for item in sources) * 2 + 4096 for index, (bucket, sources) in enumerate(plans)
                     if not list((output / "streams").glob(f"negative_stream_{index+1:04d}_*.json")))
        if shutil.disk_usage(output).free < needed:
            raise ValueError("Insufficient selected-workspace space for exact continuous evaluation WAVs")
        result_rows = [_render(root, output, identity(recipe), index + 1, bucket, sources)
                       for index, (bucket, sources) in enumerate(plans)] + passthrough
        value = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in result_rows)
        manifest = output / "manifest.jsonl"
        if manifest.exists() and manifest.read_text(encoding="utf-8") != value:
            raise ValueError("Completed continuous evaluation manifest changed")
        if not manifest.exists():
            manifest.write_text(value, encoding="utf-8")
        receipt = {"kind": "negative_stream_preparation_v1", "recipe": {"path": str(recipe_path), "sha256": digest(recipe_path)},
            "manifest": {"path": str(manifest), "sha256": digest(manifest)}, "rows": result_rows,
            "packed_originals": sum(len(sources) for _, sources in plans), "stream_count": len(plans),
            "unchanged_supplemental_recordings": len(passthrough),
            "packed_seconds": sum(row["duration_s"] for row in result_rows if row.get("evaluation_stream_components")),
            "human_listening_added": 0, "Final_accessed": False,
            "scope": "Exact contiguous source PCM, not new original hours; actual scored negative hours exclude per-stream runtime warmup"}
        receipt_path = output / "receipt.json"
        if receipt_path.exists() and read_json(receipt_path) != receipt:
            raise ValueError("Completed continuous evaluation receipt changed")
        if not receipt_path.exists():
            write_json(receipt_path, receipt)
        return receipt
