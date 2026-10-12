"""Source/family separation and reusable public-audio manifests; no STT sweep."""
from collections import defaultdict
import math
import re
from pathlib import Path
import numpy as np
import soundfile as sf

from .common import bound_file, digest, identity, read_json, read_rows, resolved, write_json
from .project import load_project

CATEGORIES = ("environment", "music", "competing_speech")


def validate_hard_negative(row, config):
    """A reviewed non-call phrase, never an unlabeled substitute Negative."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Hard Negative needs its declared non-call phrase in text")
    normalized = "".join(re.findall(r"[가-힣]", text))
    if not normalized or config["wakeword"] in normalized:
        raise ValueError("Hard Negative text cannot contain the configured wakeword")
    if row.get("speech") is not True:
        raise ValueError("Hard Negative must be a speech phrase")
    if row.get("provenance") == "qwen_tts":
        if row.get("human_reference_used") is not False or not row.get("voice_family_id"):
            raise ValueError("Synthetic Hard Negative needs a Qwen-only voice family without human reference")
    elif row.get("provenance") != "public_dataset" or not row.get("source_url") or not row.get("license"):
        raise ValueError("Hard Negative needs public source provenance or Qwen-only synthesis")


def rows_for(root, config, pool):
    path = resolved(root, config["manifests"][pool])
    return path, read_rows(path)


def source_key(row):
    return row.get("canonical_source_id", row.get("source_id"))


def coverage_interval(row, start, end):
    """Count segmented originals in their declared parent time coordinates."""
    parent = row.get("canonical_recording_id", row.get("parent_source_id"))
    if parent and row.get("parent_start_s") is not None:
        offset = float(row["parent_start_s"])
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("Invalid parent recording offset")
        return parent, (offset + start, offset + end)
    return source_key(row), (start, end)


def audit_project(project, check_files=True, include_final_files=False, extra_rows=None):
    root, config = load_project(project)
    groups, errors, totals = defaultdict(set), [], {}
    inspected = {}
    collections = [(pool, read_rows(resolved(root, manifest))) for pool, manifest in config["manifests"].items()]
    if extra_rows:
        collections.append(("external_evaluation", extra_rows))
    for pool, rows in collections:
        ids, intervals = set(), defaultdict(list)
        for row in rows:
            sid = row.get("sample_id", row.get("id"))
            split = row.get("split")
            if not sid or sid in ids:
                errors.append(f"{pool}: missing/duplicate sample ID {sid}")
            ids.add(sid)
            if split not in ("train", "validation", "final") or not source_key(row):
                errors.append(f"{pool}/{sid}: canonical source/split required")
            if pool in ("generic_negative", "korean_negative", "hard_negative", "noise", "rir", "positive_train") and split != "train":
                errors.append(f"{pool}/{sid}: training input cannot use held-out audio")
            if pool == "positive_validation" and split != "validation" or pool == "positive_final" and split != "final":
                errors.append(f"{pool}/{sid}: wrong voice-family partition")
            if pool in ("development", "final") and split != ("validation" if pool == "development" else "final"):
                errors.append(f"{pool}/{sid}: wrong evaluation split")
            if pool in ("noise_validation", "noise_final") and split != ("validation" if pool == "noise_validation" else "final"):
                errors.append(f"{pool}/{sid}: wrong held-out noise split")
            for key, value in (("source", source_key(row)), ("sha256", row.get("sha256")),
                               ("recording", row.get("canonical_recording_id", row.get("parent_source_id"))),
                               ("family", row.get("voice_family_id")), ("speaker", row.get("canonical_speaker_id")),
                               ("room", row.get("canonical_room_id"))):
                if value:
                    groups[(key, value)].add(split)
            # Mixed evaluation clips retain all component source IDs.
            for value in row.get("component_source_ids", []):
                groups[("source", value)].add(split)
            if pool.startswith("positive_"):
                if row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False or row.get("text") != config["wakeword"]:
                    errors.append(f"{pool}/{sid}: standalone Qwen/no-human provenance required")
                if not row.get("voice_family_id"):
                    errors.append(f"{pool}/{sid}: synthetic family ID required")
            elif pool == "hard_negative":
                try:
                    validate_hard_negative(row, config)
                except ValueError as e:
                    errors.append(f"{pool}/{sid}: {e}")
            elif pool not in ("development", "final", "external_evaluation"):
                if row.get("provenance") != "public_dataset" or not row.get("license") or not row.get("source_url"):
                    errors.append(f"{pool}/{sid}: public source URL/license/provenance required")
            if row.get("category") and row["category"] not in (*CATEGORIES, "clean", "rir"):
                errors.append(f"{pool}/{sid}: unsupported category")
            if check_files and (split != "final" or include_final_files):
                try:
                    candidate = resolved(root, row["path"])
                    file_key = (str(candidate), row["sha256"])
                    if file_key not in inspected:
                        inspected[file_key] = sf.info(bound_file(root, row))
                    info = inspected[file_key]
                    if info.frames == 0:
                        errors.append(f"{pool}/{sid}: empty audio")
                    start, end = float(row.get("start_s", 0)), float(row.get("end_s", info.duration))
                    if not 0 <= start < end <= info.duration + 1e-5:
                        errors.append(f"{pool}/{sid}: invalid source interval")
                    original, interval = coverage_interval(row, start, end)
                    intervals[original].append(interval)
                except (ValueError, OSError, RuntimeError) as e:
                    errors.append(f"{pool}/{sid}: {e}")
        totals[pool] = {"rows": len(rows), "unique_source_hours": union_hours(intervals) if check_files and not any(r.get("split") == "final" for r in rows) or include_final_files and check_files else None,
                       "final_audio_opened": bool(include_final_files and any(r.get("split") == "final" for r in rows))}
    leaks = [{"key": key[0], "id": key[1], "splits": sorted(values)} for key, values in groups.items() if len(values) > 1]
    errors += [f"Cross-split {r['key']} leakage: {r['id']}" for r in leaks]
    return {"kind": "project_source_and_family_audit", "passed": not errors,
            "wakeword": config["wakeword"], "pools": totals, "errors": errors, "leaks": leaks,
            "scope": "Declared IDs and exact files; acoustic near-duplicates require pilot review."}


def union_hours(intervals):
    seconds = 0.0
    for values in intervals.values():
        end = -math.inf
        for a, b in sorted(values):
            seconds += max(0, b - max(a, end))
            end = max(end, b)
    return seconds / 3600


def validate_admission(root, rows, config, role, smoke=False):
    if not rows:
        raise ValueError(f"Empty {role} manifest")
    inspected = set()
    admissions = {}
    admitted_hashes = {}
    for row in rows:
        if row.get("split") != "train" or not source_key(row):
            raise ValueError("Training inputs need immutable source IDs and train split")
        key = (row["path"], row["sha256"])
        if key not in inspected:
            bound_file(root, row)
            inspected.add(key)
        if role == "positive":
            if row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False or row.get("text") != config["wakeword"]:
                raise ValueError("Positive must be the standalone configured Qwen word without human reference")
        if role == "hard_negative":
            validate_hard_negative(row, config)
        if not smoke:
            admission = row.get("admission")
            if not isinstance(admission, dict):
                raise ValueError("Formal input needs a hash-bound admission record")
            key = (admission["path"], admission["sha256"])
            if key not in admissions:
                admissions[key] = read_json(bound_file(root, admission))
                admitted_hashes[key] = set(admissions[key].get("accepted_source_sha256", []))
            artifact = admissions[key]
            if artifact.get("state") != "accepted" or artifact.get("wakeword") not in (None, config["wakeword"]):
                raise ValueError("Admission is not accepted for this word")
            if role == "positive" and (artifact.get("human_reference_used") is not False or artifact.get("generation_policy_approved") is not True):
                raise ValueError("Approved standalone synthetic generation policy required")
            if role != "positive" and artifact.get("content_review") != "accepted":
                raise ValueError("Public material needs source/content review, not full STT")
            if role != "positive" and row["sha256"] not in admitted_hashes[key]:
                raise ValueError("Content review does not cover this exact source audio")


def prepare_windows(project, pool, output):
    """Expand admitted public recordings into fixed 2s windows without copying WAVs."""
    root, config = load_project(project)
    if pool not in ("generic_negative", "korean_negative", "hard_negative"):
        raise ValueError("Only training Negative pools use fixed windows")
    _, sources = rows_for(root, config, pool)
    validate_admission(root, sources, config, pool)
    audit = audit_project(project)
    if not audit["passed"]:
        raise ValueError("Source/split audit failed before window creation: " + str(audit["errors"][:3]))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Window manifests are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    count = 0
    with output.open("x", encoding="utf-8") as f:
        for row in sources:
            info = sf.info(resolved(root, row["path"]))
            a, b = float(row.get("start_s", 0)), float(row.get("end_s", info.duration))
            for index in range(int(math.ceil((b - a) / 2))):
                start = a + index * 2
                entry = {**row, "sample_id": f"{row['sample_id']}_window{index:06d}",
                         "start_s": start, "end_s": min(b, start + 2), "window_policy": "fixed_2s_no_audio_copy"}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    return {"rows": count, "path": str(output), "sha256": digest(output),
            "next": "Set project.json manifests pool to this file; original provenance/admission is retained."}


def seek_audio(root, row, seconds=2, rng=None):
    """Read a bounded interval before resampling; do not decode a long file per draw."""
    from .audio import read_audio
    from scipy.signal import resample_poly
    from math import gcd
    path = resolved(root, row["path"])
    with sf.SoundFile(path) as f:
        lo = float(row.get("start_s", 0))
        hi = float(row.get("end_s", f.frames / f.samplerate))
        if rng is not None:
            lo += float(rng.uniform(0, max(0, hi - lo - seconds)))
        f.seek(round(lo * f.samplerate))
        values = f.read(min(round(seconds * f.samplerate), round((hi - lo) * f.samplerate)), dtype="float64", always_2d=True).mean(axis=1)
        rate = f.samplerate
    if rate != 16000:
        d = gcd(rate, 16000)
        values = resample_poly(values, 16000 // d, rate // d)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite bounded audio")
    return values, lo
