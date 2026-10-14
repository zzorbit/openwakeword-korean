"""Prepare public recordings, then admit them using saved human review.

The two phases keep download/technical checks separate from content consent.
Provider metadata is required; this module never transcribes or guesses speech,
language, speaker identity, room identity, or permission to use a source pool.
"""
from collections import defaultdict
import csv
from math import ceil, gcd, isfinite
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from .audio import pcm16
from .common import bound_file, digest, identity, iter_rows, now, read_json, read_rows, resolved, write_json, writer_lock
from .data import CATEGORIES, audit_project, coverage_interval, union_hours, validate_admission
from .pool_review import _technical_check, approve_pool
from .project import load_project
from .streaming import manifest_index

SUPPORTED_AUDIO = frozenset({".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".au"})
TRAIN_POOLS = frozenset({"generic_negative", "korean_negative", "noise", "rir"})
_IDENTITIES = ("canonical_source_id", "canonical_recording_id", "canonical_speaker_id", "canonical_room_id")


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def _own_output(directory, filename, value):
    """A fresh output is empty; a resumed output has this exact ownership file."""
    path = directory / filename
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Preparation/admission inputs or settings changed; use a new output directory")
    else:
        if any(p.name != ".writer.lock" for p in directory.iterdir()):
            raise ValueError("Use a dedicated empty output directory; existing unowned files are preserved")
        write_json(path, value)
    return path


def _publish_rows(path, rows):
    """Publish once; a resumed run must produce the exact same manifest."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if path.exists():
            if path.stat().st_size != temporary.stat().st_size or digest(path) != digest(temporary):
                raise ValueError(f"Immutable manifest changed: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_index(path):
    if path is None:
        return None
    path = Path(path).resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if "speech" in row:
                if row["speech"] not in ("true", "false", "True", "False", ""):
                    raise ValueError("CSV speech must be true or false")
                if row["speech"]:
                    row["speech"] = row["speech"].lower() == "true"
                else:
                    del row["speech"]
    elif path.suffix.lower() == ".jsonl":
        rows = read_rows(path)
    else:
        raise ValueError("Source index must be CSV or JSONL")
    by_path = {}
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("Every source index row needs its relative path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Source index paths must stay within input_directory")
        key = relative_path.as_posix()
        if key in by_path:
            raise ValueError("Duplicate source index path")
        by_path[key] = {k: v for k, v in row.items() if v != "" and (v is not None or k == "speech")}
    return by_path


def _metadata(paths, directory, defaults, index):
    rows = []
    if index is not None and set(index) != {p.relative_to(directory).as_posix() for p in paths}:
        raise ValueError("Source index must cover exactly the discovered audio files")
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        row = {**defaults, **(index[relative] if index is not None else {})}
        if row.get("split") == "final":
            raise ValueError("Final sources stay locked; public preparation never opens Final audio")
        if row.get("split") not in (None, "train", "validation"):
            raise ValueError("Source split must be train or validation")
        provider_hash = row.pop("sha256", None)
        if provider_hash is not None:
            if not isinstance(provider_hash, str) or len(provider_hash) != 64 or any(c not in "0123456789abcdef" for c in provider_hash.lower()):
                raise ValueError("Provider SHA256 must be a 64-character hexadecimal digest")
            row["provider_sha256"] = provider_hash.lower()
        if row.get("excluded") is True:
            if (not isinstance(row.get("exclusion_reason"), str) or not row["exclusion_reason"].strip()
                    or not isinstance(row.get("dataset_id"), str) or not row["dataset_id"].strip()):
                raise ValueError("Provider exclusions need dataset_id and an explicit exclusion_reason")
            canonical = row.get("canonical_source_id", row.get("source_id", f"{row['dataset_id']}:{relative}"))
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("Excluded source needs an actual canonical source ID")
            row.update(path=str(path), original_relative_path=relative, source_id=canonical,
                       canonical_source_id=canonical, canonical_recording_id=row.get("canonical_recording_id", canonical))
            rows.append(row)
            continue
        if row.get("excluded") not in (None, False):
            raise ValueError("excluded must be an explicit boolean")
        for key in ("dataset_id", "source_url", "license", "category_basis"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Public provider metadata needs {key}")
        if row.get("category") not in (*CATEGORIES, "rir"):
            raise ValueError("Declare environment, music, competing_speech or rir category")
        row.setdefault("speech", True if row["category"] == "competing_speech" else False if row["category"] == "rir" else None)
        row.setdefault("speech_presence", "unknown" if row["speech"] is None else "declared")
        if type(row["speech"]) is not bool and not (row["speech"] is None and row["speech_presence"] == "unknown"):
            raise ValueError("speech must be a declared boolean or None with speech_presence=unknown")
        if row["category"] == "competing_speech" and row["speech"] is not True:
            raise ValueError("Competing speech requires speech=True")
        if row["category"] == "rir" and row["speech"] is not False:
            raise ValueError("RIR is a measured filter, not speech audio")
        if not isinstance(row.get("language"), str) or not row["language"].strip():
            raise ValueError("language must be declared; use unknown when metadata does not say")
        canonical = row.get("canonical_source_id", row.get("source_id", f"{row['dataset_id']}:{relative}"))
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("Canonical source ID must be nonempty")
        row.update(path=str(path), original_relative_path=relative,
                   source_id=canonical, canonical_source_id=canonical,
                   canonical_recording_id=row.get("canonical_recording_id", row.get("parent_source_id", canonical)))
        for key in _IDENTITIES:
            if row.get(key) is not None and (not isinstance(row[key], str) or not row[key].strip()):
                raise ValueError("Canonical identities must be nonempty strings")
        if row.get("canonical_speaker_ids") is not None:
            speakers = row["canonical_speaker_ids"]
            if (not isinstance(speakers, list) or not speakers
                    or any(not isinstance(s, str) or not s.strip() for s in speakers)
                    or len(set(speakers)) != len(speakers)):
                raise ValueError("canonical_speaker_ids must list distinct actual speaker IDs")
        if row["category"] == "rir" and not row.get("canonical_room_id"):
            raise ValueError("RIR needs provider's actual canonical_room_id; file names are not room IDs")
        if row["speech"] is not False and not row.get("canonical_speaker_id") and not row.get("canonical_speaker_ids"):
            row["unknown_speaker_group"] = row["dataset_id"]
        rows.append(row)
    active = [r for r in rows if not r.get("excluded")]
    if len({r["canonical_source_id"] for r in active}) != len(active):
        raise ValueError("Repeated canonical source ID; declare segments with a shared canonical_recording_id")
    return rows


def _check_existing_partitions(project, rows):
    root, config = load_project(project)
    held_ids, held_hashes = set(), set()
    for manifest in config["manifests"].values():
        for old in iter_rows(resolved(root, manifest)):
            if old.get("split") in ("validation", "final"):
                held_ids.update((key, old[key]) for key in _IDENTITIES if old.get(key))
                held_ids.update(("canonical_speaker_id", speaker) for speaker in old.get("canonical_speaker_ids", []))
                if old.get("unknown_speaker_group"):
                    held_ids.add(("unknown_speaker_group", old["unknown_speaker_group"]))
                if old.get("sha256"):
                    held_hashes.add(old["sha256"])
                if old.get("original_sha256"):
                    held_hashes.add(old["original_sha256"])
    for row in rows:
        if any((key, row[key]) in held_ids for key in _IDENTITIES if row.get(key)):
            raise ValueError("An existing held-out source/speaker/recording/room cannot be prepared for training")
        if (any(("canonical_speaker_id", speaker) in held_ids for speaker in row.get("canonical_speaker_ids", []))
                or ("unknown_speaker_group", row.get("unknown_speaker_group")) in held_ids):
            raise ValueError("An existing held-out secondary/unknown speaker group cannot be prepared for training")
    return held_hashes


def _convert(source, output, block_frames=65536):
    """Bounded, phase-aligned polyphase resampling with full FIR overlap.

    Each output block is evaluated with the neighbouring input support used by
    scipy's default filter. No independently resampled block boundary is joined.
    The only sample changes are channel averaging, resampling and PCM16 rounding;
    no per-recording gain, denoising or silence trimming is applied.
    """
    temporary = output.with_suffix(".partial.wav")
    try:
        with sf.SoundFile(source) as original:
            rate, frames, channels = original.samplerate, original.frames, original.channels
            if frames < 1 or rate < 1 or channels < 1:
                raise ValueError("Empty source audio")
            divisor = gcd(rate, 16000)
            up, down = 16000 // divisor, rate // divisor
            chunk = max(down, block_frames // down * down)
            guard = ceil(10 * max(up, down) / up) + 2 * down
            with sf.SoundFile(temporary, "w", samplerate=16000, channels=1, subtype="PCM_16") as target:
                for start in range(0, frames, chunk):
                    end = min(frames, start + chunk)
                    left = max(0, (start - guard) // down * down)
                    right = min(frames, ceil((end + guard) / down) * down)
                    original.seek(left)
                    samples = original.read(right - left, dtype="float64", always_2d=True)
                    if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0) > 1:
                        raise ValueError("Nonfinite/out-of-range source samples")
                    mono = samples.mean(axis=1)
                    resampled = resample_poly(mono, up, down) if rate != 16000 else mono
                    a = start * up // down - left * up // down
                    b = ceil(end * up / down) - left * up // down
                    target.write(pcm16(resampled[a:b]))
        os.replace(temporary, output)
        return {"input_sample_rate": rate, "input_channels": channels, "input_frames": frames}
    finally:
        temporary.unlink(missing_ok=True)


def _components(rows, *, include_people=True):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        fields = (*_IDENTITIES, "sha256", "original_sha256", "unknown_speaker_group") if include_people else (
            "canonical_source_id", "canonical_recording_id", "sha256", "original_sha256")
        keys = [(key, row[key]) for key in fields if row.get(key)]
        if include_people:
            keys.extend(("canonical_speaker_id", speaker) for speaker in row.get("canonical_speaker_ids", []))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[find(i)].append(row)
    return list(grouped.values())


def _source_interval(row):
    duration = row["duration_s"]
    start, end = row.get("start_s", 0), row.get("end_s", duration)
    if (type(start) not in (int, float) or type(end) not in (int, float)
            or not isfinite(start) or not isfinite(end) or not 0 <= start < end <= duration + 1e-9):
        raise ValueError("Source interval must be finite and within the original audio")
    return float(start), min(float(end), duration)


def _unique_seconds(rows):
    seen, intervals = set(), defaultdict(list)
    for row in rows:
        if row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        original, interval = coverage_interval(row, *_source_interval(row))
        intervals[original].append(interval)
    return union_hours(intervals) * 3600


def _representatives(rows, seed, count):
    """Prefer different provider labels/known speakers/rooms before repeats."""
    ordered = sorted(rows, key=lambda r: identity([seed, "preview", r["canonical_source_id"]]))
    selected, signatures = [], set()
    for row in ordered:
        signature = identity([row.get("class_names"), row.get("language"),
                              row.get("canonical_speaker_id"), row.get("canonical_speaker_ids"),
                              row.get("canonical_room_id")])
        if signature not in signatures:
            selected.append(row)
            signatures.add(signature)
        if len(selected) == count:
            return selected
    selected_ids = {r["canonical_source_id"] for r in selected}
    return (selected + [r for r in ordered if r["canonical_source_id"] not in selected_ids])[:count]


def _audit_groups(project, rows):
    """Also protect declared secondary speakers and unknown speaker groups."""
    root, config = load_project(project)
    from itertools import chain
    existing = chain.from_iterable(iter_rows(resolved(root, manifest)) for manifest in config["manifests"].values())
    with manifest_index(root) as index:
        for row in chain(existing, rows):
            if row.get("split") is None:
                continue
            for key in (*_IDENTITIES, "sha256", "original_sha256", "unknown_speaker_group"):
                if row.get(key):
                    index.partition(key, row[key], row["split"])
            for speaker in row.get("canonical_speaker_ids", []):
                index.partition("canonical_speaker_id", speaker, row["split"])
        if index.leaks():
            raise ValueError("Source/recording/speaker/room group leaks across Train and held-out splits")


def _partition(rows, validation_fraction, seed, target_hours):
    train, reserve = [], []
    groups = _components(rows)
    ordered = sorted(groups, key=lambda group: identity([seed, sorted(r["canonical_source_id"] for r in group)]))
    reserve_goal = _unique_seconds(rows) * validation_fraction
    reserved = 0.0
    for group in ordered:
        declared = {r.get("declared_split") for r in group} - {None}
        if len(declared) > 1:
            raise ValueError("Source/speaker/recording/room group has contradictory declared splits")
        split = next(iter(declared), "validation" if reserved < reserve_goal and len(ordered) > 1 else "train")
        selected = [{**r, "split": split} for r in group]
        if split == "validation":
            reserve.extend(selected)
            reserved += _unique_seconds(group)
        else:
            train.extend(selected)
    selected, unused, duration = [], [], 0.0
    for group in _components(train, include_people=False):
        if target_hours is None or duration < target_hours * 3600:
            selected.extend(group)
            duration += _unique_seconds(group)
        else:
            unused.extend(group)
    return selected, reserve, unused


def _source_groups_until(rows, quota):
    """Select complete recordings within an already fixed source partition."""
    chosen, seconds = [], 0.0
    for group in _components(rows, include_people=False):
        if quota is None or seconds < quota or any(r["category"] == "rir" for r in group):
            chosen.extend(group)
            seconds += _unique_seconds([r for r in group if r["category"] != "rir"])
    return chosen


def _prepare_source(root, output, row):
    source_path = Path(row["path"])
    key = identity([row["canonical_source_id"], row["original_sha256"]])
    wav, record = output / "audio" / (key + ".wav"), output / "checks" / (key + ".json")
    if record.exists():
        saved = read_json(record)
        if saved.get("original_sha256") != row["original_sha256"]:
            raise ValueError("Resume source binding changed")
        if saved["state"] == "passed":
            bound_file(root, saved["row"])
        return saved
    if shutil.disk_usage(output).free < ceil(row["duration_s"] * 32000) * 2 + 65536:
        raise ValueError("Insufficient project disk space before selected recording conversion")
    try:
        if wav.exists():
            wav.unlink()  # Owned, unfinished deterministic output only.
        original_info = _convert(source_path, wav)
        checks = _technical_check(wav, row["category"], start_s=row.get("start_s", 0), end_s=row.get("end_s"))
        if row["category"] == "rir":
            start, end = _source_interval(row)
            _technical_check(wav, "rir", start_s=start, end_s=min(end, start + 2))
        if digest(source_path) != row["original_sha256"]:
            raise ValueError("Original source changed while converting")
        prepared = {**row, "original_path": row["path"], "path": str(wav),
                    "sha256": digest(wav), "technical_pass": True,
                    "technical_qc": {**checks, **original_info}, "duration_s": checks["frames"] / 16000,
                    "individual_human_listening": False, "original_audio_opened_for_decode": True,
                    "preparation_state": "decoded_and_technical_checked"}
        _source_interval(prepared)
        saved = {"state": "passed", "original_sha256": row["original_sha256"], "row": prepared}
    except (ValueError, OSError, RuntimeError) as error:
        wav.unlink(missing_ok=True)
        saved = {"state": "rejected", "original_path": str(source_path),
                 "original_sha256": row["original_sha256"], "canonical_source_id": row["canonical_source_id"],
                 "reason": str(error), "original_preserved": source_path.exists()}
    write_json(record, saved)
    return saved


def _decode_selected(root, output, train_candidates, reserve_candidates, target_hours, validation_fraction):
    """Decode the quota, replacing unusable or duplicate Train recordings."""
    train, reserve, rejected, processed = [], [], [], set()
    seconds, seen = 0.0, {}
    for split, candidates, quota in (
            ("train", train_candidates, None if target_hours is None else target_hours * 3600),
            ("validation", reserve_candidates, None if target_hours is None else target_hours * 3600 * validation_fraction / (1 - validation_fraction))):
        seconds = 0.0
        for group in _components(candidates, include_people=False):
            if quota is not None and seconds >= quota and not any(r["category"] == "rir" for r in group):
                continue
            successful, new_unique = [], []
            for row in group:
                processed.add(row["sample_id"])
                result = _prepare_source(root, output, row)
                if result["state"] != "passed":
                    rejected.append(result)
                    continue
                prepared = result["row"]
                previous = seen.get(prepared["sha256"])
                if previous is not None:
                    if previous["split"] != split:
                        raise ValueError("Same normalized PCM appears across Train and Validation; fix source identities in a new preparation")
                    if (previous["category"], previous["language"], previous["speech"]) != (prepared["category"], prepared["language"], prepared["speech"]):
                        raise ValueError("Exact duplicate normalized audio has conflicting category/language metadata")
                else:
                    seen[prepared["sha256"]] = prepared
                    new_unique.append(prepared)
                successful.append(prepared)
            seconds += _unique_seconds([r for r in new_unique if r["category"] != "rir"])
            (train if split == "train" else reserve).extend(successful)
    unused = [{**r, "technical_pass": False, "preparation_state": "metadata_and_original_hash_only",
               "individual_human_listening": False, "training_admitted": False,
               "original_audio_opened_for_decode": False} for r in train_candidates + reserve_candidates if r["sample_id"] not in processed]
    return train, reserve, unused, rejected


def create_source_index(input_directory, *, output, dataset="manifest", metadata=None,
                        dataset_id=None, source_url=None, license=None,
                        ontology=None, room_metadata=None):
    """Translate provider metadata into the common per-file preparation index.

    ``zeroth`` reads the corpus's supplied ``*.trans.txt`` files and actual
    speaker directory names. ``manifest`` accepts provider JSONL rows or JSON
    ``files``/``tracks`` records, including the AudioSet/FMA/MIT download receipts.
    It preserves declared categories, languages and actual room/speaker IDs;
    absent metadata must still be supplied when preparing the audio folder.
    """
    directory, destination = Path(input_directory).resolve(), Path(output).resolve()
    if not directory.is_dir() or destination.is_relative_to(directory):
        raise ValueError("Use an existing audio directory and a separate index output")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO)
    rows, provider_details = [], {}
    if dataset == "zeroth":
        captions = {}
        for text in directory.rglob("*.trans.txt"):
            for line in text.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2 or parts[0] in captions:
                    raise ValueError("Zeroth provider transcript IDs must be unique and nonempty")
                captions[parts[0]] = parts[1]
        if not captions:
            raise ValueError("Zeroth needs its original provider *.trans.txt metadata")
        for path in paths:
            if path.stem not in captions:
                raise ValueError("Audio file is missing its Zeroth provider transcript")
            speaker = path.stem.split("_", 1)[0]
            if path.parent.name != speaker:
                raise ValueError("Zeroth speaker directory and utterance ID do not agree")
            row = {"path": path.relative_to(directory).as_posix(),
                         "dataset_id": dataset_id or "zeroth_korean", "source_url": source_url,
                         "license": license, "category": "competing_speech", "speech": True,
                         "language": "Korean", "category_basis": "Zeroth provider speech corpus metadata",
                         "canonical_source_id": "zeroth:utterance:" + path.stem,
                         "canonical_recording_id": "zeroth:utterance:" + path.stem,
                         "canonical_speaker_id": "zeroth:speaker:" + speaker,
                         "text": captions[path.stem], "transcript_origin": "provider_metadata_not_STT"}
            partition_names = set(path.relative_to(directory).parts) | {directory.name}
            if "test_data_01" in partition_names:
                row["split"] = "validation"
            elif "train_data_01" in partition_names:
                row["split"] = "train"
            rows.append(row)
    elif dataset in ("audioset", "fma", "mit_rir"):
        from .provider_index import build_provider_index
        prepared = build_provider_index(directory, dataset=dataset, metadata=metadata,
                         ontology=ontology, room_metadata=room_metadata, dataset_id=dataset_id,
                         source_url=source_url, license=license)
        rows = prepared["rows"]
        provider_details = {key: value for key, value in prepared.items() if key != "rows"}
    elif dataset == "manifest":
        if metadata is None:
            raise ValueError("Provider manifest/receipt is required")
        metadata = Path(metadata).resolve()
        if metadata.suffix.lower() == ".jsonl":
            supplied = read_rows(metadata)
        elif metadata.suffix.lower() == ".json":
            content = read_json(metadata)
            supplied = content if isinstance(content, list) else content.get("files", content.get("tracks"))
        else:
            raise ValueError("Provider metadata must be JSON or JSONL")
        if not isinstance(supplied, list):
            raise ValueError("Provider receipt needs files or tracks records")
        available = {str(p.resolve()): p for p in paths}
        for record in supplied:
            path = resolved(directory, record["path"])
            if str(path) not in available:
                if path.suffix.lower() in SUPPORTED_AUDIO:
                    raise ValueError("Provider audio path does not belong to input_directory")
                continue
            entry = {**record, "path": path.relative_to(directory).as_posix()}
            for key, value in (("dataset_id", dataset_id), ("source_url", source_url), ("license", license)):
                if value is not None:
                    entry.setdefault(key, value)
            # An FMA track ID is supplied identity; it is not an inferred label.
            if record.get("track_id") is not None:
                entry.setdefault("canonical_source_id", "fma:track:" + str(record["track_id"]))
            rows.append(entry)
    else:
        raise ValueError("Source index adapter must be zeroth, manifest, audioset, fma or mit_rir")
    if not rows or {r["path"] for r in rows} != {p.relative_to(directory).as_posix() for p in paths}:
        raise ValueError("Provider metadata must cover every discovered audio file exactly")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Provider metadata repeats an audio path")
    _publish_rows(destination, sorted(rows, key=lambda r: r["path"]))
    return {**provider_details, "source_index": _binding(destination), "source_count": len(rows),
            "provider_metadata_used": True, "independent_STT_runs": 0}


def _fixed_windows(rows):
    for row in rows:
        start, end = _source_interval(row)
        for index in range(ceil((end - start) / 2)):
            a = start + index * 2
            yield {**row, "sample_id": row["sample_id"] + f"_window{index:06d}",
                   "start_s": a, "end_s": min(end, a + 2),
                   "window_policy": "fixed_2s_no_audio_copy"}


def _manifest_rows(pool, rows):
    return _fixed_windows(rows) if pool in ("generic_negative", "korean_negative") else iter(rows)


def _manifest_bytes(pool, rows):
    import json
    return sum((len(json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 256)
               * (ceil((_source_interval(row)[1] - _source_interval(row)[0]) / 2)
                  if pool in ("generic_negative", "korean_negative") else 1) for row in rows)


def _rir_demonstration(output, source):
    """A fixed synthetic chirp through the actual filter, for listening only."""
    dry_path = output / "previews" / "rir_dry_chirp.wav"
    if not dry_path.exists():
        time = np.arange(12000, dtype=np.float64) / 16000
        phase = 2 * np.pi * (200 * time + (1700 - 200) / (2 * .75) * time * time)
        dry = .08 * np.sin(phase) * np.hanning(len(time))
        sf.write(dry_path, pcm16(dry), 16000, subtype="PCM_16")
    dry, rate = sf.read(dry_path, dtype="float64")
    if rate != 16000:
        raise ValueError("Frozen RIR demonstration dry audio changed")
    start, end = _source_interval(source)
    with sf.SoundFile(source["path"]) as impulse_file:
        impulse_file.seek(round(start * 16000))
        impulse = impulse_file.read(round(min(2, end - start) * 16000), dtype="float64")
    norm = float(np.sqrt(np.dot(impulse, impulse)))
    if norm <= 1e-8:
        raise ValueError("RIR training interval must contain a usable impulse")
    wet = np.r_[np.zeros(3200), fftconvolve(dry, impulse / norm), np.zeros(3200)]
    peak, level = float(np.max(np.abs(wet))), float(np.sqrt(np.mean(wet * wet)))
    gain = min(1.0, .8 / peak, .1 / level)
    construction = {"kind": "synthetic_chirp_room_response_demonstration", "dry_source": _binding(dry_path),
                    "rir_source": _binding(source["path"]), "rir_start_s": start,
                    "rir_end_s": start + len(impulse) / 16000, "rir_l2_normalization": norm, "gain": gain,
                    "training_material": False, "scored_evaluation_material": False}
    return pcm16(wet * gain), construction


def prepare_public_audio(project, *, input_directory, output, dataset_id=None,
                         source_url=None, license=None, category=None,
                         category_basis=None, language="unknown", speech=None,
                         source_index=None, target_hours=None, validation_fraction=.1,
                         seed=20260928, representatives_per_stratum=3,
                         preview_seconds=10):
    """Freeze usable Train originals, held-out reserve and one review queue.

    Supply folder-wide provider metadata, or override each audio file through a
    CSV/JSONL source index (relative path plus the same field names). Unknown
    speakers stay together at dataset level; they are never invented as distinct
    speakers. Target hours select whole recordings within fixed source partitions
    and may overshoot by the final recording. A short
    pool is reported as insufficient rather than padded or silently duplicated.
    Repeat exactly the same call to resume; changed inputs require a new output.
    """
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; prepare new training material in a new project")
    directory, output = Path(input_directory).resolve(), resolved(root, output)
    if not directory.is_dir() or not output.is_relative_to(root) or output == root:
        raise ValueError("Input folder must exist; output must be a directory inside the project")
    if output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("Input and preparation output directories must not overlap")
    if (type(validation_fraction) not in (int, float) or not isfinite(validation_fraction)
            or not 0 <= validation_fraction < 1):
        raise ValueError("validation_fraction must be finite and between zero and one")
    if type(seed) is not int or seed < 0 or type(representatives_per_stratum) is not int or representatives_per_stratum < 1:
        raise ValueError("Seed must be nonnegative and representative count positive")
    if type(preview_seconds) not in (int, float) or not isfinite(preview_seconds) or not 0 < preview_seconds <= 60:
        raise ValueError("preview_seconds must be between zero and 60")
    if target_hours is not None and (type(target_hours) not in (int, float) or not isfinite(target_hours) or target_hours <= 0):
        raise ValueError("target_hours must be positive and finite")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO)
    if not paths:
        raise ValueError("No supported audio files were found")
    if any(not p.resolve().is_relative_to(directory) for p in paths):
        raise ValueError("Input audio symlink escapes input_directory")
    defaults = {k: v for k, v in dict(dataset_id=dataset_id, source_url=source_url,
                    license=license, category=category, category_basis=category_basis,
                    language=language, speech=speech).items() if v is not None}
    rows = _metadata(paths, directory, defaults, _read_index(source_index))
    caption_screened, caption_excluded = 0, 0
    for row in rows:
        # Reuse only actual provider transcripts. A class/track label stored in
        # text is not a transcript, and the wake name without its suffix is not
        # substituted for the full configured wakeword.
        caption = row.get("transcript")
        if caption is None and row.get("transcript_origin") == "provider_metadata_not_STT":
            caption = row.get("text")
        if isinstance(caption, str):
            caption_screened += 1
            if not row.get("excluded") and config["wakeword"] in "".join(re.findall(r"[가-힣]", caption)):
                row.update(excluded=True, exclusion_reason="Provider transcript contains the full configured wakeword; no STT performed",
                           exclusion_origin="provider_transcript", provider_transcript_checked=True)
                caption_excluded += 1
    metadata_checked = set()
    for row in rows:
        for binding in row.get("provider_metadata_bindings", []):
            key = (binding.get("path"), binding.get("sha256"))
            if key not in metadata_checked:
                bound_file(directory, binding)
                metadata_checked.add(key)
    held_hashes = _check_existing_partitions(project, rows)
    # Hash after metadata checks and before decoding. An alias with the exact
    # bytes of a known held-out file is refused before any conversion.
    for row in rows:
        row["original_sha256"] = digest(row["path"])
        if row.get("provider_sha256") is not None and row["provider_sha256"] != row["original_sha256"]:
            raise ValueError("Provider source SHA256 does not match downloaded original audio")
        if row["original_sha256"] in held_hashes:
            raise ValueError("Exact held-out audio cannot be prepared for training")
    headers, initial_rejected = [], []
    for row in rows:
        source = Path(row["path"])
        if row.get("excluded") is True:
            initial_rejected.append({"state": "provider_excluded", "canonical_source_id": row["canonical_source_id"],
                                     "original_path": str(source), "original_sha256": row["original_sha256"],
                                     "reason": row["exclusion_reason"], "original_preserved": source.exists()})
            continue
        try:
            header = sf.info(source)
            if header.frames < 1 or header.samplerate < 1 or header.channels < 1:
                raise ValueError("Empty or invalid source audio header")
            key = identity([row["canonical_source_id"], row["original_sha256"]])
            candidate = {**row, "sha256": row["original_sha256"], "duration_s": ceil(header.frames * 16000 / header.samplerate) / 16000,
                         "sample_id": "public_" + key[:24], "provenance": "public_dataset", "declared_split": row.get("split"),
                         "technical_pass": False, "input_sample_rate": header.samplerate,
                         "input_frames": header.frames, "input_channels": header.channels}
            _source_interval(candidate)
            headers.append(candidate)
        except (ValueError, OSError, RuntimeError) as error:
            initial_rejected.append({"state": "header_rejected", "canonical_source_id": row["canonical_source_id"],
                                     "original_path": str(source), "original_sha256": row["original_sha256"],
                                     "reason": str(error), "original_preserved": source.exists()})
    planned_train, planned_reserve, _ = _partition(headers, validation_fraction, seed, None)
    header_plan = [{"sample_id": r["sample_id"], "split": r["split"], "duration_s": r["duration_s"],
                    "input_sample_rate": r["input_sample_rate"], "input_channels": r["input_channels"],
                    "input_frames": r["input_frames"]} for r in planned_train + planned_reserve]
    contract = {"kind": "public_audio_preparation_contract_v2", "wakeword": config["wakeword"],
                "sources": rows, "target_hours": target_hours, "validation_fraction": validation_fraction,
                "seed": seed, "representatives_per_stratum": representatives_per_stratum,
                "preview_seconds": preview_seconds, "conversion": "bounded_polyphase_mono16k_pcm16_v1",
                "selection": "metadata_header_source_partition_then_whole_recording_quota_v2",
                "header_partition_plan": header_plan,
                "quota_scope": "Non-RIR recordings; RIR filters have no duration quota"}
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output):
        contract_path = _own_output(output, "contract.json", contract)
        receipt_path = output / "preparation_receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (receipt.get("kind") != "public_audio_preparation_v1"
                    or receipt.get("wakeword") != config["wakeword"]
                    or receipt.get("contract") != _binding(contract_path)):
                raise ValueError("Frozen preparation receipt no longer binds its actual contract")
            for binding in receipt["artifacts"].values():
                bound_file(root, binding)
            for manifest in ("source_manifest", "validation_reserve", "unused_manifest", "representative_queue"):
                for row in read_rows(bound_file(root, receipt["artifacts"][manifest])):
                    bound_file(root, row)
            return {**receipt, "preparation": _binding(receipt_path), "resumed": True}
        # Reserve the selected quota's actual PCM payload. Replacement records
        # get a fresh per-source disk check if a selected source fails QC.
        required_bytes, largest = 0, 0
        expected_train = _source_groups_until(planned_train, None if target_hours is None else target_hours * 3600)
        expected_reserve = _source_groups_until(planned_reserve, None if target_hours is None else target_hours * 3600 * validation_fraction / (1 - validation_fraction))
        for row in expected_train + expected_reserve:
            key = identity([row["canonical_source_id"], row["original_sha256"]])
            if (output / "checks" / (key + ".json")).exists():
                continue
            size = ceil(row["duration_s"] * 32000) + 4096
            required_bytes += size
            largest = max(largest, size)
        expected_preview_count = min(len(headers), representatives_per_stratum * len({(r["dataset_id"], r["category"]) for r in headers}))
        estimated_bytes = required_bytes + largest + expected_preview_count * (round(preview_seconds * 32000) + 4096)
        if shutil.disk_usage(output).free < estimated_bytes:
            raise ValueError(f"Insufficient project disk space: estimated {estimated_bytes} bytes required")
        for child in ("audio", "checks", "previews"):
            (output / child).mkdir(exist_ok=True)
        train, reserve, unused, rejected = _decode_selected(root, output, planned_train, planned_reserve, target_hours, validation_fraction)
        rejected = initial_rejected + rejected
        converted = train + reserve
        if not converted:
            write_json(output / "rejected.json", {"sources": rejected, "duplicates": []})
            raise ValueError("All sources failed technical checks; see checks records")
        # Normalize exact waveform duplicates before counting hours. Conflicting
        # metadata on the same PCM is an input error, not an inferred category.
        duplicates, unique = [], {}
        for row in converted:
            if row["sha256"] in unique:
                first = unique[row["sha256"]]
                if (row["category"], row["language"], row["speech"]) != (first["category"], first["language"], first["speech"]):
                    raise ValueError("Exact duplicate normalized audio has conflicting category/language metadata")
                duplicates.append({"canonical_source_id": row["canonical_source_id"], "duplicate_of": first["canonical_source_id"], "sha256": row["sha256"]})
            else:
                unique[row["sha256"]] = row
        # Partition was fixed before decoding. Audit aliases before publication;
        # normalized duplicates cannot silently move a reserved speaker to Train.
        _audit_groups(project, train + reserve + unused)
        canonical = {r["canonical_source_id"] for r in unique.values()}
        train, reserve = ([r for r in collection if r["canonical_source_id"] in canonical]
                          for collection in (train, reserve))
        if not train:
            raise ValueError("No Train sources after conservative group splitting; adjust explicit metadata or validation fraction")
        check = audit_project(project, check_files=False, extra_rows=train + reserve + unused)
        if not check["passed"]:
            raise ValueError("Prepared source split audit failed: " + str(check["errors"][:5]))
        preview_rows = []
        strata = defaultdict(list)
        for row in train:
            strata[(row["dataset_id"], row["category"])].append(row)
        for stratum, members in sorted(strata.items()):
            for source in _representatives(members, seed, representatives_per_stratum):
                path = output / "previews" / (source["sample_id"] + ".wav")
                with sf.SoundFile(source["path"]) as stream:
                    start, end = _source_interval(source)
                    lo, hi = round(start * 16000), round(end * 16000)
                    count = min(hi - lo, round(preview_seconds * 16000))
                    center = lo + max(0, (hi - lo - count) // 2)
                    peak_offset = max(lo, min(hi - count, source["technical_qc"]["peak_frame"] - 800))
                    offset = peak_offset if source["category"] == "rir" else center
                    stream.seek(offset)
                    samples = stream.read(count, dtype="int16")
                    if not np.any(samples):
                        offset = peak_offset
                        stream.seek(offset)
                        samples = stream.read(count, dtype="int16")
                sf.write(path, samples, 16000, subtype="PCM_16")
                construction = None
                if source["category"] == "rir":
                    samples, construction = _rir_demonstration(output, source)
                    sf.write(path, samples, 16000, subtype="PCM_16")
                category_name = {"environment": "환경 소리", "music": "음악", "competing_speech": "다른 사람 말소리", "rir": "공간 울림 · 시험음에 잔향 적용"}[source["category"]]
                preview_rows.append({"sample_id": "preview_" + source["sample_id"], "path": str(path),
                    "sha256": digest(path), "split": "train", "canonical_source_id": source["canonical_source_id"],
                    "dataset_id": source["dataset_id"], "category": source["category"],
                    "language": source["language"], "reviewed_source": _binding(source["path"]),
                    "preview_start_s": offset / 16000, "preview_end_s": (offset + count) / 16000,
                    "preview_selection": "fixed_synthetic_chirp_with_rir" if source["category"] == "rir" else "center_or_non_silent_peak",
                    "review_group": source["dataset_id"] + " · " + category_name,
                    "text": source.get("text") or category_name,
                    "review_instruction": ("고정 시험음에 공간 울림을 적용했습니다. 울림은 정상이며, 심한 깨짐·잘림만 폐기하고 판단이 어려우면 재검수."
                                           if source["category"] == "rir" else
                                           "표시된 자료 종류와 맞고 소리가 정상적이면 보존. 호출어가 들리거나 깨졌으면 폐기, 판단이 어려우면 재검수.")})
                if construction is not None:
                    preview_rows[-1]["preview_construction"] = construction
        manifest_paths = {"source_manifest": output / "train_sources.jsonl", "validation_reserve": output / "validation_reserve.jsonl",
                          "unused_manifest": output / "unused_sources.jsonl", "representative_queue": output / "representatives.jsonl"}
        for key, values in zip(manifest_paths, (train, reserve, unused, preview_rows)):
            _publish_rows(manifest_paths[key], values)
        hours = _unique_seconds(train) / 3600
        write_json(output / "rejected.json", {"sources": rejected, "duplicates": duplicates})
        receipt = {"kind": "public_audio_preparation_v1", "state": "awaiting_human_review", "wakeword": config["wakeword"],
                   "created_at": now(), "contract": _binding(contract_path),
                   "artifacts": {**{key: _binding(path) for key, path in manifest_paths.items()},
                                 "rejections": _binding(output / "rejected.json")},
                   "input_files": len(rows), "technical_passed_files": len(converted), "unique_audio_files": len(unique),
                   "rejected_files": len(rejected), "excluded_provider_files": sum(r["state"] == "provider_excluded" for r in rejected),
                   "provider_transcript_screened_sources": caption_screened, "provider_transcript_excluded_sources": caption_excluded,
                   "duplicates_excluded": len(duplicates),
                   "train_source_count": len(train), "train_hours": hours,
                   "validation_source_count": len(reserve), "validation_hours": _unique_seconds(reserve) / 3600,
                   "unused_source_count": len(unused), "representative_count": len(preview_rows),
                   "waveform_decoded_files": len(converted) + sum(r["state"] == "rejected" for r in rejected),
                   "unselected_waveforms_not_decoded": len(unused),
                   "target_hours": target_hours, "target_met": target_hours is None or _unique_seconds([r for r in train if r["category"] != "rir"]) / 3600 + 1e-12 >= target_hours,
                   "target_scope": "Non-RIR recordings only; complete recordings selected inside fixed source partitions",
                   "selection_policy": "metadata_header_source_partition_then_whole_recording_quota_v2",
                   "normalized_duplicate_scope": "Selected decoded Train/Validation; every input original file hash and declared identity is checked before decoding",
                   "estimated_required_bytes": estimated_bytes, "output_directory": str(output),
                   "human_listened_count": 0, "training_admitted": False, "independent_STT_runs": 0, "Final_accessed": False,
                   "unknown_speaker_grouping": "One conservative group per dataset; no invented speaker IDs",
                   "next": "Listen using koww review with representative_queue and source_manifest, then admit-materials"}
        if (output / "previews" / "rir_dry_chirp.wav").exists():
            receipt["artifacts"]["rir_demo_dry"] = _binding(output / "previews" / "rir_dry_chirp.wav")
        write_json(receipt_path, receipt)
        return {**receipt, "preparation": _binding(receipt_path), "resumed": False}


def apply_prepared_review(project, *, preparation, answers=None, output=None,
                          pools=(), connect=False):
    """Admit exact Train hashes and optionally connect only empty pool manifests.

    Saved representative answers must include explicit whole-pool consent. A
    discarded representative excludes that original only, never an inferred
    category or the unseen pool. Review-held answers block admission. Evaluation
    reserves remain separate and are not added to any training manifest.
    """
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; material admission cannot alter training inputs")
    if isinstance(preparation, dict):
        preparation_path = bound_file(root, preparation)
    else:
        preparation_path = resolved(root, preparation)
    record = read_json(preparation_path)
    if record.get("kind") != "public_audio_preparation_v1" or record.get("wakeword") != config["wakeword"]:
        raise ValueError("Preparation is not for this project's word")
    source_path = bound_file(root, record["artifacts"]["source_manifest"])
    queue_path = bound_file(root, record["artifacts"]["representative_queue"])
    bound_file(root, record["artifacts"]["validation_reserve"])
    answer_path = resolved(root, answers or "review/pool_results.json")
    saved = read_json(answer_path)
    previews = read_rows(queue_path)
    decisions = saved.get("decisions", {})
    if saved.get("whole_pool_policy_approved") is not True:
        raise ValueError("Explicit whole-pool policy consent is required")
    if set(decisions) != {r["sample_id"] for r in previews}:
        raise ValueError("Saved answers must cover this exact representative queue")
    if any(a.get("decision") == "review" for a in decisions.values()):
        raise ValueError("Review-held representatives need resolution before admission")
    excluded = sorted({r["canonical_source_id"] for r in previews if decisions[r["sample_id"]].get("decision") == "discard"})
    if isinstance(pools, (str, bytes)) or set(pools) - TRAIN_POOLS or len(set(pools)) != len(pools):
        raise ValueError("Pools must be unique generic_negative, korean_negative, noise or rir names")
    sources = [r for r in read_rows(source_path) if r["canonical_source_id"] not in excluded]
    selected_pools = {}
    for pool in pools:
        selection = [r for r in sources if (r["category"] == "rir") == (pool == "rir")]
        if pool == "korean_negative":
            selection = [r for r in selection if r["speech"] and r["language"] in ("ko", "kor", "Korean")]
        if not selection:
            raise ValueError(f"No declared eligible Train sources for {pool}")
        selected_pools[pool] = selection
    output = resolved(root, output or (preparation_path.parent / "admitted"))
    if not output.is_relative_to(root) or output == root:
        raise ValueError("Admission output must stay inside the project")
    output.mkdir(parents=True, exist_ok=True)
    def finish(result, resumed):
        result = {**result, "resumed": resumed, "receipt": _binding(output / "admission_receipt.json")}
        if connect:
            connection = connect_materials(project, admissions=[result["receipt"]], pools=pools)
            result.update(connected=True, connection=connection)
        return result
    with writer_lock(output):
        admission_path = output / "evidence" / "admission.json"
        receipt_path = output / "admission_receipt.json"
        binding = _binding(answer_path)
        request = {"preparation": _binding(preparation_path), "answers": binding, "pools": list(pools)}
        _own_output(output, "request.json", request)
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["request"] != request:
                raise ValueError("Admission request changed; choose a new output directory")
            bound_file(root, receipt["admission"])
            for item in receipt["manifests"].values():
                bound_file(root, item)
            for item in receipt.get("source_manifests", {}).values():
                bound_file(root, item)
            return finish(receipt, True)
        if admission_path.exists():
            existing = read_json(admission_path)
            if existing.get("human_answers") != binding or existing.get("source_manifest") != record["artifacts"]["source_manifest"]:
                raise ValueError("Interrupted admission evidence changed; choose a new output")
            admission = {"path": admission_path.relative_to(root).as_posix(), "sha256": digest(admission_path)}
        else:
            approval = approve_pool(project, source_manifest=record["artifacts"]["source_manifest"],
                                    representative_queue=record["artifacts"]["representative_queue"],
                                    answers=binding, output=admission_path,
                                    excluded_source_ids=excluded)
            admission = approval["admission"]
        outputs, source_outputs = {}, {}
        required_manifest_bytes = sum(_manifest_bytes(pool, selected) for pool, selected in selected_pools.items())
        if shutil.disk_usage(output).free < required_manifest_bytes * 2:
            raise ValueError(f"Insufficient project disk space for material manifests: estimated {required_manifest_bytes * 2} bytes")
        for pool, selected in selected_pools.items():
            path = output / (pool + ".jsonl")
            admitted = [{**r, "admission": admission} for r in selected]
            original_manifest = output / (pool + "_sources.jsonl")
            _publish_rows(original_manifest, admitted)
            source_outputs[pool] = _binding(original_manifest)
            _publish_rows(path, _manifest_rows(pool, admitted))
            outputs[pool] = _binding(path)
        result = {"kind": "prepared_public_audio_admission_v1", "state": "accepted", "request": request,
                  "admission": admission, "manifests": outputs, "source_manifests": source_outputs, "connected": False,
                  "source_count": len(sources), "train_hours": _unique_seconds(sources) / 3600,
                  "excluded_source_ids": excluded, "evaluation_reserve": record["artifacts"]["validation_reserve"],
                  "Final_accessed": False, "independent_STT_runs": 0}
        write_json(receipt_path, result)
        return finish(result, False)


def connect_materials(project, *, admissions, pools, output=None):
    """Merge reviewed datasets once and connect exact, empty project manifests.

    Multiple AudioSet/music admissions can share a noise pool without manual
    JSON editing. This never appends to an established experiment. An interrupted
    connection can resume only when already published manifests equal the exact
    staged bytes; different existing material is left untouched.
    """
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; material connection cannot change inputs")
    if (isinstance(admissions, (str, bytes)) or not admissions
            or isinstance(pools, (str, bytes)) or not pools
            or set(pools) - TRAIN_POOLS or len(set(pools)) != len(pools)):
        raise ValueError("Supply admission receipts and unique supported training pools")
    bindings, records = [], []
    for entry in admissions:
        path = bound_file(root, entry) if isinstance(entry, dict) else resolved(root, entry)
        binding = _binding(path)
        record = read_json(path)
        if record.get("kind") != "prepared_public_audio_admission_v1" or record.get("state") != "accepted":
            raise ValueError("Material connection needs actual prepared admission receipts")
        actual_admission = read_json(bound_file(root, record["admission"]))
        if actual_admission.get("wakeword") != config["wakeword"]:
            raise ValueError("Admission belongs to another wakeword project")
        bound_file(root, record["evaluation_reserve"])
        bindings.append(binding)
        records.append(record)
    request = {"admissions": sorted(bindings, key=lambda b: b["path"]), "pools": sorted(pools),
               "project_config_sha256": digest(project)}
    directory = resolved(root, output or ("data/material_connections/" + identity(request)))
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("Connection artifacts must stay inside the project")
    merged, all_candidates = {}, []
    for pool in pools:
        rows, seen = [], {}
        for record in records:
            if pool not in record["manifests"]:
                continue
            bound_file(root, record["manifests"][pool])
            source_binding = record.get("source_manifests", {}).get(pool, record["manifests"][pool])
            candidates = read_rows(bound_file(root, source_binding))
            validate_admission(root, candidates, config, pool)
            all_candidates.extend(candidates)
            for row in candidates:
                if pool == "rir" and row.get("category") != "rir" or pool != "rir" and row.get("category") == "rir":
                    raise ValueError("RIR filters cannot be connected as ordinary Negative/noise recordings")
                key = (row["sha256"], *_source_interval(row))
                if key not in seen:
                    rows.append(row)
                    seen[key] = row
                elif (row["category"], row["language"], row["speech"]) != (seen[key]["category"], seen[key]["language"], seen[key]["speech"]):
                    raise ValueError("Duplicate merged audio has conflicting category/language metadata")
        if not rows:
            raise ValueError(f"Admission receipts contain no {pool} sources")
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Conflicting duplicate sample IDs in merged material")
        merged[pool] = rows
    # Audit both held-out reserves and Train rows together, without opening
    # Final audio. Reserve sources never become accepted training inputs.
    extra = all_candidates
    for record in records:
        extra.extend(read_rows(bound_file(root, record["evaluation_reserve"])))
    unique_audit_rows = {identity(row): row for row in extra}
    _audit_groups(project, list(unique_audit_rows.values()))
    audit_rows = [{**r, "sample_id": "connection_audit_" + str(i)} for i, r in enumerate(unique_audit_rows.values())]
    audit = audit_project(project, check_files=False, extra_rows=audit_rows)
    if not audit["passed"]:
        raise ValueError("Merged source/speaker/room split audit failed: " + str(audit["errors"][:5]))
    directory.mkdir(parents=True, exist_ok=True)
    estimated_manifest_bytes = sum(_manifest_bytes(pool, rows) for pool, rows in merged.items())
    if shutil.disk_usage(directory).free < estimated_manifest_bytes * 3:
        raise ValueError(f"Insufficient project disk space for merged manifests: estimated {estimated_manifest_bytes * 3} bytes")
    staged = {}
    with writer_lock(directory):
        _own_output(directory, "request.json", request)
        for pool, rows in merged.items():
            path = directory / (pool + ".jsonl")
            _publish_rows(path, _manifest_rows(pool, rows))
            staged[pool] = _binding(path)
    receipt_path = directory / "connection.json"
    with writer_lock(root / "data"):
        if receipt_path.exists():
            prior = read_json(receipt_path)
            if prior.get("request") != request:
                raise ValueError("Frozen connection request changed")
        destinations, pending = {}, []
        for pool, binding in staged.items():
            destination = resolved(root, config["manifests"][pool])
            if not destination.is_relative_to(root):
                raise ValueError("Connected manifests must stay inside the project")
            if destination.exists() and destination.stat().st_size:
                if digest(destination) != binding["sha256"]:
                    raise ValueError("Connect only empty manifests or exact resumed bytes; existing material is immutable")
            else:
                pending.append((destination, binding))
            destinations[pool] = {"path": str(destination), "sha256": binding["sha256"]}
        if pending and (any((root / "runs").rglob("contract.json")) or any((root / "features").rglob("contract.json"))):
            raise ValueError("A started experiment's inputs cannot be changed")
        transaction = {"kind": "prepared_material_connection_v1", "request": request,
                       "state": "connecting", "manifests": destinations, "Final_accessed": False}
        write_json(receipt_path, transaction)
        for destination, binding in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                shutil.copyfile(binding["path"], temporary)
                if digest(temporary) != binding["sha256"]:
                    raise ValueError("Staged material copy hash mismatch")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        for binding in destinations.values():
            bound_file(root, binding)
        transaction.update(state="connected", pool_rows={pool: sum(ceil((_source_interval(r)[1] - _source_interval(r)[0]) / 2) for r in rows)
                            if pool in ("generic_negative", "korean_negative") else len(rows) for pool, rows in merged.items()},
                           estimated_manifest_bytes=estimated_manifest_bytes)
        write_json(receipt_path, transaction)
        return {**transaction, "receipt": _binding(receipt_path), "resumed": not pending}
