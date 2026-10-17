"""Independent Final source checks, with explicit limits on unknown provenance.

No transcription, acoustic fingerprint search, or full corpus decoding is used.
Known IDs/hashes are checked first; only equal-length canonical PCM sources are
read for exact equality. Existing immutable evaluations are never rewritten.
"""
from collections import Counter
import hashlib
from pathlib import Path

import soundfile as sf

from .common import bound_file, digest, identity, iter_rows, read_json, resolved, write_json
from .project import load_project
from .source_identity import partition_keys
from .storage import storage_paths
from .streaming import manifest_index


def freeze_input_bindings(project, development_report):
    """Bind common/Train manifests and existing Final reservations, not empty outputs.

Final TTS may be generated after freezing. Nonempty pre-reserved Final manifests
are bound as well, so their sources cannot be swapped after inspecting a score.
"""
    root, config = load_project(project)
    bindings = {}
    for pool, value in config["manifests"].items():
        path = resolved(root, value)
        final = pool in ("positive_final", "noise_final", "rir_final", "final")
        if final and (not path.is_file() or not path.stat().st_size):
            continue
        bindings["manifest:" + pool] = {"path": str(path), "sha256": digest(path)}
    report_path = Path(development_report).resolve()
    bindings["development_report"] = {"path": str(report_path), "sha256": digest(report_path)}
    clips = report_path.parent / "clips.json"
    bindings["development_clips"] = {"path": str(clips), "sha256": digest(clips)}
    report = read_json(report_path)
    if report.get("evaluation_manifest"):
        entry = report["evaluation_manifest"]
        path = bound_file(root, entry)
        if entry["sha256"] != report.get("manifest_sha256"):
            raise ValueError("Development manifest binding differs from its report")
        bindings["external_development_manifest"] = {"path": str(path), "sha256": digest(path)}
    return bindings


def _rows(project, extra_rows):
    root, config = load_project(project)
    for pool, value in config["manifests"].items():
        for row in iter_rows(resolved(root, value)):
            yield pool, row
    if extra_rows is not None:
        for row in extra_rows:
            yield "external_final", row
    lock_path = root / "final_lock.json"
    if lock_path.is_file():
        lock = read_json(lock_path)
        # Selected/tuned Development reports keep the exact source metadata in
        # clips.json even when their original manifest was passed via --manifest.
        entry = lock.get("input_bindings", {}).get("development_clips")
        if entry:
            for row in read_json(bound_file(root, entry)):
                yield "frozen_development", {**row, "split": "validation"}


def _flatten(row):
    yield row
    for component in row.get("components", []):
        if isinstance(component, dict):
            yield from _flatten(component)


def _final_provenance_errors(pool, row):
    errors = []
    label = f"{pool}/{row.get('sample_id', '?')}"
    if row.get("kind") == "positive" and row.get("category") != "clean":
        components = row.get("components", [])
        roles = {c.get("role") for c in components if isinstance(c, dict)}
        if not {"call", "background"} <= roles:
            errors.append(label + ": Final mixtures need explicit call and background source components")
    if row.get("rir_applied") is True or row.get("rir_source_id") or row.get("rir_sha256"):
        if not any(c.get("role") == "rir" for c in row.get("components", []) if isinstance(c, dict)):
            errors.append(label + ": applied Final RIR needs its own source/room lineage")
    for component in _flatten(row):
        if component is row and row.get("components"):
            continue  # The constructed mixture is not a separate public dataset.
        if component.get("split") != "final":
            errors.append(label + ": every original Final component must belong to the reserved final split")
        source = component.get("canonical_source_id", component.get("source_id"))
        if not source or not component.get("sha256") or not component.get("path"):
            errors.append(label + ": Final sources need canonical source identity, original audio path and an exact file hash")
        provenance = component.get("provenance")
        if provenance == "qwen_tts":
            if not component.get("voice_family_id") or component.get("human_reference_used") is not False:
                errors.append(label + ": Final synthetic source needs a separate Qwen-only voice family")
        elif provenance == "human" and component.get("role", "call") == "call":
            if not (component.get("canonical_speaker_id") or component.get("speaker_id")):
                errors.append(label + ": Final human call needs its recorded speaker identity")
        elif provenance != "public_dataset" or not component.get("source_url") or not component.get("license"):
            errors.append(label + ": Final background/Negative/RIR source URL, license and public provenance required")
    return errors


def _pcm(path, declared, root):
    """Hash verified whole PCM in bounded blocks; container metadata is irrelevant."""
    path = Path(path)
    state = path.stat()
    source = {"sha256": declared, "bytes": state.st_size, "mtime_ns": state.st_mtime_ns}
    destination = storage_paths(root)["root"] / "source_pcm_audit" / (identity([str(path), declared]) + ".json")
    if destination.is_file():
        cached = read_json(destination)
        if cached.get("source") == source:
            return cached["pcm_sha256"], True, 0
    bound_file(root, {"path": str(path), "sha256": declared})
    h, size = hashlib.sha256(), 0
    with sf.SoundFile(path) as stream:
        for block in stream.blocks(blocksize=65536, dtype="int16"):
            payload = block.astype("<i2", copy=False).tobytes()
            h.update(payload)
            size += len(payload)
    value = h.hexdigest()
    write_json(destination, {"source": source, "pcm_sha256": value,
                             "scope": "exact_full_source_16khz_mono_pcm16"})
    return value, False, size


def audit_independent_final(project, extra_rows=None, check_files=True):
    """Reject known Train/common overlap; never claim unknown mapping is verified."""
    root, _ = load_project(project)
    errors, unknown = [], Counter()
    counts, pcm_counts = Counter(), Counter()
    final_lengths = set()
    final_paths, known_paths = {}, {}
    with manifest_index(root) as index:
        for pool, row in _rows(project, extra_rows):
            split = row.get("split")
            if pool in ("positive_final", "noise_final", "rir_final", "final", "external_final") and split != "final":
                errors.append(f"{pool}: independent Final input has the wrong split")
            if split not in ("train", "validation", "final"):
                continue
            partition = "final" if split == "final" else "train_or_common"
            counts[partition] += 1
            for kind, value in partition_keys(row):
                index.partition(kind, value, partition)
            if split == "final":
                errors.extend(_final_provenance_errors(pool, row))
            for component in _flatten(row):
                if split == "final":
                    if (component.get("category") == "competing_speech" or component.get("speech") is True
                            or component.get("language")) and not (component.get("canonical_speaker_id") or component.get("canonical_speaker_ids")):
                        unknown["speaker_identity"] += 1
                    if component.get("role") == "rir" and not component.get("canonical_room_id"):
                        unknown["room_identity"] += 1
                    if component.get("category") == "music" and not (component.get("canonical_artist_id") or component.get("canonical_artist_ids")):
                        unknown["artist_identity"] += 1
                if check_files and component.get("path") and component.get("sha256"):
                    key = (str(resolved(root, component["path"])), component["sha256"])
                    (final_paths if split == "final" else known_paths).setdefault(key, component)
        # Disk-backed identity audit works even when original audio is unavailable.
        leaks = index.leaks()
    errors.extend(f"Independent Final {row['key']} overlaps Train/common evaluation: {row['id']}" for row in leaks)
    if not counts["final"]:
        errors.append("No independent Final sources are connected")
    # Do not decode any audio once its declared lineage already proves overlap.
    if check_files and not errors:
        final_pcm = set()
        for key, row in final_paths.items():
            try:
                path = bound_file(root, row)
                info = sf.info(path)
                if (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16"):
                    unknown["noncanonical_pcm_sources"] += 1
                    continue
                value, reused, size = _pcm(path, row["sha256"], root)
                final_pcm.add((value, info.frames)); final_lengths.add(info.frames)
                pcm_counts["cache_hits"] += reused; pcm_counts["bytes_read"] += size
                pcm_counts["final_sources_checked"] += 1
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append(f"Independent Final source verification failed: {exc}")
        for (value, _), row in known_paths.items():
            try:
                info = sf.info(value)
                if (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16") or info.frames not in final_lengths:
                    continue
                value, reused, size = _pcm(value, row["sha256"], root)
                pcm_counts["cache_hits"] += reused; pcm_counts["bytes_read"] += size
                pcm_counts["known_same_length_sources_checked"] += 1
                if (value, info.frames) in final_pcm:
                    errors.append("Independent Final PCM duplicates a known Train/common original: " + str(row.get("canonical_source_id", row.get("source_id"))))
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append(f"Known Train/common source verification failed: {exc}")
    return {"kind": "independent_final_source_audit_v1", "passed": not errors,
            "errors": errors, "leaks": leaks, "manifest_rows": dict(counts), "pcm": dict(pcm_counts),
            "files_checked": bool(check_files),
            "unknown_metadata": dict(unknown), "generic_precomputed_original_mapping_available": False,
            "independence_scope": "Known Train/common source and component IDs, file/original/PCM hashes, voice families, declared speakers/artists/rooms, plus exact equal-length canonical full-source PCM. Unknown provenance, arbitrary excerpts and acoustic near-duplicates remain unverified; public Generic features have no raw-source mapping.",
            "independent_STT_runs": 0, "fully_proven_independent": False}
