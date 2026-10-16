"""Small, independently reserved Development noise; never a training pool."""
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
import time

import hashlib
import numpy as np
import soundfile as sf

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock
from .project import load_project

EVALUATION_RECIPE_VERSION = 2
_TYPE_NAMES = {
    "rain": "비", "wind": "바람", "vacuum_cleaner": "청소기", "washing_machine": "세탁기",
    "engine": "엔진", "train": "기차", "keyboard_typing": "키보드", "door_wood_knock": "문 두드림",
    "glass_breaking": "유리 충격·파손", "footsteps": "발걸음", "clock_alarm": "알람",
    "clock_tick": "시계 전자·기계음", "car_horn": "경적", "siren": "사이렌", "church_bells": "종소리",
    "music_instrumental_piano_drums": "피아노·드럼 연주", "music_instrumental_guitar_drums": "기타·드럼 연주",
    "music_singing": "보컬 멜로디", "music_vocal_pad": "보컬 패드 음악",
}


def evaluation_catalog():
    catalog = read_json(files("korean_openwakeword") / "resources/evaluation_noise_v2.json")
    if catalog.get("recipe_version") != EVALUATION_RECIPE_VERSION:
        raise ValueError("Unsupported held-out noise catalog")
    return catalog


def evaluation_preview():
    """Dataset summaries keep the first screen short; individual terms remain available."""
    catalog = evaluation_catalog()
    grouped = defaultdict(list)
    for row in catalog["assets"]:
        grouped[row["dataset_id"]].append(row)
    labels = {"esc50_held_out_v2": "환경음·전자음 평가 자료", "freesound_music_held_out_v2": "보컬·연주곡 평가 자료"}
    result = []
    for dataset, rows in grouped.items():
        result.append({"id": dataset, "dataset_id": dataset, "name": labels.get(dataset, dataset),
            "role": "held_out_evaluation_noise", "split": "validation", "evaluation_only": True,
            "file_count": len(rows), "type_count": len({r["eval_type"] for r in rows}),
            "eval_types": sorted({r["eval_type"] for r in rows}), "total_bytes": sum(r["size_bytes"] for r in rows),
            "source_url": "https://github.com/karolpiczak/ESC-50" if dataset.startswith("esc50") else "https://freesound.org/",
            "license": "ESC-50 CC-BY-NC-3.0 / ESC-10 CC-BY-3.0; 개별 원출처 조건 보존" if dataset.startswith("esc50") else "CC0-1.0 · 공개 HQ MP3 미리듣기",
            "individual_licenses": [{"id": r["id"], "name": _TYPE_NAMES.get(r["eval_type"], r["eval_type"]),
                "license": r["license"], "source_url": r["original_source_url"], "media_variant": r["media_variant"]} for r in rows]})
    return result


def _pcm_hash(path):
    h = hashlib.sha256()
    frames = 0
    with sf.SoundFile(path) as stream:
        if stream.samplerate != 16000 or stream.channels != 1 or stream.subtype != "PCM_16":
            raise ValueError("Noise split checks require canonical 16kHz mono PCM16")
        for samples in stream.blocks(blocksize=65536, dtype="int16"):
            h.update(samples.astype("<i2", copy=False).tobytes())
            frames += len(samples)
    return h.hexdigest(), frames


def _convert_evaluation_audio(source, destination):
    """Preserve the source and explicitly record any resampling headroom gain.

    These pinned excerpts are at most 36 seconds.  A single, short conversion
    avoids joining transient edges while leaving all Train conversion unchanged.
    """
    from .audio import read_audio, write_pcm16
    original_info = sf.info(source)
    original, _ = sf.read(source, dtype="float64", always_2d=True)
    clipped_fraction = float(np.mean(np.abs(original) >= 32760 / 32768))
    if clipped_fraction >= .01:
        raise ValueError("Excessively clipped original evaluation audio cannot be admitted")
    samples = read_audio(source)
    peak = float(np.max(np.abs(samples), initial=0))
    gain = .95 / peak if peak > 1 else 1.0
    write_pcm16(destination, samples * gain)
    return {"method": "mono_polyphase_resample_16000_pcm16",
            "resampled_peak_before_gain": peak, "uniform_gain": gain,
            "original_clipped_fraction": clipped_fraction,
            "original_format": original_info.format, "original_subtype": original_info.subtype,
            "original_clipping_scope": "decoded original media; MP3 decoding peaks do not prove clipping in the unavailable original WAV",
            "headroom_reason": "resampling_overshoot_only" if gain < 1 else "none",
            "noise_added": False, "silence_trimmed": False, "source_preserved": True}


def _known_train(project, prepared):
    root, config = load_project(project)
    manifests = [resolved(root, config["manifests"][role]) for role in
                 ("generic_negative", "korean_negative", "noise")]
    for queue in prepared.get("review_queues", []):
        if queue.get("pool_manifest") and not queue.get("evaluation_only"):
            manifests.append(Path(queue["pool_manifest"]))
    seen = set()
    for path in manifests:
        for row in read_rows(path):
            key = (row.get("sha256"), row.get("canonical_source_id"))
            if row.get("split") == "train" and key not in seen:
                seen.add(key)
                yield row


def assert_held_out(project, rows, prepared=None):
    """IDs/hash first; only equal-length source candidates need bounded PCM reads.

    This is exact full-source equality, not an ASR pass or a claim that every
    arbitrary excerpt/near-duplicate across different publishers is detectable.
    """
    root, _ = load_project(project)
    started = time.perf_counter()
    ids = {r["canonical_source_id"] for r in rows} | {r["canonical_recording_id"] for r in rows}
    hashes = {r["sha256"] for r in rows} | {r["original_sha256"] for r in rows}
    pcm = {(r["normalized_pcm_sha256"], r["normalized_frames"]) for r in rows}
    frames = {r["normalized_frames"] for r in rows}
    cache = root / ".studio" / "materials" / "source_pcm_cache"
    count = decoded = reused = bytes_read = 0
    seen = set()
    for row in _known_train(project, prepared or {}):
        count += 1
        if (row.get("canonical_source_id") in ids or row.get("canonical_recording_id") in ids
                or row.get("sha256") in hashes or row.get("original_sha256") in hashes):
            raise ValueError("Held-out evaluation audio overlaps a known raw Train source")
        if row.get("sha256") in seen:
            continue
        seen.add(row.get("sha256"))
        declared = row.get("duration_s")
        if declared is not None and round(float(declared) * 16000) not in frames:
            continue  # No decoding of unrelated long recordings.
        path = resolved(root, row["path"])
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames not in frames:
            continue
        state = path.stat()
        key = {"wav_sha256": row["sha256"], "bytes": state.st_size, "mtime_ns": state.st_mtime_ns}
        cached = cache / (row["sha256"] + ".json")
        if cached.is_file() and read_json(cached).get("source") == key:
            record = read_json(cached)
            value, length = record["pcm_sha256"], record["frames"]
            reused += 1
        else:
            bound_file(root, row)
            value, length = _pcm_hash(path)
            write_json(cached, {"source": key, "pcm_sha256": value, "frames": length,
                                "scope": "exact_full_source_16khz_mono_pcm16", "created_at": now()})
            decoded += 1
            bytes_read += length * 2
        if (value, length) in pcm:
            raise ValueError("Held-out evaluation PCM duplicates a known raw Train source")
    return {"known_train_sources_checked": count, "same_length_sources_pcm_read": decoded,
            "pcm_cache_reused": reused, "pcm_bytes_read": bytes_read,
            "elapsed_seconds": time.perf_counter() - started, "independent_STT_runs": 0,
            "scope": "Known raw Train canonical IDs, file hashes and exact equal-length full-source PCM; no near-duplicate or arbitrary-excerpt claim",
            "generic_precomputed_original_mapping_available": False}


def prepare_evaluation_materials(project, workspace, prepared, progress=None):
    from .studio_data import _download, _write_rows, _emit
    from .pool_review import _technical_check
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is locked; new Development noise belongs to a separate experiment")
    catalog = evaluation_catalog()
    catalog_id = identity(catalog)[:16]
    output = root / ".studio" / "materials" / ("evaluation_v2_" + catalog_id)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "preparation.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("catalog_id") != catalog_id or receipt.get("wakeword") != config["wakeword"]:
            raise ValueError("Frozen evaluation preparation differs from the source catalog")
        for binding in receipt["artifacts"].values():
            bound_file(root, binding)
        rows = read_rows(receipt["artifacts"]["source_manifest"]["path"])
        for row in rows:
            bound_file(root, row)
        audit = assert_held_out(project, rows, prepared)
        return {**receipt, "preparation": {"path": str(receipt_path), "sha256": digest(receipt_path)}, "split_audit": audit, "resumed": True}
    rows = []
    with writer_lock(output):
        for index, asset in enumerate(catalog["assets"]):
            source = _download(asset, Path(workspace).resolve() / "shared" / "downloads", progress)
            destination = output / "audio" / (asset["id"] + ".wav")
            destination.parent.mkdir(parents=True, exist_ok=True)
            check = output / "checks" / (asset["id"] + ".json")
            if check.exists():
                row = read_json(check)
                if row.get("original_sha256") != asset["sha256"]:
                    raise ValueError("Frozen evaluation original changed")
                bound_file(root, row)
            else:
                conversion = _convert_evaluation_audio(source, destination)
                technical = _technical_check(destination, asset["category"])
                pcm, count = _pcm_hash(destination)
                row = {**asset, "sample_id": asset["id"], "path": str(destination), "sha256": digest(destination),
                    "original_sha256": asset["sha256"], "original_path": str(source), "split": "validation",
                    "duration_s": count / 16000, "normalized_frames": count, "normalized_pcm_sha256": pcm,
                    "pcm_hash_scope": "exact_full_source_16khz_mono_pcm16", "technical_qc": technical,
                    "conversion": conversion,
                    "technical_pass": True, "provenance": "public_dataset", "training_admitted": False,
                    "evaluation_only": True, "individual_human_listening": False,
                    "unknown_speaker_group": asset["dataset_id"], "evaluation_content_approved": False}
                write_json(check, row)
            rows.append(row)
            _emit(progress, "학습과 분리한 평가 소음 준비", index + 1, len(catalog["assets"]), "files")
        audit = assert_held_out(project, rows, prepared)
        sources = _write_rows(output / "evaluation_sources.jsonl", rows)
        queue_rows = [{**r, "sample_id": "preview_" + r["sample_id"], "reviewed_source": {"path": r["path"], "sha256": r["sha256"]},
            "review_group": _TYPE_NAMES.get(r["eval_type"], r["eval_type"]), "text": _TYPE_NAMES.get(r["eval_type"], r["eval_type"]),
            "review_instruction": "학습에 넣지 않는 소음 평가용 자료입니다. 표시한 종류의 소리가 정상적이면 보존하고, 호출어가 들리거나 깨졌으면 폐기하세요."} for r in rows]
        queue = _write_rows(output / "representatives.jsonl", queue_rows)
        receipt = {"kind": "studio_evaluation_noise_preparation_v2", "recipe_version": 2, "catalog_id": catalog_id,
            "wakeword": config["wakeword"], "state": "awaiting_human_review", "evaluation_only": True,
            "intended_use": "held_out_development_noise", "artifacts": {"source_manifest": sources, "representative_queue": queue},
            "source_count": len(rows), "category_counts": dict(Counter(r["category"] for r in rows)),
            "eval_type_counts": dict(Counter(r["eval_type"] for r in rows)), "download_bytes": catalog["size_bytes"],
            "human_listened_count": 0, "training_admitted": False, "Final_accessed": False,
            "split_audit": audit, "limitations": catalog["limitations"], "created_at": now()}
        write_json(receipt_path, receipt)
        return {**receipt, "preparation": {"path": str(receipt_path), "sha256": digest(receipt_path)}, "resumed": False}


def admit_evaluation_review(project, queue, answer, prepared):
    """Admit actual saved listening decisions exclusively for Development noise."""
    root, config = load_project(project)
    prep_path = resolved(root, queue["preparation"])
    receipt = read_json(prep_path)
    if (receipt.get("kind") != "studio_evaluation_noise_preparation_v2" or receipt.get("wakeword") != config["wakeword"]
            or queue.get("evaluation_only") is not True or queue.get("pools") != []):
        raise ValueError("Evaluation-only preparation/queue scope is required")
    source = bound_file(root, receipt["artifacts"]["source_manifest"])
    preview = bound_file(root, receipt["artifacts"]["representative_queue"])
    answer_path = bound_file(root, answer) if isinstance(answer, dict) else resolved(root, answer)
    saved = read_json(answer_path)
    rows, previews = read_rows(source), read_rows(preview)
    decisions = saved.get("decisions", {})
    if (saved.get("wakeword") != config["wakeword"] or saved.get("whole_pool_policy_approved") is not True
            or saved.get("evaluation_only") is not True or saved.get("intended_use") != "held_out_development_noise"
            or saved.get("pools") != []
            or saved.get("consent_scope") != "Held-out Development noise evaluation only; not for training"
            or saved.get("source_scope") != "Human dispositions entered through the local review screen"
            or not saved.get("saved_at") or saved.get("source_manifest_sha256") != digest(source)
            or saved.get("queue_sha256") != digest(preview)
            or saved.get("representative_queue_sha256") != digest(preview)
            or saved.get("queued_audio_sha256") != {r["sample_id"]: r["sha256"] for r in previews}
            or set(decisions) != {r["sample_id"] for r in previews}):
        raise ValueError("Real saved evaluation listening answers and exact source/queue bindings are required")
    if (any(a.get("decision") not in ("keep", "discard") for a in decisions.values())
            or any(a.get("listened") is not True for a in decisions.values() if a.get("decision") == "keep")):
        raise ValueError("Unheard or unresolved evaluation audio cannot be approved")
    accepted = {r["reviewed_source"]["sha256"] for r in previews if decisions[r["sample_id"]]["decision"] == "keep"}
    kept = [r for r in rows if r["sha256"] in accepted]
    if not kept:
        raise ValueError("No usable held-out evaluation noise was kept")
    for row in kept:
        bound_file(root, row)
    audit = assert_held_out(project, kept, prepared)
    record = {"kind": "studio_evaluation_noise_admission_v2", "state": "accepted_for_evaluation",
        "content_review": "accepted", "evaluation_only": True, "training_admitted": False,
        "intended_use": "held_out_development_noise", "wakeword": config["wakeword"],
        "preparation": {"path": str(prep_path), "sha256": digest(prep_path)},
        "human_answers": {"path": str(answer_path), "sha256": digest(answer_path)},
        "source_manifest": receipt["artifacts"]["source_manifest"],
        "representative_queue": receipt["artifacts"]["representative_queue"],
        "accepted_source_sha256": sorted(accepted), "human_listened_count": sum(a.get("listened") is True for a in decisions.values()),
        "kept_count": len(kept), "discarded_count": len(rows) - len(kept), "split_audit_scope": audit["scope"],
        "category_counts": dict(Counter(r["category"] for r in kept)),
        "eval_type_counts": dict(Counter(r["eval_type"] for r in kept)),
        "missing_eval_types": sorted({r["eval_type"] for r in rows} - {r["eval_type"] for r in kept}),
        "generic_precomputed_original_mapping_available": False,
        "Final_accessed": False}
    path = prep_path.parent / "admissions" / (identity(record) + ".json")
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Immutable evaluation admission changed")
    else:
        write_json(path, record)
    binding = {"path": str(path), "sha256": digest(path)}
    return {"admission": binding, "rows": [{**r, "evaluation_admission": binding,
        "evaluation_content_approved": True, "individual_human_listening": True} for r in kept]}
