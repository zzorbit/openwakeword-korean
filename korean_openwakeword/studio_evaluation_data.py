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

EVALUATION_RECIPE_VERSION = 4
_TYPE_NAMES = {
    "rain": "비", "wind": "바람", "vacuum_cleaner": "청소기", "washing_machine": "세탁기",
    "engine": "엔진", "train": "기차", "keyboard_typing": "키보드", "door_wood_knock": "문 두드림",
    "glass_breaking": "유리 충격·파손", "footsteps": "발걸음", "clock_alarm": "알람",
    "clock_tick": "시계 전자·기계음", "car_horn": "경적", "siren": "사이렌", "church_bells": "종소리",
    "music_instrumental_piano_drums": "피아노·드럼 연주", "music_instrumental_guitar_drums": "기타·드럼 연주",
    "music_singing": "보컬 멜로디", "music_vocal_pad": "보컬 패드 음악",
    "korean_market_conversation": "한국어 시장 대화", "korean_restaurant_conversation": "한국어 식당 대화",
    "korean_subway_conversation": "한국어 지하철 대화", "korean_street_conversation": "한국어 거리 대화",
    "korean_transit_announcement": "한국어 교통 안내방송",
    "dog": "개 짖음", "cat": "고양이", "chirping_birds": "새소리", "crying_baby": "아기 울음",
    "sneezing": "재채기", "clapping": "박수", "breathing": "숨소리", "coughing": "기침",
    "laughing": "웃음", "snoring": "코골이", "pouring_water": "물 붓기", "toilet_flush": "변기 물 내림",
    "water_drops": "물방울", "can_opening": "캔 따기", "mouse_click": "마우스 클릭",
    "door_wood_creaks": "문 삐걱거림", "airplane": "비행기", "helicopter": "헬리콥터",
    "thunderstorm": "천둥", "sea_waves": "파도", "music_jazz": "재즈", "music_techno": "테크노",
    "music_hiphop": "힙합", "music_rock_vocals": "보컬 록", "music_rock": "록",
    "music_cinematic_orchestral": "영화풍 관현악", "music_house": "하우스 음악",
    "music_orchestral_loop": "관현악 루프",
}


def evaluation_catalog():
    catalog = read_json(files("korean_openwakeword") / "resources/evaluation_backgrounds_v4.json")
    if catalog.get("recipe_version") != EVALUATION_RECIPE_VERSION:
        raise ValueError("Unsupported held-out noise catalog")
    return catalog


def evaluation_preview():
    """Dataset summaries keep the first screen short; individual terms remain available."""
    catalog = evaluation_catalog()
    grouped = defaultdict(list)
    for row in catalog["assets"]:
        group = ("esc50_common_evaluation_v4" if row["dataset_id"].startswith("esc50") else
                 "freesound_music_common_evaluation_v4" if row["category"] == "music" else row["dataset_id"])
        grouped[group].append(row)
    labels = {"esc50_common_evaluation_v4": "환경·생활·전자음 공통 평가 자료",
              "freesound_music_common_evaluation_v4": "연주·보컬·여러 장르의 음악",
              "freesound_korean_held_out_v3": "한국어 현장 대화·안내방송 시험용 배경 음원"}
    descriptions = {
        "esc50_common_evaluation_v4": "날씨·가전·차량·생활 충격음·동물·기침·웃음·알람 등 서로 다른 원본의 지속음과 순간음입니다.",
        "freesound_music_common_evaluation_v4": "피아노·기타·드럼 연주와 보컬, 전자음악·록·재즈·관현악 등 게시자 정보로 구분한 음악입니다.",
        "freesound_korean_held_out_v3": "시장·식당·지하철·거리의 한국어 현장 말소리와 교통 안내방송입니다. TV·라디오나 한국어 보컬 음악 자료는 포함하지 않습니다.",
    }
    result = []
    for dataset, rows in grouped.items():
        korean = dataset == "freesound_korean_held_out_v3"
        result.append({"id": dataset, "dataset_id": dataset, "name": labels.get(dataset, dataset),
            "role": "held_out_korean_speech" if korean else "held_out_evaluation_noise", "split": "validation", "evaluation_only": True,
            "description": descriptions.get(dataset, "학습에 넣지 않는 시험용 배경 음원입니다."),
            "file_count": len(rows), "type_count": len({r["eval_type"] for r in rows}),
            "eval_types": sorted({r["eval_type"] for r in rows}), "total_bytes": sum(r["size_bytes"] for r in rows),
            "type_labels": [_TYPE_NAMES.get(kind, kind) for kind in sorted({r["eval_type"] for r in rows})],
            "source_url": "https://github.com/karolpiczak/ESC-50" if dataset.startswith("esc50") else "https://freesound.org/",
            "license": "ESC-50 CC-BY-NC-3.0 / ESC-10 CC-BY-3.0; 개별 원출처 조건 보존" if dataset.startswith("esc50") else "개별 Creative Commons 조건 · 공개 HQ MP3 미리듣기",
            "individual_licenses": [{"id": r["id"], "name": r.get("name", _TYPE_NAMES.get(r["eval_type"], r["eval_type"])),
                "license": r["license"], "source_url": r["original_source_url"], "media_variant": r["media_variant"]} for r in rows]})
    result.append({"id": "zeroth_validation_reserve", "dataset_id": "zeroth_validation_reserve",
        "name": "한국어 낭독 음성 · Zeroth 평가 예약분", "role": "held_out_korean_speech",
        "split": "validation", "evaluation_only": True,
        "source_url": "https://www.openslr.org/40/", "license": "Zeroth 원자료 이용 조건",
        "type_labels": ["한국어 낭독"],
        "description": "이미 내려받는 Zeroth에서 학습 화자와 분리해 남긴 한국어 낭독 음성입니다. 실제 파일 수는 자료 준비 후 확정되며 별도 추가 다운로드는 없습니다."})
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

    These pinned public recordings are at most five minutes. A single conversion
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


def _source_identity(row):
    metadata = {key: row.get(key) for key in ("canonical_recording_id", "category", "eval_type", "language", "source_url")}
    source_sha = row.get("reviewed_source", {}).get("sha256", row["sha256"])
    return row["canonical_source_id"], row["original_sha256"], source_sha, identity(metadata)


def _supported_preparation(receipt, wakeword):
    return (receipt.get("kind") in tuple(f"studio_evaluation_noise_preparation_v{v}" for v in (2, 3, 4, 5, 6))
            and receipt.get("recipe_version") in (2, 3, 4, 5, 6) and receipt.get("wakeword") == wakeword
            and receipt.get("evaluation_only") is True and receipt.get("training_admitted") is False
            and receipt.get("intended_use") == "held_out_development_noise"
            and (receipt.get("recipe_version") != 6
                 or (receipt.get("evaluation_source_scope") == "common_30h_only"
                     and receipt.get("kind") == "studio_evaluation_noise_preparation_v6"
                     and not receipt.get("inherited_admissions"))))


def assert_common_source_scope(rows):
    """Recipe 6 binds one source pool; no legacy reserves can be appended."""
    from .evaluation_materials import BUCKET_HOURS
    for row in rows:
        if row.get("evaluation_source_scope") != "common_30h_only":
            raise ValueError("Common evaluation source pool scope changed")
        if row.get("category") == "rir":
            if not row.get("canonical_room_id"):
                raise ValueError("Common evaluation RIRs require real room lineage")
        elif row.get("eval_bucket") not in BUCKET_HOURS:
            raise ValueError("Legacy supplemental or training-reserve audio is outside the common evaluation pool")


def _validated_decisions(root, config, receipt, answer_path, seen=()):
    """Recheck real answers and any immutable inherited admissions, never infer listening."""
    if not _supported_preparation(receipt, config["wakeword"]):
        raise ValueError("Evaluation-only preparation scope is required")
    source = bound_file(root, receipt["artifacts"]["source_manifest"])
    preview = bound_file(root, receipt["artifacts"]["representative_queue"])
    rows, previews = read_rows(source), read_rows(preview)
    if receipt.get("recipe_version") == 6:
        assert_common_source_scope(rows)
    inherited = {}
    for binding in receipt.get("inherited_admissions", []):
        path = bound_file(root, binding)
        if str(path) in seen:
            raise ValueError("Cyclic evaluation review inheritance")
        record = read_json(path)
        if (record.get("kind") not in tuple(f"studio_evaluation_noise_admission_v{v}" for v in (2, 3, 4, 5, 6))
                or record.get("state") != "accepted_for_evaluation" or record.get("evaluation_only") is not True
                or record.get("training_admitted") is not False or record.get("wakeword") != config["wakeword"]
                or record.get("intended_use") != "held_out_development_noise" or record.get("content_review") != "accepted"):
            raise ValueError("Inherited evaluation review scope changed")
        previous = read_json(bound_file(root, record["preparation"]))
        previous_answer = bound_file(root, record["human_answers"]) if record.get("human_answers") else None
        checked, previous_rows = _validated_decisions(root, config, previous, previous_answer, (*seen, str(path)))
        accepted = sorted({r["sha256"] for r in previous_rows if checked[_source_identity(r)]["decision"] == "keep"})
        if (record.get("accepted_source_sha256") != accepted
                or record.get("source_manifest") != previous["artifacts"]["source_manifest"]
                or record.get("representative_queue") != previous["artifacts"]["representative_queue"]):
            raise ValueError("Inherited evaluation admission differs from actual listening answers")
        inherited.update(checked)
    decisions = {}
    if previews:
        if answer_path is None:
            raise ValueError("Real saved evaluation listening answers are required")
        saved = read_json(answer_path)
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
    checked = {}
    source_rows = {_source_identity(r): r for r in rows}
    for row in previews:
        key = _source_identity(row)
        if key not in source_rows or row["reviewed_source"]["sha256"] != source_rows[key]["sha256"]:
            raise ValueError("Evaluation preview differs from its actual source")
        bound_file(root, row)
        bound_file(root, row["reviewed_source"])
        checked[key] = {**decisions[row["sample_id"]],
                        "review_scope": row.get("review_excerpt", {"scope": "full_source"})}
    if receipt.get("recipe_version") in (5, 6):
        # Consent covers the frozen group; only the actual previews are heard.
        groups = receipt.get("representative_groups", {})
        actual_groups = defaultdict(list)
        for row in rows:
            actual_groups[row["review_group_id"]].append(row["sample_id"])
        if set(groups) != set(actual_groups):
            raise ValueError("Representative group scope changed")
        for group_id, members in actual_groups.items():
            spec = groups[group_id]
            ids = spec["preview_ids"]
            if sorted(spec["source_ids"]) != sorted(members) or not ids or any(i not in decisions for i in ids):
                raise ValueError("Every frozen source group needs genuine representative decisions")
            if any(p["review_group_id"] != group_id for p in previews if p["sample_id"] in ids):
                raise ValueError("Representative preview belongs to another group")
            decision = "keep" if all(decisions[i]["decision"] == "keep" for i in ids) else "discard"
            for row in rows:
                if row["review_group_id"] == group_id:
                    checked[_source_identity(row)] = {"decision": decision, "listened": False,
                        "review_scope": {"scope": "stratified_representative_pool_not_individual_source",
                            "group_id": group_id, "preview_ids": ids}}
    for key, row in source_rows.items():
        if key not in checked:
            if key not in inherited:
                raise ValueError("New evaluation sources require their own real listening answers")
            checked[key] = inherited[key]
        bound_file(root, row)
    return checked, rows


def _previous_reviews(root, config):
    """Only admitted answers can reduce a new review queue; candidates never do."""
    candidates, bindings = {}, []
    for path in sorted((root / ".studio" / "materials").glob("evaluation_v*/admissions/*.json")):
        record = read_json(path)
        if record.get("kind") not in tuple(f"studio_evaluation_noise_admission_v{v}" for v in (2, 3, 4)):
            continue
        prep = read_json(bound_file(root, record["preparation"]))
        answer = bound_file(root, record["human_answers"]) if record.get("human_answers") else None
        binding = {"path": str(path), "sha256": digest(path)}
        # Validate the admission itself as well as all answer and source hashes.
        wrapper = {"kind": "studio_evaluation_noise_preparation_v3", "recipe_version": 3,
            "wakeword": config["wakeword"], "evaluation_only": True, "training_admitted": False,
            "artifacts": prep["artifacts"], "inherited_admissions": [binding],
            "intended_use": "held_out_development_noise"}
        checked, rows = _validated_decisions(root, config, wrapper, answer)
        bindings.append(binding)
        for row in rows:
            candidates[_source_identity(row)] = {"row": row, "decision": checked[_source_identity(row)], "admission": binding}
    return candidates, bindings


def _review_preview(row, output):
    """Keep lengthy Korean sources intact, with an explicit 15-second preview."""
    preview = {**row, "sample_id": "preview_" + row["sample_id"],
        "reviewed_source": {"path": row["path"], "sha256": row["sha256"]},
        "review_group": _TYPE_NAMES.get(row["eval_type"], row["eval_type"]),
        "text": row.get("name", _TYPE_NAMES.get(row["eval_type"], row["eval_type"]))}
    preview["review_instruction"] = "학습에 넣지 않는 시험용 배경 음원입니다. 표시한 종류의 소리가 정상적이면 보존하고, 호출어가 들리거나 깨졌으면 폐기하세요."
    if row["category"] == "competing_speech" and row["duration_s"] > 20:
        with sf.SoundFile(row["path"]) as stream:
            count = 15 * stream.samplerate
            start = max(0, (stream.frames - count) // 2)
            stream.seek(start)
            values = stream.read(count, dtype="int16")
            destination = output / "review_audio" / (row["sample_id"] + "_15s.wav")
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, values, stream.samplerate, subtype="PCM_16")
        preview.update(path=str(destination), sha256=digest(destination), duration_s=len(values) / 16000,
            review_excerpt={"scope": "representative_excerpt_not_full_source", "start_s": start / 16000,
                "end_s": (start + len(values)) / 16000, "selection": "deterministic_contiguous_center_15_seconds",
                "full_source_duration_s": row["duration_s"], "full_source_sha256": row["sha256"]},
            review_instruction="원본 전체가 아닌 15초 대표 구간입니다. 한국어 말소리·안내방송이 정상적이면 보존하세요. 불명료하거나 호출어가 들리거나 깨졌으면 폐기하세요. 보존한 원본 전체는 시험용으로만 사용하며, 전체를 청취한 것으로 기록하지 않습니다.")
    return preview


def prepare_evaluation_materials(project, workspace, prepared, progress=None):
    from .studio_data import _download, _write_rows, _emit
    from .pool_review import _technical_check
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is locked; new Development noise belongs to a separate experiment")
    catalog = evaluation_catalog()
    catalog_id = identity(catalog)[:16]
    version = catalog["recipe_version"]
    if version not in (2, 3, 4):
        raise ValueError("Unsupported evaluation background source recipe")
    output = root / ".studio" / "materials" / (f"evaluation_v{version}_" + catalog_id)
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
        for row in read_rows(receipt["artifacts"]["representative_queue"]["path"]):
            bound_file(root, row)
            bound_file(root, row["reviewed_source"])
        audit = assert_held_out(project, rows, prepared)
        return {**receipt, "preparation": {"path": str(receipt_path), "sha256": digest(receipt_path)}, "split_audit": audit, "resumed": True}
    if any((root / "features").rglob("contract.json")) or any((root / "runs").rglob("contract.json")):
        raise ValueError("Existing evaluation/training inputs are frozen; use a new experiment for revised background sources")
    previous, _ = _previous_reviews(root, config) if version >= 3 else ({}, [])
    rows, pending, inherited = [], [], {}
    with writer_lock(output):
        for index, asset in enumerate(catalog["assets"]):
            reused = next((entry for entry in previous.values()
                           if entry["row"]["original_sha256"] == asset["sha256"]
                           and entry["row"]["canonical_source_id"] == asset["canonical_source_id"]
                           and all(entry["row"].get(key) == asset.get(key) for key in
                                   ("canonical_recording_id", "category", "eval_type", "language", "source_url"))), None)
            if reused:
                row = dict(reused["row"])
                bound_file(root, row)
                rows.append(row)
                inherited[reused["admission"]["sha256"]] = reused["admission"]
                _emit(progress, "기존 청취 근거 재사용 · 시험용 배경 음원", index + 1, len(catalog["assets"]), "files")
                continue
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
                    "conversion": conversion, "eval_type_label": _TYPE_NAMES.get(asset["eval_type"], asset["eval_type"]),
                    "technical_pass": True, "provenance": "public_dataset", "training_admitted": False,
                    "evaluation_only": True, "individual_human_listening": False,
                    "unknown_speaker_group": asset["dataset_id"], "evaluation_content_approved": False}
                write_json(check, row)
            rows.append(row)
            pending.append(row)
            _emit(progress, "학습과 분리한 시험용 배경 음원 준비", index + 1, len(catalog["assets"]), "files")
        audit = assert_held_out(project, rows, prepared)
        sources = _write_rows(output / "evaluation_sources.jsonl", rows)
        queue_rows = [_review_preview(r, output) for r in pending]
        queue = _write_rows(output / "representatives.jsonl", queue_rows)
        receipt = {"kind": f"studio_evaluation_noise_preparation_v{version}", "recipe_version": version, "catalog_id": catalog_id,
            "wakeword": config["wakeword"], "state": "awaiting_human_review", "evaluation_only": True,
            "intended_use": "held_out_development_noise", "artifacts": {"source_manifest": sources, "representative_queue": queue},
            "source_count": len(rows), "category_counts": dict(Counter(r["category"] for r in rows)),
            "eval_type_counts": dict(Counter(r["eval_type"] for r in rows)), "download_bytes": catalog["size_bytes"],
            "inherited_admissions": list(inherited.values()), "inherited_source_count": len(rows) - len(pending),
            "pending_review_source_count": len(pending),
            "human_listened_count": 0, "training_admitted": False, "Final_accessed": False,
            "split_audit": audit, "limitations": catalog["limitations"], "created_at": now()}
        write_json(receipt_path, receipt)
        return {**receipt, "preparation": {"path": str(receipt_path), "sha256": digest(receipt_path)}, "resumed": False}


def prepare_common_evaluation_materials(project, workspace, prepared, progress=None):
    """30h source recipe plus held-out RIRs, with bounded representative review.

    Original duration is measured before any repeated SNR/RIR mixtures. Neither
    downloading nor technical checks invent source-wide listening or approval.
    """
    from .evaluation_materials import prepare_common_sources, evaluation_material_catalog
    from .evaluation_rir import prepare_common_rirs, render_rir_review_previews, common_rir_catalog
    from .studio_data import _write_rows, _emit
    root, config = load_project(project)
    version = prepared.get("evaluation_materials_policy", {}).get("evaluation_materials_recipe", 5)
    acquisition_policy = prepared.get("common_download_policy")
    if acquisition_policy not in (None, "selected_common_zip_v1"):
        raise ValueError("Unsupported common material download policy")
    acquisition_options = {"acquisition_policy": acquisition_policy} if acquisition_policy else {}
    if version not in (5, 6):
        raise ValueError("Unsupported common evaluation material recipe")
    strict_common = version == 6
    if (root / "final_lock.json").exists() or any((root / "features").rglob("contract.json")) or any((root / "runs").rglob("contract.json")):
        raise ValueError("Use a new experiment for the expanded common evaluation")
    catalogs = [evaluation_material_catalog(), common_rir_catalog()]
    catalog_id = identity(catalogs + ([{"evaluation_source_scope": "common_30h_only"}] if strict_common else [evaluation_catalog()]))[:16]
    output = root / ".studio" / "materials" / (f"evaluation_v{version}_" + catalog_id)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preparation.json"
    if path.exists():
        receipt = read_json(path)
        if (receipt["catalog_id"] != catalog_id or receipt["wakeword"] != config["wakeword"]
                or receipt.get("recipe_version") != version
                or (strict_common and receipt.get("evaluation_source_scope") != "common_30h_only")):
            raise ValueError("Frozen common evaluation catalog differs")
        for binding in receipt["artifacts"].values():
            bound_file(root, binding)
        stored_rows = read_rows(receipt["artifacts"]["source_manifest"]["path"])
        if strict_common:
            assert_common_source_scope(stored_rows)
        for row in stored_rows:
            bound_file(root, row)
        return {**receipt, "preparation": {"path": str(path), "sha256": digest(path)}, "resumed": True}
    supplemental = None if strict_common else prepare_evaluation_materials(project, workspace, prepared, progress)
    common = (prepare_common_sources(project, workspace, prepared, progress, strict_common_pool=True, **acquisition_options)
              if strict_common else prepare_common_sources(project, workspace, prepared, progress, **acquisition_options))
    if not common["target_met"]:
        raise ValueError("공통 평가 원본 부족: " + str(common["shortages"]))
    rir = prepare_common_rirs(project, output / "rir",
        cache_directory=Path(workspace).resolve() / "shared" / "downloads" / "common_rir",
        log=lambda value: _emit(progress, value, 0, 1, "steps"), **acquisition_options)
    raw = ([] if strict_common else read_rows(supplemental["artifacts"]["source_manifest"]["path"])) + common["source_rows"]
    audit = assert_held_out(project, raw, prepared)
    # RIRs are filters, never continuous Negative or additive noise.
    for row in rir["source_rows"]:
        raw.append({**row, "original_sha256": row["sha256"],
            "canonical_recording_id": row["canonical_room_id"],
            "eval_type": "rir_" + row["reverb_group"], "eval_type_label": "공간 울림 · " + row["reverb_group"]})
    groups, rows = defaultdict(list), []
    seen = set()
    for row in raw:
        if _source_identity(row) in seen:
            continue
        seen.add(_source_identity(row))
        key = (row.get("eval_bucket", "supplemental"), row["category"], row["eval_type"])
        group_id = "group_" + identity(key)[:16]
        row = {**row, "review_group_id": group_id, "evaluation_only": True,
               "training_admitted": False, "individual_human_listening": False}
        if strict_common:
            row["evaluation_source_scope"] = "common_30h_only"
        rows.append(row)
        groups[group_id].append(row)
    previews, specs = [], {}
    for group_id, members in groups.items():
        ordered = sorted(members, key=lambda r: identity([r["canonical_recording_id"], r["sha256"]]))
        selected = ordered[:min(3, len(ordered))] if ordered[0].get("eval_bucket") else ordered[:1]
        if ordered[0]["category"] == "rir":
            # Use actual filters applied to a fixed chirp, not raw impulses.
            rendered = render_rir_review_previews(project, ordered, output / "previews" / group_id, count=3)
            by_sha = {r["sha256"]: r for r in ordered}
            items = [{**by_sha[p["source_sha256"]], **p,
                      "sample_id": "preview_" + p["sample_id"],
                      "reviewed_source": {"path": by_sha[p["source_sha256"]]["path"], "sha256": p["source_sha256"]},
                      "review_group": "공간 울림", "text": p["description"],
                      "review_excerpt": {"scope": "rir_filter_demonstration_not_individual_source_listening"},
                      "review_instruction": "공간 울림이 정상적으로 들리는지 확인하세요. 합성 시험음이며 호출어 발음 검수가 아닙니다."} for p in rendered]
        else:
            items = [_review_preview({**r, "category": "competing_speech"} if r["duration_s"] > 20 else r, output) for r in selected]
            for item, original in zip(items, selected):
                item["category"] = original["category"]
                item["review_instruction"] = "종류가 맞고 소리가 깨지지 않았으면 보존하세요. 대표 구간 검수이며 묶음 전체를 들은 것으로 기록하지 않습니다. 하나라도 폐기하면 해당 묶음은 보류합니다."
        previews.extend(items)
        specs[group_id] = {"source_ids": sorted(r["sample_id"] for r in ordered),
                           "preview_ids": [r["sample_id"] for r in items]}
    sources = _write_rows(output / "evaluation_sources.jsonl", rows)
    queue = _write_rows(output / "representatives.jsonl", previews)
    if strict_common:
        assert_common_source_scope(rows)
    receipt = {"kind": f"studio_evaluation_noise_preparation_v{version}", "recipe_version": version,
        "catalog_id": catalog_id, "wakeword": config["wakeword"], "state": "awaiting_human_review",
        "evaluation_only": True, "training_admitted": False, "intended_use": "held_out_development_noise",
        "artifacts": {"source_manifest": sources, "representative_queue": queue},
        "source_count": len(rows), "eval_type_counts": dict(Counter(r["eval_type"] for r in rows)),
        "representative_groups": specs, "material_coverage": common["coverage"], "rir_coverage": rir["coverage"],
        "human_listened_count": 0, "pending_review_source_count": len(previews),
        "individual_listening_claimed": False, "split_audit": audit, "Final_accessed": False, "created_at": now()}
    if strict_common:
        receipt["evaluation_source_scope"] = "common_30h_only"
        receipt["negative_duration_budget"] = common.get("duration_budget")
    write_json(path, receipt)
    return {**receipt, "preparation": {"path": str(path), "sha256": digest(path)}, "resumed": False}


def admit_fixed_evaluation(project, queue, prepared, library_evidence):
    """Connect pinned, technically validated Development audio without claiming listening."""
    from .material_policy import validate_library_policy_evidence
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; evaluation inputs cannot change")
    path = resolved(root, queue["preparation"])
    receipt = read_json(path)
    if (not _supported_preparation(receipt, config["wakeword"]) or receipt.get("recipe_version") != 6
            or queue.get("evaluation_only") is not True or queue.get("pools") != []
            or receipt.get("evaluation_source_scope") != "common_30h_only"):
        raise ValueError("Library evaluation admission requires the fixed common corpus")
    record = {"kind": "studio_evaluation_noise_admission_v6", "state": "accepted_for_evaluation",
        "type": "library_fixed_catalog_technical_validation", "policy_id": "library_fixed_sources_v1",
        "content_review": "library_fixed_sources_technical_validation", "wakeword": config["wakeword"],
        "evaluation_only": True, "training_admitted": False, "intended_use": "held_out_development_noise",
        "evaluation_source_scope": "common_30h_only", "human_answers": None,
        "human_listening": False, "human_listened_count": 0,
        "library_evidence": library_evidence,
        "preparation": {"path": str(path), "sha256": digest(path)},
        "source_manifest": receipt["artifacts"]["source_manifest"],
        "individual_listening_claimed": False, "Final_accessed": False}
    validate_library_policy_evidence(root, record, role="evaluation")
    rows = read_rows(bound_file(root, receipt["artifacts"]["source_manifest"]))
    assert_common_source_scope(rows)
    if not rows:
        raise ValueError("No technically usable fixed evaluation sources")
    for row in rows:
        bound_file(root, row)
    audit = assert_held_out(project, [r for r in rows if r["category"] != "rir"], prepared)
    record.update(accepted_source_sha256=sorted({row["sha256"] for row in rows}),
        kept_count=len(rows), discarded_count=0, split_audit_scope=audit["scope"],
        category_counts=dict(Counter(row["category"] for row in rows)),
        eval_type_counts=dict(Counter(row["eval_type"] for row in rows)))
    destination = path.parent / "admissions" / ("library_fixed_" + identity(record) + ".json")
    if destination.exists():
        if read_json(destination) != record:
            raise ValueError("Immutable library evaluation admission changed")
    else:
        write_json(destination, record)
    binding = {"path": str(destination), "sha256": digest(destination)}
    return {"admission": binding, "rows": [{**row, "evaluation_admission": binding,
        "evaluation_content_approved": True,
        "evaluation_content_review_scope": {"scope": "library_fixed_catalog_technical_validation",
                                             "human_listening": False, "evidence": library_evidence},
        "individual_human_listening": False, "representative_human_listening": False} for row in rows]}


def admit_evaluation_review(project, queue, answer, prepared):
    """Admit actual saved listening decisions exclusively for Development noise."""
    root, config = load_project(project)
    prep_path = resolved(root, queue["preparation"])
    receipt = read_json(prep_path)
    if prepared.get("recipe_version") == 6 and (
            receipt.get("recipe_version") != 6 or queue.get("evaluation_source_scope") != "common_30h_only"):
        raise ValueError("Legacy evaluation admission is outside the common corpus scope")
    if (not _supported_preparation(receipt, config["wakeword"])
            or queue.get("evaluation_only") is not True or queue.get("pools") != []):
        raise ValueError("Evaluation-only preparation/queue scope is required")
    answer_path = (bound_file(root, answer) if isinstance(answer, dict) else resolved(root, answer)) if answer is not None else None
    decisions, rows = _validated_decisions(root, config, receipt, answer_path)
    kept = [r for r in rows if decisions[_source_identity(r)]["decision"] == "keep"]
    accepted = {r["sha256"] for r in kept}
    if not kept:
        raise ValueError("No usable held-out evaluation noise was kept")
    for row in kept:
        bound_file(root, row)
    audit = assert_held_out(project, [r for r in kept if r["category"] != "rir"], prepared)
    version = receipt["recipe_version"]
    record = {"kind": f"studio_evaluation_noise_admission_v{version}", "state": "accepted_for_evaluation",
        "content_review": "accepted", "evaluation_only": True, "training_admitted": False,
        "intended_use": "held_out_development_noise", "wakeword": config["wakeword"],
        "preparation": {"path": str(prep_path), "sha256": digest(prep_path)},
        "human_answers": {"path": str(answer_path), "sha256": digest(answer_path)} if answer_path else None,
        "source_manifest": receipt["artifacts"]["source_manifest"],
        "representative_queue": receipt["artifacts"]["representative_queue"],
        "accepted_source_sha256": sorted(accepted), "human_listened_count": (
            sum(a.get("listened") is True for a in read_json(answer_path)["decisions"].values())
            if receipt["recipe_version"] in (5, 6) and answer_path else sum(a.get("listened") is True for a in decisions.values())),
        "inherited_admissions": receipt.get("inherited_admissions", []),
        "kept_count": len(kept), "discarded_count": len(rows) - len(kept), "split_audit_scope": audit["scope"],
        "category_counts": dict(Counter(r["category"] for r in kept)),
        "eval_type_counts": dict(Counter(r["eval_type"] for r in kept)),
        "missing_eval_types": sorted({r["eval_type"] for r in rows} - {r["eval_type"] for r in kept}),
        "generic_precomputed_original_mapping_available": False,
        "Final_accessed": False}
    if version == 6:
        record["evaluation_source_scope"] = receipt["evaluation_source_scope"]
    path = prep_path.parent / "admissions" / (identity(record) + ".json")
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Immutable evaluation admission changed")
    else:
        write_json(path, record)
    binding = {"path": str(path), "sha256": digest(path)}
    return {"admission": binding, "rows": [{**r, "evaluation_admission": binding,
        "evaluation_content_approved": True,
        "evaluation_content_review_scope": decisions[_source_identity(r)]["review_scope"],
        "individual_human_listening": decisions[_source_identity(r)]["review_scope"]["scope"] == "full_source",
        "representative_human_listening": True} for r in kept]}
