"""Deterministic N×R cache with explicit counts, journal and exact resume."""
from collections import Counter
import time
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve

from .audio import active_speech_mask, pcm16, rms, noise_scale, measured_snr
from .common import append_row, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock
from .data import audit_project, rows_for, validate_admission, seek_audio, union_hours, source_key, coverage_interval
from .project import load_project
from .storage import project_storage


def seed_for(seed, role, sample_id, round_id):
    return int(identity([seed, role, sample_id, round_id])[:16], 16)


def prepare_noise_index(settings, noises):
    """Compile the unchanged sampling distribution once, rather than per draw."""
    available = {}
    for category in settings["category_weights"]:
        group = [n for n in noises if n["category"] == category]
        weights = np.asarray([settings["korean_speech_weight"]
            if category == "competing_speech" and n.get("language") in ("Korean", "ko", "kor") else 1
            for n in group], float)
        available[category] = (group, weights / weights.sum() if len(weights) else weights)
    return available


def augment(root, row, role, round_id, settings, noises, rirs, noise_index=None):
    seed = seed_for(settings["seed"], role, row["sample_id"], round_id)
    rng = np.random.default_rng(seed)
    clip = np.zeros(32000, dtype=np.float64)
    if role == "positive":
        a, b = row["call_start_s"], row["call_end_s"]
        source = {**row, "start_s": a, "end_s": b}
        audio, _ = seek_audio(root, source, b - a)
        if len(audio) > 29440:
            raise ValueError("Complete word must fit a 2s clip with 80ms head/tail; quarantine, do not cut syllables")
        start = int(rng.integers(1280, 32000 - len(audio) - 1280 + 1))
        clip[start:start + len(audio)] = audio
        bounds = [start / 16000, (start + len(audio)) / 16000]
        mask = active_speech_mask(clip, *bounds)
    else:
        audio, _ = seek_audio(root, row, 2)
        clip[:min(len(audio), 32000)] = audio[:32000]
        bounds = None
        mask = active_speech_mask(clip, 0, min(len(audio), 32000) / 16000) if row.get("speech") and rms(clip) > 1e-8 else None
    details = {"sample_id": row["sample_id"], "source_id": source_key(row), "round_id": round_id,
               "seed": seed, "call_bounds_in_clip": bounds, "noise_id": None, "rir_id": None,
               "snr_db": None, "measured_snr_db": None}
    if rirs and rng.random() < settings["rir_probability"]:
        rir = rirs[int(rng.integers(len(rirs)))]
        impulse, _ = seek_audio(root, rir, 2)
        norm = np.sqrt(np.sum(impulse ** 2))
        if norm <= 1e-8:
            raise ValueError("Silent RIR")
        clip = fftconvolve(clip, impulse / norm)[:32000]
        details["rir_id"] = source_key(rir)
    draw = rng.random()
    if noises and draw >= settings["clean_probability"]:
        categories = list(settings["category_weights"])
        available = prepare_noise_index(settings, noises) if noise_index is None else noise_index
        if any(not group for group, _ in available.values()):
            raise ValueError("All declared noise categories need admitted material; do not silently renormalize")
        weights = np.asarray([settings["category_weights"][k] for k in categories], float)
        category = categories[int(rng.choice(len(categories), p=weights / weights.sum()))]
        group, probabilities = available[category]
        noise = group[int(rng.choice(len(group), p=probabilities))]
        samples, offset = seek_audio(root, noise, 2, rng)
        if len(samples) < 32000:
            # Tiling is allowed augmentation, never credited as new unique hours.
            samples = np.tile(samples, int(np.ceil(32000 / len(samples))))
        samples = samples[:32000]
        details.update(noise_id=source_key(noise), noise_category=category, noise_offset_s=offset)
        if mask is not None:
            range_key = "moderate_snr_db" if draw < settings["clean_probability"] + settings["moderate_probability"] else "hard_snr_db"
            snr = float(rng.uniform(*settings[range_key]))
            scaled = samples * noise_scale(clip, samples, mask, snr)
            details.update(snr_db=snr, measured_snr_db=measured_snr(clip, scaled, mask))
        else:
            # No fictional speech SNR for pure environmental Negative.
            scaled = samples * 10 ** (float(rng.uniform(-12, 0)) / 20)
            details["noise_gain_only"] = True
        clip += scaled
    gain = float(rng.uniform(*settings["gain_db"]))
    clip *= 10 ** (gain / 20)
    peak = float(np.max(np.abs(clip)))
    headroom = min(1.0, settings["peak_limit"] / peak) if peak else 1.0
    clip *= headroom
    samples = pcm16(clip)
    details.update(gain_db=gain, headroom_gain=headroom,
                   pcm_sha256=__import__("hashlib").sha256(samples.tobytes()).hexdigest())
    return samples, details


@project_storage
def build_features(project, role, output=None, rounds=None, batch_size=32, smoke=False, resume=False, frontend=None):
    root, config = load_project(project)
    pool = "positive_train" if role == "positive" else role
    if pool not in ("positive_train", "generic_negative", "korean_negative", "hard_negative"):
        raise ValueError("Only declared training pools are supported")
    rounds = rounds or (config["augmentation_rounds"] if role == "positive" else 1)
    if rounds not in (1, 3, 5) or role != "positive" and rounds != 1:
        raise ValueError("Positive R1/R3/R5; fixed Negative R1 only")
    manifest, sources = rows_for(root, config, pool)
    if role == "positive" and not smoke and len(sources) != config["targets"]["positive_originals"]:
        raise ValueError("Positive manifest does not match the selected TTS original count; finish generation or declare a separate budget")
    validate_admission(root, sources, config, role, smoke)
    audit = audit_project(project)
    if not audit["passed"]:
        raise ValueError("Split/source audit failed: " + str(audit["errors"][:5]))
    materials = {}
    for material in ("noise", "rir"):
        _, rows = rows_for(root, config, material)
        if rows:
            validate_admission(root, rows, config, material, smoke)
        materials[material] = rows
    if not materials["noise"] and config["augmentation"]["clean_probability"] < 1:
        raise ValueError("Non-clean augmentation needs reviewed noise materials")
    noise_index = prepare_noise_index(config["augmentation"], materials["noise"])
    if output:
        directory = Path(output).resolve()
    elif role == "positive" and rounds != config["augmentation_rounds"]:
        directory = root / "features" / f"positive_N{config['targets']['positive_originals']}_R{rounds}"
    else:
        directory = resolved(root, config["pool_features"][role])
    if role != "positive" and not smoke:
        import soundfile as sf
        for row in sources:
            info = sf.info(resolved(root, row["path"]))
            if row.get("end_s", info.duration) - row.get("start_s", 0) > 2.0001:
                raise ValueError("Formal Negative requires fixed 2s windows; run koww windows first")
    qualification = None
    if frontend is None:
        from .frontend import qualified_fast_frontend
        probes = [augment(root, row, role, 0, config["augmentation"], materials["noise"], materials["rir"], noise_index)[0] for row in sources[:8]]
        frontend, qualification = qualified_fast_frontend(resolved(root, config["frontend_dir"]), probes)
    contract = {"schema_version": 1, "source_manifest_sha256": digest(manifest), "role": role,
        "rounds": rounds, "sources": sources, "materials": materials, "augmentation": config["augmentation"],
        "frontend": frontend.identity, "implementation_sha256": digest(__file__), "smoke": smoke}
    expected = len(sources) * rounds
    if expected < 1 or batch_size < 1:
        raise ValueError("Positive cache and batch sizes required")
    with writer_lock(directory):
        contract_path = directory / "contract.json"
        if contract_path.exists():
            if not resume or read_json(contract_path) != contract:
                raise ValueError("Immutable cache exists or resume identity changed")
            if (directory / "metadata.json").exists():
                return verify_cache(directory)
        else:
            write_json(contract_path, contract)
        if qualification is not None and not (directory / "frontend_qualification.json").exists():
            write_json(directory / "frontend_qualification.json", qualification)
        journal = directory / "augmentation.jsonl"
        completed = read_rows(journal)
        start = len(completed)
        path = directory / "features.npy"
        if start:
            array = np.lib.format.open_memmap(path, mode="r+")
            for i, row in enumerate(completed):
                if row["feature_row"] != i or row["feature_sha256"] != __import__("hashlib").sha256(array[i].tobytes()).hexdigest():
                    raise ValueError("Committed feature prefix changed; do not resume")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(expected, 16, 96))
        if array.shape != (expected, 16, 96):
            raise ValueError("Resume cache shape mismatch")
        started = time.perf_counter()
        pending, rows = [], []
        for index in range(start, expected):
            source, round_id = sources[index // rounds], index % rounds
            samples, info = augment(root, source, role, round_id, config["augmentation"], materials["noise"], materials["rir"], noise_index)
            pending.append(samples)
            rows.append({"feature_row": index, **info})
            if len(pending) == batch_size or index == expected - 1:
                values = frontend.embed(np.asarray(pending, dtype=np.int16))
                if values.shape != (len(rows), 16, 96) or not np.isfinite(values).all():
                    raise ValueError("Invalid frontend output")
                offset = rows[0]["feature_row"]
                array[offset:offset + len(rows)] = values
                array.flush()
                for item, value in zip(rows, values):
                    item["feature_sha256"] = __import__("hashlib").sha256(value.tobytes()).hexdigest()
                # Commit one bounded batch, not one disk fsync per feature row.
                with journal.open("a", encoding="utf-8") as handle:
                    for item in rows:
                        handle.write(__import__("json").dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
                    __import__("os").fsync(handle.fileno())
                print(f"{role} R{rounds}: {index+1}/{expected}", flush=True)
                pending, rows = [], []
        del array
        intervals = {}
        for row in sources:
            info = __import__("soundfile").info(resolved(root, row["path"]))
            original, interval = coverage_interval(row, row.get("start_s", 0), row.get("end_s", info.duration))
            intervals.setdefault(original, []).append(interval)
        metadata = {"schema_version": 1, "role": role, "split": "train", "feature_rows": expected,
            "original_count": len(sources), "rounds": rounds, "frontend_id": identity(frontend.identity),
            "frontend": frontend.identity, "provenance": "qwen_tts" if role == "positive" else
                next(iter({r["provenance"] for r in sources})) if len({r["provenance"] for r in sources}) == 1 else "mixed_reviewed_sources",
            "human_reference_used": False, "feature_sha256": digest(path), "journal_sha256": digest(journal),
            "contract_sha256": digest(contract_path), "split_audit_passed": True,
            "unique_source_hours": union_hours(intervals), "smoke_test_only": smoke,
            "source_time_scope": "Union of declared parent coordinates where supplied; caption offsets are not sample-exact original alignment",
            "elapsed_this_invocation_s": time.perf_counter() - started, "created_at": now(),
            "quality_review": "accepted" if not smoke else "unapproved_smoke_only"}
        write_json(directory / "metadata.json", metadata)
        return verify_cache(directory)


def verify_cache(directory):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    feature_path = resolved(directory, meta.get("feature_path", "features.npy"))
    if digest(feature_path) != meta["feature_sha256"]:
        raise ValueError("Feature payload hash changed")
    values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (meta["feature_rows"], 16, 96) or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Invalid feature shape/dtype")
    for i in range(0, len(values), 4096):
        if not np.isfinite(values[i:i+4096]).all():
            raise ValueError("Nonfinite feature values")
    if meta.get("imported_precomputed"):
        from .assets import GENERIC_SHA
        if meta.get("role") != "generic_negative" or meta["feature_sha256"] != GENERIC_SHA or meta["feature_rows"] != 5625000 or values.dtype != np.float16:
            raise ValueError("Unsupported imported feature cache")
        return meta
    if digest(directory / "augmentation.jsonl") != meta["journal_sha256"] or digest(directory / "contract.json") != meta["contract_sha256"]:
        raise ValueError("Feature journal/contract changed")
    rows = read_rows(directory / "augmentation.jsonl")
    if len(rows) != len(values) or meta["feature_rows"] != meta["original_count"] * meta["rounds"]:
        raise ValueError("Actual output is not N×rounds")
    per_original = {}
    for i, row in enumerate(rows):
        if row["feature_row"] != i:
            raise ValueError("Feature order changed")
        per_original.setdefault(row["sample_id"], []).append((row["round_id"], row["seed"], row["pcm_sha256"]))
    for entries in per_original.values():
        if [r[0] for r in entries] != list(range(meta["rounds"])) or len({r[1] for r in entries}) != meta["rounds"]:
            raise ValueError("Missing rounds or repeated augmentation seeds")
        if meta["rounds"] > 1 and len({r[2] for r in entries}) != meta["rounds"]:
            raise ValueError("Repeated PCM across rounds; inspect augmentation diversity")
    return meta


def compare_rounds(first, second):
    a, b = Path(first), Path(second)
    ma, mb = verify_cache(a), verify_cache(b)
    ca, cb = read_json(a / "contract.json"), read_json(b / "contract.json")
    ca.pop("rounds"); cb.pop("rounds")
    if ca != cb:
        raise ValueError("R comparison changed sources, noise distribution, frontend or code")
    rows_a = {(r["sample_id"], r["round_id"]): r for r in read_rows(a / "augmentation.jsonl")}
    rows_b = {(r["sample_id"], r["round_id"]): r for r in read_rows(b / "augmentation.jsonl")}
    shared = set(rows_a) & set(rows_b)
    if len(shared) != ma["original_count"] * min(ma["rounds"], mb["rounds"]):
        raise ValueError("Incomplete nested round coverage")
    for key in shared:
        if any(rows_a[key][k] != rows_b[key][k] for k in ("seed", "pcm_sha256", "feature_sha256")):
            raise ValueError("Shared augmentation/feature differs")
    return {"passed": True, "shared_original_rounds": len(shared), "only_rounds_changed": True}
