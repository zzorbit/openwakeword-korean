"""Deterministic N×R cache with explicit counts, journal and exact resume."""
import time
import hashlib
import json
from itertools import islice
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve

from .audio import active_speech_mask, pcm16, rms, noise_scale, measured_snr
from .common import digest, identity, iter_rows, now, read_json, resolved, write_json, writer_lock
from .data import audit_project, rows_for, validate_admission, seek_audio, source_key
from .streaming import manifest_index
from .project import load_project
from .storage import project_storage

MAX_NOISE_OFFSET_ATTEMPTS = 64
MIN_NOISE_RMS = 1e-8
FEATURE_SCAN_BLOCK_BYTES = 8 * 1024 * 1024


def _manifest_inventory(path):
    """Count rows and bind every manifest byte in one sequential read."""
    hasher, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream):
            hasher.update(raw)
            if raw.decode("utf-8-sig" if number == 0 else "utf-8").strip():
                count += 1
    return count, hasher.hexdigest()


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


def _non_silent_noise_window(root, noise, samples, offset, mask, seed):
    """Retry only offsets within the same selected source using a private RNG.

    The ordinary sampling RNG selected category/file/first offset already.
    Do not consume it here: SNR, gain, RIR and future sampling must retain
    their original draws even if the first window is silent. File-level QC
    does not imply a non-silent random 2s window or clean active region.
    """
    first_offset = offset
    scope = "clean_active_region" if mask is not None else "whole_2s_clip"
    retry_rng = None
    first_rms = None
    for attempt in range(1, MAX_NOISE_OFFSET_ATTEMPTS + 1):
        if len(samples) < 32000:
            # Preserve the existing short-file tiling and sample positions.
            samples = np.tile(samples, int(np.ceil(32000 / len(samples))))
        samples = samples[:32000]
        level = rms(samples, mask)
        if not np.isfinite(level):
            raise ValueError(f"Nonfinite noise RMS for source {source_key(noise)} in {scope}")
        if first_rms is None:
            first_rms = level
        if level > MIN_NOISE_RMS:
            return samples, offset, {
                "noise_offset_attempts": attempt,
                "noise_first_offset_s": first_offset,
                "noise_first_offset_rms": first_rms,
                "noise_first_offset_exclusion_reason": None if attempt == 1 else "silent_in_" + scope,
                "noise_offset_retry_scope": scope,
            }
        if attempt < MAX_NOISE_OFFSET_ATTEMPTS:
            if retry_rng is None:
                retry_seed = int(identity([seed, "non_silent_noise_offset_v1",
                    source_key(noise), noise["sha256"], scope])[:16], 16)
                retry_rng = np.random.default_rng(retry_seed)
            # seek_audio propagates broken/missing/nonfinite input errors.
            # No category/file replacement and no clean fallback is allowed.
            samples, offset = seek_audio(root, noise, 2, retry_rng)
    raise ValueError(
        f"No non-silent noise window for source {source_key(noise)} "
        f"(category={noise['category']}, scope={scope}) after "
        f"{MAX_NOISE_OFFSET_ATTEMPTS} offsets; first_offset_s={first_offset}. "
        "Use a versioned valid source interval; clean fallback is disabled."
    )


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
        samples, offset, offset_diagnostics = _non_silent_noise_window(root, noise, samples, offset, mask, seed)
        details.update(offset_diagnostics)
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
    manifest = resolved(root, config["manifests"][pool])
    source_count, source_manifest_sha = _manifest_inventory(manifest)
    if role == "positive" and not smoke and source_count != config["targets"]["positive_originals"]:
        raise ValueError("Positive manifest does not match the selected TTS original count; finish generation or declare a separate budget")
    source_identity, provenances = hashlib.sha256(), set()

    def sources_for_admission():
        for row in iter_rows(manifest):
            source_identity.update(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
            provenances.add(row.get("provenance"))
            yield row

    validate_admission(root, sources_for_admission(), config, role, smoke)
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
        with manifest_index(root) as index:
            for row in iter_rows(manifest):
                path = resolved(root, row["path"])
                _, duration = index.file_info((str(path), row["sha256"]), lambda: sf.info(path))
                if row.get("end_s", duration) - row.get("start_s", 0) > 2.0001:
                    raise ValueError("Formal Negative requires fixed 2s windows; run koww windows first")
    qualification = None
    if frontend is None:
        from .frontend import qualified_fast_frontend
        probes = [augment(root, row, role, 0, config["augmentation"], materials["noise"], materials["rir"], noise_index)[0]
                  for row in islice(iter_rows(manifest), 8)]
        frontend, qualification = qualified_fast_frontend(resolved(root, config["frontend_dir"]), probes)
    contract = {"schema_version": 2, "source_manifest_sha256": source_manifest_sha, "role": role,
        "rounds": rounds, "source_rows": source_count, "source_rows_identity_sha256": source_identity.hexdigest(),
        "materials": materials, "augmentation": config["augmentation"],
        "frontend": frontend.identity, "implementation_sha256": digest(__file__), "smoke": smoke}
    expected = source_count * rounds
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
        start = sum(1 for _ in iter_rows(journal))
        path = directory / "features.npy"
        if start:
            array = np.lib.format.open_memmap(path, mode="r+")
            for i, row in enumerate(iter_rows(journal)):
                if row["feature_row"] != i or row["feature_sha256"] != __import__("hashlib").sha256(array[i].tobytes()).hexdigest():
                    raise ValueError("Committed feature prefix changed; do not resume")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(expected, 16, 96))
        if array.shape != (expected, 16, 96):
            raise ValueError("Resume cache shape mismatch")
        started = time.perf_counter()
        pending, rows = [], []
        for source_index, source in enumerate(iter_rows(manifest)):
            if source_index < start // rounds:
                continue
            for round_id in range(rounds):
                index = source_index * rounds + round_id
                if index < start:
                    continue
                if index >= expected:
                    raise ValueError("Source manifest row count changed during extraction")
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
                        item["feature_sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
                    # Commit one bounded batch, not one disk fsync per feature row.
                    with journal.open("a", encoding="utf-8") as handle:
                        for item in rows:
                            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        handle.flush()
                        __import__("os").fsync(handle.fileno())
                    print(f"{role} R{rounds}: {index+1}/{expected}", flush=True)
                    pending, rows = [], []
        del array
        if _manifest_inventory(manifest) != (source_count, source_manifest_sha):
            raise ValueError("Source manifest changed during feature extraction")
        metadata = {"schema_version": 1, "role": role, "split": "train", "feature_rows": expected,
            "original_count": source_count, "rounds": rounds, "frontend_id": identity(frontend.identity),
            "journal_layout": "source_then_round_v1",
            "frontend": frontend.identity, "provenance": "qwen_tts" if role == "positive" else
                next(iter(provenances)) if len(provenances) == 1 else "mixed_reviewed_sources",
            "human_reference_used": False, "feature_sha256": digest(path), "journal_sha256": digest(journal),
            "contract_sha256": digest(contract_path), "split_audit_passed": True,
            "unique_source_hours": audit["pools"][pool]["unique_source_hours"], "smoke_test_only": smoke,
            "source_time_scope": "Union of declared parent coordinates where supplied; caption offsets are not sample-exact original alignment",
            "elapsed_this_invocation_s": time.perf_counter() - started, "created_at": now(),
            "quality_review": "accepted" if not smoke else "unapproved_smoke_only"}
        write_json(directory / "metadata.json", metadata)
        return verify_cache(directory)


def _verify_feature_payload(feature_path, expected_sha256, feature_rows):
    """Verify every file byte and every stored float in one sequential read.

    NumPy's public NPY reader validates v1/v2/v3 headers, payload bounds and
    object rejection. Its mapping is used only for header metadata, never
    for payload access. Raw C/F-order data has the same finite predicate;
    byte order and float precision stay exactly as declared in the header.
    Trailer bytes, which NumPy permits, are hashed but are not array values.
    """
    try:
        header = np.lib.format.open_memmap(feature_path, mode="r")
    except (ValueError, OSError, EOFError):
        # Match the existing hash-first rejection when a damaged header or
        # truncated file also has a mismatching digest.
        if digest(feature_path) != expected_sha256:
            raise ValueError("Feature payload hash changed")
        raise
    shape, dtype, data_offset, payload_bytes = header.shape, header.dtype, int(header.offset), int(header.nbytes)
    del header
    if shape != (feature_rows, 16, 96) or not np.issubdtype(dtype, np.floating):
        if digest(feature_path) != expected_sha256:
            raise ValueError("Feature payload hash changed")
        raise ValueError("Invalid feature shape/dtype")
    hasher = __import__("hashlib").sha256()
    finite, complete, pending = True, True, b""
    with Path(feature_path).open("rb") as stream:
        remaining = data_offset
        while remaining:
            block = stream.read(min(remaining, FEATURE_SCAN_BLOCK_BYTES))
            if not block:
                complete = False
                break
            hasher.update(block)
            remaining -= len(block)
        remaining = payload_bytes
        while complete and remaining:
            block = stream.read(min(remaining, FEATURE_SCAN_BLOCK_BYTES))
            if not block:
                complete = False
                break
            hasher.update(block)
            remaining -= len(block)
            numeric = pending + block if pending else block
            whole_bytes = len(numeric) // dtype.itemsize * dtype.itemsize
            if whole_bytes:
                values = np.frombuffer(numeric, dtype=dtype, count=whole_bytes // dtype.itemsize)
                # Check every block even if an earlier block was nonfinite.
                finite = bool(np.isfinite(values).all()) and finite
            pending = numeric[whole_bytes:]
        for block in iter(lambda: stream.read(FEATURE_SCAN_BLOCK_BYTES), b""):
            hasher.update(block)
    if hasher.hexdigest() != expected_sha256:
        raise ValueError("Feature payload hash changed")
    if not complete or remaining or pending:
        raise ValueError("Invalid feature payload length")
    if not finite:
        raise ValueError("Nonfinite feature values")
    return shape, dtype


def _round_error(entries, count, rounds):
    if count != rounds or [r[0] for r in entries] != list(range(rounds)) or len({r[1] for r in entries}) != rounds:
        return "Missing rounds or repeated augmentation seeds"
    if rounds > 1 and len({r[2] for r in entries}) != rounds:
        return "Repeated PCM across rounds; inspect augmentation diversity"
    return None


def _verify_journal(directory, meta):
    """Check old interleaved journals and new ordered journals without full lists."""
    journal = directory / "augmentation.jsonl"
    count, order_valid, error = 0, True, None
    rounds = meta["rounds"]
    with manifest_index(directory) as index:
        if meta.get("journal_layout") == "source_then_round_v1":
            current, entries, group_count = None, [], 0

            def finish_group():
                nonlocal error
                if group_count:
                    problem = _round_error(entries, group_count, rounds)
                    error = error or problem

            for position, row in enumerate(iter_rows(journal)):
                count += 1
                order_valid = order_valid and row["feature_row"] == position
                sample = row["sample_id"]
                if not group_count or sample != current:
                    finish_group()
                    if not index.unique("feature_source", sample):
                        error = error or "Missing rounds or repeated augmentation seeds"
                    current, entries, group_count = sample, [], 0
                group_count += 1
                if len(entries) < rounds:
                    entries.append((row["round_id"], row["seed"], row["pcm_sha256"]))
            finish_group()
        else:
            # Original caches permitted noncontiguous originals. Keep that
            # compatibility while putting the grouping and sort on disk.
            index.db.execute("CREATE TABLE journal(sample TEXT, position INTEGER, round_id TEXT, seed TEXT, pcm TEXT)")
            for position, row in enumerate(iter_rows(journal)):
                count += 1
                order_valid = order_valid and row["feature_row"] == position
                index.db.execute("INSERT INTO journal VALUES (?,?,?,?,?)", (
                    json.dumps(row["sample_id"]), position, json.dumps(row["round_id"]),
                    json.dumps(row["seed"]), row["pcm_sha256"]))
            index.db.execute("CREATE INDEX journal_sample ON journal(sample)")
            for sample, group_count in index.db.execute("SELECT sample,COUNT(*) FROM journal GROUP BY sample ORDER BY MIN(position)"):
                if group_count != rounds:
                    error = "Missing rounds or repeated augmentation seeds"
                    break
                entries = [(json.loads(a), json.loads(b), c) for a, b, c in index.db.execute(
                    "SELECT round_id,seed,pcm FROM journal WHERE sample=? ORDER BY position", (sample,))]
                error = _round_error(entries, group_count, rounds)
                if error:
                    break
    if count != meta["feature_rows"] or meta["feature_rows"] != meta["original_count"] * rounds:
        raise ValueError("Actual output is not N×rounds")
    if not order_valid:
        raise ValueError("Feature order changed")
    if error:
        raise ValueError(error)


def verify_cache(directory):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    feature_path = resolved(directory, meta.get("feature_path", "features.npy"))
    shape, dtype = _verify_feature_payload(feature_path, meta["feature_sha256"], meta["feature_rows"])
    if meta.get("imported_precomputed"):
        from .assets import GENERIC_SHA
        if meta.get("role") != "generic_negative" or meta["feature_sha256"] != GENERIC_SHA or meta["feature_rows"] != 5625000 or dtype != np.float16:
            raise ValueError("Unsupported imported feature cache")
        return meta
    if digest(directory / "augmentation.jsonl") != meta["journal_sha256"] or digest(directory / "contract.json") != meta["contract_sha256"]:
        raise ValueError("Feature journal/contract changed")
    _verify_journal(directory, meta)
    return meta


def compare_rounds(first, second):
    a, b = Path(first), Path(second)
    ma, mb = verify_cache(a), verify_cache(b)
    ca, cb = read_json(a / "contract.json"), read_json(b / "contract.json")
    ca.pop("rounds"); cb.pop("rounds")
    if ca != cb:
        raise ValueError("R comparison changed sources, noise distribution, frontend or code")
    with manifest_index(a) as index:
        for table, directory in (("rounds_a", a), ("rounds_b", b)):
            index.db.execute(f"CREATE TABLE {table}(sample TEXT, round_id TEXT, seed TEXT, pcm TEXT, feature TEXT, PRIMARY KEY(sample,round_id)) WITHOUT ROWID")
            index.db.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?)", (
                (json.dumps(r["sample_id"]), json.dumps(r["round_id"]), json.dumps(r["seed"]), r["pcm_sha256"], r["feature_sha256"])
                for r in iter_rows(directory / "augmentation.jsonl")))
        shared = index.db.execute("SELECT COUNT(*) FROM rounds_a a JOIN rounds_b b ON a.sample=b.sample AND a.round_id=b.round_id").fetchone()[0]
        if shared != ma["original_count"] * min(ma["rounds"], mb["rounds"]):
            raise ValueError("Incomplete nested round coverage")
        different = index.db.execute("""
            SELECT 1 FROM rounds_a a JOIN rounds_b b ON a.sample=b.sample AND a.round_id=b.round_id
            WHERE a.seed!=b.seed OR a.pcm!=b.pcm OR a.feature!=b.feature LIMIT 1
        """).fetchone()
        if different:
            raise ValueError("Shared augmentation/feature differs")
    return {"passed": True, "shared_original_rounds": shared, "only_rounds_changed": True}
