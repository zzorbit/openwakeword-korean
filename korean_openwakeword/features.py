"""Deterministic N×R cache with explicit counts, journal and exact resume."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from itertools import islice
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve

from .audio import active_speech_mask, pcm16, rms, noise_scale, measured_snr
from .checkpoint_io import write_checkpoint_json
from .common import digest, identity, iter_rows, now, read_json, resolved, write_json, writer_lock
from .data import audit_project, rows_for, validate_admission, seek_audio, source_key
from .streaming import manifest_index
from .project import load_project
from .storage import project_storage
from .task_progress import emit as report_task_progress

MAX_NOISE_OFFSET_ATTEMPTS = 64
MAX_NOISE_SOURCE_ATTEMPTS = 8
MIN_NOISE_RMS = 1e-8
FEATURE_SCAN_BLOCK_BYTES = 8 * 1024 * 1024
# The previous writer used identical augmentation and frontend math.
_PRIOR_FEATURE_WRITERS = {
    "f86b59f800747b10e0a1b87abfe4f2313a6fccbcc1606937a06e7e396d0322b6",
    "548dcd0cd13e0fa84918d3a48c2bf2fe2215b3aba1359e93d18d58f22b66a44b",
    "8add34a93580ca1c89b88099138d4b7189265ece60ba636ffb8113d28d3840b9",
    "4c8bf598d8031e3b0e231006772c7963f27befe77702329c3426235501cf77eb",
    "8ebce10167d60b8f4fbdfd948e7fdf42137e77f6c0d2a04476d7b62eccff3313",
    "e7932719d5da6332151e4cc28cc913e5040d0b41d6263053dc85c03bd7346637",
}


class FeatureGenerationPaused(RuntimeError):
    """Feature preparation stopped at a durable row boundary."""


def _check_feature_pause(pause_requested):
    if pause_requested is not None and pause_requested():
        raise FeatureGenerationPaused("특징 추출을 일시정지했습니다.")


def _feature_digest(path, pause_requested):
    if pause_requested is None:
        return digest(path)
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(FEATURE_SCAN_BLOCK_BYTES), b""):
            _check_feature_pause(pause_requested)
            hasher.update(block)
    return hasher.hexdigest()


def _resume_feature_journal(journal, checkpoint, array, contract_sha, pause_requested=None):
    """Validate the durable prefix before discarding an interrupted append."""
    record = read_json(checkpoint) if checkpoint.exists() else None
    if record is not None and (record["contract_sha256"] != contract_sha
            or record["schema_version"] != 1 or not 0 < record["rows"] <= len(array)
            or record["journal_bytes"] <= 0):
        raise ValueError("Invalid feature checkpoint")
    hasher = hashlib.sha256()
    rows, offset = 0, 0
    with journal.open("a+b") as stream:
        stream.seek(0)
        while record is None or offset < record["journal_bytes"]:
            if rows % 512 == 0:
                _check_feature_pause(pause_requested)
            raw = stream.readline()
            if not raw.endswith(b"\n"):
                break
            row = json.loads(raw)
            if (rows >= len(array) or row["feature_row"] != rows
                    or row["feature_sha256"] != hashlib.sha256(array[rows].tobytes()).hexdigest()):
                raise ValueError("Committed feature prefix changed; do not resume")
            hasher.update(raw)
            rows += 1
            offset += len(raw)
        if record is not None and (rows, offset, hasher.hexdigest()) != (
                record["rows"], record["journal_bytes"], record["journal_sha256"]):
            raise ValueError("Committed feature journal changed; do not resume")
        if stream.seek(0, os.SEEK_END) != offset:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())
    return rows, offset, hasher


def _prepared_feature_batches(items, prepare, batch_size, workers, pause_requested):
    """Keep one bounded batch queued while the current batch is embedded."""
    if workers == 1:
        while (batch := list(islice(items, batch_size))):
            _check_feature_pause(pause_requested)
            yield [prepare(item) for item in batch]
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="feature-audio") as executor:
        pending = []
        try:
            pending = [executor.submit(prepare, item) for item in islice(items, batch_size)]
            while pending:
                _check_feature_pause(pause_requested)
                ready = [future.result() for future in pending]
                pending = [executor.submit(prepare, item) for item in islice(items, batch_size)]
                yield ready
        finally:
            for future in pending:
                future.cancel()


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


def prepare_noise_index(settings, noises, *, seconds=2):
    """Compile the unchanged sampling distribution once, rather than per draw."""
    available = {}
    for category in settings["category_weights"]:
        group = [n for n in noises if n["category"] == category]
        if seconds > 2:
            # Favor one uninterrupted recording in the selected category.
            contiguous = [n for n in group if
                          (n.get("end_s") or n.get("duration_s") or 0) - n.get("start_s", 0) >= seconds]
            group = contiguous or group
        weights = np.asarray([settings["korean_speech_weight"]
            if category == "competing_speech" and n.get("language") in ("Korean", "ko", "kor") else 1
            for n in group], float)
        available[category] = (group, weights / weights.sum() if len(weights) else weights)
    return available


def _extend_background(samples, size):
    overlap = min(160, len(samples) // 4) if size > 32000 else 0
    if not overlap:
        return np.tile(samples, int(np.ceil(size / len(samples))))[:size]
    fade = np.linspace(0, 1, overlap)
    seam = (1 - fade) * samples[-overlap:] + fade * samples[:overlap]
    period = np.concatenate((seam, samples[overlap:-overlap]))
    lead = samples[:-overlap]
    return np.concatenate((lead, np.tile(period, int(np.ceil((size - len(lead)) / len(period))))))[:size]


class _SilentNoiseWindow(ValueError):
    """Every sampled offset in an admitted noise source was silent."""


def _non_silent_noise_window(root, noise, samples, offset, mask, seed, size=32000):
    """Retry only offsets within the same selected source using a private RNG.

    The ordinary sampling RNG selected category/file/first offset already.
    Do not consume it here: SNR, gain, RIR and future sampling must retain
    their original draws even if the first window is silent. File-level QC
    does not imply a non-silent random 2s window or clean active region.
    """
    first_offset = offset
    scope = "clean_active_region" if mask is not None else (
        "whole_2s_clip" if size == 32000 else "whole_context_clip")
    retry_rng = None
    first_rms = None
    for attempt in range(1, MAX_NOISE_OFFSET_ATTEMPTS + 1):
        if len(samples) < size:
            samples = _extend_background(samples, size)
        samples = samples[:size]
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
                    source_key(noise), noise.get("logical_audio_sha256", noise["sha256"]), scope])[:16], 16)
                retry_rng = np.random.default_rng(retry_seed)
            # Missing, corrupt and nonfinite audio still propagate normally.
            samples, offset = seek_audio(root, noise, size / 16000, retry_rng)
    raise _SilentNoiseWindow(
        f"No non-silent noise window for source {source_key(noise)} "
        f"(category={noise['category']}, scope={scope}) after "
        f"{MAX_NOISE_OFFSET_ATTEMPTS} offsets; first_offset_s={first_offset}. "
        "Use a versioned valid source interval; clean fallback is disabled."
    )


def _background_window(root, group, probabilities, rng, mask, seed, size):
    """Keep the normal draw and try other admitted files only after exhaustion."""
    selected = int(rng.choice(len(group), p=probabilities))
    source_rng = rng
    weights = None
    rejected = []
    for attempt in range(MAX_NOISE_SOURCE_ATTEMPTS):
        noise = group[selected]
        samples, offset = seek_audio(root, noise, size / 16000, source_rng)
        try:
            samples, offset, diagnostics = _non_silent_noise_window(root, noise, samples, offset, mask, seed, size)
        except _SilentNoiseWindow as error:
            if weights is None:
                weights = probabilities.copy()
                retry_seed = int(identity([seed, "non_silent_noise_source_v1", noise["category"]])[:16], 16)
                source_rng = np.random.default_rng(retry_seed)
            rejected.append(source_key(noise))
            for index, candidate in enumerate(group):
                if candidate["path"] == noise["path"]:
                    weights[index] = 0
            remaining = weights.sum()
            if not remaining or attempt + 1 == MAX_NOISE_SOURCE_ATTEMPTS:
                if len(rejected) == 1:
                    raise
                raise ValueError(f"{error} No usable {noise['category']} background after "
                                 f"checking {len(rejected)} admitted files.") from error
            selected = int(source_rng.choice(len(group), p=weights / remaining))
        else:
            if rejected:
                diagnostics["noise_rejected_source_ids"] = rejected
            return noise, samples, offset, diagnostics


def augment(root, row, role, round_id, settings, noises, rirs, noise_index=None, *, size=32000):
    seed = seed_for(settings["seed"], role, row["sample_id"], round_id)
    rng = np.random.default_rng(seed)
    clip = np.zeros(size, dtype=np.float64)
    duration = size / 16000
    if role == "positive":
        a, b = row["call_start_s"], row["call_end_s"]
        source = {**row, "start_s": a, "end_s": b}
        audio, _ = seek_audio(root, source, b - a)
        if len(audio) > size - 2560:
            raise ValueError(f"Complete word must fit a {duration:g}s clip with 80ms head/tail; quarantine, do not cut syllables")
        start = int(rng.integers(1280, size - len(audio) - 1280 + 1))
        clip[start:start + len(audio)] = audio
        bounds = [start / 16000, (start + len(audio)) / 16000]
        mask = active_speech_mask(clip, *bounds)
    else:
        audio, _ = seek_audio(root, row, duration)
        clip[:min(len(audio), size)] = audio[:size]
        bounds = None
        mask = active_speech_mask(clip, 0, min(len(audio), size) / 16000) if row.get("speech") and rms(clip) > 1e-8 else None
    details = {"sample_id": row["sample_id"], "source_id": source_key(row), "round_id": round_id,
               "seed": seed, "call_bounds_in_clip": bounds, "noise_id": None, "rir_id": None,
               "snr_db": None, "measured_snr_db": None}
    if rirs and rng.random() < settings["rir_probability"]:
        rir = rirs[int(rng.integers(len(rirs)))]
        impulse, _ = seek_audio(root, rir, 2)
        norm = np.sqrt(np.sum(impulse ** 2))
        if norm <= 1e-8:
            raise ValueError("Silent RIR")
        clip = fftconvolve(clip, impulse / norm)[:size]
        details["rir_id"] = source_key(rir)
    draw = rng.random()
    if noises and draw >= settings["clean_probability"]:
        categories = list(settings["category_weights"])
        available = prepare_noise_index(settings, noises, seconds=duration) if noise_index is None else noise_index
        if any(not group for group, _ in available.values()):
            raise ValueError("All declared noise categories need admitted material; do not silently renormalize")
        weights = np.asarray([settings["category_weights"][k] for k in categories], float)
        category = categories[int(rng.choice(len(categories), p=weights / weights.sum()))]
        group, probabilities = available[category]
        noise, samples, offset, offset_diagnostics = _background_window(root, group, probabilities, rng, mask, seed, size)
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



def _feature_validation_rows(rows, total, phase, label, unit, pause_requested):
    """Report a row once its consumer has fully validated it."""
    report_task_progress(phase, label, 0, total, unit, eta_eligible=False)
    for index, row in enumerate(rows, 1):
        yield row
        if index == 1 or index % 64 == 0 or index == total:
            _check_feature_pause(pause_requested)
            report_task_progress(phase, label, index, total, unit, eta_eligible=False)


@project_storage
def build_features(project, role, output=None, rounds=None, batch_size=32, smoke=False, resume=False, frontend=None,
                   *, pause_requested=None):
    _check_feature_pause(pause_requested)
    report_task_progress("feature_admission", "특징 추출 원본·승인 자료 확인", eta_eligible=False)
    root, config = load_project(project)
    from .training_context import clip_samples, clip_seconds, feature_frames
    frames = feature_frames(config["runtime"])
    shape = (frames, 96)
    size = clip_samples(frames)
    seconds = clip_seconds(frames)
    writer_sha = digest(__file__)
    if frames == 16:
        from .studio import _context_profile_compatible_digest
        writer_sha = _context_profile_compatible_digest(writer_sha)
    pool = "positive_train" if role == "positive" else role
    if pool not in ("positive_train", "generic_negative", "korean_negative", "hard_negative"):
        raise ValueError("Only declared training pools are supported")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Feature batch_size must be a positive integer")
    if rounds is None:
        rounds = config["augmentation_rounds"] if role == "positive" else 1
    if type(rounds) is not int or rounds not in (1, 3, 5) or role != "positive" and rounds != 1:
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

    source_rows = _feature_validation_rows(sources_for_admission(), source_count, "feature_admission",
                                          "특징 추출 원본 승인 검사", "개 원본", pause_requested)
    validate_admission(root, source_rows, config, role, smoke)
    report_task_progress("feature_source_audit", "특징 추출 자료의 학습·평가 분리 확인", eta_eligible=False)
    audit = audit_project(project)
    if not audit["passed"]:
        raise ValueError("Split/source audit failed: " + str(audit["errors"][:5]))
    materials = {}
    for material, label in (("noise", "소음 원본 승인 검사"), ("rir", "잔향 원본 승인 검사")):
        _check_feature_pause(pause_requested)
        report_task_progress("feature_material_audit", label, eta_eligible=False)
        _, rows = rows_for(root, config, material)
        if rows:
            review = _feature_validation_rows(rows, len(rows), "feature_material_audit", label,
                                              "개 원본", pause_requested)
            validate_admission(root, review, config, material, smoke)
        materials[material] = rows
    if not materials["noise"] and config["augmentation"]["clean_probability"] < 1:
        raise ValueError("Non-clean augmentation needs reviewed noise materials")
    noise_index = prepare_noise_index(config["augmentation"], materials["noise"], seconds=seconds)
    if output:
        directory = Path(output).resolve()
    elif role == "positive" and rounds != config["augmentation_rounds"]:
        directory = root / "features" / f"positive_N{config['targets']['positive_originals']}_R{rounds}"
    else:
        directory = resolved(root, config["pool_features"][role])
    if role != "positive" and not smoke:
        import soundfile as sf
        windows = _feature_validation_rows(iter_rows(manifest), source_count, "feature_window_validation",
                                          "비호출 음원 길이 검사", "개 구간", pause_requested)
        with manifest_index(root) as index:
            for row in windows:
                path = resolved(root, row["path"])
                _, duration = index.file_info((str(path), row["sha256"]), lambda: sf.info(path))
                if row.get("end_s", duration) - row.get("start_s", 0) > seconds + .0001:
                    raise ValueError(f"Formal Negative requires fixed {seconds:g}s windows; run koww windows first")
    def prepare_audio(row, round_id):
        args = (root, row, role, round_id, config["augmentation"], materials["noise"], materials["rir"], noise_index)
        return augment(*args) if frames == 16 else augment(*args, size=size)

    qualification = None
    if frontend is None:
        report_task_progress("feature_frontend", "특징 추출 모델 준비·동일성 확인", eta_eligible=False)
        from .frontend import qualified_fast_frontend
        probes = [prepare_audio(row, 0)[0] for row in islice(iter_rows(manifest), 8)]
        frontend, qualification = qualified_fast_frontend(resolved(root, config["frontend_dir"]), probes,
                                                         **({"frames": frames} if frames != 16 else {}))
    if frames != 16 and frontend.identity.get("feature_shape") != list(shape):
        raise ValueError("Selected frontend cannot extract the configured wake-word context")
    contract = {"schema_version": 2, "source_manifest_sha256": source_manifest_sha, "role": role,
        "rounds": rounds, "source_rows": source_count, "source_rows_identity_sha256": source_identity.hexdigest(),
        "materials": materials, "augmentation": config["augmentation"],
        "frontend": frontend.identity, "implementation_sha256": writer_sha, "smoke": smoke}
    expected = source_count * rounds
    if expected < 1 or batch_size < 1:
        raise ValueError("Positive cache and batch sizes required")
    with writer_lock(directory):
        contract_path = directory / "contract.json"
        if contract_path.exists():
            prior = read_json(contract_path)
            compatible = prior.get("implementation_sha256") in _PRIOR_FEATURE_WRITERS
            expected_contract = {**contract, "implementation_sha256": prior["implementation_sha256"]} if compatible else contract
            if not resume or prior != expected_contract:
                raise ValueError("Immutable cache exists or resume identity changed")
            if (directory / "metadata.json").exists():
                return verify_cache(directory, pause_requested=pause_requested) if pause_requested else verify_cache(directory)
        else:
            write_json(contract_path, contract)
        if qualification is not None and not (directory / "frontend_qualification.json").exists():
            write_json(directory / "frontend_qualification.json", qualification)
        journal = directory / "augmentation.jsonl"
        checkpoint = directory / "augmentation_checkpoint.json"
        contract_sha = digest(contract_path)
        path = directory / "features.npy"
        if checkpoint.exists() or journal.exists() and journal.stat().st_size:
            if not path.is_file() or not journal.is_file():
                raise ValueError("Missing resumable feature artifacts")
            array = np.lib.format.open_memmap(path, mode="r+")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(expected, *shape))
        if array.shape != (expected, *shape) or array.dtype != np.float32:
            raise ValueError("Resume cache shape mismatch")
        start, journal_bytes, journal_hash = _resume_feature_journal(journal, checkpoint, array, contract_sha, pause_requested)
        started = time.perf_counter()
        def items_to_prepare():
            sources = islice(iter_rows(manifest), start // rounds, None)
            for source_index, source in enumerate(sources, start // rounds):
                for round_id in range(rounds):
                    index = source_index * rounds + round_id
                    if index >= expected:
                        raise ValueError("Source manifest row count changed during extraction")
                    if index >= start:
                        yield index, source, round_id

        def prepare(item):
            index, source, round_id = item
            samples, info = prepare_audio(source, round_id)
            return samples, {"feature_row": index, **info}

        workers = 2 if frontend.identity.get("provider") == "CUDAExecutionProvider" else 1
        timings = dict(audio_wait=0.0, frontend=0.0, checkpoint=0.0)
        ready_at = time.perf_counter()
        committed = start
        with path.open("r+b") as feature_file, journal.open("ab") as journal_file:
            for batch in _prepared_feature_batches(items_to_prepare(), prepare, batch_size, workers, pause_requested):
                stamp = time.perf_counter()
                timings["audio_wait"] += stamp - ready_at
                samples, rows = zip(*batch)
                values = frontend.embed(np.asarray(samples, dtype=np.int16))
                if values.shape != (len(rows), *shape) or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError("Invalid frontend output")
                checkpoint_started = time.perf_counter()
                timings["frontend"] += checkpoint_started - stamp
                array[committed:committed + len(rows)] = values
                array.flush()
                os.fsync(feature_file.fileno())
                for item, value in zip(rows, values):
                    item["feature_sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
                encoded = "".join(json.dumps(item, ensure_ascii=False) + os.linesep for item in rows).encode("utf-8")
                journal_file.write(encoded)
                journal_file.flush()
                os.fsync(journal_file.fileno())
                journal_hash.update(encoded)
                journal_bytes += len(encoded)
                committed += len(rows)
                write_checkpoint_json(checkpoint, {"schema_version": 1, "contract_sha256": contract_sha,
                    "rows": committed, "journal_bytes": journal_bytes, "journal_sha256": journal_hash.hexdigest()})
                timings["checkpoint"] += time.perf_counter() - checkpoint_started
                print(f"{role} R{rounds}: {committed}/{expected}", flush=True)
                report_task_progress("feature_extraction", {
                    "generic_negative": "일반 비호출", "korean_negative": "한국어 비호출",
                    "positive": "호출어 음성", "hard_negative": "유사어 음성"}[role],
                    committed, expected, "개 특징", eta_eligible=True)
                ready_at = time.perf_counter()
        del array
        if committed != expected or _manifest_inventory(manifest) != (source_count, source_manifest_sha):
            raise ValueError("Source manifest changed during feature extraction")
        metadata = {"schema_version": 1, "role": role, "split": "train", "feature_rows": expected,
            "original_count": source_count, "rounds": rounds, "frontend_id": identity(frontend.identity),
            "journal_layout": "source_then_round_v1",
            "frontend": frontend.identity, "provenance": "qwen_tts" if role == "positive" else
                next(iter(provenances)) if len(provenances) == 1 else "mixed_reviewed_sources",
            "human_reference_used": False, "feature_sha256": _feature_digest(path, pause_requested),
            "journal_sha256": journal_hash.hexdigest(),
            "contract_sha256": contract_sha, "split_audit_passed": True,
            "unique_source_hours": audit["pools"][pool]["unique_source_hours"], "smoke_test_only": smoke,
            "source_time_scope": "Union of declared parent coordinates where supplied; caption offsets are not sample-exact original alignment",
            "elapsed_this_invocation_s": time.perf_counter() - started,
            "extraction_pipeline": {"augmentation_workers": workers, "elapsed_seconds": timings},
            "created_at": now(),
            "quality_review": "accepted" if not smoke else "unapproved_smoke_only"}
        write_json(directory / "metadata.json", metadata)
        return verify_cache(directory, pause_requested=pause_requested) if pause_requested else verify_cache(directory)


def _verify_feature_payload(feature_path, expected_sha256, feature_rows, pause_requested=None, *, frames=16):
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
        if _feature_digest(feature_path, pause_requested) != expected_sha256:
            raise ValueError("Feature payload hash changed")
        raise
    shape, dtype, data_offset, payload_bytes = header.shape, header.dtype, int(header.offset), int(header.nbytes)
    del header
    if shape != (feature_rows, frames, 96) or not np.issubdtype(dtype, np.floating):
        if _feature_digest(feature_path, pause_requested) != expected_sha256:
            raise ValueError("Feature payload hash changed")
        raise ValueError("Invalid feature shape/dtype")
    hasher = __import__("hashlib").sha256()
    finite, complete, pending = True, True, b""
    report_task_progress("feature_cache_verification", "기존 특징 배열 무결성·형식 검사", 0, payload_bytes,
                         "bytes", eta_eligible=False)
    with Path(feature_path).open("rb") as stream:
        remaining = data_offset
        while remaining:
            _check_feature_pause(pause_requested)
            block = stream.read(min(remaining, FEATURE_SCAN_BLOCK_BYTES))
            if not block:
                complete = False
                break
            hasher.update(block)
            remaining -= len(block)
        remaining = payload_bytes
        while complete and remaining:
            _check_feature_pause(pause_requested)
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
            report_task_progress("feature_cache_verification", "기존 특징 배열 무결성·형식 검사",
                                 payload_bytes - remaining, payload_bytes, "bytes", eta_eligible=False)
        for block in iter(lambda: stream.read(FEATURE_SCAN_BLOCK_BYTES), b""):
            _check_feature_pause(pause_requested)
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


def _verify_journal(directory, meta, pause_requested=None):
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
                if position % 512 == 0:
                    _check_feature_pause(pause_requested)
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
                if position % 512 == 0:
                    _check_feature_pause(pause_requested)
                count += 1
                order_valid = order_valid and row["feature_row"] == position
                index.db.execute("INSERT INTO journal VALUES (?,?,?,?,?)", (
                    json.dumps(row["sample_id"]), position, json.dumps(row["round_id"]),
                    json.dumps(row["seed"]), row["pcm_sha256"]))
            index.db.execute("CREATE INDEX journal_sample ON journal(sample)")
            for sample, group_count in index.db.execute("SELECT sample,COUNT(*) FROM journal GROUP BY sample ORDER BY MIN(position)"):
                _check_feature_pause(pause_requested)
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


def verify_cache(directory, *, pause_requested=None):
    _check_feature_pause(pause_requested)
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    feature_path = resolved(directory, meta.get("feature_path", "features.npy"))
    from .training_context import feature_frames
    feature_shape = meta.get("frontend", {}).get("feature_shape", [16, 96])
    frames = 16 if meta.get("imported_precomputed") else feature_frames({"feature_frames": feature_shape[0]})
    shape, dtype = _verify_feature_payload(feature_path, meta["feature_sha256"], meta["feature_rows"], pause_requested,
                                           **({"frames": frames} if frames != 16 else {}))
    if meta.get("imported_precomputed"):
        from .assets import GENERIC_SHA
        if meta.get("role") != "generic_negative" or meta["feature_sha256"] != GENERIC_SHA or meta["feature_rows"] != 5625000 or dtype != np.float16:
            raise ValueError("Unsupported imported feature cache")
        return meta
    report_task_progress("feature_journal_verification", "특징 완료 기록·원본 순서 확인", eta_eligible=False)
    if (_feature_digest(directory / "augmentation.jsonl", pause_requested) != meta["journal_sha256"]
            or digest(directory / "contract.json") != meta["contract_sha256"]):
        raise ValueError("Feature journal/contract changed")
    _verify_journal(directory, meta, pause_requested)
    return meta


def compare_rounds(first, second):
    a, b = Path(first), Path(second)
    ma, mb = verify_cache(a), verify_cache(b)
    ca, cb = read_json(a / "contract.json"), read_json(b / "contract.json")
    ca.pop("rounds"); cb.pop("rounds")
    from .studio import _context_profile_compatible_digest
    writer = digest(__file__)
    known_writers = _PRIOR_FEATURE_WRITERS | {writer, _context_profile_compatible_digest(writer)}
    if ca.get("implementation_sha256") in known_writers and cb.get("implementation_sha256") in known_writers:
        ca.pop("implementation_sha256")
        cb.pop("implementation_sha256")
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
