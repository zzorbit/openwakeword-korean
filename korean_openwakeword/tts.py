"""Qwen standalone generation with synthetic-only references and bounded QC.

No STT model is imported. Pronunciation approval is a pilot/policy decision;
production technical acceptance never masquerades as individual human listening.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import random
import time
import warnings
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import pcm16, active_speech_mask, mask_intervals
from .common import append_row, bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, write_immutable_rows, writer_lock, snapshot_json, require_final_lock
from .task_progress import emit as report_progress, generation_attempt
from . import clone_reference
from .project import load_project
from .reference_attempt import (check_reference_timing, prepare_reference_attempt, publish_reference_attempt,
                                raw_receipt_path, reference_raw_path)
from .storage import project_storage, local_tts_snapshot, storage_environment

# Legacy projects without an explicit sentence keep their original generation
# input. New projects record the shorter default in tts.reference_text.
REFERENCE_TEXT = "오늘은 창밖으로 따뜻한 햇살이 들어옵니다. 천천히 숨을 고르고 자연스러운 목소리로 한국어 문장을 읽습니다."
STYLE_OPTIONS = (
    ("neutral", "natural relaxed speech"), ("friendly", "warm friendly calling"),
    ("urgent", "clear urgent calling"), ("projected", "strong projected vocal style as if calling to someone far away; no echo or distance simulation"),
)

# This is an audited transition from the exact previous generator, not a
# general exception for journals produced by unknown implementations.
LEGACY_CALL_IMPLEMENTATION = "a8a62455fee84260915d489aa2bccb0249ad874c9bfb2496678a5c725292473e"
CALL_RECOVERY_POLICY = "prepared_clone_bounded_recovery_v1"
VOICE_FILL_SCHEDULE = "same_voice_fill_cuda4_v1"
PREFLIGHT_BATCH_POLICY = "preflight_mixed_voice_cuda4_v1"
EAGER_VOICE_FILL_POLICY = "same_voice_cuda_graph_or_eager_v1"
CALL_BATCH_SIZE = 4
ORIGINAL_ATTEMPTS = 3
CLONE_ATTEMPTS = 6
_MAX_CALL_DURATION_S = 8
SHORT_CLONE_POLICY = "qwen_short_clone_v1"
REPAIR_ENDING_POLICY = "call_ending_repair_v1"
CALL_BOUNDARY_POLICY = "qwen_call_boundary_qc_v1"
CANDIDATE_PCM_QC = "stored_pcm16_v1"
_QWEN_CODEC_RATE = 24000
_QWEN_CODEC_FRAME_SAMPLES = 1920
_REFERENCE_ATTEMPTS = 4
_REFERENCE_ATTEMPT_POLICY = "bounded_reference_design_retries_v1"
_REFERENCE_PREPARATION_PREVIOUS_IMPLEMENTATIONS = {
    LEGACY_CALL_IMPLEMENTATION, "9f39f255e364f0186f8e866bfb317dbef90d6c01f3481147c04637b3d0585aee"}


def _generation_evidence_sha256():
    from .call_generation_history import SCHEDULING_ONLY_TTS_DIGESTS

    current = _technical_qc_sha256()
    return SCHEDULING_ONLY_TTS_DIGESTS.get(current, current)


def _technical_qc_sha256():
    from .studio import _compute_dispatch_compatible_digest

    return _compute_dispatch_compatible_digest("tts.py", digest(__file__))


def _clone_reference_evidence_sha256():
    from .call_generation_history import CONDITIONING_ONLY_REFERENCE_DIGESTS

    current = digest(clone_reference.__file__)
    return CONDITIONING_ONLY_REFERENCE_DIGESTS.get(current, current)


def technical_qc(samples, rate):
    samples = np.asarray(samples, dtype=np.float64)
    reasons = []
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        return {"passed": False, "reasons": ["empty_nonfinite_or_nonmono"]}, None
    duration = len(samples) / rate
    if np.max(np.abs(samples)) > .999:
        reasons.append("clipping")
    level = float(np.sqrt(np.mean(samples ** 2)))
    if level < .001:
        reasons.append("near_silence")
    if not .25 <= duration <= _MAX_CALL_DURATION_S:
        reasons.append("duration_outlier")
    d = math.gcd(rate, 16000)
    mono = resample_poly(samples, 16000 // d, rate // d) if rate != 16000 else samples
    if np.max(np.abs(mono)) > 1:
        reasons.append("resampling_clipping")
    bounds = None
    if not reasons:
        try:
            mask = active_speech_mask(mono, 0, len(mono) / 16000)
            ranges = mask_intervals(mask)
            bounds = [max(0, ranges[0][0] - .04), min(len(mono) / 16000, ranges[-1][1] + .04)]
            if bounds[1] - bounds[0] > 1.84:
                reasons.append("does_not_fit_full_call_feature_window")
        except ValueError:
            reasons.append("no_active_audio")
    return {"passed": not reasons, "reasons": reasons, "duration_s": duration, "rms": level,
            "boundary_method": "energy_envelope_candidate_no_semantic_STT", "pronunciation_listened": False}, (mono, bounds)


def encode_candidate_pcm16(converted, qc):
    """Check the waveform that will be persisted before accepting an attempt."""
    samples, bounds = converted
    encoded = pcm16(np.clip(samples, -1, 1))
    if qc["passed"]:
        saved_qc, (_, bounds) = technical_qc(encoded.astype(np.float64) / 32768.0, 16000)
        if not saved_qc["passed"]:
            qc["passed"] = False
            qc["reasons"].extend(saved_qc["reasons"])
    return encoded, bounds


def _load_model(model_path, config, project_root):
    with storage_environment(project_root):
        import torch
        from qwen_tts import Qwen3TTSModel
        local_path = local_tts_snapshot(model_path, config, project_root)
        requested = config["device"]
        backend = os.environ.get("KOWW_STUDIO_TORCH_BACKEND")
        if backend == "cpu":
            requested = "cpu"
        elif backend == "cu128" and not requested.startswith("cuda"):
            requested = "auto"
        automatic = requested == "auto"
        devices = [requested]
        if automatic:
            devices = [] if backend == "cu128" else ["cpu"]
            if torch.cuda.is_available():
                current = torch.cuda.current_device()
                others = [index for index in range(torch.cuda.device_count()) if index != current]
                devices[:0] = [f"cuda:{index}" for index in [current, *others]]
            if not devices:
                raise ValueError("선택한 NVIDIA GPU를 사용할 수 없습니다. CUDA 지원 드라이버와 학습 도구를 확인하세요.")

        def load(device, dtype):
            return Qwen3TTSModel.from_pretrained(local_path, device_map=device, dtype=dtype,
                attn_implementation=config["attention"])

        for device in devices:
            if not device.startswith("cuda"):
                return load(device, torch.float32)
            # The BF16 capability check is scoped to the device receiving the model.
            with torch.cuda.device(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
                try:
                    return load(device, dtype)
                except torch.cuda.OutOfMemoryError:
                    if not automatic:
                        raise
                # The failed loader's traceback can retain tensors until the
                # except block has exited. Release the allocator cache afterwards.
                torch.cuda.empty_cache()
            warnings.warn(f"TTS model did not fit on {device}; trying the next available device.", RuntimeWarning)
        raise ValueError("선택한 GPU의 메모리가 부족해 음성 모델을 불러오지 못했습니다.")


def _qwen_audio_token_limit(model, configured_max_new_tokens, seconds, *, legacy_padding=False):
    """Convert an audio budget for the supported Qwen codec to generation tokens."""
    if type(model).__module__ != "qwen_tts.inference.qwen3_tts_model":
        return None
    try:
        if version("qwen-tts") != "0.1.1" or version("transformers") != "4.57.3":
            return None
    except PackageNotFoundError:
        return None
    if configured_max_new_tokens is None:
        configured_max_new_tokens = (getattr(model, "generate_defaults", None) or {}).get("max_new_tokens", 2048)
    if type(configured_max_new_tokens) is not int or configured_max_new_tokens < 3:
        return None
    core = model.model
    tokenizer = core.speech_tokenizer
    generation = core.talker.generation_config
    if (type(tokenizer.model).__module__ != "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
            or generation.forced_eos_token_id is not None or generation.num_beams != 1):
        return None
    sample_rate = tokenizer.get_output_sample_rate()
    samples_per_frame = tokenizer.get_decode_upsample_rate()
    if (sample_rate, samples_per_frame) != (_QWEN_CODEC_RATE, _QWEN_CODEC_FRAME_SAMPLES):
        return None
    frame_budget = seconds * sample_rate / samples_per_frame
    frames = round(frame_budget) if legacy_padding else math.floor(frame_budget)
    # The initial generation token does not produce a decoded frame.
    return min(configured_max_new_tokens, frames + 1 + int(legacy_padding))


def _short_clone_token_limit(model, configured_max_new_tokens, *, text=None, retry=False, ending_attempt=None):
    characters = sum(char.isalnum() for char in text) if isinstance(text, str) else 0
    if ending_attempt is not None:
        # Resume already-issued ending repair plans with their original budget.
        frames = 40 + 2 * characters + 18 + min(ending_attempt, 3) * 8
        seconds = min(_MAX_CALL_DURATION_S, frames * _QWEN_CODEC_FRAME_SAMPLES / _QWEN_CODEC_RATE)
        return _qwen_audio_token_limit(model, configured_max_new_tokens, seconds, legacy_padding=True)
    previous_frames = 41 + 2 * characters + (12 if retry else 0)
    seconds = min(_MAX_CALL_DURATION_S, math.ceil(previous_frames * _QWEN_CODEC_FRAME_SAMPLES / _QWEN_CODEC_RATE))
    return _qwen_audio_token_limit(model, configured_max_new_tokens, seconds)


def generate_short_clone(model, *, expected_reuses=None, retry=False, termination=None, **kwargs):
    """Use the same codec budget whether a clone runs through the runtime or directly."""
    generate = getattr(model, "generate_short_clone", None)
    if generate is not None:
        if termination is not None:
            kwargs["termination"] = termination
        return generate(expected_reuses=expected_reuses, retry=retry, **kwargs)
    configured = kwargs.get("max_new_tokens")
    limit = _short_clone_token_limit(model, configured, text=kwargs.get("text"), retry=retry)
    if limit is not None:
        kwargs["max_new_tokens"] = limit
    if termination is None:
        wavs, rate = model.generate_voice_clone(**kwargs)
    else:
        with _capture_qwen_termination(model, limit) as evidence:
            wavs, rate = model.generate_voice_clone(**kwargs)
        termination.update(evidence)
    return wavs, rate, limit


def short_clone_hit_limit(waveform, rate, token_limit):
    """Reject the codec boundary when the waveform cannot prove a natural stop."""
    # Qwen returns decoded audio and omits the stop marker. The prefill does
    # not emit audio, so a full output can contain max_new_tokens - 1 frames.
    # Clone reference slicing can round the output boundary by one sample.
    return (type(token_limit) is int and rate == _QWEN_CODEC_RATE
            and np.size(waveform) >= (token_limit - 1) * _QWEN_CODEC_FRAME_SAMPLES - 1)


def check_short_clone_audio(waveform, rate, token_limit, termination):
    """Reject incomplete inference and excessive inside pauses, preserving quiet syllables."""
    stop = termination.get("termination", "unobserved")
    failure = None
    if stop == "token_limit" or stop == "unobserved" and short_clone_hit_limit(waveform, rate, token_limit):
        failure = "generation_length_limit"
    elif stop == "without_eos":
        failure = "generation_stopped_without_eos"
    trim = None
    if token_limit is not None and failure is None:
        samples = np.asarray(waveform)
        if samples.ndim == 1 and len(samples) and np.isfinite(samples).all():
            loud = np.flatnonzero(np.abs(samples) >= 0.0005)
            if len(loud):
                margin = round(0.16 * rate)
                start = max(0, int(loud[0]) - margin)
                end = min(len(samples), int(loud[-1]) + 1 + margin)
                start = start if start >= 0.25 * rate else 0
                end = end if len(samples) - end >= 0.25 * rate else len(samples)
                if start or end < len(samples):
                    trim = {"leading_samples": start, "trailing_samples": len(samples) - end, "sample_rate": rate}
                    waveform = samples[start:end]
    qc, converted = technical_qc(waveform, rate)
    if converted is None:
        return waveform, qc, converted
    if trim is not None:
        qc["edge_silence_removed"] = trim
    if token_limit is not None:
        qc["generation_termination"] = stop
    if qc["passed"] and token_limit is not None and failure is None:
        mono = converted[0]
        intervals = mask_intervals(active_speech_mask(mono, 0, len(mono) / 16000))
        if any(right[0] - left[1] >= 0.5 for left, right in zip(intervals, intervals[1:])):
            failure = "long_pause_inside_utterance"
        elif stop != "eos" and _ending_signal(waveform, rate)["active_at_end"] and _ending_signal(mono, 16000)["active_at_end"]:
            failure = "waveform_cuts_active_ending"
    if failure:
        qc["passed"] = False
        qc["reasons"].append(failure)
    return waveform, qc, converted


def requires_ending_check(scope, family):
    """Apply stricter acceptance only to the families named by the saved review."""
    if scope is None or scope.get("ending_repair") != REPAIR_ENDING_POLICY:
        return False
    selected = scope.get("ending_repair_families")
    return selected is None or family in selected


@contextmanager
def _capture_qwen_termination(model, limit):
    """Observe the outer talker's sampled sequence before Qwen discards EOS."""
    evidence = dict(termination="unobserved", generated_tokens=None, token_limit=limit)
    if limit is None or not callable(getattr(model.model.talker, "generate", None)):
        yield evidence
        return
    talker = model.model.talker
    generate = talker.generate
    original = vars(talker).get("generate")

    def checked_generate(*args, **kwargs):
        nonlocal observed
        result = generate(*args, **kwargs)
        observed = (getattr(result, "sequences", None), kwargs.get("max_new_tokens"), kwargs.get("eos_token_id"))
        return result

    observed = None
    talker.generate = checked_generate
    try:
        yield evidence
        if observed is not None:
            sequence, observed_limit, eos_token = observed
            if (sequence is not None and sequence.ndim == 2 and sequence.shape[0] == 1
                    and sequence.shape[1] and type(observed_limit) is int and type(eos_token) is int):
                # Qwen passes inputs_embeds; sequence contains only new tokens.
                count = int(sequence.shape[1])
                ending = "eos" if sequence[0, -1].item() == eos_token else (
                    "token_limit" if count >= observed_limit else "without_eos")
                evidence.update(termination=ending, generated_tokens=count, token_limit=observed_limit)
    finally:
        if original is None:
            del talker.generate
        else:
            talker.generate = original


def _generate_ending_repair_clone(model, attempt, **kwargs):
    """Leave room for the last syllable on the first targeted attempt."""
    specialized = getattr(model, "generate_ending_repair_clone", None)
    if specialized is not None:
        return specialized(attempt=attempt, **kwargs)
    limit = _short_clone_token_limit(model, kwargs.get("max_new_tokens"), text=kwargs.get("text"), ending_attempt=attempt)
    if limit is not None:
        kwargs["max_new_tokens"] = limit
    with _capture_qwen_termination(model, limit) as evidence:
        wavs, rate = model.generate_voice_clone(**kwargs)
    return wavs, rate, evidence


def _ending_signal(samples, rate, *, pcm=False):
    """Measure a hard cut without assuming that a quiet tail proves pronunciation."""
    encoded = np.asarray(samples, dtype="<i2" if pcm else "<f4")
    signal = encoded.astype(np.float64)
    if pcm:
        signal /= 32768

    def rms(values):
        return round(float(np.sqrt(np.mean(values ** 2))), 8)

    level = rms(signal)
    tail = rms(signal[-max(1, rate * 8 // 1000):])
    edge = rms(signal[-max(1, rate * 2 // 1000):])
    return {"sample_rate": rate, "frames": len(encoded), "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
            "tail_rms": tail, "edge_rms": edge,
            "active_at_end": tail >= max(.015, .4 * level) and edge >= .75 * tail}


def _verify_ending_check(root, row, word):
    evidence = row.get("ending_check")
    if (row.get("ending_check_required") is not True or not isinstance(evidence, dict)
            or evidence.get("policy") != REPAIR_ENDING_POLICY or evidence.get("expected_last_syllable") != word[-1]
            or evidence.get("requires_listening") is not True
            or evidence.get("termination") not in ("eos", "token_limit", "without_eos", "unobserved")
            or evidence.get("token_limit") != row.get("generation_token_limit")):
        raise ValueError("End-of-call generation evidence changed")
    count, limit = evidence.get("generated_tokens"), evidence["token_limit"]
    if (evidence["termination"] == "unobserved" and count is not None
            or evidence["termination"] != "unobserved" and (
                type(count) is not int or type(limit) is not int or not 1 <= count <= limit
                or (evidence["termination"] == "token_limit") != (count == limit and evidence["termination"] != "eos"))):
        raise ValueError("End-of-call token count changed")
    if row["qc"]["passed"] and evidence["termination"] in ("token_limit", "without_eos"):
        raise ValueError("A cut-off ending cannot pass technical checks")
    _verify_ending_signal(root, row, evidence)


def _waveform_cuts_ending(evidence):
    # A naturally stopped codec can still leave energy at the file boundary.
    # Those targeted takes have a separate, mandatory listening confirmation.
    return (evidence.get("termination") != "eos"
            and evidence["synthesized"]["active_at_end"] and evidence["stored_pcm"]["active_at_end"])


def _verify_ending_signal(root, row, evidence):
    if row.get("path") is None:
        if evidence.get("synthesized") is not None or evidence.get("stored_pcm") is not None:
            raise ValueError("Invalid audio cannot advertise an ending check")
        return
    path = resolved(root, row["path"])
    samples, rate = sf.read(path, dtype="int16")
    if samples.ndim != 1 or rate != 16000 or not len(samples):
        raise ValueError("Stored call ending evidence requires mono PCM16 at 16 kHz")
    measured = _ending_signal(samples, rate, pcm=True)
    synthesized = evidence.get("synthesized")
    if (evidence.get("stored_pcm") != measured
            or not isinstance(synthesized, dict) or type(synthesized.get("frames")) is not int or synthesized["frames"] < 1
            or type(synthesized.get("sample_rate")) is not int or synthesized["sample_rate"] < 1
            or type(synthesized.get("active_at_end")) is not bool
            or measured["frames"] != (synthesized["frames"] * rate + synthesized["sample_rate"] - 1) // synthesized["sample_rate"]):
        raise ValueError("Stored call ending evidence differs from its waveform")
    if row["qc"]["passed"] and _waveform_cuts_ending(evidence):
        raise ValueError("A cut-off ending cannot pass technical checks")


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generation_journal(output, contract, filename, changed_message):
    """Load an existing frozen journal without replacing or inventing its contract.

    The caller holds the output writer lock. A journal without its original
    contract cannot prove which settings produced the saved waveforms.
    """
    contract_path = output / "contract.json"
    existing_contract = contract_path.exists()
    if existing_contract and read_json(contract_path) != contract:
        raise ValueError(changed_message)
    rows = read_rows(output / filename)
    if not existing_contract:
        if rows:
            raise ValueError("Generation journal exists without its frozen contract")
        write_json(contract_path, contract)
    return rows


def _reference_instruction(family):
    return (family["description"]
            + f" {family.get('style', 'neutral')} delivery, {family.get('speed', 'natural')} rate,"
            + f" {family.get('intonation', 'natural')} intonation, {family.get('vocal_effort', 'normal')} vocal effort."
            + " Projected means vocal style, not room echo or microphone distance. Say only the provided Korean text once. No extra speech, laughter or echo.")


def _reference_generation(config, family, text, attempt=None):
    tts = config["tts"]
    family_id = family["family_id"]
    parts = [tts["seed"], family_id]
    if attempt:
        parts += [_REFERENCE_ATTEMPT_POLICY, attempt]
    generation = {"kind": "synthetic_reference", "model": tts["design_model"],
                  "seed": int(identity(parts)[:8], 16), "text": text,
                  "instruction": _reference_instruction(family), "family_id": family_id,
                  "tts_settings_id": identity(tts)}
    if attempt is not None:
        generation.update(attempt=attempt, attempt_policy=_REFERENCE_ATTEMPT_POLICY)
    return generation


def _reference_generation_journal(root, output, contract):
    """Keep exact old reference rows/contracts; bind only newly appended rows."""
    path, journal = output / "contract.json", output / "references.jsonl"
    if not path.exists() or read_json(path) == contract:
        return _generation_journal(output, contract, "references.jsonl",
            "Reference generation settings changed; version the bank"), None
    previous = read_json(path)
    core = {key: contract[key] for key in ("tts", "families", "text")}
    if (set(previous) != {*core, "implementation_sha256"}
            or previous.get("implementation_sha256") not in _REFERENCE_PREPARATION_PREVIOUS_IMPLEMENTATIONS
            or any(previous.get(key) != value for key, value in core.items())):
        raise ValueError("Reference generation settings changed; version the bank")
    rows = read_rows(journal)
    bridge_path = output / "reference_generation_update.json"
    source = {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
    if bridge_path.exists():
        bridge = read_json(bridge_path)
        count = bridge.get("preserved_rows")
        if (type(count) is not int or not 0 <= count <= len(rows)
                or bridge.get("kind") != "generated_reference_preparation_update"
                or bridge.get("source_contract") != source or bridge.get("target_contract") != contract
                or bridge.get("preserved_prefix_sha256") != _journal_prefix(journal, count)):
            raise ValueError("Reference generation compatibility evidence changed")
    else:
        if any("reference_preparation" in row for row in rows):
            raise ValueError("New reference rows exist without their preparation compatibility evidence")
        bridge = {"kind": "generated_reference_preparation_update", "source_contract": source,
                  "target_contract": contract, "preserved_rows": len(rows),
                  "preserved_prefix_sha256": _journal_prefix(journal, len(rows))}
        write_json(bridge_path, bridge)
    binding = {"path": bridge_path.relative_to(root).as_posix(), "sha256": digest(bridge_path)}
    if any(row.get("reference_generation_update") != binding for row in rows[bridge["preserved_rows"]:]):
        raise ValueError("New reference rows do not match their preparation compatibility evidence")
    return rows, binding


@project_storage
def generate_references(project, split="train", limit=None, model=None):
    root, config = load_project(project)
    reference_text = config["tts"].get("reference_text", REFERENCE_TEXT)
    if split not in ("train", "validation", "final") or limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Use a valid split and a positive reference limit")
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final reference families stay reserved until the model/runtime is frozen")
    if split == "final":
        require_final_lock(project)
    families = [f for f in read_json(root / "voices.json")["families"] if f["split"] == split]
    selected = families[:limit] if limit is not None else families
    output = root / "data" / "references" / split

    def resident_model():
        nonlocal model
        if model is None:
            report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
            model = _load_model(config["tts"]["design_model"], config["tts"], root)
        return model

    with writer_lock(output):
        reference_contract = {"tts": config["tts"], "families": families, "text": reference_text, "implementation_sha256": _generation_evidence_sha256(),
            "reference_preparation": {"policy": clone_reference.GENERATED_POLICY, "rules": dict(clone_reference.RULES),
                                      "implementation_sha256": _clone_reference_evidence_sha256()}}
        rows, generation_update = _reference_generation_journal(root, output, reference_contract)
        completed = {r["voice_family_id"]: r for r in rows}
        family_by_id = {family["family_id"]: family for family in families}
        for row in completed.values():
            if digest(resolved(root, row["path"])) != row["sha256"]:
                raise ValueError("Synthetic reference changed")
            if "reference_preparation" in row:
                prepared = clone_reference.verify_generated_reference(root, row["reference_preparation"])
                family = family_by_id.get(row["voice_family_id"])
                attempt = row["reference_preparation"]["generation"].get("attempt")
                if (family is None or attempt is not None and
                        (type(attempt) is not int or not 0 <= attempt < _REFERENCE_ATTEMPTS)):
                    raise ValueError("Synthetic reference attempt or family changed")
                expected_generation = _reference_generation(config, family, reference_text, attempt)
                if (not prepared.samefile(resolved(root, row["path"]))
                        or row["reference_preparation"]["prepared"]["sha256"] != row["sha256"]
                        or row["reference_preparation"]["generation"] != expected_generation):
                    raise ValueError("Synthetic reference preparation changed")
                if (attempt is not None or "reference_timing_check" in row) and (
                        row.get("seed") != expected_generation["seed"]
                        or row.get("reference_timing_check") != check_reference_timing(prepared, reference_text)):
                    raise ValueError("Synthetic reference timing evidence changed")
        for family in selected:
            family_id = family["family_id"]
            if family_id in completed:
                continue
            legacy_raw = reference_raw_path(output / "raw" / f"{family_id}.wav")
            for attempt in range(_REFERENCE_ATTEMPTS):
                attempt_name = f"{family_id}_attempt{attempt}.wav"
                reuse_legacy = attempt == 0 and raw_receipt_path(legacy_raw).is_file()
                raw = legacy_raw if reuse_legacy else output / "raw" / attempt_name
                generation = _reference_generation(config, family, reference_text, None if reuse_legacy else attempt)
                try:
                    checked, timing_check, timing = prepare_reference_attempt(root, raw,
                        output / "prepared" / attempt_name, generation, config["tts"]["generation"], resident_model)
                except clone_reference.GeneratedReferenceQualityError as error:
                    if attempt == _REFERENCE_ATTEMPTS - 1:
                        raise ValueError(f"기준 목소리 {family_id} 생성 실패 ({_REFERENCE_ATTEMPTS}회): {error}") from error
                    continue
                # The attempted evidence has its own path, so a failed or
                # interrupted waveform never has to be replaced to continue.
                path = output / (f"{family_id}.wav" if attempt == 0 else attempt_name)
                preparation = publish_reference_attempt(root, path, checked)
                break
            row = {"sample_id": family["family_id"], "source_id": "qwen:" + family["family_id"],
                "voice_family_id": family["family_id"], "split": split, "path": str(path.relative_to(root)),
                "sha256": digest(path), "text": reference_text, "provenance": "qwen_synthetic_reference",
                "human_reference_used": False, "status": "pending_review", "model": config["tts"]["design_model"],
                "seed": generation["seed"], "reference_preparation": preparation, "reference_timing_check": timing_check,
                **({"reference_generation_update": generation_update} if generation_update else {}),
                "description": family["description"], "style": family.get("style", "neutral"), "speed": family.get("speed", "natural"),
                "intonation": family.get("intonation", "natural"), "vocal_effort": family.get("vocal_effort", "normal"), "call_length": "reference_style", "created_at": now(), "axes_scope": "requested reference delivery; confirm actual acoustics in the pilot"}
            append_row(output / "references.jsonl", row)
            if timing is not None:
                timing.committed(True)
            completed[family["family_id"]] = row
        return {"generated": len(completed), "review_queue": str(output / "references.jsonl"),
                "purpose": "synthetic reference voices only; not positive training utterances"}


def call_generation_location(root, split, staged_scope):
    scope_id = identity(staged_scope)[:16]
    return root / "data/call_pronunciation" / scope_id / split, scope_id


def call_generation_contract(config, split, references, pilot, method, staged_scope=None):
    contract = {"word": config["wakeword"], "split": split, "tts": config["tts"], "method": method,
                "references": references, "pilot": pilot, "implementation_sha256": _generation_evidence_sha256()}
    if staged_scope is not None:
        contract["staged_scope"] = staged_scope
    if method == "clone":
        contract["clone_input_policy"] = {"policy": CALL_RECOVERY_POLICY,
            "reference_policy": clone_reference.POLICY,
            "reference_rules": dict(clone_reference.RULES),
            "reference_implementation_sha256": _clone_reference_evidence_sha256(),
            "original_attempts": ORIGINAL_ATTEMPTS, "maximum_attempts": CLONE_ATTEMPTS}
    return contract


def _attempt_seed(config, split, slot, attempt, scope_id):
    parts = [config["tts"]["seed"], split, slot, attempt]
    if scope_id:
        parts.append(scope_id)
    return int(identity(parts)[:8], 16)


def _call_batch_policy(scope):
    if not scope:
        return None
    policy = scope.get("batch_policy")
    same_voice = scope.get("schedule") == VOICE_FILL_SCHEDULE
    if policy == PREFLIGHT_BATCH_POLICY and scope.get("kind") == "preflight" and not same_voice:
        return policy
    if same_voice and policy in (None, EAGER_VOICE_FILL_POLICY):
        return policy or VOICE_FILL_SCHEDULE
    return None


def _batch_identity(config, split, slots, scope_id, contract_id, policy=VOICE_FILL_SCHEDULE):
    seeds = [_attempt_seed(config, split, slot, 0, scope_id) for slot in slots]
    return {"policy": policy, "slots": slots,
            "seed": int(identity([contract_id, seeds, slots])[:8], 16)}


def _validate_batch_entry(config, split, row, reference_count, scope, scope_id, contract_id):
    batch = row["cuda_batch"]
    slots = batch.get("slots") if isinstance(batch, dict) else None
    policy = _call_batch_policy(scope)
    if (not policy or row.get("attempt") != 0
            or not isinstance(slots, list) or len(slots) != CALL_BATCH_SIZE
            or any(type(slot) is not int or slot < 0 for slot in slots) or row.get("slot") not in slots):
        raise ValueError("CUDA clone batch identity changed")
    if policy == PREFLIGHT_BATCH_POLICY:
        aligned = (slots[0] % CALL_BATCH_SIZE == 0 and slots[-1] < reference_count
                   and slots == list(range(slots[0], slots[0] + CALL_BATCH_SIZE)))
    else:
        aligned = (slots[0] // reference_count % CALL_BATCH_SIZE == 0
                   and slots == [slots[0] + index * reference_count for index in range(CALL_BATCH_SIZE)])
    if not aligned or batch != _batch_identity(config, split, slots, scope_id, contract_id, policy):
        raise ValueError("CUDA clone batch identity changed")
    ending = row.get("ending_boundary_qc", {})
    limit, count, stop = ending.get("token_limit"), ending.get("generated_tokens"), ending.get("termination")
    if (row.get("short_clone_policy") != SHORT_CLONE_POLICY or ending.get("policy") != CALL_BOUNDARY_POLICY
            or type(limit) is not int or limit != row.get("generation_token_limit")
            or type(count) is not int or not 1 <= count <= limit or stop not in ("eos", "token_limit", "without_eos")
            or (stop == "token_limit") != (count == limit and stop != "eos")
            or stop != "eos" and row.get("qc", {}).get("passed") is not False):
        raise ValueError("CUDA clone batch ending evidence changed")


def _finish_call_batch(output, intent):
    """Publish journal bytes once, including after an interrupted partial append."""
    journal = output / "candidates.jsonl"
    offset, rows = intent["offset"], intent["rows"]
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + os.linesep for row in rows).encode("utf-8")
    with journal.open("r+b" if journal.exists() else "w+b") as stream:
        size = stream.seek(0, os.SEEK_END)
        if type(offset) is not int or not 0 <= offset <= size or size - offset > len(payload):
            raise ValueError("CUDA clone batch journal boundary changed")
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != b"\n":
                raise ValueError("CUDA clone batch begins inside another journal row")
        stream.seek(offset)
        tail = stream.read()
        if not payload.startswith(tail):
            raise ValueError("CUDA clone batch journal does not match its pending rows")
        stream.write(payload[len(tail):])
        stream.flush()
        os.fsync(stream.fileno())
    (output / "pending_cuda_batch.json").unlink()


def _recover_call_batch(root, output, config, split, references, scope, scope_id, settings):
    pending = output / "pending_cuda_batch.json"
    if not pending.exists():
        return
    intent = read_json(pending)
    contract_id = identity(settings)
    rows = intent.get("rows") if isinstance(intent, dict) else None
    if (not isinstance(rows, list) or len(rows) != CALL_BATCH_SIZE or intent.get("contract_id") != contract_id
            or read_json(output / "contract.json") != settings or not _call_batch_policy(scope)):
        raise ValueError("Pending CUDA clone batch does not belong to this generation contract")
    for row in rows:
        _validate_batch_entry(config, split, row, len(references), scope, scope_id, contract_id)
        if row.get("path") is not None:
            bound_file(root, row)
    if [row["slot"] for row in rows] != rows[0]["cuda_batch"]["slots"] or any(
            row["cuda_batch"] != rows[0]["cuda_batch"] for row in rows):
        raise ValueError("Pending CUDA clone batch order changed")
    _finish_call_batch(output, intent)


def _journal_prefix(path, count):
    """Hash exact old JSONL bytes; never normalize or replace old history."""
    value = hashlib.sha256()
    if count:
        if not path.exists():
            raise ValueError("Saved generation journal prefix is missing")
        with path.open("rb") as journal:
            for _ in range(count):
                line = journal.readline()
                if not line:
                    raise ValueError("Saved generation journal prefix is missing")
                value.update(line)
    return value.hexdigest()


def call_journal_context(root, output, expected, *, create=False):
    """Validate a current journal or the single exact audited legacy bridge.

    Legacy contract.json and its journal prefix remain byte-for-byte intact.
    Only appended attempts use the new contract and its immutable receipt.
    Reading a complete old bank creates no update receipt or derived audio.
    """
    path, journal = output / "contract.json", output / "candidates.jsonl"
    rows = read_rows(journal)
    if not path.exists():
        if rows:
            raise ValueError("Generation journal exists without its frozen contract")
        if create:
            write_json(path, expected)
        return rows, {"current_id": identity(expected), "legacy_count": 0, "update": None}
    actual = read_json(path)
    from .call_generation_history import preserved_context
    preserved = preserved_context(root, output, expected, actual, create=create)
    if preserved is not None:
        return preserved
    if actual == expected:
        if (output / "generation_update.json").exists():
            raise ValueError("Unexpected generation update attached to a current contract")
        return rows, {"current_id": identity(expected), "legacy_count": 0, "update": None}
    legacy = {key: value for key, value in expected.items() if key != "clone_input_policy"}
    legacy["implementation_sha256"] = LEGACY_CALL_IMPLEMENTATION
    if expected.get("method") != "clone" or actual != legacy:
        raise ValueError("TTS model/reference/settings changed; use an audited compatible update or versioned output")
    old_id, new_id = identity(actual), identity(expected)
    update_path = output / "generation_update.json"
    if update_path.exists():
        update = read_json(update_path)
        count = update.get("legacy_row_count")
        if type(count) is not int or count < 0 or count > len(rows):
            raise ValueError("Invalid generation update journal boundary")
        expected_update = {"kind": CALL_RECOVERY_POLICY, "source_contract": {
            "path": str(path.relative_to(root)), "sha256": digest(path)},
            "source_contract_id": old_id, "target_contract": expected,
            "target_contract_id": new_id, "legacy_row_count": count,
            "legacy_journal_prefix_sha256": _journal_prefix(journal, count)}
        if update != expected_update:
            raise ValueError("Generation recovery receipt or its original journal prefix changed")
        binding = {"path": str(update_path.relative_to(root)), "sha256": digest(update_path)}
    else:
        count, binding = len(rows), None
        if any(row.get("generation_contract_id") != old_id for row in rows):
            raise ValueError("Legacy generation rows have no matching update receipt")
        if create:
            update = {"kind": CALL_RECOVERY_POLICY, "source_contract": {
                "path": str(path.relative_to(root)), "sha256": digest(path)},
                "source_contract_id": old_id, "target_contract": expected,
                "target_contract_id": new_id, "legacy_row_count": count,
                "legacy_journal_prefix_sha256": _journal_prefix(journal, count)}
            write_json(update_path, update)
            binding = {"path": str(update_path.relative_to(root)), "sha256": digest(update_path)}
    return rows, {"current_id": new_id, "legacy_id": old_id, "legacy_count": count, "update": binding}


def validate_call_rows(root, config, split, references, rows, context, *, scope=None, method="clone"):
    """Check both historical and newly prepared attempts without inventing QC."""
    tts_id, samples, passing, counts, verified = identity(config["tts"]), set(), {}, Counter(), set()
    scope_id = identity(scope)[:16] if scope is not None else ""
    implementation_sha = _generation_evidence_sha256() if method == "clone" else None
    production = scope is not None and scope.get("kind") == "production"
    old_prompt_attempts = Counter()
    batches = {}
    for index, row in enumerate(rows):
        slot = row.get("slot")
        if type(slot) is not int or slot < 0 or not references:
            raise ValueError("Invalid generated call slot")
        reference = references[slot % len(references)]
        family = reference.get("voice_family_id", reference.get("family_id"))
        historical = index < context["legacy_count"]
        expected_id = context.get("legacy_id") if historical else context["current_id"]
        expected_implementation, expected_update = implementation_sha, context["update"]
        for epoch in context.get("epochs", []):
            if index < epoch["stop"]:
                historical, expected_id = epoch["historical"], epoch["current_id"]
                expected_implementation, expected_update = epoch["implementation_sha256"], epoch["update"]
                break
        if (row.get("sample_id") in samples or row.get("voice_family_id") != family
                or row.get("split") != split or row.get("text") != config["wakeword"]
                or row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False
                or row.get("generation_method") != method or row.get("generation_contract_id") != expected_id
                or row.get("tts_settings_id") != tts_id
                or (scope is not None and row.get("voice_reference_sha256") != reference["sha256"])):
            raise ValueError("Generated call identity, reference assignment or generation provenance changed")
        if "candidate_pcm_qc" in row and (row["candidate_pcm_qc"] != CANDIDATE_PCM_QC or not row.get("path")):
            raise ValueError("Stored PCM quality evidence changed")
        attempt = counts[slot]
        # Legacy journals have deterministic attempt IDs but no explicit seed.
        sample_id = f"{split}_{slot:06d}_attempt{attempt}"
        if scope_id:
            sample_id = scope_id + "_" + sample_id
        if row["sample_id"] != sample_id or slot in passing:
            raise ValueError("Generated call attempt sequence changed")
        if historical and attempt >= ORIGINAL_ATTEMPTS:
            raise ValueError("Legacy generation retry budget changed")
        if method == "clone" and not historical:
            preparation = row.get("clone_reference_preparation")
            policy = preparation.get("policy") if isinstance(preparation, dict) else None
            compacted = policy == clone_reference.INTERNAL_SILENCE_POLICY
            extended = attempt >= CLONE_ATTEMPTS
            if (compacted and not production or extended and (not compacted or attempt >= 2 * CLONE_ATTEMPTS
                    or old_prompt_attempts[slot] != CLONE_ATTEMPTS)
                    or row.get("attempt") != attempt
                    or row.get("seed") != _attempt_seed(config, split, slot, attempt, scope_id)
                    or row.get("clone_input_policy") != CALL_RECOVERY_POLICY
                    or row.get("retry_phase") != ("conditioned_recovery" if extended else
                                                  "recovery" if attempt >= ORIGINAL_ATTEMPTS else "initial")
                    or row.get("individual_human_listening") is not False
                    or row.get("generation_implementation_sha256") != expected_implementation
                    or row.get("generation_update") != expected_update):
                raise ValueError("Prepared generation attempt policy or version changed")
            if policy == clone_reference.POLICY and not extended:
                old_prompt_attempts[slot] += 1
            key = identity([reference, preparation])
            if key not in verified:
                clone_reference.verify_clone_reference(root, reference, preparation)
                verified.add(key)
            if row.get("short_clone_policy") is not None:
                limit = row.get("generation_token_limit")
                if (row["short_clone_policy"] != SHORT_CLONE_POLICY or type(limit) is not int or limit < 3
                        or ("generation_length_limit" in row["qc"]["reasons"] and row["qc"]["passed"] is not False)):
                    raise ValueError("Short-clone generation evidence changed")
        if "cuda_batch" in row:
            _validate_batch_entry(config, split, row, len(references), scope, scope_id, expected_id)
            ordered_slots = tuple(row["cuda_batch"]["slots"])
            batches.setdefault(ordered_slots, []).append(slot)
        if row.get("generation_error"):
            if (historical or method != "clone" or row["generation_error"] != {"kind": "invalid_waveform", "audio_saved": False}
                    or row.get("path") is not None or row.get("sha256") is not None
                    or row.get("qc") != {"passed": False, "reasons": ["empty_nonfinite_or_nonmono"]}
                    or row.get("call_start_s") is not None or row.get("call_end_s") is not None):
                raise ValueError("Invalid waveform failure record was changed or admitted")
        else:
            if not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
                raise ValueError("Generated call waveform binding is missing")
            bound_file(root, row)
        if requires_ending_check(scope, family):
            if "ending_boundary_qc" in row:
                raise ValueError("Targeted ending checks have their own evidence")
            _verify_ending_check(root, row, config["wakeword"])
        elif "ending_check" in row or "ending_check_required" in row:
            raise ValueError("Unexpected ending check outside the agreed repair scope")
        elif "ending_boundary_qc" in row:
            evidence = row["ending_boundary_qc"]
            if (method != "clone" or row.get("short_clone_policy") != SHORT_CLONE_POLICY
                    or not isinstance(evidence, dict) or evidence.get("policy") != CALL_BOUNDARY_POLICY):
                raise ValueError("Unexpected call waveform boundary evidence")
            _verify_ending_signal(root, row, evidence)
        samples.add(row["sample_id"])
        counts[slot] += 1
        if row.get("qc", {}).get("passed") is True:
            passing[slot] = row
    if any(list(slots) != recorded for slots, recorded in batches.items()):
        raise ValueError("CUDA clone batch journal does not contain all four outputs in order")
    return passing, counts


def _exhausted_slots(rows, references, target, passing, limit, split, *, recovery_slots=()):
    result, by_slot = [], {}
    for row in rows:
        by_slot.setdefault(row["slot"], []).append(row)
    for slot in range(target):
        attempts = by_slot.get(slot, [])
        slot_limit = 2 * limit if slot in recovery_slots else limit
        if slot in passing or len(attempts) < slot_limit:
            continue
        reference = references[slot % len(references)]
        group = "학습용" if split == "train" else "공통 평가용" if split == "validation" else "최종 평가용"
        gender = {"man": "남성", "woman": "여성"}.get(reference.get("gender"), "")
        age = {"child": "어린이", "teen": "청소년", "teenager": "청소년", "young-adult": "청년", "young adult": "청년",
               "young_adult": "청년", "middle-aged": "중년", "older adult": "노년", "elderly": "노년", "senior": "노년"}.get(reference.get("age_group"), "")
        family = reference.get("voice_family_id", reference.get("family_id"))
        peers = [ref for ref in references if ref.get("gender") == reference.get("gender")
                 and ref.get("age_group") == reference.get("age_group")]
        number = next((index for index, ref in enumerate(peers, 1) if ref == reference), slot + 1)
        label = " · ".join(item for item in (group, gender, age, f"{number}번") if item)
        result.append({"split": split, "slot": slot, "label": label,
            "voice_family_id": family,
            "attempts": len(attempts), "maximum_attempts": slot_limit,
            "reference_sha256": reference.get("sha256"), "automatic_retry_available": False,
            "failures": [{"sample_id": row["sample_id"], "qc": row["qc"]} for row in attempts]})
    return result


class CallGenerationPaused(Exception):
    """Stop before starting another attempt; completed audio is journaled."""


@project_storage
def generate(project, split="train", limit=None, pilot=False, method="clone", model=None, *,
             staged_scope=None, attempt_ceiling=None, excluded_sha256=(), pause_requested=None):
    root, config = load_project(project)
    if split not in ("train", "validation", "final") or method not in ("clone", "design") or limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Use a valid split/method and a positive generation limit")
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final generation stays locked until model/runtime freeze")
    if split == "final":
        require_final_lock(project)
    approval = read_json(root / "approval.json")
    tts_settings_id = identity(config["tts"])
    if not pilot and split in ("train", "validation") and approval.get("call_preflight_reviews"):
        raise ValueError("Use the Studio pronunciation workflow for its reviewed Train and Validation calls")
    if not pilot and (approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True or
                      approval.get("wakeword") != config["wakeword"] or approval.get("tts_settings_id") != tts_settings_id or approval.get("generation_method") != method):
        raise ValueError("Approve the new-word standalone pilot before bulk generation")
    admission = snapshot_json(root, approval, "generation_policy") if not pilot else None
    if staged_scope is not None and (not pilot or method != "clone" or split == "final"):
        raise ValueError("Scoped pronunciation generation is only for staged Train or Validation calls")
    scope_kind = staged_scope.get("kind") if isinstance(staged_scope, dict) else None
    production = scope_kind == "production"
    same_voice_fill = staged_scope is not None and staged_scope.get("schedule") == VOICE_FILL_SCHEDULE
    batch_policy = _call_batch_policy(staged_scope)
    mixed_preflight = batch_policy == PREFLIGHT_BATCH_POLICY
    if staged_scope and "batch_policy" in staged_scope and not (
            mixed_preflight or same_voice_fill and batch_policy == EAGER_VOICE_FILL_POLICY):
        raise ValueError("Invalid staged call batch policy")
    if same_voice_fill and (scope_kind not in ("production", "postgeneration_repair") or attempt_ceiling is not None):
        raise ValueError("Voice-fill scheduling requires a complete production or repair call target")
    if attempt_ceiling is not None or excluded_sha256:
        if scope_kind not in ("preflight", "production"):
            raise ValueError("Retry scheduling is only for staged preflight or production calls")
        if attempt_ceiling is not None and (type(attempt_ceiling) is not int or not 1 <= attempt_ceiling <= CLONE_ATTEMPTS):
            raise ValueError("Invalid staged preflight attempt ceiling")
    if not isinstance(excluded_sha256, (tuple, list, set, frozenset)) or any(
            type(sha) is not str or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha)
            for sha in excluded_sha256):
        raise ValueError("Invalid staged preflight audio exclusion")
    references = (staged_scope["references"] if staged_scope is not None
                  else read_rows(root / "data" / "references" / split / "references.jsonl"))
    if staged_scope and ("ending_repair" in staged_scope or "ending_repair_families" in staged_scope):
        selected = staged_scope.get("ending_repair_families")
        if (scope_kind not in ("preflight", "postgeneration_repair")
                or staged_scope.get("ending_repair") != REPAIR_ENDING_POLICY
                or selected is not None and (not isinstance(selected, list) or not selected
                    or any(not isinstance(family, str) for family in selected)
                    or len(set(selected)) != len(selected)
                    or set(selected) - {row["voice_family_id"] for row in references})):
            raise ValueError("Invalid targeted call ending repair scope")
    if method == "clone":
        if staged_scope is None:
            references = [r for r in references if r.get("status") == "accepted"
                          or r["voice_family_id"] in approval.get("accepted_reference_families", [])]
        if not references:
            raise ValueError("No approved synthetic references; run references and review/import their provenance first")
        for row in references:
            if row.get("human_reference_used") is not False or row.get("provenance") != "qwen_synthetic_reference" or row.get("split") != split:
                raise ValueError("Only Qwen-created synthetic references from this split are allowed")
            if digest(resolved(root, row["path"])) != row["sha256"]:
                raise ValueError("Reference audio changed")
    else:
        references = [f for f in read_json(root / "voices.json")["families"] if f["split"] == split]
    target_key = "positive_originals" if split == "train" else "validation_originals" if split == "validation" else "final_originals"
    target = min(config["targets"][target_key], limit) if limit is not None else config["targets"][target_key]
    if pilot and limit is None:
        target = min(12, len(references))
    if mixed_preflight and target > len(references):
        raise ValueError("Call preflight generates at most one sample per reference voice")
    output = root / "data" / ("pilot" if pilot else "generated") / split
    scope_id = ""
    if staged_scope is not None:
        output, scope_id = call_generation_location(root, split, staged_scope)
    settings = call_generation_contract(config, split, references, pilot, method, staged_scope)
    with writer_lock(output):
        if method == "clone":
            if batch_policy:
                _recover_call_batch(root, output, config, split, references, staged_scope, scope_id, settings)
            rows, context = call_journal_context(root, output, settings)
            by_slot, attempts_by_slot = validate_call_rows(root, config, split, references, rows, context, scope=staged_scope)
        else:
            rows = _generation_journal(output, settings, "candidates.jsonl",
                                       "TTS model/reference/settings changed; use a new project or versioned output")
            by_slot = {r["slot"]: r for r in rows if r["qc"]["passed"]}
            attempts_by_slot = Counter(row["slot"] for row in rows)
            for row in rows:
                if digest(resolved(root, row["path"])) != row["sha256"]:
                    raise ValueError("Generated candidate changed")
        initial_row_count = len(rows)
        pending = [i for i in range(target) if i not in by_slot]
        attempt_limit = CLONE_ATTEMPTS if method == "clone" else ORIGINAL_ATTEMPTS
        # An intermediate sweep does not change the frozen per-slot budget.
        ceiling = attempt_limit if attempt_ceiling is None else attempt_ceiling
        runnable = [slot for slot in pending if attempts_by_slot[slot] < ceiling]

        def check_pause():
            if pause_requested is not None and pause_requested():
                raise CallGenerationPaused

        prepared = {}
        recovery_slots = set()
        if production and attempt_ceiling is None and not same_voice_fill:
            # Re-open a spent slot only when all six attempts used the earlier,
            # unconditioned prompt. Persisted rows preserve that boundary.
            old_prompt_attempts = Counter(row["slot"] for row in rows
                if row.get("attempt", CLONE_ATTEMPTS) < CLONE_ATTEMPTS
                and (row.get("clone_reference_preparation") or {}).get("policy") == clone_reference.POLICY)
            recovery_slots = {slot for slot in pending if old_prompt_attempts[slot] == CLONE_ATTEMPTS}
        if method == "clone" and (runnable or recovery_slots):
            check_pause()
            # Prepare/verify before model loading, and only for unfinished slots.
            report_progress("tts_prompt_preparation", "기준 목소리 복제 입력 확인", eta_eligible=False)
            for slot in [*runnable, *sorted(recovery_slots)]:
                reference = references[slot % len(references)]
                key = (reference["voice_family_id"], reference["sha256"])
                if key not in prepared:
                    check_pause()
                    if production:
                        prepared[key] = clone_reference.prepare_clone_reference(root, reference, compact_internal_silence=True)
                    else:
                        prepared[key] = clone_reference.prepare_clone_reference(root, reference)
            recovery_slots = {slot for slot in recovery_slots if prepared[
                (references[slot % len(references)]["voice_family_id"], references[slot % len(references)]["sha256"])
            ][1]["policy"] == clone_reference.INTERNAL_SILENCE_POLICY}
            runnable.extend(slot for slot in sorted(recovery_slots) if attempts_by_slot[slot] < 2 * attempt_limit)
            if runnable:
                rows, context = call_journal_context(root, output, settings, create=True)
        if model is None and runnable:
            check_pause()
            report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
            model = _load_model(config["tts"]["clone_model" if method == "clone" else "design_model"], config["tts"], root)
        prompts = {}
        existing_hashes = {r["sha256"] for r in rows if r.get("sha256")}
        if staged_scope is not None:
            existing_hashes.update(staged_scope.get("excluded_sha256", ()))
        existing_hashes.update(excluded_sha256)
        # These immutable settings and retry counts are independent of output.
        generation_contract_id = context["current_id"] if method == "clone" else identity(settings)
        remaining = (Counter(slot % len(references) for slot in runnable)
                     if method == "clone" else {})
        completed_for_target = sum(0 <= slot < target for slot in by_slot)
        def record_attempt(slot, attempt, waveform, rate, token_limit, termination, axes, started, batch=None):
            reference = references[slot % len(references)]
            family = reference.get("voice_family_id", reference.get("family_id"))
            check_ending = requires_ending_check(staged_scope, family)
            sample_id = f"{split}_{slot:06d}_attempt{attempt}"
            if scope_id:
                sample_id = scope_id + "_" + sample_id
            seed = _attempt_seed(config, split, slot, attempt, scope_id)
            if check_ending:
                # Keep the existing waveform rules for already-frozen plans.
                qc, converted = technical_qc(waveform, rate)
                stop = termination["termination"]
                if converted is not None:
                    if stop == "token_limit" or stop == "unobserved" and short_clone_hit_limit(waveform, rate, token_limit):
                        qc["passed"] = False
                        qc["reasons"].append("generation_length_limit")
                    elif stop == "without_eos":
                        qc["passed"] = False
                        qc["reasons"].append("generation_stopped_without_eos")
            else:
                waveform, qc, converted = check_short_clone_audio(waveform, rate, token_limit, termination)
            if converted is None and method != "clone":
                raise ValueError("Nonfinite/empty TTS output; cannot persist or silently accept")
            path, sha, bounds = None, None, None
            ending = None
            if check_ending:
                ending = {"policy": REPAIR_ENDING_POLICY, "expected_last_syllable": config["wakeword"][-1],
                          **termination, "requires_listening": True, "synthesized": None, "stored_pcm": None}
            elif token_limit is not None:
                ending = {"policy": CALL_BOUNDARY_POLICY, **termination, "synthesized": None, "stored_pcm": None}
            if converted is not None:
                samples, bounds = encode_candidate_pcm16(converted, qc)
                if ending:
                    ending["synthesized"] = _ending_signal(waveform, int(rate))
                    ending["stored_pcm"] = _ending_signal(samples, 16000, pcm=True)
                    if _waveform_cuts_ending(ending) and "waveform_cuts_active_ending" not in qc["reasons"]:
                        qc["passed"] = False
                        qc["reasons"].append("waveform_cuts_active_ending")
                path = output / (sample_id + ".wav")
                if (production or batch_policy) and path.exists():
                    # An interruption can leave a WAV without its journal row.
                    # Keep that uncommitted output before retrying its exact seed.
                    orphan = output / "interrupted_audio" / f"{sample_id}_{digest(path)}.wav"
                    orphan.parent.mkdir(exist_ok=True)
                    if not orphan.exists():
                        path.rename(orphan)
                    elif digest(path) != digest(orphan):
                        raise ValueError("An interrupted generation WAV differs from its archived copy")
                sf.write(path, samples, 16000, subtype="PCM_16")
                sha = digest(path)
                if sha in existing_hashes:
                    qc["passed"] = False; qc["reasons"].append("exact_duplicate")
            row = {"sample_id": sample_id, "slot": slot, "source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                "path": str(path.relative_to(root)) if path is not None else None, "sha256": sha, "split": split,
                "voice_family_id": family, "text": config["wakeword"], "provenance": "qwen_tts",
                "human_reference_used": False, "call_start_s": bounds[0] if bounds else None,
                "call_end_s": bounds[1] if bounds else None, "qc": qc, "axes": {"voice_family": family, **axes, "text_profile": "standalone"},
                "generation_contract_id": generation_contract_id, "generation_seconds": time.perf_counter() - started,
                "generation_method": method, "tts_settings_id": tts_settings_id,
                "individual_human_listening": False, "created_at": now()}
            if converted is not None:
                row["candidate_pcm_qc"] = CANDIDATE_PCM_QC
            if check_ending:
                row.update(ending_check_required=True, ending_check=ending)
            elif ending:
                row["ending_boundary_qc"] = ending
            if method == "clone":
                row.update(attempt=attempt, seed=seed, clone_input_policy=CALL_RECOVERY_POLICY,
                    retry_phase=("conditioned_recovery" if attempt >= CLONE_ATTEMPTS else
                                 "recovery" if attempt >= ORIGINAL_ATTEMPTS else "initial"),
                    generation_implementation_sha256=settings["implementation_sha256"],
                    generation_update=context["update"],
                    clone_reference_preparation=prepared[(family, reference["sha256"])][1])
                if token_limit is not None:
                    row.update(generation_token_limit=token_limit, short_clone_policy=SHORT_CLONE_POLICY)
                if converted is None:
                    row["generation_error"] = {"kind": "invalid_waveform", "audio_saved": False}
            if staged_scope is not None:
                row["voice_reference_sha256"] = reference["sha256"]
            if not pilot and qc["passed"]:
                row["admission"] = admission
            if batch is not None:
                row["cuda_batch"] = batch
            return row

        def track(row):
            nonlocal completed_for_target
            slot, sha = row["slot"], row["sha256"]
            rows.append(row)
            if sha:
                existing_hashes.add(sha)
            attempts_by_slot[slot] += 1
            if row["qc"]["passed"]:
                by_slot[slot] = row
                completed_for_target += 1

        def clone_options(reference):
            prompt_key = (reference["voice_family_id"], reference["sha256"])
            if prompt_key not in prompts:
                report_progress("tts_prompt_preparation", "기준 목소리 복제 정보 준비", eta_eligible=False)
                prompt_path, _ = prepared[prompt_key]
                prompts[prompt_key] = model.create_voice_clone_prompt(ref_audio=str(prompt_path),
                    ref_text=reference["text"], x_vector_only_mode=False)
            return dict(text=config["wakeword"], language="Korean", voice_clone_prompt=prompts[prompt_key],
                        **config["tts"]["generation"])

        def clone_axes(reference):
            return {key: reference.get(key, "reference_style")
                    for key in ("style", "speed", "intonation", "vocal_effort", "call_length")}

        def progress():
            print(f"TTS {split}: accepted technical {completed_for_target}/{target}; attempts {len(rows)}", flush=True)

        def run_single(slot):
            reference_index = slot % len(references)
            reference = references[reference_index]
            family = reference.get("voice_family_id", reference.get("family_id"))
            check_ending = requires_ending_check(staged_scope, family)
            slot_ceiling = 2 * ceiling if slot in recovery_slots else ceiling
            for attempt in range(attempts_by_slot[slot], slot_ceiling):
                check_pause()
                _seed(_attempt_seed(config, split, slot, attempt, scope_id))
                started = time.perf_counter()
                token_limit, termination = None, {}
                if method == "clone":
                    generation = clone_options(reference)
                    axes = clone_axes(reference)
                    with generation_attempt() as timing:
                        if check_ending:
                            wavs, rate, termination = _generate_ending_repair_clone(model, attempt, **generation)
                            token_limit = termination["token_limit"]
                        else:
                            wavs, rate, token_limit = generate_short_clone(model, retry=attempt > 0,
                                expected_reuses=remaining[reference_index], termination=termination, **generation)
                else:
                    style, instruction = STYLE_OPTIONS[slot % len(STYLE_OPTIONS)]
                    speed = ("slow", "natural", "quick")[(slot // len(references)) % 3]
                    axes = {"style": style, "speed": speed, "intonation": "natural variation",
                            "vocal_effort": style, "call_length": speed}
                    with generation_attempt() as timing:
                        wavs, rate = model.generate_voice_design(text=config["wakeword"], language="Korean",
                            instruct=reference["description"] + f" {instruction}. {speed} speaking rate. Say only the provided word once. No other words, breaths, laughs or background.",
                            **config["tts"]["generation"])
                waveform = wavs[0] if wavs is not None and len(wavs) else []
                row = record_attempt(slot, attempt, waveform, rate, token_limit, termination, axes, started)
                append_row(output / "candidates.jsonl", row)
                if timing is not None:
                    timing.committed(row["qc"]["passed"])
                track(row)
                if row["qc"]["passed"]:
                    break
            if remaining:
                remaining[reference_index] -= 1
            progress()

        def run_batch(slots, batches_left, generate_batch):
            check_pause()
            indexes = [slot % len(references) for slot in slots]
            reference = references[indexes[0]]
            batch = _batch_identity(config, split, slots, scope_id, generation_contract_id, batch_policy)
            _seed(_attempt_seed(config, split, slots[0], 0, scope_id))
            generation = clone_options(reference)
            if mixed_preflight:
                generation["voice_clone_prompts"] = [clone_options(references[index])["voice_clone_prompt"]
                                                        for index in indexes]
                del generation["voice_clone_prompt"]
            else:
                generation["expected_reuses"] = batches_left
                if batch_policy == EAGER_VOICE_FILL_POLICY:
                    generation["allow_eager"] = True
            started = time.perf_counter()
            with generation_attempt() as timing:
                result = generate_batch(batch_seed=batch["seed"], **generation)
            if result is None:
                return
            wavs, rate, token_limit, endings = result
            if (len(wavs) != CALL_BATCH_SIZE or len(endings) != CALL_BATCH_SIZE
                    or type(token_limit) is not int
                    or any(ending.get("token_limit") != token_limit for ending in endings)):
                raise ValueError("CUDA clone batch did not return one bounded result per slot")
            synthesis_share = (time.perf_counter() - started) / CALL_BATCH_SIZE
            published = []
            for slot, index, waveform, termination in zip(slots, indexes, wavs, endings, strict=True):
                axes = clone_axes(references[index])
                row = record_attempt(slot, 0, waveform, rate, token_limit, termination, axes,
                                     time.perf_counter() - synthesis_share, batch)
                _validate_batch_entry(config, split, row, len(references), staged_scope, scope_id, generation_contract_id)
                published.append(row)
                track(row)
            journal = output / "candidates.jsonl"
            intent = {"contract_id": generation_contract_id, "offset": journal.stat().st_size if journal.exists() else 0,
                      "rows": published}
            write_json(output / "pending_cuda_batch.json", intent)
            _finish_call_batch(output, intent)
            for index, row in zip(indexes, published, strict=True):
                remaining[index] -= row["qc"]["passed"]
            if timing is not None:
                timing.committed(sum(row["qc"]["passed"] for row in published))
            progress()

        if same_voice_fill:
            generate_batch = getattr(model, "generate_short_clone_batch", None)
            allowed = set(runnable)
            reference_count = len(references)
            for reference_index, reference in enumerate(references):
                assigned = list(range(reference_index, target, reference_count))
                groups = [assigned[start:start + CALL_BATCH_SIZE] for start in range(0, len(assigned), CALL_BATCH_SIZE)]
                batch_groups = {index for index, slots in enumerate(groups)
                                if generate_batch is not None and not requires_ending_check(staged_scope, reference["voice_family_id"])
                                and len(slots) == CALL_BATCH_SIZE
                                and all(slot in allowed and attempts_by_slot[slot] == 0 for slot in slots)}
                batches_left = len(batch_groups)
                for index, slots in enumerate(groups):
                    if index in batch_groups:
                        run_batch(slots, batches_left, generate_batch)
                        batches_left -= 1
                    for slot in slots:
                        if slot in allowed and slot not in by_slot:
                            run_single(slot)
                    if any(slot not in by_slot for slot in slots):
                        break
                if any(slot not in by_slot for slot in assigned):
                    break
        elif mixed_preflight:
            generate_batch = getattr(model, "generate_short_clone_mixed_voice_batch", None)
            allowed = set(runnable)
            for first in range(0, target, CALL_BATCH_SIZE):
                slots = range(first, min(first + CALL_BATCH_SIZE, target))
                if (generate_batch is not None and len(slots) == CALL_BATCH_SIZE and all(
                        slot in allowed and attempts_by_slot[slot] == 0 and not requires_ending_check(
                            staged_scope, references[slot]["voice_family_id"]) for slot in slots)):
                    run_batch(list(slots), None, generate_batch)
                for slot in slots:
                    if slot in allowed and slot not in by_slot:
                        run_single(slot)
        else:
            for slot in runnable:
                run_single(slot)
        report_progress("tts_finalizing", "생성 음원 목록·검사 결과 저장", eta_eligible=False)
        passed = sorted(by_slot.values(), key=lambda r: r["slot"])
        # Atomically publish the currently accepted candidates, including partial
        # runs. The summary records whether the selected target was reached.
        if not pilot:
            dest = resolved(root, config["manifests"]["positive_" + split])
            temporary = dest.with_suffix(".tmp")
            temporary.write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in passed), encoding="utf-8")
            temporary.replace(dest)
        journal = output / "candidates.jsonl"
        queue = journal
        if any(row.get("generation_error") for row in rows):
            # Failure-only attempts are audit facts, not playable audio. Keep
            # real QC-failed WAVs available to the standalone CLI reviewer.
            reviewable = [row for row in rows if row.get("path") and row.get("sha256")]
            queue = output / ("reviewable_" + identity(reviewable)[:16] + ".jsonl")
            write_immutable_rows(queue, reviewable)

        def summary_for(goal):
            selected_rows = [row for row in passed if 0 <= row["slot"] < goal]
            count = len(selected_rows)
            axes = ("voice_family", "style", "speed", "intonation", "vocal_effort", "call_length", "text_profile")
            result = {"attempts": sum(attempts for slot, attempts in attempts_by_slot.items() if 0 <= slot < goal),
                "technical_pass": count, "target": goal, "pilot": pilot, "human_listened": 0,
                "policy_admitted": 0 if pilot else count, "independent_STT_runs": 0,
                "distribution": {axis: dict(Counter(row["axes"][axis] for row in selected_rows)) for axis in axes},
                "review_queue": str(queue), "full_target_reached": count == goal,
                "exhausted_slots": _exhausted_slots(rows, references, goal, by_slot, attempt_limit, split,
                                                    recovery_slots=recovery_slots)}
            if queue != journal:
                result["attempt_journal"] = str(journal)
            if method == "clone":
                result.update(clone_input_policy=CALL_RECOVERY_POLICY, maximum_attempts_per_slot=attempt_limit)
            return result

        summary = summary_for(target)
        recovery = method == "clone" and ("legacy_id" in context or "epochs" in context)
        summary_path = output / ("summary_recovery_" + context["current_id"][:16] + ".json" if recovery else "summary.json")
        # A narrower rerun reports its own target without replacing a larger
        # run's evidence. If it filled a gap, refresh the larger summary.
        if not recovery or len(rows) > initial_row_count:
            saved_target = read_json(summary_path).get("target") if summary_path.exists() else None
            if type(saved_target) is not int or saved_target <= target:
                write_json(summary_path, summary)
            elif len(rows) > initial_row_count:
                write_json(summary_path, summary_for(saved_target))
        return summary
