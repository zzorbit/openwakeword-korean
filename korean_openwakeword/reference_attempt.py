"""Prepare and check one generated reference attempt without deciding admission."""

from unicodedata import normalize

import numpy as np
import soundfile as sf

from .clone_reference import GeneratedReferenceQualityError, RULES, prepare_generated_reference
from .common import digest, identity, read_json, write_json
from .task_progress import generation_attempt


MAX_INTERNAL_SILENCE_S = 2.0
_TIMING_POLICY = "reference_candidate_timing_v1"
_MIN_SPAN_LIMIT_S = 8.0
_SPAN_HEADROOM_S = 2.0
_SECONDS_PER_TEXT_CHARACTER = 0.5
_MIN_SECONDS_PER_TEXT_CHARACTER = 0.05
_MAX_REFERENCE_GENERATION_S = 30
_STOP_POLICY = "qwen_reference_termination_v1"


def _text_units(text):
    return sum(character.isalnum() for character in normalize("NFC", text))


def _reference_generation_seconds(text):
    # One extra second beyond the audible-span rule leaves room for edge silence.
    return min(_MAX_REFERENCE_GENERATION_S, max(8, 3 + (_text_units(text) + 1) // 2))


def check_reference_timing(path, text):
    """Measure the audible span of the reference sentence after edge cleanup."""
    samples, rate = sf.read(path, dtype="float64")
    frame = max(1, round(rate * RULES["frame_ms"] / 1000))
    starts = np.arange(0, len(samples), frame)
    ends = np.minimum(starts + frame, len(samples))
    powers = np.add.reduceat(np.square(samples), starts) / (ends - starts)
    active = powers >= RULES["activity_rms"] ** 2
    active_starts, active_ends = starts[active], ends[active]
    if not len(active_starts):
        raise GeneratedReferenceQualityError("기준 목소리 후보에서 발화 구간을 찾지 못했습니다.")
    longest_pause = np.max(active_starts[1:] - active_ends[:-1], initial=0)
    if longest_pause >= MAX_INTERNAL_SILENCE_S * rate:
        raise GeneratedReferenceQualityError("기준 목소리 후보의 발화 사이에 긴 무음이 있습니다.")
    # Allow deliberate slow delivery and longer configured sentences.
    text_units = _text_units(text)
    span_limit_s = max(_MIN_SPAN_LIMIT_S, _SPAN_HEADROOM_S + text_units * _SECONDS_PER_TEXT_CHARACTER)
    audible_span_s = float(active_ends[-1] - active_starts[0]) / rate
    if audible_span_s > span_limit_s:
        raise GeneratedReferenceQualityError("기준 목소리 후보의 발화가 예문 길이에 비해 너무 깁니다.")
    return {"policy": _TIMING_POLICY, "audible_span_s": audible_span_s,
            "span_limit_s": span_limit_s, "longest_internal_pause_s": float(longest_pause) / rate}


def raw_receipt_path(path):
    return path.with_name(path.name + ".raw.json")


def _stop_receipt_path(path):
    return path.with_name(path.name + ".stop.json")


def _check_new_reference_stop(raw, generation, source_sha, timing_check):
    path = _stop_receipt_path(raw)
    if not path.exists():
        return  # Legacy raw recordings have no observed stop to revalidate.
    record = read_json(path)
    if (not isinstance(record, dict) or record.get("policy") != _STOP_POLICY
            or record.get("source_sha256") != source_sha or record.get("generation_id") != identity(generation)):
        raise ValueError("기준 목소리 원본의 합성 종료 기록이 다릅니다.")
    termination = record.get("termination")
    token_limit, count = record.get("token_limit"), record.get("generated_tokens")
    observed = termination in ("eos", "token_limit", "without_eos")
    if (not observed and termination not in ("waveform_at_limit", "unobserved")
            or type(token_limit) is not int or token_limit < 3
            or not observed and count is not None
            or observed and (type(count) is not int or not 1 <= count <= token_limit
                or termination == "token_limit" and count != token_limit
                or termination == "without_eos" and count == token_limit)):
        raise ValueError("기준 목소리 원본의 합성 종료 기록이 다릅니다.")
    if termination in ("token_limit", "without_eos", "waveform_at_limit"):
        raise GeneratedReferenceQualityError("기준 목소리 후보가 발화를 정상 종료하기 전에 멈췄습니다.")
    # Reject only implausibly short readings. Timing alone cannot attest a transcript.
    minimum_span = _text_units(generation["text"]) * _MIN_SECONDS_PER_TEXT_CHARACTER
    if timing_check["audible_span_s"] < minimum_span:
        raise GeneratedReferenceQualityError("기준 목소리 후보의 발화가 예문 길이에 비해 너무 짧습니다.")


def reference_raw_path(raw):
    base, recovery = raw, 0
    while raw.exists() and not raw_receipt_path(raw).exists():
        # An interrupted write without a receipt cannot be re-attested.
        recovery += 1
        raw = base.with_name(base.stem + f"_recovery{recovery}.wav")
    return raw


def prepare_reference_attempt(root, raw, prepared, generation, settings, load_model):
    """Retain raw evidence and reject invalid attempts before they are published."""
    from .tts import _capture_qwen_termination, _qwen_audio_token_limit, _seed, short_clone_hit_limit

    raw = reference_raw_path(raw)
    timing = stop = None
    if not raw.exists():
        if raw_receipt_path(raw).exists() or _stop_receipt_path(raw).exists():
            raise ValueError("기준 목소리 원본의 생성 기록은 있지만 음원이 없습니다.")
        model = load_model()
        limit = _qwen_audio_token_limit(model, settings.get("max_new_tokens"),
                                      _reference_generation_seconds(generation["text"]))
        options = dict(settings, max_new_tokens=limit) if limit is not None else settings
        _seed(generation["seed"])
        with generation_attempt() as timing:
            with _capture_qwen_termination(model, limit) as stop:
                waves, rate = model.generate_voice_design(text=generation["text"], language="Korean",
                    instruct=generation["instruction"], **options)
        if len(waves) != 1:
            raise ValueError("목소리 후보 생성 결과의 수량이 다릅니다.")
        raw.parent.mkdir(parents=True, exist_ok=True)
        sf.write(raw, waves[0], rate, subtype="FLOAT")
        if limit is None:
            stop = None
        elif stop["termination"] == "unobserved" and short_clone_hit_limit(waves[0], rate, limit):
            stop["termination"] = "waveform_at_limit"
    source_sha = digest(raw)
    if stop is not None:
        write_json(_stop_receipt_path(raw), {**stop, "policy": _STOP_POLICY, "source_sha256": source_sha,
                                            "generation_id": identity(generation)})
    try:
        preparation = prepare_generated_reference(root,
            {"path": raw.relative_to(root).as_posix(), "sha256": source_sha, "generation": generation}, prepared)
        timing_check = check_reference_timing(prepared, generation["text"])
        _check_new_reference_stop(raw, generation, source_sha, timing_check)
    except GeneratedReferenceQualityError:
        if timing is not None:
            timing.committed(False)
        raise
    return preparation, timing_check, timing


def publish_reference_attempt(root, target, preparation):
    """Publish a checked attempt with a receipt bound to its final review path."""
    return prepare_generated_reference(root, {**preparation["source"], "generation": preparation["generation"]}, target)
