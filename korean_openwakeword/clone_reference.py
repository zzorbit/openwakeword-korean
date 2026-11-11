"""Hash-bound, native-rate edge-silence conditioning for cloned speech.

The selected voice, its original file and human review evidence are immutable.
Only excessive quiet outside the first/last energy-active native frame is
removed from a separate prompt input. This is not semantic speech detection.
"""
from pathlib import Path, PureWindowsPath
import json
import shutil
import threading
import uuid

import numpy as np
import soundfile as sf

from .common import bound_file, digest, identity, read_json, writer_lock, write_json
from .reference_qc import _check_wave


POLICY = "clone_reference_edge_silence_v1"
RULES = {"frame_ms": 20, "context_ms": 200, "minimum_removed_edge_ms": 750,
         "activity_rms": 0.0001, "format": "WAV", "subtype": "FLOAT"}
_LOCK = threading.RLock()
GENERATED_POLICY = "generated_reference_edge_silence_v1"


class GeneratedReferenceQualityError(ValueError):
    """A newly generated raw/prepared voice failed unchanged signal quality."""


def _path_identity(path):
    """Compare resolved Windows names in one namespace without changing I/O.

    During concurrent directory creation, Windows realpath can return either
    F:\\... or \\\\?\\F:\\... for the same physical path. Add the namespace
    prefix for comparison rather than stripping verbatim path semantics.
    """
    if isinstance(path, PureWindowsPath):
        name = str(path)
        if name.startswith(("\\\\?\\", "\\\\.\\")):
            return PureWindowsPath(name)
        if name.startswith("\\\\"):
            return PureWindowsPath("\\\\?\\UNC\\" + name[2:])
        if path.is_absolute():
            return PureWindowsPath("\\\\?\\" + name)
    return path


def _binding(root, path):
    try:
        name = _path_identity(path).relative_to(_path_identity(root)).as_posix()
    except ValueError:
        name = _path_identity(path).as_posix()
    return {"path": name, "sha256": digest(path)}


def _source_plan(root, reference):
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("복제할 기준 목소리의 원본 연결을 확인하세요.")
    try:
        source = bound_file(root, reference)
        reasons, _ = _check_wave(source)
        if reasons:
            raise ValueError("복제할 기준 목소리의 기술 검사 실패: " + ", ".join(reasons))
        samples, rate = sf.read(source, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, KeyError) as error:
        raise ValueError("복제할 기준 목소리를 읽지 못했습니다.") from error
    if digest(source) != reference.get("sha256"):
        raise ValueError("분석하는 동안 기준 목소리 원본이 변경되었습니다.")
    # Keep native samples; no downmix, resampling, gain, compression or filtering.
    if samples.shape[1] != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("복제할 기준 목소리는 유효한 단일 채널 음원이어야 합니다.")
    mono = samples[:, 0]
    frame = max(1, round(rate * RULES["frame_ms"] / 1000))
    active = [start for start in range(0, len(mono), frame)
              if np.sqrt(np.mean(mono[start:start + frame].astype(np.float64) ** 2)) >= RULES["activity_rms"]]
    if not active:
        raise ValueError("기준 목소리에 복제할 수 있는 소리 구간이 없습니다.")
    context = round(rate * RULES["context_ms"] / 1000)
    minimum = round(rate * RULES["minimum_removed_edge_ms"] / 1000)
    start = max(0, active[0] - context)
    end = min(len(mono), active[-1] + frame + context)
    if start < minimum:
        start = 0
    if len(mono) - end < minimum:
        end = len(mono)
    source_binding = _binding(root, source)
    if source_binding["sha256"] != reference["sha256"]:
        raise ValueError("분석하는 동안 기준 목소리 원본이 변경되었습니다.")
    plan = {"kind": "clone_reference_preparation", "policy": POLICY, "rules": dict(RULES),
            "source": source_binding, "sample_rate": rate, "source_frames": len(mono),
            "start_frame": start, "end_frame": end,
            "source_duration_s": len(mono) / rate, "prepared_duration_s": (end - start) / rate,
            "leading_removed_s": start / rate, "trailing_removed_s": (len(mono) - end) / rate,
            "trimmed": start != 0 or end != len(mono),
            "source_scope": "Automatic native-sample edge-silence conditioning; not human review or voice replacement"}
    return source, mono, plan


def _locations(root, plan):
    key = plan["source"]["sha256"][:16] + "_" + identity(plan)[:16]
    cache = (root / "data" / "clone_references").resolve()
    folder = (cache / key).resolve()
    if (not _path_identity(cache).is_relative_to(_path_identity(root))
            or not _path_identity(folder).is_relative_to(_path_identity(cache))):
        raise ValueError("기준 목소리 사본 저장 위치가 프로젝트 밖을 가리킵니다.")
    # Use that same namespace for returned I/O paths, not just comparison and
    # receipts. Concurrent normal/verbatim resolve results must be identical.
    folder = Path(_path_identity(folder))
    return folder, folder / "reference.wav", folder / "derivation.json"


def _check_prepared(root, audio, samples, plan):
    try:
        actual, rate = sf.read(audio, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("정리한 기준 목소리 사본을 읽지 못했습니다.") from error
    expected = samples[plan["start_frame"]:plan["end_frame"]]
    if rate != plan["sample_rate"] or actual.shape != (len(expected), 1) or not np.array_equal(actual[:, 0], expected):
        raise ValueError("정리한 기준 목소리가 원본의 보존 구간과 다릅니다.")
    reasons, _ = _check_wave(audio)
    if reasons:
        raise ValueError("정리한 기준 목소리의 기술 검사 실패: " + ", ".join(reasons))
    return {**plan, "prepared": _binding(root, audio)}


def _verify_existing(root, samples, plan, audio, record):
    body = _check_prepared(root, audio, samples, plan)
    try:
        if read_json(record) != body:
            raise ValueError("기준 목소리 정리 기록의 원본·설정·구간 연결이 다릅니다.")
    except (OSError, KeyError, TypeError) as error:
        raise ValueError("기준 목소리 정리 기록을 읽지 못했습니다.") from error
    return {**body, "derivation_receipt": _binding(root, record)}


def verify_clone_reference(root, reference, metadata):
    """Verify a saved conditioning input without creating or modifying files."""
    root = Path(root).resolve()
    if not isinstance(metadata, dict) or metadata.get("policy") != POLICY:
        raise ValueError("기준 목소리 정리 정책이 다릅니다.")
    source, samples, plan = _source_plan(root, reference)
    if not plan["trimmed"]:
        expected = {**plan, "prepared": plan["source"]}
        if metadata != expected:
            raise ValueError("원본 기준 목소리의 정리 기록이 다릅니다.")
        return Path(_path_identity(source))
    _, audio, record = _locations(root, plan)
    expected = _verify_existing(root, samples, plan, audio, record)
    if metadata != expected:
        raise ValueError("정리한 기준 목소리 사본의 연결 또는 해시가 다릅니다.")
    return audio


def prepare_clone_reference(root, reference):
    """Return (prompt WAV path, immutable preparation metadata).

    A missing receipt after an interrupted atomic WAV publish is recoverable
    only when that WAV is exactly the expected native source slice. Existing
    receipts or WAVs are never silently replaced if their content differs.
    """
    root = Path(root).resolve()
    source, samples, plan = _source_plan(root, reference)
    if not plan["trimmed"]:
        return Path(_path_identity(source)), {**plan, "prepared": plan["source"]}
    folder, audio, record = _locations(root, plan)
    with _LOCK:
        if audio.exists() and record.exists():
            return audio, _verify_existing(root, samples, plan, audio, record)
        if record.exists():
            raise ValueError("기준 목소리 정리 기록은 있지만 사본 음원이 없습니다.")
        with writer_lock(folder):
            if not audio.exists():
                temporary = folder / ("reference-" + uuid.uuid4().hex + ".tmp.wav")
                try:
                    sf.write(temporary, samples[plan["start_frame"]:plan["end_frame"]], plan["sample_rate"],
                             format=RULES["format"], subtype=RULES["subtype"])
                    _check_prepared(root, temporary, samples, plan)
                    temporary.replace(audio)
                finally:
                    temporary.unlink(missing_ok=True)
            body = _check_prepared(root, audio, samples, plan)
            if record.exists():
                if read_json(record) != body:
                    raise ValueError("기준 목소리 정리 기록이 동시에 변경되었습니다.")
            else:
                write_json(record, body)
            return audio, _verify_existing(root, samples, plan, audio, record)


def _generated_paths(root, raw_reference, target):
    source = bound_file(root, raw_reference)
    target = Path(target)
    target = (target if target.is_absolute() else root / target).resolve()
    if (not _path_identity(source).is_relative_to(_path_identity(root))
            or not _path_identity(target).is_relative_to(_path_identity(root))
            or _path_identity(source) == _path_identity(target)):
        raise ValueError("새 기준 목소리 원본과 검수용 사본은 프로젝트 내부의 서로 다른 파일이어야 합니다.")
    return Path(_path_identity(source)), Path(_path_identity(target))


def _generated_raw_record(root, raw_reference, source, *, publish):
    source_binding = _binding(root, source)
    if source_binding["sha256"] != raw_reference["sha256"]:
        raise ValueError("새 기준 목소리 원본이 변경되었습니다.")
    generation = json.loads(json.dumps(raw_reference.get("generation", {}), allow_nan=False))
    if not isinstance(generation, dict):
        raise ValueError("새 기준 목소리의 생성 조건 기록이 다릅니다.")
    reasons, metrics = _check_wave(source)
    body = {"kind": "generated_reference_raw", "policy": GENERATED_POLICY,
            "source": source_binding, "generation": generation,
            "technical_reasons": reasons, "technical_metrics": metrics,
            "source_scope": "Preserved raw synthetic model output before automatic edge-silence cleanup"}
    record = source.with_name(source.name + ".raw.json")
    if record.exists():
        if read_json(record) != body:
            raise ValueError("새 기준 목소리 원본의 생성 기록이 다릅니다.")
    elif publish:
        write_json(record, body)
    else:
        raise ValueError("새 기준 목소리 원본의 생성 기록이 없습니다.")
    return body, _binding(root, record)


def _generated_plan(root, raw_reference, source, *, publish_raw):
    raw, raw_binding = _generated_raw_record(root, raw_reference, source, publish=publish_raw)
    # Excess edge silence can make the raw clip overlong or dilute its mean
    # level. Everything else is a hard failure; the prepared clip still has to
    # pass the original, unchanged complete reference quality check.
    hard_reasons = set(raw["technical_reasons"]) - {"duration_outlier", "near_silence"}
    if hard_reasons:
        raise GeneratedReferenceQualityError("새 기준 목소리 원본의 기술 검사 실패: " + ", ".join(sorted(hard_reasons)))
    try:
        samples, rate = sf.read(source, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise GeneratedReferenceQualityError("새 기준 목소리 원본을 읽지 못했습니다.") from error
    if samples.shape[1] != 1 or not len(samples) or not np.isfinite(samples).all():
        raise GeneratedReferenceQualityError("새 기준 목소리 원본은 유효한 단일 채널 음원이어야 합니다.")
    mono = samples[:, 0]
    frame = max(1, round(rate * RULES["frame_ms"] / 1000))
    active = [start for start in range(0, len(mono), frame)
              if np.sqrt(np.mean(mono[start:start + frame].astype(np.float64) ** 2)) >= RULES["activity_rms"]]
    if not active:
        raise GeneratedReferenceQualityError("새 기준 목소리에 보존할 수 있는 소리 구간이 없습니다.")
    context = round(rate * RULES["context_ms"] / 1000)
    minimum = round(rate * RULES["minimum_removed_edge_ms"] / 1000)
    start, end = max(0, active[0] - context), min(len(mono), active[-1] + frame + context)
    if start < minimum:
        start = 0
    if len(mono) - end < minimum:
        end = len(mono)
    if digest(source) != raw_reference["sha256"]:
        raise ValueError("정리하는 동안 새 기준 목소리 원본이 변경되었습니다.")
    plan = {"kind": "generated_reference_preparation", "policy": GENERATED_POLICY, "rules": dict(RULES),
            "source": raw["source"], "raw_receipt": raw_binding, "generation": raw["generation"],
            "sample_rate": rate, "source_frames": len(mono), "start_frame": start, "end_frame": end,
            "source_duration_s": len(mono) / rate, "prepared_duration_s": (end - start) / rate,
            "leading_removed_s": start / rate, "trailing_removed_s": (len(mono) - end) / rate,
            "trimmed": start != 0 or end != len(mono),
            "source_technical_reasons": raw["technical_reasons"],
            "source_scope": "Automatic edge-silence cleanup of a newly generated synthetic reference; not human review or voice replacement"}
    return mono, plan


def _generated_body(root, target, samples, plan):
    try:
        actual, rate = sf.read(target, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("새 기준 목소리의 검수용 음원을 읽지 못했습니다.") from error
    expected = samples[plan["start_frame"]:plan["end_frame"]]
    if rate != plan["sample_rate"] or actual.shape != (len(expected), 1) or not np.array_equal(actual[:, 0], expected):
        raise ValueError("새 기준 목소리 사본이 원본의 보존 구간과 다릅니다.")
    reasons, metrics = _check_wave(target)
    if reasons:
        raise GeneratedReferenceQualityError("새 기준 목소리 검수용 음원의 기술 검사 실패: " + ", ".join(reasons))
    return {**plan, "prepared": _binding(root, target), "technical_metrics": metrics}


def prepare_generated_reference(root, raw_reference, target):
    """Publish a new review/clone WAV while preserving its raw generation.

    The caller holds its generation writer lock and supplies a distinct target
    for each attempt. Existing target files and receipts are never replaced.
    Even rejected raw model outputs retain a hash-bound raw receipt.
    """
    root = Path(root).resolve()
    source, target = _generated_paths(root, raw_reference, target)
    with _LOCK:
        samples, plan = _generated_plan(root, raw_reference, source, publish_raw=True)
        receipt = target.with_name(target.name + ".preparation.json")
        if receipt.exists() and not target.exists():
            raise ValueError("새 기준 목소리 정리 기록은 있지만 검수용 음원이 없습니다.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp.wav")
            try:
                if plan["trimmed"]:
                    sf.write(temporary, samples[plan["start_frame"]:plan["end_frame"]], plan["sample_rate"],
                             format=RULES["format"], subtype=RULES["subtype"])
                else:
                    shutil.copyfile(source, temporary)
                _generated_body(root, temporary, samples, plan)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        body = _generated_body(root, target, samples, plan)
        if receipt.exists():
            if read_json(receipt) != body:
                raise ValueError("새 기준 목소리의 정리 기록이 다릅니다.")
        else:
            write_json(receipt, body)
        return {**body, "derivation_receipt": _binding(root, receipt)}


def verify_generated_reference(root, metadata):
    """Validate new-reference provenance read-only; do not reinterpret old rows."""
    root = Path(root).resolve()
    if not isinstance(metadata, dict) or metadata.get("policy") != GENERATED_POLICY:
        raise ValueError("새 기준 목소리 정리 정책이 다릅니다.")
    raw_reference = {**metadata["source"], "generation": metadata["generation"]}
    source, target = _generated_paths(root, raw_reference, metadata["prepared"]["path"])
    samples, plan = _generated_plan(root, raw_reference, source, publish_raw=False)
    body = _generated_body(root, target, samples, plan)
    receipt = target.with_name(target.name + ".preparation.json")
    if read_json(bound_file(root, metadata["derivation_receipt"])) != body:
        raise ValueError("새 기준 목소리의 정리 기록이 다릅니다.")
    expected = {**body, "derivation_receipt": _binding(root, receipt)}
    if metadata != expected:
        raise ValueError("새 기준 목소리 원본·사본·기록 연결이 다릅니다.")
    return target
