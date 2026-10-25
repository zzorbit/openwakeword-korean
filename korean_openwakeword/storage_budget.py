"""Auditable storage ledger, not a disk-space guarantee or reservation.

The commercial source set and independent Final inputs are selected later.
Keep their unknown sizes unknown; never turn a partial sum into a minimum,
maximum, or a recommendation with an unexplained safety multiplier.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path


def storage_evidence():
    path = Path(__file__).with_name("resources") / "storage_evidence_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def commercial_storage_budget(settings, *, base_archive_bytes):
    """Describe a new automatic_v3 run, including a labelled manual Final case.

    File payload only: allocation units, an existing workspace, another run,
    and unselected provider batches cannot be inferred from this recipe.
    This function performs no download, audio read, approval or Final access.
    """
    evidence = storage_evidence()
    originals = _positive_integer(settings["originals"])
    rounds = _positive_integer(settings["rounds"])
    per_phrase = settings.get("hard_samples_per_phrase")
    if per_phrase is not None:
        per_phrase = _positive_integer(per_phrase)
        phrase_count = settings.get("hard_phrase_count")
        if phrase_count is not None:
            phrase_count = _positive_integer(phrase_count)
        hard = None if phrase_count is None else phrase_count * per_phrase
    else:
        # Completed older recipes keep their original total-count contract.
        hard = _positive_integer(settings["hard_originals"])
        phrase_count = None
    pool_hours = [float(settings[key]) for key in ("generic_hours", "korean_hours")]
    hours = sum(pool_hours)
    if any(not math.isfinite(h) or h < 0 for h in pool_hours) or hours <= 0:
        raise ValueError("Storage ledger requires finite positive Negative hours")
    policy = settings["evaluation_policy"]
    # Counts vary by benchmark version; the per-WAV bounds come from the
    # unchanged call/padding/RIR limits recorded in storage_evidence_v1.
    supported = {"common_development_v4": (5000, 80000, 32000, 112000, 1000),
                 "common_development_v5": (2000, 32000, 16000, 48000, 500)}
    counts = tuple(policy.get(key) for key in ("validation_originals", "dry_positive_sessions",
                   "reverb_positive_sessions", "positive_sessions", "reverb_originals"))
    if (settings.get("material_acquisition_policy") != "automatic_v3"
            or counts != supported.get(policy.get("id"))
            or settings.get("validation_originals") != counts[0]):
        raise ValueError("Storage evidence does not cover this workflow policy")
    final = evidence["independent_final_scenario"]
    pilot_slots = _positive_integer(settings["train_references"]) if per_phrase is not None else 24
    variables = {
        "negative_hours": hours,
        # Source-boundary rounding and whole-recording quota overshoot are not
        # known yet. This is explicitly a nominal payload, not the final shape.
        "negative_feature_rows": math.ceil(hours * 3600 / 2),
        "positive_feature_rows": originals * rounds,
        "tts_slots": originals + settings["validation_originals"] + final["originals"] + (hard or 0) + pilot_slots,
        "pilot_slots": pilot_slots,
        "reference_count": settings["train_references"] + settings["validation_references"] + 10,
        "hard_originals": hard,
    }
    hard_budget = {
        "policy": "per_phrase_v1" if per_phrase is not None else "legacy_total",
        "samples_per_phrase": per_phrase,
        "phrase_count": phrase_count,
        "total_originals": hard,
        "status": "pending_phrase_count" if hard is None else "calculated",
        "included_in_reference_subtotal": hard is not None,
    }
    if per_phrase is not None:
        hard_budget.update({
            "tts_slots_per_phrase": per_phrase,
            "tts_audio_bytes_per_phrase": per_phrase * 1566856,
            "feature_bytes_per_phrase": per_phrase * 6144,
            "incremental_bytes_per_phrase": per_phrase * (1566856 + 6144),
        })
    rows = []

    def row(key, label, classification, detail, **terms):
        amount = math.ceil(sum(value * (1 if term == "constant" else variables[term])
                               for term, value in terms.items()))
        rows.append({"id": key, "label": label, "classification": classification,
                     "bytes": amount, "detail": detail, "terms": terms})

    row("negative_audio", "비호출 학습 음원", "reference",
        "16kHz 모노 PCM16 환산: 시간 × 115,200,000B. 실제 보존본은 FLAC이며 전체 압축 크기는 미측정입니다.",
        negative_hours=115200000)
    row("negative_features", "비호출 특징", "reference",
        "2초마다 16×96 float32, 행당 6,144B의 명목 계산. 원본별 올림·목표 초과분은 별도입니다.",
        negative_feature_rows=6144)
    row("positive_features", "호출어 증강 특징", "calculated",
        "원본 수 × rounds × 6,144B. 세 모델이 같은 특징을 재사용하며 증강 WAV를 따로 저장하지 않습니다.",
        positive_feature_rows=6144)
    if hard is not None:
        row("hard_features", "유사어 특징·NPY 헤더", "calculated",
            "유사어 R1 행당 6,144B + 네 특징 파일의 128B 헤더.",
            hard_originals=6144, constant=512)
    else:
        row("feature_headers", "네 특징 파일의 NPY 헤더", "calculated",
            "네 특징 파일의 128B 헤더. 문구 수가 정해지지 않은 유사어 특징 데이터는 별도 미확정 항목입니다.",
            constant=512)
    row("base_archives", "공통 평가·기본 자료 묶음", "published",
        "현재 고정 catalog의 보존 다운로드 합계. 대규모 학습 shard와 독립 Final 원본 다운로드는 제외합니다.",
        constant=_nonnegative_integer(base_archive_bytes))
    row("fma_catalog", "FMA 원곡 목록", "measured",
        "전체 tracks.csv 한 사본260,414,445B 실측 참조입니다. 메타 ZIP 전체를 받지 않으며 보충 candidate의 추가 사본과 생성 목록은 미확정 항목에 남깁니다.",
        constant=evidence["fma_catalog_reference_bytes"])
    row("common_originals", "공통 평가 원본·해제 중간본(확인분)", "reference",
        "고정 목록의 FSD·Seoul·MUSAN 원본, Seoul 중첩 ZIP·결합 WAV, Zeroth FLAC·TXT와 공통 RIR80개를 합산합니다. 원본 목록·아카이브 member 크기·실측 헤더에 근거합니다.",
        constant=evidence["common_source_known_bytes"])
    row("training_rirs", "학습 MIT 잔향 원본·FLAC", "measured",
        "고정270개 원본8,374,902B + 표준 변환 FLAC1,554,203B 실측입니다. 270개 모두 PCM 일치를 확인했으며 경로별 sidecar 크기는 관리 파일에 별도 남깁니다.",
        constant=evidence["mit_rir_audio_bytes"])
    row("tts_wavs", "TTS 음원·재시도 보존" + ("(유사어 제외)" if hard is None else ""), "calculated",
        "기본 Qwen 256-token 설정으로 목표를 완료한 경우의 음원 상한 계산입니다. 성공 슬롯당 실패 2개와 성공 1개, 합성 참조를 포함합니다. 평균 예상 크기가 아닙니다."
        + (" 유사어는 문구 수가 정해지지 않아 이 금액에서 제외했습니다." if hard is None else ""),
        tts_slots=1566856, reference_count=1966160)
    row("qwen", "Qwen 모델 두 개", "published",
        "고정 VoiceDesign·Base revision 전체 파일 합계. 저장소별 tokenizer 사본도 각각 계산합니다.",
        constant=evidence["qwen_snapshot_bytes"])
    row("environment", "Python 학습 도구 환경", "measured",
        "Windows·Python 3.12·Torch 2.8 cu128 설치 파일 실측 참조입니다. 공유 Python·설치 다운로드 캐시·설치 임시본은 제외하며 환경에 따라 달라집니다.",
        constant=evidence["environment_reference_bytes"])
    evaluation = dict(evidence["evaluation_audio"])
    evaluation["common_positive"] = policy["dry_positive_sessions"] * 167724 + policy["reverb_positive_sessions"] * 199722
    evaluation["site_copies"] = sum(evaluation[key] for key in
        ("common_positive", "final_positive", "common_streams", "final_negative"))
    for key, label, detail in (
        ("common_positive", "공통 평가 호출 WAV", f"{policy['positive_sessions']:,}개: 무잔향 {policy['dry_positive_sessions']:,} + 잔향 {policy['reverb_positive_sessions']:,}. 호출 구간 최대 1.84초와 지정 패딩·잔향 길이로 계산한 음원 상한입니다."),
        ("final_positive", "독립 Final 호출 WAV(가이드 예시)", "자동 실행 기준이 아닌 가이드의 5,000원본 × 76조건 = 380,000개 시나리오입니다. Final 구성 동결 후 다시 계산해야 합니다."),
        ("common_sources", "공통 평가 변환 음원", "30시간 PCM16 명목값. 실제 원본별 길이·목표 초과분은 준비 후 확인합니다."),
        ("common_streams", "공통 평가 연속 재생 WAV", "원본 변환 음원과 별도로 저장하는 30시간 명목 사본. 유효 시간에서 제외되는 warmup·꼬리는 추가됩니다."),
        ("final_negative", "독립 Final 비호출 WAV", "100시간 PCM16 명목값. 원본 다운로드·변환 중간본·유효 시간 보충은 미확정입니다."),
        ("site_copies", "공통·Final 결과 사이트 음원 사본", "현재 사이트 내보내기는 실제 WAV 복사입니다. 위 호출 WAV와 연속 비호출 WAV를 한 번씩 더 보관하는 시나리오입니다."),
    ):
        row(key, label, "reference", detail, constant=evaluation[key])
    row("checkpoints", "학습 체크포인트 9개", "reference",
        "CNN128·AdamW와 명목 특징 행 수로 직렬화한 파일 실측 참조. 실제 행 수·CUDA RNG에 따라 달라지고 저장 중 임시본은 별도입니다.",
        constant=evidence["checkpoint_reference_bytes"])
    row("deployment_files", "ONNX·frontend 사본", "reference",
        "기존 CNN128 ONNX 실측으로 세 학습 출력·배포 묶음·3개 평균 선택본을 계산한 시나리오입니다. 공통 frontend는 프로젝트·배포·선택본의 세 사본이며 설치 패키지의 사본과 별개입니다.",
        constant=evidence["deployment_files_reference_bytes"])
    unknowns = deepcopy(evidence["unknowns"])
    if hard is None:
        unknowns.append({
            "id": "hard_features", "label": "유사어 음원·특징(문구 수 미정)",
            "detail": f"문구당 {per_phrase:,}개를 생성합니다. 문구 수가 정해지면 TTS·재시도 보존과 R1 특징을 합쳐 문구당 {hard_budget['incremental_bytes_per_phrase']:,}B씩 참고 합계에 추가합니다. 현재 참고 합계에 이 분량은 포함하지 않았습니다.",
        })
    if policy["id"] == "common_development_v5":
        for item in unknowns:
            if item["id"] == "scores_and_reports":
                item["detail"] = "현재 평가의 실제 점수 프레임 수와 NPZ 압축·JSONL·clips/report·선택 사본 크기는 미확정입니다. 평가 중 frames.npy와 압축 NPZ가 동시에 존재합니다."
    for item in unknowns:
        rows.append({**item, "bytes": None, "classification": "unresolved"})
    return {
        "version": 1, "complete": False, "variables": variables, "rows": rows,
        "hard_generation": hard_budget,
        "tts_slots_scope": "excluding_unselected_hard_phrases" if hard is None else "including_hard_phrases",
        "reference_subtotal_bytes": sum(r["bytes"] for r in rows if r["bytes"] is not None),
        "total_required_bytes": None, "peak_required_bytes": None,
        "scope": "conditional_file_payload_comparison_not_total_or_bound",
        "independent_final_scenario": deepcopy(final),
        "unknowns": unknowns,
        "removed_legacy_allowance_bytes": 33285996544,
        "assumptions": [
            "GB/TB는 10진 바이트, 화면의 GiB/TiB는 2진 바이트입니다.",
            "참고 합계는 PCM 환산·설정 상한·명목값·환경 실측을 섞은 비교값이며 전체 필요 공간의 하한이나 상한이 아닙니다.",
            "독립 Final은 수동 별도 단계입니다. 이 계산은 자료를 열거나 평가 조건을 동결·변경하지 않습니다.",
            "임의의 작업 여유·보충률·안전 배수는 합산하지 않았습니다. 미확정 항목을 0B로 간주하지 않습니다.",
            "동시 최대 사용량은 단계별 보존 파일 + 현재 다운로드·해제·변환·점수 임시본의 최댓값으로, 전체 전송량의 합과 다릅니다.",
        ],
        "evidence_version": evidence["version"],
    }


def _positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("Expected a positive integer storage setting")
    return value


def _nonnegative_integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer byte count")
    return value
