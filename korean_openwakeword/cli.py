"""koww: stable entry points without historical project paths."""
import argparse
import json
import sys
from pathlib import Path

from .common import resolved, write_json
from .project import create_project, load_project, status, default_profile, workload_summary, save_configuration
from .budgets import DEFAULT_GENERIC_HOURS, DEFAULT_KOREAN_HOURS, DEFAULT_FMA_HOURS


def setup_arguments(a):
    basic = a.add_argument_group("초기 선택: 작업량·자료 규모")
    basic.add_argument("--originals", type=int, help="증강 전 Positive TTS 원본 수; 시작값 100000")
    basic.add_argument("--rounds", type=int, choices=(1, 3, 5), help="원본당 증강 횟수; 시작값 5")
    basic.add_argument("--updates", type=int, help="모델 한 개의 optimizer update 수; 시작값 100000")
    basic.add_argument("--generic-hours", type=float, help=f"일반 비호출 자료 목표 시간; 시작값 {DEFAULT_GENERIC_HOURS}")
    basic.add_argument("--korean-hours", type=float, help=f"한국어 비호출 자료 목표 시간; 시작값 {DEFAULT_KOREAN_HOURS}")
    basic.add_argument("--noise-hours", type=float, help="환경소음 증강 재료 목표 시간; 시작값 30, 다양성도 별도 확인")
    mode = basic.add_mutually_exclusive_group()
    mode.add_argument("--interactive", dest="interactive", action="store_true", help="작업량 선택 화면 열기")
    mode.add_argument("--non-interactive", dest="interactive", action="store_false", help="질문 없이 명시한 값과 기본값 저장")
    a.set_defaults(interactive=None)
    detail = a.add_argument_group("세부 설정: 필요할 때만 변경")
    detail.add_argument("--advanced", action="store_true", help="모델 구조·학습 방법의 세부 선택 화면 열기")
    detail.add_argument("--architecture", choices=("cnn64", "cnn128", "mlp32", "mlp128", "mlp256"))
    detail.add_argument("--optimizer", choices=("Adam", "AdamW"))
    detail.add_argument("--learning-rate", type=float)
    detail.add_argument("--weight-decay", type=float)
    detail.add_argument("--dropout", type=float)
    detail.add_argument("--positive-loss-weight", type=float)
    detail.add_argument("--schedule", choices=("constant", "cosine_with_2percent_warmup"))
    detail.add_argument("--schedule-horizon", type=int, help="통제 비교용 고정 horizon; 기본은 선택 update 수")
    detail.add_argument("--checkpoints", type=int, nargs="+")
    detail.add_argument("--models", type=int, choices=(1, 3), help="독립 seed 모델 수; 기본 3")
    detail.add_argument("--device", help="auto / cpu / cuda[:index]; auto는 CPU/GPU 실측 비교")


def selected_choices(a, config):
    from .setup import setup_choices
    keys = ("architecture", "optimizer", "learning_rate", "weight_decay", "dropout",
            "positive_loss_weight", "schedule", "schedule_horizon", "checkpoints", "device")
    overrides = {key: getattr(a, key) for key in keys if getattr(a, key) is not None}
    if a.models is not None:
        seeds = config["training"]["seeds"][:a.models]
        while len(seeds) < a.models:
            seeds.append(max(seeds) + 1)
        overrides["seeds"] = seeds
    return setup_choices(config, interactive=a.interactive, advanced=a.advanced,
        positive_originals=a.originals, augmentation_rounds=a.rounds,
        updates=a.updates, generic_hours=a.generic_hours, korean_hours=a.korean_hours,
        noise_hours=a.noise_hours, training_overrides=overrides)


def parser():
    p = argparse.ArgumentParser(prog="koww", description="한국어 전용 openWakeWord 학습·평가 라이브러리")
    from . import __version__
    p.add_argument("--version", action="version", version="korean-openwakeword " + __version__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("web", help="하나의 웹 화면에서 자료 준비·검수·학습·시험")
    a.add_argument("--workspace", type=Path, help="프로젝트·공유 자료·캐시 저장 폴더; 기본은 현재 폴더")
    a.add_argument("--port", type=int, default=8793)
    a.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    a = sub.add_parser("init", help="새 한국어 호출어 프로젝트")
    a.add_argument("--wakeword", required=True); a.add_argument("--id", required=True); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--validation-voices", type=int, choices=(10, 30), default=30,
                   help="평가용 기준 목소리 수; 기본 30개, 이전 10개 설정도 지원")
    setup_arguments(a)
    a = sub.add_parser("configure", help="기존 설정을 보존하고 별도 작업량·세부 설정 snapshot 작성")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    setup_arguments(a)
    a = sub.add_parser("settings", help="현재 작업량 또는 세부 학습 설정 확인")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--advanced", action="store_true")
    a = sub.add_parser("check", aliases=["doctor"], help="설치·GPU 상태 확인")
    a.add_argument("--tts", action="store_true", help="Qwen·PyTorch·torchaudio 버전과 실제 import 확인 (생성 없음)")
    a = sub.add_parser("sample", help="춘식아 예시 모델 검증 또는 복사")
    a.add_argument("--output", type=Path)
    a = sub.add_parser("demo", help="동봉 예시 WAV 재채점·결과 사이트 생성")
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("infer", help="실제 WAV의 CPU ONNX 호출 결과")
    a.add_argument("--bundle", type=Path); a.add_argument("--audio", required=True, type=Path); a.add_argument("--output", type=Path)
    a = sub.add_parser("mic", help="브라우저에서 마이크 선택·실시간 호출 시험 (녹음 저장 없음)")
    a.add_argument("--bundle", type=Path, help="학습한 실행 bundle; 생략하면 동봉 춘식아 모델")
    a.add_argument("--port", type=int, default=8792, help="로컬 시험 화면 포트; 기본 8792")
    a = sub.add_parser("benchmark", help="CPU/GPU의 학습 업데이트 비용·수치 차이 측정")
    a.add_argument("--output", type=Path, required=True); a.add_argument("--steps", type=int, default=20)
    a = sub.add_parser("download-generic", help="검증된 공개 Generic feature 다운로드 (~17.28GB)")
    a.add_argument("--output", type=Path, required=True)
    sub.add_parser("datasets", help="춘식아에 사용한 공개 자료의 출처·고정 버전·용량 확인")
    a = sub.add_parser("negative-plan", help="상용 비호출의 종류별 준비 목표와 실제 공개 출처 확인 (다운로드 없음)")
    a.add_argument("--generic-hours", type=float, default=DEFAULT_GENERIC_HOURS)
    a.add_argument("--korean-hours", type=float, default=DEFAULT_KOREAN_HOURS)
    a = sub.add_parser("download-source", help="상용 비호출의 원본 묶음 선택·용량 확인·다운로드 (검수·시간 집계 별도)")
    a.add_argument("--source", choices=("audioset_unbalanced", "mls", "yodas2", "fma_large", "fma_full"), required=True)
    a.add_argument("--parts", nargs="+", help="AudioSet/YODAS 묶음 번호 또는 MLS 화자·책 묶음 ID")
    a.add_argument("--language", help="MLS의 제공자 언어명: german, french, spanish 등")
    a.add_argument("--subset", help="YODAS2의 제공자 분할명: ko000, en000 등")
    a.add_argument("--output", type=Path, help="원본과 완료 기록을 보존할 폴더")
    a.add_argument("--max-download-gib", type=float, help="실제 다운로드에 허용할 최대 GiB; 생략하면 메타데이터만 확인")
    a.add_argument("--metadata-only", action="store_true", help="명시한 용량 안에서 제공자 설명·시간·전사만 받고 원본 오디오는 받지 않음")
    a.add_argument("--whole-archive", action="store_true", help="FMA full 전체 ZIP을 명시적으로 선택; 기본은 필요한 원곡만 다운로드")
    a.add_argument("--hours", type=float, default=DEFAULT_FMA_HOURS, help=f"FMA full 선택 다운로드 목표 시간; 기본 {DEFAULT_FMA_HOURS:g}")
    a.add_argument("--seed", type=int, default=20261021)
    a.add_argument("--license-policy", choices=("allow-derivatives", "commercial-only"), default="allow-derivatives")
    a.add_argument("--exclude", type=Path, help="추가로 제외할 FMA 곡·음악가·원출처 목록 JSON")
    a = sub.add_parser("download-fma", help="FMA full에서 필요한 원곡만 선택·검증·이어받기 (재인코딩 없음)")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--hours", type=float, default=DEFAULT_FMA_HOURS)
    a.add_argument("--seed", type=int, default=20261021)
    a.add_argument("--plan-only", action="store_true", help="목록·메타데이터만 받고 고정 선곡·예상 용량 저장; 원곡 다운로드 없음")
    a.add_argument("--max-download-gib", type=float, help="실제 다운로드의 최대 GiB; 원곡을 받으려면 필수")
    a.add_argument("--license-policy", choices=("allow-derivatives", "commercial-only"), default="allow-derivatives")
    a.add_argument("--exclude", type=Path, help="추가로 제외할 FMA 곡·음악가·원출처 목록 JSON")
    a = sub.add_parser("download-data", help="고정 SHA256 공개 자료 다운로드·중단 재개 (내용 승인 별도)")
    a.add_argument("--dataset", required=True); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("import-generic", help="전체 공개 Generic feature를 외부 경로로 재사용")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--features", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("index-data", help="공급자 메타데이터로 음원별 출처·화자 목록 만들기 (STT 없음)")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--dataset", choices=("zeroth", "manifest", "audioset", "fma", "mit_rir", "mls", "yodas2"), default="manifest")
    a.add_argument("--metadata", type=Path)
    a.add_argument("--ontology", type=Path, help="AudioSet의 제공자 ontology.json")
    a.add_argument("--room-metadata", type=Path, help="제공자가 명시한 RIR 파일별 공간 목록")
    a.add_argument("--dataset-id"); a.add_argument("--source-url"); a.add_argument("--license")
    a = sub.add_parser("prepare-materials", help="공개 음원 변환·자료 분리·대표 검수 큐 준비")
    a.add_argument("--project", type=Path, required=True)
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--source-index", type=Path, help="index-data 결과 또는 원출처 CSV/JSONL")
    a.add_argument("--dataset-id"); a.add_argument("--source-url"); a.add_argument("--license")
    a.add_argument("--category", choices=("environment", "music", "competing_speech", "rir"))
    a.add_argument("--category-basis", help="제공자가 명시한 자료 범주 근거")
    a.add_argument("--language", default="unknown")
    speech = a.add_mutually_exclusive_group()
    speech.add_argument("--speech", dest="speech", action="store_true")
    speech.add_argument("--no-speech", dest="speech", action="store_false")
    a.set_defaults(speech=None)
    a.add_argument("--hours", type=float, help="고유 학습 원본의 선택 분량; 부족하면 결과에 표시")
    a.add_argument("--validation-fraction", type=float, default=.1, help="원본·화자·공간 그룹을 분리할 개발용 예약 비율")
    a.add_argument("--seed", type=int, default=20260928)
    a.add_argument("--representatives", type=int, default=3, help="자료셋·범주당 대표 수")
    a.add_argument("--preview-seconds", type=float, default=10)
    a = sub.add_parser("admit-materials", help="저장된 검수 답변 적용·폐기 제외·학습 자료 연결")
    a.add_argument("--project", type=Path, required=True)
    a.add_argument("--preparation", type=Path, required=True, help="prepare-materials의 preparation_receipt.json")
    a.add_argument("--answers", type=Path, help="생략하면 이 프로젝트의 저장된 pool_results.json")
    a.add_argument("--output", type=Path)
    a.add_argument("--pools", nargs="+", choices=("generic_negative", "korean_negative", "noise", "rir"), default=[])
    a.add_argument("--connect", action="store_true", help="현재 프로젝트의 비어 있는 해당 자료 목록에 연결")
    a = sub.add_parser("connect-materials", help="여러 승인 자료 묶음을 검증·병합해 한 번에 연결")
    a.add_argument("--project", type=Path, required=True)
    a.add_argument("--admissions", type=Path, nargs="+", required=True, help="admit-materials의 admission_receipt.json 목록")
    a.add_argument("--pools", nargs="+", choices=("generic_negative", "korean_negative", "noise", "rir"), required=True)
    a.add_argument("--output", type=Path)
    for name in ("status", "audit", "prepare-frontend", "storage"):
        a = sub.add_parser(name); a.add_argument("--project", type=Path, required=True)
    a = sub.add_parser("audit-final", help="동결 후 독립 최종 자료의 학습·공통 평가 중복 검사 (채점 없음)")
    a.add_argument("--project", type=Path, required=True)
    a.add_argument("--manifest", type=Path, help="추가 최종 평가 manifest; 프로젝트 예약 자료와 함께 검사")
    a.add_argument("--output", type=Path, required=True, help="새 검사 보고서 JSON; 기존 파일 덮어쓰기 금지")
    a = sub.add_parser("windows", help="공개 Negative 원본의 고정 2초 구간 manifest")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--pool", choices=("generic_negative", "korean_negative", "hard_negative"), required=True)
    a.add_argument("--output", type=Path, required=True)
    for name in ("references", "generate"):
        a = sub.add_parser(name); a.add_argument("--project", type=Path, required=True)
        a.add_argument("--split", choices=("train", "validation", "final"), default="train"); a.add_argument("--limit", type=int)
        if name == "generate":
            a.add_argument("--pilot", action="store_true"); a.add_argument("--method", choices=("clone", "design"), default="clone")
            a.add_argument("--tts-backend", choices=("auto", "eager"), default="auto", help="GPU 가속은 원본과 PCM/QC/RNG가 같은지 참조별 확인 후 사용")
            a.add_argument("--tts-cpu-threads", type=int, default=4)
    a = sub.add_parser("check-references", help="기준 목소리 자동 품질 검사·연결; 호출어 발음 검수는 별도")
    a.add_argument("--project", type=Path, required=True)
    a.add_argument("--train-count", type=int, help="학습용 기준 목소리 수; 생략하면 생성된 수량")
    a.add_argument("--validation-count", type=int, help="공통 평가용 기준 목소리 수; 생략하면 생성된 수량")
    a = sub.add_parser("generate-hard", help="Qwen 유사어 후보 생성; 별도 내용 승인 전에는 학습하지 않음")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--phrases", nargs="+", required=True)
    hard_budget = a.add_mutually_exclusive_group()
    hard_budget.add_argument("--samples-per-phrase", type=int, help="유사어 한 문구당 생성할 음원 수")
    hard_budget.add_argument("--count", type=int, help="기존 방식의 전체 유사어 원본 수; 생략 시 문구당 학습 목소리 수")
    a.add_argument("--tts-backend", choices=("auto", "eager"), default="auto")
    a.add_argument("--tts-cpu-threads", type=int, default=4)
    a = sub.add_parser("review", help="검수 큐 전체 연속 청취·직접 저장")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--queue", type=Path, required=True)
    a.add_argument("--pool-manifest", type=Path, help="대표 청취와 별도 동의로 승인할 고정 자료 묶음; content 전용")
    a.add_argument("--kind", choices=("references", "pilot", "content"), default="pilot"); a.add_argument("--port", type=int, default=8790)
    a = sub.add_parser("features", help="실제 N×rounds 증강 feature")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--role", choices=("positive", "generic_negative", "korean_negative", "hard_negative"), required=True)
    a.add_argument("--output", type=Path); a.add_argument("--rounds", type=int, choices=(1, 3, 5)); a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--smoke", action="store_true"); a.add_argument("--resume", action="store_true")
    a = sub.add_parser("verify-cache"); a.add_argument("directory", type=Path)
    a = sub.add_parser("compare-rounds"); a.add_argument("first", type=Path); a.add_argument("second", type=Path)
    a = sub.add_parser("train", help="고정 batch·독립 seed·GPU/CPU 학습 및 ONNX export")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--output", type=Path)
    a.add_argument("--smoke", action="store_true"); a.add_argument("--resume", action="store_true")
    a = sub.add_parser("package", help="새 모델·공유 frontend·설정 묶음")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--run", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("prepare-eval", help="고정 소음 구간/SNR의 실제 평가 WAV 생성")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--split", choices=("validation", "final"), default="validation"); a.add_argument("--originals", type=int, default=100)
    a.add_argument("--noise-sections", type=int, default=5)
    a.add_argument("--reverb-originals", type=int, default=0, help="두 잔향 강도에 공통으로 사용할 원본 수; 원본 2000개·소음 구간 1개·잔향 500개이면 총 48,000회")
    a.add_argument("--reverb-selection", choices=["legacy", "metadata_axes"], default="legacy", help="잔향 원본 선별: 새 공통 평가에는 metadata_axes 사용; 기존 recipe는 legacy 유지")
    a.add_argument("--noise-selection", choices=("legacy", "balanced_sources"), default="legacy",
                   help="legacy: 기존 원본 선택 / balanced_sources: 소음 세부 종류와 원본 균형 선택")
    a = sub.add_parser("setup-gpu", help="평가용 GPU 도구를 프로젝트 캐시에 준비")
    a.add_argument("--project", type=Path, required=True)
    a = sub.add_parser("evaluate", help="동일 WAV의 단일·평균 모델 평가")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--manifest", type=Path); a.add_argument("--stage", choices=("development", "final"), default="development")
    a.add_argument("--output", type=Path, required=True); a.add_argument("--smoke", action="store_true")
    a.add_argument("--candidate", help="채택 판정을 확인할 모델 ID; 생략하면 현재 bundle의 실제 출력")
    a.add_argument("--workers", choices=("auto", "1", "2", "4"), default="auto", help="독립 음원 단위 CPU 병렬 평가")
    a.add_argument("--resume", action="store_true", help="입력 해시가 같은 중단된 평가만 재개")
    a.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto", help="평가 장치: 기본은 GPU 우선 자동 선택, CUDA는 FP32 배치·CPU 경계 재확인")
    a.add_argument("--gpu-batch-size", type=int, default=128)
    a.add_argument("--gpu-boundary-margin", type=float, default=.001)
    a.add_argument("--accept-gpu-numerical-differences", action="store_true", default=None, help="이전 명령 호환 옵션; GPU 평가의 미세한 점수 차이는 기본 허용하고 경계는 CPU 재확인")
    a.add_argument("--cpu-prefix-import", type=Path, help="검증된 CPU 완료분을 보존하는 명시적 이관 증빙")
    a = sub.add_parser("select-candidate", help="저장된 개발 평가로 단일/앙상블 실행 묶음 선택")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--evaluation", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--candidate", required=True, help="개발 평가에 있는 정확한 모델 ID 또는 mean")
    a = sub.add_parser("tune", help="개발 점수 캐시로 임계값·연속 프레임 비교")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--evaluation", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--thresholds", type=float, nargs="+", required=True); a.add_argument("--consecutive", type=int, nargs="+", default=[2])
    a.add_argument("--candidate", help="감지 기준을 비교할 모델 ID; 생략하면 실제 bundle 출력")
    a = sub.add_parser("select-runtime", help="선택한 개발 실행 설정을 새 프로젝트에 적용")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a = sub.add_parser("import-references", help="청취 근거가 있는 Qwen 합성 참조 뱅크 재사용")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--source-project", type=Path, required=True)
    a.add_argument("--split", choices=["train", "validation", "final"], default="train")
    a = sub.add_parser("freeze", help="개발 채택 결과로 모델·실행 설정 Final 동결")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--development-report", type=Path, required=True)
    a = sub.add_parser("site", help="완료 평가의 통계·연속 청취 사이트 파일")
    a.add_argument("--evaluation", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("serve", help="저장된 결과 사이트 서버")
    a.add_argument("--directory", type=Path, required=True); a.add_argument("--port", type=int, default=8791)
    return p


def dispatch(a):
    if a.command == "web":
        from .studio import serve_studio
        return serve_studio(a.workspace, a.port, not a.no_browser)
    if a.command == "datasets":
        from .downloads import dataset_catalog
        return dataset_catalog()
    if a.command == "negative-plan":
        from .commercial_sources import commercial_negative_plan
        return commercial_negative_plan(a.generic_hours, a.korean_hours)
    if a.command == "download-fma":
        from .fma_download import download_fma
        import math
        if a.max_download_gib is not None and (not math.isfinite(a.max_download_gib)
                or a.max_download_gib <= 0 or not math.isfinite(a.max_download_gib * 1024**3)):
            raise ValueError("--max-download-gib must be positive and finite")
        return download_fma(a.output, hours=a.hours, seed=a.seed, plan_only=a.plan_only,
            max_download_bytes=int(a.max_download_gib * 1024**3) if a.max_download_gib is not None else None,
            license_policy=a.license_policy, exclusions=a.exclude)
    if a.command == "download-source":
        from .commercial_sources import resolve_commercial_source, download_commercial_source
        import math
        if a.max_download_gib is not None:
            if (not math.isfinite(a.max_download_gib) or a.max_download_gib <= 0
                    or not math.isfinite(a.max_download_gib * 1024**3) or not a.output):
                raise ValueError("다운로드하려면 양수 --max-download-gib와 --output을 지정하세요.")
        elif a.output is not None or a.metadata_only:
            raise ValueError("--output에 다운로드하려면 최대 용량 --max-download-gib도 지정하세요.")
        if a.whole_archive and a.source != "fma_full":
            raise ValueError("--whole-archive is only for the explicit FMA full ZIP")
        selective_fma = a.source == "fma_full" and not a.whole_archive and not a.metadata_only and a.parts != ["metadata"]
        if selective_fma:
            if a.parts not in (None, ["full"]):
                raise ValueError("FMA full uses full/metadata; do not mix archive variants")
            if a.max_download_gib is None:
                return {"source": "fma_full", "download_mode": "selective", "target_hours": a.hours,
                    "next": "download-fma --output <folder> --hours <hours> --plan-only",
                    "audio_download_started": False, "content_approved": False}
            from .fma_download import download_fma
            return download_fma(a.output, hours=a.hours, seed=a.seed,
                max_download_bytes=int(a.max_download_gib * 1024**3),
                license_policy=a.license_policy, exclusions=a.exclude)
        default_fma_part = {"fma_large": "large", "fma_full": "full"}.get(a.source)
        parts = a.parts or ([default_fma_part] if default_fma_part else [])
        if a.source in ("audioset_unbalanced", "yodas2"):
            try:
                parts = [int(value) for value in parts]
            except ValueError as error:
                raise ValueError("AudioSet·YODAS2 묶음 번호는 정수여야 합니다.") from error
        selection = {"parts": parts}
        if a.language:
            selection["language"] = a.language
        if a.subset:
            selection["subset"] = a.subset
        record = resolve_commercial_source(a.source, selection=selection)
        if a.max_download_gib is None:
            return record
        return download_commercial_source(record, a.output,
            max_download_bytes=int(a.max_download_gib * 1024**3),
            **({"metadata_only": True} if a.metadata_only else {}))
    if a.command == "download-data":
        from .downloads import download_dataset
        return download_dataset(a.dataset, a.output)
    if a.command == "storage":
        from .storage import describe_storage
        return describe_storage(a.project)
    if a.command == "init":
        from .studio_steps import common_evaluation_policy
        choices = selected_choices(a, default_profile())
        project = create_project(a.output, a.wakeword, a.id, **choices,
                                 validation_voice_families=a.validation_voices)
        # Only the newly initialized configuration adopts the current common
        # call count. The pinned profile and saved experiments stay intact.
        _, config = load_project(project)
        config["targets"]["validation_originals"] = common_evaluation_policy()["validation_originals"]
        write_json(project, config)
        return {"project": str(project), "workload": workload_summary(load_project(project)[1]),
                "next": "참조·파일럿·자료 준비 후 train 실행; init은 학습을 시작하지 않습니다."}
    if a.command == "configure":
        _, config = load_project(a.project)
        project = save_configuration(a.project, a.output, **selected_choices(a, config))
        return {"project": str(project), "workload": workload_summary(load_project(project)[1]),
                "note": "기존 설정·모델·receipt 보존. 새 설정은 새 학습 output에 사용하세요."}
    if a.command == "settings":
        _, config = load_project(a.project)
        result = {"workload": workload_summary(config)}
        if a.advanced:
            result.update(training=config["training"], batch=config["batch"], augmentation=config["augmentation"],
                          runtime=config["runtime"], evaluation_targets=config["evaluation_targets"])
        return result
    if a.command in ("check", "doctor"):
        from .benchmark import doctor
        report = doctor(include_tts=True) if a.tts else doctor()
        if a.tts and not report["tts_check"]["ready"]:
            raise ValueError("TTS 설치 확인 실패: " + "; ".join(report["tts_check"]["reasons"]))
        return report
    if a.command == "benchmark":
        from .benchmark import benchmark_training
        return benchmark_training(a.output, a.steps)
    if a.command == "sample":
        from .assets import export_sample
        from .runtime import sample_bundle, verify_bundle
        return export_sample(a.output) if a.output else verify_bundle(sample_bundle())
    if a.command == "demo":
        from .demo import run_demo
        return run_demo(a.output)
    if a.command == "infer":
        from .runtime import WakeWordDetector
        result = WakeWordDetector(a.bundle).predict_file(a.audio)
        if a.output:
            write_json(a.output, result)
        return result
    if a.command == "mic":
        from .microphone import serve_microphone
        return serve_microphone(a.bundle, a.port)
    if a.command == "status":
        return status(a.project)
    if a.command == "audit":
        from .data import audit_project
        report = audit_project(a.project)
        root, _ = load_project(a.project)
        write_json(root / "reports" / "source_audit.json", report)
        if not report["passed"]:
            raise ValueError("Source audit failed: " + str(report["errors"][:10]))
        return report
    if a.command == "audit-final":
        from .common import require_final_lock, read_rows
        from .final_integrity import audit_independent_final
        require_final_lock(a.project)
        if a.output.exists():
            raise ValueError("Final source audit reports are immutable; choose a new output file")
        report = audit_independent_final(a.project, extra_rows=read_rows(a.manifest) if a.manifest else None)
        write_json(a.output, report)
        if not report["passed"]:
            raise ValueError("Independent Final source audit failed: " + str(report["errors"][:10]))
        return report
    if a.command == "prepare-frontend":
        from .assets import prepare_frontend
        root, config = load_project(a.project)
        return prepare_frontend(resolved(root, config["frontend_dir"]))
    if a.command == "download-generic":
        from .assets import download_generic
        return download_generic(a.output)
    if a.command == "import-generic":
        from .assets import import_generic
        from .frontend import StreamingFrontend
        root, config = load_project(a.project)
        frontend = StreamingFrontend(resolved(root, config["frontend_dir"]))
        return import_generic(a.features, a.output, frontend.identity)
    if a.command == "index-data":
        from .preparation import create_source_index
        return create_source_index(a.input.resolve(), output=a.output.resolve(), dataset=a.dataset,
                                   metadata=a.metadata.resolve() if a.metadata else None,
                                   ontology=a.ontology.resolve() if a.ontology else None,
                                   room_metadata=a.room_metadata.resolve() if a.room_metadata else None,
                                   dataset_id=a.dataset_id, source_url=a.source_url, license=a.license)
    if a.command == "prepare-materials":
        from .preparation import prepare_public_audio
        return prepare_public_audio(a.project, input_directory=a.input.resolve(), output=a.output.resolve(),
            dataset_id=a.dataset_id, source_url=a.source_url, license=a.license, category=a.category,
            category_basis=a.category_basis, language=a.language, speech=a.speech,
            source_index=a.source_index.resolve() if a.source_index else None,
            target_hours=a.hours, validation_fraction=a.validation_fraction,
            seed=a.seed, representatives_per_stratum=a.representatives, preview_seconds=a.preview_seconds)
    if a.command == "admit-materials":
        from .preparation import apply_prepared_review
        return apply_prepared_review(a.project, preparation=a.preparation.resolve(),
            answers=a.answers.resolve() if a.answers else None,
            output=a.output.resolve() if a.output else None, pools=a.pools, connect=a.connect)
    if a.command == "connect-materials":
        from .preparation import connect_materials
        return connect_materials(a.project, admissions=[p.resolve() for p in a.admissions],
                                 pools=a.pools, output=a.output.resolve() if a.output else None)
    if a.command == "windows":
        from .data import prepare_windows
        return prepare_windows(a.project, a.pool, a.output)
    if a.command in ("references", "generate"):
        from .tts import generate_references
        if a.command == "references":
            return generate_references(a.project, a.split, a.limit)
        if a.pilot and a.method == "clone" and a.split == "train":
            from .pilot import prepare_pilot
            return prepare_pilot(a.project, train_count=a.limit,
                                 tts_backend=a.tts_backend, tts_cpu_threads=a.tts_cpu_threads)
        from .tts_runtime import generate_with_runtime
        return generate_with_runtime(a.project, a.split, a.limit, a.pilot, a.method, a.tts_backend, a.tts_cpu_threads)
    if a.command == "check-references":
        from .reference_qc import admit_references
        if any(value is not None and value < 1 for value in (a.train_count, a.validation_count)):
            raise ValueError("기준 목소리 수는 1 이상이어야 합니다.")
        from .common import read_rows
        root, _ = load_project(a.project)
        counts = {split: value if value is not None else len(read_rows(root / "data/references" / split / "references.jsonl"))
                  for split, value in (("train", a.train_count), ("validation", a.validation_count))}
        counts = {split: count for split, count in counts.items() if count != 0}
        return admit_references(a.project, counts)
    if a.command == "generate-hard":
        from .hard_tts import generate_hard
        from .tts_runtime import CloneRuntime
        root, config = load_project(a.project)
        if a.tts_cpu_threads < 1:
            raise ValueError("CPU thread count must be positive")
        model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, a.tts_backend, a.tts_cpu_threads)
        if a.count is not None:
            return generate_hard(a.project, a.phrases, a.count, model=model)
        per_phrase = a.samples_per_phrase
        if per_phrase is None:
            from .common import read_json, read_rows
            accepted = set(read_json(root / "approval.json").get("accepted_reference_families", []))
            per_phrase = sum(row.get("status") == "accepted" or row.get("voice_family_id") in accepted
                             for row in read_rows(root / "data/references/train/references.jsonl"))
        return generate_hard(a.project, a.phrases, model=model, samples_per_phrase=per_phrase)
    if a.command == "review":
        from .review import serve_review
        return serve_review(a.project, a.queue, a.kind, a.port, a.pool_manifest)
    if a.command == "features":
        from .features import build_features
        return build_features(a.project, a.role, a.output, a.rounds, a.batch_size, a.smoke, a.resume)
    if a.command == "verify-cache":
        from .features import verify_cache
        return verify_cache(a.directory)
    if a.command == "compare-rounds":
        from .features import compare_rounds
        return compare_rounds(a.first, a.second)
    if a.command == "train":
        from .train import train
        return train(a.project, a.output, a.smoke, a.resume)
    if a.command == "package":
        from .bundle import package_run
        return package_run(a.project, a.run, a.output)
    if a.command == "prepare-eval":
        from .paired import prepare_evaluation
        return prepare_evaluation(a.project, a.output, a.split, a.originals, a.noise_sections,
                                  noise_selection=a.noise_selection, reverb_originals=a.reverb_originals,
                                  reverb_selection=a.reverb_selection)
    if a.command == "setup-gpu":
        from .gpu_runtime_install import setup_gpu_runtime
        return setup_gpu_runtime(a.project.resolve().parent)
    if a.command == "evaluate":
        from .evaluate import evaluate
        return evaluate(a.project, a.bundle, a.output, a.stage, a.manifest, a.smoke, a.candidate,
                        workers=a.workers, resume=a.resume, backend=a.backend,
                        gpu_batch_size=a.gpu_batch_size, boundary_margin=a.gpu_boundary_margin,
                        numerical_differences_approved=a.accept_gpu_numerical_differences,
                        cpu_prefix_import=a.cpu_prefix_import)
    if a.command == "select-candidate":
        from .selection import select_candidate
        return select_candidate(a.project, a.bundle, a.evaluation, a.output, a.candidate)
    if a.command == "tune":
        from .tune import tune
        return tune(a.project, a.bundle, a.evaluation, a.output, a.thresholds, a.consecutive, a.candidate)
    if a.command == "select-runtime":
        from .tune import select_runtime
        return select_runtime(a.project, a.bundle)
    if a.command == "import-references":
        from .reference_bank import import_references
        return import_references(a.project, a.source_project, a.split)
    if a.command == "freeze":
        from .bundle import freeze
        return freeze(a.project, a.bundle, a.development_report)
    if a.command == "site":
        from .site import export_site
        return export_site(a.evaluation, a.output)
    if a.command == "serve":
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        if not (a.directory / "index.html").is_file():
            raise ValueError("No exported index.html in this directory")
        print(f"http://127.0.0.1:{a.port}/", flush=True)
        return ThreadingHTTPServer(("127.0.0.1", a.port), partial(SimpleHTTPRequestHandler, directory=str(a.directory.resolve()))).serve_forever()
    raise ValueError("Unknown command")


def main(argv=None):
    # UTF-8 reports/help in pipes on Korean Windows; leave the native console alone.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8")
    p = parser()
    args = p.parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        p.exit(1, "koww: " + str(error) + "\n")
