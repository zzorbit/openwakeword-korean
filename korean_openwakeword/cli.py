"""koww: stable entry points without historical project paths."""
import argparse
import json
import sys
from pathlib import Path

from .common import resolved, write_json
from .project import create_project, load_project, status, default_profile, workload_summary, save_configuration


def setup_arguments(a):
    basic = a.add_argument_group("초기 선택: 작업량·자료 규모")
    basic.add_argument("--originals", type=int, help="증강 전 Positive TTS 원본 수; 시작값 100000")
    basic.add_argument("--rounds", type=int, choices=(1, 3, 5), help="원본당 증강 횟수; 시작값 5")
    basic.add_argument("--updates", type=int, help="모델 한 개의 optimizer update 수; 시작값 100000")
    basic.add_argument("--generic-hours", type=float, help="일반 비호출 자료 목표 시간; 시작값 7500")
    basic.add_argument("--korean-hours", type=float, help="한국어 비호출 자료 목표 시간; 시작값 2500")
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
    a = sub.add_parser("init", help="새 한국어 호출어 프로젝트")
    a.add_argument("--wakeword", required=True); a.add_argument("--id", required=True); a.add_argument("--output", type=Path, required=True)
    setup_arguments(a)
    a = sub.add_parser("configure", help="기존 설정을 보존하고 별도 작업량·세부 설정 snapshot 작성")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    setup_arguments(a)
    a = sub.add_parser("settings", help="현재 작업량 또는 세부 학습 설정 확인")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--advanced", action="store_true")
    sub.add_parser("check", aliases=["doctor"], help="설치·GPU 상태 확인")
    a = sub.add_parser("sample", help="춘식아 예시 모델 검증 또는 복사")
    a.add_argument("--output", type=Path)
    a = sub.add_parser("demo", help="동봉 예시 WAV 재채점·결과 사이트 생성")
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("infer", help="실제 WAV의 CPU ONNX 호출 결과")
    a.add_argument("--bundle", type=Path); a.add_argument("--audio", required=True, type=Path); a.add_argument("--output", type=Path)
    a = sub.add_parser("benchmark", help="CPU/GPU의 학습 업데이트 비용·수치 차이 측정")
    a.add_argument("--output", type=Path, required=True); a.add_argument("--steps", type=int, default=20)
    a = sub.add_parser("download-generic", help="검증된 공개 Generic feature 다운로드 (~17.28GB)")
    a.add_argument("--output", type=Path, required=True)
    sub.add_parser("datasets", help="춘식아에 사용한 공개 자료의 출처·고정 버전·용량 확인")
    a = sub.add_parser("download-data", help="고정 SHA256 공개 자료 다운로드·중단 재개 (내용 승인 별도)")
    a.add_argument("--dataset", required=True); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("import-generic", help="전체 공개 Generic feature를 외부 경로로 재사용")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--features", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    for name in ("status", "audit", "prepare-frontend", "storage"):
        a = sub.add_parser(name); a.add_argument("--project", type=Path, required=True)
    a = sub.add_parser("windows", help="공개 Negative 원본의 고정 2초 구간 manifest")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--pool", choices=("generic_negative", "korean_negative", "hard_negative"), required=True)
    a.add_argument("--output", type=Path, required=True)
    for name in ("references", "generate"):
        a = sub.add_parser(name); a.add_argument("--project", type=Path, required=True)
        a.add_argument("--split", choices=("train", "validation", "final"), default="train"); a.add_argument("--limit", type=int)
        if name == "generate":
            a.add_argument("--pilot", action="store_true"); a.add_argument("--method", choices=("clone", "design"), default="clone")
    a = sub.add_parser("review", help="검수 큐 전체 연속 청취·직접 저장")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--queue", type=Path, required=True)
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
    a = sub.add_parser("evaluate", help="동일 WAV의 단일·평균 모델 평가")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--manifest", type=Path); a.add_argument("--stage", choices=("development", "final"), default="development")
    a.add_argument("--output", type=Path, required=True); a.add_argument("--smoke", action="store_true")
    a = sub.add_parser("tune", help="개발 점수 캐시로 임계값·연속 프레임 비교")
    a.add_argument("--project", type=Path, required=True); a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--evaluation", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--thresholds", type=float, nargs="+", required=True); a.add_argument("--consecutive", type=int, nargs="+", default=[2])
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
    if a.command == "datasets":
        from .downloads import dataset_catalog
        return dataset_catalog()
    if a.command == "download-data":
        from .downloads import download_dataset
        return download_dataset(a.dataset, a.output)
    if a.command == "storage":
        from .storage import describe_storage
        return describe_storage(a.project)
    if a.command == "init":
        choices = selected_choices(a, default_profile())
        project = create_project(a.output, a.wakeword, a.id, **choices)
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
        return doctor()
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
    if a.command == "windows":
        from .data import prepare_windows
        return prepare_windows(a.project, a.pool, a.output)
    if a.command in ("references", "generate"):
        from .tts import generate, generate_references
        return generate_references(a.project, a.split, a.limit) if a.command == "references" else generate(a.project, a.split, a.limit, a.pilot, a.method)
    if a.command == "review":
        from .review import serve_review
        return serve_review(a.project, a.queue, a.kind, a.port)
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
        return prepare_evaluation(a.project, a.output, a.split, a.originals, a.noise_sections)
    if a.command == "evaluate":
        from .evaluate import evaluate
        return evaluate(a.project, a.bundle, a.output, a.stage, a.manifest, a.smoke)
    if a.command == "tune":
        from .tune import tune
        return tune(a.project, a.bundle, a.evaluation, a.output, a.thresholds, a.consecutive)
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
