"""Launch Studio and provide utilities for already trained models."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__


def parser():
    p = argparse.ArgumentParser(
        prog="koww",
        description="한국어 호출어 모델의 웹 학습과 실행 도구",
        epilog="새 모델의 자료 준비·검수·생성·학습·평가는 koww web으로 연 웹 화면에서 진행합니다.",
    )
    p.add_argument("--version", action="version", version="korean-openwakeword " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    web = sub.add_parser("web", help="웹 학습 화면 열기")
    web.add_argument("--workspace", type=Path, help="프로젝트·공유 자료·캐시 저장 폴더; 기본은 현재 폴더")
    web.add_argument("--port", type=int, default=8793)
    web.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")

    check = sub.add_parser("check", aliases=["doctor"], help="설치·GPU 상태 확인")
    check.add_argument("--tts", action="store_true", help="Qwen·PyTorch·torchaudio 버전과 실제 import 확인")

    sample = sub.add_parser("sample", help="춘식아 예시 모델 검증 또는 복사")
    sample.add_argument("--output", type=Path)
    demo = sub.add_parser("demo", help="동봉 모델과 예시 WAV로 실행 예제 열기")
    demo.add_argument("--output", type=Path, required=True)

    infer = sub.add_parser("infer", help="학습된 모델로 WAV의 호출 감지")
    infer.add_argument("--bundle", type=Path)
    infer.add_argument("--audio", required=True, type=Path)
    infer.add_argument("--output", type=Path)
    mic = sub.add_parser("mic", help="학습된 모델의 브라우저 마이크 시험")
    mic.add_argument("--bundle", type=Path, help="학습한 실행 bundle; 생략하면 동봉 춘식아 모델")
    mic.add_argument("--port", type=int, default=8792, help="로컬 시험 화면 포트; 기본 8792")

    sub.add_parser("datasets", help="동봉 모델에 사용한 공개 자료의 출처·버전·용량 확인")
    return p


def dispatch(args):
    if args.command == "web":
        from .studio import serve_studio
        return serve_studio(args.workspace, args.port, not args.no_browser)
    if args.command in ("check", "doctor"):
        from .benchmark import doctor
        report = doctor(include_tts=True) if args.tts else doctor()
        if args.tts and not report["tts_check"]["ready"]:
            raise ValueError("TTS 설치 확인 실패: " + "; ".join(report["tts_check"]["reasons"]))
        return report
    if args.command == "sample":
        from .assets import export_sample
        from .runtime import sample_bundle, verify_bundle
        return export_sample(args.output) if args.output else verify_bundle(sample_bundle())
    if args.command == "demo":
        from .demo import run_demo
        return run_demo(args.output)
    if args.command == "infer":
        from .common import write_json
        from .runtime import WakeWordDetector
        result = WakeWordDetector(args.bundle).predict_file(args.audio)
        if args.output:
            write_json(args.output, result)
        return result
    if args.command == "mic":
        from .microphone import serve_microphone
        return serve_microphone(args.bundle, args.port)
    if args.command == "datasets":
        from .downloads import dataset_catalog
        return dataset_catalog()
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
