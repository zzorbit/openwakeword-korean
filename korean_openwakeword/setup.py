"""Small terminal setup: workload first, learning internals only on request."""
import sys
import re

from .project import configure_budget, workload_summary


def _ask(label, default, convert, valid, input_fn, output_fn):
    while True:
        try:
            answer = input_fn(f"{label} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise ValueError("Setup cancelled; no project configuration was saved") from None
        try:
            value = convert(answer) if answer else default
            if valid(value):
                return value
        except (ValueError, TypeError):
            pass
        output_fn("입력값을 다시 확인하세요.")


def setup_choices(config, *, interactive=None, advanced=False, positive_originals=None,
                  augmentation_rounds=None, updates=None, training_overrides=None,
                  input_fn=None, output_fn=None):
    """TTY opens the wizard; redirected/script invocations never block on input."""
    input_fn, output_fn = input_fn or input, output_fn or print
    if interactive is None:
        interactive = sys.stdin.isatty()
    if advanced and not interactive:
        raise ValueError("--advanced opens the interactive settings; use --interactive or explicit setting flags")
    choices = dict(positive_originals=positive_originals, augmentation_rounds=augmentation_rounds,
                   updates=updates, training_overrides=dict(training_overrides or {}))
    if not interactive:
        configure_budget(config, **choices)  # fail before any files are created
        return choices
    output_fn("초기 설정: 작업량 3가지만 선택합니다. Enter는 표시된 시작값을 사용합니다.")
    output_fn("원본 수는 증강 전 TTS 수입니다. R은 원본당 증강 횟수, 학습량은 모델 한 개의 optimizer update 수입니다.")
    if positive_originals is None:
        choices["positive_originals"] = _ask("TTS 원본 수 (예: 20000 / 50000 / 100000 또는 직접 입력)",
            config["targets"]["positive_originals"], int, lambda x: type(x) is int and x > 0, input_fn, output_fn)
    if augmentation_rounds is None:
        choices["augmentation_rounds"] = _ask("증강 횟수 (1 / 3 / 5)", config["augmentation_rounds"], int,
            lambda x: x in (1, 3, 5), input_fn, output_fn)
    if updates is None:
        chosen_updates = _ask("모델당 학습 update 수 (예: 10000 / 20000 / 50000 또는 직접 입력)",
            config["training"]["updates"], int, lambda x: x > 0, input_fn, output_fn)
        if chosen_updates != config["training"]["updates"]:
            choices["updates"] = chosen_updates
    if not advanced:
        advanced = _ask("세부 설정 열기? (y / n)", "n", str.lower,
            lambda x: x in ("y", "n"), input_fn, output_fn) == "y"
    if advanced:
        output_fn("세부 설정: 이번 춘식아 실험의 권장 시작값입니다. 모든 한국어 호출어의 최적값은 아닙니다.")
        current = configure_budget(config, **choices)["training"]
        fields = (
            ("architecture", "판별기 (cnn64 / cnn128 / mlp32 / mlp128 / mlp256)", str, lambda x: x in ("cnn64", "cnn128", "mlp32", "mlp128", "mlp256")),
            ("optimizer", "Optimizer (Adam / AdamW)", str, lambda x: x in ("Adam", "AdamW")),
            ("learning_rate", "학습률", float, lambda x: 0 < x < float("inf")),
            ("weight_decay", "Weight decay", float, lambda x: 0 <= x < float("inf")),
            ("dropout", "Dropout", float, lambda x: 0 <= x < 1),
            ("positive_loss_weight", "Positive loss 가중치", float, lambda x: 0 < x < float("inf")),
            ("schedule", "학습률 변화 (constant / cosine_with_2percent_warmup)", str, lambda x: x in ("constant", "cosine_with_2percent_warmup")),
        )
        for key, label, convert, valid in fields:
            choices["training_overrides"][key] = _ask(label, current[key], convert, valid, input_fn, output_fn)
        members = _ask("독립 모델 수 (1 / 3; 3개는 점수 평균)", len(current["seeds"]), int,
            lambda x: x in (1, 3), input_fn, output_fn)
        seeds = current["seeds"][:members]
        while len(seeds) < members:
            seeds.append(max(seeds) + 1)
        choices["training_overrides"]["seeds"] = seeds
        choices["training_overrides"]["device"] = _ask("학습 자원 (auto / cpu / cuda:0)", current["device"], str,
            lambda x: x in ("auto", "cpu") or re.fullmatch(r"cuda(?::[0-9]+)?", x) is not None, input_fn, output_fn)
    selected = configure_budget(config, **choices)
    summary = workload_summary(selected)
    output_fn(f"선택 결과: 원본 {summary['positive_tts_originals']:,} × R{summary['augmentation_rounds']} = {summary['positive_feature_rows']:,}개 Positive feature")
    output_fn(f"Positive feature 용량 약 {summary['positive_feature_payload_gib']:.3f} GiB (WAV·Negative·기록 제외)")
    output_fn(f"{summary['model_count']}개 {summary['architecture']} × {summary['updates_per_model']:,} update = 총 {summary['total_optimizer_updates']:,} update")
    output_fn("실제 시간은 이 컴퓨터에서 측정해야 합니다. 이 단계는 설정만 저장하고 생성·학습을 시작하지 않습니다.")
    return choices
