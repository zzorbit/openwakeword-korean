"""Measure resources before large work; distinguish speed from numeric equality."""
import time
import importlib.metadata
import os
import re
import subprocess
import sys
import numpy as np
from .common import now, write_json
from .train import build_classifier, apply_backend
from .project import default_profile, training_batch


def _tts_import_messages(output):
    """Summarize only known optional Qwen 0.1.1 import noise.

    Qwen imports its unused 25Hz tokenizer even for our supported 12Hz models.
    Keep any other warnings visible, and never apply this filter to failures.
    """
    notes = []
    if "SoX could not be found!" in output:
        notes.append("SoX 미설치: 이 라이브러리의 Qwen 12Hz 음성 생성에는 필요하지 않습니다.")
        output = re.sub(r"SoX could not be found!.*?path variables\.[ \t]*", "", output, flags=re.S)
        # PySOX's shell probe writes directly to native stderr in these locales.
        for message in (
            r"'sox' is not recognized as an internal or external command,\s*operable program or batch file\.",
            r"'sox'은\(는\) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는\s*배치 파일이 아닙니다\.",
            r"(?:/[^\s:]+sh|sh): (?:\d+: )?sox: (?:not found|command not found)",
        ):
            output = re.sub(message, "", output)
    flash_warning = (
        "Warning: flash-attn is not installed. Will only run the manual PyTorch version. "
        "Please install flash-attn for faster inference."
    )
    if flash_warning in output:
        notes.append("flash-attn 미설치: 기본 SDPA 설정으로 사용할 수 있습니다.")
        output = re.sub(r"\*{8}\s*" + re.escape(flash_warning) + r"\s*\*{8}", "", output)
    return output.strip(), notes


def _check_tts_imports():
    """Capture native child stderr too, without changing vendor imports or PATH."""
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torchaudio; from qwen_tts import Qwen3TTSModel"],
            capture_output=True, timeout=120,
            env=environment, check=False,
        )
    except subprocess.TimeoutExpired:
        return ["TTS binary/import check timed out after 120 seconds"], "", []
    except OSError as error:
        return [f"TTS binary/import check could not start: {error}"], "", []
    # Python emits UTF-8; CMD's missing-command message may still use Windows ACP.
    lines = []
    for part in (result.stdout, result.stderr):
        for line in part.splitlines():
            try:
                lines.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append(line.decode("mbcs" if os.name == "nt" else "utf-8", errors="replace"))
    output = "\n".join(lines)
    if result.returncode:
        return [f"TTS binary/import check failed (exit {result.returncode}): {output[-6000:] or 'no diagnostic output'}"], "", []
    warnings, notes = _tts_import_messages(output)
    return [], warnings[-3000:], notes


def doctor(include_tts=False):
    versions = {}
    for name in ("korean-openwakeword", "openwakeword", "torch", "torchaudio", "onnxruntime", "qwen-tts", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"versions": versions, "reference_openwakeword_ready": versions["openwakeword"] == "0.6.0"}
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            report["GPU"] = {"name": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total}
    except ImportError:
        report["cuda_available"] = False
    if include_tts:
        reasons = []
        for name, expected in (("torch", "2.8.0"), ("torchaudio", "2.8.0"), ("qwen-tts", "0.1.1")):
            if not versions[name] or versions[name].split("+", 1)[0] != expected:
                reasons.append(f"{name}=={expected} is required for the supported TTS path")
        warnings, notes = "", []
        if not reasons:
            reasons, warnings, notes = _check_tts_imports()
        report["tts_check"] = {"ready": not reasons, "reasons": reasons,
                               "import_warnings": warnings, "notes": notes,
                               "model_download_or_generation": False,
                               "installation_guide": "https://pytorch.org/get-started/previous-versions/"}
    return report


def benchmark_training(output, steps=20, architecture="cnn128", batch=None):
    import torch
    if steps < 2:
        raise ValueError("Use at least two measured steps")
    settings = default_profile()["training"]
    batch = training_batch(default_profile()) if batch is None else dict(batch)
    batch_size = sum(batch.values())
    apply_backend(settings["backend"])
    torch.set_num_threads(settings["cpu_threads"])
    rng = np.random.default_rng(20260928)
    input_array = rng.normal(size=(batch_size, 16, 96)).astype(np.float32)
    torch.manual_seed(20260928)
    model = build_classifier(architecture).eval()
    with torch.no_grad():
        reference = model(torch.from_numpy(input_array)).numpy()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    results = {}
    for device in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        model = build_classifier(architecture)
        model.load_state_dict(initial); model.to(device).eval()
        x = torch.from_numpy(input_array).to(device)
        with torch.no_grad():
            probe = model(x).cpu().numpy()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
        y = torch.cat([torch.full((count, 1), float(role == "positive"), device=device)
                       for role, count in batch.items()])
        timings = []
        for step in range(steps + 3):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward(); optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if step >= 3:
                timings.append(time.perf_counter() - start)
        results[device] = {"warm_train_update_median_ms": float(np.median(timings) * 1000),
            "warm_train_update_p95_ms": float(np.quantile(timings, .95) * 1000),
            "initial_logits_max_difference_from_CPU": float(np.max(np.abs(probe - reference))),
            "initial_logits_bitwise_equal_to_CPU": bool(np.array_equal(probe.view("u4"), reference.view("u4"))),
            "scope": f"{architecture} representative resident batch; excludes full pool transfer/disk IO; not recall/FA equivalence"}
    report = {"kind": "CPU_GPU_training_resource_probe", "created_at": now(), "steps": steps, "architecture": architecture,
        "resources": doctor(), "results": results, "batch": batch, "batch_size": batch_size,
        "fastest_measured": min(results, key=lambda k: results[k]["warm_train_update_median_ms"]),
        "GPU_neural_inference_automatically_adopted": False,
        "note": "Benchmark is a functional resource probe, not another formal wakeword model."}
    write_json(output, report)
    return report
