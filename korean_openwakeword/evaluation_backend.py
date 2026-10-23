"""Choose an evaluation device without changing an existing score contract.

New Development runs prefer the qualified CUDA batch path. A separate process
checks real CUDA sessions before selection; the presence of a driver or a GPU
package alone is not proof that its DLLs and kernels work.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import read_json
from .frontend import PINNED_ASSETS

NUMERICAL_POLICY = "cuda_fp32_whole_session_cpu_boundary_v1"


def _runtime_environment(project_root):
    environment = {}
    # An explicit runtime path is an intentional override of the managed one.
    if project_root is not None and not os.environ.get("KOWW_GPU_RUNTIME_PATH"):
        from .gpu_runtime_install import gpu_runtime_environment
        environment.update(gpu_runtime_environment(project_root))
    for key in ("KOWW_GPU_RUNTIME_PATH", "KOWW_GPU_DLL_PATHS"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    return environment


def _bundle_limitation(bundle):
    if bundle is None:
        return "CUDA 평가에는 모델 묶음 경로가 필요합니다."
    manifest = read_json(Path(bundle).resolve() / "manifest.json")
    if len(manifest.get("members", [])) != 3 or manifest.get("aggregation") != "ordered_float64_mean_to_float32":
        return "현재 GPU 배치 평가는 독립 모델 3개와 평균 점수 구성을 지원합니다. 이 모델은 CPU로 평가합니다."
    frontend = {Path(row["path"]).name: row.get("sha256") for row in manifest.get("frontend", [])}
    if any(frontend.get(name) != expected for name, expected in PINNED_ASSETS.items()):
        return "공유 음성 처리 모델이 GPU 배치 검증에 사용한 버전과 다릅니다."
    return None


def _loaded_runtime_conflict(environment):
    loaded = sys.modules.get("onnxruntime")
    if loaded is None:
        return None
    target = environment.get("KOWW_GPU_RUNTIME_PATH")
    module_path = getattr(loaded, "__file__", None)
    if target and (not module_path or not Path(module_path).resolve().is_relative_to(Path(target).resolve())):
        return "이 Python 프로세스에서 다른 ONNX Runtime을 이미 불러왔습니다. GPU 런타임은 새 프로세스에서 시작해야 합니다."
    if "CUDAExecutionProvider" not in loaded.get_available_providers():
        return "이 Python 프로세스에서 CPU용 ONNX Runtime을 이미 불러왔습니다. GPU 런타임은 새 프로세스에서 시작해야 합니다."
    return None


def probe_cuda_backend(bundle, *, environment=None, timeout=30):
    """Return capability evidence from an isolated, bounded CUDA test process."""
    child_environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    child_environment.update(environment or {})
    paths = [str(Path(__file__).resolve().parent.parent)]
    if child_environment.get("KOWW_GPU_RUNTIME_PATH"):
        paths.insert(0, child_environment["KOWW_GPU_RUNTIME_PATH"])
    if child_environment.get("PYTHONPATH"):
        paths.append(child_environment["PYTHONPATH"])
    child_environment["PYTHONPATH"] = os.pathsep.join(paths)
    command = [sys.executable, "-B", "-m", "korean_openwakeword.evaluation_backend", "--probe-cuda", str(Path(bundle).resolve())]
    try:
        completed = subprocess.run(command, env=child_environment, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            cwd=Path(__file__).resolve().parent.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"available": False, "reason_code": "cuda_probe_timeout",
                "reason": "GPU 실행 확인이 제한 시간 안에 끝나지 않아 CPU를 선택했습니다."}
    except OSError as error:
        return {"available": False, "reason_code": "cuda_probe_start_failed", "reason": str(error)}
    lines = completed.stdout.splitlines()
    try:
        result = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
    except (StopIteration, ValueError):
        return {"available": False, "reason_code": "cuda_probe_failed",
                "reason": "GPU 실행 확인에 실패했습니다.",
                "detail": (completed.stderr or completed.stdout)[-2000:], "returncode": completed.returncode}
    if not isinstance(result, dict) or type(result.get("available")) is not bool:
        return {"available": False, "reason_code": "cuda_probe_invalid", "reason": "GPU 실행 확인 결과를 읽을 수 없습니다."}
    if completed.returncode and result["available"]:
        return {"available": False, "reason_code": "cuda_probe_failed", "reason": "GPU 실행 확인 프로세스가 정상 종료되지 않았습니다.",
                "returncode": completed.returncode}
    return result


def resolve_evaluation_backend(requested="auto", *, stage="development", bundle=None,
                               resume_backend=None, project_root=None):
    """Resolve a new run or preserve the backend of a resumable checkpoint.

    The returned runtime environment is applied only by the caller's dedicated
    evaluation worker. This function does not install packages, import another
    ORT build into the caller, change environment variables or rewrite records.
    Explicit CUDA and CUDA resumes fail closed instead of becoming CPU runs.
    """
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Evaluation backend must be auto, cpu or cuda")
    if stage not in ("development", "final"):
        raise ValueError("Unknown evaluation stage")
    if resume_backend not in (None, "cpu", "cuda"):
        raise ValueError("Existing evaluation backend must be cpu or cuda")
    if resume_backend and requested not in ("auto", resume_backend):
        raise ValueError("An existing evaluation backend cannot change during resume")
    explicit_cuda = requested == "cuda" or resume_backend == "cuda"
    if stage == "final" and explicit_cuda:
        raise ValueError("Independent Final retains its frozen CPU reference protocol")
    result = {"requested_backend": requested, "backend": "cpu", "resume_backend": resume_backend,
              "runtime_environment": {}, "numerical_policy": None, "probe": None}
    if stage == "final":
        return {**result, "reason_code": "final_cpu_reference", "reason": "독립 최종 평가는 고정된 CPU 기준으로 진행합니다."}
    if requested == "cpu" or resume_backend == "cpu":
        return {**result, "reason_code": "resume_cpu_reference" if resume_backend else "explicit_cpu",
                "reason": "기존 CPU 평가의 연산 방식을 유지합니다." if resume_backend else "CPU 평가를 선택했습니다."}
    limitation = _bundle_limitation(bundle)
    if limitation:
        if explicit_cuda:
            raise ValueError(limitation)
        return {**result, "reason_code": "unsupported_gpu_bundle", "reason": limitation}
    try:
        environment = _runtime_environment(project_root)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        if explicit_cuda:
            raise RuntimeError("CUDA runtime configuration is unavailable: " + str(error)) from error
        return {**result, "reason_code": "cuda_runtime_configuration_failed",
                "reason": "GPU 런타임 구성을 확인할 수 없어 CPU로 평가합니다.", "detail": str(error)[:2000]}
    conflict = _loaded_runtime_conflict(environment)
    if conflict:
        if explicit_cuda:
            raise RuntimeError(conflict)
        return {**result, "reason_code": "runtime_already_loaded", "reason": conflict}
    probe = probe_cuda_backend(bundle, environment=environment)
    if not probe["available"]:
        if explicit_cuda:
            raise RuntimeError("CUDA evaluation is unavailable: " + probe.get("reason", "GPU probe failed"))
        return {**result, "reason_code": probe.get("reason_code", "cuda_unavailable"),
                "reason": probe.get("reason", "GPU를 사용할 수 없어 CPU로 평가합니다."), "probe": probe}
    return {**result, "backend": "cuda", "runtime_environment": environment,
            "reason_code": "resume_cuda" if resume_backend else "cuda_available",
            "reason": "GPU 배치 평가를 사용하고 임계값 근처의 음원은 CPU로 다시 확인합니다.",
            "numerical_policy": NUMERICAL_POLICY, "probe": probe}


def _probe_main(bundle):
    try:
        import numpy as np
        from .gpu_evaluation import _GpuBatchScorer
        from .runtime import bundle_candidates
        manifest = read_json(Path(bundle) / "manifest.json")
        stats = {}
        scorer = _GpuBatchScorer(bundle, [row["id"] for row in bundle_candidates(manifest)], stats)
        scores = scorer.score_batch([np.zeros(1280 * 3, dtype=np.int16), np.zeros(1280 * 2, dtype=np.int16)])
        if [value.shape for value in scores] != [(3, 4), (2, 4)] or any(not np.isfinite(value).all() for value in scores):
            raise ValueError("CUDA session execution returned invalid scores")
        print(json.dumps({"available": True, "reason_code": "cuda_sessions_verified", "reason": "CUDA FP32 세션 실행 확인 완료",
            "onnxruntime_version": stats["onnxruntime_version"], "onnxruntime_module": stats["onnxruntime_module"],
            "providers": stats["providers"], "precision": stats["precision"], "mel_derivation": stats["mel_derivation"]},
            ensure_ascii=True), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"available": False, "reason_code": "cuda_unavailable", "reason": "GPU 실행에 필요한 구성 요소를 사용할 수 없습니다.",
                          "detail": str(error)[:2000]}, ensure_ascii=True), flush=True)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--probe-cuda":
        raise SystemExit("This module is an internal evaluation capability probe")
    raise SystemExit(_probe_main(sys.argv[2]))
