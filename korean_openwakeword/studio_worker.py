"""Studio subprocess entry points, launched by StudioManager (not a public CLI)."""
from pathlib import Path
from contextlib import nullcontext
from collections import deque
import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

from .common import bound_file, digest, identity, now, read_json, write_json, writer_lock
from .storage import storage_environment
from .project import load_project


_TORCH_STAGES = {"references", "pilot", "positives", "positive_review", "hard_pilot", "hard_generation", "training"}


def _torch_install_command(backend):
    from .studio_steps import _dependency_plan, _installed_dependencies

    commands = _dependency_plan(_installed_dependencies(), backend)
    if commands and any(package.startswith("torch==") for package in commands[0]):
        return commands[0]


def _verify_selected_torch(backend):
    if backend not in ("cpu", "cu128"):
        return
    from .studio_steps import _dependency_plan

    try:
        import torch
        import torchaudio

        versions = {"torch": torch.__version__, "torchaudio": torchaudio.__version__}
        if any(spec.startswith("torch==") for spec in _dependency_plan(versions, backend)[0]):
            raise ValueError("설치된 PyTorch와 TorchAudio가 선택한 장치의 버전과 다릅니다.")
        if backend == "cu128":
            if not torch.cuda.is_available():
                raise ValueError("선택한 NVIDIA GPU를 사용할 수 없습니다. 드라이버를 확인하거나 CPU를 선택하세요.")
            torch.ones(1, device="cuda:0").add_(1).item()
    except (ImportError, OSError, RuntimeError) as error:
        raise ValueError("선택한 PyTorch 실행 도구로 연산할 수 없습니다. 설치 로그를 확인하세요.") from error


def _prepare_selected_torch(project_root, backend):
    """Restore the selected build if this virtual environment was switched."""
    if backend not in ("cpu", "cu128"):
        return
    command = _torch_install_command(backend)
    if command:
        if sys.prefix == sys.base_prefix:
            raise ValueError("선택한 PyTorch 도구를 사용하려면 웹 화면을 가상환경에서 실행하세요.")
        if "torch" in sys.modules:
            raise ValueError("학습 도구가 변경되었습니다. 현재 작업을 재개해 새 실행 환경을 불러오세요.")
        print("이 프로젝트에서 선택한 PyTorch 실행 도구를 확인하고 있습니다.", flush=True)
        try:
            with storage_environment(project_root):
                subprocess.run(command, cwd=str(project_root), check=True)
        except subprocess.CalledProcessError as error:
            raise ValueError("선택한 PyTorch 도구를 설치하지 못했습니다. 설치 로그를 확인한 뒤 재개하세요.") from error
        importlib.invalidate_caches()
    _verify_selected_torch(backend)


def _training_torch_backend(root, config, backend):
    if backend not in ("cpu", "cu128"):
        return None
    seed = config["training"]["seeds"][0]
    prior = root / "runs/studio_run" / f"member1_seed{seed}" / "contract.json"
    if prior.is_file():
        version = read_json(prior).get("versions", {}).get("torch", "")
        if version == "2.8.0+cu128":
            return "cu128"
        if version == ("2.8.0" if sys.platform == "darwin" else "2.8.0+cpu"):
            return "cpu"
        return None  # The existing training contract will verify its exact build.
    return backend


def _hard_stage(request, request_path, run_stage):
    from .hard_tts import HardGenerationPaused

    pause_path = request_path.parent / "pause.json"

    def pause_requested():
        try:
            return read_json(pause_path).get("task_id") == request["task_id"] if pause_path.exists() else False
        except FileNotFoundError:
            return False

    try:
        if not pause_requested():
            return run_stage(request["project"], request["workspace"], request["settings"], request["stage"],
                             request.get("context", {}), pause_requested=pause_requested)
    except HardGenerationPaused:
        pass
    return {"state": "paused"}


def _task_progress_writer(request, directory, request_sha256):
    """Best-effort display hints; receipt and artifact writes remain strict."""
    owner_path = directory / "owner.json"
    owner = read_json(owner_path) if owner_path.is_file() else {}
    previous_key, previous_time, previous_done = None, None, None
    phase, phase_started_at = None, None
    rates = deque(maxlen=24)
    rate_phase = None

    def report(record):
        nonlocal previous_key, previous_time, previous_done, phase, phase_started_at, rate_phase
        stamp = time.monotonic()
        wall = time.time()
        new_phase = record["phase"]
        if new_phase != phase:
            phase, phase_started_at = new_phase, wall
        hint = {**record, "phase_started_at": phase_started_at}
        sample = hint.pop("rate_sample", None)
        if phase in ("tts_generation", "training"):
            # A worker may alternate between prompt setup and synthesis; keep
            # recent completed samples while reporting the current phase.
            if phase != rate_phase:
                rates.clear()
                rate_phase = phase
            while rates and stamp - rates[0][0] > 120:
                rates.popleft()
            if sample is not None:
                count, seconds = sample
                if count >= 0 and math.isfinite(seconds) and seconds > 0:
                    rates.append((stamp, count, seconds))
            count = sum(row[1] for row in rates)
            active_seconds = sum(row[2] for row in rates)
            hint["worker_rate"] = None
            if count >= 3 and active_seconds >= 1:
                hint["worker_rate"] = {"per_second": count / active_seconds,
                                       "measured_at": wall - (stamp - rates[-1][0])}
        if phase == "model_initialization" and rate_phase == "training":
            rates.clear()
        # Before the first timing sample, a new worker's generation/training
        # phase already owns the estimate. Do not substitute API poll timing.
        # Throttle ordinary updates; always publish phase transitions promptly.
        key = ((phase,) if phase == "tts_generation" else
               tuple(record.get(field) for field in ("phase", "label", "total", "unit")))
        activity = record.get("activity") or {}
        key += tuple(activity.get(field) for field in ("phase", "total", "unit", "batch_files"))
        done = activity.get("done") if activity else record.get("done")
        reset = (key == previous_key and isinstance(done, (int, float))
                 and isinstance(previous_done, (int, float)) and done < previous_done)
        finished = record.get("total") is not None and record.get("done") == record["total"]
        if key == previous_key and previous_time is not None and stamp - previous_time < 1 and not finished and not reset:
            return
        previous_key, previous_time, previous_done = key, stamp, done
        try:
            write_json(directory / "progress.json", {**hint, "kind": "studio_task_progress_v1",
                "task_id": request["task_id"], "request_sha256": request_sha256,
                "stage": request.get("stage"), "pid": os.getpid(),
                "creation_identity": owner.get("creation_identity"), "updated_at": now()})
        except OSError:
            # Retry with a fresh hint when the worker reports again. This
            # optional write does not determine whether the task succeeded.
            return
    return report


def execute(request_path, *, reference_review=False, reference_preparation=False, load_model=None):
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    from .studio import process_identity
    owner = {"task_id": request["task_id"], "request_sha256": digest(request_path),
             "pid": os.getpid(), "creation_identity": process_identity(os.getpid()),
             "parent_pid": os.getppid(), "started_at": now()}
    prior_path = request_path.parent / "owner.json"
    if prior_path.exists():
        prior = read_json(prior_path)
        if prior.get("pid") != os.getpid() and prior.get("creation_identity") and process_identity(prior.get("pid")) == prior["creation_identity"]:
            raise RuntimeError("This stage already has a live Python worker")
    write_json(prior_path, owner)
    with writer_lock(request_path.parent):
        if reference_preparation:
            return _execute_reference_preparation(request_path, load_model=load_model)
        if reference_review:
            return _execute_reference_review(request_path, load_model=load_model)
        return _execute(request_path)


def _execute_reference_preparation(request_path, *, load_model=None):
    from .reference_candidates import prepare_initial_candidate_batches
    from .studio import _implementation

    request = read_json(request_path)
    result = {"task_id": request["task_id"], "request_sha256": digest(request_path),
              "config_sha256": request["config_sha256"]}
    destination = request_path.parent / "result.json"
    if destination.exists():
        prior = read_json(destination)
        if any(prior.get(key) != value for key, value in result.items()):
            raise ValueError("기준 목소리 준비 결과가 실행 요청과 다릅니다.")
        return int(bool(prior.get("error")))
    try:
        if (request.get("kind") != "reference_initial_candidate_preparation_v1"
                or _implementation() != request["implementation_id"]
                or digest(request["project"]) != request["config_sha256"]):
            raise ValueError("후보 준비 실행 코드 또는 모델 설정이 변경되었습니다.")
        root = Path(request["project"]).parent.resolve()
        _prepare_selected_torch(root, os.environ.get("KOWW_STUDIO_TORCH_BACKEND"))
        if Path(request["pause_path"]).resolve() != request_path.parent / "pause.json":
            raise ValueError("후보 준비 일시정지 요청의 저장 위치가 다릅니다.")
        ledger_path = root / ".studio/reference-candidates/state.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else {}
        previous = ledger.get("job") or {}
        if previous.get("task_id") and previous.get("request_sha256"):
            old_path = root / ".studio/reference-candidates/jobs" / previous["task_id"] / "request.json"
            if digest(old_path) != previous["request_sha256"]:
                raise ValueError("재개할 기존 후보 요청이 변경되었습니다.")
            old = read_json(old_path)
            if old.get("project") != request["project"] or old.get("config_sha256") != request["config_sha256"]:
                raise ValueError("재개할 기존 후보 요청의 모델 설정이 다릅니다.")
            if old.get("implementation_id") != request["implementation_id"]:
                receipts = [read_json(bound_file(root, binding)) for binding in request.get("migrations", [])]
                reached = {old.get("implementation_id")}
                for _ in receipts:
                    for receipt in receipts:
                        if receipt.get("old_implementation_id") in reached:
                            reached.add(receipt.get("new_implementation_id"))
                if request["implementation_id"] not in reached:
                    raise ValueError("구버전 후보 계획을 재개할 명시적 이관 근거가 없습니다.")
                envelope = {"kind": "audited_reference_preparation_resume_v1",
                    "old_request": {"path": old_path.relative_to(root).as_posix(), "sha256": digest(old_path)},
                    "old_implementation_id": old["implementation_id"], "new_implementation_id": request["implementation_id"],
                    "new_request_sha256": digest(request_path), "plan_id": identity(old["plan"]),
                    "migration_receipts": request["migrations"]}
                envelope_path = request_path.parent / "audited_resume.json"
                if envelope_path.exists() and read_json(envelope_path) != envelope:
                    raise ValueError("후보 준비 재개 연결 기록이 변경되었습니다.")
                write_json(envelope_path, envelope)
        model_cache = {}
        def resident_model(model_path, config, root):
            from .tts import _load_model
            key = (str(root), model_path, config["device"], config["attention"],
                   config.get("model_revisions", {}).get(model_path))
            if key not in model_cache:
                model_cache[key] = (load_model or _load_model)(model_path, config, root)
            return model_cache[key]
        result["result"] = prepare_initial_candidate_batches(request["project"], request["originals"],
            load_model=resident_model, progress_path=request_path.parent / "progress.json", pause_path=request["pause_path"])
        if digest(request["project"]) != request["config_sha256"]:
            raise ValueError("후보 준비 중 모델 설정이 변경되었습니다.")
    except Exception as error:
        traceback.print_exc()
        result["error"] = str(error) if isinstance(error, ValueError) else "기준 목소리 후보 준비를 완료하지 못했습니다. 작업 로그를 확인하세요."
    write_json(destination, result)
    return int(bool(result.get("error")))


def _execute_reference_review(request_path, *, load_model=None):
    from .reference_voice_review import generate_replacements
    from .studio import _implementation

    request = read_json(request_path)
    destination = request_path.parent / "result.json"
    result = {"task_id": request["task_id"], "request_sha256": digest(request_path)}
    if destination.exists():
        prior = read_json(destination)
        if any(prior.get(key) != value for key, value in result.items()):
            raise ValueError("기준 목소리 다시 생성 기록이 실행 요청과 다릅니다.")
        return int(bool(prior.get("error")))
    try:
        if _implementation() != request["implementation_id"] or digest(request["project"]) != request["config_sha256"]:
            raise ValueError("다시 생성하기 전에 실행 코드 또는 모델 설정이 바뀌었습니다.")
        _prepare_selected_torch(Path(request["project"]).parent, os.environ.get("KOWW_STUDIO_TORCH_BACKEND"))
        options = {"load_model": load_model} if load_model is not None else {}
        if request.get("kind") == "reference_candidate_batch_v2":
            from .reference_candidates import generate_candidate_batch
            result["candidates"] = generate_candidate_batch(request["project"], request_path.parent / "audio",
                request["task_id"], request["plan"], request["all_hashes"], **options)
        else:
            result["replacements"] = generate_replacements(request["project"], request_path.parent / "audio",
                request["task_id"], request["rows"], request["all_hashes"], **options)
        if digest(request["project"]) != request["config_sha256"]:
            raise ValueError("다시 생성하는 동안 모델 설정이 바뀌었습니다.")
    except Exception as error:
        traceback.print_exc()
        result["error"] = str(error) if isinstance(error, ValueError) else "기준 목소리를 다시 생성하지 못했습니다. 작업 로그를 확인하세요."
    write_json(destination, result)
    return int(bool(result.get("error")))


def _serve_reference_reviews():
    """Handle one request path per line until the manager closes the pipe."""
    model = model_key = None

    def load_model(model_path, config, root):
        from .tts import _load_model

        nonlocal model, model_key
        key = (root, model_path, config["device"], config["attention"],
               config.get("model_revisions", {}).get(model_path))
        if model is None or model_key != key:
            model = None
            model = _load_model(model_path, config, root)
            model_key = key
        return model

    for line in sys.stdin.buffer:
        execute(json.loads(line), reference_review=True, load_model=load_model)
    return 0


def _execute(request_path):
    request = read_json(request_path)
    request_hash = digest(request_path)
    destination = Path(request["result_path"]).resolve()
    if destination.parent != request_path.parent or destination.name != "result.json":
        raise ValueError("Worker results must stay beside their own request")
    if destination.exists():
        prior = read_json(destination)
        if prior.get("task_id") != request["task_id"] or prior.get("request_sha256") != request_hash:
            raise ValueError("Existing stage result belongs to another request; preserved")
        return 1 if prior.get("error") else 0
    result = {"task_id": request["task_id"], "request_sha256": request_hash,
              "config_sha256": request["config_sha256"], "started_at": now()}
    try:
        stage = request["stage"]
        before = load_project(request["project"])[1]
        if digest(request["project"]) != request["config_sha256"]:
            raise ValueError("Project settings changed before execution")
        if request.get("implementation_id"):
            from .studio import _implementation
            if _implementation() != request["implementation_id"]:
                raise ValueError("Library implementation changed before execution; previous artifacts preserved")
        from .studio_steps import run_stage
        if request["stage"] == "results":
            from .site import studio_result_audio
            result_audio = studio_result_audio(request["project"])
        else:
            result_audio = nullcontext()
        from .task_progress import task_progress
        report = _task_progress_writer(request, request_path.parent, request_hash)
        if request["stage"] == "training":
            from .train import training_progress
            preparation = training_progress(report)
        else:
            preparation = nullcontext()
        with storage_environment(Path(request["project"]).parent), result_audio, preparation, task_progress(report):
            if stage in _TORCH_STAGES:
                backend = os.environ.get("KOWW_STUDIO_TORCH_BACKEND")
                if stage == "training":
                    backend = _training_torch_backend(Path(request["project"]).parent, before, backend)
                _prepare_selected_torch(Path(request["project"]).parent, backend)
            if stage in ("hard_pilot", "hard_generation"):
                result["result"] = _hard_stage(request, request_path, run_stage)
            else:
                context = request.get("context", {})
                settings = request["settings"]
                if stage in ("frontend", "evaluation"):
                    settings = _stage_evaluation_settings(stage, settings, context.get("inputs", {}),
                                                          Path(request["project"]).parent)
                result["result"] = run_stage(request["project"], request["workspace"], settings, stage, context)
                if stage == "frontend" and result["result"]["state"] == "completed":
                    _verify_selected_torch(os.environ.get("KOWW_STUDIO_TORCH_BACKEND"))
        after = load_project(request["project"])[1]
        if after != before:
            if request["stage"] not in ("data_connect", "hard_connect"):
                raise ValueError("Stage changed the frozen project configuration")
            allowed = dict(before, manifests=after["manifests"], pool_features=after["pool_features"])
            if after != allowed:
                raise ValueError("Material connection changed training settings")
        result["result_config_sha256"] = digest(request["project"])
        result["artifact_bindings"] = [{"path": str(Path(p).resolve()), "sha256": digest(p)}
                                       for p in result["result"].get("artifacts", [])]
    except Exception as error:
        traceback.print_exc()
        result["error"] = {"message": str(error), "type": type(error).__name__,
                           "stage": request["stage"], "resolution": "아래 실행 로그를 확인하세요. 완료한 파일은 보존됩니다."}
        if request["stage"] in ("data_connect", "hard_connect") and "before" in locals():
            try:
                after = load_project(request["project"])[1]
                if after == dict(before, manifests=after["manifests"], pool_features=after["pool_features"]):
                    result["result_config_sha256"] = digest(request["project"])
            except (OSError, ValueError, KeyError):
                pass  # Never accept altered hyperparameters just to resume.
    result["completed_at"] = now()
    write_json(request["result_path"], result)
    return 1 if result.get("error") else 0


def _stage_evaluation_settings(stage, settings, inputs, root):
    from .studio import _studio_evaluation_backend

    backend = _studio_evaluation_backend(settings, inputs)
    effective = {**settings, "evaluation_backend": backend}
    if backend == "cpu":
        effective.pop("gpu_cpu_prefix_import", None)
    elif stage == "evaluation" and not os.environ.get("KOWW_GPU_RUNTIME_PATH"):
        from .gpu_runtime_install import _managed_reason, gpu_runtime_environment
        from .studio_steps import _prepare_evaluation_runtime
        from .task_progress import emit

        if _managed_reason() is None and not gpu_runtime_environment(root):
            emit("evaluation_environment", "GPU 평가 실행환경 설치·확인", eta_eligible=False)
            _prepare_evaluation_runtime(root)
    if inputs.get("evaluation_backend") == "cuda":
        effective.setdefault("gpu_numerical_differences_approved", True)
    return effective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument("--request")
    request.add_argument("--reference-review")
    request.add_argument("--reference-preparation")
    request.add_argument("--reference-review-worker", action="store_true")
    args = parser.parse_args()
    if args.reference_review_worker:
        raise SystemExit(_serve_reference_reviews())
    raise SystemExit(execute(args.request or args.reference_review or args.reference_preparation,
                            reference_review=bool(args.reference_review), reference_preparation=bool(args.reference_preparation)))
