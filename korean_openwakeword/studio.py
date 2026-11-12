"""Local training workspace: persistent orchestration around the public library.

The HTTP process never runs TTS/training itself. A single identified subprocess
executes one stage, publishes its result atomically, and can outlive the page.
Human answers remain real, hash-bound review evidence, never a job status flag.
"""
from datetime import datetime
from array import array
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
import ast
import hashlib
import hmac
import json
import math
import mimetypes
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
from sysconfig import get_platform
import threading
import time
import uuid

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, snapshot_json, write_json
from .hard_pilot import HARD_SAMPLE_REVIEW_POLICY
from .pilot import COMBINED_PILOT_POLICY, POSTGEN_PILOT_POLICY, PREPOST_PILOT_POLICY, TRAIN_PILOT_POLICY
from .project import create_project, load_project
from .web_page import render_page_config


_NAVIGATION_START = (
    ("downloads", ("frontend", "data_download", "data_review", "data_connect")),
    ("voices", ("references", "references_review")),
)
_NAVIGATION_END = (
    ("features", ("features",)),
    ("training", ("training", "package")),
    ("evaluation", ("development", "evaluation")),
    ("results", ("results",)),
)
_LEGACY_TRAINING_PREPARATION_PHASES = {
    "feature_validation", "device_selection", "model_initialization", "gpu_cache_loading",
}
_TRAINING_PREPARATION_PHASES = _LEGACY_TRAINING_PREPARATION_PHASES | {
    "feature_cache_verification", "training_finalizing",
}

_REFERENCE_PREVIEW_TRAIT_LABELS = {
    "gender": {"man": "남성", "woman": "여성"},
    "age_group": {"child": "어린이", "teenager": "청소년", "young adult": "청년",
                  "middle-aged": "중년", "older adult": "노년", "adult": "성인(연령 미지정)"},
    "timbre": {"clear light": "맑고 가벼운", "warm rounded": "따뜻하고 둥근", "low resonant": "낮고 울림 있는",
               "bright nasal": "밝고 비음 섞인", "slightly husky": "약간 허스키한", "soft airy": "부드럽고 공기 섞인",
               "crisp focused": "또렷하고 선명한", "deep mellow": "깊고 부드러운", "smooth balanced": "매끄럽고 균형 잡힌",
               "rich textured": "질감이 풍부한", "gentle thin": "온화하고 얇은", "firm dry": "단단하고 건조한",
               "light breathy": "가볍고 숨결 섞인", "dense midrange": "중음이 촘촘한", "clear ringing": "맑고 울림 있는"},
    "style": {"neutral": "담담하게", "friendly": "친근하게", "urgent": "다급하게", "projected": "또렷하게"},
    "vocal_effort": {"soft": "부드럽게", "normal": "보통", "strong": "강하게"},
    "speed": {"slow": "느리게", "natural": "자연스럽게", "quick": "빠르게"},
    "intonation": {"level": "평탄하게", "rising": "올리는 억양", "falling": "내리는 억양", "natural": "자연스러운 억양"},
}


def process_identity(pid):
    """Creation identity, not just PID existence (PIDs may be recycled)."""
    if type(pid) is not int or pid < 1:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            code = wintypes.DWORD()
            if (not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259
                    or not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                                 ctypes.byref(kernel_time), ctypes.byref(user_time))):
                return None
            return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        # macOS/BSD have no /proc. lstart is stable for a given process lifetime.
        try:
            result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True,
                                    text=True, timeout=2, check=False)
            return result.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None


def _safe_root(path):
    path = Path(path).expanduser().resolve()
    if any(part.lower() in (".codex", ".git") for part in path.parts):
        raise ValueError("학습 저장 폴더로 .codex 또는 .git 내부를 사용할 수 없습니다.")
    return path


def _model_choice(word, preset, settings):
    """Match runs using the same evaluation and pronunciation review policies."""
    return {"wakeword": word, "preset": preset,
            "evaluation_policy": (settings.get("evaluation_policy") or {}).get("id"),
            "voice_design_policy": settings.get("voice_design_policy"),
            "pilot_review_policy": settings.get("pilot_review_policy"),
            "hard_review_policy": settings.get("hard_review_policy"),
            **{key: settings[key] for key in ("originals", "rounds", "updates")},
            **{key: float(settings[key]) for key in ("generic_hours", "korean_hours")}}


def _dependency_package_requirements():
    """Use the worker's package requirements without probing installed packages."""
    from .gpu_runtime_install import PINS
    from .studio_steps import _dependency_plan

    evaluation_runtime = [f"{name}=={version}" for name, version in PINS.items()]
    packages = {
        backend: [argument for command in _dependency_plan({}, backend) for argument in _dependency_specs(command)]
        for backend in ("cpu", "cu128")
    }
    packages["cu128"].extend(evaluation_runtime)
    return packages, evaluation_runtime


def _dependency_specs(arguments):
    return [argument for argument in arguments if re.match(r"^[\w.-]+(?:==|>=)", argument)]


def _download_setup_rows(backend, evaluation_runtime, phase, *, complete=False, log=""):
    """Project the install plan using worker phases and its existing pip log.

    Packages in a pip invocation become complete when the worker reaches the
    next invocation or phase. Wheel names refine the active item within the
    invocation, without treating pip's console output as a completion receipt.
    """
    from .studio_steps import _dependency_plan

    def row(identifier, label):
        return {"id": identifier, "label": label, "state": "waiting"}

    def package_rows(specs):
        return [row(re.split(r"==|>=", spec, maxsplit=1)[0], spec) for spec in specs]

    commands = _dependency_plan({}, backend) if backend in ("cpu", "cu128") else []
    packages = [("environment_install", package_rows(_dependency_specs(command))) for command in commands]
    runtime = []
    if evaluation_runtime:
        runtime_rows = ([row("evaluation-runtime", "GPU 평가 실행 환경 확인")]
                        if os.environ.get("KOWW_GPU_RUNTIME_PATH") else package_rows(evaluation_runtime))
        runtime = [("evaluation_environment", runtime_rows)]
    batches = [("environment_check", [row("environment-check", "학습·평가 실행 환경 확인")]), *packages]
    if backend == "cu128":
        batches.extend(runtime)
    batches.append(("environment_onnx", [row("environment-onnx", "모델 내보내기 환경 확인")]))
    if backend != "cu128":
        batches.extend(runtime)
    batches.append(("frontend_assets", [row("frontend-assets", "음성 특징 추출 모델")]))

    # The GPU installer runs inside environment_install when training uses CUDA.
    # Later worker phases identify their item directly.
    active_phase, required, wheel, filename = phase, set(), "", ""
    if phase in ("environment_install", "evaluation_environment"):
        for line in log.splitlines():
            if line.startswith("필요한 학습 패키지 설치: "):
                if phase == "environment_install":
                    active_phase = "environment_install"
                    required = set(_dependency_specs(line.partition(": ")[2].split()))
                wheel = filename = ""
            elif line.startswith("별도 GPU 실행 환경을 프로젝트 캐시에 설치합니다."):
                active_phase, required = "evaluation_environment", set()
                wheel = filename = ""
            elif line.startswith(("Installing collected packages:", "Successfully installed ")):
                wheel = filename = ""
            elif match := re.match(r"^\s*(?:Downloading|Using cached) (\S+)", line):
                name = unquote(match[1].partition("?")[0].rsplit("/", 1)[-1])
                if name.lower().endswith(".whl"):
                    wheel = name.split("-", 1)[0].lower().replace("_", "-")
                    filename = name

    active = next((index for index, (batch_phase, rows) in enumerate(batches)
                   if batch_phase == active_phase and (not required or any(r["label"] in required for r in rows))), None)
    if active is None and phase == "environment_install":
        active = 0
    for index, (_, rows) in enumerate(batches):
        if complete or active is not None and index < active:
            for item in rows:
                item["state"] = "complete"
        elif index == active:
            tracked_wheel = any(item["id"] == wheel for item in rows)
            for item in rows:
                if required and item["label"] not in required:
                    item["state"] = "complete"
                elif not tracked_wheel or item["id"] == wheel:
                    item["state"] = "installing"
                    if tracked_wheel:
                        item["filename"] = filename
    return [item for _, rows in batches for item in rows]


# Content-Length of the pinned win_amd64 / cp312 wheels from download.pytorch.org.
_WINDOWS_CP312_TORCH_WHEEL_BYTES = {
    "torch==2.8.0+cpu": 619_352_539,
    "torchaudio==2.8.0+cpu": 2_495_725,
    "torch==2.8.0+cu128": 3_461_384_651,
    "torchaudio==2.8.0+cu128": 4_673_203,
}


def _dependency_download_bytes(packages):
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12) or get_platform() != "win-amd64":
        return None
    specs = [spec for spec in packages if spec.startswith(("torch==", "torchaudio=="))]
    if len(specs) != 2 or any(spec not in _WINDOWS_CP312_TORCH_WHEEL_BYTES for spec in specs):
        return None
    return sum(_WINDOWS_CP312_TORCH_WHEEL_BYTES[spec] for spec in specs)


def _dependency_setup_options(recommended_backend):
    packages, evaluation_runtime = _dependency_package_requirements()
    return {"recommended_backend": "cu128" if recommended_backend == "cu128" else "cpu",
            "packages": packages, "evaluation_runtime_packages": evaluation_runtime,
            "download_bytes": {backend: _dependency_download_bytes(specs) for backend, specs in packages.items()}}


def _elapsed(start):
    try:
        return max(0, time.time() - datetime.fromisoformat(start).timestamp())
    except (ValueError, TypeError):
        return 0


def _lock_paths(root):
    """Known writer directories only; don't walk hundreds of thousands of WAVs."""
    roots = (root / "features", root / "runs", root / "data", root / ".studio")
    patterns = (".writer.lock", "*/.writer.lock", "*/*/.writer.lock", "*/*/*/.writer.lock")
    return [path for base in roots if base.exists() for pattern in patterns for path in base.glob(pattern)]


def _implementation():
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) + sorted((root / "resources").glob("*.json"))
    def source_digest(path):
        # These are small source/config files. Keep checking every byte on
        # every call without allocating the artifact hasher's 8 MiB reads.
        value = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(128 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    return identity({path.relative_to(root).as_posix(): source_digest(path) for path in paths})


_EXECUTION_CONTRACT_CACHE = {}


def _execution_contract(*, legacy_reference_review=False):
    """Conservative computation fingerprint, separate from display/orchestration.

    A UI fix may change the worker implementation without changing any input,
    generation, training or scoring code. A changed computational file remains
    incompatible: a maintainer must provide an explicit audited migration.
    """
    root = Path(__file__).parent
    presentation = {"__init__.py", "__main__.py", "cli.py", "demo.py", "microphone.py",
                    "paged_site.py", "site.py", "storage_budget.py", "studio.py", "studio_worker.py", "studio_analysis.py", "web_page.py"}
    paths = [p for p in sorted(root.glob("*.py")) if p.name not in presentation]
    paths += sorted((root / "resources").glob("*.json"))
    bindings = {p.relative_to(root).as_posix(): digest(p) for p in paths}
    if bindings.get("reference_voice_review.py") == _REFERENCE_REVIEW_REUSABLE_LOADER_SHA256:
        # This exact version only allows the worker to reuse an already loaded model.
        bindings["reference_voice_review.py"] = _REFERENCE_REVIEW_COMPATIBLE_FILES[1]
    if legacy_reference_review:
        # Only project the previous contract for this audited, opt-in change.
        if (bindings.get("pilot.py"), bindings.get("reference_voice_review.py")) != _REFERENCE_REVIEW_COMPATIBLE_FILES:
            return None
        bindings["pilot.py"] = "5a1bf0206af409b7ec3b81ba4cc8b0865bbb9e49d78dfb909dce44cfa915ea84"
        bindings.pop("reference_voice_review.py")
    # Recheck every source byte, but reuse the expensive AST dependency graph
    # only for this exact source tree. Callers receive independent dictionaries.
    cache_key = (str(root.resolve()), digest(__file__), identity(bindings))
    if cache_key in _EXECUTION_CONTRACT_CACHE:
        return deepcopy(_EXECUTION_CONTRACT_CACHE[cache_key])
    # Stage-only modules cannot affect an already finished stage. Unknown new
    # modules remain shared, so adding a dependency never silently bypasses the
    # compatibility gate.
    groups = {
        "data": {"automatic_sources.py", "canonical_sources.py", "commercial_sources.py", "compact_sources.py",
                 "downloads.py", "download_control.py", "evaluation_materials.py", "fma_download.py", "fma_selection.py", "fma_streaming.py",
                 "material_downloads.py", "material_policy.py", "provider_index.py", "range_zip.py", "source_waveform_index.py",
                 "studio_data.py", "studio_evaluation_data.py", "pool_review.py"},
        "tts": {"reference_bank.py", "reference_qc.py", "reference_voice_review.py", "reference_candidates.py", "clone_reference.py", "call_generation_history.py", "pilot.py", "hard_tts.py", "tts.py", "tts_runtime.py", "tts_cuda_graph.py",
                "tts_cuda_predictor.py", "tts_cuda_sampling.py"},
        "hard_pilot": {"hard_pilot.py"},
        "postgeneration_repair": {"postgeneration_repair.py"},
        "features": {"features.py", "paired.py", "preparation.py"},
        "training": {"train.py"},
        "models": {"bundle.py", "final_integrity.py", "runtime.py", "events.py"},
        "results": {"selection.py", "cache_analysis.py", "calibration.py"},
        "evaluation": {"evaluate.py", "evaluation_backend.py", "evaluation_coverage.py", "evaluation_rir.py",
                       "evaluation_workers.py", "gpu_evaluation.py", "negative_streams.py", "tune.py", "calibration.py"}}
    stage_groups = {
        "frontend": set(), "data_download": {"data"}, "data_review": {"data"}, "data_connect": {"data"},
        "references": {"tts"}, "references_review": {"tts"}, "pilot": {"tts"}, "pilot_review": {"tts"},
        "positives": {"tts"}, "positive_review": {"tts", "postgeneration_repair"}, "hard_phrases": {"tts"}, "hard_pilot": {"tts", "hard_pilot"},
        "hard_generation": {"tts", "hard_pilot"}, "hard_review": {"tts", "data", "hard_pilot"},
        "hard_connect": {"tts", "data", "hard_pilot"}, "features": {"features"}, "training": {"training"},
        "package": {"models"}, "development": {"data", "evaluation", "models"},
        "evaluation": {"evaluation", "models"}, "results": {"results", "evaluation", "models"}}
    tree = ast.parse((root / "studio_steps.py").read_text(encoding="utf-8-sig"))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_stage")
    branches = {}
    shared = []
    for node in run.body:
        test = node.test if isinstance(node, ast.If) else None
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "stage_id"
                and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
                and isinstance(test.comparators[0].value, str) and isinstance(test.ops[0], ast.Eq)):
            branches[test.comparators[0].value] = node
        else:
            shared.append(node)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name not in ("run_stage", "preset_settings", "workflow_plan", "common_evaluation_policy")]
    classified = set().union(*groups.values())
    dependencies = {}
    for path in paths:
        if path.suffix != ".py" or path.name == "studio_steps.py":
            continue
        parsed = ast.parse(path.read_text(encoding="utf-8-sig"))
        dependencies[path.name] = {node.module.split(".")[0] + ".py" for node in ast.walk(parsed)
                                   if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}
    stages = {}
    for stage, needed_groups in stage_groups.items():
        needed = set().union(*(groups[group] for group in needed_groups)) if needed_groups else set()
        stage_files = {name: value for name, value in bindings.items()
                       if name != "studio_steps.py" and (name not in classified or name in needed)}
        pending = list(stage_files)
        while pending:
            for dependency in dependencies.get(pending.pop(), ()):
                if dependency in bindings and dependency not in stage_files and dependency != "studio_steps.py":
                    stage_files[dependency] = bindings[dependency]
                    pending.append(dependency)
        code = ast.dump(ast.Module(body=shared + helpers + [branches[stage]], type_ignores=[]), include_attributes=False)
        stages[stage] = identity({"files": stage_files, "stage_code": code})
    result = {"schema_version": 1, "computation_sha256": identity(bindings), "files": bindings, "stages": stages}
    if len(_EXECUTION_CONTRACT_CACHE) >= 4:
        _EXECUTION_CONTRACT_CACHE.clear()
    _EXECUTION_CONTRACT_CACHE[cache_key] = result
    return deepcopy(result)


_REFERENCE_REVIEW_COMPATIBLE_FILES = (
    "2b982ab0f2728a5af5d7f6d7ca7a897ef62a14be6f715693769d3c2f28b0f804",
    "5b5315240fd0ee91abf6d48a23d592b291c25d9bf475114b57b4aa2db03d4247",
)
_REFERENCE_REVIEW_REUSABLE_LOADER_SHA256 = "ca1554f1a5a797024c83e0d1d98bc2a44d1c363940c4eb954877c4adfc462ab3"
_REFERENCE_WORKER_IDLE_SECONDS = 120

# Populated only after testing the exact candidate-pool release. This cannot
# authorize a general computational-code rebind or alter a completed model.
_AUDITED_REFERENCE_CANDIDATE_UPDATES = {
    "bc713df47d2af09fa60490aec0b5fcbb489b9a2bbc632b37cdbf05921d645ddb": {
        "source_stages_sha256": "b0ce021ffccfd3465a137b31fb70c1d6c5c00a8bcbf46f259b6c4e3c62bd1138",
        "target_computation_sha256": "026185d1369e2614bf792da0572217cfc69949655ae864f95f011608353026a4",
        "target_stages_sha256": "d7131dabd26a5100f88d140c24d69603160219d594573cc5da189df7ed437240",
        "changed_files": ("reference_candidates.py", "reference_voice_review.py", "resources/voice_design_options.json"),
        "evidence_sha256": "3e1788fc9bc6700d376c8dbe3a99d5c4708d36bee0fbb07e711fe842ac2191f4",
    },
}
_AUDITED_REFERENCE_CANDIDATE_MERGE_UPDATES = {
    "026185d1369e2614bf792da0572217cfc69949655ae864f95f011608353026a4": {
        "source_stages_sha256": "d7131dabd26a5100f88d140c24d69603160219d594573cc5da189df7ed437240",
        "target_computation_sha256": "151d41977e7092dea0fe5f66cec05e3c7d02abfac2503a2b93528a3ad9559fad",
        "target_stages_sha256": "34a433d3e4525d895f7d8b42db11481d2634cf83b5eefd8195fcd631328a7414",
        "changed_files": ("pilot.py", "reference_candidates.py", "reference_voice_review.py", "studio_steps.py"),
        "evidence_sha256": "498d04bdfb1aae793d5402be6578db7127475833420f5f6f84dcc57a1e3e103e",
    },
}

# Exact release pins are added only after the preparation and preservation tests pass.
_AUDITED_REFERENCE_CANDIDATE_PREPARATION_UPDATES = {
    "151d41977e7092dea0fe5f66cec05e3c7d02abfac2503a2b93528a3ad9559fad": {
        "source_stages_sha256": "34a433d3e4525d895f7d8b42db11481d2634cf83b5eefd8195fcd631328a7414",
        "target_computation_sha256": "9badcb7d019ea1d04e7784ea177c82fbd5d613c6283a28ea9cb7985ef2bb9b6b",
        "target_stages_sha256": "26f7efbaaae71b689d10c53a1d4dd9c56919d8b44a43dd0bd17b1802e1301fb9",
        "changed_files": ("reference_candidates.py", "studio_steps.py"),
        "evidence_sha256": "9ae2f67b3bcee1de4286569a91b45647dfacfb044ad33bbd421b258c018c5f94",
    },
}


# Exact optional-listening releases are pinned after selection and compatibility checks.
_AUDITED_REFERENCE_OPTIONAL_LISTENING_UPDATES = {
    "9badcb7d019ea1d04e7784ea177c82fbd5d613c6283a28ea9cb7985ef2bb9b6b": {
        "source_stages_sha256": "26f7efbaaae71b689d10c53a1d4dd9c56919d8b44a43dd0bd17b1802e1301fb9",
        "target_computation_sha256": "37414060bd8360fe99e199d30beede76ab9be863aab56d57ff21f51895c7b44d",
        "target_stages_sha256": "5dd679355fcac55872fa253b63df9337db2fb90acf4d70c281bfbac6296e3810",
        "changed_files": ("reference_candidates.py",),
        "evidence_sha256": "0c160c9c62dca26a848773e4f199b43f98d460f0433185ef0d8c5e3e626831e8",
    },
    # Compose the already audited preparation update with the same review-only change.
    "151d41977e7092dea0fe5f66cec05e3c7d02abfac2503a2b93528a3ad9559fad": {
        "source_stages_sha256": "34a433d3e4525d895f7d8b42db11481d2634cf83b5eefd8195fcd631328a7414",
        "target_computation_sha256": "37414060bd8360fe99e199d30beede76ab9be863aab56d57ff21f51895c7b44d",
        "target_stages_sha256": "5dd679355fcac55872fa253b63df9337db2fb90acf4d70c281bfbac6296e3810",
        "changed_files": ("reference_candidates.py", "studio_steps.py"),
        "evidence_sha256": "0c160c9c62dca26a848773e4f199b43f98d460f0433185ef0d8c5e3e626831e8",
    },
}


def _audited_reference_candidate_update(previous, current):
    source = previous.get("computation_sha256")
    optional_listening = source in _AUDITED_REFERENCE_OPTIONAL_LISTENING_UPDATES
    preparation = source in _AUDITED_REFERENCE_CANDIDATE_PREPARATION_UPDATES
    merged = source in _AUDITED_REFERENCE_CANDIDATE_MERGE_UPDATES
    audit = (_AUDITED_REFERENCE_OPTIONAL_LISTENING_UPDATES if optional_listening else
             _AUDITED_REFERENCE_CANDIDATE_PREPARATION_UPDATES if preparation else
             _AUDITED_REFERENCE_CANDIDATE_MERGE_UPDATES if merged else
             _AUDITED_REFERENCE_CANDIDATE_UPDATES).get(source)
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current.get("computation_sha256") != audit.get("target_computation_sha256")
            or identity(previous.get("stages")) != audit.get("source_stages_sha256")
            or identity(current.get("stages")) != audit.get("target_stages_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("evidence_sha256", ""))):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    allowed = ({"reference_candidates.py", "studio_steps.py"} if preparation else
               {"reference_candidates.py"} if optional_listening else
               {"reference_candidates.py", "reference_voice_review.py", "resources/voice_design_options.json"})
    if merged:
        allowed.update({"pilot.py", "studio_steps.py"})
    if (changed != sorted(audit.get("changed_files", [])) or not changed or not set(changed).issubset(allowed)):
        return None
    return {"kind": "audited_reference_candidate_preparation_update" if preparation else
                    "audited_reference_optional_listening_update" if optional_listening else
                    "audited_reference_candidate_merge_update" if merged else "audited_reference_candidate_update", "changed_files": changed,
            "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": audit["evidence_sha256"]}


_AUDITED_CALL_GENERATION_RECOVERY_UPDATES = {
    "6db73e2aedb147470d1f386e2482bfaa6aa4d5af63f25ef314864e0b2ee75d2e": {
        "kind": "audited_postgeneration_voice_repair_update",
        "source_stages_sha256": "5c757fbab706cf6ebb0126efa5d1d2ae374debd67590daa502e6782b78401b18",
        "target_computation_sha256": "8ec72f66469ab1ebbc7a58202bc39e68d75d0d8dcae4a1fbdd9e16dc220b9012",
        "target_stages_sha256": "37e5396812e3c28715d2818a51fa120c17834bcbdd4ef7f3fd2400601ae29c34",
        "changed_files": ("postgeneration_repair.py", "studio_steps.py"),
        "evidence_sha256": "40b6e3e4d8d297c4d6f8715913e555de6b9cae3454d505ec8acdec3d5653e165",
        "reason": "기존 적합 음원·판정과 생성 기록을 보존하고, 본검수에서 부적합한 목소리의 호출어만 다시 생성해 재검수합니다.",
    },
    "6431480661a735e7d43424958721adc6b9a602850bb08f6433a5924263d2af7d": {'kind': 'audited_preflight_autosave_update', 'source_stages_sha256': 'de2168f88958a953def0c2857aa7aa5c9d19852c888832874fdef78b03b140af', 'target_computation_sha256': '6db73e2aedb147470d1f386e2482bfaa6aa4d5af63f25ef314864e0b2ee75d2e', 'target_stages_sha256': '5c757fbab706cf6ebb0126efa5d1d2ae374debd67590daa502e6782b78401b18', 'changed_files': ['call_generation_history.py', 'clone_reference.py', 'pilot.py', 'reference_candidates.py', 'review.py', 'tts.py'], 'evidence_sha256': '6cc3e376ef231ab2811951511e0dc7616e3fc20f32b080e8d59fccaa51be92e5', 'reason': '기존 목소리·샘플·검수 기록은 보존하고 사전 검수를 항목별 자동 저장으로 바꿉니다. 새 기준 목소리는 원본을 보존한 채 긴 앞뒤 무음만 정리합니다.'},
    "37414060bd8360fe99e199d30beede76ab9be863aab56d57ff21f51895c7b44d": {
        "source_stages_sha256": "5dd679355fcac55872fa253b63df9337db2fb90acf4d70c281bfbac6296e3810",
        "target_computation_sha256": "6431480661a735e7d43424958721adc6b9a602850bb08f6433a5924263d2af7d",
        "target_stages_sha256": "de2168f88958a953def0c2857aa7aa5c9d19852c888832874fdef78b03b140af",
        "changed_files": ("clone_reference.py", "pilot.py", "studio_steps.py", "tts.py"),
        "evidence_sha256": "3d98904afa79beb9b3ac8937834785e2b196e84b95af8180e5929f31b911ba60",
    },
}


def _audited_call_generation_recovery_update(previous, current):
    source = previous.get("computation_sha256")
    audit = _AUDITED_CALL_GENERATION_RECOVERY_UPDATES.get(source)
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != source or identity(new_files) != current.get("computation_sha256")
            or identity(previous.get("stages", {})) != audit["source_stages_sha256"]
            or current.get("computation_sha256") != audit["target_computation_sha256"]
            or identity(current.get("stages", {})) != audit["target_stages_sha256"]
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("evidence_sha256", ""))):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    if changed != sorted(audit["changed_files"]):
        return None
    return {"kind": audit.get("kind", "audited_call_generation_recovery_update"), "changed_files": changed,
            "source_computation_sha256": source,
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": audit["evidence_sha256"],
            **({"reason": audit["reason"]} if audit.get("reason") else {})}


# Exact 0.14.3 source identity and computational fingerprint, audited before
# adding migration support. Other snapshot-less versions are never guessed.
_LEGACY_COMPATIBLE_CONTRACTS = {
    "9d05b6d391a18920f28c964cf8f12c26faec7c5f98152688c73a5a499b4dd141":
        "28f88142c4ab88361199fd2fae2ec1fca1ba863412c206fb0cd5e6b5515def14"}


# An acquisition or fixed-public-source validation update still changes
# computational files. It requires an audited, exact source/target pair, not a
# blanket exception for download or admission code.
# Final release fingerprints and the preserved-input equivalence receipt are
# filled only after the complete source tree has been frozen and tested.
_AUDITED_ACQUISITION_UPDATES = {
    # Exact CRLF and LF pairs in validation/steady_download_gain_v1.json.
    "a89dd8f89ed2534935a153104333b79181d0367292c4e46bd086a41b5d4ae77e": {
        "source_stages_sha256": "3b723b31d0a8e004620e79f6abdb67424fa3c2b85cae36c82e33cef4a59ab941",
        "target_computation_sha256": "1a55d55a872bc02d08feb6709c24157c2f6ef8fd197502dd10e643469aed1db1",
        "target_stages_sha256": "05c07d33e0264ba7ef697e90dfc1a273c134efc065008a4bdbc65b44f14affb2",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "b8d1c9b40abb91dbbe1500a6ab922f769e08bafbff5e375e2260e4cee328e829": {
        "source_stages_sha256": "8589b8bc6d86d7e04e33bdefbbb47a1216265092d074befd234cd62d6cdcd734",
        "target_computation_sha256": "1a55d55a872bc02d08feb6709c24157c2f6ef8fd197502dd10e643469aed1db1",
        "target_stages_sha256": "05c07d33e0264ba7ef697e90dfc1a273c134efc065008a4bdbc65b44f14affb2",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "4c6fdddc4349ac998869657de3066a4c143a36369275583ae6dd4fe1fd8b6978": {
        "source_stages_sha256": "56910287134bf8f3688743b6f11c54f6bf3dc5c14b1778e84f3dd6357f79d4dd",
        "target_computation_sha256": "1a55d55a872bc02d08feb6709c24157c2f6ef8fd197502dd10e643469aed1db1",
        "target_stages_sha256": "05c07d33e0264ba7ef697e90dfc1a273c134efc065008a4bdbc65b44f14affb2",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "0002ac6e810f3656cd0ba4400a9f434b088e936c1f7f69ffb1311e251ca79a3e": {
        "source_stages_sha256": "ba58808e7fa0de597f6a97b8b60800272b403d0887259b6b63ddbacdc691dfdf",
        "target_computation_sha256": "1a55d55a872bc02d08feb6709c24157c2f6ef8fd197502dd10e643469aed1db1",
        "target_stages_sha256": "05c07d33e0264ba7ef697e90dfc1a273c134efc065008a4bdbc65b44f14affb2",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "1098e002acf28cfa8d863b0dd6701a1b3c366de19f3267ef7c3eaf380bef402f": {
        "source_stages_sha256": "5bb03ab00820d47d6d0bd4bd6cb20aa107c88533c0b543b567aefe672536a8c3",
        "target_computation_sha256": "1a55d55a872bc02d08feb6709c24157c2f6ef8fd197502dd10e643469aed1db1",
        "target_stages_sha256": "05c07d33e0264ba7ef697e90dfc1a273c134efc065008a4bdbc65b44f14affb2",
        "changed_files": ("downloads.py", "gpu_runtime_install.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "e93c1365da51c7777ab12fa9a6982418cc508df49bc95577c36c63c6a228af71": {
        "source_stages_sha256": "b7973889bfc2002f19fb8568daf72a2bc18c867ba297addb9f33965698b5f625",
        "target_computation_sha256": "fe8a0de0868bed40f830651a07146a5bbc58cc7b77c2f245c8b4ecb3ebafb78b",
        "target_stages_sha256": "aecacf85cf420f3c5e946b49b2670415576e52d1d3cbfbd23269400e61b45216",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "5adf2d4c54e21988777354b18cbaec0ed7b41426c7b397bd8cd8066061fe99da": {
        "source_stages_sha256": "35990c7529bad56c4ae9605f2fd3abc050b916dc271063b873f96575e90dad2e",
        "target_computation_sha256": "fe8a0de0868bed40f830651a07146a5bbc58cc7b77c2f245c8b4ecb3ebafb78b",
        "target_stages_sha256": "aecacf85cf420f3c5e946b49b2670415576e52d1d3cbfbd23269400e61b45216",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "688c877c6a38656dae27c34a209e8623c7a171277aba91a5973e7f65986b9183": {
        "source_stages_sha256": "7b464f3270916a7e65359731791f8d636514156b77ab6bd92b0b3e23ec12a720",
        "target_computation_sha256": "fe8a0de0868bed40f830651a07146a5bbc58cc7b77c2f245c8b4ecb3ebafb78b",
        "target_stages_sha256": "aecacf85cf420f3c5e946b49b2670415576e52d1d3cbfbd23269400e61b45216",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "cc89568eca719ab7a44cd752ea14a1d9a5929c2548b70c71b3158eb3d8e90e6e": {
        "source_stages_sha256": "f17e93b18f2858bb895d47dc58706373a8894d3919a132929accc472ffbd75ed",
        "target_computation_sha256": "fe8a0de0868bed40f830651a07146a5bbc58cc7b77c2f245c8b4ecb3ebafb78b",
        "target_stages_sha256": "aecacf85cf420f3c5e946b49b2670415576e52d1d3cbfbd23269400e61b45216",
        "changed_files": ("downloads.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    "7ce9423ef77c82e8517ec2218298ba4dac15d86bda010ea7ee8f84fbc306193d": {
        "source_stages_sha256": "9c174445612be55c8f1918b4f28d85e53cfe40c835a900fb4beae63937584c1d",
        "target_computation_sha256": "fe8a0de0868bed40f830651a07146a5bbc58cc7b77c2f245c8b4ecb3ebafb78b",
        "target_stages_sha256": "aecacf85cf420f3c5e946b49b2670415576e52d1d3cbfbd23269400e61b45216",
        "changed_files": ("downloads.py", "gpu_runtime_install.py", "material_downloads.py",),
        "evidence_sha256": "6cd0caafc69b0dad12a30b644c880a4e01a8e888c52dde1e96e893a98e2442c7",
    },
    # Exact non-material VoiceDesign option pin exclusion; all source evidence stays frozen.
    '8ec72f66469ab1ebbc7a58202bc39e68d75d0d8dcae4a1fbdd9e16dc220b9012': {'source_stages_sha256': '37e5396812e3c28715d2818a51fa120c17834bcbdd4ef7f3fd2400601ae29c34', 'target_computation_sha256': '1770934c2cf9b6e4e84edd532d1591991836f69604a7930cb6da6f3c199917f6', 'target_stages_sha256': '0e766b081f2651106fcb613fcbfd0208c4527a2974714e25068c6e25350e3a8d', 'changed_files': ['material_policy.py'], 'evidence_sha256': '807cb103d36fd144d7834ca803c0e1d3d3d8e89d941f3496723fa60ac54ef26f'},
    "04d4bd5dcfa9667ff2a650e4163007181fafe9b0cdd2e8cc72ad3e9526380d04": {
        "source_stages_sha256": "aef7b4226a131faa19df24a46220015e901d696f3c130fb67064114fbb2b38de",
        "target_computation_sha256": "9074dae60dd74e2439198457ecbfb639df5a418831a2d73937feae53ee22de70",
        "target_stages_sha256": "a01c4fcc065be2c5dd003f82fbe9c3d630a3e7a840d4c0da935c2551b51c702c",
        "changed_files": ('downloads.py', 'evaluation_materials.py', 'evaluation_rir.py', 'material_downloads.py', 'official_music.py', 'studio_data.py', 'studio_steps.py'),
        "evidence_sha256": "250073e4bae086eefaf11e1a00a57baf169fcbff8376a0dea0e4ff12196c818c",
    },
    # Exact 0.24.0 -> 0.24.1 original-owner validation repair.
    "41e7cbd1606ebbed420f5b07f971296207d263ed878acbab3d26a7b388f9edec": {'source_stages_sha256': 'f13dad4d451cefc5269ff1f1bad5e9f4a740bbd97cc7b25298b51c67fa5dca7e', 'target_computation_sha256': '190472c48c47978effa8a0360fd56287ca72c8c250e321619bbe00984f518ccb', 'target_stages_sha256': '24b984a10bd10d23663ef9e463286730fbd51ea0f658235b363e0a09b2cdf966', 'changed_files': ('material_policy.py',), 'evidence_sha256': '30765776c7006abc7630249662f36039bbf2f7080d520dc412d43b2a9e98e0b6'},
    # Exact GPU-required evaluation / Windows publication repair.
    "190472c48c47978effa8a0360fd56287ca72c8c250e321619bbe00984f518ccb": {'source_stages_sha256': '24b984a10bd10d23663ef9e463286730fbd51ea0f658235b363e0a09b2cdf966', 'target_computation_sha256': '93eb771702f0dce745787f0e1493d0150118ea652896d22521c649f08e03f0a1', 'target_stages_sha256': '8f4ca49e07e13b564543574f39649dd0ea725267152bb5fd017bcc35dffa82d3', 'changed_files': ('evaluation_backend.py', 'gpu_runtime_install.py', 'studio_steps.py'), 'evidence_sha256': '9010a49f2879a605120a816d6e6847ec97428da7439856ad3f2fe8278b10afd4'},
}
_ACQUISITION_UPDATE_FILES = {
    "downloads.py", "download_control.py", "material_downloads.py", "studio_data.py",
    "evaluation_materials.py", "evaluation_rir.py", "studio_evaluation_data.py", "official_music.py", "studio_steps.py",
    "material_policy.py", "evaluation_backend.py", "gpu_runtime_install.py",
}

# Only the audited reference-bank expansion can rebind this release. Existing
# generation code, reference waveforms and evaluation families stay unchanged.
_AUDITED_REFERENCE_UPDATES = {
    "9074dae60dd74e2439198457ecbfb639df5a418831a2d73937feae53ee22de70": {
        "source_stages_sha256": "a01c4fcc065be2c5dd003f82fbe9c3d630a3e7a840d4c0da935c2551b51c702c",
        "target_computation_sha256": "5c883156eb3b70a0ff3dffb0b469d6281add62884c9891f3aff0f43c5691b289",
        "target_stages_sha256": "a85ec45815ccc7ba85a2c9a82ab8960280bed3e8d77d36fb33ec9e3cb07e3abe",
        "evidence_sha256": "7592afc3d3c2d85512d7a7aed1b54ee12e2a96c2f91260b8d873c413c4bd4df3",
    },
}

# Reference admission changes approval provenance, never the generated audio or
# the mandatory pronunciation pilot. Only frozen, tested release pairs qualify.
_AUDITED_REFERENCE_ADMISSION_UPDATES = {
    "1df90740a1b663b970b81042f570ae7e614fe09ab723f4d6242e60e6743902d1": {
        "source_stages_sha256": "ccbc03ebd7c3c608bfa83184fbff9bc81001d2345047c473d78fc1c1e694e4e3",
        "target_computation_sha256": "744dedd32fdbeed761227caf17b89b10e679122130557c575075d1aa74e25d18",
        "target_stages_sha256": "e1dbe0d98f0cd2b765f8e4115f068a6d3cb7c1f28ed8ba59691020b2ca4e78b5",
        "changed_files": ("reference_bank.py", "reference_qc.py", "studio_steps.py"),
        # Canonical JSON identity of docs/reference_admission_0.21.0_audit.json.
        "evidence_sha256": "2aeb248eadb1a676619b940b1d765cb31f1325c7b0cce266c9897cdd44a49180",
    },
}


def _audited_reference_admission_update(previous, current):
    audit = _AUDITED_REFERENCE_ADMISSION_UPDATES.get(previous.get("computation_sha256"))
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (previous.get("schema_version") != 1 or current.get("schema_version") != 1
            or not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current.get("computation_sha256") != audit.get("target_computation_sha256")
            or identity(previous.get("stages")) != audit.get("source_stages_sha256")
            or identity(current.get("stages")) != audit.get("target_stages_sha256")):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    allowed = {"reference_qc.py", "reference_bank.py", "review.py", "studio_steps.py"}
    if (not changed or changed != sorted(audit.get("changed_files", ()))
            or not set(changed).issubset(allowed)
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("evidence_sha256", ""))):
        return None
    return {"kind": "audited_reference_admission_update", "changed_files": changed,
            "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": audit["evidence_sha256"]}


_AUDITED_PILOT_UPDATES = {
    "744dedd32fdbeed761227caf17b89b10e679122130557c575075d1aa74e25d18": {
        "source_stages_sha256": "e1dbe0d98f0cd2b765f8e4115f068a6d3cb7c1f28ed8ba59691020b2ca4e78b5",
        "target_computation_sha256": "194ed2bcfda035f24009c11d6a9b71f6cd79639f9a5e8a03361ca4466db4ac4c",
        "target_stages_sha256": "3a137e646251e373319242ac3c518999a06d8ab430fe5dff66c2e50516b7ec8a",
        # Canonical JSON identity of docs/pilot_one_per_voice_0.22.0_audit.json.
        "evidence_sha256": "720ef26f5f850ed9e35a8bd4d20f55deb2d22a97ba917e77c904caa1a4acd6d7",
    },
}


def _audited_pilot_update(previous, current):
    audit = _AUDITED_PILOT_UPDATES.get(previous.get("computation_sha256"))
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (previous.get("schema_version") != 1 or current.get("schema_version") != 1
            or not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current.get("computation_sha256") != audit.get("target_computation_sha256")
            or identity(previous.get("stages")) != audit.get("source_stages_sha256")
            or identity(current.get("stages")) != audit.get("target_stages_sha256")):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    if (changed != ["pilot.py", "studio_steps.py"]
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("evidence_sha256", ""))):
        return None
    return {"kind": "audited_one_call_per_training_voice", "changed_files": changed,
            "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": audit["evidence_sha256"]}


_AUDITED_HARD_PER_PHRASE_UPDATES = {
    "194ed2bcfda035f24009c11d6a9b71f6cd79639f9a5e8a03361ca4466db4ac4c": {
        "source_stages_sha256": "3a137e646251e373319242ac3c518999a06d8ab430fe5dff66c2e50516b7ec8a",
        "target_computation_sha256": "de8c59b13dd11d3f0688409c091c79016f27453a5d849d6b596bf31b6865758a",
        "target_stages_sha256": "e73d5b7a8280e34852bf33090ea1f2d69ff46d79470e1bd82549480d2f20c0a5",
        # Canonical JSON identity of docs/hard_per_phrase_0.23.0_audit.json.
        "evidence_sha256": "1a129b6eeebe366dce54b763acd4132c4e37c4cca125b059c12b01d509be0c81",
    },
}


def _audited_hard_per_phrase_update(previous, current):
    """Only a tested, pinned release may change an unstarted Hard budget."""
    audit = _AUDITED_HARD_PER_PHRASE_UPDATES.get(previous.get("computation_sha256"))
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (previous.get("schema_version") != 1 or current.get("schema_version") != 1
            or not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current.get("computation_sha256") != audit.get("target_computation_sha256")
            or identity(previous.get("stages")) != audit.get("source_stages_sha256")
            or identity(current.get("stages")) != audit.get("target_stages_sha256")):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    if (changed != ["hard_tts.py", "pool_review.py", "studio_steps.py"]
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("evidence_sha256", ""))):
        return None
    return {"kind": "audited_hard_per_phrase_update", "changed_files": changed,
            "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": audit["evidence_sha256"]}


_AUDITED_HARD_REVIEW_UPDATES = {
    "de8c59b13dd11d3f0688409c091c79016f27453a5d849d6b596bf31b6865758a": {
        "source_stages_sha256": "e73d5b7a8280e34852bf33090ea1f2d69ff46d79470e1bd82549480d2f20c0a5",
        "target_computation_sha256": "41e7cbd1606ebbed420f5b07f971296207d263ed878acbab3d26a7b388f9edec",
        "target_stages_sha256": "f13dad4d451cefc5269ff1f1bad5e9f4a740bbd97cc7b25298b51c67fa5dca7e",
        "evidence_sha256": "e0227e6cb109e7479b9aa10f4f721cbe1b072b4ee665ff2f3e6e896e647bfd8b",
    },
}


def _audited_hard_review_update(previous, current):
    """A release-bound review change cannot relax an unrelated computation."""
    audit = _AUDITED_HARD_REVIEW_UPDATES.get(previous.get("computation_sha256"))
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (previous.get("schema_version") != 1 or current.get("schema_version") != 1
            or not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current.get("computation_sha256") != audit.get("target_computation_sha256")
            or identity(previous.get("stages")) != audit.get("source_stages_sha256")
            or identity(current.get("stages")) != audit.get("target_stages_sha256")):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    if (changed != ["pool_review.py", "studio_steps.py"]
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("evidence_sha256", ""))):
        return None
    return {"kind": "audited_one_hard_representative_per_phrase", "changed_files": changed,
            "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": audit["evidence_sha256"]}


def _audited_reference_update(previous, current):
    audit = _AUDITED_REFERENCE_UPDATES.get(previous.get("computation_sha256"))
    if not isinstance(audit, dict):
        return None
    old_files, new_files = previous.get("files"), current.get("files")
    if (not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current.get("computation_sha256") != audit.get("target_computation_sha256")
            or identity(previous.get("stages")) != audit.get("source_stages_sha256")
            or identity(current.get("stages")) != audit.get("target_stages_sha256")):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys()
                     if old_files.get(name) != new_files.get(name))
    if changed != ["project.py", "studio_steps.py"]:
        return None
    evidence = audit.get("evidence_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", evidence):
        return None
    return {"kind": "audited_reference_budget_update", "changed_files": changed,
            "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"],
            "evidence_sha256": evidence}


def _audited_acquisition_update(previous, current):
    """Return a release-bound audit for an exact acquisition/admission diff."""
    audit = _AUDITED_ACQUISITION_UPDATES.get(previous.get("computation_sha256"))
    if not isinstance(audit, dict):
        return None
    hash_keys = ("source_stages_sha256", "target_computation_sha256", "target_stages_sha256", "evidence_sha256")
    if any(not isinstance(audit.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", audit[key]) for key in hash_keys):
        return None  # An unfinished release audit can never authorize a rebind.
    old_files, new_files = previous.get("files"), current.get("files")
    if (previous.get("schema_version") != 1 or current.get("schema_version") != 1
            or not isinstance(old_files, dict) or not isinstance(new_files, dict)
            or identity(old_files) != previous.get("computation_sha256")
            or identity(new_files) != current.get("computation_sha256")
            or current["computation_sha256"] != audit["target_computation_sha256"]
            or identity(previous.get("stages")) != audit["source_stages_sha256"]
            or identity(current.get("stages")) != audit["target_stages_sha256"]):
        return None
    changed = sorted(name for name in old_files.keys() | new_files.keys() if old_files.get(name) != new_files.get(name))
    approved = audit.get("changed_files")
    if (not isinstance(approved, (list, tuple)) or not approved or changed != sorted(approved)
            or not set(changed).issubset(_ACQUISITION_UPDATE_FILES)):
        return None  # Catalogs, pins, and every unlisted module must be intact.
    return {"kind": "audited_acquisition_update", "source_computation_sha256": previous["computation_sha256"],
            "target_computation_sha256": current["computation_sha256"], "changed_files": changed,
            "source_stages_sha256": audit["source_stages_sha256"], "target_stages_sha256": audit["target_stages_sha256"],
            "evidence_sha256": audit["evidence_sha256"]}


def _timestamp(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def _interval_seconds(intervals):
    """Union intervals: receipts and the current worker must never double count."""
    end, seconds = None, 0.0
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        seconds += stop - max(start, end if end is not None else start) if end is None or stop > end else 0
        end = max(end if end is not None else stop, stop)
    return seconds


def _job_log_tail(job, limit=24000):
    """Read a bounded tail from this task's own log, never another attempt."""
    if not job.get("task_id") or not job.get("log"):
        return ""
    path = Path(job["log"])
    directory = Path(job.get("directory", path.parent))
    if (directory.name != job["task_id"] or path.name != "output.log"
            or path.parent.resolve() != directory.resolve()):
        return ""
    try:
        with path.open("rb") as handle:
            length = handle.seek(0, 2)
            handle.seek(max(0, length - limit))
            return handle.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _evaluation_log_progress(job):
    """Session counts only; polling never opens score caches or evaluation WAVs."""
    pattern = (r"(?m)^Evaluate (?:development|final): (\d+)/(\d+) "
               r"\((?:CPU workers=(\d+)|CUDA batch=(\d+), CPU rechecked=(\d+))\)\r?$")
    for match in reversed(list(re.finditer(pattern, _job_log_tail(job)))):
        done, total = int(match[1]), int(match[2])
        if not 0 <= done <= total or total == 0:
            continue
        cuda = match[4] is not None
        result = {"done": done, "total": total, "unit": "개 음원",
                  "label": "성능 평가 · GPU" if cuda else "성능 평가 · CPU",
                  "phase": "finalizing" if done == total else "scoring",
                  "source": "current_task_log", "task_id": job["task_id"],
                  "backend": "cuda" if cuda else "cpu"}
        if cuda:
            result.update(batch_size=int(match[4]), cpu_rechecked_sessions=int(match[5]))
        else:
            result["workers"] = int(match[3])
        return result
    return {"done": None, "total": None, "unit": "개 음원", "label": "평가 음원 검증 중",
            "phase": "validation", "source": "current_task_log", "task_id": job.get("task_id")}


@contextmanager
def workspace_lease(workspace):
    """Only one Studio server owns a workspace, even on different ports."""
    directory = _safe_root(workspace) / ".koww-studio"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "server.json"
    lock = (directory / "server.lock").open("a+b")
    if lock.seek(0, 2) == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock.close()
        raise RuntimeError("이 작업 폴더의 학습 화면이 이미 실행 중입니다. 기존 화면을 사용하세요.") from error
    owner = {"pid": os.getpid(), "creation_identity": process_identity(os.getpid()),
             "started_at": now(), "lease_id": uuid.uuid4().hex}
    try:
        if path.exists():
            write_json(directory / "previous_servers" / (uuid.uuid4().hex + ".json"), read_json(path))
        write_json(path, owner)
        yield
    finally:
        if path.exists() and read_json(path).get("lease_id") == owner["lease_id"]:
            path.unlink()
        lock.seek(0)
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


class StudioManager:
    def __init__(self, workspace, *, popen=None):
        self.workspace = _safe_root(workspace)
        self.directory = self.workspace / ".koww-studio"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.directory / "projects.json"
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._popen = popen or subprocess.Popen
        self._children = {}
        self._reference_worker = None
        self._implementation_id = _implementation()
        self._execution_contract = _execution_contract()
        self._pre_reference_review_contract = _execution_contract(legacy_reference_review=True)
        self._timing_receipts = {}
        self._microphones = {}
        self._journals = {}
        self._rates = {}
        self._progress_snapshots = {}
        self._history_metadata = {}
        self._positive_preview_indexes = {}
        self._history_download_plans = {}
        self._closing = threading.Event()
        self._thread = None
        self.registry = read_json(self.registry_path) if self.registry_path.exists() else {"projects": {}, "active_project_id": None}
        self._recover()

    def _path(self, project_id):
        if project_id not in self.registry["projects"]:
            raise ValueError("등록되지 않은 모델 학습입니다.")
        project = _safe_root(self.registry["projects"][project_id])
        return project.parent / ".studio" / "workflow.json"

    def _load(self, project_id):
        state = read_json(self._path(project_id))
        if state.get("id") != project_id or state.get("workspace") != str(self.workspace):
            raise ValueError("저장된 모델 학습의 실행 근거가 일치하지 않습니다.")
        return state

    def _save(self, state):
        stamp = now()
        # Record observed status boundaries from this version onward. Unknown
        # legacy/offline intervals stay unclassified instead of becoming CPU
        # time merely because a status file once said "running".
        timing = state.setdefault("timing_observations", {"schema_version": 1, "segments": []})
        segments = timing["segments"]
        if not segments or segments[-1]["status"] != state["status"]:
            if segments and "end" not in segments[-1]:
                segments[-1]["end"] = stamp
            segments.append({"status": state["status"], "start": stamp})
        if state["status"] == "completed":
            segments[-1]["end"] = stamp
            state.setdefault("completed_at", stamp)
        state["updated_at"] = stamp
        write_json(self._path(state["id"]), state)

    def _timing(self, state):
        start = _timestamp(state.get("started_at") or state.get("created_at"))
        end = time.time()
        complete = state.get("status") == "completed"
        if complete:
            finished = state.get("completed_at")
            if not finished:
                finished = next((entry.get("completed_at") for entry in reversed(state.get("history", []))
                                 if entry.get("completed_at")), state.get("updated_at"))
            end = _timestamp(finished) or start or end
        start = min(start if start is not None else end, end)
        intervals = []
        for entry in state.get("history", []):
            key = (entry.get("result"), entry.get("sha256"))
            if not all(key):
                continue
            if key not in self._timing_receipts:
                try:
                    result = read_json(key[0])
                    self._timing_receipts[key] = (_timestamp(result.get("started_at")),
                                                  _timestamp(result.get("completed_at")))
                except (OSError, ValueError, TypeError):
                    self._timing_receipts[key] = (None, None)
            left, right = self._timing_receipts[key]
            if left is not None and right is not None:
                intervals.append((max(start, left), min(end, right)))
        job = state.get("job") or {}
        if not complete and job and self._job_alive(job):
            left = _timestamp(job.get("started_at"))
            if left is not None:
                intervals.append((max(start, left), end))
        active = min(end - start, _interval_seconds(intervals))
        review_intervals, paused_intervals = [], []
        for segment in (state.get("timing_observations") or {}).get("segments", []):
            left = _timestamp(segment.get("start"))
            right = _timestamp(segment.get("end")) if segment.get("end") else end
            if left is None or right is None:
                continue
            target = review_intervals if segment.get("status") in ("waiting_review", "waiting_input") else paused_intervals
            if segment.get("status") in ("waiting_review", "waiting_input", "ready", "paused", "interrupted", "error"):
                target.append((max(start, left), min(end, right)))
        total = max(0, end - start)
        review = min(total - active, _interval_seconds(review_intervals))
        paused = min(total - active - review, _interval_seconds(paused_intervals))
        unknown = max(0, total - active - review - paused)
        return {"total_seconds": round(total, 1), "active_seconds": round(active, 1),
                "review_wait_seconds": round(review, 1), "paused_seconds": round(paused, 1),
                "unclassified_seconds": round(unknown, 1), "complete": complete,
                "active_scope": "verified_worker_lifetime", "legacy_estimate": unknown > 1}

    def _recover(self):
        for project_id in self.registry["projects"]:
            state = self._load(project_id)
            if state["status"] in ("running", "pausing"):
                job = state.get("job") or {}
                if not self._job_alive(job):
                    try:
                        if not self._finish_job(state):
                            state.update(status="interrupted", next_action="완료한 자료를 보존했습니다. 이어서 진행을 누르세요.")
                    except (OSError, ValueError, KeyError) as error:
                        state.update(status="error", auto_continue=False, error={"message": str(error), "type": type(error).__name__},
                                     next_action="이전 작업의 완료 근거를 확인해야 합니다. 원본은 보존했습니다.")
                    self._save(state)

    def recipes(self):
        from .studio_data import material_recipe
        from .studio_steps import _recommended_dependency_backend, preset_settings
        from .project import voice_candidates
        presets = []
        for key, label in (("check", "동작 확인용"), ("trial", "시험 모델용"), ("production", "상용 모델용")):
            settings = preset_settings(key)
            recipe = material_recipe(key, evaluation_policy=settings["evaluation_policy"],
                                     material_acquisition_policy=settings.get("material_acquisition_policy"),
                                     fma_acquisition_policy=settings.get("fma_acquisition_policy"),
                                     material_review_policy=settings.get("material_review_policy"),
                                     official_noise_acquisition_policy=settings.get("official_noise_acquisition_policy"))
            if key == "production":
                from .storage_budget import commercial_storage_budget
                recipe["storage_budget"] = commercial_storage_budget(
                    settings, base_archive_bytes=recipe["known_download_bytes"])
                recipe["estimated_workspace_bytes_scope"] = "legacy_partial_not_total"
            bank = voice_candidates(validation_count=settings["validation_references"],
                                    train_count=settings["train_references"],
                                    policy=settings["voice_design_policy"])
            presets.append({**settings, **recipe, "id": key, "label": label,
                            "reference_design": self._voice_design(bank, settings)})
        return {"presets": presets, "default_directory": str(self.workspace / "models"),
                "dependency_setup": _dependency_setup_options(_recommended_dependency_backend()),
                "storage_note": "호출어별 학습 자료와 결과를 models 폴더에 자동으로 저장합니다. 공통 자료와 캐시는 작업 폴더 안에서 공유합니다."}

    def _model_request(self, payload):
        """Validate creation and preview inputs without changing stored state."""
        from .studio_steps import preset_settings, workflow_plan
        if not isinstance(payload, dict) or not isinstance(payload.get("wakeword"), str):
            raise ValueError("호출어를 문자열로 입력하세요.")
        preset = payload.get("preset", "check")
        if not isinstance(preset, str):
            raise ValueError("작업 규모를 선택하세요.")
        settings = preset_settings(preset)
        if payload.get("reference_voice_review") is True:
            # New interactive projects choose candidate voices directly. Older
            # matched projects retain their frozen source-bank policy.
            settings["reference_selection_policy"] = "candidate_pool_v3"
            settings["reference_preparation_policy"] = "initial_candidate_batches_v1"
        overrides = payload.get("advanced") or {}
        if not isinstance(overrides, dict):
            raise ValueError("세부 설정 형식이 올바르지 않습니다.")
        if payload.get("directory") not in (None, ""):
            raise ValueError("모델 저장 폴더는 자동으로 정해집니다. 저장 경로를 입력하지 마세요.")
        mapping = {"positive_originals": "originals", "augmentation_rounds": "rounds", "updates": "updates",
                   "generic_hours": "generic_hours", "korean_hours": "korean_hours"}
        if set(overrides) - set(mapping):
            raise ValueError("지원하지 않는 세부 설정입니다.")
        for key, value in overrides.items():
            if value is not None and value != "":
                settings[mapping[key]] = value
        if type(settings["rounds"]) is not int:
            raise ValueError("증강 횟수는 정수 1 / 3 / 5로 입력하세요.")
        word = payload["wakeword"].strip()
        if not re.fullmatch(r"[가-힣]+", word):
            raise ValueError("호출어는 한글로 입력하세요. 경로나 특수 문자는 사용할 수 없습니다.")
        return word, preset, settings, workflow_plan(settings)

    def _model_location(self, word, preset, settings):
        """Read-only resolver shared by previews and actual model creation."""
        choice = _model_choice(word, preset, settings)
        # Old registered paths and their immutable settings remain in place,
        # including the Hangul folders made by 0.17.1 and projects/... paths.
        for known_id in self.registry["projects"]:
            saved = self._load(known_id)
            if _model_choice(saved["wakeword"], saved["preset"], saved["settings"]) == choice:
                target = Path(saved["project"]).parent
                if (target / "final_lock.json").exists() and saved["status"] != "completed":
                    raise ValueError("최종 평가로 잠긴 모델 학습입니다. 기존 자료와 평가 잠금을 유지합니다.")
                return target, saved
        models = self.workspace / "models"
        if _safe_root(models) != models or (models.exists() and not models.is_dir()):
            raise ValueError("models 저장 폴더가 다른 위치를 가리키거나 파일과 겹칩니다. 기존 자료를 확인하세요.")
        numbers = [int(match.group(1)) for entry in models.iterdir()
                   if (match := re.fullmatch(r"model_([0-9]+)", entry.name, re.IGNORECASE))] if models.exists() else []
        target = models / f"model_{max(numbers, default=0) + 1:04d}"
        if _safe_root(target) != target:
            raise ValueError("모델 저장 폴더가 다른 위치를 가리킵니다. 기존 자료를 확인하세요.")
        return target, None

    def model_location(self, payload):
        with self._lock:
            word, preset, settings, _ = self._model_request(payload)
            target, _ = self._model_location(word, preset, settings)
            return {"directory": str(target)}

    def create(self, payload):
        with self._lock:
            if isinstance(payload, dict):
                for key in ("download_checkpoint", "navigation_checkpoints", "reference_voice_review"):
                    if key in payload and type(payload[key]) is not bool:
                        raise ValueError("화면 진행 확인 설정은 참 또는 거짓으로 지정하세요.")
                if payload.get("download_checkpoint") and payload.get("navigation_checkpoints"):
                    raise ValueError("다운로드 전용 확인과 전체 단계 확인 중 하나만 선택하세요.")
                if payload.get("reference_voice_review") and not payload.get("navigation_checkpoints"):
                    raise ValueError("기준 목소리 검수에는 전체 단계 확인이 필요합니다.")
            word, preset, settings, plan = self._model_request(payload)
            dependency_backend = payload.get("dependency_backend")
            if "dependency_backend" in payload and dependency_backend not in ("cpu", "cu128"):
                raise ValueError("학습 도구는 CPU 또는 NVIDIA GPU를 선택하세요.")
            target, saved = self._model_location(word, preset, settings)
            if saved:
                self.registry["active_project_id"] = saved["id"]
                write_json(self.registry_path, self.registry)
                return self.describe(saved)
            # Claim a fresh folder exclusively: even an empty directory made
            # between resolution and creation belongs to someone else.
            target.mkdir(parents=True, exist_ok=False)
            project_id = uuid.uuid4().hex[:12]
            project = target / "project.json"
            create_project(target, word, "word_" + project_id,
                           positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                           updates=settings["updates"], generic_hours=settings["generic_hours"],
                           korean_hours=settings["korean_hours"],
                           validation_voice_families=settings["validation_references"])
            _, config = load_project(project)
            config["targets"]["validation_originals"] = settings["validation_originals"]
            config["audio_storage_policy"] = settings["audio_storage_policy"]
            if settings.get("fma_acquisition_policy"):
                config["fma_acquisition_policy"] = settings["fma_acquisition_policy"]
            if settings.get("official_noise_acquisition_policy"):
                config["official_noise_acquisition_policy"] = settings["official_noise_acquisition_policy"]
            write_json(project, config)
            state = {"schema_version": 1, "id": project_id, "workspace": str(self.workspace),
                     "storage_workspace": str(self.workspace),
                     "project": str(project), "wakeword": word, "preset": preset, "settings": settings,
                     "config_sha256": digest(project), "implementation_id": self._implementation_id,
                     "created_at": now(), "status": "ready", "index": 0,
                     "plan": plan, "results": {}, "inputs": {}, "job": None,
                     "error": None, "next_action": "시작하면 자료 준비부터 자동으로 진행합니다."}
            if payload.get("download_checkpoint") is True:
                # Web navigation is orchestration only: preserve the canonical
                # settings, computational plan and every worker request.
                state["download_checkpoint"] = {"policy": "explicit_download_v1", "state": "awaiting_start"}
                state["next_action"] = "자료 다운로드 시작을 누르세요."
            if payload.get("navigation_checkpoints") is True:
                self._navigation_bounds(state)
                state["navigation_checkpoint"] = {"policy": "explicit_story_v1", "step": "downloads", "state": "awaiting_start"}
                state["next_action"] = "자료 다운로드 시작을 누르세요."
            if dependency_backend is not None:
                state["inputs"]["dependency_backend"] = dependency_backend
                state["dependency_setup_selected"] = True
                state["next_action"] = "선택한 학습 도구 준비를 시작하세요."
            if payload.get("reference_voice_review") is True:
                state["reference_voice_review_policy"] = "candidate_pool_v2"
            state["execution_contract"] = {**self._execution_contract,
                "settings_sha256": identity(settings), "plan_sha256": identity(state["plan"])}
            selected_cache = os.environ.get("KOWW_CACHE_ROOT")
            state["cache_root"] = str(_safe_root(selected_cache) if selected_cache else Path(state["storage_workspace"]) / "cache")
            self.registry["projects"][project_id] = str(project)
            self.registry["active_project_id"] = project_id
            self._save(state)
            write_json(self.registry_path, self.registry)
            return self.describe(state)

    def _job_alive(self, job):
        child = self._children.get(job.get("task_id"))
        if child is not None and child.poll() is None:
            return True
        owner = self._worker_owner(job)
        return any(record.get("pid") and record.get("creation_identity") and
                   process_identity(record["pid"]) == record["creation_identity"] for record in (owner, job))

    def _worker_owner(self, job):
        path = Path(job["directory"]) / "owner.json" if job.get("directory") else None
        if not path or not path.exists():
            return {}
        owner = read_json(path)
        if owner.get("task_id") != job.get("task_id") or owner.get("request_sha256") != job.get("request_sha256"):
            raise ValueError("실제 작업 프로세스의 소유 기록이 실행 요청과 다릅니다.")
        return owner

    def _guard_config(self, state):
        # A running manager stamps requests with the source identity captured
        # at startup. Refuse before allocating work if files have since changed;
        # the strict worker gate must never receive a knowingly stale request.
        if _implementation() != self._implementation_id:
            raise ValueError("웹 서버 실행 코드가 업데이트되었습니다. 서버를 다시 시작한 뒤, 호환 업데이트가 표시되면 적용하고 이어서 진행하세요. 기존 자료와 검수 기록은 보존됩니다.")
        if digest(state["project"]) != state["config_sha256"]:
            raise ValueError("모델 학습 설정이 변경되었습니다. 실행 중인 실험 설정은 덮어쓰지 않습니다.")
        if state.get("implementation_id") != self._implementation_id:
            root = Path(state["project"]).parent
            jobs = root / ".studio" / "jobs"
            unstarted = (state.get("status") == "ready" and type(state.get("index")) is int
                         and state["index"] == 0 and state.get("results", {}) == {}
                         and state.get("inputs", {}) == {} and state.get("history", []) == []
                         and state.get("job") in (None, {}) and state.get("last_job") in (None, {})
                         and not (jobs.exists() and any(jobs.iterdir())) and not _lock_paths(root))
            if not unstarted:
                upgrade = self._migration_view(state)
                if upgrade and upgrade["state"] == "available":
                    raise ValueError("라이브러리 코드가 변경되었습니다. 호환되는 업데이트 적용 후 이어가기 버튼을 누르세요. 기존 자료는 보존됩니다.")
                raise ValueError("라이브러리 코드가 변경되었습니다. " + (upgrade or {}).get("reason", "기존 실행 자료를 보존하고 새 실험을 만드세요."))
            # A configuration made before the first worker has no execution or
            # human decisions to reinterpret. Preserve its exact prior state,
            # then bind only future requests to this implementation. Started
            # experiments still require an explicit compatibility migration.
            directory = root / ".studio" / "migrations" / uuid.uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            before = directory / "workflow_before.json"
            before.write_bytes(self._path(state["id"]).read_bytes())
            receipt = directory / "receipt.json"
            write_json(receipt, {
                "schema_version": 1, "kind": "unstarted_workflow_implementation_rebind",
                "old_implementation_id": state.get("implementation_id"),
                "new_implementation_id": self._implementation_id,
                "project": state["project"], "config_sha256": state["config_sha256"],
                "previous_workflow": {"path": str(before), "sha256": digest(before)},
                "migrated_at": now(),
                "reason": "No workflow execution or human decisions exist; frozen project configuration is unchanged",
                "preserved_scope": "Existing downloads, audio, approvals, models and evaluation files are unchanged"})
            state["implementation_id"] = self._implementation_id
            state["execution_contract"] = {**self._execution_contract,
                "settings_sha256": identity(state["settings"]), "plan_sha256": identity(state["plan"])}
            state.setdefault("migrations", []).append({"path": str(receipt), "sha256": digest(receipt)})
            self._save(state)

    def _reference_budget_change(self, state):
        """Expand only before call generation; never reinterpret a trained run."""
        from .studio_steps import preset_settings
        settings = state["settings"]
        old = settings.get("train_references")
        target = preset_settings(state["preset"])["train_references"]
        if type(old) is not int or old >= target:
            return None
        positions = {entry["id"]: i for i, entry in enumerate(state["plan"])}
        if ("references" not in positions or "references_review" not in positions
                or state["index"] > positions["references_review"]):
            return None
        root = Path(state["project"]).parent
        approval = read_json(root / "approval.json")
        if (root / "final_lock.json").exists() or approval.get("generation_policy_approved"):
            return None
        for directory in (root / "data/pilot", root / "data/generated", root / "runs"):
            if directory.exists() and any(p.is_file() for p in directory.rglob("*")):
                return None
        families = read_json(root / "voices.json")["families"]
        if sum(f.get("split") == "train" for f in families) < target:
            return None  # A frozen, smaller bank needs its own explicit plan.
        return {"old_train_references": old, "train_references": target,
                "validation_references": settings["validation_references"],
                "resume_index": min(state["index"], positions["references"]),
                "preserves_existing_bank": True}

    def _migration_view(self, state):
        if state.get("implementation_id") == self._implementation_id or state.get("status") == "completed":
            return None
        def blocked(reason):
            return {"state": "blocked", "reason": reason, "action": None}
        if self._job_alive(state.get("job") or {}):
            return blocked("현재 작업이 끝난 뒤 업데이트 호환성을 다시 확인합니다.")
        previous = state.get("execution_contract") or {}
        audited_update = None
        reference_update = None
        reference_admission_update = None
        pilot_update = None
        hard_update = None
        hard_review_update = None
        candidate_update = None
        call_recovery_update = None
        if previous:
            if (previous.get("schema_version") != 1 or previous.get("settings_sha256") != identity(state["settings"])
                    or previous.get("plan_sha256") != identity(state["plan"])):
                return blocked("저장된 설정 또는 단계 계약이 달라 자동으로 이관할 수 없습니다. 기존 자료는 보존됩니다.")
            remaining = [entry["id"] for entry in state["plan"][state["index"]:]]
            changed = [stage for stage in remaining if not previous.get("stages", {}).get(stage)
                       or previous["stages"][stage] != self._execution_contract["stages"].get(stage)]
            if changed:
                comparison = self._execution_contract
                legacy = self._pre_reference_review_contract
                if (legacy and "reference_voice_review.py" not in previous.get("files", {})
                        and not state.get("inputs", {}).get("reference_voice_review")):
                    comparison = legacy
                    changed = [stage for stage in remaining
                               if previous.get("stages", {}).get(stage) != comparison["stages"].get(stage)]
                audited_update = _audited_acquisition_update(previous, comparison) if changed else None
                reference_update = _audited_reference_update(previous, comparison) if changed else None
                reference_admission_update = _audited_reference_admission_update(previous, comparison) if changed else None
                pilot_update = _audited_pilot_update(previous, comparison) if changed else None
                hard_update = _audited_hard_per_phrase_update(previous, comparison) if changed else None
                hard_review_update = _audited_hard_review_update(previous, comparison) if changed else None
                candidate_update = _audited_reference_candidate_update(previous, comparison) if changed else None
                call_recovery_update = _audited_call_generation_recovery_update(previous, comparison) if changed else None
                if call_recovery_update and not self._can_update_call_generation_recovery(state):
                    return blocked("호출어 자동 복구 업데이트는 실행 중인 작업과 학습 결과를 보존하기 위해 특징 추출 전 작업에만 적용할 수 있습니다.")
                if call_recovery_update:
                    generic_binding = self._call_recovery_generic_import(state)
                    if generic_binding:
                        call_recovery_update["preserved_precomputed_generic_metadata"] = generic_binding
                candidate_compatible = (not candidate_update or
                                        (self._can_update_reference_optional_listening(state)
                                         if candidate_update.get("kind") == "audited_reference_optional_listening_update"
                                         else self._can_update_reference_candidates(state)))
                if candidate_update and not candidate_compatible:
                    return blocked("기준 목소리 후보 선택은 호출어 샘플 생성 전의 미확정 작업에만 적용할 수 있습니다. 기존 음원과 검수 기록은 보존됩니다.")
                if hard_update and not self._can_update_hard_budget(state):
                    return blocked("유사어 생성이 시작된 작업은 수량과 기존 기록을 유지합니다. 새 문구당 설정은 별도 모델에서 사용하세요.")
                if pilot_update and not self._can_update_pilot(state):
                    return blocked("목소리별 발음 검수 수량 변경은 대량 생성 전 작업에만 적용할 수 있습니다. 기존 음원과 검수 기록은 보존됩니다.")
                if reference_admission_update and not self._can_update_reference_admission(state):
                    return blocked("기준 목소리 자동 검사는 호출어 생성 전 작업에만 적용할 수 있습니다. 기존 자료와 검수 기록은 보존됩니다.")
                if changed and all(update is None for update in (audited_update, reference_update, reference_admission_update, pilot_update, hard_update, hard_review_update, candidate_update, call_recovery_update)):
                    return blocked("남은 단계의 자료·생성·학습·채점 코드가 달라 호환 이관할 수 없습니다. 기존 자료를 보존하고 새 실험을 만드세요.")
        elif (_LEGACY_COMPATIBLE_CONTRACTS.get(state.get("implementation_id"))
              != self._execution_contract["computation_sha256"]):
            return blocked("이 구버전에는 검증된 단계 계약이 없습니다. 기존 자료를 보존하고 별도 호환 검증이 필요합니다.")
        budget_change = self._reference_budget_change(state) if reference_update else None
        return {"state": "available", "action": "migrate",
                "reason": (call_recovery_update.get("reason") or "기존 기준 목소리와 통과한 호출어를 보존하고, 긴 앞뒤 무음을 정리한 입력으로 실패한 호출어만 제한된 횟수 안에서 자동 복구합니다. 검사 기준과 실제 검수 기록은 유지합니다."
                           if call_recovery_update else
                           "기존 후보와 선택·청취·확정 기록을 보존하고, 기준 목소리의 청취 필수 조건을 해제합니다. 기존 확정 기록은 변경하지 않습니다."
                           if candidate_update and candidate_update.get("kind") == "audited_reference_optional_listening_update" else
                           "기존 후보와 선택·청취·확정 기록을 보존하고, 없는 분류의 첫 후보만 준비합니다. 기존 완료 단계는 다시 실행하지 않습니다."
                           if candidate_update and candidate_update.get("kind") == "audited_reference_candidate_preparation_update" else
                           "기존 기준 목소리와 검수 기록을 보존하고, 성별·연령별 후보를 다섯 개씩 듣고 선택하는 검수로 전환합니다. 학습용과 공통 평가용을 각각 확정합니다."
                           if candidate_update else
                           "기존 음원과 검수 기록을 보존합니다. 아직 답변하지 않은 유사어 검수는 문구별 대표 1개로 줄이고, 전체 듣기는 별도로 제공합니다."
                           if hard_review_update else
                           "기존 호출어 음원과 검수 기록을 보존하고, 아직 시작하지 않은 유사어 생성을 문구당 수량으로 설정합니다."
                           if hard_update else
                           "기존 음원과 검수 기록을 보존하고 학습용 목소리마다 호출어 1개씩 확인하도록 목록을 갱신합니다."
                           if pilot_update else
                           "기존 목소리와 검수 기록을 보존하고 기준 목소리를 자동 검사합니다. 호출어 발음 대표 검수는 유지합니다."
                           if reference_admission_update else
                           "기존 목소리·검수 결과를 보존하고 부족한 학습용 기준 목소리만 추가합니다. 호출어 생성 전 작업에만 수량 변경을 적용합니다."
                           if reference_update else
                           "재사용한 공개 자료를 원래 준비 기록에 맞춰 검증하도록 수정한 업데이트입니다. 기존 음원·검수 결과·설정은 보존됩니다."
                           if audited_update and audited_update.get("changed_files") == ["material_policy.py"] else
                           "GPU 평가 도구 설치를 복구합니다. 검증을 통과한 설치 파일과 기존 학습 기록을 재사용합니다."
                           if audited_update and audited_update.get("changed_files") == ["gpu_runtime_install.py"] else
                           "GPU 평가에 실패하면 CPU로 전환하지 않고 멈춥니다. 기존 CPU 점수는 보존하고 별도의 GPU 평가로 진행합니다."
                           if audited_update and audited_update.get("changed_files") == ["evaluation_backend.py", "gpu_runtime_install.py", "studio_steps.py"] else
                           "다운로드 순서·진행 표시 변경의 자료 동일성을 확인한 업데이트입니다. 기존 설정과 완료 결과를 검증한 뒤 적용할 수 있습니다."
                           if audited_update else "남은 단계의 계산·자료 계약이 같습니다. 기존 설정과 완료 결과를 검증하고 업데이트를 적용할 수 있습니다."),
                **({"audited_acquisition_update": audited_update} if audited_update else {}),
                **({"audited_reference_update": reference_update} if reference_update else {}),
                **({"audited_reference_admission_update": reference_admission_update} if reference_admission_update else {}),
                **({"audited_pilot_update": pilot_update} if pilot_update else {}),
                **({"audited_hard_per_phrase_update": hard_update} if hard_update else {}),
                **({"audited_hard_review_update": hard_review_update} if hard_review_update else {}),
                **({"audited_reference_candidate_update": candidate_update} if candidate_update else {}),
                **({"audited_call_generation_recovery_update": call_recovery_update} if call_recovery_update else {}),
                **({"reference_budget_change": budget_change} if budget_change else {}),
                "preserves": ["project_settings", "completed_artifacts", "review_answers", "history"]}

    def _can_update_call_generation_recovery(self, state):
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        root = Path(state["project"]).parent
        if (state.get("status") == "completed" or state["index"] >= positions.get("features", -1)
                or state["settings"].get("pilot_review_policy") != PREPOST_PILOT_POLICY
                or self._job_alive(state.get("job") or {}) or self._reference_review_job_alive(state)
                or self._reference_preparation_job_alive(state) or (root / "final_lock.json").exists()):
            return False
        later = {stage["id"] for stage in state["plan"][positions["features"]:]}
        if (any(entry.get("stage") in later for entry in state.get("history", []))
                or any(stage in state.get("results", {}) for stage in later)):
            return False
        generic_binding = self._call_recovery_generic_import(state)
        allowed_metadata = bound_file(root, generic_binding) if generic_binding else None
        return not any(path.is_file() and path.stat().st_size
                       and not (allowed_metadata and path.samefile(allowed_metadata))
                       for folder in ("features", "runs") if (root / folder).exists()
                       for path in (root / folder).rglob("*"))

    def _call_recovery_generic_import(self, state):
        """A completed public-data import is not model-specific feature extraction."""
        from .frontend import PINNED_ASSETS
        root = Path(state["project"]).parent
        try:
            config = read_json(state["project"])
            path = resolved(root, config["pool_features"]["generic_negative"]) / "metadata.json"
            metadata = read_json(path)
            source = resolved(root, metadata["feature_path"])
            if (metadata.get("role") != "generic_negative" or metadata.get("split") != "train"
                    or metadata.get("imported_precomputed") is not True
                    or metadata.get("quality_review") != "accepted"
                    or type(metadata.get("feature_rows")) is not int or metadata["feature_rows"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", metadata.get("feature_sha256", ""))
                    or metadata.get("canonical_frontend_assets") != {
                        "melspectrogram": PINNED_ASSETS["melspectrogram.onnx"],
                        "embedding": PINNED_ASSETS["embedding_model.onnx"]}
                    or not source.is_file()):
                return None
            for entry in state.get("history", []):
                if entry.get("stage") != "data_connect" or digest(entry["result"]) != entry.get("sha256"):
                    continue
                completed = read_json(entry["result"])
                result = completed.get("result", {})
                imported = result.get("result", {}).get("generic_feature_import", {})
                if (result.get("state") == "completed" and imported.get("imported") is True
                        and resolved(root, imported["output"]).samefile(path.parent)
                        and resolved(root, imported["features"]).samefile(source)):
                    return {"path": str(path.relative_to(root)), "sha256": digest(path)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _can_update_reference_candidates(self, state, *, allow_frozen=False):
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        root = Path(state["project"]).parent
        if (state.get("status") == "completed" or state["index"] > positions.get("pilot", -1)
                or state["settings"].get("pilot_review_policy") != PREPOST_PILOT_POLICY
                or state.get("inputs", {}).get("reference_voice_review") and not allow_frozen
                or self._job_alive(state.get("job") or {}) or self._reference_review_job_alive(state)
                or (root / "final_lock.json").exists()
                or read_json(root / "approval.json").get("generation_policy_approved")):
            return False
        downstream = set(positions) - {"frontend", "data_download", "data_review", "data_connect", "references", "references_review"}
        if any(entry.get("stage") in downstream for entry in state.get("history", [])):
            return False
        if any(stage in state.get("results", {}) for stage in downstream):
            return False
        # A detached downstream artifact must not be hidden by an absent receipt.
        return not any(path.is_file() and path.stat().st_size for folder in ("data/pilot", "data/positive", "data/generated", "runs")
                       if (root / folder).exists() for path in (root / folder).rglob("*"))

    def _can_update_reference_optional_listening(self, state):
        if not self._can_update_reference_candidates(state, allow_frozen=True):
            return False
        binding = state.get("inputs", {}).get("reference_voice_review")
        if not binding:
            return True
        try:
            root = Path(state["project"]).parent
            manifest = read_json(bound_file(root, binding))
            if (manifest.get("kind") not in ("reference_voice_review_v2", "reference_voice_review_v3")
                    or "listening_policy" in manifest):
                return False
            from .reference_voice_review import overlay_reference_banks
            _, originals = self.reference_preview_rows(state["id"], include_design=True)
            banks = {split: [row for row in originals if row["split"] == split] for split in ("train", "validation")}
            overlay_reference_banks(root, banks, binding)
            return True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return False

    def _can_update_hard_review(self, state):
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        root = Path(state["project"]).parent
        if (state["index"] > positions["hard_review"] or self._job_alive(state.get("job") or {})
                or (root / "final_lock.json").exists()
                or state.get("inputs", {}).get("review_answers", {}).get("hard")
                or state.get("inputs", {}).get("review_results", {}).get("hard")):
            return False
        _, config = load_project(state["project"])
        if read_rows(resolved(root, config["manifests"]["hard_negative"])) or state.get("results", {}).get("hard_connect"):
            return False
        # A detached saved answer must not be evaded by rebuilding its queue.
        old_queues = {digest(path) for path in (root / ".studio").glob("hard_representatives*.jsonl")}
        for path in (root / ".studio/answers").glob("*.json"):
            if read_json(path).get("queue_sha256") in old_queues:
                return False
        return not (root / "review/content_results.json").exists()

    def _can_update_hard_budget(self, state):
        from .studio_steps import preset_settings
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        if state["index"] > positions["hard_generation"] or self._job_alive(state.get("job") or {}):
            return False
        if state["settings"]["train_references"] != preset_settings(state["preset"])["train_references"]:
            return False
        root = Path(state["project"]).parent
        if (root / "final_lock.json").exists() or "hard_generation" in state.get("results", {}):
            return False
        if any(item.get("stage") == "hard_generation" for item in state.get("history", [])):
            return False
        return not any(path.is_file() for folder in ("data/generated/hard", "runs")
                       if (root / folder).exists() for path in (root / folder).rglob("*"))

    def _can_update_reference_admission(self, state):
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        if state["index"] > positions.get("pilot", positions["positives"]) or self._job_alive(state.get("job") or {}):
            return False
        root = Path(state["project"]).parent
        if (root / "final_lock.json").exists() or read_json(root / "approval.json").get("generation_policy_approved"):
            return False
        return not any(path.is_file() for folder in ("data/pilot", "data/generated", "runs")
                       if (root / folder).exists() for path in (root / folder).rglob("*"))

    def _can_update_pilot(self, state):
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        if state["index"] > positions["pilot_review"] or self._job_alive(state.get("job") or {}):
            return False
        root = Path(state["project"]).parent
        if (root / "final_lock.json").exists() or read_json(root / "approval.json").get("generation_policy_approved"):
            return False
        return not any(path.is_file() for folder in ("data/generated", "runs")
                       if (root / folder).exists() for path in (root / folder).rglob("*"))

    def migrate(self, project_id):
        """Explicit, audited compatibility rebind; never an implicit resume."""
        with self._lock:
            state = self._load(project_id)
            if state.get("status") == "completed":
                return self.describe(state)  # Historical workflows are never rewritten for a UI update.
            if state.get("implementation_id") == self._implementation_id:
                return self.describe(state)
            compatibility = self._migration_view(state)
            if not compatibility or compatibility["state"] != "available":
                raise ValueError((compatibility or {}).get("reason", "이 업데이트는 자동 이관할 수 없습니다."))
            if digest(state["project"]) != state["config_sha256"]:
                raise ValueError("모델 학습 설정이 변경되어 업데이트를 적용할 수 없습니다.")
            for other_id in self.registry["projects"]:
                other = self._load(other_id)
                if self._job_alive(other.get("job") or {}) or self._reference_review_job_alive(other):
                    raise ValueError("실행 중인 작업이 있습니다. 해당 작업이 끝난 뒤 업데이트를 적용하세요.")
            self._verify_history(state)
            self._recover_dead_locks(state)
            if _lock_paths(Path(state["project"]).parent):
                raise ValueError("소유자를 확인할 수 없는 자료 잠금이 있어 업데이트를 적용할 수 없습니다.")
            directory = Path(state["project"]).parent / ".studio" / "migrations" / uuid.uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            before = directory / "workflow_before.json"
            before.write_bytes(self._path(project_id).read_bytes())
            receipt = directory / "receipt.json"
            write_json(receipt, {"schema_version": 1, "kind": "compatible_workflow_update",
                "migrated_at": now(), "old_implementation_id": state.get("implementation_id"),
                "new_implementation_id": self._implementation_id, "config_sha256": state["config_sha256"],
                "previous_workflow": {"path": str(before), "sha256": digest(before)},
                "old_execution_contract": state.get("execution_contract"),
                "new_computation_sha256": self._execution_contract["computation_sha256"],
                "remaining_stages": [entry["id"] for entry in state["plan"][state["index"]:]],
                "history_sha256": identity(state.get("history", [])), "inputs_sha256": identity(state.get("inputs", {})),
                **({"audited_acquisition_update": compatibility["audited_acquisition_update"]}
                   if compatibility.get("audited_acquisition_update") else {}),
                **({"audited_reference_update": compatibility["audited_reference_update"]}
                   if compatibility.get("audited_reference_update") else {}),
                **({"audited_reference_admission_update": compatibility["audited_reference_admission_update"]}
                   if compatibility.get("audited_reference_admission_update") else {}),
                **({"audited_pilot_update": compatibility["audited_pilot_update"]}
                   if compatibility.get("audited_pilot_update") else {}),
                **({"audited_hard_per_phrase_update": compatibility["audited_hard_per_phrase_update"]}
                   if compatibility.get("audited_hard_per_phrase_update") else {}),
                **({"audited_hard_review_update": compatibility["audited_hard_review_update"]}
                   if compatibility.get("audited_hard_review_update") else {}),
                **({"audited_reference_candidate_update": compatibility["audited_reference_candidate_update"]}
                   if compatibility.get("audited_reference_candidate_update") else {}),
                **({"audited_call_generation_recovery_update": compatibility["audited_call_generation_recovery_update"]}
                   if compatibility.get("audited_call_generation_recovery_update") else {}),
                **({"reference_budget_change": compatibility["reference_budget_change"]}
                   if compatibility.get("reference_budget_change") else {}),
                "preserved": compatibility["preserves"]})
            budget_change = compatibility.get("reference_budget_change")
            if compatibility.get("audited_call_generation_recovery_update"):
                stage = state["plan"][state["index"]]["id"]
                result = state.get("results", {}).get(stage, {})
                if (state.get("status") == "waiting_input" and stage in ("pilot", "positives")
                        and result.get("input_kind") == "generation_policy"
                        and result.get("result", {}).get("full_target_reached") is False):
                    state.update(status="paused", auto_continue=False, error=None,
                                 next_action="통과한 호출어는 유지하고 실패한 항목만 자동 복구합니다. 이어서 진행하세요.")
            if compatibility.get("audited_reference_candidate_update"):
                # Separate state/outputs preserve every original WAV, manifest and
                # v1 answer; only a later explicit approval freezes selected voices.
                if compatibility["audited_reference_candidate_update"].get("kind") == "audited_reference_optional_listening_update":
                    pass  # Only the audited execution binding changes; review data and position are immutable here.
                elif compatibility["audited_reference_candidate_update"].get("kind") == "audited_reference_candidate_preparation_update":
                    state["reference_preparation_policy"] = "initial_candidate_batches_v1"
                    state.update(status="paused", auto_continue=False, error=None,
                                 next_action="기준 목소리 준비에서 빠진 분류의 첫 후보를 준비한 뒤 검수를 이어가세요.")
                else:
                    state["reference_voice_review_policy"] = "candidate_pool_v2"
                    state.update(status="paused", auto_continue=False, error=None,
                                 next_action="기준 목소리 후보를 듣고 성별·연령별로 선택하세요.")
            if compatibility.get("audited_hard_review_update") and self._can_update_hard_review(state):
                state["settings"]["hard_review_policy"] = "one_per_phrase_v1"
                if state["plan"][state["index"]]["id"] == "hard_review":
                    from .studio_steps import _hard_queue
                    descriptor = _hard_queue(state["project"], Path(state["project"]).parent / ".studio", "one_per_phrase_v1")
                    state["results"]["hard_review"] = {"state": "waiting_review", "reviews": [descriptor],
                                                       "message": "문구별 대표 음원을 확인하고 저장하세요."}
                    state.update(status="waiting_review", error=None)
            if compatibility.get("audited_hard_per_phrase_update"):
                from .studio_steps import preset_settings
                state["settings"].pop("hard_originals", None)
                state["settings"].update(hard_generation_policy="per_phrase_v1",
                    hard_samples_per_phrase=preset_settings(state["preset"])["hard_samples_per_phrase"])
            if compatibility.get("audited_pilot_update"):
                from .studio_steps import workflow_plan
                state["settings"]["pilot_review_policy"] = TRAIN_PILOT_POLICY
                state["plan"] = workflow_plan(state["settings"])
                pilot_index = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot")
                state.update(index=min(state["index"], pilot_index), status="paused", auto_continue=False,
                             error=None, next_action="이어서 진행하면 기존 음원으로 목소리별 발음 검수 목록을 준비합니다.")
            if compatibility.get("audited_reference_admission_update"):
                from .studio_steps import workflow_plan
                state["settings"]["reference_admission_policy"] = "automatic_technical_v1"
                state["plan"] = workflow_plan(state["settings"])
                review_index = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "references_review")
                state.update(index=min(state["index"], review_index), status="paused", auto_continue=False,
                             error=None, next_action="이어서 진행하면 기준 목소리를 자동 검사합니다. 호출어 발음 검수는 그대로 진행합니다.")
            if budget_change:
                state["settings"]["train_references"] = budget_change["train_references"]
                state.update(index=budget_change["resume_index"], status="paused", auto_continue=False,
                             error=None, next_action="이어서 진행하면 기존 음원은 유지하고 부족한 학습용 목소리만 추가합니다.")
            state["implementation_id"] = self._implementation_id
            state["execution_contract"] = {**self._execution_contract,
                "settings_sha256": identity(state["settings"]), "plan_sha256": identity(state["plan"])}
            state.setdefault("migrations", []).append({"path": str(receipt), "sha256": digest(receipt)})
            if state.get("status") in ("running", "pausing"):
                state.update(status="interrupted", auto_continue=False)
            self._save(state)
            return self.describe(state)

    def _verify_history(self, state):
        verified = []
        for entry in state.get("history", []):
            if digest(entry["result"]) != entry["sha256"]:
                raise ValueError("이전 단계 완료 기록이 변경되었습니다.")
            verified.append((entry, read_json(entry["result"])))
        root, config = load_project(state["project"])
        policy = state["settings"].get("pilot_review_policy")
        positive_writer = {POSTGEN_PILOT_POLICY: "pilot_review", PREPOST_PILOT_POLICY: "positive_review"}.get(policy, "positives")
        writers = {resolved(root, config["manifests"][role]): writer for role, writer in
                   (("positive_train", positive_writer), ("positive_validation", positive_writer),
                    ("hard_negative", "hard_connect"))}
        latest = {}
        for index, (entry, completed) in enumerate(verified):
            if completed.get("error") or completed.get("result", {}).get("state") != "completed":
                continue
            for binding in completed.get("artifact_bindings", []):
                path = resolved(root, binding["path"])
                if writers.get(path) == entry["stage"]:
                    latest[path] = (index, binding["sha256"])
        observed = {}
        for index, (entry, completed) in enumerate(verified):
            for binding in completed.get("artifact_bindings", []):
                path = resolved(root, binding["path"])
                if path not in observed:
                    observed[path] = digest(path)
                if observed[path] == binding["sha256"]:
                    continue
                # data_connect creates empty future manifests. Only a later,
                # verified successful writer may fill those exact placeholders.
                # Earlier receipts and all non-placeholder artifacts stay intact.
                replacement = latest.get(path)
                if (entry["stage"] == "data_connect"
                        and binding["sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                        and path in writers):
                    if replacement and replacement[0] > index and replacement[1] == observed[path]:
                        continue
                    if (not replacement and writers[path] in ("pilot_review", "positive_review")
                            and state["index"] < len(state["plan"])
                            and state["plan"][state["index"]]["id"] == writers[path]):
                        from .pilot import matches_postgeneration_publication
                        if matches_postgeneration_publication(state["project"], self._effective_positive_generated(state), path):
                            continue
                raise ValueError("이전 단계의 결과 파일이 변경되었습니다: " + binding["path"])

    def _checkpoint_prefix_complete(self, state, count):
        stages = tuple(row["id"] for row in state["plan"][:count])
        if (len(stages) != count or state["index"] < count
                or any(state.get("results", {}).get(stage, {}).get("state") != "completed" for stage in stages)):
            return False
        connected = state["results"].get("data_connect", {})
        if connected.get("result", connected).get("shortages"):
            return False
        # A display marker alone cannot stand in for actual completed
        # worker receipts. Only read their small JSON, not downloaded archives.
        completed = {}
        for entry in state.get("history", []):
            if entry.get("stage") not in stages:
                continue
            if digest(entry["result"]) != entry.get("sha256"):
                raise ValueError("이전 단계 완료 기록이 변경되었습니다.")
            receipt = read_json(entry["result"])
            result = receipt.get("result", {})
            if not receipt.get("error") and result.get("state") == "completed":
                completed[entry["stage"]] = result
        return all(completed.get(stage) == state["results"][stage] for stage in stages)

    def _download_group_complete(self, state):
        return ([row["id"] for row in state["plan"][:4]] == ["frontend", "data_download", "data_review", "data_connect"]
                and self._checkpoint_prefix_complete(state, 4))

    @staticmethod
    def _navigation_bounds(state):
        plan = [row["id"] for row in state["plan"]]
        if "positive_review" in plan:
            positive_groups = (("call-pilot-generation", ("pilot",)), ("call-pilot-review", ("pilot_review",)),
                               ("positive-generation", ("positives",)), ("pronunciation", ("positive_review",)))
        elif "pilot" in plan:
            positive_groups = (("pronunciation", ("pilot", "pilot_review")), ("positive-generation", ("positives",)))
        else:
            positive_groups = (("positive-generation", ("positives",)), ("pronunciation", ("pilot_review",)))
        if "hard_pilot" in plan:
            hard_groups = (("hard-review", ("hard_pilot", "hard_review")),
                           ("hard-generation", ("hard_generation", "hard_connect")))
        else:
            hard_groups = (("hard-generation", ("hard_generation",)),
                           ("hard-review", ("hard_review", "hard_connect")))
        groups = _NAVIGATION_START + positive_groups + (("phrases", ("hard_phrases",)),) + hard_groups + _NAVIGATION_END
        if plan != [stage for _, stages in groups for stage in stages]:
            raise ValueError("화면 진행 확인과 실제 작업 순서가 다릅니다.")
        bounds, start = {}, 0
        for step, stages in groups:
            bounds[step] = (start, start + len(stages))
            start += len(stages)
        return bounds

    def _navigation_checkpoint_transition(self, state):
        checkpoint = state.get("navigation_checkpoint")
        if checkpoint is None:
            return None
        if (not isinstance(checkpoint, dict) or checkpoint.get("policy") != "explicit_story_v1"
                or checkpoint.get("state") not in ("awaiting_start", "running", "awaiting_next", "completed")):
            raise ValueError("화면 진행 확인 기록이 올바르지 않습니다.")
        bounds = self._navigation_bounds(state)
        if checkpoint.get("step") not in bounds:
            raise ValueError("화면 진행 확인 단계가 올바르지 않습니다.")
        phase, step, index = checkpoint["state"], checkpoint["step"], state["index"]
        start, end = bounds[step]
        if phase == "awaiting_start":
            if step != "downloads" or index != 0 or state.get("history") or state.get("job"):
                raise ValueError("작업 시작 대기 기록이 실제 진행 상태와 다릅니다.")
            return step
        if phase == "running":
            if not start <= index < end:
                raise ValueError("해당 화면의 작업 완료를 확인한 뒤 다음 단계로 진행하세요.")
            return step
        if (phase != "awaiting_next" or index != end or end >= len(state["plan"])
                or not self._checkpoint_prefix_complete(state, end)):
            raise ValueError("해당 화면의 작업과 필수 확인이 모두 완료되지 않았습니다.")
        return next((name for name, (left, _) in bounds.items() if left == end), None)

    def _download_checkpoint_transition(self, state):
        checkpoint = state.get("download_checkpoint")
        if checkpoint is None:
            return None
        if (not isinstance(checkpoint, dict) or checkpoint.get("policy") != "explicit_download_v1"
                or checkpoint.get("state") not in ("awaiting_start", "running", "awaiting_next", "released")):
            raise ValueError("다운로드 진행 확인 기록이 올바르지 않습니다.")
        phase, index = checkpoint["state"], state["index"]
        if phase == "awaiting_start":
            if index != 0 or state.get("history") or state.get("job"):
                raise ValueError("다운로드 시작 대기 기록이 실제 진행 상태와 다릅니다.")
            return "running"
        if phase == "running":
            if index >= 4:
                raise ValueError("다운로드 준비 완료를 확인한 뒤 다음 단계로 진행하세요.")
            return "running"
        if not self._download_group_complete(state) or phase == "awaiting_next" and index != 4:
            raise ValueError("다운로드·검사·자료 연결이 모두 완료되지 않았습니다.")
        return "released"

    def _launch(self, state):
        self._guard_config(state)
        self._stop_reference_worker()
        stage = state["plan"][state["index"]]
        if stage["id"] == "references" and self._reference_preparation_required(state):
            previous = state.get("job") or state.get("last_job") or {}
            self._recover_reference_preparation_outputs(state, previous if previous.get("stage") == "references" else {})
            # A new explicit start/resume clears the previous pause request;
            # the archived marker remains available for recovery auditing.
            marker = Path(state["project"]).parent / ".studio/reference-candidates/preparation/pause.json"
            if marker.exists():
                marker.replace(marker.with_name("pause-before-" + uuid.uuid4().hex + ".json"))
        task_id = uuid.uuid4().hex
        task_dir = Path(state["project"]).parent / ".studio" / "jobs" / task_id
        task_dir.mkdir(parents=True)
        storage_workspace = Path(state.get("storage_workspace", self.workspace))
        request = {"task_id": task_id, "project": state["project"], "workspace": str(storage_workspace),
                   "settings": state["settings"], "stage": stage["id"],
                   "context": {"results": state["results"], "inputs": state["inputs"]},
                   "config_sha256": state["config_sha256"], "implementation_id": self._implementation_id,
                   "result_path": str(task_dir / "result.json")}
        write_json(task_dir / "request.json", request)
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                           KOWW_CACHE_ROOT=state.get("cache_root", str(storage_workspace / "cache")))
        environment.pop("KOWW_STUDIO_REFERENCE_REVIEW", None)
        reference_review = state["inputs"].get("reference_voice_review")
        if reference_review:
            bound_file(Path(state["project"]).parent, reference_review)
            environment["KOWW_STUDIO_REFERENCE_REVIEW"] = json.dumps(reference_review)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        log_path = task_dir / "output.log"
        with log_path.open("ab") as log:
            child = self._popen([sys.executable, "-m", "korean_openwakeword.studio_worker", "--request", str(task_dir / "request.json")],
                                stdout=log, stderr=subprocess.STDOUT, cwd=Path(state["project"]).parent,
                                env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        creation = process_identity(child.pid)
        self._children[task_id] = child
        state.update(status="running", error=None, next_action="작업을 진행하고 있습니다.",
                     job={"task_id": task_id, "pid": child.pid, "creation_identity": creation,
                          "stage": stage["id"], "started_at": now(), "directory": str(task_dir),
                          "request_sha256": digest(task_dir / "request.json"), "log": str(log_path)})
        state.setdefault("started_at", now())
        self._save(state)

    def _finish_job(self, state):
        job = state.get("job") or {}
        if not job.get("directory"):
            return False
        path = Path(job["directory"]) / "result.json"
        if not path.exists():
            return False
        result = read_json(path)
        if (result.get("task_id") != job["task_id"] or result.get("request_sha256") != job["request_sha256"]
                or result.get("config_sha256") != state["config_sha256"]):
            raise ValueError("단계 완료 기록이 실제 실행 요청과 일치하지 않습니다.")
        for binding in result.get("artifact_bindings", []):
            if digest(binding["path"]) != binding["sha256"]:
                raise ValueError("단계 결과 파일이 변경되었습니다: " + binding["path"])
        current_hash = digest(state["project"])
        if current_hash != result.get("result_config_sha256", state["config_sha256"]):
            raise ValueError("단계 이후 모델 학습 설정이 변경되었습니다.")
        state["config_sha256"] = current_hash
        state["last_job"] = job
        state["job"] = None
        if result.get("error"):
            state.update(status="error", error=result["error"], next_action="오류 원인을 확인한 뒤 이어서 진행을 누르세요.")
        else:
            output = result["result"]
            stage = job["stage"]
            state["results"][stage] = output
            phase = output.get("state", "completed")
            if phase == "waiting_review":
                for review in output.get("reviews", []):
                    if digest(review["queue"]) != review.get("queue_sha256"):
                        raise ValueError("완료 시점의 검수 목록이 변경되었습니다.")
                    if review.get("pool_manifest"):
                        review["pool_sha256"] = digest(review["pool_manifest"])
                    # An invalidated saved answer must not hide a repeated gate.
                    for key in ("review_answers", "review_results"):
                        state["inputs"].get(key, {}).pop(review["id"], None)
                state.update(status=phase, next_action="아래 음원을 확인하고 검수 완료·저장을 누르세요.")
            elif phase == "waiting_input":
                state.update(status=phase, next_action=output.get("message", "아래 선택을 확인하세요."))
            elif phase == "completed":
                state["index"] += 1
                state.update(status="completed" if state["index"] >= len(state["plan"]) else "ready", next_action="다음 단계를 준비합니다.")
                if state["status"] == "completed":
                    state["auto_continue"] = False
                    state["completed_at"] = result.get("completed_at") if _timestamp(result.get("completed_at")) else now()
            elif phase == "paused" and stage == "references" and self._reference_preparation_required(state):
                state.update(status="paused", auto_continue=False,
                             next_action="현재 후보 묶음을 저장했습니다. 재개하면 남은 분류를 준비합니다.")
            else:
                state.update(status="error", error="지원하지 않는 단계 종료 상태: " + str(phase))
        state.setdefault("history", []).append({"stage": job["stage"], "started_at": job["started_at"],
                                                "completed_at": now(), "result": str(path), "sha256": digest(path)})
        checkpoint = state.get("download_checkpoint")
        if (checkpoint and job["stage"] == "data_connect" and state["status"] == "ready"
                and state["index"] == 4):
            if (not isinstance(checkpoint, dict) or checkpoint.get("policy") != "explicit_download_v1"
                    or checkpoint.get("state") != "running" or not self._download_group_complete(state)):
                raise ValueError("다운로드 준비 완료 근거를 확인할 수 없습니다.")
            checkpoint["state"] = "awaiting_next"
            state.update(status="paused", auto_continue=False,
                         next_action="자료 준비를 마쳤습니다. 다음을 누르세요.")
        navigation = state.get("navigation_checkpoint")
        if navigation and state["status"] in ("ready", "completed"):
            if not isinstance(navigation, dict) or navigation.get("policy") != "explicit_story_v1":
                raise ValueError("화면 진행 확인 기록이 올바르지 않습니다.")
            bounds = self._navigation_bounds(state)
            step = navigation.get("step")
            if step not in bounds or navigation.get("state") != "running":
                raise ValueError("화면 진행 확인 기록이 실제 작업과 다릅니다.")
            if state["index"] == bounds[step][1]:
                if not self._checkpoint_prefix_complete(state, state["index"]):
                    raise ValueError("해당 화면의 작업 완료 근거를 확인할 수 없습니다.")
                if state["status"] == "completed":
                    navigation["state"] = "completed"
                else:
                    navigation["state"] = "awaiting_next"
                    state.update(status="paused", auto_continue=False,
                                 next_action="이 단계의 작업을 마쳤습니다. 다음을 누르세요.")
        self._save(state)
        return True

    def start(self, project_id):
        with self._lock:
            state = self._load(project_id)
            if state["status"] == "completed":
                return self.describe(state)
            self._guard_config(state)
            if self._reference_preparation_job_alive(state):
                raise ValueError("기준 목소리 후보 준비 작업이 끝난 뒤 진행하세요.")
            if (self._reference_review_required(state) and self._reference_preparation_required(state)
                    and not self._reference_preparation_view(state)["ready"]):
                raise ValueError("모든 분류의 첫 후보 준비를 완료한 뒤 검수로 이동하세요.")
            checkpoint_state = self._download_checkpoint_transition(state)
            navigation_step = self._navigation_checkpoint_transition(state)
            if self._reference_review_required(state):
                reference_review = state["inputs"].get("reference_voice_review")
                if not reference_review:
                    raise ValueError("기준 목소리의 모든 묶음을 검수한 뒤 다음 단계로 이동하세요.")
                bound_file(Path(state["project"]).parent, reference_review)
            if state["status"] in ("running", "pausing") and self._job_alive(state.get("job") or {}):
                state["auto_continue"] = True
                self._save(state)
                return self.describe(state)
            for other_id in self.registry["projects"]:
                other = self._load(other_id)
                if self._reference_review_job_alive(other):
                    raise ValueError("기준 목소리 다시 생성이 끝난 뒤 학습을 시작하세요.")
                if self.analysis.busy(other_id):
                    raise ValueError("모델 비교·선택 작업이 끝난 뒤 학습을 시작하세요.")
                scheduled = other["status"] in ("ready", "running", "pausing") and other.get("auto_continue")
                if other_id != project_id and (scheduled or self._job_alive(other.get("job") or {})):
                    raise ValueError("다른 모델 학습이 실행 중입니다. 해당 작업을 먼저 일시정지하세요.")
            if state["status"] == "completed":
                return self.describe(state)
            if state["status"] == "waiting_review":
                return self.describe(state)
            if state["status"] == "waiting_input" and not state.get("input_submitted"):
                return self.describe(state)
            state.update(auto_continue=True, input_submitted=False)
            self._verify_history(state)
            # Finish a worker that completed while the browser/server was closed.
            if state.get("job") and self._finish_job(state):
                if (state["status"] in ("waiting_review", "waiting_input", "completed", "error")
                        or (state.get("download_checkpoint") or {}).get("state") == "awaiting_next"
                        or (state.get("navigation_checkpoint") or {}).get("state") == "awaiting_next"):
                    return self.describe(state)
            if checkpoint_state:
                state["download_checkpoint"]["state"] = checkpoint_state
            if navigation_step:
                state["navigation_checkpoint"].update(step=navigation_step, state="running")
            self._recover_dead_locks(state)
            self._launch(state)
            return self.describe(state)

    def pause(self, project_id):
        with self._lock:
            state = self._load(project_id)
            if self._reference_preparation_job_alive(state):
                return self.prepare_pause(project_id)
            state["auto_continue"] = False
            if (self._reference_preparation_required(state) and self._job_alive(state.get("job") or {})
                    and (state.get("job") or {}).get("stage") == "references"):
                write_json(Path(state["project"]).parent / ".studio/reference-candidates/preparation/pause.json",
                           {"task_id": state["job"]["task_id"], "requested_at": now()})
            if self._job_alive(state.get("job") or {}):
                state.update(status="pausing", next_action="현재 단계를 완료한 뒤 일시정지합니다. 자료를 강제로 끊지 않습니다.")
            elif state["status"] not in ("completed", "waiting_review", "waiting_input"):
                state.update(status="paused", next_action="이어서 진행을 누르면 재개합니다.")
            self._save(state)
            return self.describe(state)

    def _recover_dead_locks(self, state):
        previous = state.get("job") or state.get("last_job")
        if not previous or self._job_alive(previous):
            return
        # Only the exact identified previous worker may own recoverable locks.
        root = Path(state["project"]).parent
        shared = Path(state.get("storage_workspace", self.workspace)) / "shared"
        paths = _lock_paths(root)
        for pattern in ("*/.writer.lock", "downloads/*/.writer.lock", "downloads/common_rir/*/.writer.lock", "mit_rir/16khz/*/.writer.lock",
                        "automatic_sources_v3/parts/*/.writer.lock",
                        "automatic_sources_v3/parts/*/.canonical_lock/.writer.lock"):
            paths.extend(shared.glob(pattern))
        for path in paths:
            record = read_json(path)
            owners = {previous.get("pid"), self._worker_owner(previous).get("pid")}
            if record.get("pid") not in owners:
                raise ValueError("다른 작업의 자료 잠금이 남아 있습니다: " + str(path))
            if process_identity(record["pid"]) is not None:
                raise ValueError("잠금 소유 PID가 현재 살아 있습니다. 재사용된 PID라도 임의 해제하지 않습니다.")
            archive = root / ".studio" / "recovered_locks" / (identity(str(path)) + "_" + uuid.uuid4().hex + ".json")
            write_json(archive, {"original": str(path), "record": record, "job": previous, "recovered_at": now()})
            path.unlink()

    def tick(self):
        with self._lock:
            self._expire_reference_worker()
            for project_id in self.registry["projects"]:
                state = self._load(project_id)
                if state.get("reference_preparation_job"):
                    self._finish_reference_preparation(state)
                if state["status"] not in ("running", "pausing"):
                    continue
                job = state.get("job") or {}
                child = self._children.get(job.get("task_id"))
                if child is not None and child.poll() is None or self._job_alive(job):
                    continue
                continuing = state.get("auto_continue", False)
                try:
                    finished = self._finish_job(state)
                except (OSError, ValueError, KeyError) as error:
                    state.update(status="error", auto_continue=False, error={"message": str(error), "type": type(error).__name__},
                                 next_action="완료 근거 확인에 실패했습니다. 로그를 확인하세요.")
                    self._save(state)
                    continue
                if not finished:
                    state.update(status="interrupted", auto_continue=False,
                                 next_action="작업이 중단됐습니다. 로그를 확인한 뒤 이어서 진행하세요.")
                    self._save(state)
                elif state["status"] == "ready":
                    if continuing:
                        if (state.get("implementation_id") != self._implementation_id
                                and (state.get("execution_contract") or {}).get("computation_sha256")
                                == self._execution_contract["computation_sha256"]):
                            # A user-started run can adopt a display-only update after
                            # the current worker's completion has been verified.
                            self.migrate(project_id)
                            state = self._load(project_id)
                        self._launch(state)
                    else:
                        state.update(status="paused", next_action="일시정지했습니다. 이어서 진행을 누르세요.")
                        self._save(state)

    def _loop(self):
        while not self._closing.wait(.75):
            try:
                self.tick()
            except (OSError, ValueError, KeyError) as error:
                # Persist orchestration failures instead of abandoning a daemon.
                write_json(self.directory / "manager_error.json", {"error": str(error), "at": now()})

    def run(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def close(self):
        self._closing.set()
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            self._stop_reference_worker(wait=False)
        for service in list(self._microphones.values()):
            with service._lock:
                service._close()

    def submit_input(self, project_id, payload):
        with self._lock:
            state = self._load(project_id)
            if state["status"] != "waiting_input":
                raise ValueError("현재 입력을 기다리는 단계가 아닙니다.")
            defer_dependency_install = payload.get("defer_dependency_install", False)
            if type(defer_dependency_install) is not bool:
                raise ValueError("학습 도구 설치 연기 설정은 참 또는 거짓으로 지정하세요.")
            if defer_dependency_install:
                stage = state["plan"][state["index"]]["id"]
                result = state["results"].get(stage, {})
                input_kind = (result.get("input") or {}).get("kind", result.get("input_kind"))
                backend = payload.get("dependency_backend")
                if (stage != "frontend" or input_kind != "dependency_setup" or backend not in ("cpu", "cu128")
                        or set(payload) - {"dependency_backend", "defer_dependency_install"}):
                    raise ValueError("학습 도구를 선택하는 단계에서만 설치를 미룰 수 있습니다.")
                state["inputs"]["dependency_backend"] = backend
                state.update(status="paused", input_submitted=True, auto_continue=False,
                             dependency_setup_selected=True, next_action="설치·다운로드 시작을 누르세요.")
                self._save(state)
                return self.describe(state)
            if "phrases" in payload:
                phrases = payload["phrases"]
                if not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) for p in phrases):
                    raise ValueError("유사어를 한 줄에 하나씩 입력하세요.")
                from .hard_tts import _phrases
                _, config = load_project(state["project"])
                phrases = _phrases(phrases, config["wakeword"])
                state["inputs"].update(hard_phrases=phrases, hard_phrases_approved=True)
            elif "admissions" in payload:
                admissions = payload["admissions"]
                if not isinstance(admissions, list) or not admissions or any(not isinstance(p, str) for p in admissions):
                    raise ValueError("준비된 자료 묶음의 승인 기록 경로가 필요합니다.")
                state["inputs"]["material_admissions"] = [str(_safe_root(p)) for p in admissions]
            elif payload.get("dependency_backend") in ("cpu", "cu128"):
                state["inputs"]["dependency_backend"] = payload["dependency_backend"]
            elif "source" in payload:
                source = payload["source"]
                if not isinstance(source, dict) or not isinstance(source.get("directory"), str):
                    raise ValueError("추가 자료의 음원 폴더를 선택하세요.")
                source = dict(source, directory=str(_safe_root(source["directory"])))
                state["inputs"].setdefault("additional_sources", []).append(source)
            else:
                raise ValueError("지원하지 않는 입력입니다.")
            state.update(status="paused", input_submitted=True)
            self._save(state)
            return self.start(project_id)

    def review_descriptors(self, state):
        if state["status"] != "waiting_review":
            return []
        stage = state["plan"][state["index"]]["id"]
        reviews = state["results"][stage].get("reviews", [])
        return [r for r in reviews if r["id"] not in state["inputs"].get("review_results", {})]

    @staticmethod
    def _call_review_phase(state, review):
        if state["settings"].get("pilot_review_policy") == PREPOST_PILOT_POLICY:
            return {"pilot": "pre", "positive": "post"}.get(review["kind"])
        return None

    def review_queue(self, project_id, queue_id):
        state = self._load(project_id)
        review = next((r for r in self.review_descriptors(state) if r["id"] == queue_id), None)
        if not review:
            raise ValueError("현재 검수 목록이 아닙니다. 화면을 새로고침하세요.")
        path = Path(review["queue"])
        if digest(path) != review.get("queue_sha256") or (review.get("pool_manifest") and
                digest(review["pool_manifest"]) != review.get("pool_sha256")):
            raise ValueError("검수 목록 또는 승인할 원본 목록이 변경되었습니다.")
        if review.get("synthetic_review_policy"):
            from .common import bound_file
            selection = read_json(bound_file(Path(state["project"]).parent, review["representative_selection"]))
            if (review["synthetic_review_policy"] not in ("one_per_phrase_v1", HARD_SAMPLE_REVIEW_POLICY)
                    or selection.get("policy") != review["synthetic_review_policy"]
                    or selection.get("source_manifest", {}).get("sha256") != review["pool_sha256"]
                    or selection.get("representative_queue", {}).get("sha256") != review["queue_sha256"]):
                raise ValueError("유사어 대표 선정 기록이 현재 검수 목록과 다릅니다.")
        rows = read_rows(path)
        if not rows:
            raise ValueError("검수할 음원이 없습니다.")
        return state, review, rows

    @staticmethod
    def _preflight_draft_path(state, review):
        return Path(state["project"]).parent / ".studio/review-drafts" / (
            identity([review["id"], review["queue_sha256"]]) + ".json")

    def _preflight_draft(self, state, review, rows):
        from .review import PLAY_STARTED_POLICY, validate_started_decision
        if (self._call_review_phase(state, review) != "pre" or any(
                row.get("pronunciation_policy") != PREPOST_PILOT_POLICY
                or "preflight_round" not in row or row.get("qc", {}).get("passed") is not True for row in rows)):
            raise ValueError("자동 저장할 호출어 사전 검수 목록이 아닙니다.")
        expected = {"policy": PLAY_STARTED_POLICY, "project_id": state["id"],
                    "queue_id": review["id"], "queue_sha256": review["queue_sha256"],
                    "queued_audio_sha256": {row["sample_id"]: row["sha256"] for row in rows}}
        path = self._preflight_draft_path(state, review)
        if not path.exists():
            return {**expected, "revision": 0, "decisions": {}}
        draft = read_json(path)
        if (any(draft.get(key) != value for key, value in expected.items())
                or type(draft.get("revision")) is not int or draft["revision"] < 0
                or not isinstance(draft.get("decisions"), dict)
                or not set(draft["decisions"]).issubset(expected["queued_audio_sha256"])):
            raise ValueError("자동 저장한 사전 검수 초안의 음원 연결이 바뀌었습니다.")
        if (draft["revision"] == 0 and draft["decisions"]
                or draft["revision"] > 0 and (
                    not isinstance(draft.get("saved_at"), str) or not draft["saved_at"]
                    or draft.get("source_scope") != "Playback events and explicit decisions entered through the local review screen")):
            raise ValueError("자동 저장한 사전 검수 초안의 입력 출처가 올바르지 않습니다.")
        for answer in draft["decisions"].values():
            validate_started_decision(answer)
        return draft

    @staticmethod
    def _draft_response(draft, rows):
        return {"state": "draft_saved", "autosave_policy": draft["policy"],
                "queue_sha256": draft["queue_sha256"],
                "draft_decisions": draft["decisions"], "draft_revision": draft["revision"],
                "complete": set(draft["decisions"]) == {row["sample_id"] for row in rows},
                "kept": sum(answer["decision"] == "keep" for answer in draft["decisions"].values()), "total": len(rows)}

    def preflight_draft(self, project_id, queue_id):
        with self._lock:
            state, review, rows = self.review_queue(project_id, queue_id)
            return self._draft_response(self._preflight_draft(state, review, rows), rows)

    def _save_preflight_draft(self, state, review, rows, payload):
        from .review import validate_started_decision
        draft = self._preflight_draft(state, review, rows)
        if type(payload.get("revision")) is not int or payload["revision"] != draft["revision"]:
            raise ValueError("다른 화면에서 판정이 바뀌었습니다. 저장된 초안을 다시 불러오세요.")
        sample = next((row for row in rows if row["sample_id"] == payload.get("sample_id")), None)
        if not sample or payload.get("audio_sha256") != sample["sha256"]:
            raise ValueError("자동 저장할 음원이 바뀌었습니다. 화면을 새로고침하세요.")
        bound_file(Path(state["project"]).parent, sample)
        decision = payload.get("decision")
        started, completed = payload.get("playback_started"), payload.get("playback_completed")
        if type(started) is not bool or type(completed) is not bool or completed and not started:
            raise ValueError("재생 시작과 재생 완료 기록을 확인하세요.")
        previous = draft["decisions"].get(sample["sample_id"], {})
        if decision == "clear":
            if started or completed or previous.get("playback_started") or previous.get("playback_completed"):
                raise ValueError("실제로 재생한 기록은 지울 수 없습니다. 정상 또는 다시 생성을 선택하세요.")
            changed = draft["decisions"].pop(sample["sample_id"], None) is not None
        else:
            answer = {"decision": decision, "playback_started": started or previous.get("playback_started", False),
                      "playback_completed": completed or previous.get("playback_completed", False)}
            answer["listened"] = answer["playback_completed"]
            validate_started_decision(answer)
            changed = answer != previous
            draft["decisions"][sample["sample_id"]] = answer
        if changed:
            draft.update(revision=draft["revision"] + 1, saved_at=now(),
                         source_scope="Playback events and explicit decisions entered through the local review screen")
            write_json(self._preflight_draft_path(state, review), draft)
        return self._draft_response(draft, rows)

    @staticmethod
    def _call_voice_reference(row):
        reference = row.get("preflight_reference")
        if not isinstance(reference, dict) or any(row.get(key) and reference.get(key)
                and row[key] != reference[key] for key in ("split", "voice_family_id")):
            return row
        return reference

    @staticmethod
    def _preflight_voice_label(row, reference=None, *, number=None):
        fallback = reference if isinstance(reference, dict) else {}
        direct = StudioManager._call_voice_reference(row)
        def trait(source, axis):
            value = source.get(axis)
            return _REFERENCE_PREVIEW_TRAIT_LABELS[axis].get(value, "") if isinstance(value, str) else ""
        gender = trait(direct, "gender") or trait(fallback, "gender")
        age = trait(direct, "age_group") or trait(fallback, "age_group")
        voice_number = row.get("voice_number")
        if type(voice_number) is not int or voice_number < 1:
            voice_number = fallback.get("voice_number")
        if type(number) is not int or number < 1:
            number = voice_number
        split = row.get("split")
        group = row.get("review_group") or row.get("group") or ({"train": "학습용", "validation": "공통 평가용"}.get(split, "") if isinstance(split, str) else "")
        group = group if isinstance(group, str) else ""
        return {"gender": gender, "age_group": age,
                **({"voice_number": voice_number} if type(voice_number) is int and voice_number > 0 else {}),
                "label": " · ".join(value for value in (group, gender, age,
                    f"{number}번" if type(number) is int and number > 0 else "") if value)}

    def _call_voice_label_index(self, state):
        """Small frozen reference metadata only; never scan generated audio pools.

        A family can be regenerated under the same id. Exact reference revisions
        therefore remain distinct; only a row with no reference hash may use the
        unambiguous family fallback retained for older projects.
        """
        index, ambiguous = {}, set()

        def read(binding, *, rows=False):
            if (not isinstance(binding, dict) or not isinstance(binding.get("path"), str)
                    or not isinstance(binding.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])):
                raise ValueError("기준 목소리 표시 근거가 없습니다.")
            return self._history_read(state, binding["path"], sha256=binding["sha256"], rows=rows,
                                      max_bytes=16 * 1024**2)

        def add(split, family, revision, metadata):
            if split not in ("train", "validation") or not isinstance(family, str) or not family:
                return
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision):
                revision = None
            value = {key: metadata[key] for key in ("gender", "age_group", "voice_number") if metadata.get(key) is not None}
            for key in ((split, family, revision), (split, family, None)):
                if key in ambiguous:
                    continue
                previous = index.get(key, {})
                if any(previous.get(axis) is not None and previous[axis] != item for axis, item in value.items()):
                    ambiguous.add(key); index.pop(key, None)
                else:
                    index[key] = {**previous, **value}

        try:
            manifest = read(state.get("inputs", {}).get("reference_voice_review"))
            if not isinstance(manifest, dict):
                raise ValueError("기준 목소리 표시 기록이 아닙니다.")
            bank = read(manifest.get("voice_bank"))
            if not isinstance(bank, dict):
                raise ValueError("기준 목소리 표시 은행이 아닙니다.")
            families = bank.get("families", [])
            if not isinstance(families, list) or len(families) > 10000:
                raise ValueError("기준 목소리 표시 목록이 너무 큽니다.")
            originals, replacements = manifest.get("originals", {}), manifest.get("replacements", {})
            originals = originals if isinstance(originals, dict) else {}
            replacements = replacements if isinstance(replacements, dict) else {}
            seen = set()
            for split, setting in (("train", "train_references"), ("validation", "validation_references")):
                limit = state.get("settings", {}).get(setting)
                selected = [row for row in families if isinstance(row, dict) and row.get("split") == split]
                if type(limit) is int and limit >= 0:
                    selected = selected[:limit]
                for number, family in enumerate(selected, 1):
                    family_id = family.get("family_id")
                    if not isinstance(family_id, str) or not family_id:
                        continue
                    key = (split, family_id)
                    if key in seen:
                        for existing in list(index):
                            if existing[:2] == key:
                                ambiguous.add(existing); index.pop(existing, None)
                        ambiguous.add((*key, None)); continue
                    seen.add(key)
                    replacement = replacements.get(family_id, {})
                    replacement = replacement if isinstance(replacement, dict) else {}
                    if (replacement.get("voice_family_id") not in (None, family_id)
                            or replacement.get("split") not in (None, split)):
                        replacement = {}
                    metadata = {axis: replacement.get(axis, family.get(axis)) for axis in ("gender", "age_group")}
                    metadata["voice_number"] = number
                    revision = replacement.get("sha256") or originals.get(family_id)
                    add(split, family_id, revision if isinstance(revision, str) else None, metadata)
        except (ValueError, KeyError, OSError, TypeError):
            pass  # Missing legacy display evidence never changes audio or approval.
        records = state.get("inputs", {}).get("call_pilot_reviews", [])
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            try:
                rows = read(record.get("queue"), rows=True)
                if len(rows) > 1000:
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    reference = row.get("preflight_reference")
                    if (not isinstance(reference, dict) or reference.get("voice_family_id") != row.get("voice_family_id")
                            or reference.get("split") != row.get("split") or not isinstance(reference.get("sha256"), str)):
                        continue
                    add(row["split"], row["voice_family_id"], reference["sha256"],
                        {**{axis: reference.get(axis) for axis in ("gender", "age_group")}, "voice_number": row.get("voice_number")})
            except (ValueError, KeyError, OSError, TypeError):
                continue
        return index

    def _call_voice_labels(self, state, rows, *, numbers=None):
        direct = [self._preflight_voice_label(row) for row in rows]
        index = self._call_voice_label_index(state) if any(
            not label["gender"] or not label["age_group"] or "voice_number" not in label for label in direct) else {}
        labels = []
        for position, row in enumerate(rows):
            revision = row.get("voice_reference_sha256")
            split, family = row.get("split"), row.get("voice_family_id")
            valid_identity = split in ("train", "validation") and isinstance(family, str)
            metadata = index.get((split, family, revision), {}) if valid_identity and (revision is None or isinstance(revision, str)) else {}
            if revision and not metadata:
                preparation = row.get("clone_reference_preparation")
                preparation = preparation if isinstance(preparation, dict) else {}
                source, prepared = preparation.get("source"), preparation.get("prepared")
                source = source if isinstance(source, dict) else {}
                prepared = prepared if isinstance(prepared, dict) else {}
                if (preparation.get("kind") == "clone_reference_preparation"
                        and preparation.get("policy") == "clone_reference_edge_silence_v1"
                        and prepared.get("sha256") == revision and valid_identity
                        and isinstance(source.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
                    metadata = index.get((split, family, source["sha256"]), {})
            labels.append(self._preflight_voice_label(row, metadata, number=numbers[position] if numbers is not None else None))
        return labels

    def _effective_positive_generated(self, state, *, presentation=False):
        generated = self._history_result(state, "positives")
        rounds = state.get("inputs", {}).get("postgeneration_repair_rounds", [])
        if generated is not None and rounds:
            if presentation:
                # Status/history/preview only read the small already-bound
                # completion evidence. Explicit saves and worker admission use
                # the full resolver below, including every generation journal.
                output = state.get("results", {}).get("positive_review", {})
                output = output.get("result", output)
                current = output.get("effective_generated")
                if current is None:
                    plan = self._history_read(state, rounds[-1]["request"]["plan"]["path"],
                        sha256=rounds[-1]["request"]["plan"]["sha256"])
                    binding = plan["parent_generated"]
                    current = self._history_read(state, binding["path"], sha256=binding["sha256"])
                metadata = current.get("postgeneration_repair")
                if metadata:
                    if metadata.get("plan") not in [entry["request"]["plan"] for entry in rounds]:
                        raise ValueError("표시할 호출어 복구 내역의 순서가 다릅니다.")
                    binding = metadata["evidence"]
                    evidence = self._history_read(state, binding["path"], sha256=binding["sha256"])
                    if evidence.get("plan") != metadata["plan"] or current != {**evidence.get("generated", {}), "postgeneration_repair": metadata}:
                        raise ValueError("표시할 호출어 복구 완료 근거가 다릅니다.")
                elif current != generated:
                    raise ValueError("표시할 호출어 원본 생성 기록이 다릅니다.")
                return current
            from .postgeneration_repair import resolve_postgeneration_repair
            generated = resolve_postgeneration_repair(state["project"], generated, rounds,
                reference_review=state.get("inputs", {}).get("reference_voice_review"))
        return generated

    def _postgeneration_repair_review(self, state, descriptor):
        if not state.get("inputs", {}).get("postgeneration_repair_rounds"):
            return {}
        from .postgeneration_repair import postgeneration_repair_review
        generated = self._effective_positive_generated(state)
        if not generated or generated.get("queue_binding", {}).get("sha256") != descriptor["queue_sha256"]:
            raise ValueError("다시 생성한 호출어 검수 목록의 연결이 다릅니다.")
        return postgeneration_repair_review(state["project"], generated,
            reference_review=state.get("inputs", {}).get("reference_voice_review"))

    def _start_postgeneration_repair(self, state, rows, report_binding):
        from .postgeneration_repair import prepare_postgeneration_repair
        generated = self._effective_positive_generated(state)
        if not generated:
            raise ValueError("본생성 완료 근거가 없어 다시 생성할 수 없습니다.")
        request = prepare_postgeneration_repair(state["project"], generated, report_binding,
            reference_review=state.get("inputs", {}).get("reference_voice_review"))
        root = Path(state["project"]).parent
        report = read_json(bound_file(root, report_binding))
        bad = [row for row in rows if report["decisions"][row["sample_id"]]["decision"] == "discard"]
        record = {"request": request, "source_queue_sha256": report["queue_sha256"],
                  "family_labels": [item["label"] for item in self._call_voice_labels(state, bad)]}
        rounds = state["inputs"].setdefault("postgeneration_repair_rounds", [])
        if not rounds or rounds[-1]["request"]["plan"] != request["plan"]:
            rounds.append(record)
        state["results"].pop("positive_review", None)
        checkpoint = state.get("navigation_checkpoint")
        if checkpoint:
            checkpoint.update(step="pronunciation", state="running")
        state.update(status="paused", auto_continue=False, input_submitted=False,
                     next_action="부적합한 목소리의 호출어만 다시 생성합니다. 기존 적합 판정은 보존합니다.")
        self._save(state)
        return self._resume_postgeneration_repair(state)

    def _resume_postgeneration_repair(self, state):
        try:
            view = self.start(state["id"])
        except (OSError, ValueError) as error:
            return {"state": "postgeneration_repair_pending", "review_finalized": True,
                    "workflow_action": "repair", "message": str(error)}
        started = view.get("status") in ("running", "pausing")
        return {"state": "postgeneration_repair_started" if started else "postgeneration_repair_pending",
                "review_finalized": True, "workflow_action": "repair"}

    def _postgeneration_repair_view(self, state):
        rounds = state.get("inputs", {}).get("postgeneration_repair_rounds", [])
        if not rounds:
            return None
        from .postgeneration_repair import postgeneration_repair_progress
        record, stage = rounds[-1], state["plan"][state["index"]]["id"] if state["index"] < len(state["plan"]) else "completed"
        request = record["request"]
        view = {"state": "completed" if stage != "positive_review" else
                "waiting_review" if state["status"] == "waiting_review" else
                "running" if state["status"] in ("running", "pausing") else
                "failed" if state["status"] in ("error", "waiting_input") else "paused",
                "round": request["round"], "repair_id": request["plan"]["sha256"],
                "source_queue_sha256": record["source_queue_sha256"],
                "total": request["target"], "unit": "개", "label": "부적합 목소리 호출어 다시 생성",
                "family_count": len(request["affected_families"]), "family_labels": record.get("family_labels", []),
                "retained_count": request["carried_review_count"], "recheck_count": request["pending_review_count"]}
        try:
            progress = postgeneration_repair_progress(state["project"], request)
            view.update(done=progress["done"], phase=progress.get("phase"))
        except (OSError, ValueError, KeyError) as error:
            view.update(done=None, state="failed", error=str(error))
        return view

    def retry_postgeneration(self, project_id, payload=None):
        """Recover the already saved human rejection without re-entering answers."""
        with self._lock:
            state = self._load(project_id)
            self._guard_config(state)
            if (state["index"] >= len(state["plan"]) or state["plan"][state["index"]]["id"] != "positive_review"
                    or state["settings"].get("pilot_review_policy") != PREPOST_PILOT_POLICY):
                raise ValueError("현재 호출어 발음 검수에서만 다시 생성할 수 있습니다.")
            rounds = state.get("inputs", {}).get("postgeneration_repair_rounds", [])
            if rounds and state["status"] in ("running", "pausing", "paused", "interrupted", "error"):
                expected = (payload or {}).get("queue_sha256")
                if expected and expected != rounds[-1].get("source_queue_sha256"):
                    raise ValueError("다시 생성할 호출어 목록이 바뀌었습니다.")
                if self._job_alive(state.get("job") or {}):
                    return {"state": "postgeneration_repair_started", "review_finalized": True, "workflow_action": "repair"}
                return self._resume_postgeneration_repair(state)
            state, descriptor, rows = self.review_queue(project_id, "positive")
            if (payload or {}).get("queue_sha256", descriptor["queue_sha256"]) != descriptor["queue_sha256"]:
                raise ValueError("다시 생성할 호출어 목록이 바뀌었습니다.")
            root = Path(state["project"]).parent
            report = read_json(root / "review/positive_results.json")
            decisions = report.get("decisions", {})
            if (report.get("kind") != "positive_listening_review" or report.get("queue_sha256") != descriptor["queue_sha256"]
                    or report.get("source_scope") != "Human dispositions entered through the local review screen"
                    or not report.get("saved_at") or set(decisions) != {row["sample_id"] for row in rows}
                    or report.get("queued_audio_sha256") != {row["sample_id"]: row["sha256"] for row in rows}
                    or any(not isinstance(answer, dict) or answer.get("decision") not in ("keep", "discard") or answer.get("listened") is not True for answer in decisions.values())
                    or not any(answer["decision"] == "discard" for answer in decisions.values())):
                raise ValueError("실제로 저장한 부적합 판정과 현재 호출어 목록이 일치하지 않습니다.")
            archived = root / "review/evidence" / f"positive_review_{identity(report)}.json"
            if not archived.is_file() or read_json(archived) != report:
                raise ValueError("저장된 부적합 판정의 원본 근거가 없습니다.")
            return self._start_postgeneration_repair(state, rows,
                {"path": archived.relative_to(root).as_posix(), "sha256": digest(archived)})

    def save_review(self, project_id, queue_id, payload):
        from .review import PLAY_STARTED_POLICY, apply_review
        with self._lock:
            state, review, rows = self.review_queue(project_id, queue_id)
            call_phase = self._call_review_phase(state, review)
            if call_phase and payload.get("queue_sha256") != review["queue_sha256"]:
                raise ValueError("검수할 음원이 바뀌었습니다. 화면을 새로고침하세요.")
            if call_phase:
                self._guard_config(state)
            playback_policy = None
            if payload.get("action") in ("autosave", "finalize"):
                if call_phase != "pre":
                    raise ValueError("자동 저장은 호출어 사전 검수에서만 사용할 수 있습니다.")
                self._guard_config(state)
                if payload["action"] == "autosave":
                    return self._save_preflight_draft(state, review, rows, payload)
                draft = self._preflight_draft(state, review, rows)
                if type(payload.get("revision")) is not int or payload["revision"] != draft["revision"]:
                    raise ValueError("다른 화면에서 판정이 바뀌었습니다. 저장된 초안을 다시 불러오세요.")
                if set(draft["decisions"]) != {row["sample_id"] for row in rows}:
                    raise ValueError("아직 판정하지 않은 호출어가 있습니다. 재생하거나 다시 생성을 선택하세요.")
                payload = {"queue_sha256": review["queue_sha256"], "decisions": draft["decisions"]}
                playback_policy = PLAY_STARTED_POLICY
            evaluation_only = review.get("evaluation_only") is True
            if evaluation_only and (not review.get("pool_manifest") or review.get("pools") != []):
                raise ValueError("평가 전용 검수 자료는 학습 풀에 연결할 수 없습니다.")
            decisions = payload.get("decisions", payload)
            sample_first = review.get("synthetic_review_policy") == HARD_SAMPLE_REVIEW_POLICY
            allowed = ("keep", "discard") if call_phase else ("keep", "discard", "review")
            if (not isinstance(decisions, dict) or set(decisions) != {r["sample_id"] for r in rows}
                    or any(not isinstance(a, dict) or a.get("decision") not in allowed
                           or playback_policy is None and (call_phase or a.get("decision") == "keep" or sample_first and a.get("decision") == "discard")
                           and a.get("listened") is not True for a in decisions.values())):
                raise ValueError("모든 항목을 듣고 보존하거나 폐기·재검수를 표시하세요.")
            if call_phase == "post":
                carried = self._postgeneration_repair_review(state, review).get("carried_decisions", {})
                if any(decisions.get(sample, {}).get("decision") != answer.get("decision")
                       or decisions.get(sample, {}).get("listened") is not True for sample, answer in carried.items()):
                    raise ValueError("기존 적합 판정은 보존됩니다. 새로 만든 호출어만 다시 검수하세요.")
            if review.get("pool_manifest"):
                if payload.get("policy_approved") is not True:
                    raise ValueError("자료 묶음 사용 동의에 체크한 뒤 저장하세요.")
                for row in rows:
                    if digest(resolved(Path(state["project"]).parent, row["path"])) != row["sha256"]:
                        raise ValueError("검수 음원이 변경되었습니다.")
                # Public previews needn't repeat their original's provenance.
                # Record real answers now; worker verifies the frozen source
                # pool's provenance, exclusions and technical admission later.
                result = {"state": "review_saved", "kept": sum(a["decision"] == "keep" for a in decisions.values()),
                          "total": len(rows), "admission_pending": True}
                if evaluation_only:
                    result.update(evaluation_only=True, intended_use="held_out_development_noise", pools=[])
            else:
                result = apply_review(state["project"], review["queue"], decisions, review["kind"],
                                      **({"playback_policy": playback_policy} if playback_policy else {}))
            if call_phase == "pre":
                state["inputs"].setdefault("call_pilot_reviews", []).append({
                    "queue": {"path": review["queue"], "sha256": review["queue_sha256"]},
                    "report": result["report"],
                })
                if result.get("policy_approved") is not True:
                    result = self._retry_call_pilot(state, result)
                    return {**result, "review_finalized": True, "workflow_action": "retry"} if playback_policy else result
            if call_phase == "post" and result.get("policy_approved") is not True:
                repair = self._start_postgeneration_repair(state, rows, result["report"])
                return {**result, **repair}
            if (any(a.get("decision") == "review" for a in decisions.values())
                    or (review["kind"] in ("pilot", "positive") and result.get("policy_approved") is not True)):
                state.update(next_action=("본검수에서 제외한 음원이 있어 학습 자료를 확정하지 않았습니다."
                                         if call_phase == "post" else
                                         "재검수 또는 폐기 항목을 확인하세요. 승인 전에는 다음 작업을 시작하지 않습니다."))
                self._save(state)
                return result
            if not result.get("kept", result.get("source_count", 0)):
                state.update(next_action="보존한 음원이 없습니다. 생성 방식이나 자료 선택을 확인하세요.")
                self._save(state)
                return result
            answer = {"kind": "content_listening_review" if review.get("pool_manifest") else review["kind"] + "_listening_review",
                      "wakeword": state["wakeword"], "decisions": decisions, "saved_at": now(),
                      "source_scope": "Human dispositions entered through the local review screen",
                      "queue_sha256": digest(review["queue"]), "representative_queue_sha256": digest(review["queue"]),
                      "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in rows},
                      "whole_pool_policy_approved": payload.get("policy_approved") is True}
            if playback_policy:
                answer.update(autosave_policy=playback_policy, draft_revision=draft["revision"])
            if review.get("pool_manifest"):
                answer["source_manifest_sha256"] = digest(review["pool_manifest"])
            if review.get("synthetic_review_policy"):
                answer.update(synthetic_review_policy=review["synthetic_review_policy"],
                              representative_selection_sha256=review["representative_selection"]["sha256"],
                              consent_scope=("Heard one sample per phrase before approving bulk generation and use for kept phrases"
                                             if sample_first else "Heard one representative per phrase; other technical-pass originals are policy-admitted, not individually heard"))
            if evaluation_only:
                answer.update(evaluation_only=True, intended_use="held_out_development_noise", pools=[],
                              consent_scope="Held-out Development noise evaluation only; not for training")
            path = Path(state["project"]).parent / ".studio" / "answers" / (identity(answer) + ".json")
            write_json(path, answer)
            state["inputs"].setdefault("review_answers", {})[queue_id] = {"path": str(path), "sha256": digest(path)}
            state["inputs"].setdefault("review_results", {})[queue_id] = result
            if not self.review_descriptors(state):
                state.update(status="paused", input_submitted=True)
            self._save(state)
            if state["status"] == "paused" and state.get("auto_continue"):
                self.start(project_id)
            if playback_policy:
                current = self._load(project_id)
                return {**result, "review_finalized": True,
                        "workflow_action": "resume" if current["status"] == "paused" and current.get("input_submitted") else "none"}
            return result

    def _retry_call_pilot(self, state, result):
        # The saved queue and decision snapshot remain immutable. The worker
        # uses their bindings to carry approved families into the next round.
        pilot_index = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot")
        for stage in ("pilot", "pilot_review"):
            state["results"].pop(stage, None)
        for key in ("review_answers", "review_results"):
            state["inputs"].get(key, {}).pop("pilot", None)
        checkpoint = state.get("navigation_checkpoint")
        if checkpoint:
            checkpoint.update(step="call-pilot-generation", state="running")
        state.update(index=pilot_index, status="paused", auto_continue=False,
                     next_action="다시 생성할 목소리를 저장했습니다. 이어서 진행하면 해당 목소리만 다시 생성합니다.")
        self._save(state)
        try:
            self.start(state["id"])
        except (OSError, ValueError) as error:
            return {**result, "state": "pilot_retry_pending", "message": str(error)}
        return {**result, "state": "pilot_retry_started"}

    @staticmethod
    def _reference_preview_design(family, row):
        description = row.get("description") or family.get("description", "")
        # Journal values describe the saved audio. Some older journals carry
        # these three traits only in the generation prompt.
        saved = family if not row.get("description") or description == family.get("description") else {}
        attributes = {**saved, **row}
        for axis, pattern in (
            ("gender", r"\b(man|woman|male|female)\b"),
            ("age_group", r"\b(child|teenager|young adult|middle-aged|older adult|adult)\b"),
            ("timbre", r"with a (.*?) timbre\b"),
        ):
            if not attributes.get(axis):
                match = re.search(pattern, description)
                if match:
                    attributes[axis] = match.group(1)
        gender = attributes.get("gender")
        if gender in ("male", "female"):
            attributes["gender"] = "man" if gender == "male" else "woman"
        design = {}
        for axis, labels in _REFERENCE_PREVIEW_TRAIT_LABELS.items():
            value = attributes.get(axis)
            if isinstance(value, str) and value in labels:
                design[axis] = {"value": value, "label": labels[value]}
        return design

    def reference_preview_rows(self, project_id, *, include_design=False):
        state = self._load(project_id)
        if state["settings"].get("reference_selection_policy") == "candidate_pool_v3":
            from .reference_candidates import source_slots
            return state, source_slots(state["project"], state["settings"])
        root = Path(state["project"]).parent
        bank = read_json(root / "voices.json")["families"]
        rows = []
        for split, key in (("train", "train_references"), ("validation", "validation_references")):
            families = [row for row in bank if row["split"] == split][:state["settings"][key]]
            selected = {family["family_id"]: (number, family) for number, family in enumerate(families, 1)}
            for row in read_rows(root / "data/references" / split / "references.jsonl"):
                if row.get("voice_family_id") in selected:
                    if (row.get("split") != split or row.get("provenance") != "qwen_synthetic_reference"
                            or row.get("human_reference_used") is not False):
                        raise ValueError("기준 목소리 출처가 올바르지 않습니다.")
                    if include_design:
                        number, family = selected[row["voice_family_id"]]
                        row = {**row, "voice_number": number, "voice_design": self._reference_preview_design(family, row)}
                    rows.append(row)
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("기준 목소리 목록에 중복 번호가 있습니다.")
        return state, rows

    @staticmethod
    def _reference_review_required(state):
        checkpoint = state.get("navigation_checkpoint") or {}
        return (state.get("reference_voice_review_policy") in ("grouped_v1", "candidate_pool_v2")
                and checkpoint.get("step") == "voices" and checkpoint.get("state") == "awaiting_next"
                and state["status"] == "paused" and state["settings"].get("pilot_review_policy") == PREPOST_PILOT_POLICY
                and state["index"] < len(state["plan"]) and state["plan"][state["index"]]["id"] == "pilot"
                and not (Path(state["project"]).parent / "final_lock.json").exists())

    @staticmethod
    def _reference_review_path(state):
        return Path(state["project"]).parent / ".studio/reference-review/state.json"

    def _stop_reference_worker(self, *, wait=True):
        worker = self._reference_worker
        if worker is None:
            return
        child = worker["child"]
        try:
            child.stdin.close()
        except OSError:
            pass
        if wait:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise ValueError("이전 기준 목소리 작업을 종료하고 있습니다. 잠시 뒤 다시 시도하세요.") from error
        self._reference_worker = None

    def _expire_reference_worker(self):
        worker = self._reference_worker
        if worker is None:
            return
        if worker["child"].poll() is not None:
            self._stop_reference_worker()
        elif worker["result_path"].is_file():
            current = time.monotonic()
            if worker["idle_since"] is None:
                worker["idle_since"] = current
            elif current - worker["idle_since"] >= _REFERENCE_WORKER_IDLE_SECONDS:
                self._stop_reference_worker()

    def _reference_review_running(self, job):
        if (job.get("persistent") and job.get("directory")
                and (Path(job["directory"]) / "result.json").is_file()):
            # An atomic receipt ends this request; the worker can stay alive for the next one.
            # This also holds after the HTTP server restarts and no longer owns
            # the old resident-process handle. The caller verifies the receipt.
            return False
        return self._job_alive(job)

    def _reference_review_job_alive(self, state):
        if self._reference_preparation_job_alive(state):
            return True
        # Both ledgers can coexist after an audited migration. Neither may
        # permit another GPU writer while an outstanding request is live.
        for path in (self._reference_review_path(state), self._reference_candidate_path(state)):
            if path.exists():
                job = read_json(path).get("job") or {}
                if job.get("status") == "running" and self._reference_review_running(job):
                    return True
        return False

    def _reference_review_load(self, state, originals):
        path = self._reference_review_path(state)
        hashes = {row["voice_family_id"]: row["sha256"] for row in originals}
        try:
            saved = read_json(path)
        except FileNotFoundError:
            return {"originals": hashes, "replacements": {}, "heard": {}, "selected": {}, "approved": [], "job": None}
        if saved.get("originals") != hashes:
            raise ValueError("기준 목소리 원본이 기존 검수 기록과 일치하지 않습니다.")
        job = saved.get("job")
        if not job or job["status"] != "running":
            return saved
        if self._reference_review_running(job):
            return saved
        root = Path(state["project"]).parent
        result_path = Path(job["directory"]) / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            if result.get("request_sha256") != job["request_sha256"] or result.get("task_id") != job["task_id"]:
                raise ValueError("기준 목소리 다시 생성 결과가 요청과 일치하지 않습니다.")
            if result.get("error"):
                job.update(status="failed", message=result["error"])
            else:
                replacements = result["replacements"]
                if set(replacements) != set(job["selected"]):
                    raise ValueError("선택한 목소리와 다시 생성한 목소리가 일치하지 않습니다.")
                for family, row in replacements.items():
                    if row["previous_reference_sha256"] != job["selected"][family]:
                        raise ValueError("교체 대상 기준 목소리가 변경되었습니다.")
                    bound_file(root, row)
                saved["replacements"].update(replacements)
                for family in job["families"]:
                    saved["heard"].pop(family, None)
                for family in job["selected"]:
                    saved["selected"].pop(family, None)
                saved["job"] = None
            self._children.pop(job["task_id"], None)
        else:
            self._children.pop(job["task_id"], None)
            job.update(status="failed", message="다시 생성 작업이 중단되었습니다. 선택한 목소리로 다시 시도하세요.")
        write_json(path, saved)
        return saved

    @staticmethod
    def _reference_review_rows(originals, saved):
        groups = {}
        rows = []
        genders = _REFERENCE_PREVIEW_TRAIT_LABELS["gender"]
        ages = _REFERENCE_PREVIEW_TRAIT_LABELS["age_group"]
        for original in originals:
            family, split = original["voice_family_id"], original["split"]
            design = original["voice_design"]
            gender = design.get("gender", {}).get("value", "unknown")
            age = design.get("age_group", {}).get("value", "unknown")
            key = f"{split}:{gender}:{age}"
            group = "학습용" if split == "train" else "공통 평가용"
            labels = [group, genders.get(gender), ages.get(age)]
            if gender == age == "unknown":
                labels.append("생성 설정 정보 없음")
            groups[key] = {"key": key, "label": " · ".join(filter(None, labels)), "approved": key in saved["approved"]}
            row = {**original, **saved["replacements"].get(family, {}), "group_key": key, "group": group}
            row["heard"] = saved["heard"].get(family) == row["sha256"]
            row["decision"] = "compare" if saved["selected"].get(family) == row["sha256"] else None
            rows.append(row)
        order = {value: index for index, value in enumerate((*ages, "unknown"))}
        return rows, sorted(groups.values(), key=lambda group: (group["key"].startswith("validation:"),
                      group["key"].split(":")[1], order[group["key"].rsplit(":", 1)[1]]))

    def _launch_reference_review(self, state, saved, group, rows):
        chosen = [row for row in rows if row["group_key"] == group and row["decision"] == "compare"]
        families = [row["voice_family_id"] for row in rows if row["group_key"] == group]
        if not chosen:
            raise ValueError("다시 생성할 목소리 번호를 선택하세요.")
        for project_id in self.registry["projects"]:
            other = self._load(project_id)
            if (self._job_alive(other.get("job") or {}) or self._reference_review_job_alive(other) or self.analysis.busy(project_id)
                    or other["status"] in ("ready", "running", "pausing") and other.get("auto_continue")):
                raise ValueError("다른 생성 또는 학습 작업이 끝난 뒤 다시 시도하세요.")
        root = Path(state["project"]).parent
        task_id = uuid.uuid4().hex
        directory = root / ".studio/reference-review/jobs" / task_id
        directory.mkdir(parents=True)
        request_path = directory / "request.json"
        hashes = sorted({row[key] for row in rows for key in ("sha256", "decoded_pcm_sha256") if row.get(key)})
        # Keep UI fields out of generation evidence.
        chosen = [{key: value for key, value in row.items() if key not in
                   ("group", "group_key", "heard", "decision", "voice_design", "voice_number")} for row in chosen]
        write_json(request_path, {"task_id": task_id, "project": state["project"], "config_sha256": state["config_sha256"],
                                  "rows": chosen, "all_hashes": hashes, "implementation_id": self._implementation_id})
        saved["job"] = self._dispatch_reference_request(state, request_path, group) | {
            "families": families, "selected": {row["voice_family_id"]: row["sha256"] for row in chosen}}

    def _dispatch_reference_request(self, state, request_path, group, *, before_send=None):
        """Share one resident model between legacy replacement and candidate batches."""
        root, directory = Path(state["project"]).parent, request_path.parent
        task_id = directory.name
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                           KOWW_CACHE_ROOT=state.get("cache_root", str(root / "cache")))
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        key = (state["project"], state["config_sha256"], environment["KOWW_CACHE_ROOT"])
        worker = self._reference_worker
        if worker is not None and (worker["key"] != key or worker["child"].poll() is not None
                                   or worker["child"].stdin.closed):
            self._stop_reference_worker()
            worker = None
        if worker is None:
            log_path = directory / "output.log"
            with log_path.open("ab") as log:
                child = self._popen([sys.executable, "-m", "korean_openwakeword.studio_worker", "--reference-review-worker"],
                                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=root, env=environment,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            worker = self._reference_worker = {"key": key, "child": child, "log": log_path}
        child = worker["child"]
        worker.update(result_path=directory / "result.json", idle_since=None)
        job = {"task_id": task_id, "pid": child.pid, "creation_identity": process_identity(child.pid),
               "directory": str(directory), "request_sha256": digest(request_path), "group": group,
               "status": "running", "persistent": True, "log": str(worker["log"])}
        if before_send is not None:
            before_send(job)
        try:
            child.stdin.write(json.dumps(str(request_path)).encode("ascii") + b"\n")
            child.stdin.flush()
        except OSError as error:
            self._stop_reference_worker()
            raise ValueError("기준 목소리 생성 작업을 시작하지 못했습니다. 다시 시도하세요.") from error
        return job

    @staticmethod
    def _reference_candidate_path(state):
        return Path(state["project"]).parent / ".studio/reference-candidates/state.json"

    @staticmethod
    def _reference_preparation_required(state):
        policy = state.get("reference_preparation_policy") or state["settings"].get("reference_preparation_policy")
        return (policy == "initial_candidate_batches_v1" and state.get("status") != "completed"
                and not state.get("inputs", {}).get("reference_voice_review"))

    def _reference_preparation_job_alive(self, state):
        job = state.get("reference_preparation_job") or {}
        return job.get("status") in ("running", "pausing") and self._job_alive(job)

    def _reference_preparation_view(self, state):
        from .reference_candidates import preparation_status

        root = Path(state["project"]).parent
        required = self._reference_preparation_required(state)
        policy = state.get("reference_preparation_policy") or state["settings"].get("reference_preparation_policy")
        stage = state["plan"][state["index"]]["id"] if state["index"] < len(state["plan"]) else "completed"
        operation = "stage" if state["settings"].get("reference_preparation_policy") == "initial_candidate_batches_v1" and stage in (
            "frontend", "data_download", "data_review", "data_connect", "references", "references_review") else "helper"
        if policy != "initial_candidate_batches_v1":
            return {"policy": None, "required": False, "ready": True, "status": "completed", "operation": operation,
                    "rows": [], "done_groups": 20, "total_groups": 20,
                    "can_start": False, "can_pause": False, "can_resume": False}
        try:
            raw = preparation_status(root, verify=False)
        except FileNotFoundError:
            raw = {}
        helper = state.get("reference_preparation_job") or {}
        stage_active = (stage in ("references", "references_review") and self._job_alive(state.get("job") or {}))
        helper_active = self._reference_preparation_job_alive(state)
        active = stage_active or helper_active
        done_groups = int(raw.get("completed_groups", raw.get("done_groups", 0)))
        receipt = raw.get("receipt")
        if receipt:
            bound_file(root, receipt)
        ready = (done_groups == 20 and not active and raw.get("status") == "completed" and bool(receipt)
                 and (not helper or helper.get("status") == "completed")
                 and not (operation == "stage" and stage in ("references", "references_review")))
        status = "completed" if ready else helper.get("status") if helper else raw.get("status", "pending")
        if active:
            status = "pausing" if helper.get("status") == "pausing" or state.get("status") == "pausing" else "running"
        elif operation == "stage" and stage in ("references", "references_review") and state.get("status") in ("error", "interrupted"):
            status = "failed" if state["status"] == "error" else "interrupted"
        elif (operation == "stage" and stage in ("references", "references_review") and state.get("status") == "paused"
              and ((state.get("last_job") or {}).get("stage") == "references" or stage == "references_review")):
            status = "paused"
        elif status not in ("paused", "failed", "interrupted", "completed"):
            status = "pending"
        current_group = (raw.get("current") or {}).get("group", "")
        values = {row["split"]: row for row in raw.get("rows", [])}
        rows = []
        for split, label in (("train", "학습용 목소리 후보"), ("validation", "공통 평가용 목소리 후보")):
            entry = values.get(split, {})
            done = min(50, max(0, int(entry.get("completed", entry.get("done", 0)))))
            rows.append({"split": split, "label": label, "done": done, "total": 50,
                         "active": active and current_group.startswith(split + ":"), "completed": done == 50})
        eligible = self._reference_review_required(state)
        error = helper.get("error")
        return {"policy": policy, "required": required, "ready": ready, "status": status, "operation": operation, "rows": rows,
                "done_groups": done_groups, "total_groups": 20,
                "reused_candidates": raw.get("reused_candidates", 0),
                "generated_candidates": raw.get("generated_candidates", 0),
                "message": error or raw.get("message", ""), "error": error,
                "can_start": not ready and not active and ((operation == "helper" and eligible and not helper)
                             or operation == "stage" and stage in ("references", "references_review") and status == "pending"),
                "can_pause": active, "can_resume": operation == "helper" and eligible and not ready and not active
                and status in ("paused", "failed", "interrupted") or operation == "stage" and not ready and not active
                and stage in ("references", "references_review") and status in ("paused", "failed", "interrupted")}

    def _finish_reference_preparation(self, state):
        job = state.get("reference_preparation_job") or {}
        if job.get("status") not in ("running", "pausing") or self._job_alive(job):
            return
        directory = Path(job["directory"])
        result_path = directory / "result.json"
        if not result_path.exists():
            job.update(status="interrupted", error="후보 준비가 중단되었습니다. 재개하면 저장된 후보부터 이어갑니다.")
        else:
            result = read_json(result_path)
            if (digest(directory / "request.json") != job["request_sha256"]
                    or result.get("task_id") != job["task_id"] or result.get("request_sha256") != job["request_sha256"]
                    or result.get("config_sha256") != state["config_sha256"]):
                raise ValueError("기준 목소리 준비 결과가 저장한 실행 요청과 다릅니다.")
            if result.get("error"):
                job.update(status="failed", error=result["error"])
            else:
                prepared = result["result"]
                if prepared.get("status") == "completed":
                    binding = prepared["receipt"]
                    bound_file(Path(state["project"]).parent, binding)
                    state["inputs"]["reference_candidate_preparation"] = binding
                    job.update(status="completed", error=None)
                elif prepared.get("status") == "paused":
                    job.update(status="paused", error=None)
                else:
                    raise ValueError("기준 목소리 준비 작업의 종료 상태가 다릅니다.")
        self._children.pop(job["task_id"], None)
        self._save(state)

    def _recover_reference_preparation_outputs(self, state, prior):
        """Recover only a proved interrupted output, including its deeper lock."""
        root = Path(state["project"]).parent.resolve()
        ledger_path = self._reference_candidate_path(state)
        saved = read_json(ledger_path) if ledger_path.exists() else {}
        candidates = []
        owners = {prior.get("pid"), self._worker_owner(prior).get("pid")} if prior else set()
        if prior:
            candidates.append((Path(prior["directory"]) / ".writer.lock", owners))
        manual = saved.get("job") or {}
        manual_request = None
        if manual.get("task_id") and manual.get("request_sha256"):
            directory = (root / ".studio/reference-candidates/jobs" / manual["task_id"]).resolve()
            request_path = directory / "request.json"
            if (directory != Path(manual["directory"]).resolve() or digest(request_path) != manual["request_sha256"]):
                raise ValueError("기존 후보 작업의 재개 근거가 변경되었습니다.")
            manual_request = read_json(request_path)
            if manual_request.get("project") != state["project"] or manual_request.get("config_sha256") != state["config_sha256"]:
                raise ValueError("기존 후보 작업의 모델 설정이 다릅니다.")
            manual_owners = {manual.get("pid"), self._worker_owner(manual).get("pid")}
            candidates.extend(((directory / ".writer.lock", manual_owners), (directory / "audio/.writer.lock", manual_owners)))
        phase_path = root / ".studio/reference-candidates/preparation/state.json"
        current = read_json(phase_path).get("current") if phase_path.exists() else None
        if current:
            output = resolved(root, current["output"])
            plan = next((plan for plan in saved.get("batches", {}).get(current["group"], [])
                         if plan["batch_id"] == current["batch_id"]), None)
            jobs = root / ".studio/reference-candidates/jobs"
            if (plan is None or output.name != "audio" or not output.is_relative_to(jobs)
                    or output.parent.parent != jobs):
                raise ValueError("준비 중단 위치가 등록된 후보 생성 계획과 다릅니다.")
            plan_path = output / "plan.json"
            expected_new = output.parent.name == "initial_" + plan["batch_id"]
            expected_manual = manual_request is not None and manual_request.get("plan") == plan and output.parent == Path(manual["directory"]).resolve()
            if (plan_path.exists() and read_json(plan_path) != plan) or not (expected_new or expected_manual):
                raise ValueError("중단된 후보 출력의 실행 계획이 다릅니다.")
            candidates.append((output / ".writer.lock", owners if expected_new else owners | manual_owners))
        targets = {}
        for path, allowed in candidates:
            targets.setdefault(path.resolve(), set()).update(allowed)
        recover = {}
        for path, allowed in targets.items():
            if not path.exists():
                continue
            path = path.resolve()
            record = read_json(path)
            pid = record.get("pid")
            if not path.is_relative_to(root) or not pid or pid not in allowed or process_identity(pid) is not None:
                raise ValueError("후보 출력 잠금의 소유자가 살아 있거나 확인되지 않아 재개할 수 없습니다.")
            recover[path] = record
        for path, record in recover.items():
            archive = path.with_name("recovered-lock-" + uuid.uuid4().hex + ".json")
            write_json(archive, {"original": str(path), "record": record, "job": prior, "recovered_at": now()})
            path.unlink()

    def prepare_start(self, project_id):
        with self._lock:
            state = self._load(project_id)
            self._guard_config(state)
            self._finish_reference_preparation(state)
            view = self._reference_preparation_view(state)
            if view["ready"]:
                return self.describe(state)
            if not view["required"] or view["operation"] != "helper" or not self._reference_review_required(state):
                raise ValueError("현재 단계에서는 기준 목소리 후보 준비를 시작할 수 없습니다.")
            if self._reference_preparation_job_alive(state):
                return self.describe(state)
            for other_id in self.registry["projects"]:
                other = self._load(other_id)
                if (self._job_alive(other.get("job") or {}) or self._reference_review_job_alive(other) or self.analysis.busy(other_id)
                        or other["status"] in ("ready", "running", "pausing") and other.get("auto_continue")):
                    raise ValueError("다른 생성 또는 학습 작업이 끝난 뒤 후보 준비를 시작하세요.")
            self._verify_history(state)
            _, originals = self.reference_preview_rows(project_id, include_design=True)
            self._reference_candidate_load(state, originals)  # Reconcile a completed manual request before the exclusive writer.
            self._stop_reference_worker()
            prior = state.get("reference_preparation_job")
            if prior:
                self._recover_dead_locks({**state, "job": prior})
            self._recover_reference_preparation_outputs(state, prior)
            root = Path(state["project"]).parent
            task_id = uuid.uuid4().hex
            directory = root / ".studio/reference-candidates/preparation/jobs" / task_id
            directory.mkdir(parents=True)
            request_path = directory / "request.json"
            write_json(request_path, {"task_id": task_id, "kind": "reference_initial_candidate_preparation_v1",
                "project": state["project"], "config_sha256": state["config_sha256"],
                "implementation_id": self._implementation_id, "originals": originals,
                "migrations": state.get("migrations", []),
                "pause_path": str(directory / "pause.json")})
            log_path = directory / "output.log"
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                               KOWW_CACHE_ROOT=state.get("cache_root", str(root / "cache")))
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment.get("PYTHONPATH", "")
            with log_path.open("ab") as log:
                child = self._popen([sys.executable, "-m", "korean_openwakeword.studio_worker", "--reference-preparation", str(request_path)],
                    stdout=log, stderr=subprocess.STDOUT, cwd=root, env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._children[task_id] = child
            state["reference_preparation_job"] = {"task_id": task_id, "pid": child.pid,
                "creation_identity": process_identity(child.pid), "directory": str(directory),
                "request_sha256": digest(request_path), "status": "running", "log": str(log_path)}
            self._save(state)
            return self.describe(state)

    def prepare_resume(self, project_id):
        return self.prepare_start(project_id)

    def prepare_pause(self, project_id):
        with self._lock:
            state = self._load(project_id)
            job = state.get("reference_preparation_job") or {}
            if self._reference_preparation_job_alive(state):
                write_json(Path(job["directory"]) / "pause.json", {"task_id": job["task_id"], "requested_at": now()})
                job["status"] = "pausing"
                self._save(state)
            else:
                self._finish_reference_preparation(state)
            return self.describe(state)

    def _reference_candidate_load(self, state, originals):
        from .reference_candidates import create_candidate_review, ingest_candidate_batch, validate_candidate_review

        root, path = Path(state["project"]).parent, self._reference_candidate_path(state)
        try:
            saved = read_json(path)
        except FileNotFoundError:
            return create_candidate_review(originals)
        validate_candidate_review(saved, originals)
        if self._reference_preparation_job_alive(state):
            return saved  # The preparation worker is the exclusive ledger writer.
        job = saved.get("job") or {}
        if job.get("status") != "running" or self._reference_review_running(job):
            return saved
        directory = (root / ".studio/reference-candidates/jobs" / job["task_id"]).resolve()
        if directory != Path(job["directory"]).resolve():
            raise ValueError("기준 목소리 후보 작업의 저장 위치가 다릅니다.")
        try:
            request_path = directory / "request.json"
            if digest(request_path) != job["request_sha256"]:
                raise ValueError("기준 목소리 후보 생성 요청이 변경되었습니다.")
            request = read_json(request_path)
            result = read_json(directory / "result.json")
            if (result.get("request_sha256") != job["request_sha256"]
                    or result.get("task_id") != job["task_id"]
                    or request.get("project") != state["project"]
                    or request.get("config_sha256") != state["config_sha256"]):
                raise ValueError("기준 목소리 후보 생성 결과가 요청과 일치하지 않습니다.")
            if result.get("error"):
                # A finalized failure may have exhausted all deterministic QC
                # attempts. Draw a fresh plan on the next explicit More action.
                job.update(status="failed", resumable=False, message=str(result["error"]))
            else:
                ingest_candidate_batch(root, saved, request["plan"], result["candidates"])
                saved["job"] = None
        except FileNotFoundError:
            job.update(status="failed", resumable=True,
                       message="후보 생성이 중단되었습니다. 후보 더 뽑기로 남은 후보를 이어서 생성하세요.")
        self._children.pop(job["task_id"], None)
        write_json(path, saved)
        return saved

    def _launch_reference_candidates(self, state, saved, group, originals):
        from .reference_candidates import plan_candidate_batch, register_candidate_batch

        for project_id in self.registry["projects"]:
            other = self._load(project_id)
            if (self._job_alive(other.get("job") or {}) or self._reference_review_job_alive(other) or self.analysis.busy(project_id)
                    or other["status"] in ("ready", "running", "pausing") and other.get("auto_continue")):
                raise ValueError("다른 생성 또는 학습 작업이 끝난 뒤 다시 시도하세요.")
        root = Path(state["project"]).parent
        prior = saved.get("job") or {}
        if prior.get("status") == "failed" and prior.get("resumable") and prior.get("group") == group:
            directory = (root / ".studio/reference-candidates/jobs" / prior["task_id"]).resolve()
            request_path = directory / "request.json"
            if (Path(prior["directory"]).resolve() != directory or digest(request_path) != prior["request_sha256"]):
                raise ValueError("재개할 후보 생성 요청이 변경되었습니다.")
            request = read_json(request_path)
            if (request.get("project") != state["project"] or request.get("config_sha256") != state["config_sha256"]
                    or request.get("implementation_id") != self._implementation_id
                    or request.get("kind") != "reference_candidate_batch_v2"
                    or request["plan"]["group_key"] != group):
                raise ValueError("저장한 후보 생성 조건이 달라 재개할 수 없습니다.")
            # Recover only locks owned by this exact, confirmed-dead request.
            # Keep their contents as evidence instead of deleting unknown locks.
            owner = read_json(directory / "owner.json") if (directory / "owner.json").exists() else {}
            locks = list(directory.rglob(".writer.lock"))
            for path in locks:
                lock = read_json(path)
                if (lock.get("pid") not in {prior.get("pid"), owner.get("pid")}
                        or not lock.get("pid") or process_identity(lock["pid"]) is not None):
                    raise ValueError("후보 생성 잠금의 소유자가 살아 있거나 확인되지 않아 재개할 수 없습니다.")
            for path in locks:
                path.replace(path.with_name("recovered-lock-" + uuid.uuid4().hex + ".json"))
            result_path = directory / "result.json"
            if result_path.exists():
                result_path.replace(directory / ("failed-result-" + uuid.uuid4().hex + ".json"))
            def persist_retry(job):
                saved["job"] = job
                write_json(self._reference_candidate_path(state), saved)
            self._dispatch_reference_request(state, request_path, group, before_send=persist_retry)
            return
        plan = plan_candidate_batch(state["project"], saved, group)
        task_id = uuid.uuid4().hex
        directory = root / ".studio/reference-candidates/jobs" / task_id
        directory.mkdir(parents=True)
        request_path = directory / "request.json"
        hashes = {row[key] for row in originals for key in ("sha256", "decoded_pcm_sha256") if row.get(key)}
        # Include every prior candidate, including unselected batches, so no
        # train/evaluation or previously auditioned duplicate can be admitted.
        def collect(value):
            if isinstance(value, dict):
                for key, entry in value.items():
                    if key in ("sha256", "decoded_pcm_sha256") and isinstance(entry, str):
                        hashes.add(entry)
                    elif isinstance(entry, (list, dict)):
                        collect(entry)
            elif isinstance(value, list):
                for entry in value:
                    collect(entry)
        collect(saved)
        write_json(request_path, {"task_id": task_id, "kind": "reference_candidate_batch_v2", "plan": plan,
                                  "project": state["project"], "config_sha256": state["config_sha256"],
                                  "all_hashes": sorted(hashes), "implementation_id": self._implementation_id})
        register_candidate_batch(saved, plan)
        def persist(job):
            saved["job"] = job
            write_json(self._reference_candidate_path(state), saved)
        try:
            self._dispatch_reference_request(state, request_path, group, before_send=persist)
        except (OSError, ValueError):
            saved["job"] = {"status": "failed", "group": group,
                            "message": "기준 목소리 후보 생성을 시작하지 못했습니다. 다시 시도하세요."}
            write_json(self._reference_candidate_path(state), saved)
            raise

    @staticmethod
    def _reference_candidate_settings(row):
        """Describe saved option IDs without modifying generation evidence."""
        from .reference_candidates import OPTIONS

        age_key = row.get("age_group")
        age = OPTIONS["ages"].get(age_key) if isinstance(age_key, str) else None
        controls = row.get("voice_controls")
        if not age or not isinstance(controls, dict):
            return []
        settings = []
        for field in OPTIONS["fields"]:
            option = next((option for option in age["controls"].get(field["id"], [])
                           if option["id"] == controls.get(field["id"])), None)
            settings.append({"id": field["id"], "label": field["label"],
                             "value": option["label"] if option else ""})
        return settings

    def _freeze_reference_candidate_selection(self, state, saved, originals):
        from .reference_candidates import freeze_candidate_review
        from .reference_voice_review import overlay_reference_banks

        root = Path(state["project"]).parent
        if state["settings"].get("reference_selection_policy") != "candidate_pool_v3":
            from .reference_qc import admitted_reference_rows
            if admitted_reference_rows(root, read_json(root / "approval.json")) != saved["originals"]:
                raise ValueError("자동 검사를 통과한 원본 목소리 목록과 일치하지 않습니다.")
        manifest = freeze_candidate_review(root, saved, originals)
        banks = {split: [row for row in originals if row["split"] == split] for split in ("train", "validation")}
        frozen = snapshot_json(root, manifest, "reference_voice_review")
        overlay_reference_banks(root, banks, frozen)
        state["inputs"]["reference_voice_review"] = frozen
        self._save(state)
        return frozen

    @staticmethod
    def _reference_candidate_batches(saved):
        """List finished batches in this group without changing its latest batch."""
        from .reference_candidates import active_candidate_group, BATCH_SIZE

        group = active_candidate_group(saved)
        batches, rows = [], {}
        for plan in saved["batches"].get(group, []):
            candidate_ids = [item["candidate_id"] for item in plan["candidates"]]
            if (len(candidate_ids) != BATCH_SIZE or len(set(candidate_ids)) != BATCH_SIZE
                    or any(cid not in saved["candidates"] for cid in candidate_ids)):
                continue
            items = [saved["candidates"][cid] for cid in candidate_ids]
            if any(row["batch_id"] != plan["batch_id"] or row["group_key"] != group for row in items):
                continue
            batches.append({"id": plan["batch_id"], "number": len(batches) + 1, "count": BATCH_SIZE})
            rows[plan["batch_id"]] = items
        return group, saved["current_batch"].get(group), batches, rows

    def _reference_candidate_queue(self, state, originals, payload, *, batch_id=None):
        from .reference_candidates import candidate_action, candidate_review_view

        root = Path(state["project"]).parent
        saved = self._reference_candidate_load(state, originals)
        binding = state.get("inputs", {}).get("reference_voice_review")
        complete = bool(binding)
        editable = self._reference_review_required(state) and not complete
        if self._reference_preparation_required(state) and not self._reference_preparation_view(state)["ready"]:
            editable = False
        if binding:
            frozen = read_json(bound_file(root, binding))
            groups = [group["key"] for group in saved["groups"]]
            selected = frozen.get("selections", {})
            if (frozen.get("kind") != saved.get("kind") or frozen.get("approved_groups") != groups
                    or set(selected) != set(groups)
                    or any(cid not in saved["candidates"] for ids in selected.values() for cid in ids)):
                raise ValueError("확정한 기준 목소리와 후보 검수 기록의 연결이 다릅니다.")
            # The hash-bound frozen manifest is authoritative if an old server
            # stopped after saving its workflow input but before its last ledger
            # write. Hydrate only this response; never replay generation/approval.
            saved = {**saved, "approved": list(frozen["approved_groups"]),
                     "selected": selected, "heard": frozen["heard"]}
        if batch_id is not None:
            _, _, _, history = self._reference_candidate_batches(saved)
            if not isinstance(batch_id, str) or batch_id not in history:
                raise ValueError("현재 분류에서 생성이 끝난 목소리 후보 묶음을 선택하세요.")
            if payload is not None and payload.get("action") not in ("hear", "toggle"):
                raise ValueError("후보 묶음 보기 설정은 청취·선택 요청에만 함께 사용할 수 있습니다.")
        if payload is None and editable and candidate_review_view(root, saved)["review_complete"]:
            # All human approvals are already durable. A server interruption
            # between publication and the workflow binding can safely repeat
            # this idempotent, file-only finalization; it never loads a model.
            self._guard_config(state)
            self._freeze_reference_candidate_selection(state, saved, originals)
            complete, editable = True, False
        if payload is not None:
            # Guard every mutation, not just GPU work: an out-of-date server
            # cannot save judgments against a different computational contract.
            self._guard_config(state)
            if not editable or (saved.get("job") or {}).get("status") == "running":
                raise ValueError("현재 기준 목소리 후보 검수를 변경할 수 없습니다.")
            action = payload.get("action")
            if action == "more" and set(payload) == {"action", "group"}:
                self._launch_reference_candidates(state, saved, payload["group"], originals)
            elif action in ("hear", "toggle") and set(payload) == {"action", "id", "sha256"}:
                candidate_action(root, saved, action, candidate_id=payload["id"], sha256=payload["sha256"])
            elif action == "approve" and set(payload) == {"action", "group"}:
                candidate_action(root, saved, action, group_key=payload["group"])
                failed = saved.get("job") or {}
                if failed.get("status") == "failed" and failed.get("group") == payload["group"]:
                    # Selecting enough older candidates can finish a group even
                    # after a newer draw failed. Its stored request/result stay
                    # intact; that failure must not block the next group.
                    saved["job"] = None
                if candidate_review_view(root, saved)["review_complete"]:
                    # Commit the user's last approval before publication. The
                    # read reconciliation above handles an interrupted freeze.
                    write_json(self._reference_candidate_path(state), saved)
                    self._freeze_reference_candidate_selection(state, saved, originals)
                    complete, editable = True, False
            else:
                raise ValueError("기준 목소리 후보 검수 요청을 확인하세요.")
            write_json(self._reference_candidate_path(state), saved)
        view = candidate_review_view(root, saved)
        group, current_batch, batches, history = self._reference_candidate_batches(saved)
        if (batch_id is None and not view["items"] and batches
                and self._reference_preparation_required(state)
                and self._reference_preparation_view(state)["ready"]
                and (saved.get("job") or {}).get("status") != "running"):
            # Preparation preserves older pointers, including a failed initial
            # attempt. Show a proved finished batch without rewriting that pointer.
            batch_id = batches[0]["id"]
        if batch_id is not None:
            selected = set(saved["selected"].get(group, []))
            view["items"] = [{**row, "id": row["candidate_id"],
                              "selected": row["candidate_id"] in selected,
                              "heard": saved["heard"].get(row["candidate_id"]) == row["sha256"]}
                             for row in history[batch_id]]
        view.update(batches=batches, current_batch=current_batch,
                    view_batch=batch_id if batch_id is not None else current_batch)
        for name in ("items", "accepted"):
            for number, row in enumerate(view[name], 1):
                row["id"] = row.get("candidate_id", row.get("id"))
                row.setdefault("number", number)
                row["voice_settings"] = self._reference_candidate_settings(row)
                row["audio"] = (f"/reference-preview/candidate-audio/{quote(row['id'], safe='')}"
                                f"?project={quote(state['id'])}&rev={row['sha256']}")
        job = saved.get("job")
        public_job = {key: job[key] for key in ("group", "status", "message") if key in job} if job else None
        if job and job.get("status") == "running":
            progress_path = Path(job["directory"]) / "progress.json"
            try:
                progress = read_json(progress_path)
            except (OSError, ValueError):
                progress = {}
            public_job.update({key: progress[key] for key in ("completed", "total", "message") if key in progress})
        if (not public_job and not editable and not complete
                and state["settings"].get("reference_selection_policy") == "candidate_pool_v3"):
            # The merged chapter can open while its two metadata-only stages
            # advance. Keep the iframe polling without issuing a GPU request.
            preparing = {"phase": "preparing", "completed": 0, "total": 5}
            if state.get("implementation_id") != self._implementation_id:
                public_job = {**preparing, "status": "failed", "message": "업데이트를 적용한 뒤 목소리 분류 준비를 계속하세요."}
            elif state.get("status") in ("error", "interrupted", "waiting_input", "waiting_review", "completed"):
                public_job = {**preparing, "status": "failed", "message": "목소리 분류 준비가 중단되었습니다. 작업 상태를 확인하고 재개하세요."}
            else:
                public_job = {**preparing, "status": "running", "message": "목소리 분류를 준비하고 있습니다."}
        view.update(editable=editable, review_complete=complete, job=public_job)
        return view

    def reference_candidate_audio(self, project_id, candidate_id, revision):
        from .reference_candidates import candidate_audio
        with self._lock:
            state, originals = self.reference_preview_rows(project_id, include_design=True)
            if state.get("reference_voice_review_policy") != "candidate_pool_v2":
                raise ValueError("이 모델은 기준 목소리 후보 검수를 사용하지 않습니다.")
            saved = self._reference_candidate_load(state, originals)
            return candidate_audio(Path(state["project"]).parent, saved, candidate_id, revision)

    def reference_preview_queue(self, project_id, payload=None, *, batch_id=None):
        from .reference_voice_review import REVIEW_KIND, overlay_reference_banks

        with self._lock:
            state, originals = self.reference_preview_rows(project_id, include_design=True)
            if state.get("reference_voice_review_policy") == "candidate_pool_v2":
                return self._reference_candidate_queue(state, originals, payload, batch_id=batch_id)
            if batch_id is not None:
                raise ValueError("이 모델은 기준 목소리 후보 묶음 이동을 지원하지 않습니다.")
            root = Path(state["project"]).parent
            saved = self._reference_review_load(state, originals)
            binding = state["inputs"].get("reference_voice_review")
            complete = bool(binding)
            if binding:
                frozen = read_json(bound_file(root, binding))
                saved.update(replacements=frozen["replacements"], heard=frozen["heard"], approved=frozen["approved_groups"])
            rows, groups = self._reference_review_rows(originals, saved)
            editable = self._reference_review_required(state) and not complete
            if payload is not None:
                active = next((group["key"] for group in groups if not group["approved"]), None)
                action = payload.get("action")
                job = saved.get("job") or {}
                if not editable or job.get("status") == "running" or active is None:
                    raise ValueError("현재 묶음의 검수를 변경할 수 없습니다.")
                if action in ("hear", "toggle") and set(payload) == {"action", "id", "sha256"}:
                    row = next((row for row in rows if row["sample_id"] == payload["id"] and row["group_key"] == active), None)
                    if row is None or row["sha256"] != payload["sha256"]:
                        raise ValueError("기준 목소리 음원이 바뀌었습니다. 목록을 다시 불러오세요.")
                    bound_file(root, row)
                    family = row["voice_family_id"]
                    if action == "hear":
                        saved["heard"][family] = row["sha256"]
                    elif row["heard"]:
                        if row["decision"]:
                            saved["selected"].pop(family, None)
                        else:
                            saved["selected"][family] = row["sha256"]
                    else:
                        raise ValueError("음성을 끝까지 들은 뒤 교체할 목소리를 선택하세요.")
                elif action in ("regenerate", "approve") and set(payload) == {"action", "group"} and payload["group"] == active:
                    if not all(row["heard"] for row in rows if row["group_key"] == active):
                        raise ValueError("이 묶음의 모든 목소리를 끝까지 들어 주세요.")
                    if action == "regenerate":
                        self._guard_config(state)
                        self._launch_reference_review(state, saved, active, rows)
                    else:
                        if saved["selected"]:
                            raise ValueError("교체로 표시한 목소리를 다시 생성한 뒤 묶음을 확정하세요.")
                        if len(saved["approved"]) + 1 == len(groups):
                            from .reference_qc import admitted_reference_rows
                            if admitted_reference_rows(root, read_json(root / "approval.json")) != saved["originals"]:
                                raise ValueError("자동 검사를 통과한 기준 목소리 목록과 일치하지 않습니다.")
                            manifest = {"kind": REVIEW_KIND, "originals": saved["originals"],
                                        "replacements": saved["replacements"], "heard": saved["heard"],
                                        "approved_groups": [*saved["approved"], active]}
                            binding = snapshot_json(root, manifest, "reference_voice_review")
                            banks = {split: [row for row in originals if row["split"] == split] for split in ("train", "validation")}
                            overlay_reference_banks(root, banks, binding)
                            state["inputs"]["reference_voice_review"] = binding
                            self._save(state)
                            complete, editable = True, False
                        saved["approved"].append(active)
                else:
                    raise ValueError("기준 목소리 검수 요청을 확인하세요.")
                write_json(self._reference_review_path(state), saved)
                rows, groups = self._reference_review_rows(originals, saved)
            job = saved.get("job")
            public_job = {key: job[key] for key in ("group", "status", "message") if key in job} if job else None
            items = [{key: row[key] for key in ("sha256", "group", "group_key", "voice_number", "voice_design", "heard", "decision")}
                     | {"id": row["sample_id"], "text": row.get("text", ""),
                        "audio": f"/reference-preview/audio/{index}?project={quote(project_id)}&rev={row['sha256']}"}
                     for index, row in enumerate(rows)]
            return {"items": items, "groups": groups, "editable": editable, "review_complete": complete, "job": public_job}

    def reference_preview_audio(self, project_id, index, revision):
        with self._lock:
            state, originals = self.reference_preview_rows(project_id, include_design=True)
            saved = self._reference_review_load(state, originals)
            binding = state["inputs"].get("reference_voice_review")
            if binding:
                saved["replacements"] = read_json(bound_file(Path(state["project"]).parent, binding))["replacements"]
            rows, _ = self._reference_review_rows(originals, saved)
            if not 0 <= index < len(rows) or revision not in (None, rows[index]["sha256"]):
                raise ValueError("기준 목소리가 바뀌었습니다. 목록을 다시 불러오세요.")
            return bound_file(Path(state["project"]).parent, rows[index])

    def hard_preview_rows(self, project_id):
        """Read the frozen full pool without granting consent or recording listening."""
        state = self._load(project_id)
        generated = state.get("results", {}).get("hard_generation", {})
        descriptor = generated.get("result", generated).get("review")
        if not descriptor:
            raise ValueError("준비된 유사어 음원이 없습니다.")
        root = Path(state["project"]).parent
        path = resolved(root, descriptor["pool_manifest"])
        expected = descriptor.get("pool_sha256")
        if expected is None:
            # Older generation receipts bound the full pool in the following
            # review gate, rather than inside the generation result itself.
            for review in state.get("results", {}).get("hard_review", {}).get("reviews", []):
                if resolved(root, review.get("pool_manifest", "")) == path:
                    expected = review.get("pool_sha256")
            answer = state.get("inputs", {}).get("review_answers", {}).get("hard")
            if expected is None and answer:
                from .common import bound_file
                expected = read_json(bound_file(root, answer)).get("source_manifest_sha256")
        if not path.is_relative_to(root) or not expected or digest(path) != expected:
            raise ValueError("유사어 전체 음원 목록이 변경되었습니다.")
        rows = read_rows(path)
        if not rows or any(row.get("split") != "train" or row.get("provenance") != "qwen_tts"
                           or row.get("technical_pass") is not True for row in rows):
            raise ValueError("유사어 전체 듣기 목록의 출처가 올바르지 않습니다.")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("유사어 전체 듣기 목록에 중복 번호가 있습니다.")
        return state, rows

    def positive_preview_rows(self, project_id, start, limit):
        """Read a bounded slice from the original, hash-bound generation manifests."""
        state = self._load(project_id)
        result = self._effective_positive_generated(state, presentation=True)
        completed = next((entry for entry in reversed(state.get("history", [])) if entry["stage"] == "positives"), None)
        if result is None or not completed or not completed.get("sha256") or start < 0:
            raise ValueError("준비된 호출어 음원이 없습니다.")
        receipt = self._history_read(state, completed["result"], sha256=completed["sha256"])
        if receipt.get("result", {}).get("state") != "completed":
            raise ValueError("호출어 생성 완료 기록이 없습니다.")
        root = Path(state["project"]).parent.resolve()
        sources = result.get("preview_sources")
        artifacts = {resolved(root, entry["path"]): entry["sha256"] for entry in receipt["artifact_bindings"]}
        if result.get("postgeneration_repair"):
            artifacts.update({resolved(root, binding["path"]): binding["sha256"] for binding in sources.values()})
        rows, total = [], 0
        for split in ("train", "validation"):
            binding = sources[split] if sources else {"path": f"data/positive/{split}.jsonl"}
            path = resolved(root, binding["path"])
            expected = artifacts.get(path)
            if not path.is_relative_to(root) or not expected or binding.get("sha256", expected) != expected:
                raise ValueError("호출어 전체 음원 목록의 저장 근거가 다릅니다.")
            with path.open("rb") as handle:
                info = os.fstat(handle.fileno())
                stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, expected)
                cached = self._positive_preview_indexes.get(path)
                if cached and cached[0] == stamp:
                    offsets = cached[1]
                else:
                    offsets, cursor = array("Q"), 0
                    hasher = hashlib.sha256()
                    for line in handle:
                        hasher.update(line)
                        if line.strip():
                            offsets.append(cursor)
                        cursor += len(line)
                    if hasher.hexdigest() != expected:
                        raise ValueError("호출어 전체 음원 목록이 변경되었습니다.")
                    self._positive_preview_indexes[path] = (stamp, offsets)
                for index in range(max(0, start - total), min(len(offsets), start + limit - total)):
                    handle.seek(offsets[index])
                    row = json.loads(handle.readline().decode("utf-8-sig"))
                    if (row.get("split") != split or row.get("text") != state["wakeword"]
                            or row.get("provenance") != "qwen_tts" or row.get("qc", {}).get("passed") is not True):
                        raise ValueError("호출어 전체 듣기 음원의 출처가 올바르지 않습니다.")
                    rows.append((total + index, index + 1, row))
                total += len(offsets)
        if start >= total:
            raise ValueError("잘못된 호출어 음원 번호입니다.")
        return state, rows, total

    @staticmethod
    def _voice_design(bank, settings):
        from .voice_design import summarize_voice_design
        return {"policy": bank.get("policy", "legacy"),
                "counts": {split: summarize_voice_design(
                    [row for row in bank["families"] if row["split"] == split][:settings[key]])
                    for split, key in (("train", "train_references"), ("validation", "validation_references"))},
                "scope": "requested_conditions_not_measured_voices"}

    def _history_read(self, state, path, *, sha256=None, rows=False, max_bytes=2 * 1024**2):
        """Read small historical metadata, never re-hash an audio/archive pool.

        Completed views are presentation only. A stat-bound cache avoids opening
        the same frozen queues and receipts on every status poll; a changed or
        missing binding becomes unavailable rather than invented completion.
        """
        root = Path(state["project"]).parent.resolve()
        path = resolved(root, path)
        allowed = (root, self.workspace, Path(state.get("storage_workspace", self.workspace)).resolve())
        if not any(path.is_relative_to(directory) for directory in allowed):
            raise ValueError("저장된 내역의 경로가 작업 폴더 밖에 있습니다.")
        info = path.stat()
        if not path.is_file() or info.st_size > max_bytes:
            raise ValueError("저장된 세부 내역의 형식이나 크기를 확인하세요.")
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, sha256, rows)
        cached = self._history_metadata.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
        if sha256 is not None and digest(path) != sha256:
            raise ValueError("저장된 세부 내역이 변경되었습니다.")
        value = read_rows(path) if rows else read_json(path)
        self._history_metadata[str(path)] = (stamp, value)
        return value

    def _history_result(self, state, stage):
        completed = {row["id"] for row in state["plan"][:state["index"]]}
        entry = state.get("results", {}).get(stage)
        if stage not in completed or not isinstance(entry, dict) or entry.get("state") != "completed":
            return None
        result = entry.get("result", entry)
        return result if isinstance(result, dict) else None

    def result_audio_reference(self, state, site, name, *, query=False):
        """Resolve a Studio-only scored WAV through immutable result evidence.

        Large reference indexes are parsed once per exact file identity. Only
        the requested WAV is hashed; the frozen evaluation receipt provides
        the report/clip bindings without reopening the complete clip manifest.
        """
        root = Path(state["project"]).parent.resolve()
        site = Path(site).resolve()
        provenance_path = site / "provenance.json"
        if not provenance_path.is_file():
            return None  # Older copied sites do not need reference metadata.

        def metadata(path, sha256=None, max_bytes=2 * 1024**2):
            path = Path(path).resolve()
            def stamp():
                info = path.stat()
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
            before = stamp()
            value = self._history_read(state, path, sha256=sha256, max_bytes=max_bytes)
            if stamp() != before:
                self._history_metadata.pop(str(path), None)
                raise ValueError("Result evidence changed while reading")
            return value

        def completed(stage, location, key):
            for entry in reversed(state.get("history", [])):
                if entry.get("stage") != stage:
                    continue
                receipt = metadata(entry["result"], sha256=entry["sha256"], max_bytes=16 * 1024**2)
                output = receipt.get("result", {})
                payload = output.get("result", output)
                if (receipt.get("error") or output.get("state") != "completed"
                        or not isinstance(payload.get(key), str)
                        or Path(payload[key]).resolve() != location):
                    continue
                bindings = {}
                for binding in receipt.get("artifact_bindings", []):
                    path = resolved(root, binding["path"])
                    value = binding["sha256"]
                    if not re.fullmatch(r"[0-9a-f]{64}", value) or path in bindings:
                        raise ValueError("Invalid completed result artifact binding")
                    bindings[path] = value
                return bindings
            return {}

        try:
            artifacts = completed("results", site / "index.html", "site")
            provenance_sha = artifacts.get(provenance_path)
            provenance = metadata(provenance_path, sha256=provenance_sha, max_bytes=32 * 1024**2)
            storage = provenance.get("audio_storage")
            if storage is None:
                return None
            if (not isinstance(storage, dict) or storage.get("mode") != "studio_reference_v1"
                    or provenance_sha is None or not site.is_relative_to(root)):
                raise ValueError("Unbound Studio result reference metadata")
            match = re.fullmatch(r"audio/([0-9a-f]{64})\.wav", name)
            if match is None or query:
                raise ValueError("Invalid result audio name")
            references = storage.get("references")
            if (not isinstance(references, dict) or set(references) != {"path", "sha256"}
                    or references["path"] != "audio-references.json"
                    or not isinstance(references["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", references["sha256"])):
                raise ValueError("Invalid result audio reference binding")
            index = metadata(site / references["path"],
                sha256=references["sha256"], max_bytes=128 * 1024**2)
            evaluation_value = index.get("evaluation")
            if (index.get("schema_version") != 1 or index.get("kind") != "studio_scored_audio_references_v1"
                    or not isinstance(index.get("project"), str) or not Path(index["project"]).is_absolute()
                    or Path(index["project"]).resolve() != root / "project.json"
                    or not isinstance(evaluation_value, str) or not Path(evaluation_value).is_absolute()):
                raise ValueError("Result audio belongs to a different project")
            evaluation = Path(evaluation_value).resolve()
            if not evaluation.is_relative_to(root):
                raise ValueError("Result evaluation escaped its project")
            evaluation_artifacts = completed("evaluation", evaluation, "evaluation")
            receipt_sha = evaluation_artifacts.get(evaluation / "receipt.json")
            if (receipt_sha is None or receipt_sha != index.get("evaluation_receipt_sha256")
                    or receipt_sha != provenance.get("evaluation_receipt_sha256")):
                raise ValueError("Result audio evaluation binding changed")
            receipt = metadata(evaluation / "receipt.json", sha256=receipt_sha)
            for field, filename in (("report_sha256", "report.json"), ("clips_sha256", "clips.json")):
                if receipt.get(field) != index.get(field) or receipt.get(field) != evaluation_artifacts.get(evaluation / filename):
                    raise ValueError("Result audio scored report binding changed")
            registered = index.get("audio")
            reference = registered.get(match[1]) if isinstance(registered, dict) else None
            if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                    or reference.get("sha256") != match[1] or not isinstance(reference.get("path"), str)
                    or not Path(reference["path"]).is_absolute()):
                raise ValueError("Result audio is not registered")
            audio = Path(reference["path"]).resolve()
            allowed = (root, self.workspace, Path(state.get("storage_workspace", self.workspace)).resolve())
            if audio.suffix.lower() != ".wav" or not any(audio.is_relative_to(path) for path in allowed):
                raise ValueError("Result audio escaped the selected workspace")
            before = audio.stat()
            if not audio.is_file() or digest(audio) != match[1]:
                raise ValueError("Result audio changed")
            after = audio.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Result audio changed during verification")
            return audio
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # Original storage paths and the private reference index never
            # become browser error details.
            raise ValueError("결과 음원 참조를 확인할 수 없습니다. 원본 음원과 완료 기록을 확인하세요.") from None

    def _history_download_files(self, state, result, *, planned=False):
        """List planned or verified files without opening the downloads.

        Byte counts describe the verified material, including cache reuse. They
        are not a claim that these bytes crossed the network during this run.
        The original large downloads are deliberately not opened or re-hashed.
        """
        from .studio_data import _cached_asset, _official_assets, MIT_REV
        settings = state["settings"]
        policy = {key: settings.get(key) for key in ("evaluation_policy", "noise_material_policy",
            "material_acquisition_policy", "fma_acquisition_policy", "material_review_policy",
            "official_noise_acquisition_policy")}
        key = identity([state["preset"], policy, result.get("recipe_version")])
        if key not in self._history_download_plans:
            # Only load static source pins. Calling preparation/planning helpers
            # here can import numerical runtimes or perform acquisition checks.
            assets = [a for a in _official_assets() if a["id"] not in ("dns_fan", "dns_door", "electronic_validation")]
            if policy["noise_material_policy"] == "reuse_negative_waveforms":
                assets = [a for a in assets if a["id"] not in ("audioset", "audioset_segments", "audioset_ontology", "fma", "fma_metadata")]
            if policy["material_acquisition_policy"] in ("automatic_v1", "automatic_v2", "automatic_v3") and state["preset"] in ("trial", "production"):
                assets = [a for a in assets if a["id"] != "zeroth" and not (state["preset"] == "production" and a["id"] == "generic")]
            resources = files("korean_openwakeword") / "resources"
            version = result.get("recipe_version", (policy["evaluation_policy"] or {}).get("evaluation_materials_recipe", 4))
            recipe = {"assets": assets, "recipe_version": version, "additional_download": {"files": 270}}
            if version in (5, 6):
                recipe["common_evaluation_catalog"] = read_json(resources / "evaluation_materials_v1.json")
                recipe["rir_archive"] = read_json(resources / "evaluation_rirs_v1.json")["archive"]
            if version < 6:
                recipe["evaluation_assets"] = read_json(resources / "evaluation_backgrounds_v4.json")["assets"]
            if policy["official_noise_acquisition_policy"]:
                from .official_music import official_music_preview
                recipe["official_music_bytes"] = official_music_preview()["selected_payload_bytes"]
            self._history_download_plans[key] = recipe
        recipe = self._history_download_plans[key]
        shared = Path(state.get("storage_workspace", self.workspace)) / "shared"
        directory, files_out, seen = shared / "downloads", [], set()

        def item(identifier, label, group, group_label, done=None, total=None, *, unit="bytes", complete=False):
            return {"id": identifier, "label": label, "group": group, "group_label": group_label,
                "state": "waiting" if planned else "complete" if complete else "unknown",
                "done": 0 if planned else done, "total": total, "unit": unit,
                "network_bytes": 0, "bytes_per_second": None, "eta_seconds": None}

        def receipt_file(asset, identifier, group, label, target=None):
            if target is None:
                target = (_cached_asset(asset, directory) if planned and asset.get("checksum_algorithm") != "md5"
                          else directory / asset["filename"])
            if str(target) in seen:
                return
            seen.add(str(target))
            size = asset.get("size_bytes")
            verified = False
            for candidate in (target.with_name(target.name + ".receipt.json"),
                              directory / (target.name + ".reused.receipt.json")):
                try:
                    record = self._history_read(state, candidate)
                    algorithm = asset.get("checksum_algorithm", "sha256")
                    checksum = record.get("publisher_md5") if algorithm == "md5" else record.get(algorithm)
                    verified = (record.get("completed_at") is not None and record.get("bytes") == size
                        and checksum == asset.get(algorithm) and checksum is not None
                        and record.get("source_url") == asset.get("source_url"))
                    if verified:
                        break
                except (ValueError, KeyError, OSError):
                    continue
            if planned:
                row = item(identifier, label, group, label, total=size)
                row["filename"] = Path(asset["filename"]).name
                if verified:
                    try:
                        row["cache_candidate"] = target.is_file() and target.stat().st_size == size
                    except OSError:
                        pass
                files_out.append(row)
                return
            row = item(identifier, label, group, label, size if verified else None, size, complete=verified)
            files_out.append({**row, "filename": Path(asset["filename"]).name})

        names = {"generic": ("generic_negative", "일반 비호출 · openWakeWord"),
                 "zeroth": ("zeroth", "한국어 낭독 · Zeroth"),
                 "audioset": ("audioset", "소음 증강 · AudioSet"),
                 "audioset_segments": ("audioset", "소음 증강 · AudioSet"),
                 "audioset_ontology": ("audioset", "소음 증강 · AudioSet"),
                 "fma": ("official_music", "소음 증강 음악 · FMA"),
                 "fma_metadata": ("official_music", "소음 증강 음악 · FMA")}
        selected_music = recipe.get("official_music_bytes") if planned else result.get("official_music_preparation")
        for asset in recipe["assets"]:
            if selected_music and asset["id"] in ("fma", "fma_metadata"):
                continue
            group, label = names.get(asset["id"], (asset["id"], asset["filename"]))
            receipt_file(asset, "base:" + asset["id"], group, label)
        for asset in recipe.get("evaluation_assets", []):
            receipt_file(asset, "evaluation:" + asset["id"], "evaluation_legacy", "평가용 배경 음원")
        for asset in recipe.get("common_evaluation_catalog", {}).get("archives", []):
            group, label = (("fsd50k", "평가용 환경음·전자음 · FSD50K") if asset["id"].startswith("fsd_") else
                ("musan", "평가용 말소리·음악 · MUSAN") if asset["id"] == "musan" else
                ("seoul", "평가용 한국어 대화 · Seoul") if asset["id"] == "seoul" else
                names.get(asset["id"], (asset["id"], asset["filename"])))
            receipt_file(asset, "common:" + asset["id"], group, label)
        if recipe.get("recipe_version") in (5, 6):
            asset = recipe["rir_archive"]
            receipt_file(asset, "common:rir_archive", "common_rir", "평가용 공간 울림 · OpenSLR RIR",
                         directory / "common_rir" / asset["filename"])
        music = result.get("official_music_preparation", {}).get("receipt")
        if planned and recipe.get("official_music_bytes"):
            files_out.append(item("official_music", "소음 증강 음악 · FMA", "official_music",
                "소음 증강 음악 · FMA", total=recipe["official_music_bytes"]))
        elif music:
            size = None
            try:
                record = self._history_read(state, music["path"], sha256=music["sha256"])
                members = record.get("files", [])
                if (record.get("kind") == "selective_fma_small_v1" and record.get("completed_at")
                        and members and all(type(row.get("compressed_bytes")) is int for row in members)):
                    size = sum(row["compressed_bytes"] for row in members)
            except (ValueError, KeyError, OSError):
                pass
            files_out.append(item("official_music", "소음 증강 음악 · FMA", "official_music",
                "소음 증강 음악 · FMA", size, size, complete=size is not None))
        # MIT WAVs have individual completion receipts, not a large ZIP receipt.
        try:
            inventory = self._history_read(state, shared / "mit_rir/provider_tree.json")
            expected_url = f"https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/{MIT_REV}/16khz?limit=1000"
            entries = [row for row in inventory.get("entries", []) if row.get("type") == "file" and row.get("path", "").endswith(".wav")]
            if inventory.get("source_url") != expected_url or len(entries) != recipe["additional_download"]["files"]:
                raise ValueError("저장된 공간 울림 자료 목록이 다릅니다.")
            size, completed = 0, 0
            for row in () if planned else entries:
                target = shared / "mit_rir" / row["path"]
                try:
                    record = self._history_read(state, target.with_name(target.name + ".receipt.json"))
                    if (record.get("completed_at") and record.get("bytes") == row["size"]
                            and record.get("sha256") == row.get("lfs", {}).get("oid")):
                        completed += 1
                        size += record["bytes"]
                except (ValueError, KeyError, OSError):
                    continue
            total = sum(row["size"] for row in entries)
            files_out.append(item("mit_rir", "학습용 공간 울림 · MIT RIR", "mit_rir", "학습용 공간 울림 · MIT RIR",
                size if completed else None, total, complete=completed == len(entries)))
        except (ValueError, KeyError, OSError):
            files_out.append(item("mit_rir", "학습용 공간 울림 · MIT RIR", "mit_rir", "학습용 공간 울림 · MIT RIR"))
        automatic = result.get("automatic_source_preparation", {})
        if planned and policy["material_acquisition_policy"] in ("automatic_v1", "automatic_v2", "automatic_v3"):
            from .studio_data import _automatic_source_plan
            components = _automatic_source_plan(state["preset"], settings["generic_hours"], settings["korean_hours"])["components"]
            for component in components:
                identifier = "automatic:" + component["id"]
                files_out.append(item(identifier, component["name"], identifier, component["name"],
                    total=component["target_hours"], unit="hours"))
        targets = {row["id"]: row["target_hours"] for row in result.get("automatic_source_plan", {}).get("components", [])}
        for component, actual in automatic.get("component_actual_hours", {}).items():
            target = targets.get(component)
            files_out.append(item("automatic:" + component, component, "automatic:" + component,
                "학습용 비호출 · " + component, actual, target, unit="hours", complete=target is not None and actual >= target))
        return files_out

    def _live_download_files(self, state, record, *, running):
        """Keep planned rows visible through preflight, transfer and pause."""
        planned = self._history_download_files(state, {}, planned=True)
        files = record.get("files") if record.get("stage") == "data_download" else None
        if not isinstance(files, list) or not files:
            return planned
        current = {row["id"]: row for row in files if isinstance(row, dict) and isinstance(row.get("id"), str)}
        mit = [current.pop(identifier) for identifier, row in list(current.items()) if row.get("group") == "mit_rir"]
        if mit:
            active_states = ("failed", "downloading", "retrying", "verifying", "waiting")
            states = {row.get("state") for row in mit}
            if states == {"reused"}:
                status = "reused"
            else:
                status = "complete" if states <= {"complete", "reused"} else next(
                    (status for status in active_states if status in states), "unknown")
            def total_for(key):
                amounts = [row.get(key) for row in mit]
                return sum(amounts) if all(type(value) is int and value >= 0 for value in amounts) else None
            current["mit_rir"] = {"id": "mit_rir", "state": status, "unit": "bytes",
                "done": total_for("done"), "total": total_for("total")}
        merged = []
        for row in planned:
            live = current.pop(row["id"], {})
            if live.get("deferred") and live.get("unit") is None:
                live = {}
            merged.append({**row, **live, "label": row["label"], "group_label": row["group_label"]})
        merged.extend(current.values())
        if not running:
            merged = [{**row, "network_bytes": 0, "bytes_per_second": None, "eta_seconds": None} for row in merged]
        return merged

    def history_review_rows(self, project_id, step):
        """Read the exact saved representative queue without reopening approval."""
        state = self._load(project_id)
        policy = state["settings"].get("pilot_review_policy")
        if policy == PREPOST_PILOT_POLICY and step == "call-pilot-review":
            return self._history_call_pilot_rows(state)
        pronunciation = ("positive_review", "positive") if policy == PREPOST_PILOT_POLICY else ("pilot_review", "pilot")
        stage, queue_id = {"pronunciation": pronunciation, "hard-review": ("hard_review", "hard")}.get(step, (None, None))
        if not stage or self._history_result(state, stage) is None:
            raise ValueError("완료된 발음 확인 기록이 아닙니다.")
        binding = state.get("inputs", {}).get("review_answers", {}).get(queue_id)
        if not binding:
            raise ValueError("저장된 발음 확인 기록이 없습니다.")
        answer = self._history_read(state, binding["path"], sha256=binding["sha256"])
        if answer.get("wakeword") != state["wakeword"]:
            raise ValueError("저장된 발음 확인의 호출어가 다릅니다.")
        policy = answer.get("synthetic_review_policy")
        if step == "hard-review" and policy in ("one_per_phrase_v1", HARD_SAMPLE_REVIEW_POLICY):
            selection_path = ".studio/hard_representative_selection_v1.json"
            if policy == HARD_SAMPLE_REVIEW_POLICY:
                pilot = self._history_result(state, "hard_pilot") or {}
                selection_path = pilot["review"]["representative_selection"]["path"]
            selection = self._history_read(state, selection_path, sha256=answer["representative_selection_sha256"])
            if (selection.get("policy") != answer["synthetic_review_policy"]
                    or selection.get("source_manifest", {}).get("sha256") != answer.get("source_manifest_sha256")):
                raise ValueError("저장된 유사어 대표 선정 기록이 다릅니다.")
            queue = selection["representative_queue"]
        else:
            positive_stage = "positives" if state["settings"].get("pilot_review_policy") in (
                POSTGEN_PILOT_POLICY, PREPOST_PILOT_POLICY) else "pilot"
            result = self._history_result(state, positive_stage if step == "pronunciation" else "hard_generation") or {}
            if step == "pronunciation" and positive_stage == "positives" and state.get("inputs", {}).get("postgeneration_repair_rounds"):
                result = self._effective_positive_generated(state, presentation=True) or {}
            descriptor = result.get("review", {})
            queue = {"path": result.get("queue") or result.get("review_queue") or descriptor.get("queue"),
                     "sha256": answer.get("queue_sha256")}
        if not queue.get("path") or queue.get("sha256") != answer.get("queue_sha256"):
            raise ValueError("저장된 발음 확인 목록이 다릅니다.")
        rows = self._history_read(state, queue["path"], sha256=queue["sha256"], rows=True)
        decisions = answer.get("decisions", {})
        if (not rows or len(rows) > 1000 or len({row["sample_id"] for row in rows}) != len(rows)
                or set(decisions) != {row["sample_id"] for row in rows}
                or any(answer.get("queued_audio_sha256", {}).get(row["sample_id"]) != row.get("sha256") for row in rows)):
            raise ValueError("저장된 발음 확인과 음원 목록이 일치하지 않습니다.")
        return state, answer, rows

    def _history_call_pilot_rows(self, state):
        from .review import PLAY_STARTED_POLICY, validate_started_decision
        if self._history_result(state, "pilot_review") is None:
            raise ValueError("완료된 호출어 사전 검수 기록이 아닙니다.")
        rows, decisions = [], {}
        saved_at, started_policy_used = None, False
        for round_record in state["inputs"].get("call_pilot_reviews", []):
            queue, report = round_record["queue"], round_record["report"]
            candidates = self._history_read(state, queue["path"], sha256=queue["sha256"], rows=True)
            answer = self._history_read(state, report["path"], sha256=report["sha256"])
            if (answer.get("wakeword") != state["wakeword"] or answer.get("queue_sha256") != queue["sha256"]
                    or set(answer["decisions"]) != {row["sample_id"] for row in candidates}
                    or any(answer["queued_audio_sha256"].get(row["sample_id"]) != row["sha256"] for row in candidates)):
                raise ValueError("저장된 사전 검수와 음원 목록이 일치하지 않습니다.")
            saved_at = answer.get("saved_at")
            policy = answer.get("autosave_policy")
            if policy not in (None, PLAY_STARTED_POLICY):
                raise ValueError("지원하지 않는 저장된 사전 검수 판정 방식입니다.")
            if policy == PLAY_STARTED_POLICY:
                for decision in answer["decisions"].values():
                    validate_started_decision(decision)
                if (answer.get("kind") != "pilot_listening_review"
                        or answer.get("source_scope") != "Human dispositions entered through the local review screen"
                        or answer.get("playback_started_count") != sum(decision["playback_started"] for decision in answer["decisions"].values())
                        or answer.get("playback_completed_count") != sum(decision["playback_completed"] for decision in answer["decisions"].values())):
                    raise ValueError("저장된 사전 검수의 실제 재생 기록이 올바르지 않습니다.")
                started_policy_used = True
            for candidate in candidates:
                sample_id = candidate["sample_id"]
                decision = answer["decisions"][sample_id]
                if decision.get("decision") == "keep" and (policy == PLAY_STARTED_POLICY or decision.get("listened") is True):
                    rows.append(candidate)
                    decisions[sample_id] = decision
        rows.sort(key=lambda row: (row["split"], row.get("voice_number") or 0, row["voice_family_id"]))
        families = {row["voice_family_id"] for row in rows}
        settings = state["settings"]
        if len(rows) != len(families) or len(rows) != settings["train_references"] + settings["validation_references"]:
            raise ValueError("승인된 사전 검수 목소리 구성이 올바르지 않습니다.")
        answer = {"decisions": decisions, "saved_at": saved_at}
        if started_policy_used:
            answer["autosave_policy"] = PLAY_STARTED_POLICY
        return state, answer, rows

    def _history_validation_samples(self, state):
        """Select one playable, previously unreviewed original per saved evaluation voice."""
        completed = next((entry for entry in reversed(state["history"]) if entry["stage"] == "positives"), None)
        if not completed or not re.fullmatch(r"[0-9a-f]{64}", completed.get("sha256", "")):
            return []
        receipt = self._history_read(state, completed["result"], sha256=completed["sha256"])
        if receipt["result"]["state"] != "completed":
            return []
        root = Path(state["project"]).parent
        manifest = root / "data/positive/validation.jsonl"
        binding = next((entry for entry in receipt["artifact_bindings"] if Path(entry["path"]) == manifest), None)
        if not binding or not re.fullmatch(r"[0-9a-f]{64}", binding.get("sha256", "")):
            return []
        expected = receipt["result"]["result"]["outputs"][1]["distribution"]["voice_family"]
        if not isinstance(expected, dict) or not expected or len(expected) != state["settings"]["validation_references"]:
            return []
        selected = {}
        # The frozen manifest is sorted by generation slot. Stop after the
        # first technical pass for each voice, leaving the larger pool intact.
        for clip in self._history_read(state, manifest, sha256=binding["sha256"], rows=True, max_bytes=8 * 1024**2):
            voice = clip.get("voice_family_id")
            if (voice in expected and voice not in selected and clip.get("split") == "validation"
                    and clip.get("text") == state["wakeword"] and clip.get("provenance") == "qwen_tts"
                    and clip.get("generation_method") == "clone" and clip.get("human_reference_used") is False
                    and clip.get("individual_human_listening") is False and clip.get("qc", {}).get("passed") is True
                    and Path(clip["path"]).parent == Path("data/generated/validation")
                    and Path(clip["path"]).suffix.lower() == ".wav"
                    and re.fullmatch(r"[0-9a-f]{64}", clip.get("sha256", ""))):
                selected[voice] = clip
                if len(selected) == len(expected):
                    return [selected[voice] for voice in sorted(selected)]
        return []

    def history_playback_rows(self, project_id, step):
        state, answer, rows = self.history_review_rows(project_id, step)
        if (step != "pronunciation" or state["status"] != "completed"
                or state["settings"].get("pilot_review_policy") != TRAIN_PILOT_POLICY
                or len(rows) != state["settings"].get("train_references")
                or any(clip.get("split") != "train" for clip in rows)
                or len({clip.get("voice_family_id") for clip in rows}) != len(rows)):
            return state, answer, rows
        try:
            validation = self._history_validation_samples(state)
            combined = rows + validation
            if (validation and len({clip["sample_id"] for clip in combined}) == len(combined)
                    and len({clip["sha256"] for clip in combined}) == len(combined)):
                return state, answer, ([{**clip, "review_group": "학습용"} for clip in rows]
                                       + [{**clip, "review_group": "공통 평가용"} for clip in validation])
        except (ValueError, KeyError, IndexError, OSError):
            pass  # Missing playback extras must not hide an intact review record.
        return state, answer, rows

    def _completed_stage_views(self, state):
        """Saved final values stay separate from the active worker's progress."""
        views = {}
        def row(identifier, label, actual, target=None, unit="개"):
            return {"id": identifier, "label": label, "done": actual, "total": target, "unit": unit}
        def add(step, stage, rows, **extra):
            if self._history_result(state, stage) is not None:
                views[step] = {"completed": True, "state": "completed", "rows": rows, **extra}
        settings = state["settings"]
        downloaded = self._history_result(state, "data_download")
        if downloaded is not None:
            try:
                records = self._history_download_files(state, downloaded)
                known = records and all(r["state"] == "complete" and r["unit"] == "bytes" and r["done"] is not None for r in records)
                size = sum(r["done"] for r in records) if known else None
                add("downloads", "data_download", [], progress={"done": size, "total": size, "unit": "bytes",
                    "label": "자료 다운로드", "files": records}, quantity_scope="verified_material_including_reuse_not_network_traffic")
            except (ValueError, KeyError, OSError):
                add("downloads", "data_download", [], progress=None)
        references = self._history_result(state, "references") or {}
        outputs = references.get("outputs", [])
        add("voices", "references_review", [row("train", "학습용 목소리", outputs[0].get("generated") if outputs else None, settings.get("train_references")),
            row("validation", "공통 평가용 목소리", outputs[1].get("generated") if len(outputs) > 1 else None, settings.get("validation_references"))])
        positives = self._history_result(state, "positives") or {}
        outputs = positives.get("outputs", [])
        add("positive-generation", "positives", [row("train", "학습용 호출어 음성", outputs[0].get("technical_pass") if outputs else None, settings.get("originals")),
            row("validation", "공통 평가용 호출어 음성", outputs[1].get("technical_pass") if len(outputs) > 1 else None, settings.get("validation_originals"))])
        phrases = self._history_result(state, "hard_phrases") or {}
        add("phrases", "hard_phrases", [], phrases=phrases.get("phrases", state.get("inputs", {}).get("hard_phrases", [])))
        hard = self._history_result(state, "hard_generation") or {}
        add("hard-generation", "hard_generation", [row("hard", "유사어 음성", hard.get("technical_pass"), hard.get("target_originals"))])
        features = self._history_result(state, "features") or {}
        labels = {"positive": "호출어 음성", "generic_negative": "일반 비호출", "korean_negative": "한국어 비호출", "hard_negative": "유사어 음성"}
        add("features", "features", [row(key, labels.get(key, key), value.get("feature_rows"),
            features.get("positive_expected_rows") if key == "positive" else value.get("feature_rows"), "개 특징")
            for key, value in features.get("pools", {}).items()])
        training = self._history_result(state, "training") or {}
        add("training", "training", [row(str(i), f"모델 {i + 1}", member.get("updates"), settings.get("updates"), "회")
            for i, member in enumerate(training.get("members", []))])
        evaluation = self._history_result(state, "evaluation") or {}
        development = self._history_result(state, "development") or {}
        sessions = evaluation.get("report", {}).get("sessions")
        target = development.get("audio_files")
        if target is not None and development.get("negative_recordings") is not None:
            target += development["negative_recordings"]
        add("evaluation", "evaluation", [row("evaluation", "성능 평가 음원", sessions, target, "개")])
        prepost = settings.get("pilot_review_policy") == PREPOST_PILOT_POLICY
        if prepost:
            pilot = self._history_result(state, "pilot") or {}
            add("call-pilot-generation", "pilot", [row("pilot", "호출어 샘플", pilot.get("technical_pass"), pilot.get("target"))])
        review_stages = [("pronunciation", "positive_review" if prepost else "pilot_review"), ("hard-review", "hard_review")]
        if prepost:
            review_stages.insert(0, ("call-pilot-review", "pilot_review"))
        for step, stage in review_stages:
            if self._history_result(state, stage) is None:
                continue
            try:
                _, answer, audio_rows = self.history_playback_rows(state["id"], step)
                decisions = answer["decisions"]
                voice_labels = self._call_voice_labels(state, audio_rows) if step in ("call-pilot-review", "pronunciation") else [{} for _ in audio_rows]
                items = [{"id": audio["sample_id"], "text": audio.get("text", ""),
                    "group": audio.get("review_group") or audio.get("voice_family_id"),
                    **({"voice_number": audio["voice_number"]} if "voice_number" in audio else {}),
                    "decision": decisions.get(audio["sample_id"], {}).get("decision"),
                    "listened": decisions.get(audio["sample_id"], {}).get("listened") is True,
                    "saved_review": audio["sample_id"] in decisions,
                    **voice_label,
                    **({
                        **{key: decisions.get(audio["sample_id"], {})[key] for key in ("playback_started", "playback_completed")
                           if key in decisions.get(audio["sample_id"], {})}} if step == "call-pilot-review" else {}),
                    "audio": f"/history-stage/audio/{i}?project={quote(state['id'])}&step={step}"}
                    for i, (audio, voice_label) in enumerate(zip(audio_rows, voice_labels))]
                add(step, stage, [row("review", "발음 확인", len(decisions), len(decisions))],
                    review={"items": items, "saved": True, "saved_at": answer.get("saved_at"),
                            **({"autosave_policy": answer["autosave_policy"]} if answer.get("autosave_policy") else {})})
            except (ValueError, KeyError, OSError):
                add(step, stage, [], review={"items": [], "saved": False})
        return views

    def _download_quality(self, state, progress, saved_download):
        """Read the generic feature scan and its published completion metadata."""
        from .assets import GENERIC_SHA

        root = Path(state["project"]).parent.resolve()
        try:
            imported = (saved_download or {}).get("generic_feature_import")
            output = imported.get("output") if isinstance(imported, dict) and imported.get("imported") is True else None
            if output is None:
                config = self._history_read(state, state["project"])
                output = config["pool_features"]["generic_negative"]
            metadata_path = resolved(root, output) / "metadata.json"
            if metadata_path.is_relative_to(root):
                record = self._history_read(state, metadata_path)
                total = record.get("feature_rows") if isinstance(record, dict) else None
                if (type(total) is int and total > 0
                        and record.get("role") == "generic_negative" and record.get("split") == "train"
                        and record.get("feature_sha256") == GENERIC_SHA
                        and record.get("imported_precomputed") is True
                        and record.get("quality_review") == "accepted"
                        and record.get("scanned_rows") == total):
                    return {"state": "complete", "done": total, "total": total}
        except (KeyError, OSError, TypeError, ValueError):
            pass
        if progress.get("stage") == "data_download" and progress.get("phase") == "quality" and progress.get("unit") == "rows":
            done, total = progress.get("done"), progress.get("total")
            if type(done) is int and type(total) is int and total > 0 and 0 <= done <= total:
                return {"state": "checking", "done": done, "total": total}
        return None

    def _evaluation_checkpoint_progress(self, state, progress):
        """Prefer this live worker's durable count over its 100-session log."""
        job = state.get("job") or {}
        if progress.get("source") != "current_task_log" or progress.get("phase") != "scoring":
            return progress
        owner = self._worker_owner(job)
        pid, creation = owner.get("pid"), owner.get("creation_identity")
        if not pid or not creation or process_identity(pid) != creation:
            return progress
        locations = getattr(self, "_evaluation_progress_locations", None)
        if locations is None:
            locations = self._evaluation_progress_locations = {}
        key = job["task_id"]
        folder = locations.get(key)
        if folder is None:
            reports = Path(state["project"]).parent / "reports"
            if not reports.is_dir():
                return progress
            # A bounded directory listing, never a walk through WAVs or scores.
            for candidate in reports.iterdir():
                if not re.fullmatch(r"studio_evaluation(?:_attempt[0-9]+)?", candidate.name):
                    continue
                try:
                    lock = read_json(candidate / ".writer.lock")
                    if (lock.get("pid") == pid and _timestamp(lock.get("started_at")) is not None
                            and _timestamp(job.get("started_at")) is not None
                            and _timestamp(lock["started_at"]) >= _timestamp(job["started_at"])):
                        folder = locations[key] = candidate
                        break
                except (OSError, ValueError, TypeError):
                    continue
        if folder is None:
            return progress
        try:
            lock = read_json(folder / ".writer.lock")
            if lock.get("pid") != pid:
                return progress
            path = folder / "_evaluation_work/checkpoint.json"
            if path.stat().st_size > 4096:
                return progress
            checkpoint = read_json(path)
            count = checkpoint.get("committed_sessions")
            if (checkpoint.get("schema_version") != 1 or type(count) is not int
                    or not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("binding", ""))
                    or not progress["done"] < count <= progress["total"]):
                return progress
            return {**progress, "done": count, "source": "current_task_checkpoint",
                    "phase": "finalizing" if count == progress["total"] else "scoring"}
        except (OSError, ValueError, TypeError, KeyError):
            return progress

    def _read_optional_progress(self, path, scope, view, *, max_bytes=None):
        """Read display-only JSON; retain same-task data through a sharing denial."""
        try:
            if max_bytes is not None and path.stat().st_size > max_bytes:
                self._progress_snapshots.pop(path, None)
                return {}
            record = read_json(path)
        except PermissionError:
            view["progress_pending"] = True
            previous_scope, previous = self._progress_snapshots.get(path, (None, {}))
            return previous if previous_scope == scope else {}
        except (OSError, ValueError):
            self._progress_snapshots.pop(path, None)
            return {}
        if not isinstance(record, dict):
            self._progress_snapshots.pop(path, None)
            return {}
        self._progress_snapshots[path] = scope, record
        return record

    def _owned_task_progress(self, state, view):
        """Read small, current-worker telemetry without inspecting its outputs.

        Unlike durable result receipts this record is only a display hint. A
        missing, stale or malformed record must never advance a workflow.
        """
        job = state.get("job") or {}
        if not job.get("directory") or not job.get("request_sha256"):
            return {}
        directory = Path(job["directory"]).resolve()
        expected = Path(state["project"]).parent.resolve() / ".studio/jobs" / job["task_id"]
        if directory != expected:
            return {}
        path = directory / "progress.json"
        try:
            scope = (job.get("stage"), job.get("task_id"))
            record = self._read_optional_progress(path, scope, view, max_bytes=32768)
            if (record.get("kind") != "studio_task_progress_v1"
                    or record.get("task_id") != job.get("task_id")
                    or record.get("request_sha256") != job["request_sha256"]
                    or record.get("stage") != job.get("stage")):
                return {}
            owner = self._worker_owner(job)
            if (not owner.get("pid") or not owner.get("creation_identity")
                    or record.get("pid") != owner["pid"]
                    or record.get("creation_identity") != owner["creation_identity"]
                    or process_identity(owner["pid"]) != owner["creation_identity"]):
                return {}
            started, updated = _timestamp(job.get("started_at")), _timestamp(record.get("updated_at"))
            if updated is None or started is not None and updated < started:
                return {}
            return record
        except (OSError, TypeError, ValueError):
            return {}

    def _progress_eta(self, state, job, progress):
        """Use the worker's measured rate, or a stable window for older producers."""
        project = state["id"]

        def valid_number(value):
            return type(value) in (int, float) and math.isfinite(value)

        done, total = progress.get("done"), progress.get("total")
        counter = progress.get("rate_done", done)
        phase = progress.get("rate_phase", progress.get("phase") or progress.get("label"))
        excluded = {"retry", "download_many", "feature_validation", "device_selection",
                    "model_initialization", "gpu_cache_loading", "preparation_validation",
                    "preparation_finalizing", "validation", "finalizing"}
        if (state.get("status") != "running" or progress.get("eta_eligible") is False
                or phase in excluded or progress.get("unit") == "완료 단계"
                or not all(valid_number(value) for value in (done, total, counter))
                or total <= 0 or not 0 <= done < total or counter < 0):
            self._rates.pop(project, None)
            return None
        if "worker_rate" in progress:
            self._rates.pop(project, None)
            rate = progress["worker_rate"]
            if not isinstance(rate, dict):
                return None
            speed, measured_at = rate.get("per_second"), rate.get("measured_at")
            if (not valid_number(speed) or speed <= 0 or not valid_number(measured_at)
                    or not -1 <= time.time() - measured_at <= 60):
                return None
            estimate = (total - done) / speed
            return estimate if math.isfinite(estimate) else None
        stamp = time.monotonic()
        scope = (job.get("task_id"), job.get("stage"), phase, total, progress.get("unit"))
        window = self._rates.get(project)
        if (not isinstance(window, dict) or window.get("scope") != scope
                or counter < window["points"][-1][1] or stamp < window["points"][-1][0]
                or stamp - window["points"][-1][0] > 90):
            window = self._rates[project] = {"scope": scope, "points": [(stamp, counter)]}
            return None
        points = window["points"]
        if counter > points[-1][1]:
            if points[-1][1] == 0:
                # A long wait for the first output is startup, not per-item cost.
                points[:] = [(stamp, counter)]
            else:
                points.append((stamp, counter))
                points[:] = [point for point in points[-8:] if stamp - point[0] <= 90]
        if len(points) < 4 or points[-1][0] - points[0][0] < 15 or stamp - points[-1][0] > 30:
            return None
        rates = [(right[1] - left[1]) / (right[0] - left[0])
                 for left, right in zip(points, points[1:]) if right[0] > left[0]]
        if len(rates) < 3 or min(rates) <= 0 or max(rates) > 2 * min(rates):
            return None
        estimate = (total - done) / statistics.median(rates)
        return estimate if math.isfinite(estimate) and estimate >= 0 else None

    def describe(self, state):
        index, plan = state["index"], state["plan"]
        stage = plan[index] if index < len(plan) else {"id": "completed", "label": "완료"}
        view = {key: state.get(key) for key in ("id", "wakeword", "preset", "status", "error", "next_action", "created_at")}
        dependency_backend = state.get("inputs", {}).get("dependency_backend")
        view["dependency_backend"] = dependency_backend
        view["dependency_setup_selected"] = state.get("dependency_setup_selected") is True
        view["dependency_install_pending"] = (view["dependency_setup_selected"] and stage["id"] == "frontend"
                                              and state["status"] == "paused" and state.get("input_submitted") is True)
        if dependency_backend in ("cpu", "cu128"):
            packages, evaluation_runtime = _dependency_package_requirements()
            view["dependency_packages"] = packages[dependency_backend]
            view["dependency_download_bytes"] = _dependency_download_bytes(view["dependency_packages"])
            if dependency_backend == "cu128" or state["settings"].get("evaluation_backend") in ("auto", "cuda"):
                view["evaluation_runtime_packages"] = evaluation_runtime
        if isinstance(state.get("download_checkpoint"), dict):
            view["download_checkpoint"] = {key: state["download_checkpoint"].get(key) for key in ("policy", "state")}
        if isinstance(state.get("navigation_checkpoint"), dict):
            view["navigation_checkpoint"] = {key: state["navigation_checkpoint"].get(key) for key in ("policy", "step", "state")}
        timing = self._timing(state)
        view.update(directory=str(Path(state["project"]).parent), stage=stage, stage_id=stage["id"],
                    settings={key: state["settings"][key] for key in (
                        "originals", "rounds", "updates", "generic_hours", "korean_hours",
                        "train_references", "validation_references", "validation_originals",
                        "hard_originals", "hard_samples_per_phrase", "hard_review_policy", "pilot_review_policy", "noise_material_policy", "reference_selection_policy", "reference_preparation_policy")
                        if key in state["settings"]},
                    reference_counts={"train": state["settings"]["train_references"],
                                      "validation": state["settings"]["validation_references"]},
                    reference_design=self._voice_design(read_json(Path(state["project"]).parent / "voices.json"), state["settings"]),
                    evaluation_policy=state.get("settings", {}).get("evaluation_policy"),
                    evaluation_policy_sha256=identity(state["settings"]["evaluation_policy"]) if state.get("settings", {}).get("evaluation_policy") else None,
                    stages=[{**entry, "status": "completed" if i < index else state["status"] if i == index else "pending"}
                            for i, entry in enumerate(plan)],
                    stage_progress={"done": index, "total": len(plan), "unit": "완료 단계"},
                    progress={"done": index, "total": len(plan), "unit": "완료 단계"},
                    elapsed_seconds=timing["total_seconds"], timing=timing,
                    migration=self._migration_view(state), eta_seconds=None)
        preview_counts = {split: self._journal_counts(Path(state["project"]).parent / "data/references" / split / "references.jsonl")["passed"]
                          for split in ("train", "validation")}
        candidate_selection = state.get("reference_voice_review_policy") == "candidate_pool_v2"
        if candidate_selection:
            view["reference_preparation"] = self._reference_preparation_view(state)
        planned_candidates = state["settings"].get("reference_selection_policy") == "candidate_pool_v3"
        if sum(preview_counts.values()) or planned_candidates:
            view["reference_preview"] = {"url": f"/reference-preview/?project={state['id']}",
                                         "count": sum(preview_counts.values()), **preview_counts,
                                         "candidate_selection": candidate_selection,
                                         "planned_count": (state["settings"]["train_references"] + state["settings"]["validation_references"]),
                                         "review_required": self._reference_review_required(state),
                                         "review_complete": bool(state.get("inputs", {}).get("reference_voice_review"))}
        hard_result = state.get("results", {}).get("hard_generation", {})
        hard_result = hard_result.get("result", hard_result)
        if hard_result.get("full_target_reached") and hard_result.get("review"):
            view["hard_preview"] = {"url": f"/hard-preview/?project={state['id']}",
                                    "count": hard_result.get("technical_pass", hard_result.get("target_originals"))}
        positive_result = self._history_result(state, "positives") or {}
        outputs = positive_result.get("outputs", [])
        if len(outputs) == 2 and all(row.get("full_target_reached") is True for row in outputs):
            view["positive_preview"] = {"url": f"/positive-preview/?project={quote(state['id'])}",
                                        "count": sum(row["technical_pass"] for row in outputs)}
        job = state.get("job") or {}
        progress_scope = (stage["id"], job.get("task_id") or (state.get("last_job") or {}).get("task_id"))

        def read_progress(path):
            return self._read_optional_progress(path, progress_scope, view)

        material_progress = Path(state["project"]).parent / ".studio" / "progress.json"
        download_record = (read_progress(material_progress)
                           if "data_download" in (stage["id"], job.get("stage")) else {})
        owned_progress = self._owned_task_progress(state, view) if job else {}
        if state.get("status") != "running":
            self._rates.pop(state["id"], None)
        if job:
            view["current_stage_elapsed_seconds"] = round(_elapsed(job.get("started_at")), 1)
            view["process_alive"] = self._job_alive(job)
            progress = Path(state["project"]).parent / ".studio" / "task_progress.json"
            if progress.exists():
                record = read_progress(progress)
                if record.get("task_id") == job["task_id"]:
                    view["progress"] = record
                    view["eta_seconds"] = record.get("eta_seconds")
            if owned_progress and job["stage"] != "evaluation":
                current_download = (job["stage"] == "data_download"
                    and download_record.get("stage") == "data_download"
                    and download_record.get("updated_at", "") >= job["started_at"])
                # The scoped file is intentionally small. File rows and the
                # current hash event remain in the material producer's record.
                view["progress"] = {**download_record, **owned_progress} if current_download else owned_progress
                view["eta_seconds"] = None
            if (job["stage"] == "data_download" and download_record.get("stage") == "data_download"
                    and download_record.get("updated_at", "") >= job["started_at"]
                    and not owned_progress):
                view["progress"] = download_record
                view["eta_seconds"] = download_record.get("eta_seconds")
            if job["stage"] == "training":
                preparation = {}
                if owned_progress.get("phase") in _TRAINING_PREPARATION_PHASES:
                    preparation = owned_progress
                elif not owned_progress:
                    # Older workers wrote preparation hints using the same path.
                    previous = self._read_optional_progress(Path(job["directory"]) / "progress.json",
                                                            progress_scope, view, max_bytes=32768)
                    if (previous.get("kind") == "training_preparation_v1"
                            and previous.get("task_id") == job["task_id"]
                            and previous.get("phase") in _LEGACY_TRAINING_PREPARATION_PHASES):
                        preparation = previous
                if preparation:
                    view["training_preparation"] = view["progress"] = preparation
                else:
                    members = Path(state["project"]).parent / "runs" / "studio_run"
                    updates = state["settings"]["updates"]
                    done = 0
                    for path in members.glob("member*/progress.json"):
                        record = read_progress(path)
                        count = record.get("completed_updates", record.get("update", 0))
                        if type(count) is int and 0 <= count <= updates:
                            done += count
                    view["progress"] = {"done": done, "total": updates * 3, "unit": "학습 업데이트",
                                        "label": "모델 3개 누적 학습량"}
                    if owned_progress.get("phase") == "training" and "worker_rate" in owned_progress:
                        view["progress"].update(phase="training", worker_rate=owned_progress["worker_rate"])
            if job["stage"] in ("references", "pilot", "positives", "hard_pilot", "hard_generation"):
                root = Path(state["project"]).parent
                folder = root / "data"
                pilot_targets = None
                initial_preparation = None
                if job["stage"] == "references" and self._reference_preparation_required(state):
                    initial_preparation = view.get("reference_preparation") or self._reference_preparation_view(state)
                    paths = []
                    total = sum(row["total"] for row in initial_preparation["rows"])
                elif job["stage"] == "references":
                    paths = [folder / "references" / split / "references.jsonl" for split in ("train", "validation")]
                    total = state["settings"]["train_references"] + state["settings"]["validation_references"]
                elif (job["stage"] in ("pilot", "positives")
                      and state["settings"].get("pilot_review_policy") == PREPOST_PILOT_POLICY):
                    paths, total = self._call_generation_progress(state, folder, job["stage"])
                elif job["stage"] == "pilot":
                    policy = state["settings"].get("pilot_review_policy")
                    splits = ("train", "validation") if policy == COMBINED_PILOT_POLICY else ("train",)
                    paths = [folder / "pilot" / split / "candidates.jsonl" for split in splits]
                    if policy in (TRAIN_PILOT_POLICY, COMBINED_PILOT_POLICY):
                        pilot_targets = [state["settings"][f"{split}_references"] for split in splits]
                    total = sum(pilot_targets) if pilot_targets else 24
                elif job["stage"] in ("hard_pilot", "hard_generation"):
                    from .studio_steps import hard_generation_budget
                    phrases = state.get("inputs", {}).get("hard_phrases", [])
                    if job["stage"] == "hard_pilot":
                        total = len(phrases)
                        paths = [folder / "generated/hard_pilot/candidates.jsonl"]
                    else:
                        approved = self._history_result(state, "hard_review") or {}
                        total = approved.get("total") if state["settings"].get("hard_review_policy") == HARD_SAMPLE_REVIEW_POLICY else None
                        if total is None:
                            total = hard_generation_budget(state["settings"], phrases)["total"]
                        paths = [folder / "generated/hard/candidates.jsonl"]
                else:
                    bank = "pilot" if state["settings"].get("pilot_review_policy") == POSTGEN_PILOT_POLICY else "generated"
                    paths = [folder / bank / split / "candidates.jsonl" for split in ("train", "validation")]
                    total = state["settings"]["originals"] + state["settings"]["validation_originals"]
                counters = [self._journal_counts(p) if p is not None else {"passed": 0, "attempts": 0} for p in paths]
                done = sum(c["passed"] for c in counters)
                if pilot_targets:
                    done = sum(min(target, len(counter.get("passed_families", ())))
                               for target, counter in zip(pilot_targets, counters))
                labels = {"references": "기준 목소리 생성 완료", "pilot": "사전 검수용 호출어 생성 완료",
                          "positives": "호출어 생성 완료", "hard_pilot": "발음 확인용 유사어 생성 완료",
                          "hard_generation": "유사어 음성 생성 완료"}
                view["progress"] = {"done": done, "total": total, "unit": "개", "label": labels[job["stage"]]}
                view["counts"] = {"tts_generated": sum(c["attempts"] for c in counters), "technical_passed": done}
                if initial_preparation is not None:
                    done = sum(row["done"] for row in initial_preparation["rows"])
                    view["progress"].update(done=done, phase="reference_candidates_generation",
                        rate_done=initial_preparation["generated_candidates"])
                    # Generated/reused candidates are not a count of TTS attempts.
                    view["counts"] = {"technical_passed": done}
                if owned_progress:
                    for field in ("phase", "eta_eligible", "activity", "worker_rate", "phase_started_at"):
                        if field in owned_progress:
                            view["progress"][field] = owned_progress[field]
                    if owned_progress.get("eta_eligible") is False:
                        view["progress"]["activity"] = {"phase": owned_progress.get("phase"),
                            "label": owned_progress.get("label", "음성 생성 준비 중")}
                if job["stage"] == "positives":
                    view["next_action"] = (f"학습용 {counters[0]['passed']:,} / {state['settings']['originals']:,}개 · "
                                           f"공통 평가용 {counters[1]['passed']:,} / {state['settings']['validation_originals']:,}개")
            if job["stage"] == "features" and not owned_progress:
                matches = re.findall(r"(positive|korean_negative|hard_negative|generic_negative) R\d+: (\d+)/(\d+)", _job_log_tail(job))
                if matches:
                    role, done, total = matches[-1]
                    view["progress"] = {"done": int(done), "total": int(total), "unit": "개 특징", "label": role}
            if job["stage"] == "evaluation":
                view["progress"] = self._evaluation_checkpoint_progress(state, _evaluation_log_progress(job))
                view["progress_note"] = "음원 수 기준이며, 긴 비호출 음원은 더 오래 걸립니다."
                view["eta_seconds"] = None
                phase = view["progress"]["phase"]
                if phase == "validation":
                    view["next_action"] = "평가 음원의 무결성과 형식을 확인하고 있습니다."
                elif phase == "finalizing":
                    view["next_action"] = "채점을 마쳤으며 결과를 정리하고 있습니다."
                if material_progress.exists():
                    record = read_progress(material_progress)
                    if (record and record.get("stage") == "evaluation" and record.get("phase") == "runtime_recommendation"
                            and record.get("updated_at", "") >= job.get("started_at", "")
                            and record.get("task_id", job["task_id"]) == job["task_id"]):
                        view["progress"] = record
                        view["progress_note"] = "저장된 점수로 모델별 감지 기준과 연속 감지 수를 비교합니다."
                        view["next_action"] = "채점을 마쳤으며 모델별 추천 감지 설정을 찾고 있습니다."
                if (owned_progress.get("phase") == "scoring_activity"
                        and view["progress"].get("phase") == "scoring"
                        and isinstance(owned_progress.get("activity"), dict)):
                    view["progress"]["activity"] = owned_progress["activity"]
            current = view["progress"]
            phase_started_at = current.get("phase_started_at")
            if type(phase_started_at) in (int, float) and math.isfinite(phase_started_at):
                # A UI-only elapsed clock, independent of the worker's rate.
                current = view["progress"] = {**current, "phase_elapsed_seconds": max(0, time.time() - phase_started_at)}
            if job["stage"] != "evaluation":
                estimate = self._progress_eta(state, job, current)
                if view["eta_seconds"] is None:
                    view["eta_seconds"] = estimate
        repair_progress = self._postgeneration_repair_view(state)
        if repair_progress is not None:
            view["postgeneration_repair"] = repair_progress
            if stage["id"] == "positive_review" and state["status"] != "waiting_review":
                view["progress"] = {key: repair_progress.get(key) for key in ("done", "total", "unit", "label", "phase")}
                view["eta_seconds"] = None
        reviews = self.review_descriptors(state)
        material_results = state.get("results", {})
        downloaded = material_results.get("data_download", {})
        downloaded = downloaded.get("result", downloaded)
        connected = material_results.get("data_connect", {})
        connected = connected.get("result", connected)
        if state.get("settings", {}).get("material_acquisition_policy") in ("automatic_v1", "automatic_v2", "automatic_v3"):
            candidate = downloaded.get("candidate_unique_hours")
            admitted = connected.get("admitted_unique_hours")
            nominal = connected.get("generic_publisher_nominal_hours")
            if nominal is None:
                nominal = downloaded.get("generic_feature_import", {}).get("publisher_declared_hours")
            view["training_materials"] = {
                "targets": {"generic_negative": state["settings"]["generic_hours"],
                            "korean_negative": state["settings"]["korean_hours"]},
                "candidate": candidate, "admitted": admitted,
                "generic_publisher_nominal_hours": nominal}
            canonical_storage = downloaded.get("automatic_source_preparation", {})
            if canonical_storage.get("storage_policy") == "canonical_flac_only_v1":
                view["training_materials"]["storage"] = {key: canonical_storage.get(key)
                    for key in ("storage_policy", "canonical_storage_bytes", "source_storage_bytes",
                                "retired_original_bytes", "acquired_original_peak_bytes")}
        candidate_coverage = downloaded.get("evaluation_material_coverage")
        admitted_coverage = connected.get("evaluation_material_coverage")
        if candidate_coverage or admitted_coverage:
            view["evaluation_materials"] = {"candidate": candidate_coverage, "admitted": admitted_coverage,
                "scope": "원본 고유 시간 · 다운로드/기술 검사와 대표 검수 승인은 별도 · 실제 채점 시간은 결과에서 확인"}
            if (state.get("settings", {}).get("evaluation_policy") or {}).get("id") in ("common_development_v4", "common_development_v5"):
                def duration_view(coverage, budget):
                    if not coverage:
                        return None
                    budget = budget or {}
                    eligible = budget.get("eligible_hours")
                    if eligible is None and budget.get("eligible_seconds") is not None:
                        eligible = budget["eligible_seconds"] / 3600
                    return {"core_hours": (coverage.get("core_coverage") or {}).get("actual_total_hours"),
                            "initialization_margin_hours": coverage.get("initialization_margin_hours"),
                            "eligible_hours": eligible}
                view["evaluation_materials"]["duration"] = {
                    "candidate": duration_view(candidate_coverage, downloaded.get("negative_duration_budget")),
                    "admitted": duration_view(admitted_coverage, connected.get("negative_duration_budget"))}
        if reviews:
            r = reviews[0]
            call_phase = self._call_review_phase(state, r)
            revision = f"&rev={r['queue_sha256']}" if call_phase else ""
            view["review"] = {"queue_id": r["id"], "title": r.get("title", "음원 확인"),
                              "queue_sha256": r["queue_sha256"],
                              "description": r.get("description", "정상 음원은 보존하고 문제가 있는 항목만 표시하세요."),
                              "remaining_queues": len(reviews),
                              "url": f"/review/?project={state['id']}&queue={quote(r['id'])}" + revision}
        if state["status"] == "waiting_input":
            result = state["results"].get(stage["id"], {})
            view["input"] = result.get("input", {"kind": result.get("input_kind", "diagnostic"),
                                                   "description": result.get("message", "자료 준비가 필요합니다."),
                                                   "suggestions": result.get("suggestions", [])})
            input_kind = (view.get("input") or {}).get("kind")
            if input_kind == "dependency_setup":
                recommended = view["input"].get("recommended_backend", result.get("recommended_backend"))
                view["dependency_setup"] = _dependency_setup_options(recommended)
                view["input"] = {**view["input"], "recommended_backend": view["dependency_setup"]["recommended_backend"]}
            if input_kind == "hard_phrases":
                per_phrase = state["settings"].get("hard_samples_per_phrase")
                view["next_action"] = "유사어 문구를 입력하고 저장하세요."
                view["input"] = {**view["input"], "title": "유사어 문구 선택",
                    "description": (f"문구마다 {per_phrase:,}개의 유사어 음성을 만듭니다." if per_phrase is not None else
                                    f"입력한 문구로 총 {state['settings']['hard_originals']:,}개의 유사어 음성을 만듭니다."),
                    "label": "유사어 문구",
                    "hint": "문구 수는 직접 정하세요. 한글 1~16자, 띄어쓰기와 중복 없이 입력하세요.",
                    **({"samples_per_phrase": per_phrase, "voice_count": state["settings"]["train_references"]}
                       if per_phrase is not None else {})}
        final = state["results"].get("results", {})
        if state["status"] == "completed" or final.get("site"):
            result = state.get("results", {}).get("evaluation", {})
            report = result.get("result", result).get("report", {})
            assessment = result.get("result", result).get("common_evaluation_assessment")
            hours = [r.get("negative", {}).get("hours") for r in report.get("models", [])]
            hours = [value for value in hours if isinstance(value, (int, float))]
            observed = min(hours) if hours else None
            common = state.get("settings", {}).get("evaluation_policy") is not None
            scope = ("공통 개발 평가 결과입니다." if common else "기존 설정의 개발 평가 결과입니다.")
            scope += " 독립 최종 평가는 아직 완료하지 않았습니다."
            if observed is not None:
                scope += f" 비호출 실측 {observed:.2f}시간"
                minimum = (state.get("settings", {}).get("evaluation_policy") or {}).get("development_negative_target_hours", [20])[0]
                if observed < minimum:
                    scope += f" · 계획의 {minimum}시간 이상 장시간 오호출 검증은 미완료입니다."
                else:
                    scope += "."
            view["results"] = {"url": f"/results/{state['id']}/", "mic_url": f"/mic/{state['id']}/",
                               "scope": scope, "negative_hours": observed,
                               "common_evaluation_assessment": assessment,
                               "independent_Final_completed": False}
        view["completed_stage_views"] = self._completed_stage_views(state)
        completed_downloads = view["completed_stage_views"].get("downloads", {}).get("progress") or {}
        if stage["id"] in ("frontend", "data_download", "data_review", "data_connect") or "downloads" in view["completed_stage_views"]:
            frontend_complete = (self._history_result(state, "frontend") is not None
                                 or "downloads" in view["completed_stage_views"])
            frontend_progress = owned_progress if job.get("stage") == "frontend" else {}
            phase = frontend_progress.get("phase")
            evaluation_runtime = view.get("evaluation_runtime_packages")
            if evaluation_runtime is None and state["settings"].get("evaluation_backend") in ("auto", "cuda"):
                _, evaluation_runtime = _dependency_package_requirements()
            setup_items = _download_setup_rows(dependency_backend, evaluation_runtime, phase,
                complete=frontend_complete,
                log=_job_log_tail(job) if phase in ("environment_install", "evaluation_environment") else "")
            view["download_setup_items"] = setup_items
            if frontend_progress:
                view["progress"] = {**view["progress"], "setup_items": setup_items}
        if stage["id"] in ("frontend", "data_download", "data_review", "data_connect"):
            view["download_files"] = completed_downloads.get("files") or self._live_download_files(
                state, download_record, running=job.get("stage") == "data_download" and view.get("process_alive", False))
        download_files = view.get("download_files", completed_downloads.get("files", ()))
        if any(row.get("id") == "base:generic" for row in download_files):
            quality = self._download_quality(state, view["progress"], self._history_result(state, "data_download"))
            if quality is not None:
                view["download_quality"] = quality
        return view

    def _call_generation_progress(self, state, folder, stage):
        rounds = state["inputs"].get("call_pilot_reviews", [])
        revision = identity(rounds)
        kind = "preflight" if stage == "pilot" else "production"
        journals = {}
        for contract in (folder / "call_pronunciation").glob("*/*/contract.json"):
            scope = self._history_read(state, contract, max_bytes=16 * 1024**2).get("staged_scope", {})
            if scope.get("kind") == kind and scope.get("revision") == revision:
                journals[contract.parent.name] = contract.with_name("candidates.jsonl")
        settings = state["settings"]
        if stage == "positives":
            total = settings["originals"] + settings["validation_originals"]
        elif rounds:
            report = rounds[-1]["report"]
            reviewed = self._history_read(state, report["path"], sha256=report["sha256"])
            total = sum(answer["decision"] == "discard" for answer in reviewed["decisions"].values())
        else:
            total = settings["train_references"] + settings["validation_references"]
        return [journals.get(split) for split in ("train", "validation")], total

    def _journal_counts(self, path):
        """Incremental line scanning; UI polling never re-reads the full pool."""
        if not path.exists():
            return {"attempts": 0, "passed": 0}
        record = self._journals.setdefault(str(path), {"offset": 0, "attempts": 0, "passed": 0, "passed_families": set()})
        if path.stat().st_size < record["offset"]:
            record.update(offset=0, attempts=0, passed=0, passed_families=set())
        with path.open("rb") as handle:
            handle.seek(record["offset"])
            while True:
                before = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    record["offset"] = before
                    break
                row = json.loads(line)
                record["attempts"] += 1
                if row.get("qc", {}).get("passed", row.get("technical_pass", True)):
                    record["passed"] += 1
                    if row.get("voice_family_id"):
                        record["passed_families"].add(row["voice_family_id"])
                record["offset"] = handle.tell()
        return record

    def _project_list_entry(self, project_id, filename):
        # The selector needs a label and folder without the workflow history.
        entry = {"id": project_id}
        try:
            project = _safe_root(filename)
            entry["directory"] = str(project.parent)
            config = read_json(project)
            if isinstance(config, dict) and isinstance(config.get("wakeword"), str):
                entry["wakeword"] = config["wakeword"]
        except (OSError, ValueError):
            pass  # A missing label must not hide the remaining projects.
        return entry

    def snapshot(self, *, studio_view=False, project_id=None):
        with self._lock:
            registry = self.registry["projects"]
            active = self.registry.get("active_project_id")
            if studio_view:
                if project_id not in registry:
                    project_id = active if active in registry else next(iter(registry), None)
                state = self._load(project_id) if project_id is not None else None
                detail = self.describe(state) if state is not None else None
                projects = [detail if key == project_id else self._project_list_entry(key, filename)
                            for key, filename in registry.items()]
                states = [state] if state is not None else []
            else:
                states = [self._load(key) for key in registry]
                projects = [self.describe(state) for state in states]
            system = {"free_bytes": shutil.disk_usage(self.workspace).free, "python": sys.version.split()[0]}
            for state in states:
                preflight = state["results"].get("frontend", {}).get("result", {}).get("check", {})
                if preflight:
                    system.update(preflight)
                    break
            view = {"workspace": str(self.workspace), "token": self.token, "projects": projects,
                    "active_project_id": active, "system": system}
            if studio_view:
                view["selected_project_id"] = project_id
            return view

    def log(self, project_id):
        state = self._load(project_id)
        job = state.get("job") or state.get("last_job") or {}
        return _job_log_tail(job) or "아직 실행 로그가 없습니다."

    @property
    def analysis(self):
        with self._lock:
            if not hasattr(self, "_analysis_service"):
                from .studio_analysis import StudioAnalysis
                self._analysis_service = StudioAnalysis(self)
            return self._analysis_service

    def runtime_recommendations(self, state):
        """Read a separately completed recommendation, never rewrite history.

        Serving its frozen summary needs the bound reports, not another scan
        of the hundreds of MB of score/clip evidence already checked at search.
        """
        def result(key):
            row = state.get("results", {}).get(key, {})
            return row.get("result", row)
        bundle, evaluation = result("package").get("bundle"), result("evaluation").get("evaluation")
        if not bundle or not evaluation:
            return None
        root = Path(state["project"]).resolve().parent
        report = Path(evaluation) / "report.json"
        from .calibration import read_recommendation, SEARCH_POLICY
        directory = root / "reports" / ("studio_runtime_recommendations_" + digest(report)[:16] + "_" + identity(SEARCH_POLICY)[:12])
        if not (directory / "receipt.json").is_file():
            return None
        return read_recommendation(state["project"], bundle, evaluation, directory, verify_inputs=False)

    def microphone(self, project_id):
        if project_id not in self._microphones:
            with self._lock:
                if project_id in self._microphones:
                    return self._microphones[project_id]
                state = self._load(project_id)
                if "package" not in state["results"]:
                    raise ValueError("모델 학습과 실행 묶음 준비 후 시험할 수 있습니다.")
                from .microphone import MicrophoneService
                from .runtime import WakeWordDetector
                package = state["results"]["package"]
                original_bundle = package.get("result", package)["bundle"]
                original = read_json(Path(original_bundle) / "manifest.json")
                labels = {Path(row["path"]).stem: f"모델 {index + 1}"
                          for index, row in enumerate(original["members"])}
                # The microphone is a comparison of all original models. Old
                # per-candidate selection artifacts remain separate records.
                recommendation = self.runtime_recommendations(state)
                # A best tradeoff is not a protected recommendation. Keep that
                # model's recorded settings unless the user makes a manual trial.
                runtimes = {row["id"]: {key: (row["original_runtime"] if row.get("recommendation_available") is False
                                              else row["runtime"])[key] for key in ("threshold", "consecutive_frames")}
                            for row in recommendation["candidates"]} if recommendation else None
                self._microphones[project_id] = MicrophoneService(WakeWordDetector(original_bundle), model_labels=labels,
                    runtime_recommendations=recommendation, recommended_runtimes=runtimes)
        return self._microphones[project_id]


def make_studio_server(manager, port=8793):
    resources = files("korean_openwakeword") / "resources"

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass

        def guard(self, mutate=False, token=None):
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                raise PermissionError("로컬 주소로 접속하세요.")
            origin = self.headers.get("Origin")
            if origin is not None and origin != "http://" + host:
                raise PermissionError("다른 사이트의 요청은 허용하지 않습니다.")
            if mutate and not hmac.compare_digest(self.headers.get("X-Koww-Token", ""), token or manager.token):
                raise PermissionError("화면을 새로고침하세요.")

        def respond(self, value, status=200, kind="application/json; charset=utf-8"):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Permissions-Policy", "microphone=(self)")
            self.end_headers()
            self.wfile.write(data)

        def body(self, max_bytes=1000000, kind="application/json"):
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked requests are not supported")
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= max_bytes:
                raise ValueError("요청 크기가 올바르지 않습니다.")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != kind:
                raise ValueError("요청 형식이 올바르지 않습니다.")
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("요청을 모두 받지 못했습니다.")
            if kind != "application/json":
                return data
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("JSON object is required")
            return value

        def do_GET(self):
            try:
                self.guard()
                url = urlsplit(self.path)
                path, query = unquote(url.path), parse_qs(url.query)
                if path == "/api/state":
                    if query.get("view") == ["studio"]:
                        return self.respond(manager.snapshot(studio_view=True, project_id=query.get("project", [None])[0]))
                    return self.respond(manager.snapshot())
                if path == "/api/recipes":
                    return self.respond(manager.recipes())
                if path in ("/", "/studio.js", "/studio.css", "/studio_story.js", "/studio_story.css", "/audio_listening.css", "/listening_player.js",
                            "/studio_background_grain.svg", "/studio_background_atmosphere.svg"):
                    name = "studio.html" if path == "/" else path[1:]
                    kind = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                            ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}[Path(name).suffix]
                    return self.respond((resources / name).read_bytes(), kind=kind)
                if path.startswith("/api/projects/") and path.endswith("/log"):
                    return self.respond({"log": manager.log(path.split("/")[3])})
                if path.startswith("/history-stage/audio/"):
                    state, _, rows = manager.history_playback_rows(query["project"][0], query["step"][0])
                    index = int(path.rsplit("/", 1)[1])
                    if not 0 <= index < len(rows):
                        raise ValueError("잘못된 저장 음원 번호입니다.")
                    root = Path(state["project"]).parent.resolve()
                    audio = resolved(root, rows[index]["path"])
                    if not audio.is_relative_to(root) or digest(audio) != rows[index]["sha256"]:
                        raise ValueError("저장된 확인 음원이 없거나 변경되었습니다.")
                    try:
                        return self.send_file(audio)
                    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                        # Changing story pages cancels the browser's old audio
                        # request. Its closed socket cannot receive an error.
                        return
                if path.startswith("/hard-preview/"):
                    project_id = query["project"][0]
                    state, rows = manager.hard_preview_rows(project_id)
                    base = f"?project={quote(project_id)}"
                    if path == "/hard-preview/":
                        return self.respond((resources / "reference_preview.html").read_bytes(), kind="text/html; charset=utf-8")
                    if path == "/hard-preview/queue":
                        return self.respond({"kind": "hard", "count": len(rows), "items": [{
                            "id": row["sample_id"], "text": row["text"], "voice_family_id": row["voice_family_id"],
                            "group": row["text"] + " · " + row["voice_family_id"],
                            "audio": f"/hard-preview/audio/{i}" + base} for i, row in enumerate(rows)]})
                    if path.startswith("/hard-preview/audio/"):
                        index = int(path.rsplit("/", 1)[1])
                        if not 0 <= index < len(rows):
                            raise ValueError("잘못된 유사어 음원 번호입니다.")
                        from .common import bound_file
                        audio = bound_file(Path(state["project"]).parent, rows[index])
                        return self.send_file(audio)
                if path.startswith("/positive-preview/"):
                    project_id = query["project"][0]
                    base = f"?project={quote(project_id)}"
                    if path == "/positive-preview/":
                        return self.respond((resources / "reference_preview.html").read_bytes(), kind="text/html; charset=utf-8")
                    if path == "/positive-preview/queue":
                        start = int(query.get("start", ["0"])[0])
                        state, rows, total = manager.positive_preview_rows(project_id, start, 100)
                        labels = manager._call_voice_labels(state, [row for _, _, row in rows], numbers=[number for _, number, _ in rows])
                        return self.respond({"kind": "positive", "count": total, "start": start, "items": [{
                            "id": row["sample_id"], "text": row["text"], "number": number,
                            "group": "학습용" if row["split"] == "train" else "공통 평가용",
                            **label,
                            "audio": f"/positive-preview/audio/{index}" + base} for (index, number, row), label in zip(rows, labels)]})
                    if path.startswith("/positive-preview/audio/"):
                        index = int(path.rsplit("/", 1)[1])
                        state, rows, _ = manager.positive_preview_rows(project_id, index, 1)
                        from .common import bound_file
                        return self.send_file(bound_file(Path(state["project"]).parent, rows[0][2]))
                if path.startswith("/reference-preview/"):
                    project_id = query["project"][0]
                    base = f"?project={quote(project_id)}"
                    if path == "/reference-preview/":
                        manager._load(project_id)
                        html = (resources / "reference_preview.html").read_text(encoding="utf-8")
                        html = render_page_config(html, {"save_url": "/reference-preview/save" + base, "token": manager.token})
                        return self.respond(html.encode(), kind="text/html; charset=utf-8")
                    if path == "/reference-preview/queue":
                        return self.respond(manager.reference_preview_queue(project_id, batch_id=query.get("batch", [None])[0]))
                    if path.startswith("/reference-preview/candidate-audio/"):
                        candidate_id = path.rsplit("/", 1)[1]
                        return self.send_file(manager.reference_candidate_audio(project_id, candidate_id, query.get("rev", [None])[0]))
                    if path.startswith("/reference-preview/audio/"):
                        index = int(path.rsplit("/", 1)[1])
                        return self.send_file(manager.reference_preview_audio(project_id, index, query.get("rev", [None])[0]))
                if path.startswith("/review/"):
                    project_id, queue_id = query["project"][0], query["queue"][0]
                    state, review, rows = manager.review_queue(project_id, queue_id)
                    base = f"?project={project_id}&queue={quote(queue_id)}"
                    call_phase = manager._call_review_phase(state, review)
                    if call_phase:
                        revision = review["queue_sha256"]
                        if query.get("rev") != [revision]:
                            raise ValueError("검수할 음원이 바뀌었습니다. 화면을 새로고침하세요.")
                        base += f"&rev={revision}"
                    if path == "/review/":
                        html = (resources / "review.html").read_text(encoding="utf-8")
                        html = render_page_config(html, {"queue_url": "/review/queue" + base,
                            "save_url": "/review/save" + base, "token": manager.token,
                            "saved_message": "저장 완료 · 다음 단계를 준비합니다."})
                        return self.respond(html.encode(), kind="text/html; charset=utf-8")
                    if path == "/review/queue":
                        purpose = "합성 목소리가 자연스럽고 서로 다른지 확인하세요." if review["kind"] == "references" else f"'{state['wakeword']}'만 명확하게 들리고 다른 말이 없는지 확인하세요."
                        if review["kind"] == "content":
                            purpose = "안내한 종류의 소리가 정상인지 확인하세요. 호출어 자체가 들리거나 깨진 음원은 폐기하세요."
                        payload = {"purpose": review.get("description", purpose), "wakeword": state["wakeword"],
                                   "queue_sha256": digest(review["queue"]),
                                   "items": [{"id": r["sample_id"], "text": r.get("text", r.get("category", "배경음")),
                                              "group": r.get("review_group"), "instruction": r.get("review_instruction"),
                                              **({"voice_number": r["voice_number"]} if "voice_number" in r else {}),
                                              "audio": f"/review/audio/{i}" + base} for i, r in enumerate(rows)]}
                        if call_phase:
                            payload.update(review_phase=call_phase, require_listened_discard=True)
                            for item, label in zip(payload["items"], manager._call_voice_labels(state, rows)):
                                item.update(label)
                        if call_phase == "post" and state.get("inputs", {}).get("postgeneration_repair_rounds"):
                            repair = manager._postgeneration_repair_review(state, review)
                            by_id = {row["sample_id"]: row for row in rows}
                            payload.update(retained_decisions={sample: {"decision": answer["decision"], "listened": answer["listened"],
                                "audio_sha256": by_id[sample]["sha256"]} for sample, answer in repair.get("carried_decisions", {}).items()},
                                focus_sample_ids=repair.get("pending_ids", []))
                            for item, row in zip(payload["items"], rows):
                                item["audio_sha256"] = row["sha256"]
                        if call_phase == "pre":
                            draft = manager.preflight_draft(project_id, queue_id)
                            payload.update({key: value for key, value in draft.items() if key != "state"})
                            payload["require_listened_discard"] = False
                            for item, row in zip(payload["items"], rows):
                                item.update(audio_sha256=row["sha256"])
                        if review.get("pool_manifest"):
                            if review.get("evaluation_only") is True:
                                payload.update(policy_review=True, evaluation_only=True,
                                    intended_use="held_out_development_noise", pools=[],
                                    policy_text="이 자료를 학습에는 사용하지 않고, 학습과 분리한 소음 평가에만 사용하는 데 동의합니다.")
                            elif review.get("synthetic_review_policy") == HARD_SAMPLE_REVIEW_POLICY:
                                payload.update(policy_review=True, require_listened_discard=True,
                                    policy_text="정상으로 판정한 문구의 전체 음원을 생성하고, 자동 검사를 통과한 음원을 학습에 사용하는 데 동의합니다.")
                            elif review.get("synthetic_review_policy") == "one_per_phrase_v1":
                                payload.update(policy_review=True,
                                    policy_text="문구별 대표 음원을 듣고, 보존한 문구의 나머지 자동 검사 통과 음원도 학습에 사용하는 데 동의합니다. 폐기한 문구의 음원은 제외합니다.")
                            else:
                                payload.update(policy_review=True, policy_text="대표 음원 청취와 출처·기술 검사에 근거해 이 자료 묶음을 학습에 사용하는 데 동의합니다. 모든 원본을 들었다는 뜻은 아닙니다.")
                        return self.respond(payload)
                    if path.startswith("/review/audio/"):
                        index = int(path.rsplit("/", 1)[1])
                        if not 0 <= index < len(rows):
                            raise ValueError("잘못된 음원 번호입니다.")
                        audio = resolved(Path(state["project"]).parent, rows[index]["path"])
                        if digest(audio) != rows[index]["sha256"]:
                            raise ValueError("검수 음원이 변경되었습니다.")
                        return self.send_file(audio)
                if path.startswith("/results/"):
                    parts = path.split("/", 3)
                    state = manager._load(parts[2])
                    completed = state["results"].get("results", {})
                    site_path = completed.get("result", completed).get("site")
                    if not site_path:
                        raise ValueError("평가 결과 화면이 아직 준비되지 않았습니다.")
                    site = Path(site_path).parent
                    target = (site / (parts[3] or "index.html")).resolve()
                    if not target.is_relative_to(site.resolve()):
                        raise PermissionError("잘못된 결과 경로입니다.")
                    # View assets can improve without changing the archived
                    # evaluation or its original, immutable site export.
                    view_assets = {"index.html": "results_paged.html" if (site / "viewer-index.json").is_file() else "results.html",
                        **{name: name for name in ("results_ui.js", "results_scope.js", "results_paged.js", "results.js", "results.css", "audio_listening.css", "listening_player.js")}}
                    name = parts[3] or "index.html"
                    if (":" in name or "\\" in name
                            or any(part != part.rstrip(" .") for part in name.split("/"))):
                        raise PermissionError("잘못된 결과 리소스 이름입니다.")
                    if target.name.casefold() == "audio-references.json":
                        raise PermissionError("결과 음원 참조 목록은 직접 열 수 없습니다.")
                    if name.startswith("audio/"):
                        audio = manager.result_audio_reference(state, site, name, query=bool(url.query))
                        if audio is not None:
                            try:
                                return self.send_file(audio)
                            except OSError:
                                return self.respond({"error": "결과 원본 음원을 열 수 없습니다."}, 404)
                    if name in view_assets:
                        kind = "text/html; charset=utf-8" if name.endswith("html") else "text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8"
                        return self.respond((resources / view_assets[name]).read_bytes(), kind=kind)
                    if name == "results-data.js":
                        data = target.read_bytes()
                        recommendation = manager.runtime_recommendations(state)
                        if recommendation:
                            data += ("\nwindow.KOWW_RESULTS.report.runtime_recommendations=" +
                                     json.dumps(recommendation, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c") + ";\n").encode("utf-8")
                        return self.respond(data, kind="text/javascript; charset=utf-8")
                    return self.send_file(target)
                if path.startswith("/mic/"):
                    return self.mic_get(path)
                if path.startswith("/analysis/"):
                    return self.analysis_get(path)
                self.respond({"error": "페이지를 찾을 수 없습니다."}, 404)
            except ConnectionError:
                return  # A cancelled read has no recipient for an error response.
            except PermissionError as error:
                self.respond({"error": str(error)}, 403)
            except (ValueError, KeyError, OSError, IndexError, RuntimeError, ImportError) as error:
                self.respond({"error": str(error)}, getattr(error, "status", 400))

        def send_file(self, path):
            if not path.is_file():
                return self.respond({"error": "파일이 없습니다."}, 404)
            # Stream long evaluation WAVs; do not retain them in server RAM.
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)

        def mic_get(self, path):
            parts = path.split("/", 3)
            project_id, rest = parts[2], parts[3] if len(parts) > 3 else ""
            prefix = "/mic/" + project_id
            if not rest and not path.endswith("/"):
                self.send_response(308)
                self.send_header("Location", "/mic/" + quote(project_id, safe="") + "/")
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            if rest == "api/info":
                if manager.analysis.busy(project_id):
                    raise ValueError("모델 비교·선택 작업을 마친 뒤 마이크를 연결하세요.")
                return self.respond(manager.microphone(project_id).info())
            names = {"": "microphone.html", "microphone.js": "microphone.js", "microphone.css": "microphone.css", "microphone_worklet.js": "microphone_worklet.js"}
            if rest not in names:
                return self.respond({"error": "Not found"}, 404)
            name = names[rest]
            content = (resources / name).read_bytes()
            if name == "microphone.html":
                content = render_page_config(content.decode("utf-8"), {"api_base": prefix,
                    "worklet_url": prefix + "/microphone_worklet.js"}).encode("utf-8")
            kind = "text/html; charset=utf-8" if name.endswith("html") else "text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8"
            return self.respond(content, kind=kind)

        def analysis_get(self, path):
            parts = path.split("/", 3)
            project_id, rest = parts[2], parts[3] if len(parts) > 3 else ""
            manager._load(project_id)
            if rest == "api/status":
                return self.respond(manager.analysis.status(project_id))
            if rest.startswith("download/"):
                return self.send_file(manager.analysis.download(project_id, rest.split("/", 1)[1]))
            if not rest:
                # Old bookmarks open the consolidated trial screen. Historical
                # analysis records and authenticated APIs remain available.
                self.send_response(308)
                self.send_header("Location", "/mic/" + quote(project_id, safe="") + "/")
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            return self.respond({"error": "페이지를 찾을 수 없습니다."}, 404)

        def do_POST(self):
            try:
                path = urlsplit(self.path).path
                if path.startswith("/mic/"):
                    self.guard()
                    project_id, action = path.split("/")[2], path.rsplit("/", 1)[1]
                    service = manager.microphone(project_id)
                    self.guard(mutate=True, token=service.token)
                    if action == "frame":
                        payload = self.body(max_bytes=4096, kind="application/octet-stream")
                        return self.respond(service.process_frame(self.headers.get("X-Koww-Session"), int(self.headers.get("X-Koww-Sequence", "-1")), payload))
                    payload = self.body(max_bytes=4096)
                    if action == "start":
                        with manager._lock:
                            if manager.analysis.busy(project_id):
                                raise ValueError("모델 비교·선택 작업을 마친 뒤 마이크를 연결하세요.")
                            service = manager.microphone(project_id)
                            self.guard(mutate=True, token=service.token)
                            return self.respond(service.start(payload))
                    if action == "stop":
                        return self.respond(service.stop(payload.get("session_id")))
                    raise ValueError("Unsupported microphone action")
                self.guard(mutate=True)
                payload = self.body()
                if path.startswith("/analysis/"):
                    parts = path.split("/")
                    if len(parts) != 5 or parts[3] != "api" or parts[4] not in ("compare", "select"):
                        raise ValueError("지원하지 않는 분석 작업입니다.")
                    with manager._lock:
                        if any(manager._reference_review_job_alive(manager._load(project_id))
                               for project_id in manager.registry["projects"]):
                            raise ValueError("기준 목소리 후보 준비·생성이 끝난 뒤 모델을 비교하세요.")
                        manager._stop_reference_worker()
                        return self.respond(manager.analysis.launch(parts[2], parts[4], payload))
                if path == "/api/projects":
                    return self.respond(manager.create(payload))
                if path == "/api/model-location":
                    return self.respond(manager.model_location(payload))
                if path == "/review/save":
                    query = parse_qs(urlsplit(self.path).query)
                    return self.respond(manager.save_review(query["project"][0], query["queue"][0], payload))
                if path == "/reference-preview/save":
                    query = parse_qs(urlsplit(self.path).query)
                    return self.respond(manager.reference_preview_queue(query["project"][0], payload,
                                                                         batch_id=query.get("batch", [None])[0]))
                if path.startswith("/api/projects/"):
                    project_id, action = path.split("/")[3:5]
                    if action == "retry_postgeneration":
                        return self.respond(manager.retry_postgeneration(project_id, payload))
                    methods = {"start": manager.start, "resume": manager.start, "pause": manager.pause,
                               "prepare_start": manager.prepare_start, "prepare_pause": manager.prepare_pause,
                               "prepare_resume": manager.prepare_resume,
                               "migrate": manager.migrate}
                    if action == "input":
                        return self.respond(manager.submit_input(project_id, payload))
                    if action in methods:
                        return self.respond(methods[action](project_id))
                raise ValueError("Unsupported action")
            except PermissionError as error:
                self.respond({"error": str(error)}, 403)
            except (ValueError, KeyError, OSError, IndexError, RuntimeError) as error:
                self.respond({"error": str(error)}, getattr(error, "status", 400))

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def service_actions(self):
            for service in list(manager._microphones.values()):
                service.expire()

    return Server(("127.0.0.1", port), Handler)


def serve_studio(workspace=None, port=8793, open_browser=True):
    if not 1 <= port <= 65535:
        raise ValueError("포트는 1~65535 사이여야 합니다.")
    with workspace_lease(workspace or Path.cwd()):
        _serve_studio(workspace or Path.cwd(), port, open_browser)


def _serve_studio(workspace, port, open_browser):
    manager = StudioManager(workspace)
    try:
        server = make_studio_server(manager, port)
    except OSError as error:
        raise RuntimeError(f"포트 {port}를 열 수 없습니다. 이미 열린 화면을 사용하거나 --port로 변경하세요.") from error
    manager.run()
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"한국어 호출어 학습: {url}\n작업 폴더: {manager.workspace}\n종료: Ctrl+C · 진행 상황은 저장됩니다.", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        server.server_close()
