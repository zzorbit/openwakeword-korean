"""Optional, project-owned CUDA runtime; never replace the base CPU wheel.

The managed payload is qualified on Windows x64 / CPython 3.12. Other systems
can use a separately supplied CUDA provider; this installer does not guess their
native dependencies. Native validation runs in a fresh subprocess, so an already
imported CPU onnxruntime in the caller stays intact.
"""
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
import marshal
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
from types import CodeType
import uuid

from .common import digest, now, read_json, write_json
from .storage import storage_environment, storage_paths


PINS = {"onnxruntime-gpu": "1.31.0", "nvidia-cublas": "13.8.0.4",
        "nvidia-cuda-runtime": "13.4.92", "nvidia-cudnn-cu13": "9.27.0.42"}
RUNTIME_NAME = "onnxruntime_1_31_cuda13"
# Measured unpacked qualified Windows payload: ORT 242,971,855 + NVIDIA
# 1,186,284,164 bytes. One additional payload is an explicit temporary pip
# download/unpack allowance, not a claim about every pip version's exact peak.
REFERENCE_PAYLOAD_BYTES = 1429256019
MINIMUM_FREE_BYTES = REFERENCE_PAYLOAD_BYTES * 2
_REQUIRED_NATIVE = {"onnxruntime_pybind11_state.pyd", "onnxruntime_providers_cuda.dll",
                    "cublas64_13.dll", "cublaslt64_13.dll", "cudart64_13.dll", "cudnn64_9.dll"}
_PROBE = r'''
import base64, json, os, sys
sys.path.insert(0, sys.argv[1])
paths = [p for p in sys.argv[2].split(os.pathsep) if p]
handles = [os.add_dll_directory(p) for p in paths] if os.name == 'nt' else []
os.environ['PATH'] = os.pathsep.join(paths) + os.pathsep + os.environ.get('PATH', '')
if 'onnxruntime' in sys.modules:
    raise RuntimeError('CPU onnxruntime was imported before the isolated runtime')
import numpy as np
import onnxruntime as ort
if ort.__version__ != '1.31.0':
    raise RuntimeError('Unexpected isolated ONNX Runtime version')
ort.preload_dlls(cuda=False, cudnn=False, msvc=True)
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.log_severity_level = 3
options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
model = base64.b64decode('CAg68gEKEgoBeAoBdxICbW0iBk1hdE11bAobCgVpbWFnZQoGa2VybmVsEgRjb252IgRDb252EhxxdWFsaWZpZWRfY3VkYV9ydW50aW1lX3Byb2JlKhsIAggCEAEiEAAAgD8AAIA/AACAPwAAgD9CAXcqGAgBCAEIAQgBEAEiBAAAgD9CBmtlcm5lbFoTCgF4Eg4KDAgBEggKAggCCgIIAlofCgVpbWFnZRIWChQIARIQCgIIAQoCCAEKAggDCgIIA2IUCgJtbRIOCgwIARIICgIIAgoCCAJiHgoEY29udhIWChQIARIQCgIIAQoCCAEKAggDCgIIA0IECgAQDQ==')
session = ort.InferenceSession(model, sess_options=options,
    providers=[('CUDAExecutionProvider', {'use_tf32': '0', 'cudnn_conv_algo_search': 'DEFAULT'})])
if session.get_providers()[0] != 'CUDAExecutionProvider':
    raise RuntimeError('Isolated CUDA probe fell back to CPU')
result = session.run(None, {'x': np.ones((2,2), np.float32), 'image': np.ones((1,1,3,3), np.float32)})
if not (np.array_equal(result[0], np.full((2,2), 2, np.float32)) and np.array_equal(result[1], np.ones((1,1,3,3), np.float32))):
    raise RuntimeError('Isolated CUDA MatMul/Conv probe produced unexpected output')
print(json.dumps({'version': ort.__version__, 'module': ort.__file__, 'providers': session.get_providers(),
    'cuda_session_validated': True, 'cpu_fallback_disabled': True, 'operators': ['MatMul','Conv']}), flush=True)
'''


def _managed_reason():
    if sys.platform != "win32":
        return "Managed CUDA 13 libraries are qualified on Windows x64 only; use an existing, separately installed CUDA provider on this platform."
    if platform.machine().lower() not in ("amd64", "x86_64") or sys.version_info[:2] != (3, 12):
        return "The managed CUDA 13 payload is qualified for CPython 3.12 on Windows x64; configure a separate compatible CUDA runtime for this interpreter."
    return None


def _target(project_root):
    root = Path(project_root).resolve()
    if any(part.lower() in (".codex", ".git") for part in root.parts):
        raise ValueError("GPU runtime storage must stay in the selected project/cache")
    target = storage_paths(root)["root"] / "gpu_runtime" / RUNTIME_NAME
    if any(part.lower() == ".git" for part in target.parts):
        raise ValueError("GPU runtime storage must not be inside Git metadata")
    for name in ("purelib", "platlib"):
        library = sysconfig.get_paths().get(name)
        if library and target.is_relative_to(Path(library).resolve()):
            raise ValueError("GPU runtime must use a separate project cache, not Python site-packages")
    return target


def _versions(target):
    return {dist.metadata["Name"].lower().replace("_", "-"): dist.version
            for dist in importlib.metadata.distributions(path=[str(target)]) if dist.metadata.get("Name")}


def _dll_paths(target):
    return sorted({str(path.parent.resolve()) for path in (target / "nvidia").rglob("*.dll")})


def _validate_layout(target):
    versions = _versions(target)
    if versions != PINS:
        raise ValueError("The isolated GPU runtime package versions differ or contain a CPU-wheel collision")
    if not (target / "onnxruntime/__init__.py").is_file():
        raise ValueError("The isolated GPU runtime package is missing")
    native = {path.name.lower() for path in target.rglob("*") if path.is_file() and path.suffix.lower() in (".dll", ".pyd")}
    if not _REQUIRED_NATIVE <= native:
        raise ValueError("The isolated GPU runtime is missing required CUDA/cuDNN/provider libraries")
    return versions


def _environment(target):
    return {"KOWW_GPU_RUNTIME_PATH": str(target.resolve()), "KOWW_GPU_DLL_PATHS": os.pathsep.join(_dll_paths(target))}


def gpu_runtime_environment(project_root):
    """Locate a completed managed runtime without importing native modules.

    This is a locator, not a certification of the GPU or current native bytes;
    setup verifies recorded hashes and the resolver probes actual CUDA sessions.
    """
    if _managed_reason() is not None:
        return {}
    target = _target(project_root)
    receipt = target / "receipt.json"
    if not receipt.is_file():
        return {}
    try:
        record = read_json(receipt)
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict):
        return {}
    if record.get("kind") != "project_isolated_cuda_runtime_v1" or record.get("state") != "completed" or record.get("versions") != PINS:
        return {}
    return _environment(target) if (target / "onnxruntime/__init__.py").is_file() else {}


@contextmanager
def _install_lock(path):
    """OS releases this lock after a crash; no stale PID lock to steal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _probe_runtime(target, environment):
    command = [sys.executable, "-I", "-B", "-c", _PROBE, str(target), environment["KOWW_GPU_DLL_PATHS"]]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", timeout=120,
            env=dict(os.environ, **environment))
    except subprocess.CalledProcessError as error:
        detail = ((error.stderr or "") + "\n" + (error.stdout or "")).strip()[-8000:]
        raise ValueError("The separate GPU runtime failed its CUDA session probe: " + detail) from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("The separate GPU runtime CUDA session probe exceeded 120 seconds") from error
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    probe = json.loads(lines[-1]) if lines else {}
    module = Path(probe.get("module", "")).resolve()
    if (probe.get("version") != PINS["onnxruntime-gpu"] or probe.get("cuda_session_validated") is not True
            or probe.get("cpu_fallback_disabled") is not True or probe.get("providers", [])[:1] != ["CUDAExecutionProvider"]
            or not module.is_relative_to((target / "onnxruntime").resolve())):
        raise ValueError("The separate GPU runtime did not pass its CUDA session probe")
    return probe


def _artifacts(target):
    return [{"path": path.relative_to(target).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(target.rglob("*")) if path.is_file() and path != target / "receipt.json"]


def _source_code(code):
    # Moving the staged package changes filenames, not executable bytecode.
    return code.replace(co_filename="", co_consts=tuple(_source_code(value) if isinstance(value, CodeType) else value
                                                        for value in code.co_consts))


def _generated_bytecode(target, relative, recorded_files):
    """Allow only a standard import cache matching an already bound source."""
    path = target / relative
    if path.parent.name != "__pycache__" or path.suffix != ".pyc":
        return False
    try:
        source = Path(importlib.util.source_from_cache(str(path)))
        if source.relative_to(target).as_posix() not in recorded_files:
            return False
        candidates = {Path(importlib.util.cache_from_source(str(source), optimization=level)): level
                      for level in ("", "1", "2")}
        if path not in candidates:
            return False
        data = path.read_bytes()
        if data[:4] != importlib.util.MAGIC_NUMBER or int.from_bytes(data[4:8], "little") not in (0, 1, 3):
            return False
        code = marshal.loads(data[16:])
        expected = compile(source.read_bytes(), str(source), "exec", dont_inherit=True,
                           optimize=int(candidates[path] or "0"))
        return isinstance(code, CodeType) and _source_code(code) == _source_code(expected)
    except (OSError, ValueError, EOFError, TypeError, SyntaxError):
        return False


def _verify_receipt(target):
    try:
        receipt = read_json(target / "receipt.json")
    except (OSError, ValueError) as error:
        raise ValueError("Existing isolated GPU runtime has no readable receipt; it was preserved") from error
    if not isinstance(receipt, dict):
        raise ValueError("Existing isolated GPU runtime has an invalid receipt; it was preserved")
    if receipt.get("kind") != "project_isolated_cuda_runtime_v1" or receipt.get("state") != "completed" or receipt.get("versions") != PINS:
        raise ValueError("Existing isolated GPU runtime is unverified; it was preserved")
    if _validate_layout(target) != PINS:
        raise ValueError("Existing isolated GPU runtime metadata changed; it was preserved")
    artifacts = receipt.get("artifacts")
    if (not isinstance(artifacts, list) or not artifacts or any(not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("bytes"), int) or not isinstance(entry.get("sha256"), str) for entry in artifacts)
            or len({entry["path"] for entry in artifacts}) != len(artifacts)):
        raise ValueError("Existing GPU runtime has no complete integrity receipt")
    actual_files = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file() and path != target / "receipt.json"}
    recorded_files = {entry["path"] for entry in artifacts}
    if not recorded_files <= actual_files:
        raise ValueError("Existing isolated GPU runtime inventory changed; it was preserved")
    for entry in artifacts:
        path = (target / entry["path"]).resolve()
        if not path.is_relative_to(target.resolve()) or not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Existing isolated GPU runtime bytes changed; it was preserved")
    if any(not _generated_bytecode(target, relative, recorded_files) for relative in actual_files - recorded_files):
        raise ValueError("Existing isolated GPU runtime has unverified extra files; it was preserved")
    return receipt


def setup_gpu_runtime(project_root):
    """Install only into a fresh project-cache target; leave CPU ORT untouched.

    Failed staging directories remain for diagnosis. The successful canonical
    target is never overwritten, including when its verification fails.
    """
    root = Path(project_root).resolve()
    target = _target(root)
    reason = _managed_reason()
    if reason:
        return {"state": "unsupported", "path": str(target), "managed": False, "reason": reason,
                "scope": "Existing CUDA provider environments remain supported by the runtime resolver; no packages were installed."}
    if sys.prefix == sys.base_prefix:
        raise ValueError("GPU runtime automatic setup requires the library's virtual environment")
    target.parent.mkdir(parents=True, exist_ok=True)
    with _install_lock(target.parent / (RUNTIME_NAME + ".install.lock")), storage_environment(root):
        if target.exists():
            receipt = _verify_receipt(target)
            probe = _probe_runtime(target, _environment(target))
            return {**receipt, "reused": True, "current_probe": probe}
        free = shutil.disk_usage(target.parent).free
        if free < MINIMUM_FREE_BYTES:
            raise ValueError(f"Insufficient selected-cache space for isolated GPU runtime setup: {free} free, {MINIMUM_FREE_BYTES} bytes required by the measured payload plus pip temporary allowance")
        staging = target.with_name(target.name + ".installing_" + uuid.uuid4().hex)
        staging.mkdir(exist_ok=False)
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                   "--only-binary=:all:", "--no-deps", "--no-compile", "--target", str(staging),
                   "--index-url", "https://pypi.org/simple", *[name + "==" + version for name, version in PINS.items()]]
        print("별도 GPU 실행 환경을 프로젝트 캐시에 설치합니다. 기본 CPU onnxruntime은 유지합니다.", flush=True)
        subprocess.run(command, cwd=str(root), check=True)
        versions = _validate_layout(staging)
        probe = _probe_runtime(staging, _environment(staging))
        receipt = {"kind": "project_isolated_cuda_runtime_v1", "state": "completed", "created_at": now(),
            "path": str(target), "versions": versions, "commands": [command], "reused": False,
            "platform": "win_amd64_cp312", "cuda_session_probe": probe, "artifacts": _artifacts(staging),
            "probe_path_scope": "Probe module path records the fresh staging directory before atomic publication; artifact hashes bind the published payload.",
            "storage_reference_installed_bytes": REFERENCE_PAYLOAD_BYTES, "setup_minimum_free_bytes": MINIMUM_FREE_BYTES,
            "scope": "Separate project/cache native payload; no base Python package, parent sys.path or machine-wide environment changed. CUDA MatMul/Conv probe is not a full model-performance qualification.",
            "native_payload_scope": "Pinned libraries from the qualified precompiled CUDA path; optional NVRTC runtime compilation libraries are not installed. Unsupported native configurations fail the actual session probe."}
        write_json(staging / "receipt.json", receipt)
        if target.exists():
            raise RuntimeError("GPU target appeared during setup; both targets were preserved")
        staging.rename(target)
        return receipt
