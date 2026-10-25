"""Local training workspace: persistent orchestration around the public library.

The HTTP process never runs TTS/training itself. A single identified subprocess
executes one stage, publishes its result atomically, and can outlive the page.
Human answers remain real, hash-bound review evidence, never a job status flag.
"""
from datetime import datetime
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
import ast
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .common import digest, identity, now, read_json, read_rows, resolved, write_json
from .project import create_project, load_project
from .web_page import render_page_config


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
    """Different benchmark versions need separate runs; old IDs stay resumable."""
    return {"wakeword": word, "preset": preset,
            "evaluation_policy": (settings.get("evaluation_policy") or {}).get("id"),
            "voice_design_policy": settings.get("voice_design_policy"),
            **{key: settings[key] for key in ("originals", "rounds", "updates")},
            **{key: float(settings[key]) for key in ("generic_hours", "korean_hours")}}


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
    return identity({path.relative_to(root).as_posix(): digest(path) for path in paths})


def _execution_contract():
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
    # Stage-only modules cannot affect an already finished stage. Unknown new
    # modules remain shared, so adding a dependency never silently bypasses the
    # compatibility gate.
    groups = {
        "data": {"automatic_sources.py", "canonical_sources.py", "commercial_sources.py", "compact_sources.py",
                 "downloads.py", "download_control.py", "evaluation_materials.py", "fma_download.py", "fma_selection.py", "fma_streaming.py",
                 "material_downloads.py", "material_policy.py", "provider_index.py", "range_zip.py", "source_waveform_index.py",
                 "studio_data.py", "studio_evaluation_data.py", "pool_review.py"},
        "tts": {"reference_bank.py", "reference_qc.py", "pilot.py", "hard_tts.py", "tts.py", "tts_runtime.py", "tts_cuda_graph.py",
                "tts_cuda_predictor.py", "tts_cuda_sampling.py"},
        "features": {"features.py", "paired.py", "preparation.py"},
        "training": {"train.py"},
        "models": {"bundle.py", "final_integrity.py", "runtime.py", "events.py"},
        "results": {"selection.py", "cache_analysis.py"},
        "evaluation": {"evaluate.py", "evaluation_backend.py", "evaluation_coverage.py", "evaluation_rir.py",
                       "evaluation_workers.py", "gpu_evaluation.py", "negative_streams.py", "tune.py"}}
    stage_groups = {
        "frontend": set(), "data_download": {"data"}, "data_review": {"data"}, "data_connect": {"data"},
        "references": {"tts"}, "references_review": {"tts"}, "pilot": {"tts"}, "pilot_review": {"tts"},
        "positives": {"tts"}, "hard_phrases": {"tts"}, "hard_generation": {"tts"}, "hard_review": {"tts", "data"},
        "hard_connect": {"tts", "data"}, "features": {"features"}, "training": {"training"},
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
    return {"schema_version": 1, "computation_sha256": identity(bindings), "files": bindings, "stages": stages}


# Exact 0.14.3 source identity and computational fingerprint, audited before
# adding migration support. Other snapshot-less versions are never guessed.
_LEGACY_COMPATIBLE_CONTRACTS = {
    "9d05b6d391a18920f28c964cf8f12c26faec7c5f98152688c73a5a499b4dd141":
        "28f88142c4ab88361199fd2fae2ec1fca1ba863412c206fb0cd5e6b5515def14"}


# An acquisition-only update still changes computational files. It requires an
# audited, exact source/target pair, not a blanket exception for download code.
# Final release fingerprints and the preserved-input equivalence receipt are
# filled only after the complete source tree has been frozen and tested.
_AUDITED_ACQUISITION_UPDATES = {
    "04d4bd5dcfa9667ff2a650e4163007181fafe9b0cdd2e8cc72ad3e9526380d04": {
        "source_stages_sha256": "aef7b4226a131faa19df24a46220015e901d696f3c130fb67064114fbb2b38de",
        "target_computation_sha256": "9074dae60dd74e2439198457ecbfb639df5a418831a2d73937feae53ee22de70",
        "target_stages_sha256": "a01c4fcc065be2c5dd003f82fbe9c3d630a3e7a840d4c0da935c2551b51c702c",
        "changed_files": ('downloads.py', 'evaluation_materials.py', 'evaluation_rir.py', 'material_downloads.py', 'official_music.py', 'studio_data.py', 'studio_steps.py'),
        "evidence_sha256": "250073e4bae086eefaf11e1a00a57baf169fcbff8376a0dea0e4ff12196c818c",
    },
}
_ACQUISITION_UPDATE_FILES = {
    "downloads.py", "download_control.py", "material_downloads.py", "studio_data.py",
    "evaluation_materials.py", "evaluation_rir.py", "studio_evaluation_data.py", "official_music.py", "studio_steps.py",
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
    """Return a release-bound audit only for its fully pinned acquisition diff."""
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


class _ProgressReadPending(RuntimeError):
    """A live progress view is temporarily unreadable; no execution failure."""


def _read_optional_progress(path):
    """Defer access-denied live views only, never frozen execution evidence."""
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}  # It can disappear after the caller's existence check.
    except PermissionError as error:
        raise _ProgressReadPending(
            "진행 정보를 잠시 읽지 못했습니다. 마지막 상태를 유지하고 자동으로 다시 확인합니다.") from error


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
        self._implementation_id = _implementation()
        self._execution_contract = _execution_contract()
        self._timing_receipts = {}
        self._microphones = {}
        self._journals = {}
        self._rates = {}
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
        from .studio_steps import preset_settings
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
            word, preset, settings, plan = self._model_request(payload)
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
        if previous:
            if (previous.get("schema_version") != 1 or previous.get("settings_sha256") != identity(state["settings"])
                    or previous.get("plan_sha256") != identity(state["plan"])):
                return blocked("저장된 설정 또는 단계 계약이 달라 자동으로 이관할 수 없습니다. 기존 자료는 보존됩니다.")
            remaining = [entry["id"] for entry in state["plan"][state["index"]:]]
            changed = [stage for stage in remaining if not previous.get("stages", {}).get(stage)
                       or previous["stages"][stage] != self._execution_contract["stages"].get(stage)]
            if changed:
                audited_update = _audited_acquisition_update(previous, self._execution_contract)
                reference_update = _audited_reference_update(previous, self._execution_contract)
                reference_admission_update = _audited_reference_admission_update(previous, self._execution_contract)
                pilot_update = _audited_pilot_update(previous, self._execution_contract)
                if pilot_update and not self._can_update_pilot(state):
                    return blocked("목소리별 발음 검수 수량 변경은 대량 생성 전 작업에만 적용할 수 있습니다. 기존 음원과 검수 기록은 보존됩니다.")
                if reference_admission_update and not self._can_update_reference_admission(state):
                    return blocked("기준 목소리 자동 검사는 호출어 생성 전 작업에만 적용할 수 있습니다. 기존 자료와 검수 기록은 보존됩니다.")
                if audited_update is None and reference_update is None and reference_admission_update is None and pilot_update is None:
                    return blocked("남은 단계의 자료·생성·학습·채점 코드가 달라 호환 이관할 수 없습니다. 기존 자료를 보존하고 새 실험을 만드세요.")
        elif (_LEGACY_COMPATIBLE_CONTRACTS.get(state.get("implementation_id"))
              != self._execution_contract["computation_sha256"]):
            return blocked("이 구버전에는 검증된 단계 계약이 없습니다. 기존 자료를 보존하고 별도 호환 검증이 필요합니다.")
        budget_change = self._reference_budget_change(state) if reference_update else None
        return {"state": "available", "action": "migrate",
                "reason": ("기존 음원과 검수 기록을 보존하고 학습용 목소리마다 호출어 1개씩 확인하도록 목록을 갱신합니다."
                           if pilot_update else
                           "기존 목소리와 검수 기록을 보존하고 기준 목소리를 자동 검사합니다. 호출어 발음 대표 검수는 유지합니다."
                           if reference_admission_update else
                           "기존 목소리·검수 결과를 보존하고 부족한 학습용 기준 목소리만 추가합니다. 호출어 생성 전 작업에만 수량 변경을 적용합니다."
                           if reference_update else
                           "다운로드 순서·진행 표시 변경의 자료 동일성을 확인한 업데이트입니다. 기존 설정과 완료 결과를 검증한 뒤 적용할 수 있습니다."
                           if audited_update else "남은 단계의 계산·자료 계약이 같습니다. 기존 설정과 완료 결과를 검증하고 업데이트를 적용할 수 있습니다."),
                **({"audited_acquisition_update": audited_update} if audited_update else {}),
                **({"audited_reference_update": reference_update} if reference_update else {}),
                **({"audited_reference_admission_update": reference_admission_update} if reference_admission_update else {}),
                **({"audited_pilot_update": pilot_update} if pilot_update else {}),
                **({"reference_budget_change": budget_change} if budget_change else {}),
                "preserves": ["project_settings", "completed_artifacts", "review_answers", "history"]}

    def _can_update_reference_admission(self, state):
        positions = {stage["id"]: i for i, stage in enumerate(state["plan"])}
        if state["index"] > positions["pilot"] or self._job_alive(state.get("job") or {}):
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
                if self._job_alive(self._load(other_id).get("job") or {}):
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
                **({"reference_budget_change": compatibility["reference_budget_change"]}
                   if compatibility.get("reference_budget_change") else {}),
                "preserved": compatibility["preserves"]})
            budget_change = compatibility.get("reference_budget_change")
            if compatibility.get("audited_pilot_update"):
                from .studio_steps import workflow_plan
                state["settings"]["pilot_review_policy"] = "one_per_training_voice_v1"
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
        for entry in state.get("history", []):
            if digest(entry["result"]) != entry["sha256"]:
                raise ValueError("이전 단계 완료 기록이 변경되었습니다.")
            completed = read_json(entry["result"])
            for binding in completed.get("artifact_bindings", []):
                if digest(binding["path"]) != binding["sha256"]:
                    raise ValueError("이전 단계의 결과 파일이 변경되었습니다: " + binding["path"])

    def _launch(self, state):
        self._guard_config(state)
        stage = state["plan"][state["index"]]
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
            else:
                state.update(status="error", error="지원하지 않는 단계 종료 상태: " + str(phase))
        state.setdefault("history", []).append({"stage": job["stage"], "started_at": job["started_at"],
                                                "completed_at": now(), "result": str(path), "sha256": digest(path)})
        self._save(state)
        return True

    def start(self, project_id):
        with self._lock:
            state = self._load(project_id)
            if state["status"] == "completed":
                return self.describe(state)
            self._guard_config(state)
            if state["status"] in ("running", "pausing") and self._job_alive(state.get("job") or {}):
                state["auto_continue"] = True
                self._save(state)
                return self.describe(state)
            for other_id in self.registry["projects"]:
                other = self._load(other_id)
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
                if state["status"] in ("waiting_review", "waiting_input", "completed", "error"):
                    return self.describe(state)
            self._recover_dead_locks(state)
            self._launch(state)
            return self.describe(state)

    def pause(self, project_id):
        with self._lock:
            state = self._load(project_id)
            state["auto_continue"] = False
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
            for project_id in self.registry["projects"]:
                state = self._load(project_id)
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
        for service in list(self._microphones.values()):
            with service._lock:
                service._close()

    def submit_input(self, project_id, payload):
        with self._lock:
            state = self._load(project_id)
            if state["status"] != "waiting_input":
                raise ValueError("현재 입력을 기다리는 단계가 아닙니다.")
            if "phrases" in payload:
                phrases = payload["phrases"]
                if not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) for p in phrases):
                    raise ValueError("유사어를 한 줄에 하나씩 입력하세요.")
                from .hard_phrase_policy import validate_noncall_text
                _, config = load_project(state["project"])
                for phrase in phrases:
                    validate_noncall_text(phrase, config["wakeword"])
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

    def review_queue(self, project_id, queue_id):
        state = self._load(project_id)
        review = next((r for r in self.review_descriptors(state) if r["id"] == queue_id), None)
        if not review:
            raise ValueError("현재 검수 목록이 아닙니다. 화면을 새로고침하세요.")
        path = Path(review["queue"])
        if digest(path) != review.get("queue_sha256") or (review.get("pool_manifest") and
                digest(review["pool_manifest"]) != review.get("pool_sha256")):
            raise ValueError("검수 목록 또는 승인할 원본 목록이 변경되었습니다.")
        rows = read_rows(path)
        if not rows:
            raise ValueError("검수할 음원이 없습니다.")
        return state, review, rows

    def save_review(self, project_id, queue_id, payload):
        from .review import apply_review
        with self._lock:
            state, review, rows = self.review_queue(project_id, queue_id)
            evaluation_only = review.get("evaluation_only") is True
            if evaluation_only and (not review.get("pool_manifest") or review.get("pools") != []):
                raise ValueError("평가 전용 검수 자료는 학습 풀에 연결할 수 없습니다.")
            decisions = payload.get("decisions", payload)
            if (not isinstance(decisions, dict) or set(decisions) != {r["sample_id"] for r in rows}
                    or any(not isinstance(a, dict) or a.get("decision") not in ("keep", "discard", "review")
                           or a.get("decision") == "keep" and a.get("listened") is not True for a in decisions.values())):
                raise ValueError("모든 항목을 듣고 보존하거나 폐기·재검수를 표시하세요.")
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
                result = apply_review(state["project"], review["queue"], decisions, review["kind"])
            if any(a.get("decision") == "review" for a in decisions.values()) or (review["kind"] == "pilot" and result.get("policy_approved") is not True):
                state.update(next_action="재검수 또는 폐기 항목을 확인하세요. 승인 전에는 다음 작업을 시작하지 않습니다.")
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
            if review.get("pool_manifest"):
                answer["source_manifest_sha256"] = digest(review["pool_manifest"])
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
            return result

    def reference_preview_rows(self, project_id):
        state = self._load(project_id)
        root = Path(state["project"]).parent
        bank = read_json(root / "voices.json")["families"]
        rows = []
        for split, key in (("train", "train_references"), ("validation", "validation_references")):
            selected = {row["family_id"] for row in [row for row in bank if row["split"] == split][:state["settings"][key]]}
            for row in read_rows(root / "data/references" / split / "references.jsonl"):
                if row.get("voice_family_id") in selected:
                    if (row.get("split") != split or row.get("provenance") != "qwen_synthetic_reference"
                            or row.get("human_reference_used") is not False):
                        raise ValueError("기준 목소리 출처가 올바르지 않습니다.")
                    rows.append(row)
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("기준 목소리 목록에 중복 번호가 있습니다.")
        return state, rows

    @staticmethod
    def _voice_design(bank, settings):
        from .voice_design import summarize_voice_design
        return {"policy": bank.get("policy", "legacy"),
                "counts": {split: summarize_voice_design(
                    [row for row in bank["families"] if row["split"] == split][:settings[key]])
                    for split, key in (("train", "train_references"), ("validation", "validation_references"))},
                "scope": "requested_conditions_not_measured_voices"}

    def describe(self, state):
        index, plan = state["index"], state["plan"]
        stage = plan[index] if index < len(plan) else {"id": "completed", "label": "완료"}
        view = {key: state.get(key) for key in ("id", "wakeword", "preset", "status", "error", "next_action", "created_at")}
        timing = self._timing(state)
        view.update(directory=str(Path(state["project"]).parent), stage=stage, stage_id=stage["id"],
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
        if sum(preview_counts.values()):
            view["reference_preview"] = {"url": f"/reference-preview/?project={state['id']}",
                                         "count": sum(preview_counts.values()), **preview_counts}
        job = state.get("job") or {}
        if job:
            view["current_stage_elapsed_seconds"] = round(_elapsed(job.get("started_at")), 1)
            view["process_alive"] = self._job_alive(job)
            progress = Path(state["project"]).parent / ".studio" / "task_progress.json"
            if progress.exists():
                record = _read_optional_progress(progress)
                if record.get("task_id") == job["task_id"]:
                    view["progress"] = record
                    view["eta_seconds"] = record.get("eta_seconds")
            material_progress = Path(state["project"]).parent / ".studio" / "progress.json"
            if job["stage"] == "data_download" and material_progress.exists():
                record = _read_optional_progress(material_progress)
                if record.get("stage") == "data_download" and record.get("updated_at", "") >= job["started_at"]:
                    view["progress"] = record
                    view["eta_seconds"] = record.get("eta_seconds")
            if job["stage"] == "training":
                progress_files = list((Path(state["project"]).parent / "runs" / "studio_run").glob("member*/progress.json"))
                records = [_read_optional_progress(p) for p in progress_files]
                done = sum(r.get("completed_updates", r.get("update", 0)) for r in records)
                view["progress"] = {"done": done, "total": state["settings"]["updates"] * 3,
                                    "unit": "학습 업데이트", "label": "모델 3개 누적 학습량"}
            if job["stage"] in ("references", "pilot", "positives", "hard_generation"):
                root = Path(state["project"]).parent
                folder = root / "data"
                if job["stage"] == "references":
                    paths = [folder / "references" / split / "references.jsonl" for split in ("train", "validation")]
                    total = state["settings"]["train_references"] + state["settings"]["validation_references"]
                elif job["stage"] == "pilot":
                    paths = [folder / "pilot/train/candidates.jsonl"]
                    total = state["settings"]["train_references"] if state["settings"].get("pilot_review_policy") == "one_per_training_voice_v1" else 24
                elif job["stage"] == "hard_generation":
                    paths, total = [folder / "generated/hard/candidates.jsonl"], state["settings"]["hard_originals"]
                else:
                    paths = [folder / "generated" / split / "candidates.jsonl" for split in ("train", "validation")]
                    total = state["settings"]["originals"] + state["settings"]["validation_originals"]
                counters = [self._journal_counts(p) for p in paths]
                done = sum(c["passed"] for c in counters)
                if job["stage"] == "pilot" and state["settings"].get("pilot_review_policy") == "one_per_training_voice_v1":
                    done = min(total, len(set().union(*(c.get("passed_families", set()) for c in counters))))
                view["progress"] = {"done": done, "total": total, "unit": "개", "label": "자동 기술 검사 통과 음성"}
                view["counts"] = {"tts_generated": sum(c["attempts"] for c in counters), "technical_passed": done}
            if job["stage"] == "features":
                import re
                matches = re.findall(r"(positive|korean_negative|hard_negative|generic_negative) R\d+: (\d+)/(\d+)", self.log(state["id"]))
                if matches:
                    role, done, total = matches[-1]
                    view["progress"] = {"done": int(done), "total": int(total), "unit": "개 특징", "label": role}
            if job["stage"] == "evaluation":
                view["progress"] = _evaluation_log_progress(job)
                view["progress_note"] = "음원 수 기준이며, 긴 비호출 음원은 더 오래 걸립니다."
                view["eta_seconds"] = None
                phase = view["progress"]["phase"]
                if phase == "validation":
                    view["next_action"] = "평가 음원의 무결성과 형식을 확인하고 있습니다."
                elif phase == "finalizing":
                    view["next_action"] = "채점을 마쳤으며 결과를 정리하고 있습니다."
            current = view["progress"]
            if (job["stage"] != "evaluation" and current.get("phase") not in ("retry", "download_many")
                    and view["eta_seconds"] is None and current.get("total") and current.get("done") is not None
                    and current.get("unit") != "완료 단계"):
                key = (job["task_id"], current.get("phase"), current.get("label"), current.get("total"))
                baseline = self._rates.setdefault(key, (time.monotonic(), current["done"]))
                seconds, delta = time.monotonic() - baseline[0], current["done"] - baseline[1]
                if seconds >= 10 and delta > 0:
                    view["eta_seconds"] = max(0, (current["total"] - current["done"]) * seconds / delta)
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
            view["review"] = {"queue_id": r["id"], "title": r.get("title", "음원 확인"),
                              "description": r.get("description", "정상 음원은 보존하고 문제가 있는 항목만 표시하세요."),
                              "remaining_queues": len(reviews),
                              "url": f"/review/?project={state['id']}&queue={quote(r['id'])}"}
        if state["status"] == "waiting_input":
            result = state["results"].get(stage["id"], {})
            view["input"] = result.get("input", {"kind": result.get("input_kind", "diagnostic"),
                                                   "description": result.get("message", "자료 준비가 필요합니다."),
                                                   "suggestions": result.get("suggestions", [])})
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
        return view

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

    def snapshot(self):
        with self._lock:
            projects = [self.describe(self._load(p)) for p in self.registry["projects"]]
            system = {"free_bytes": shutil.disk_usage(self.workspace).free, "python": sys.version.split()[0]}
            for key in self.registry["projects"]:
                preflight = self._load(key)["results"].get("frontend", {}).get("result", {}).get("check", {})
                if preflight:
                    system.update(preflight)
                    break
            return {"workspace": str(self.workspace), "token": self.token, "projects": projects,
                    "active_project_id": self.registry.get("active_project_id"),
                    "system": system}

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
                bundle = self.analysis.active_bundle(project_id) or original_bundle
                original = read_json(Path(original_bundle) / "manifest.json")
                labels = {Path(row["path"]).stem: f"모델 {index + 1}"
                          for index, row in enumerate(original["members"])}
                self._microphones[project_id] = MicrophoneService(WakeWordDetector(bundle), model_labels=labels)
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
                    return self.respond(manager.snapshot())
                if path == "/api/recipes":
                    return self.respond(manager.recipes())
                if path in ("/", "/studio.js", "/studio.css"):
                    name = "studio.html" if path == "/" else path[1:]
                    kind = "text/html; charset=utf-8" if name.endswith("html") else "text/javascript; charset=utf-8" if name.endswith("js") else "text/css; charset=utf-8"
                    return self.respond((resources / name).read_bytes(), kind=kind)
                if path.startswith("/api/projects/") and path.endswith("/log"):
                    return self.respond({"log": manager.log(path.split("/")[3])})
                if path.startswith("/reference-preview/"):
                    project_id = query["project"][0]
                    state, rows = manager.reference_preview_rows(project_id)
                    base = f"?project={quote(project_id)}"
                    if path == "/reference-preview/":
                        return self.respond((resources / "reference_preview.html").read_bytes(), kind="text/html; charset=utf-8")
                    if path == "/reference-preview/queue":
                        return self.respond({"items": [{"id": row["sample_id"], "text": row.get("text", ""),
                            "group": "학습용" if row["split"] == "train" else "공통 평가용",
                            "audio": f"/reference-preview/audio/{i}" + base} for i, row in enumerate(rows)]})
                    if path.startswith("/reference-preview/audio/"):
                        index = int(path.rsplit("/", 1)[1])
                        if not 0 <= index < len(rows):
                            raise ValueError("잘못된 목소리 번호입니다.")
                        audio = resolved(Path(state["project"]).parent, rows[index]["path"])
                        if digest(audio) != rows[index]["sha256"]:
                            raise ValueError("기준 목소리 음원이 변경되었습니다.")
                        return self.send_file(audio)
                if path.startswith("/review/"):
                    project_id, queue_id = query["project"][0], query["queue"][0]
                    state, review, rows = manager.review_queue(project_id, queue_id)
                    base = f"?project={project_id}&queue={quote(queue_id)}"
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
                                              "audio": f"/review/audio/{i}" + base} for i, r in enumerate(rows)]}
                        if review.get("pool_manifest"):
                            if review.get("evaluation_only") is True:
                                payload.update(policy_review=True, evaluation_only=True,
                                    intended_use="held_out_development_noise", pools=[],
                                    policy_text="이 자료를 학습에는 사용하지 않고, 학습과 분리한 소음 평가에만 사용하는 데 동의합니다.")
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
                    return self.send_file(target)
                if path.startswith("/mic/"):
                    return self.mic_get(path)
                if path.startswith("/analysis/"):
                    return self.analysis_get(path)
                self.respond({"error": "페이지를 찾을 수 없습니다."}, 404)
            except _ProgressReadPending as error:
                self.respond({"error": str(error), "code": "progress_read_pending"}, 503)
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
            names = {"": "analysis.html", "analysis.js": "analysis.js", "analysis.css": "analysis.css"}
            if rest not in names:
                return self.respond({"error": "페이지를 찾을 수 없습니다."}, 404)
            name = names[rest]
            kind = "text/html; charset=utf-8" if name.endswith("html") else "text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8"
            return self.respond((resources / name).read_bytes(), kind=kind)

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
                            return self.respond(service.start())
                    if action == "stop":
                        return self.respond(service.stop(payload.get("session_id")))
                    raise ValueError("Unsupported microphone action")
                self.guard(mutate=True)
                payload = self.body()
                if path.startswith("/analysis/"):
                    parts = path.split("/")
                    if len(parts) != 5 or parts[3] != "api" or parts[4] not in ("compare", "select"):
                        raise ValueError("지원하지 않는 분석 작업입니다.")
                    return self.respond(manager.analysis.launch(parts[2], parts[4], payload))
                if path == "/api/projects":
                    return self.respond(manager.create(payload))
                if path == "/api/model-location":
                    return self.respond(manager.model_location(payload))
                if path == "/review/save":
                    query = parse_qs(urlsplit(self.path).query)
                    return self.respond(manager.save_review(query["project"][0], query["queue"][0], payload))
                if path.startswith("/api/projects/"):
                    project_id, action = path.split("/")[3:5]
                    methods = {"start": manager.start, "resume": manager.start, "pause": manager.pause,
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
