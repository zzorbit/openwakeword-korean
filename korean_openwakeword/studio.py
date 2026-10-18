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
import hmac
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .common import digest, identity, now, read_json, read_rows, resolved, write_json
from .project import create_project, load_project


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
        self._microphones = {}
        self._journals = {}
        self._rates = {}
        self._closing = threading.Event()
        self._thread = None
        self.registry = read_json(self.registry_path) if self.registry_path.exists() else {"projects": {}, "active_project_id": None}
        self._recover()

    def _path(self, project_id):
        if project_id not in self.registry["projects"]:
            raise ValueError("등록되지 않은 프로젝트입니다.")
        project = _safe_root(self.registry["projects"][project_id])
        return project.parent / ".studio" / "workflow.json"

    def _load(self, project_id):
        state = read_json(self._path(project_id))
        if state.get("id") != project_id or state.get("workspace") != str(self.workspace):
            raise ValueError("저장된 프로젝트 실행 근거가 일치하지 않습니다.")
        return state

    def _save(self, state):
        state["updated_at"] = now()
        write_json(self._path(state["id"]), state)

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
        presets = []
        for key, label in (("check", "동작 확인용"), ("trial", "시험 모델용"), ("production", "상용 모델용")):
            settings = preset_settings(key)
            recipe = material_recipe(key, evaluation_policy=settings["evaluation_policy"])
            presets.append({**settings, **recipe, "id": key, "label": label})
        return {"presets": presets, "default_directory": str(self.workspace / "projects"),
                "storage_note": "선택한 작업 폴더에 자료·모델·캐시를 저장합니다. 공개 자료와 Qwen 모델의 첫 다운로드는 별도 용량이 필요합니다."}

    def create(self, payload):
        from .studio_steps import preset_settings, workflow_plan
        with self._lock:
            if not isinstance(payload, dict) or not isinstance(payload.get("wakeword"), str):
                raise ValueError("호출어를 문자열로 입력하세요.")
            preset = payload.get("preset", "check")
            if not isinstance(preset, str):
                raise ValueError("작업 규모를 선택하세요.")
            settings = preset_settings(preset)
            overrides = payload.get("advanced") or {}
            if not isinstance(overrides, dict):
                raise ValueError("세부 설정 형식이 올바르지 않습니다.")
            if payload.get("directory") is not None and not isinstance(payload["directory"], (str, Path)):
                raise ValueError("저장 폴더 경로를 문자열로 입력하세요.")
            mapping = {"positive_originals": "originals", "augmentation_rounds": "rounds", "updates": "updates",
                       "generic_hours": "generic_hours", "korean_hours": "korean_hours"}
            if set(overrides) - set(mapping):
                raise ValueError("지원하지 않는 세부 설정입니다.")
            for key, value in overrides.items():
                if value is not None and value != "":
                    settings[mapping[key]] = value
            project_id = uuid.uuid4().hex[:12]
            word = payload.get("wakeword", "").strip()
            target = _safe_root(payload.get("directory") or self.workspace / "projects" / ("word_" + project_id))
            project = target / "project.json"
            if project.exists():
                _, config = load_project(project)
                expected = (word, settings["originals"], settings["rounds"], settings["updates"])
                actual = (config["wakeword"], config["targets"]["positive_originals"], config["augmentation_rounds"], config["training"]["updates"])
                if expected != actual or (target / "final_lock.json").exists():
                    raise ValueError("기존 프로젝트와 선택한 호출어·작업량이 다릅니다. 기존 자료는 보존하고 새 폴더를 선택하세요.")
                for known_id, known_path in self.registry["projects"].items():
                    if Path(known_path) == project:
                        self.registry["active_project_id"] = known_id
                        write_json(self.registry_path, self.registry)
                        return self.describe(self._load(known_id))
                if (target / ".studio" / "workflow.json").exists():
                    raise ValueError("이 프로젝트의 웹 실행 기록이 이미 있습니다. 처음 사용한 작업 폴더로 웹 화면을 여세요. 기존 기록은 덮어쓰지 않습니다.")
                if _lock_paths(target):
                    raise ValueError("기존 작업의 잠금이 있습니다. 실행 상태를 확인한 뒤 연결하세요.")
                settings.update(generic_hours=config["targets"]["generic_hours"], korean_hours=config["targets"]["korean_hours"])
                bank = read_json(target / "voices.json")
                if sum(f["split"] == "validation" for f in bank["families"]) < settings["validation_references"]:
                    # Reusing an existing project does not rewrite its voice bank
                    # or pretend that the expanded protocol has been prepared.
                    from .studio_steps import common_evaluation_policy
                    policy = common_evaluation_policy(1)
                    settings.update(evaluation_policy=policy,
                                    **{k: policy[k] for k in ("validation_references", "validation_originals", "noise_sections")})
            else:
                create_project(target, word, "word_" + project_id,
                               positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                               updates=settings["updates"], generic_hours=settings["generic_hours"],
                               korean_hours=settings["korean_hours"],
                               validation_voice_families=settings["validation_references"])
            state = {"schema_version": 1, "id": project_id, "workspace": str(self.workspace),
                     "storage_workspace": str(self.workspace if target.is_relative_to(self.workspace) else target),
                     "project": str(project), "wakeword": word, "preset": preset, "settings": settings,
                     "config_sha256": digest(project), "implementation_id": self._implementation_id,
                     "created_at": now(), "status": "ready", "index": 0,
                     "plan": workflow_plan(settings), "results": {}, "inputs": {}, "job": None,
                     "error": None, "next_action": "시작하면 자료 준비부터 자동으로 진행합니다."}
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
            raise ValueError("프로젝트 설정이 변경되었습니다. 실행 중인 실험 설정은 덮어쓰지 않습니다.")
        if state.get("implementation_id") != self._implementation_id:
            root = Path(state["project"]).parent
            jobs = root / ".studio" / "jobs"
            unstarted = (state.get("status") == "ready" and type(state.get("index")) is int
                         and state["index"] == 0 and state.get("results", {}) == {}
                         and state.get("inputs", {}) == {} and state.get("history", []) == []
                         and state.get("job") in (None, {}) and state.get("last_job") in (None, {})
                         and not (jobs.exists() and any(jobs.iterdir())) and not _lock_paths(root))
            if not unstarted:
                raise ValueError("라이브러리 코드가 변경되었습니다. 기존 실행 자료를 보존하고 새 실험을 만드세요.")
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
            state.setdefault("migrations", []).append({"path": str(receipt), "sha256": digest(receipt)})
            self._save(state)

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
            raise ValueError("단계 이후 프로젝트 설정이 변경되었습니다.")
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
            else:
                state.update(status="error", error="지원하지 않는 단계 종료 상태: " + str(phase))
        state.setdefault("history", []).append({"stage": job["stage"], "started_at": job["started_at"],
                                                "completed_at": now(), "result": str(path), "sha256": digest(path)})
        self._save(state)
        return True

    def start(self, project_id):
        with self._lock:
            state = self._load(project_id)
            self._guard_config(state)
            if state["status"] in ("running", "pausing") and self._job_alive(state.get("job") or {}):
                state["auto_continue"] = True
                self._save(state)
                return self.describe(state)
            for other_id in self.registry["projects"]:
                other = self._load(other_id)
                if other_id != project_id and (other.get("auto_continue") or self._job_alive(other.get("job") or {})):
                    raise ValueError("다른 프로젝트가 실행 중입니다. 해당 작업을 먼저 일시정지하세요.")
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
        for pattern in ("*/.writer.lock", "downloads/*/.writer.lock", "mit_rir/16khz/*/.writer.lock"):
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

    def describe(self, state):
        index, plan = state["index"], state["plan"]
        stage = plan[index] if index < len(plan) else {"id": "completed", "label": "완료"}
        view = {key: state.get(key) for key in ("id", "wakeword", "preset", "status", "error", "next_action", "created_at")}
        view.update(directory=str(Path(state["project"]).parent), stage=stage, stage_id=stage["id"],
                    evaluation_policy=state.get("settings", {}).get("evaluation_policy"),
                    evaluation_policy_sha256=identity(state["settings"]["evaluation_policy"]) if state.get("settings", {}).get("evaluation_policy") else None,
                    stages=[{**entry, "status": "completed" if i < index else state["status"] if i == index else "pending"}
                            for i, entry in enumerate(plan)],
                    progress={"done": index, "total": len(plan), "unit": "완료 단계"},
                    elapsed_seconds=round(_elapsed(state.get("started_at")), 1), eta_seconds=None)
        job = state.get("job") or {}
        if job:
            view["current_stage_elapsed_seconds"] = round(_elapsed(job.get("started_at")), 1)
            view["process_alive"] = self._job_alive(job)
            progress = Path(state["project"]).parent / ".studio" / "task_progress.json"
            if progress.exists():
                record = read_json(progress)
                if record.get("task_id") == job["task_id"]:
                    view["progress"] = record
                    view["eta_seconds"] = record.get("eta_seconds")
            material_progress = Path(state["project"]).parent / ".studio" / "progress.json"
            if job["stage"] == "data_download" and material_progress.exists():
                record = read_json(material_progress)
                if record.get("stage") == "data_download" and record.get("updated_at", "") >= job["started_at"]:
                    view["progress"] = record
            if job["stage"] == "training":
                progress_files = list((Path(state["project"]).parent / "runs" / "studio_run").glob("member*/progress.json"))
                records = [read_json(p) for p in progress_files]
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
                    paths, total = [folder / "pilot/train/candidates.jsonl"], 24
                elif job["stage"] == "hard_generation":
                    paths, total = [folder / "generated/hard/candidates.jsonl"], state["settings"]["hard_originals"]
                else:
                    paths = [folder / "generated" / split / "candidates.jsonl" for split in ("train", "validation")]
                    total = state["settings"]["originals"] + state["settings"]["validation_originals"]
                counters = [self._journal_counts(p) for p in paths]
                done = sum(c["passed"] for c in counters)
                view["progress"] = {"done": done, "total": total, "unit": "개", "label": "자동 기술 검사 통과 음성"}
                view["counts"] = {"tts_generated": sum(c["attempts"] for c in counters), "technical_passed": done}
            if job["stage"] == "features":
                import re
                matches = re.findall(r"(positive|korean_negative|hard_negative|generic_negative) R\d+: (\d+)/(\d+)", self.log(state["id"]))
                if matches:
                    role, done, total = matches[-1]
                    view["progress"] = {"done": int(done), "total": int(total), "unit": "개 특징", "label": role}
            current = view["progress"]
            if current.get("total") and current.get("done") is not None and current.get("unit") != "완료 단계":
                key = (job["task_id"], current.get("label"), current.get("total"))
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
        candidate_coverage = downloaded.get("evaluation_material_coverage")
        admitted_coverage = connected.get("evaluation_material_coverage")
        if candidate_coverage or admitted_coverage:
            view["evaluation_materials"] = {"candidate": candidate_coverage, "admitted": admitted_coverage,
                "scope": "원본 고유 시간 · 다운로드/기술 검사와 대표 검수 승인은 별도 · 실제 채점 시간은 결과에서 확인"}
            if (state.get("settings", {}).get("evaluation_policy") or {}).get("id") == "common_development_v4":
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
        record = self._journals.setdefault(str(path), {"offset": 0, "attempts": 0, "passed": 0})
        if path.stat().st_size < record["offset"]:
            record.update(offset=0, attempts=0, passed=0)
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
        path = Path(job["log"]) if job.get("log") else None
        if path and path.exists():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 24000))
                return handle.read().decode("utf-8", errors="replace")
        return "아직 실행 로그가 없습니다."

    def microphone(self, project_id):
        if project_id not in self._microphones:
            with self._lock:
                state = self._load(project_id)
                if "package" not in state["results"]:
                    raise ValueError("모델 학습과 실행 묶음 준비 후 시험할 수 있습니다.")
                from .microphone import MicrophoneService
                from .runtime import WakeWordDetector
                package = state["results"]["package"]
                bundle = package.get("result", package)["bundle"]
                self._microphones[project_id] = MicrophoneService(WakeWordDetector(bundle))
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
                if path.startswith("/review/"):
                    project_id, queue_id = query["project"][0], query["queue"][0]
                    state, review, rows = manager.review_queue(project_id, queue_id)
                    base = f"?project={project_id}&queue={quote(queue_id)}"
                    if path == "/review/":
                        html = (resources / "review.html").read_text(encoding="utf-8")
                        html = html.replace("fetch('/queue')", "fetch(" + json.dumps("/review/queue" + base) + ")")
                        html = html.replace("fetch('/save',", "fetch(" + json.dumps("/review/save" + base) + ",")
                        html = html.replace("headers:{'Content-Type':'application/json'}", "headers:{'Content-Type':'application/json','X-Koww-Token':" + json.dumps(manager.token) + "}")
                        html = html.replace("저장 완료 · 채팅에서 완료했다고 알려주세요.", "저장 완료 · 다음 단계를 준비합니다.")
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
                self.respond({"error": "페이지를 찾을 수 없습니다."}, 404)
            except PermissionError as error:
                self.respond({"error": str(error)}, 403)
            except (ValueError, KeyError, OSError, IndexError) as error:
                self.respond({"error": str(error)}, 400)

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
            if rest == "api/info":
                return self.respond(manager.microphone(project_id).info())
            names = {"": "microphone.html", "microphone.js": "microphone.js", "microphone.css": "microphone.css", "microphone_worklet.js": "microphone_worklet.js"}
            if rest not in names:
                return self.respond({"error": "Not found"}, 404)
            name = names[rest]
            content = (resources / name).read_text(encoding="utf-8")
            if name == "microphone.html":
                content = content.replace('href="/microphone.css"', 'href="' + prefix + '/microphone.css"').replace('src="/microphone.js"', 'src="' + prefix + '/microphone.js"')
            elif name == "microphone.js":
                content = content.replace('fetch(path,', 'fetch(' + json.dumps(prefix) + ' + path,').replace('"/microphone_worklet.js"', json.dumps(prefix + "/microphone_worklet.js"))
            kind = "text/html; charset=utf-8" if name.endswith("html") else "text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8"
            return self.respond(content.encode(), kind=kind)

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
                        return self.respond(service.start())
                    if action == "stop":
                        return self.respond(service.stop(payload.get("session_id")))
                    raise ValueError("Unsupported microphone action")
                self.guard(mutate=True)
                payload = self.body()
                if path == "/api/projects":
                    return self.respond(manager.create(payload))
                if path == "/review/save":
                    query = parse_qs(urlsplit(self.path).query)
                    return self.respond(manager.save_review(query["project"][0], query["queue"][0], payload))
                if path.startswith("/api/projects/"):
                    project_id, action = path.split("/")[3:5]
                    methods = {"start": manager.start, "resume": manager.start, "pause": manager.pause}
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
