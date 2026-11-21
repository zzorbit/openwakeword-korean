"""Worker identity and cooperative pause requests for the local Studio."""

from pathlib import Path
import os
import re
import subprocess

from .common import now, write_json


STAGE_PAUSE_MESSAGES = {
    "pilot": "현재 발음 후보를 저장한 뒤 일시정지합니다.",
    "positives": "현재 생성 중인 음성을 저장한 뒤 일시정지합니다.",
    "hard_pilot": "현재 생성 중인 음성을 저장한 뒤 일시정지합니다.",
    "hard_generation": "현재 생성 중인 음성을 저장한 뒤 일시정지합니다.",
    "features": "현재 특징 추출 묶음을 저장한 뒤 일시정지합니다.",
    "training": "현재 학습 묶음을 저장한 뒤 일시정지합니다.",
    "development": "현재 원본의 평가 음원을 저장한 뒤 일시정지합니다.",
    "evaluation": "현재 채점 또는 감지 기준 비교를 저장한 뒤 일시정지합니다.",
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


def worker_lock_paths(root):
    """Known writer directories only; don't walk hundreds of thousands of WAVs."""
    roots = (root / "features", root / "runs", root / "data", root / ".studio")
    patterns = (".writer.lock", "*/.writer.lock", "*/*/.writer.lock", "*/*/*/.writer.lock")
    revision_locks = (root / "data/generated/hard_phrase_revisions").glob("*/*/.writer.lock")
    paths = [path for base in roots if base.exists() for pattern in patterns for path in base.glob(pattern)]
    paths.extend(revision_locks)
    for path in (root / "reports").glob("studio_*/.writer.lock"):
        if re.fullmatch(r"studio_(?:(?:evaluation|paired)(?:_attempt[0-9]+)?|runtime_recommendations_[0-9a-f]{16}_[0-9a-f]{12})",
                        path.parent.name):
            paths.append(path)
    return paths


def request_stage_pause(state, job, *, candidate_preparation):
    """Send a cooperative pause request to stages that support it."""
    if job["stage"] in STAGE_PAUSE_MESSAGES:
        path = Path(job["directory"]) / "pause.json"
        write_json(path, {"task_id": job["task_id"], "requested_at": now()})
        return STAGE_PAUSE_MESSAGES[job["stage"]]
    if candidate_preparation:
        path = Path(state["project"]).parent / ".studio/reference-candidates/preparation/pause.json"
        write_json(path, {"task_id": job["task_id"], "requested_at": now()})
    return "현재 단계를 완료한 뒤 일시정지합니다. 자료를 강제로 끊지 않습니다."
