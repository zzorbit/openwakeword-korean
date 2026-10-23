"""Explicit, versioned post-training choices; original experiments stay frozen."""
from pathlib import Path
import argparse
import math
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile

from .common import digest, now, read_json, write_json, writer_lock


def _result(state, key):
    value = state.get("results", {}).get(key, {})
    return value.get("result", value)


def _inside(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("분석 파일은 해당 프로젝트 안에 있어야 합니다.")
    return path


def _parameters(payload, report):
    if not isinstance(payload, dict):
        raise ValueError("모델 선택 내용을 확인하세요.")
    candidates = report.get("candidate_definitions", [])
    candidate = payload.get("candidate")
    if candidate not in {row["id"] for row in candidates}:
        raise ValueError("평가한 모델 중 하나를 선택하세요.")
    thresholds = payload.get("thresholds", [report["runtime"]["threshold"]])
    consecutive = payload.get("consecutive", [report["runtime"]["consecutive_frames"]])
    if (not isinstance(thresholds, list) or not 1 <= len(thresholds) <= 7
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 0 < value < 1 for value in thresholds)):
        raise ValueError("감지 기준은 0과 1 사이의 숫자를 최대 7개 입력하세요.")
    if (not isinstance(consecutive, list) or not 1 <= len(consecutive) <= 3
            or any(type(value) is not int or not 1 <= value <= 10 for value in consecutive)):
        raise ValueError("연속 감지 수는 1~10 사이의 정수를 최대 3개 입력하세요.")
    return {"candidate": candidate, "thresholds": list(dict.fromkeys(thresholds)),
            "consecutive": list(dict.fromkeys(consecutive))}


class StudioAnalysis:
    def __init__(self, manager, *, popen=None):
        self.manager = manager
        self._popen = popen or subprocess.Popen
        self._lock = threading.RLock()
        self._children = {}

    def source(self, project_id):
        state = self.manager._load(project_id)
        root = Path(state["project"]).resolve().parent
        if state.get("status") != "completed":
            raise ValueError("학습과 공통 평가를 마친 뒤 모델을 비교할 수 있습니다.")
        bundle = _inside(root, _result(state, "package")["bundle"])
        evaluation = _inside(root, _result(state, "evaluation")["evaluation"])
        report = read_json(evaluation / "report.json")
        if report.get("stage") != "development":
            raise ValueError("공통 평가 결과만 모델 선택에 사용할 수 있습니다.")
        if digest(state["project"]) != state["config_sha256"]:
            raise ValueError("원래 프로젝트 설정이 변경되었습니다.")
        return state, root, bundle, evaluation, report

    def _directory(self, project_id):
        state = self.manager._load(project_id)
        return Path(state["project"]).resolve().parent / ".studio" / "analysis"

    def _alive(self, job):
        from .studio import process_identity
        request = Path(job["directory"]) / "request.json"
        owner_path = request.with_name("owner.json")
        records = [job]
        if owner_path.is_file():
            owner = read_json(owner_path)
            if owner.get("task_id") != job["task_id"] or owner.get("request_sha256") != digest(request):
                raise ValueError("분석 프로세스의 실행 근거가 다릅니다.")
            records.append(owner)
        return any(record.get("creation_identity") and process_identity(record.get("pid")) == record["creation_identity"]
                   for record in records)

    def _refresh(self, project_id):
        directory = self._directory(project_id)
        path = directory / "state.json"
        if not path.is_file():
            return {"status": "idle"}
        state = read_json(path)
        job = state.get("job") or {}
        if state.get("status") != "running":
            return state
        result_path = Path(job["directory"]) / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            request_path = result_path.with_name("request.json")
            if result.get("task_id") != job["task_id"] or result.get("request_sha256") != digest(request_path):
                raise ValueError("분석 완료 기록이 실행 요청과 다릅니다.")
            state.update(status="error" if result.get("error") else "completed",
                         error=result.get("error"), result=result.get("result"),
                         completed_at=result["completed_at"])
            if not result.get("error") and state["action"] == "select":
                root = Path(self.manager._load(project_id)["project"]).resolve().parent
                bundle = _inside(root, result["result"]["bundle"])
                if not bundle.is_relative_to(Path(job["directory"]).resolve()):
                    raise ValueError("선택 결과가 별도 분석 폴더를 벗어났습니다.")
                write_json(directory / "active.json", {"job_id": job["task_id"], "bundle": str(bundle),
                    "bundle_sha256": digest(bundle / "manifest.json"), "candidate": result["result"]["candidate"],
                    "runtime": result["result"]["runtime"], "summary": result["result"].get("summary"),
                    "selected_at": now()})
                self.manager._microphones.pop(project_id, None)
            write_json(path, state)
        elif not self._alive(job):
            state.update(status="interrupted", error="분석이 중단됐습니다. 기존 결과는 보존됩니다. 다시 비교하거나 선택하세요.")
            write_json(path, state)
        return state

    def busy(self, project_id):
        with self._lock:
            return self._refresh(project_id).get("status") == "running"

    def active_bundle(self, project_id):
        with self._lock:
            self._refresh(project_id)
            directory = self._directory(project_id)
            path = directory / "active.json"
            if not path.is_file():
                return None
            value = read_json(path)
            bundle = _inside(directory, value["bundle"])
            if digest(bundle / "manifest.json") != value["bundle_sha256"]:
                raise ValueError("선택한 모델의 실행 묶음이 변경되었습니다.")
            return str(bundle)

    def status(self, project_id):
        with self._lock:
            workflow, root, bundle, evaluation, report = self.source(project_id)
            state = self._refresh(project_id)
            active = self._directory(project_id) / "active.json"
            selection = read_json(active) if active.is_file() else None
            candidates = report.get("candidate_definitions", [])
            models = []
            for row in report.get("models", []):
                models.append({key: row.get(key) for key in ("id", "kind", "rank", "clean", "noise_cell_mean", "negative", "primary")})
            view = {"token": self.manager.token, "wakeword": workflow["wakeword"], "status": state["status"],
                    "models": models, "candidate_definitions": candidates,
                    "runtime": report["runtime"], "default_candidate": report.get("deployment_candidate", {}).get("id"),
                    "adoptions": report.get("candidate_adoptions", {}), "selection": selection,
                    "selection_allowed": not (root / "final_lock.json").exists(),
                    "job": {key: state.get("job", {}).get(key) for key in ("task_id", "started_at")},
                    "action": state.get("action"), "error": state.get("error")}
            if selection:
                active_bundle = _inside(self._directory(project_id), selection["bundle"])
                if digest(active_bundle / "manifest.json") != selection["bundle_sha256"]:
                    raise ValueError("선택한 모델의 실행 묶음이 변경되었습니다.")
                view["selected"] = {key: selection.get(key) for key in ("candidate", "runtime", "bundle", "summary")}
                view["download_url"] = f"/analysis/{project_id}/download/{selection['job_id']}"
            completed = state.get("result") or {}
            if state["status"] == "running" and state.get("action") == "compare":
                from .studio import _job_log_tail
                matches = list(re.finditer(r"(?m)^Cache comparison: (\d+)/(\d+) candidate=", _job_log_tail(state["job"])))
                if matches:
                    done, total = map(int, matches[-1].groups())
                    if 0 <= done <= total and total > 0:
                        view["progress"] = {"done": done, "total": total}
            if state["status"] == "completed":
                if state.get("action") == "compare":
                    comparison = _inside(root, completed["comparison"])
                    view["comparison"] = read_json(comparison)
                    view["comparison_id"] = state["job"]["task_id"]
                else:
                    view["selected"] = {key: completed.get(key) for key in ("candidate", "runtime", "bundle", "adoption", "summary")}
                    view["download_url"] = f"/analysis/{project_id}/download/{state['job']['task_id']}"
            return view

    def launch(self, project_id, action, payload):
        from .studio import process_identity, _implementation
        if action not in ("compare", "select"):
            raise ValueError("지원하지 않는 분석 작업입니다.")
        with self.manager._lock, self._lock:
            state, root, bundle, evaluation, report = self.source(project_id)
            if (root / "final_lock.json").exists():
                raise ValueError("최종 평가를 위해 잠긴 프로젝트는 새 실험에서 비교·선택하세요.")
            if self._refresh(project_id)["status"] == "running":
                raise ValueError("이미 비교·선택 작업이 진행 중입니다.")
            for other_id in self.manager.registry["projects"]:
                other = self.manager._load(other_id)
                if self.manager._job_alive(other.get("job") or {}):
                    raise ValueError("진행 중인 학습 작업을 마친 뒤 비교하세요.")
            microphone = self.manager._microphones.get(project_id)
            if microphone is not None and microphone.active():
                raise ValueError("마이크 시험을 중지한 뒤 모델을 비교하거나 선택하세요.")
            parameters = _parameters(payload, report)
            if action == "select" and (len(parameters["thresholds"]) != 1 or len(parameters["consecutive"]) != 1):
                raise ValueError("시험할 모델에는 감지 기준과 연속 감지 수를 하나씩 선택하세요.")
            comparison = None
            comparison_id = payload.get("comparison_id")
            if comparison_id:
                if not isinstance(comparison_id, str) or not re_full_id(comparison_id):
                    raise ValueError("비교 기록 번호가 잘못됐습니다.")
                prior = self._directory(project_id) / "jobs" / comparison_id
                result = read_json(prior / "result.json")
                request = read_json(prior / "request.json")
                if result.get("error") or request.get("action") != "compare" or result.get("request_sha256") != digest(prior / "request.json"):
                    raise ValueError("완료한 민감도 비교 기록이 필요합니다.")
                comparison = str(_inside(prior, result["result"]["comparison"]))
            task_id = uuid.uuid4().hex
            directory = self._directory(project_id) / "jobs" / task_id
            directory.mkdir(parents=True, exist_ok=False)
            request = {"task_id": task_id, "project": state["project"], "bundle": str(bundle),
                       "evaluation": str(evaluation), "action": action, "parameters": parameters,
                       "comparison": comparison, "config_sha256": state["config_sha256"],
                       "implementation_id": _implementation()}
            request_path = directory / "request.json"
            write_json(request_path, request)
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment.get("PYTHONPATH", "")
            with (directory / "output.log").open("ab") as log:
                child = self._popen([sys.executable, "-m", "korean_openwakeword.studio_analysis", "--request", str(request_path)],
                    stdout=log, stderr=subprocess.STDOUT, cwd=root, env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._children[task_id] = child
            job = {"task_id": task_id, "directory": str(directory), "pid": child.pid,
                   "creation_identity": process_identity(child.pid), "started_at": now(),
                   "log": str(directory / "output.log")}
            write_json(self._directory(project_id) / "state.json", {"status": "running", "action": action, "job": job})
            return {"status": "running", "task_id": task_id}

    def download(self, project_id, task_id):
        if not re_full_id(task_id):
            raise ValueError("모델 기록 번호가 잘못됐습니다.")
        directory = self._directory(project_id) / "jobs" / task_id
        result = read_json(directory / "result.json")
        request = read_json(directory / "request.json")
        if request.get("action") != "select" or result.get("error") or result.get("request_sha256") != digest(directory / "request.json"):
            raise ValueError("완료된 모델 선택 기록이 필요합니다.")
        archive = _inside(directory, result["result"]["archive"])
        if digest(archive) != result["result"]["archive_sha256"]:
            raise ValueError("모델 다운로드 파일이 변경되었습니다.")
        return archive


def re_full_id(value):
    return isinstance(value, str) and len(value) == 32 and all(char in "0123456789abcdef" for char in value)


def execute(request_path):
    from .studio import process_identity, _implementation
    from .storage import storage_environment
    from .tune import compare_cached, select_cached_configuration
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    if request_path.name != "request.json" or request_path.parent.name != request["task_id"] or not re_full_id(request["task_id"]):
        raise ValueError("분석 실행 요청 경로가 잘못됐습니다.")
    root = Path(request["project"]).resolve().parent
    if not request_path.is_relative_to(root / ".studio" / "analysis" / "jobs"):
        raise ValueError("분석 작업은 해당 프로젝트 안에서 실행해야 합니다.")
    owner_path = request_path.with_name("owner.json")
    if owner_path.is_file():
        owner = read_json(owner_path)
        if owner.get("creation_identity") and process_identity(owner.get("pid")) == owner["creation_identity"]:
            raise ValueError("이 분석 작업이 이미 실행 중입니다.")
    write_json(owner_path, {"task_id": request["task_id"], "request_sha256": digest(request_path),
        "pid": os.getpid(), "creation_identity": process_identity(os.getpid()), "parent_pid": os.getppid(), "started_at": now()})
    result_path = request_path.with_name("result.json")
    with writer_lock(request_path.parent):
        if result_path.exists():
            raise ValueError("완료된 분석 결과는 다시 쓰지 않습니다.")
        result = {"task_id": request["task_id"], "request_sha256": digest(request_path), "started_at": now()}
        try:
            if digest(request["project"]) != request["config_sha256"] or _implementation() != request["implementation_id"]:
                raise ValueError("분석 전에 설정 또는 라이브러리가 변경되었습니다.")
            parameters = _parameters(request["parameters"], read_json(Path(request["evaluation"]) / "report.json"))
            output = request_path.parent / "output"
            with storage_environment(root):
                if request["action"] == "compare":
                    result["result"] = compare_cached(request["project"], request["bundle"], request["evaluation"], output,
                        parameters["thresholds"], parameters["consecutive"], parameters["candidate"])
                    # The durable comparison lives separately; the worker receipt stays small.
                    result["result"].pop("comparisons", None)
                elif request["action"] == "select":
                    if len(parameters["thresholds"]) != 1 or len(parameters["consecutive"]) != 1:
                        raise ValueError("모델 선택에는 한 가지 감지 설정만 사용할 수 있습니다.")
                    value = select_cached_configuration(request["project"], request["bundle"], request["evaluation"], output,
                        parameters["candidate"], parameters["thresholds"][0], parameters["consecutive"][0], request.get("comparison"))
                    selected_report = read_json(Path(value["evaluation"]) / "report.json")
                    selected_summary = next(row for row in selected_report["models"] if row["id"] == parameters["candidate"])
                    value["summary"] = {key: selected_summary.get(key)
                                        for key in ("clean", "noise_cell_mean", "negative", "primary")}
                    archive = request_path.parent / "bundle.zip"
                    selected = Path(value["bundle"]).resolve()
                    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
                        for path in sorted(selected.rglob("*")):
                            if path.is_file():
                                if not path.resolve().is_relative_to(selected):
                                    raise ValueError("선택 모델의 파일이 실행 묶음을 벗어났습니다.")
                                zipped.write(path, path.relative_to(selected).as_posix())
                    value.update(archive=str(archive), archive_sha256=digest(archive))
                    result["result"] = value
                else:
                    raise ValueError("지원하지 않는 분석 작업입니다.")
        except Exception as error:
            result["error"] = str(error)
        result["completed_at"] = now()
        write_json(result_path, result)
        return 1 if result.get("error") else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    raise SystemExit(execute(parser.parse_args().request))
