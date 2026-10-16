"""One bounded Studio stage, launched by StudioManager (not a public CLI)."""
from pathlib import Path
import argparse
import os
import traceback

from .common import digest, now, read_json, write_json, writer_lock
from .storage import storage_environment
from .project import load_project


def execute(request_path):
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
        return _execute(request_path)


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
        before = load_project(request["project"])[1]
        if digest(request["project"]) != request["config_sha256"]:
            raise ValueError("Project settings changed before execution")
        if request.get("implementation_id"):
            from .studio import _implementation
            if _implementation() != request["implementation_id"]:
                raise ValueError("Library implementation changed before execution; previous artifacts preserved")
        from .studio_steps import run_stage
        with storage_environment(Path(request["project"]).parent):
            result["result"] = run_stage(request["project"], request["workspace"], request["settings"],
                                         request["stage"], request.get("context", {}))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    raise SystemExit(execute(parser.parse_args().request))
