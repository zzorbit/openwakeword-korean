"""A stale manager refuses mutation before a strict worker could reject it."""
from copy import deepcopy

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json, write_json
from test_reference_candidate_manager import review, send
from test_korean_library_studio import manager


def files_under(root):
    return {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}


def test_stale_manager_refuses_more_before_any_candidate_allocation_or_dispatch(review, monkeypatch):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    before = files_under(root)
    monkeypatch.setattr(studio, "_implementation", lambda: "source-replaced-after-manager-startup")

    with pytest.raises(ValueError, match="서버를 다시 시작"):
        send(service, project_id, "more", group=view["active_group"])

    assert files_under(root) == before
    assert not service.residents
    assert not list((root / ".studio/reference-candidates/jobs").glob("*"))


def test_fresh_manager_keeps_normal_guard_and_real_configuration_checks(review, monkeypatch):
    service, project_id, root = review
    state = service._load(project_id)
    service._guard_config(state)
    config = read_json(state["project"])
    config["wakeword"] = "변경된 호출어"
    write_json(state["project"], config)
    with pytest.raises(ValueError, match="모델 학습 설정이 변경"):
        service._guard_config(state)
    assert not service.residents


def test_freshness_does_not_authorize_a_changed_computational_contract(review):
    service, project_id, _ = review
    state = service._load(project_id)
    current = deepcopy(service._execution_contract)
    current["files"]["tts.py"] = "0" * 64
    current["computation_sha256"] = identity(current["files"])
    for stage in current["stages"]:
        current["stages"][stage] = "1" * 64
    service._implementation_id = "new-runtime-with-changed-tts"
    service._execution_contract = current
    assert service._migration_view(state)["state"] == "blocked"
    with pytest.raises(ValueError, match="라이브러리 코드가 변경"):
        service._guard_config(state)


def test_display_only_migration_keeps_failed_candidate_job_and_all_prior_evidence(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    saved_path = root / ".studio/reference-candidates/state.json"
    saved = read_json(saved_path)
    failed = saved["job"]
    request_path = root / ".studio/reference-candidates/jobs" / failed["task_id"] / "request.json"
    write_json(request_path.parent / "result.json", {
        "task_id": failed["task_id"], "request_sha256": digest(request_path),
        "error": "다시 생성하기 전에 실행 코드 또는 모델 설정이 바뀌었습니다.",
    })
    service.residents[-1].returncode = 1
    service.reference_preview_queue(project_id)
    saved = read_json(saved_path)
    assert saved["job"]["status"] == "failed" and saved["job"]["resumable"] is False
    before = files_under(root)
    old_state = service._load(project_id)
    service._implementation_id = "freshly-restarted-display-only-runtime"
    service.migrate(project_id)
    after = files_under(root)
    new_state = service._load(project_id)

    assert read_json(saved_path) == saved
    assert all(after[path] == sha for path, sha in before.items() if path != ".studio/workflow.json")
    for key in ("settings", "plan", "index", "history", "inputs", "results", "status", "config_sha256"):
        assert new_state.get(key) == old_state.get(key)
    assert new_state["implementation_id"] == service._implementation_id
    assert len(new_state.get("migrations", [])) == len(old_state.get("migrations", [])) + 1
    failed_evidence = {name: digest(request_path.parent / name) for name in ("request.json", "result.json")}
    send(service, project_id, "more", group=view["active_group"])
    retry = read_json(saved_path)["job"]
    assert retry["task_id"] != failed["task_id"] and retry["status"] == "running"
    retry_request = read_json(root / ".studio/reference-candidates/jobs" / retry["task_id"] / "request.json")
    assert retry_request["implementation_id"] == service._implementation_id
    assert all(digest(request_path.parent / name) == sha for name, sha in failed_evidence.items())


def test_worker_still_rejects_a_source_change_after_dispatch_before_loading_a_model(review, monkeypatch):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    monkeypatch.setattr(studio, "_implementation", lambda: "source-changed-after-dispatch")
    assert service.finish_candidates() == 1
    assert service.fake_model.counter == 0
    job = read_json(root / ".studio/reference-candidates/state.json")["job"]
    result = read_json(root / ".studio/reference-candidates/jobs" / job["task_id"] / "result.json")
    assert result["error"] == "다시 생성하기 전에 실행 코드 또는 모델 설정이 바뀌었습니다."
