"""Web choices use cached synthetic fixtures, never the user's trained model."""
from pathlib import Path
from types import SimpleNamespace
import http.client
import json
import shutil
import threading
import zipfile

import pytest

from test_korean_library_selection import experiment
from korean_openwakeword import studio
from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.studio_analysis import StudioAnalysis, _parameters, execute


@pytest.fixture
def analysis(experiment, monkeypatch):
    project, old_bundle, old_evaluation, ids = experiment
    root = project.parent
    bundle, evaluation = root / "source_bundle", root / "evaluation"
    shutil.copytree(old_bundle, bundle)
    shutil.copytree(old_evaluation, evaluation)
    state = {"project": str(project), "wakeword": "춘식아", "status": "completed", "job": None,
             "config_sha256": digest(project), "results": {
                 "package": {"result": {"bundle": str(bundle)}},
                 "evaluation": {"result": {"evaluation": str(evaluation)}}}}
    manager = SimpleNamespace(_load=lambda project_id: state, _lock=threading.RLock(),
        token="analysis-local-token", registry={"projects": {"fixture": str(project)}},
        _job_alive=lambda job: False, _microphones={})
    calls = []
    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(pid=41000)
    monkeypatch.setattr(studio, "process_identity", lambda pid: "worker-birth" if pid == 41000 else None)
    service = StudioAnalysis(manager, popen=popen)
    return service, state, project, bundle, evaluation, ids, calls


def protected(project, bundle, evaluation):
    return {path: digest(path) for path in [project, *bundle.rglob("*"), *evaluation.rglob("*")] if path.is_file()}


def test_initial_status_does_not_select_or_write_anything(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    before = protected(project, bundle, evaluation)
    view = service.status("fixture")
    assert view["status"] == "idle" and view["selection"] is None
    assert view["default_candidate"] == ids[-1]
    assert not (project.parent / ".studio" / "analysis").exists()
    assert not calls and before == protected(project, bundle, evaluation)


@pytest.mark.parametrize("field,value", [("candidate", "invented"), ("thresholds", []),
    ("thresholds", [True]), ("thresholds", [float("nan")]), ("thresholds", [1]),
    ("thresholds", [.5] * 8), ("consecutive", [True]), ("consecutive", [0]),
    ("consecutive", [11]), ("consecutive", [1, 2, 3, 4])])
def test_untrusted_settings_are_bounded_before_worker_launch(analysis, field, value):
    service, state, project, bundle, evaluation, ids, calls = analysis
    payload = {"candidate": ids[0], "thresholds": [.5], "consecutive": [2], field: value}
    with pytest.raises(ValueError):
        service.launch("fixture", "compare", payload)
    assert not calls


def test_compare_runs_in_separate_identified_worker_and_preserves_sources(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    before = protected(project, bundle, evaluation)
    result = service.launch("fixture", "compare", {"candidate": ids[1]})
    assert result["status"] == "running"
    args, options = calls[-1]
    assert args[1:3] == ["-m", "korean_openwakeword.studio_analysis"]
    assert options["cwd"] == project.parent
    assert service.busy("fixture")
    with pytest.raises(ValueError, match="이미"):
        service.launch("fixture", "compare", {"candidate": ids[1]})
    assert execute(args[-1]) == 0
    view = service.status("fixture")
    assert view["status"] == "completed"
    assert view["comparison"]["candidate"]["id"] == ids[1]
    assert view["selection"] is None
    assert before == protected(project, bundle, evaluation)


def test_explicit_selection_exports_portable_bundle_and_releases_old_choice(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    before = protected(project, bundle, evaluation)
    service.launch("fixture", "select", {"candidate": ids[1]})
    assert execute(calls[-1][0][-1]) == 0
    view = service.status("fixture")
    assert view["selection"]["candidate"] == ids[1]
    assert service.active_bundle("fixture") == view["selected"]["bundle"]
    archive = service.download("fixture", view["job"]["task_id"])
    with zipfile.ZipFile(archive) as zipped:
        assert "manifest.json" in zipped.namelist()
        assert len([name for name in zipped.namelist() if name.startswith("models/")]) == 1
    assert before == protected(project, bundle, evaluation)
    active = Path(service.active_bundle("fixture")) / "manifest.json"
    active.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="변경"):
        service.active_bundle("fixture")


def test_changed_settings_require_a_verified_comparison_and_no_original_apply(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    before = protected(project, bundle, evaluation)
    service.launch("fixture", "select", {"candidate": ids[1], "thresholds": [.7], "consecutive": [1]})
    assert execute(calls[-1][0][-1]) == 1
    assert service.status("fixture")["status"] == "error"
    service.launch("fixture", "compare", {"candidate": ids[1], "thresholds": [.7], "consecutive": [1]})
    assert execute(calls[-1][0][-1]) == 0
    compare = service.status("fixture")
    service.launch("fixture", "select", {"candidate": ids[1], "thresholds": [.7], "consecutive": [1], "comparison_id": compare["comparison_id"]})
    assert execute(calls[-1][0][-1]) == 0
    assert service.status("fixture")["selection"]["runtime"]["threshold"] == .7
    assert before == protected(project, bundle, evaluation)


def test_live_microphone_or_training_blocks_new_analysis(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    service.manager._microphones["fixture"] = SimpleNamespace(active=lambda: True)
    with pytest.raises(ValueError, match="마이크"):
        service.launch("fixture", "compare", {"candidate": ids[0]})
    service.manager._microphones.clear()
    service.manager._job_alive = lambda job: True
    with pytest.raises(ValueError, match="학습"):
        service.launch("fixture", "compare", {"candidate": ids[0]})
    assert not calls


def test_final_lock_and_changed_project_are_not_bypassed(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    (project.parent / "final_lock.json").write_text("{}", encoding="utf-8")
    assert not service.status("fixture")["selection_allowed"]
    with pytest.raises(ValueError, match="최종"):
        service.launch("fixture", "compare", {"candidate": ids[0]})
    state["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="설정"):
        service.status("fixture")


def test_interrupted_worker_is_not_a_running_lock_or_fabricated_success(analysis, monkeypatch):
    service, state, project, bundle, evaluation, ids, calls = analysis
    service.launch("fixture", "compare", {"candidate": ids[0]})
    monkeypatch.setattr(studio, "process_identity", lambda pid: None)
    view = service.status("fixture")
    assert view["status"] == "interrupted" and view["selection"] is None
    assert not service.busy("fixture")


def test_http_analysis_assets_tokens_and_explicit_job_launch(analysis):
    service, state, project, bundle, evaluation, ids, calls = analysis
    manager = service.manager
    manager.analysis = service
    server = studio.make_studio_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(path, method="GET", payload=None, **headers):
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        else:
            body = None
        client.request(method, path, body=body, headers=headers)
        response = client.getresponse()
        result = response.status, response.read()
        client.close()
        return result
    try:
        base = "/analysis/fixture/"
        assert request(base)[0] == 200
        assert request(base + "analysis.js")[0] == 200
        assert request(base + "analysis.css")[0] == 200
        status, content = request(base + "api/status")
        assert status == 200 and json.loads(content)["selection"] is None
        assert request(base + "api/compare", "POST", {"candidate": ids[1]})[0] == 403
        assert request(base + "api/compare", "POST", {"candidate": ids[1]},
                       **{"X-Koww-Token": manager.token, "Origin": "https://outside.invalid"})[0] == 403
        assert request(base + "api/compare", "POST", {"candidate": ids[1]},
                       **{"X-Koww-Token": manager.token})[0] == 200
        assert len(calls) == 1
        assert request(base + "download/../../secret")[0] == 400
        assert request(base + "unrecognized")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
