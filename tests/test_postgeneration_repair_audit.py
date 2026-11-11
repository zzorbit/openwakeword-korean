"""The frozen selective-repair release preserves completed computation identities."""
from copy import deepcopy
from pathlib import Path

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json
from test_korean_library_studio import manager  # noqa: F401
from test_reference_candidate_manager import review  # noqa: F401


def previous_contract():
    return read_json(Path(__file__).parent / "fixtures/postgeneration_repair_previous_contract.json")


def target_contract():
    # Historical release audits stay fixed when unrelated later releases change.
    contract = read_json(Path(__file__).parent / "fixtures/postgeneration_repair_target_contract.json")
    assert identity(contract["files"]) == contract["computation_sha256"]
    return contract


def pin_previous(service, project_id):
    service._execution_contract = target_contract()
    state = service._load(project_id)
    state["implementation_id"] = "0" * 64
    state["execution_contract"] = {**previous_contract(),
        "settings_sha256": identity(state["settings"]), "plan_sha256": identity(state["plan"])}
    service._save(state)
    return state


def test_exact_release_only_and_completed_generation_stage_stays_identical():
    previous, current = previous_contract(), target_contract()
    update = studio._audited_call_generation_recovery_update(previous, current)
    assert update["kind"] == "audited_postgeneration_voice_repair_update"
    assert update["changed_files"] == ["postgeneration_repair.py", "studio_steps.py"]
    assert [stage for stage in previous["stages"] if previous["stages"][stage] != current["stages"][stage]] == ["positive_review"]
    for side in ("source", "target"):
        old, new = deepcopy(previous), deepcopy(current)
        altered = old if side == "source" else new
        altered["files"]["tts.py"] = "0" * 64
        altered["computation_sha256"] = identity(altered["files"])
        assert studio._audited_call_generation_recovery_update(old, new) is None
        old, new = deepcopy(previous), deepcopy(current)
        (old if side == "source" else new)["stages"]["positives"] = "0" * 64
        assert studio._audited_call_generation_recovery_update(old, new) is None


def test_migration_preserves_all_audio_reports_results_and_human_answers(review):
    service, project_id, root = review
    state = service._load(project_id)
    state.update(status="waiting_review", auto_continue=False,
                 index=next(i for i, item in enumerate(state["plan"]) if item["id"] == "positive_review"))
    state["results"]["positive_review"] = {"state": "waiting_review", "reviews": []}
    service._save(state)
    frozen = root / "data/approved.wav"
    frozen.write_bytes(b"existing accepted call and its exact identity")
    before = deepcopy(pin_previous(service, project_id))
    files = {path: digest(path) for path in root.rglob("*") if path.is_file()}
    assert service._migration_view(before)["state"] == "available"
    service.migrate(project_id)
    after = service._load(project_id)
    for key in ("settings", "plan", "inputs", "results", "history", "index", "config_sha256", "last_job"):
        assert after.get(key) == before.get(key), key
    assert after["status"] == "waiting_review" and after["auto_continue"] is False
    assert all(digest(path) == value for path, value in files.items() if path != service._path(project_id))
    receipt = read_json(after["migrations"][-1]["path"])
    assert receipt["audited_call_generation_recovery_update"]["kind"] == "audited_postgeneration_voice_repair_update"


def test_trained_locked_active_or_completed_projects_cannot_be_rebound(review, monkeypatch):
    service, project_id, root = review
    state = pin_previous(service, project_id)
    assert service._can_update_call_generation_recovery(state)
    feature_position = next(i for i, item in enumerate(state["plan"]) if item["id"] == "features")
    for changes in ({"status": "completed"}, {"index": feature_position},
                    {"history": [{"stage": "training"}]}, {"results": {"training": {"state": "completed"}}}):
        altered = deepcopy(state)
        altered.update(changes)
        assert not service._can_update_call_generation_recovery(altered)
    monkeypatch.setattr(service, "_job_alive", lambda job: bool(job))
    active = deepcopy(state)
    active["job"] = {"status": "running"}
    assert not service._can_update_call_generation_recovery(active)
    (root / "final_lock.json").write_text("{}", encoding="utf-8")
    assert not service._can_update_call_generation_recovery(state)
