"""Only a frozen, audited release may rebind a pre-training workflow."""
from copy import deepcopy
from pathlib import Path
import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json, write_json
from test_korean_library_studio import manager  # noqa: F401
from test_reference_candidate_manager import review  # noqa: F401


@pytest.fixture(autouse=True)
def frozen_autosave_release(monkeypatch):
    """Keep historical release proofs independent of later audited releases."""
    release = read_json(Path(__file__).parent / "fixtures/postgeneration_repair_previous_contract.json")
    monkeypatch.setattr(studio, "_execution_contract", lambda **kwargs: deepcopy(release))


def previous_contract():
    return read_json(Path(__file__).parent / "fixtures/preflight_autosave_previous_contract.json")


def pin_previous(service, project_id):
    state = service._load(project_id)
    state["implementation_id"] = "0" * 64
    state["execution_contract"] = {**previous_contract(),
        "settings_sha256": identity(state["settings"]), "plan_sha256": identity(state["plan"])}
    service._save(state)
    return state


def test_only_exact_audited_source_target_and_stage_contracts_are_accepted():
    previous, current = previous_contract(), studio._execution_contract()
    update = studio._audited_call_generation_recovery_update(previous, current)
    assert update["kind"] == "audited_preflight_autosave_update"
    assert update["changed_files"] == ["call_generation_history.py", "clone_reference.py", "pilot.py",
                                       "reference_candidates.py", "review.py", "tts.py"]
    for side in ("source", "target"):
        old, new = deepcopy(previous), deepcopy(current)
        altered = old if side == "source" else new
        altered["files"]["train.py"] = "0" * 64
        altered["computation_sha256"] = identity(altered["files"])
        assert studio._audited_call_generation_recovery_update(old, new) is None
        old, new = deepcopy(previous), deepcopy(current)
        (old if side == "source" else new)["stages"]["pilot"] = "0" * 64
        assert studio._audited_call_generation_recovery_update(old, new) is None
    unbound = deepcopy(previous)
    unbound["files"]["tts.py"] = "0" * 64
    assert studio._audited_call_generation_recovery_update(unbound, current) is None


def test_original_recovery_release_is_still_an_exact_historical_binding():
    original = read_json(Path(__file__).parent / "fixtures/call_recovery_previous_contract.json")
    recovered = previous_contract()
    update = studio._audited_call_generation_recovery_update(original, recovered)
    assert update["kind"] == "audited_call_generation_recovery_update"
    assert update["target_computation_sha256"] == "6431480661a735e7d43424958721adc6b9a602850bb08f6433a5924263d2af7d"
    assert studio._audited_call_generation_recovery_update(original, studio._execution_contract()) is None


def test_partial_generation_migration_preserves_artifacts_and_review_then_unblocks_resume(review):
    service, project_id, root = review
    service.reference_preview_queue(project_id)
    state = service._load(project_id)
    state.update(status="waiting_input", auto_continue=False)
    state["results"]["pilot"] = {"state": "waiting_input", "input_kind": "generation_policy",
        "result": {"target": 50, "technical_pass": 49, "full_target_reached": False}}
    service._save(state)
    preserved_audio = root / "data/call_pronunciation/proven/train/passed.wav"
    preserved_audio.parent.mkdir(parents=True)
    preserved_audio.write_bytes(b"existing passed output is immutable")
    before = deepcopy(pin_previous(service, project_id))
    files_before = {path: digest(path) for path in root.rglob("*") if path.is_file()}
    assert service._migration_view(before)["state"] == "available"
    service.migrate(project_id)
    after = service._load(project_id)
    for key in ("settings", "plan", "index", "inputs", "history", "results", "config_sha256", "last_job"):
        assert after.get(key) == before.get(key), key
    assert after["status"] == "paused" and after["auto_continue"] is False
    assert all(digest(path) == sha for path, sha in files_before.items() if path != service._path(project_id))
    receipt = read_json(after["migrations"][-1]["path"])
    assert receipt["audited_call_generation_recovery_update"]["target_computation_sha256"] == studio._execution_contract()["computation_sha256"]
    assert after["execution_contract"]["settings_sha256"] == before["execution_contract"]["settings_sha256"]
    assert after["execution_contract"]["plan_sha256"] == before["execution_contract"]["plan_sha256"]


def test_other_waiting_input_is_not_silently_submitted(review):
    service, project_id, _ = review
    state = service._load(project_id)
    state.update(status="waiting_input", input_submitted=False)
    state["results"]["pilot"] = {"state": "waiting_input", "input_kind": "dependency_backend"}
    service._save(state)
    pin_previous(service, project_id)
    service.migrate(project_id)
    after = service._load(project_id)
    assert after["status"] == "waiting_input" and after["input_submitted"] is False


def test_active_trained_and_final_work_are_rejected_without_reinterpreting_old_results(review, monkeypatch):
    service, project_id, root = review
    state = pin_previous(service, project_id)
    assert service._can_update_call_generation_recovery(state)
    positions = {row["id"]: i for i, row in enumerate(state["plan"])}
    for changes in ({"status": "completed"}, {"index": positions["features"]},
                    {"results": {"training": {"state": "completed"}}},
                    {"history": [{"stage": "features"}]}):
        altered = deepcopy(state)
        altered.update(changes)
        assert not service._can_update_call_generation_recovery(altered)
    monkeypatch.setattr(service, "_job_alive", lambda job: bool(job))
    active = deepcopy(state)
    active["job"] = {"status": "running"}
    assert not service._can_update_call_generation_recovery(active)
    (root / "final_lock.json").write_text("{}", encoding="utf-8")
    assert not service._can_update_call_generation_recovery(state)


def test_detached_features_block_the_update_and_completed_workflows_are_not_rewritten(review):
    service, project_id, root = review
    state = pin_previous(service, project_id)
    feature = root / "features/existing-feature.npy"
    feature.parent.mkdir(exist_ok=True)
    feature.write_bytes(b"existing feature evidence")
    assert not service._can_update_call_generation_recovery(state)
    state["status"] = "completed"
    service._save(state)
    before = digest(service._path(project_id))
    service.migrate(project_id)
    assert digest(service._path(project_id)) == before


def test_verified_precomputed_generic_metadata_is_preserved_but_model_features_still_block(review):
    service, project_id, root = review
    from korean_openwakeword.frontend import PINNED_ASSETS
    state = pin_previous(service, project_id)
    config = read_json(state["project"])
    folder = root / config["pool_features"]["generic_negative"]
    folder.mkdir(parents=True, exist_ok=True)
    source = root / "public-features.npy"
    source.write_bytes(b"existing public feature fixture")
    path = folder / "metadata.json"
    metadata = {"role": "generic_negative", "split": "train", "imported_precomputed": True,
        "quality_review": "accepted", "feature_rows": 12, "feature_path": str(source),
        "feature_sha256": digest(source), "canonical_frontend_assets": {
            "melspectrogram": PINNED_ASSETS["melspectrogram.onnx"],
            "embedding": PINNED_ASSETS["embedding_model.onnx"]}}
    write_json(path, metadata)
    assert not service._can_update_call_generation_recovery(state)
    result = root / "public-import-result.json"
    write_json(result, {"result": {"state": "completed", "result": {"generic_feature_import": {
        "imported": True, "features": str(source), "output": str(folder)}}}})
    state.setdefault("history", []).append({"stage": "data_connect", "result": str(result), "sha256": digest(result)})
    service._save(state)
    assert service._can_update_call_generation_recovery(state)
    view = service._migration_view(state)
    binding = view["audited_call_generation_recovery_update"]["preserved_precomputed_generic_metadata"]
    assert binding["sha256"] == digest(path)
    before = digest(path)
    service.migrate(project_id)
    assert digest(path) == before
    (folder / "model-specific.npy").write_bytes(b"derived model feature")
    assert not service._can_update_call_generation_recovery(state)
