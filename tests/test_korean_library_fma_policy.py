"""FMA transaction selection, legacy compatibility and storage forecasting."""
from pathlib import Path

import pytest

from korean_openwakeword import automatic_sources as module
from korean_openwakeword.common import digest, identity, now, read_json, write_json
from korean_openwakeword.project import create_project
from korean_openwakeword.studio import StudioManager
from korean_openwakeword.studio_data import material_recipe, _automatic_space_plan
from korean_openwakeword.studio_steps import common_evaluation_policy, preset_settings, workflow_plan


POLICY = "per_track_flac_prefetch_v1"


def test_streaming_forecast_keeps_sources_payload_and_transfer_unchanged():
    config = {"targets": {"positive_originals": 100000}, "augmentation_rounds": 5}
    old = module.automatic_storage_plan(config, material_acquisition_policy="automatic_v3")
    new = module.automatic_storage_plan(config, material_acquisition_policy="automatic_v3",
                                        fma_acquisition_policy=POLICY)
    assert "fma_acquisition_policy" not in old
    assert new["fma_acquisition_policy"] == POLICY
    for key in ("minimum_known_bytes", "normalized_pcm16_bytes", "negative_feature_payload_bytes",
                "positive_feature_payload_bytes", "estimated_download_bytes"):
        assert new[key] == old[key]
    assert old["transient_source_bytes"] == 62_500_000_000
    assert new["transient_source_bytes"] == 8 * 1024**3
    assert old["estimated_total_bytes"] - new["estimated_total_bytes"] == 62_500_000_000 - 8 * 1024**3


@pytest.mark.parametrize("policy, storage", [("unknown", "automatic_v3"),
    (POLICY, "automatic_v1"), (POLICY, "automatic_v2")])
def test_incompatible_policy_is_rejected_before_download(policy, storage):
    with pytest.raises(ValueError, match="FMA|FLAC"):
        module.automatic_source_policy(material_acquisition_policy=storage, fma_acquisition_policy=policy)


def test_streaming_dispatch_has_its_own_namespace_and_no_batch_acquisition(tmp_path, monkeypatch):
    from korean_openwakeword import fma_streaming
    calls = []
    def prepare(directory, **kwargs):
        calls.append((Path(directory), kwargs))
        return {"marker": "streaming"}
    monkeypatch.setattr(fma_streaming, "prepare_fma_streaming_source", prepare)
    monkeypatch.setattr(module.PublicSourceBackend, "_prepare", lambda *a, **k: pytest.fail("batch path"))
    backend = module.PublicSourceBackend(tmp_path, storage_policy="canonical_flac_only_v1",
                                        fma_acquisition_policy=POLICY)
    result = backend.prepare({"component": "fma_full"}, tmp_path / "part", target_hours=500,
        max_download_bytes=1000, max_extract_bytes=2000, excluded_track_ids=(12, 13))
    assert result == {"marker": "streaming"}
    assert calls[0][0].is_relative_to(tmp_path / "part" / POLICY)
    assert calls[0][1]["excluded_track_ids"] == (12, 13)


def test_recipe_and_custom_project_storage_plan_keep_explicit_policy():
    kwargs = {"material_acquisition_policy": "automatic_v3", "evaluation_policy": common_evaluation_policy()}
    old = material_recipe("production", **kwargs)
    new = material_recipe("production", fma_acquisition_policy=POLICY, **kwargs)
    assert "fma_acquisition_policy" not in old
    assert new["fma_acquisition_policy"] == new["automatic_space_plan"]["fma_acquisition_policy"] == POLICY
    assert old["known_download_bytes"] == new["known_download_bytes"]
    assert old["automatic_source_plan"] == new["automatic_source_plan"]
    config = {"targets": {"generic_hours": 3500, "korean_hours": 1500, "positive_originals": 100000},
              "augmentation_rounds": 5, "fma_acquisition_policy": POLICY}
    space = _automatic_space_plan("production", config, "automatic_v3")
    assert space["fma_acquisition_policy"] == POLICY
    assert space["transient_source_bytes"] == 8 * 1024**3
    assert module.automatic_storage_plan(config, material_acquisition_policy="automatic_v3")["fma_acquisition_policy"] == POLICY


def test_resuming_registered_legacy_project_does_not_enable_new_fma_transaction(tmp_path):
    directory = tmp_path / "old"
    create_project(directory, "보라야", "bora", positive_originals=100000,
                   augmentation_rounds=5, updates=100000, generic_hours=3500, korean_hours=1500,
                   validation_voice_families=30)
    project = directory / "project.json"
    original = project.read_bytes()
    manager = StudioManager(tmp_path / "workspace")
    try:
        # Saved web registrations remain readable at their original location.
        # Construct the historical record explicitly: web creation no longer
        # accepts arbitrary directory payloads or attaches unregistered data.
        settings = preset_settings("production")
        settings.pop("fma_acquisition_policy")
        settings.pop("audio_storage_policy")
        settings["material_acquisition_policy"] = "automatic_v2"
        plan = workflow_plan(settings)
        state = {"schema_version": 1, "id": "legacy-bora", "workspace": str(manager.workspace),
                 "storage_workspace": str(directory), "project": str(project), "wakeword": "보라야",
                 "preset": "production", "settings": settings, "plan": plan,
                 "config_sha256": digest(project), "implementation_id": manager._implementation_id,
                 "created_at": now(), "status": "ready", "index": 0, "results": {}, "inputs": {},
                 "job": None, "error": None, "next_action": "기존 학습 이어서 진행"}
        state["execution_contract"] = {**manager._execution_contract,
            "settings_sha256": identity(settings), "plan_sha256": identity(plan)}
        manager.registry["projects"][state["id"]] = str(project)
        write_json(manager.registry_path, manager.registry)
        manager._save(state)
        workflow = manager._path(state["id"]).read_bytes()
        view = manager.create({"wakeword": "보라야", "preset": "production"})
        assert view["id"] == state["id"] and Path(view["directory"]) == directory
        assert "fma_acquisition_policy" not in manager._load(view["id"])["settings"]
        assert project.read_bytes() == original
        assert manager._path(state["id"]).read_bytes() == workflow
        assert "fma_acquisition_policy" not in read_json(project)
        assert not (manager.workspace / "models").exists()
    finally:
        manager.close()


def test_new_commercial_web_project_persists_measured_policy(tmp_path):
    manager = StudioManager(tmp_path / "workspace")
    try:
        recipes = manager.recipes()["presets"]
        production = next(row for row in recipes if row["id"] == "production")
        assert production["fma_acquisition_policy"] == POLICY
        assert production["automatic_space_plan"]["estimated_total_bytes"] == 699_246_834_450
        assert all("fma_acquisition_policy" not in row for row in recipes if row["id"] != "production")
        view = manager.create({"wakeword": "보라야", "preset": "production"})
        state = manager._load(view["id"])
        assert state["settings"]["fma_acquisition_policy"] == POLICY
        assert read_json(state["project"])["fma_acquisition_policy"] == POLICY
    finally:
        manager.close()
