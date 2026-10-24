"""Public defaults agree; explicit budgets and historical projects stay intact."""
import pytest

from korean_openwakeword.automatic_sources import automatic_storage_plan, automatic_source_policy
from korean_openwakeword.cli import dispatch, parser
from korean_openwakeword.commercial_sources import commercial_negative_plan
from korean_openwakeword.project import create_project, default_profile, load_project, validate_project, workload_summary
from korean_openwakeword.studio import StudioManager
from korean_openwakeword.studio_steps import preset_settings


def test_new_project_cli_and_studio_share_the_5000_hour_plan(tmp_path):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    config = load_project(project)[1]
    studio = next(row for row in StudioManager(tmp_path / "web").recipes()["presets"]
                  if row["id"] == "production")
    cli = dispatch(parser().parse_args(["negative-plan"]))
    assert cli == studio["training_negative_plan"] == commercial_negative_plan()
    assert (config["targets"]["generic_hours"], config["targets"]["korean_hours"]) == (3500, 1500)
    assert [row["target_hours"] for row in cli["components"]] == [1500, 1500, 500, 1500]
    assert cli["total_hours"] == 5000
    automatic = automatic_source_policy()
    assert list(automatic["component_targets"].values()) == [1500, 1500, 500, 1500]
    assert studio["automatic_target_hours"] == {"generic_raw": 3500, "korean_train": 1500}
    assert (config["targets"]["positive_originals"], config["augmentation_rounds"],
            config["training"]["updates"]) == (100000, 5, 100000)
    assert config["batch"] == {"generic_negative": 768, "korean_negative": 256,
                               "positive": 100, "hard_negative": 100}
    assert preset_settings("check")["korean_hours"] == 1
    assert preset_settings("trial")["korean_hours"] == pytest.approx(2000 / 3)


def test_half_hours_reduce_variable_storage_without_halving_fixed_costs():
    config = default_profile()
    old = automatic_storage_plan(config, generic_hours=7000, korean_hours=3000,
                                 material_acquisition_policy="automatic_v3")
    new = automatic_storage_plan(config, material_acquisition_policy="automatic_v3")
    for key in ("normalized_pcm16_bytes", "negative_feature_payload_bytes",
                "estimated_download_bytes", "transient_source_bytes"):
        assert new[key] * 2 == old[key]
    assert new["positive_feature_payload_bytes"] == old["positive_feature_payload_bytes"] == 3072000000
    assert new["future_buffer_bytes"] == old["future_buffer_bytes"]
    assert new["canonical_flac_estimated_bytes"] is None
    assert new["estimated_total_bytes"] == 705457934592


def test_saved_10000_hour_project_is_never_rewritten_by_loading(tmp_path):
    project = create_project(tmp_path / "old", "보라야", "old",
                             generic_hours=7000, korean_hours=3000)
    before = project.read_bytes()
    config = load_project(project)[1]
    plan = commercial_negative_plan(config["targets"]["generic_hours"], config["targets"]["korean_hours"])
    assert [row["target_hours"] for row in plan["components"]] == [3000, 3000, 1000, 3000]
    assert config["targets"]["generic_hours"] == 7000
    assert project.read_bytes() == before


def test_selective_music_cli_defaults_follow_the_new_source_budget(tmp_path):
    for argv in (["download-source", "--source", "fma_full"],
                 ["download-fma", "--output", str(tmp_path)]):
        assert parser().parse_args(argv).hours == 500


def test_explicit_noise_target_does_not_require_a_legacy_range():
    config = default_profile()
    config["targets"]["noise_hours"] = 40
    del config["targets"]["noise_hours_phase1"]
    assert validate_project(config) == config
    assert workload_summary(config)["data_targets"]["environment_noise_hours"] == 40


@pytest.mark.parametrize("key", ["generic_hours", "korean_hours"])
def test_missing_negative_hours_are_not_replaced_by_noise_hours(key):
    config = default_profile()
    del config["targets"][key]
    with pytest.raises(ValueError, match="hours"):
        validate_project(config)


def test_missing_noise_target_reports_a_validation_error():
    config = default_profile()
    del config["targets"]["noise_hours_phase1"]
    with pytest.raises(ValueError, match="Noise preparation hours"):
        validate_project(config)


def test_new_cli_and_web_initialization_use_2000_common_calls_without_rewriting_old_profiles(tmp_path):
    baseline = default_profile()
    assert baseline["targets"]["validation_originals"] == 5000
    old_project = create_project(tmp_path / "saved", "보라야", "saved")
    old_bytes = old_project.read_bytes()
    parsed = parser().parse_args(["init", "--wakeword", "보라야", "--id", "bora",
        "--output", str(tmp_path / "cli"), "--non-interactive", "--validation-voices", "30"])
    result = dispatch(parsed)
    config = load_project(result["project"])[1]
    assert config["targets"]["validation_originals"] == 2000
    assert config["targets"]["final_originals"] == 5000
    manager = StudioManager(tmp_path / "web")
    try:
        for preset in ("check", "trial", "production"):
            model = manager.create({"wakeword": "보라야", "preset": preset})
            state = manager._load(model["id"])
            assert load_project(state["project"])[1]["targets"]["validation_originals"] == 2000
            assert state["settings"]["evaluation_policy"]["positive_sessions"] == 48000
        assert manager._children == {}
    finally:
        manager.close()
    assert old_project.read_bytes() == old_bytes
    assert default_profile() == baseline
