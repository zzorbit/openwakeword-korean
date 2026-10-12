"""Workload selection must reach generation/features/training without weakening admission."""
import json
from copy import deepcopy

import pytest

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.project import (
    BATCH, configure_budget, create_project, default_profile, load_project,
    save_configuration, workload_summary,
)
from korean_openwakeword.setup import setup_choices
from korean_openwakeword.train import validate_training_scale


def test_selected_budget_keeps_learning_defaults_and_syncs_paths_schedule(tmp_path):
    baseline = default_profile()
    project = create_project(tmp_path / "word", "보라야", "bora",
        positive_originals=20000, augmentation_rounds=3, updates=20000)
    config = load_project(project)[1]
    assert config["training"]["schedule_horizon"] == 20000
    assert config["training"]["checkpoints"] == [4000, 8000, 20000]
    assert config["pool_features"]["positive"] == "features/positive_N20000_R3"
    assert config["batch"] == BATCH and config["hard_negative_samples"] == 100
    for key in ("architecture", "optimizer", "learning_rate", "dropout", "seeds", "backend"):
        assert config["training"][key] == baseline["training"][key]
    for key in ("validation_originals", "final_originals", "generic_hours", "korean_hours"):
        assert config["targets"][key] == baseline["targets"][key]
    summary = read_json(project.parent / "reports/setup_summary.json")
    assert summary["positive_feature_rows"] == 60000
    assert summary["positive_feature_payload_bytes"] == 60000 * 16 * 96 * 4
    assert summary["total_optimizer_updates"] == 60000
    assert summary["warmup_updates"] == 400


def test_formal_selected_budget_accepts_matching_counts_only():
    config = configure_budget(default_profile(), positive_originals=20000, augmentation_rounds=3, updates=20000,
                              generic_hours=2000, korean_hours=200)
    # Unit metadata fixture, not an audio approval or trained model.
    meta = {"positive": {"original_count": 20000, "rounds": 3, "feature_rows": 60000},
            "generic_negative": {"unique_source_hours": 2000}, "korean_negative": {"unique_source_hours": 200}}
    validate_training_scale(config, meta)
    for key, value in (("original_count", 100000), ("rounds", 1), ("feature_rows", 20000)):
        bad = deepcopy(meta); bad["positive"][key] = value
        with pytest.raises(ValueError, match="selected"):
            validate_training_scale(config, bad)
    meta["korean_negative"]["unique_source_hours"] = 199
    with pytest.raises(ValueError, match="200h"):
        validate_training_scale(config, meta)


def test_independent_workload_choices_and_shared_horizon():
    base = default_profile()
    for n, r, u in ((12000, 1, 10000), (50000, 3, 30000), (100000, 5, 50000)):
        selected = configure_budget(base, positive_originals=n, augmentation_rounds=r, updates=u)
        assert selected["targets"]["positive_originals"] == n
        assert selected["augmentation_rounds"] == r
        assert selected["training"]["updates"] == selected["training"]["schedule_horizon"] == u
    selected = configure_budget(base, updates=20000, training_overrides={"schedule_horizon": 50000})
    assert workload_summary(selected)["warmup_updates"] == 1000
    assert base == default_profile()  # no default mutation


def test_invalid_choices_fail_before_creating_a_project(tmp_path):
    for changes in ({"positive_originals": 0}, {"positive_originals": True}, {"augmentation_rounds": 2},
                    {"updates": 0}, {"updates": -1}, {"updates": True},
                    {"updates": 20000, "training_overrides": {"schedule_horizon": 10000}},
                    {"updates": 20000, "training_overrides": {"checkpoints": [10000, 50000]}},
                    {"training_overrides": {"architecture": "cnn999"}},
                    {"training_overrides": {"learning_rate": float("nan")}}):
        with pytest.raises(ValueError):
            create_project(tmp_path / "never_created", "보라야", "bora", **changes)
        assert not (tmp_path / "never_created").exists()


def test_terminal_wizard_basic_only_and_invalid_entry():
    answers = iter(["0", "20000", "3", "20000", "", "", "", "n"])
    output = []
    choices = setup_choices(default_profile(), interactive=True,
        input_fn=lambda prompt: next(answers), output_fn=output.append)
    config = configure_budget(default_profile(), **choices)
    assert config["training"]["architecture"] == "cnn128"
    assert config["training"]["updates"] == 20000
    assert choices["training_overrides"] == {}
    assert any("총 60,000" in line for line in output)
    assert any("다시 확인" in line for line in output)


def test_advanced_wizard_and_cancelled_setup():
    answers = iter(["50000", "3", "20000", "", "", "", "mlp128", "Adam", "0.0002", "0", "0.2", "2", "constant", "1", "cpu"])
    choices = setup_choices(default_profile(), interactive=True, advanced=True,
        input_fn=lambda prompt: next(answers), output_fn=lambda s: None)
    config = configure_budget(default_profile(), **choices)
    assert config["training"]["architecture"] == "mlp128"
    assert config["training"]["optimizer"] == "Adam"
    assert len(config["training"]["seeds"]) == 1
    assert workload_summary(config)["total_optimizer_updates"] == 20000
    def cancelled(prompt):
        raise EOFError()
    with pytest.raises(ValueError, match="cancelled"):
        setup_choices(default_profile(), interactive=True, input_fn=cancelled)


def test_script_mode_never_prompts_and_preserves_default():
    def never_ask(prompt):
        raise AssertionError("Script mode tried to prompt")
    choices = setup_choices(default_profile(), interactive=False, input_fn=never_ask)
    assert configure_budget(default_profile(), **choices) == default_profile()
    with pytest.raises(ValueError, match="interactive"):
        setup_choices(default_profile(), interactive=False, advanced=True, input_fn=never_ask)


def test_data_hours_selection_reaches_setup_summary_without_changing_learning(tmp_path):
    baseline = default_profile()
    answers = iter(["", "", "", "1000", "50", "nan", "40", "n"])
    choices = setup_choices(baseline, interactive=True,
        input_fn=lambda prompt: next(answers), output_fn=lambda s: None)
    project = create_project(tmp_path / "word", "보라야", "bora", **choices)
    root, config = load_project(project)
    assert config["targets"]["generic_hours"] == 1000
    assert config["targets"]["korean_hours"] == 50
    assert config["targets"]["noise_hours"] == 40
    assert config["training"] == baseline["training"] and config["batch"] == BATCH
    assert config["augmentation"] == baseline["augmentation"]
    summary = read_json(root / "reports/setup_summary.json")["data_targets"]
    assert summary["environment_noise_hours"] == 40
    assert summary["generic_negative_hours"] == 1000
    assert all((root / path).read_text() == "" for path in config["manifests"].values())
    assert baseline == default_profile()


def test_formal_training_checks_selected_negative_hours_and_publisher_nominal():
    config = configure_budget(default_profile(), generic_hours=1000, korean_hours=50)
    meta = {"positive": {"original_count": 100000, "rounds": 5, "feature_rows": 500000},
            "generic_negative": {"legacy_generic_bridge": True, "publisher_declared_hours": 2000},
            "korean_negative": {"unique_source_hours": 50}}
    validate_training_scale(config, meta)
    bad = deepcopy(meta); bad["korean_negative"]["unique_source_hours"] = 49.9
    with pytest.raises(ValueError, match="50h"):
        validate_training_scale(config, bad)
    with pytest.raises(ValueError, match="3000h"):
        validate_training_scale(configure_budget(config, generic_hours=3000), meta)
    bad = deepcopy(meta); bad["korean_negative"]["unique_source_hours"] = float("nan")
    with pytest.raises(ValueError):
        validate_training_scale(config, bad)


def test_invalid_data_hours_never_create_a_project(tmp_path):
    for key in ("generic_hours", "korean_hours", "noise_hours"):
        for value in (0, -1, True, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="hours"):
                create_project(tmp_path / "invalid", "보라야", "bora", **{key: value})
            assert not (tmp_path / "invalid").exists()


def test_check_command_and_legacy_doctor_share_environment_diagnostic(monkeypatch):
    from korean_openwakeword import benchmark
    from korean_openwakeword.cli import parser, dispatch
    report = {"versions": {"openwakeword": "0.6.0"}, "cuda_available": False}
    monkeypatch.setattr(benchmark, "doctor", lambda: report)
    assert dispatch(parser().parse_args(["check"])) == report
    assert dispatch(parser().parse_args(["doctor"])) == report


def test_cli_data_hours_and_snapshot_preserve_original(tmp_path, capsys):
    from korean_openwakeword.cli import main
    root = tmp_path / "word"
    main(["init", "--wakeword", "보라야", "--id", "bora", "--output", str(root),
          "--non-interactive", "--generic-hours", "2000", "--korean-hours", "250", "--noise-hours", "40"])
    result = json.loads(capsys.readouterr().out)
    assert result["workload"]["data_targets"]["korean_negative_hours"] == 250
    original = digest(root / "project.json")
    snapshot = root / "project_more_data.json"
    main(["configure", "--project", str(root / "project.json"), "--output", str(snapshot),
          "--non-interactive", "--korean-hours", "300"])
    capsys.readouterr()
    assert digest(root / "project.json") == original
    assert load_project(snapshot)[1]["targets"]["korean_hours"] == 300
    assert load_project(snapshot)[1]["targets"]["noise_hours"] == 40


def test_configuration_snapshot_preserves_original_and_final_lock(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    original = digest(project)
    snapshot = save_configuration(project, project.parent / "project_small.json",
        positive_originals=20000, augmentation_rounds=3, updates=20000)
    assert digest(project) == original
    assert load_project(snapshot)[1]["targets"]["positive_originals"] == 20000
    with pytest.raises(ValueError, match="overwrite"):
        save_configuration(project, snapshot, updates=10000)
    with pytest.raises(ValueError, match="beside"):
        save_configuration(project, tmp_path / "elsewhere.json", updates=10000)
    write_json(project.parent / "final_lock.json", {"test": "fixture only"})
    with pytest.raises(ValueError, match="frozen"):
        save_configuration(project, project.parent / "project_after_final.json", updates=10000)


def test_cli_selection_writes_actual_config_and_advanced_settings(tmp_path, capsys):
    from korean_openwakeword.cli import main
    root = tmp_path / "word"
    main(["init", "--wakeword", "보라야", "--id", "bora", "--output", str(root),
          "--non-interactive", "--originals", "20000", "--rounds", "3", "--updates", "20000"])
    result = json.loads(capsys.readouterr().out)
    assert result["workload"]["positive_feature_rows"] == 60000
    main(["configure", "--project", str(root / "project.json"), "--output", str(root / "project_single.json"),
          "--non-interactive", "--architecture", "cnn64", "--models", "1", "--learning-rate", "0.0002"])
    capsys.readouterr()
    main(["settings", "--project", str(root / "project_single.json"), "--advanced"])
    result = json.loads(capsys.readouterr().out)
    assert result["training"]["architecture"] == "cnn64"
    assert result["workload"]["total_optimizer_updates"] == 20000
    assert result["training"]["learning_rate"] == .0002


@pytest.mark.parametrize("rounds", [1, 3, 5])
def test_selected_positive_cache_path_produces_actual_n_times_rounds(tmp_path, rounds):
    from test_korean_library import wave, StubFrontend
    from korean_openwakeword.features import build_features, verify_cache
    project = create_project(tmp_path / "word", "보라야", "bora",
        positive_originals=3, augmentation_rounds=rounds, updates=4)
    root, config = load_project(project)
    config["augmentation"].update(clean_probability=1, moderate_probability=0, rir_probability=0)
    write_json(project, config)
    rows = []
    for i in range(3):
        audio = root / "data" / f"fixture_{i}.wav"; wave(audio, seed=i+1)
        rows.append({"sample_id": f"fixture_{i}", "source_id": f"synthetic-fixture:{i}",
            "path": str(audio.relative_to(root)), "sha256": digest(audio), "split": "train",
            "provenance": "qwen_tts", "human_reference_used": False, "text": "보라야",
            "voice_family_id": f"train_fixture_{i}", "call_start_s": 0, "call_end_s": 1})
    (root / config["manifests"]["positive_train"]).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    meta = build_features(project, "positive", smoke=True, frontend=StubFrontend())
    assert meta["original_count"] == 3 and meta["feature_rows"] == 3 * rounds
    assert verify_cache(root / config["pool_features"]["positive"])["feature_rows"] == 3 * rounds
    assert meta["smoke_test_only"] is True  # synthetic fixture is never listening approval
