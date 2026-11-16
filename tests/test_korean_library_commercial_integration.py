"""Commercial defaults and source routes stay separate from actual admissions."""
import pytest

from korean_openwakeword.commercial_sources import commercial_negative_plan
from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.project import BATCH, create_project, default_profile, load_project
from korean_openwakeword.studio import StudioManager
from korean_openwakeword.studio_steps import preset_settings


def test_studio_api_exposes_commercial_plan_and_preserves_smaller_presets(tmp_path):
    manager = StudioManager(tmp_path)
    recipes = {record["id"]: record for record in manager.recipes()["presets"]}
    production = recipes["production"]
    plan = commercial_negative_plan()
    assert production["training_negative_plan"] == plan
    assert (production["generic_hours"], production["korean_hours"]) == (3500, 1500)
    assert sum(row["target_hours"] for row in plan["components"]) == 5000
    assert plan["id"] == "commercial_negative_v3" and plan["version"] == 3
    assert [row["target_hours"] for row in plan["components"]] == [1500, 1500, 500, 1500]
    assert [row["sources"][0]["id"] for row in plan["components"]] == ["audioset_unbalanced", "mls", "fma_full", "yodas2"]
    assert (recipes["check"]["generic_hours"], recipes["check"]["korean_hours"]) == (2000, 1)
    assert (recipes["trial"]["generic_hours"], recipes["trial"]["korean_hours"]) == (2000, 2000 / 3)
    assert all(record["batch"] == BATCH for record in recipes.values())
    assert all("training_negative_plan" not in recipes[key] for key in ("check", "trial"))
    assert production["material_acquisition_policy"] == "automatic_v3"
    assert production["automatic_target_hours"] == {"generic_raw": 3500, "korean_train": 1500}
    assert recipes["trial"]["automatic_target_hours"] == {"generic_nominal": 2000, "korean_train": 2000 / 3}
    assert recipes["check"]["automatic_target_hours"] == {"generic_nominal": 2000, "korean_train": 1}
    assert not plan["content_approved"]


def test_loading_a_historical_project_does_not_apply_new_hours(tmp_path):
    project = create_project(tmp_path / "old", "보라야", "historical")
    old = read_json(project)
    old["targets"].update(generic_hours=7500, korean_hours=2500)
    write_json(project, old)
    before = project.read_bytes()
    loaded = load_project(project)[1]
    assert loaded["targets"]["generic_hours"] == 7500
    assert loaded["targets"]["korean_hours"] == 2500
    assert project.read_bytes() == before


def test_commercial_plan_defaults_do_not_change_sampling():
    profile = default_profile()
    assert profile["targets"]["generic_hours"] == 3500 and profile["targets"]["korean_hours"] == 1500
    assert profile["batch"] == preset_settings("production")["batch"] == BATCH
    plan = commercial_negative_plan(3500, 1500)
    assert [row["target_hours"] for row in plan["components"]] == [1500, 1500, 500, 1500]


def test_fma_full_preparation_counts_actual_long_track_and_preserves_original(tmp_path):
    import csv
    import numpy as np
    import soundfile as sf
    from korean_openwakeword.common import digest, read_rows
    from korean_openwakeword.preparation import create_source_index, prepare_public_audio

    project = create_project(tmp_path / "project", "보라야", "full_music_fixture")
    audio = tmp_path / "full_originals"
    audio.mkdir()
    original = audio / "000002.wav"
    sf.write(original, np.random.default_rng(42).normal(0, .04, 8000 * 45), 8000, subtype="PCM_16")
    original_sha = digest(original)
    tracks = tmp_path / "tracks.csv"
    with tracks.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([["", "artist", "track", "track", "set"],
            ["", "id", "license", "url", "split"], ["track_id", "", "", "", ""],
            [2, 17, "Attribution 4.0", "https://example.org/full/track2", "training"]])
    indexed = create_source_index(input_directory=audio, output=tmp_path / "full_index.jsonl",
        dataset="fma", dataset_id="fma_full_fixture", metadata=tracks)
    result = prepare_public_audio(project, input_directory=audio, output="data/full_music",
        source_index=indexed["source_index"]["path"], validation_fraction=0, target_hours=1/3600)
    sources = read_rows(result["artifacts"]["source_manifest"]["path"])
    assert len(sources) == 1 and sources[0]["duration_s"] == 45
    assert result["train_hours"] == pytest.approx(45/3600)
    assert sources[0]["canonical_source_id"] == "fma:track:2"
    assert sources[0]["license"] == "Attribution 4.0"
    assert digest(original) == original_sha
    assert result["training_admitted"] is False and result["human_listened_count"] == 0
    assert result["independent_STT_runs"] == 0
