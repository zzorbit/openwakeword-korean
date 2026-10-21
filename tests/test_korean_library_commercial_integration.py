"""Commercial defaults and source routes stay separate from actual admissions."""
from copy import deepcopy

import pytest

from korean_openwakeword.cli import dispatch, parser
from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.project import BATCH, create_project, default_profile, load_project
from korean_openwakeword.studio import StudioManager
from korean_openwakeword.studio_steps import preset_settings


def test_studio_api_exposes_the_same_new_plan_as_cli_and_preserves_smaller_presets(tmp_path):
    manager = StudioManager(tmp_path)
    recipes = {record["id"]: record for record in manager.recipes()["presets"]}
    production = recipes["production"]
    plan = dispatch(parser().parse_args(["negative-plan"]))
    assert production["training_negative_plan"] == plan
    assert (production["generic_hours"], production["korean_hours"]) == (7000, 3000)
    assert sum(row["target_hours"] for row in plan["components"]) == 10000
    assert plan["id"] == "commercial_negative_v3" and plan["version"] == 3
    assert [row["target_hours"] for row in plan["components"]] == [3000, 3000, 1000, 3000]
    assert [row["sources"][0]["id"] for row in plan["components"]] == ["audioset_unbalanced", "mls", "fma_full", "yodas2"]
    assert (recipes["check"]["generic_hours"], recipes["check"]["korean_hours"]) == (2000, 1)
    assert (recipes["trial"]["generic_hours"], recipes["trial"]["korean_hours"]) == (2000, 2000 / 3)
    assert all(record["batch"] == BATCH for record in recipes.values())
    assert all("training_negative_plan" not in recipes[key] for key in ("check", "trial"))
    assert production["material_acquisition_policy"] == "automatic_v3"
    assert production["automatic_target_hours"] == {"generic_raw": 7000, "korean_train": 3000}
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


def test_source_cli_without_a_budget_only_resolves_metadata(monkeypatch):
    from korean_openwakeword import commercial_sources as sources
    captured = {}
    def resolve(source_id, *, selection):
        captured.update(source_id=source_id, selection=selection)
        return {"kind": "resolved", "prepared_hours": 0, "admitted_hours": 0}
    monkeypatch.setattr(sources, "resolve_commercial_source", resolve)
    monkeypatch.setattr(sources, "download_commercial_source", lambda *a, **k: pytest.fail("preview downloaded audio"))
    result = dispatch(parser().parse_args(["download-source", "--source", "yodas2", "--subset", "ko000", "--parts", "0"]))
    assert captured == {"source_id": "yodas2", "selection": {"parts": [0], "subset": "ko000"}}
    assert result["prepared_hours"] == result["admitted_hours"] == 0


def test_source_cli_requires_a_budget_and_passes_only_the_selected_parts(monkeypatch, tmp_path):
    from korean_openwakeword import commercial_sources as sources
    monkeypatch.setattr(sources, "resolve_commercial_source", lambda *a, **k: pytest.fail("invalid input reached network"))
    with pytest.raises(ValueError, match="max-download-gib"):
        dispatch(parser().parse_args(["download-source", "--source", "mls", "--parts", "1_2_000", "--language", "german", "--output", str(tmp_path)]))
    record = {"kind": "selected"}
    monkeypatch.setattr(sources, "resolve_commercial_source", lambda *a, **k: deepcopy(record))
    def download(resolved, output, *, max_download_bytes):
        assert resolved == record and output == tmp_path
        assert max_download_bytes == 2 * 1024**3
        return {"state": "downloaded", "content_approved": False}
    monkeypatch.setattr(sources, "download_commercial_source", download)
    result = dispatch(parser().parse_args(["download-source", "--source", "mls", "--parts", "1_2_000", "--language", "german", "--output", str(tmp_path), "--max-download-gib", "2"]))
    assert not result["content_approved"]


@pytest.mark.parametrize("options", [["--metadata-only"], ["--max-download-gib", "nan"],
    ["--max-download-gib", "inf"], ["--max-download-gib", "1e308"], ["--max-download-gib", "0"]])
def test_invalid_source_download_budget_is_rejected_before_network(monkeypatch, tmp_path, options):
    from korean_openwakeword import commercial_sources as sources
    monkeypatch.setattr(sources, "resolve_commercial_source", lambda *a, **k: pytest.fail("invalid input reached network"))
    with pytest.raises(ValueError, match="max-download-gib"):
        dispatch(parser().parse_args(["download-source", "--source", "fma_large",
                                    "--output", str(tmp_path), *options]))


def test_cli_default_targets_and_new_negative_plan_do_not_change_sampling():
    profile = default_profile()
    assert profile["targets"]["generic_hours"] == 7000 and profile["targets"]["korean_hours"] == 3000
    assert profile["batch"] == preset_settings("production")["batch"] == BATCH
    plan = dispatch(parser().parse_args(["negative-plan", "--generic-hours", "3500", "--korean-hours", "1500"]))
    assert [row["target_hours"] for row in plan["components"]] == [1500, 1500, 500, 1500]


@pytest.mark.parametrize("source_id,expected_part", [("fma_large", "large"), ("fma_full", "full")])
def test_fma_cli_selects_the_requested_original_variant_without_a_download(monkeypatch, source_id, expected_part):
    from korean_openwakeword import commercial_sources as sources
    captured = {}
    def resolve(source_id, *, selection):
        captured.update(source_id=source_id, selection=selection)
        return {"kind": "preview", "content_approved": False}
    monkeypatch.setattr(sources, "resolve_commercial_source", resolve)
    monkeypatch.setattr(sources, "download_commercial_source", lambda *a, **k: pytest.fail("preview downloaded audio"))
    result = dispatch(parser().parse_args(["download-source", "--source", source_id,
        *(["--whole-archive"] if source_id == "fma_full" else [])]))
    assert captured == {"source_id": source_id, "selection": {"parts": [expected_part]}}
    assert not result["content_approved"]


def test_fma_full_cli_uses_real_download_route_with_the_explicit_budget(monkeypatch, tmp_path):
    from korean_openwakeword import commercial_sources as sources
    record = {"source_id": "fma_full", "selection": {"parts": ["full"]}}
    def resolve(source_id, *, selection):
        assert source_id == record["source_id"] and selection == record["selection"]
        return record
    def download(resolved, output, *, max_download_bytes):
        assert resolved == record and output == tmp_path
        assert max_download_bytes == 900 * 1024**3
        return {"state": "downloaded", "content_approved": False}
    monkeypatch.setattr(sources, "resolve_commercial_source", resolve)
    monkeypatch.setattr(sources, "download_commercial_source", download)
    result = dispatch(parser().parse_args(["download-source", "--source", "fma_full", "--parts", "full",
        "--whole-archive", "--output", str(tmp_path), "--max-download-gib", "900"]))
    assert result["state"] == "downloaded" and not result["content_approved"]


def test_fma_full_cli_defaults_to_selected_tracks_not_the_whole_archive(monkeypatch, tmp_path):
    from korean_openwakeword import commercial_sources, fma_download
    monkeypatch.setattr(commercial_sources, "resolve_commercial_source", lambda *a, **k: pytest.fail("whole archive resolver called"))
    seen = {}
    def download(output, **kwargs):
        seen.update(output=output, **kwargs)
        return {"mode": "selective_original_mp3", "content_approved": False}
    monkeypatch.setattr(fma_download, "download_fma", download)
    preview = dispatch(parser().parse_args(["download-source", "--source", "fma_full"]))
    assert preview["download_mode"] == "selective" and not seen
    result = dispatch(parser().parse_args(["download-source", "--source", "fma_full", "--output", str(tmp_path),
        "--max-download-gib", "150"]))
    assert seen["max_download_bytes"] == 150 * 1024**3 and seen["hours"] == 1000
    assert result["mode"] == "selective_original_mp3"


def test_selective_fma_cli_preserves_plan_only_and_policy_choices(monkeypatch, tmp_path):
    from korean_openwakeword import fma_download
    captured = {}
    def download(output, **kwargs):
        captured.update(output=output, **kwargs)
        return {"state": "planned", "audio_download_started": False}
    monkeypatch.setattr(fma_download, "download_fma", download)
    result = dispatch(parser().parse_args(["download-fma", "--output", str(tmp_path), "--hours", "0.1",
        "--plan-only", "--license-policy", "commercial-only"]))
    assert captured["hours"] == .1 and captured["plan_only"] is True
    assert captured["max_download_bytes"] is None and captured["license_policy"] == "commercial-only"
    assert result["audio_download_started"] is False


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
