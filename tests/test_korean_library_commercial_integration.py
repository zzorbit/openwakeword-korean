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
    assert [row["target_hours"] for row in plan["components"]] == [3200, 2600, 900, 300, 3000]
    assert (recipes["check"]["generic_hours"], recipes["check"]["korean_hours"]) == (2000, 1)
    assert (recipes["trial"]["generic_hours"], recipes["trial"]["korean_hours"]) == (2000, 2000 / 3)
    assert all(record["batch"] == BATCH for record in recipes.values())
    assert all("training_negative_plan" not in recipes[key] for key in ("check", "trial"))
    assert production["automatic_target_hours"] == {"generic_nominal": 2000, "korean_train": 1}
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
    assert [row["target_hours"] for row in plan["components"]] == [1600, 1300, 450, 150, 1500]
