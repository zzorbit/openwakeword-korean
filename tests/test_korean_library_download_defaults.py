from importlib.resources import files

import pytest

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.project import create_project, load_project


def test_new_projects_pin_selective_transport_without_changing_original_profile(tmp_path):
    resource = files("korean_openwakeword")/"resources/default_profile.json"
    before = digest(resource)
    project = create_project(tmp_path/"new", "보라야", "bora")
    assert read_json(project)["common_download_policy"] == "selected_common_zip_v1"
    assert "common_download_policy" not in read_json(resource)
    assert digest(resource) == before


def test_loading_legacy_project_does_not_add_transport_policy_or_rewrite_it(tmp_path):
    project = create_project(tmp_path/"old", "보라야", "bora")
    config = read_json(project); config.pop("common_download_policy")
    write_json(project, config)
    before = digest(project)
    _, loaded = load_project(project)
    assert "common_download_policy" not in loaded and digest(project) == before


def test_unknown_transport_policy_is_rejected(tmp_path):
    project = create_project(tmp_path/"invalid", "보라야", "bora")
    config = read_json(project); config["common_download_policy"] = "unverified"
    write_json(project, config)
    with pytest.raises(ValueError, match="download policy"):
        load_project(project)
