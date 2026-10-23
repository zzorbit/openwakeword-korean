from importlib.resources import files

import pytest

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.project import create_project, load_project, require_standard_common_download


def test_new_projects_use_standard_download_without_changing_original_profile(tmp_path):
    resource = files("korean_openwakeword")/"resources/default_profile.json"
    before = digest(resource)
    project = create_project(tmp_path/"new", "보라야", "bora")
    assert "common_download_policy" not in read_json(project)
    assert "common_download_policy" not in read_json(resource)
    assert digest(resource) == before


def test_loading_legacy_project_does_not_add_transport_policy_or_rewrite_it(tmp_path):
    project = create_project(tmp_path/"old", "보라야", "bora")
    before = digest(project)
    _, loaded = load_project(project)
    assert "common_download_policy" not in loaded and digest(project) == before


def test_retired_project_remains_readable_without_rewriting_records(tmp_path):
    project = create_project(tmp_path/"retired", "보라야", "bora")
    config = read_json(project); config["common_download_policy"] = "selected_common_zip_v1"
    write_json(project, config)
    before = digest(project)
    _, loaded = load_project(project)
    assert loaded["common_download_policy"] == "selected_common_zip_v1"
    assert digest(project) == before
    with pytest.raises(ValueError, match="0.16.0"):
        require_standard_common_download(loaded["common_download_policy"])
    assert digest(project) == before


def test_standard_download_policy_needs_no_migration():
    assert require_standard_common_download(None) is None


def test_unknown_transport_policy_is_rejected(tmp_path):
    project = create_project(tmp_path/"invalid", "보라야", "bora")
    config = read_json(project); config["common_download_policy"] = "unverified"
    write_json(project, config)
    with pytest.raises(ValueError, match="download policy"):
        load_project(project)
