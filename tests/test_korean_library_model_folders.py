"""Automatic web model storage, preserving registered and unknown prior data."""
from pathlib import Path
import re
import shutil

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import read_json, write_json


@pytest.fixture
def manager(tmp_path):
    value = studio.StudioManager(tmp_path / "workspace")
    yield value
    value.close()


def create(manager, **payload):
    return manager.create({"wakeword": "보라야", "preset": "check", **payload})


def test_web_uses_word_named_model_folder_without_callers_path(manager):
    view = create(manager)
    root = manager.workspace / "models" / "보라야"
    assert Path(view["directory"]) == root
    assert Path(manager._load(view["id"])["project"]) == root / "project.json"
    assert read_json(root / "project.json")["wakeword"] == "보라야"
    assert manager.recipes()["default_directory"] == str(manager.workspace / "models")
    assert not (manager.workspace / "projects").exists()


def test_same_choices_reuse_saved_state_without_rewriting_or_creating_work(manager):
    first = create(manager)
    state = manager._load(first["id"])
    state.update(status="completed", index=len(state["plan"]))
    manager._save(state)
    root = Path(state["project"]).parent
    marker = root / "models" / "preserved.onnx"
    marker.write_bytes(b"existing model")
    write_json(root / "final_lock.json", {"state": "frozen"})
    before = {p: p.read_bytes() for p in (Path(state["project"]), manager._path(first["id"]), marker)}
    reopened = create(manager, wakeword=" 보라야 ", advanced={"generic_hours": 2000.0})
    assert reopened["id"] == first["id"] and reopened["status"] == "completed"
    assert len(manager.registry["projects"]) == 1
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert manager._children == {}


def test_incomplete_registered_final_locked_run_cannot_be_resumed_by_create(manager):
    first = create(manager)
    state = manager._load(first["id"])
    root = Path(state["project"]).parent
    write_json(root / "final_lock.json", {"state": "frozen"})
    paths = (Path(state["project"]), manager._path(first["id"]), manager.registry_path)
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError, match="최종 평가로 잠긴"):
        create(manager)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert manager._children == {}


@pytest.mark.parametrize("advanced", [
    {"positive_originals": 2000}, {"augmentation_rounds": 3}, {"updates": 20000},
    {"generic_hours": 2001}, {"korean_hours": 2},
])
def test_each_changed_budget_creates_stable_separate_folder(manager, advanced):
    original = create(manager)
    old = Path(manager._load(original["id"])["project"])
    contents = old.read_bytes()
    changed = create(manager, advanced=advanced)
    assert re.fullmatch(r"보라야_[0-9a-f]{12}", Path(changed["directory"]).name)
    assert changed["id"] != original["id"]
    assert create(manager, advanced=advanced)["id"] == changed["id"]
    assert old.read_bytes() == contents


def test_different_preset_gets_a_separate_run_and_identical_choices_resume(manager):
    check = create(manager)
    trial = create(manager, preset="trial")
    assert check["id"] != trial["id"]
    assert Path(trial["directory"]).name.startswith("보라야_")
    assert create(manager, preset="trial")["id"] == trial["id"]


def test_registered_legacy_location_and_frozen_policy_are_resumed_in_place(manager):
    view = create(manager)
    state = manager._load(view["id"])
    original = Path(state["project"]).parent
    legacy = manager.workspace / "projects" / "bora"
    legacy.parent.mkdir()
    shutil.move(str(original), str(legacy))
    state["project"] = str(legacy / "project.json")
    state["settings"].pop("official_noise_acquisition_policy")
    state["settings"]["evaluation_backend"] = "cpu"
    manager.registry["projects"][view["id"]] = state["project"]
    write_json(manager.registry_path, manager.registry)
    manager._save(state)
    evidence = manager._path(view["id"]).read_bytes()
    config = Path(state["project"]).read_bytes()
    reopened = create(manager)
    assert reopened["id"] == view["id"]
    assert Path(reopened["directory"]) == legacy
    assert manager._path(view["id"]).read_bytes() == evidence
    assert Path(state["project"]).read_bytes() == config
    assert not original.exists()


def test_unknown_nonempty_base_is_preserved_and_not_adopted(manager):
    root = manager.workspace / "models" / "보라야"
    root.mkdir(parents=True)
    marker = root / "important.bin"
    marker.write_bytes(b"unregistered data")
    created = create(manager)
    assert Path(created["directory"]) != root
    assert marker.read_bytes() == b"unregistered data"
    assert not (root / "project.json").exists()


def test_unknown_base_and_variant_are_not_overwritten_or_claimed(manager):
    from korean_openwakeword.studio_steps import preset_settings
    root = manager.workspace / "models" / "보라야"
    variant = root.with_name(root.name + "_" + studio.identity(
        studio._model_choice("보라야", "check", preset_settings("check")))[:12])
    for folder in (root, variant):
        folder.mkdir(parents=True)
        (folder / "important.bin").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="등록되지 않은"):
        create(manager)
    assert not manager.registry["projects"]
    assert all((folder / "important.bin").read_bytes() == b"preserve" for folder in (root, variant))


@pytest.mark.parametrize("directory", ["F:\\custom", 123, [], {}, Path("elsewhere")])
def test_nonempty_directory_payload_is_rejected_before_model_creation(manager, directory):
    with pytest.raises(ValueError, match="자동"):
        create(manager, directory=directory)
    assert not (manager.workspace / "models").exists()
    assert manager.registry["projects"] == {}


@pytest.mark.parametrize("word", ["", " ", "../보라야", "..\\보라야", "보라야/다음아", "CON", "보라야.", "보라:야"])
def test_invalid_names_cannot_become_paths(manager, word):
    with pytest.raises(ValueError, match="한글"):
        create(manager, wakeword=word)
    assert not (manager.workspace / "models").exists()
    assert manager.registry["projects"] == {}


@pytest.mark.parametrize("key,value", [("updates", 0), ("augmentation_rounds", 2), ("augmentation_rounds", True),
    ("generic_hours", float("nan")), ("korean_hours", -1)])
def test_invalid_workloads_do_not_create_any_model_folder(manager, key, value):
    with pytest.raises(ValueError):
        create(manager, advanced={key: value})
    assert not (manager.workspace / "models").exists()
    assert manager.registry["projects"] == {}


def test_invalid_boolean_rounds_do_not_reuse_registered_run_or_change_active_id(manager):
    first = create(manager)
    second = create(manager, wakeword="다음아")
    before = manager.registry_path.read_bytes()
    with pytest.raises(ValueError, match="정수"):
        create(manager, advanced={"augmentation_rounds": True})
    assert manager.registry_path.read_bytes() == before
    assert manager.registry["active_project_id"] == second["id"]
    assert len(manager.registry["projects"]) == 2


@pytest.mark.parametrize("inside", [False, True])
def test_models_root_redirects_are_rejected_even_inside_workspace(manager, tmp_path, inside):
    destination = manager.workspace / "other-data" if inside else tmp_path / "outside"
    destination.mkdir()
    try:
        (manager.workspace / "models").symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable in this environment")
    with pytest.raises(ValueError, match="다른 위치"):
        create(manager)
    assert list(destination.iterdir()) == []


def test_very_long_valid_call_uses_short_stable_name_without_losing_original_word(manager):
    word = "가" * 300
    created = create(manager, wakeword=word)
    root = Path(created["directory"])
    assert len(root.name) == 77
    assert read_json(root / "project.json")["wakeword"] == word
    assert create(manager, wakeword=word)["id"] == created["id"]
