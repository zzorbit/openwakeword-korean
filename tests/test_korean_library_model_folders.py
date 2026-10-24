"""Automatic web model storage, preserving registered and unknown prior data."""
from pathlib import Path
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


def test_web_uses_library_numbered_folder_without_callers_path(manager):
    view = create(manager)
    root = manager.workspace / "models" / "model_0001"
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
    for method in (manager.create, manager.model_location):
        with pytest.raises(ValueError, match="최종 평가로 잠긴"):
            method({"wakeword": "보라야", "preset": "check"})
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert manager._children == {}


@pytest.mark.parametrize("advanced", [
    {"positive_originals": 2000}, {"augmentation_rounds": 3}, {"updates": 20000},
    {"generic_hours": 2001}, {"korean_hours": 2},
])
def test_each_changed_budget_creates_separate_numbered_folder(manager, advanced):
    original = create(manager)
    old = Path(manager._load(original["id"])["project"])
    contents = old.read_bytes()
    changed = create(manager, advanced=advanced)
    assert Path(changed["directory"]).name == "model_0002"
    assert changed["id"] != original["id"]
    assert create(manager, advanced=advanced)["id"] == changed["id"]
    assert old.read_bytes() == contents


def test_different_preset_gets_a_separate_run_and_identical_choices_resume(manager):
    check = create(manager)
    trial = create(manager, preset="trial")
    assert check["id"] != trial["id"]
    assert Path(trial["directory"]).name == "model_0002"
    assert create(manager, preset="trial")["id"] == trial["id"]


@pytest.mark.parametrize("relative", ["projects/bora", "models/보라야"])
def test_registered_legacy_location_and_frozen_policy_are_resumed_in_place(manager, relative):
    view = create(manager)
    state = manager._load(view["id"])
    original = Path(state["project"]).parent
    legacy = manager.workspace / relative
    legacy.parent.mkdir(exist_ok=True)
    shutil.move(str(original), str(legacy))
    state["project"] = str(legacy / "project.json")
    state["settings"].pop("official_noise_acquisition_policy")
    state["settings"]["evaluation_backend"] = "cpu"
    manager.registry["projects"][view["id"]] = state["project"]
    write_json(manager.registry_path, manager.registry)
    manager._save(state)
    evidence = manager._path(view["id"]).read_bytes()
    config = Path(state["project"]).read_bytes()
    registry = manager.registry_path.read_bytes()
    for _ in range(3):
        assert Path(manager.model_location({"wakeword": "보라야", "preset": "check"})["directory"]) == legacy
    assert manager.registry_path.read_bytes() == registry
    assert manager._path(view["id"]).read_bytes() == evidence
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


def test_sequence_counts_existing_empty_directories_files_and_gaps(manager):
    models = manager.workspace / "models"
    (models / "model_0001").mkdir(parents=True)
    (models / "model_0004").write_bytes(b"unknown file")
    (models / "model_0008").mkdir()
    (models / "model_0008" / "important.bin").write_bytes(b"preserve")
    (models / "model_other").mkdir()
    (models / "보라야").mkdir()
    preview = manager.model_location({"wakeword": "보라야", "preset": "check"})
    assert Path(preview["directory"]) == models / "model_0009"
    assert Path(create(manager)["directory"]) == Path(preview["directory"])
    assert list((models / "model_0001").iterdir()) == []
    assert (models / "model_0004").read_bytes() == b"unknown file"
    assert (models / "model_0008" / "important.bin").read_bytes() == b"preserve"
    assert list((models / "보라야").iterdir()) == []


def test_sequence_grows_past_four_digits_without_reusing_gaps(manager):
    models = manager.workspace / "models"
    models.mkdir()
    (models / "model_9999").write_bytes(b"keep")
    assert Path(create(manager)["directory"]) == models / "model_10000"


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_sequence_skips_uppercase_existing_names_on_all_platforms(manager, kind):
    models = manager.workspace / "models"
    models.mkdir()
    occupied = models / "MODEL_0007"
    if kind == "file":
        occupied.write_bytes(b"preserve")
    else:
        occupied.mkdir()
    preview = manager.model_location({"wakeword": "보라야", "preset": "check"})
    assert Path(preview["directory"]) == models / "model_0008"
    assert Path(create(manager)["directory"]) == models / "model_0008"
    assert occupied.is_dir() if kind == "directory" else occupied.read_bytes() == b"preserve"


def _workspace_snapshot(manager):
    return {str(path.relative_to(manager.workspace)): path.read_bytes() if path.is_file() else None
            for path in manager.workspace.rglob("*")}


def test_repeated_preview_never_creates_or_reserves_any_folder_or_state(manager):
    before = _workspace_snapshot(manager)
    payload = {"wakeword": "보라야", "preset": "check"}
    previews = [manager.model_location(payload) for _ in range(4)]
    assert all(item == {"directory": str(manager.workspace / "models" / "model_0001")} for item in previews)
    assert _workspace_snapshot(manager) == before
    assert manager.registry == {"projects": {}, "active_project_id": None}
    assert Path(create(manager)["directory"]) == Path(previews[0]["directory"])


def test_existing_run_preview_does_not_switch_active_run_or_rewrite_saved_records(manager):
    first = create(manager)
    second = create(manager, wakeword="다음아")
    before = _workspace_snapshot(manager)
    preview = manager.model_location({"wakeword": "보라야", "preset": "check"})
    assert preview["directory"] == first["directory"]
    assert manager.registry["active_project_id"] == second["id"]
    assert _workspace_snapshot(manager) == before


def test_preview_and_create_resolve_same_path_after_budget_changes(manager):
    create(manager)
    for advanced in ({"updates": 20000}, {"generic_hours": 2100}, {"positive_originals": 2000}):
        payload = {"wakeword": "보라야", "preset": "check", "advanced": advanced}
        preview = manager.model_location(payload)
        assert manager.create(payload)["directory"] == preview["directory"]


def test_creation_refuses_even_empty_folder_that_appears_after_resolution(manager, monkeypatch):
    original = manager._model_location
    def external_collision(*args):
        target, saved = original(*args)
        target.mkdir(parents=True)
        return target, saved
    monkeypatch.setattr(manager, "_model_location", external_collision)
    with pytest.raises(FileExistsError):
        create(manager)
    assert manager.registry["projects"] == {}
    assert list((manager.workspace / "models" / "model_0001").iterdir()) == []


@pytest.mark.parametrize("directory", ["F:\\custom", 123, [], {}, Path("elsewhere")])
def test_nonempty_directory_payload_is_rejected_before_model_creation(manager, directory):
    for method in (manager.create, manager.model_location):
        with pytest.raises(ValueError, match="자동"):
            method({"wakeword": "보라야", "preset": "check", "directory": directory})
    assert not (manager.workspace / "models").exists()
    assert manager.registry["projects"] == {}


@pytest.mark.parametrize("word", ["", " ", "../보라야", "..\\보라야", "보라야/다음아", "CON", "보라야.", "보라:야"])
def test_invalid_names_cannot_become_paths(manager, word):
    for method in (manager.create, manager.model_location):
        with pytest.raises(ValueError, match="한글"):
            method({"wakeword": word, "preset": "check"})
    assert not (manager.workspace / "models").exists()
    assert manager.registry["projects"] == {}


@pytest.mark.parametrize("key,value", [("updates", 0), ("augmentation_rounds", 2), ("augmentation_rounds", True),
    ("generic_hours", float("nan")), ("korean_hours", -1)])
def test_invalid_workloads_do_not_create_any_model_folder(manager, key, value):
    for method in (manager.create, manager.model_location):
        with pytest.raises(ValueError):
            method({"wakeword": "보라야", "preset": "check", "advanced": {key: value}})
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
    for method in (manager.create, manager.model_location):
        with pytest.raises(ValueError, match="다른 위치"):
            method({"wakeword": "보라야", "preset": "check"})
    assert list(destination.iterdir()) == []


def test_numbered_child_redirect_is_skipped_without_using_its_target(manager, tmp_path):
    models = manager.workspace / "models"
    models.mkdir()
    destination = tmp_path / "outside"
    destination.mkdir()
    try:
        (models / "model_0001").symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable in this environment")
    preview = manager.model_location({"wakeword": "보라야", "preset": "check"})
    assert Path(preview["directory"]) == models / "model_0002"
    assert Path(create(manager)["directory"]) == models / "model_0002"
    assert list(destination.iterdir()) == []


def test_very_long_valid_call_never_enters_folder_name_or_loses_original_word(manager):
    word = "가" * 300
    created = create(manager, wakeword=word)
    root = Path(created["directory"])
    assert root.name == "model_0001"
    assert read_json(root / "project.json")["wakeword"] == word
    assert create(manager, wakeword=word)["id"] == created["id"]
