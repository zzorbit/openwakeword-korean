"""Download rows remain visible while the same stage extracts and converts."""
from types import SimpleNamespace

from korean_openwakeword import official_music, studio_data, studio_steps
from korean_openwakeword.common import identity, read_json, write_json
from korean_openwakeword.project import create_project
from korean_openwakeword.studio import StudioManager


def test_stage_retains_all_download_rows_during_processing_without_changing_inputs(tmp_path, monkeypatch):
    settings = studio_steps.preset_settings("check")
    project = create_project(tmp_path / "model", "보라야", "bora",
                             positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
                             updates=settings["updates"], generic_hours=settings["generic_hours"],
                             korean_hours=settings["korean_hours"])
    before = {path: path.read_bytes() for path in (project, project.parent / "approval.json")}

    def prepare(*args, progress, **kwargs):
        progress({"phase": "download_many", "files": [
            {"id": "train", "state": "waiting", "done": 0, "total": 10},
            {"id": "evaluation", "state": "waiting", "done": 0, "total": 20}]})
        progress({"phase": "download_many", "files": [
            {"id": "train", "state": "reused", "done": 10, "total": 10},
            {"id": "evaluation", "state": "complete", "done": 20, "total": 20}]})
        progress({"phase": "extract", "done": 3, "total": 9, "unit": "files"})
        return {"review_queues": [], "source_bytes_unchanged": True}

    monkeypatch.setattr(studio_data, "prepare_materials", prepare)
    result = studio_steps.run_stage(project, tmp_path, settings, "data_download")
    record = read_json(project.parent / ".studio/progress.json")
    assert result["state"] == "completed" and result["result"]["source_bytes_unchanged"]
    assert record["phase"] == "extract" and record["done"] == 3
    assert [row["id"] for row in record["files"]] == ["train", "evaluation"]
    assert [row["state"] for row in record["files"]] == ["reused", "complete"]
    assert "bytes_per_second" not in record
    assert all(path.read_bytes() == data for path, data in before.items())


def test_paused_download_uses_current_hours_target_before_worker_reports_again():
    row = {"id": "automatic:yodas2", "label": "한국어 웹 음성", "group_label": "한국어 웹 음성",
           "unit": "hours", "total": 500}
    manager = SimpleNamespace(_history_download_files=lambda *args, **kwargs: [row],
                              _saved_automatic_download_hours=lambda state: {})
    saved = {**row, "done": 7, "total": 2000 / 3, "target_hours": 2000 / 3, "state": "downloading"}
    record = {"stage": "data_download", "files": [saved]}

    result, = StudioManager._live_download_files(manager, {}, record, running=False)

    assert (result["done"], result["total"], result["target_hours"]) == (7, 500, 500)
    assert saved["total"] == 2000 / 3


def test_missing_download_snapshot_restores_fixed_caches_and_saved_korean_hours(tmp_path, monkeypatch):
    manager = StudioManager(tmp_path)
    state = {"project": str(tmp_path / "model/project.json"), "preset": "trial", "settings": studio_steps.preset_settings("trial")}
    shared = tmp_path / "shared"
    track = {"name": "fma_small/000/000001.mp3", "size_bytes": 7, "compressed_bytes": 5}
    catalog = {"archives": {"music": {"members": [track], "index_bytes": 1, "size_bytes": 10}}}
    monkeypatch.setattr(official_music, "official_music_catalog", lambda: catalog)
    music = official_music._destination(shared, "music", track)
    music.parent.mkdir(parents=True)
    music.write_bytes(b"1234567")
    receipt = {"kind": official_music.POLICY, "request": {"policy": official_music.POLICY, "catalog_sha256": identity(catalog)},
               "completed_at": "completed", "files": [{"path": str(music), **track}]}
    write_json(shared / "official_music_selective_v1" / ("receipt_" + identity(receipt) + ".json"), receipt)
    manager._history_download_files(state, {}, planned=True)
    next(iter(manager._history_download_plans.values()))["additional_download"]["files"] = 1
    rir = shared / "mit_rir/16khz/one.wav"
    rir.parent.mkdir(parents=True)
    rir.write_bytes(b"rir")
    write_json(shared / "mit_rir/provider_tree.json", {"source_url": "https://huggingface.co/api/datasets/davidscripka/"
               f"MIT_environmental_impulse_responses/tree/{studio_data.MIT_REV}/16khz?limit=1000",
               "entries": [{"type": "file", "path": "16khz/one.wav", "size": 3, "lfs": {"oid": "rir-pin"}}]})
    write_json(rir.with_name("one.wav.receipt.json"), {"completed_at": "completed", "bytes": 3, "sha256": "rir-pin"})
    write_json(tmp_path / "model/.studio/materials/automatic_sources_v3/progress.json",
               {"component_actual_hours": {"yodas2": 60.85}})

    def visible(record):
        return {row["id"]: row for row in manager._live_download_files(state, record, running=True)}

    rows = visible({})
    assert (rows["official_music"]["done"], rows["mit_rir"]["done"], rows["automatic:yodas2"]["done"]) == (5, 3, 60.85)
    assert all(rows[key]["state"] == "waiting" for key in ("official_music", "mit_rir", "automatic:yodas2"))
    assert all(rows[key]["cache_candidate"] for key in ("official_music", "mit_rir"))
    assert rows["automatic:yodas2"]["total"] == 500
    assert rows["automatic:yodas2"]["estimated_download_bytes"] == 70_000_000_000

    latest = {"stage": "data_download", "files": [
        {"id": "official_music", "done": 0, "state": "failed"},
        {"id": "automatic:yodas2", "done": 0, "state": "waiting", "deferred": True},
    ]}
    rows = visible(latest)
    assert rows["official_music"]["state"] == "failed" and rows["official_music"]["done"] == 0
    assert rows["automatic:yodas2"]["done"] == 60.85
    music.unlink()
    rir.unlink()
    rows = visible({})
    assert rows["official_music"]["done"] == rows["mit_rir"]["done"] == 0
    state["settings"]["korean_hours"] = 250
    assert visible({})["automatic:yodas2"]["estimated_download_bytes"] == 35_000_000_000


def test_download_poll_retains_file_rows_when_material_snapshot_has_only_a_phase(tmp_path):
    manager = StudioManager(tmp_path)
    state = {"id": "trial", "project": str(tmp_path / "model/project.json"), "status": "ready"}
    progress_path = tmp_path / "model/.studio/progress.json"
    rows = [{"id": "official_music", "state": "reused", "done": 10, "total": 10}]
    write_json(progress_path, {"stage": "data_download", "files": rows})
    assert manager._describe_progress(state, {"id": "data_download"}, {}, {})[1]["files"] == rows
    write_json(progress_path, {"stage": "data_download", "phase": "extract"})
    latest = manager._describe_progress(state, {"id": "data_download"}, {}, {})[1]
    assert latest["files"] == rows and latest["phase"] == "extract"
    progress_path.unlink()
    assert manager._describe_progress(state, {"id": "data_download"}, {}, {})[1]["files"] == rows
    state["last_job"] = {"task_id": "another-task"}
    assert manager._describe_progress(state, {"id": "data_download"}, {}, {})[1] == {}
