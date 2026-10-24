"""Download rows remain visible while the same stage extracts and converts."""
from korean_openwakeword import studio_data, studio_steps
from korean_openwakeword.common import read_json
from korean_openwakeword.project import create_project


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
