"""Exact source audits and bounded journals for large fixed-window manifests."""
import json
from pathlib import Path
import sqlite3
import tracemalloc

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import data, features, streaming
from korean_openwakeword.common import digest, iter_rows, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.storage import storage_paths
from korean_openwakeword.streaming import manifest_index


def public_project(tmp_path):
    project = create_project(tmp_path / "project", "보라야", "bora", positive_originals=3)
    root, config = load_project(project)
    audio = root / "data" / "source.wav"
    sf.write(audio, .1 * np.sin(np.arange(32000) / 25), 16000, subtype="PCM_16")
    row = dict(sample_id="source", source_id="fixture:source", canonical_recording_id="fixture:parent",
               path=str(audio.relative_to(root)), sha256=digest(audio), provenance="public_dataset",
               license="synthetic-test-fixture", source_url="https://example.org/fixture",
               split="train", category="environment", speech=False)
    return project, root, config, row


def write_windows(path, row, count):
    with path.open("w", encoding="utf-8") as stream:
        for number in range(count):
            stream.write(json.dumps(dict(row, sample_id=f"window_{number:07d}",
                             start_s=0, end_s=2, parent_start_s=number * 2)) + "\n")


def test_iter_rows_preserves_bom_blank_lines_and_order_without_read_ahead(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('\ufeff\n\u00a0\n{"id":1}\n\n{"id":2}\n', encoding="utf-8")
    iterator = iter_rows(path)
    assert next(iterator) == {"id": 1}
    assert next(iterator) == {"id": 2}
    assert list(iterator) == []
    assert read_rows(path) == [{"id": 1}, {"id": 2}]
    assert list(iter_rows(tmp_path / "absent.jsonl")) == []


def test_audit_uses_disk_uniqueness_and_exact_parent_union(tmp_path, monkeypatch):
    project, root, config, row = public_project(tmp_path)
    rows = [dict(row, sample_id=str(i), parent_start_s=start) for i, start in enumerate((3, 0, 1, 8))]
    manifest = root / config["manifests"]["korean_negative"]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    original_info, calls = data.sf.info, []

    def measured_info(path):
        calls.append(path)
        return original_info(path)

    monkeypatch.setattr(data.sf, "info", measured_info)
    monkeypatch.setattr(data, "read_rows", lambda *args: pytest.fail("Audit cannot load window lists"))
    result = data.audit_project(project)
    assert result["passed"]
    assert result["pools"]["korean_negative"] == {
        "rows": 4, "unique_source_hours": 7 / 3600, "final_audio_opened": False}
    assert len(calls) == 1
    manifest.write_text(manifest.read_text(encoding="utf-8") + json.dumps(rows[0]) + "\n", encoding="utf-8")
    result = data.audit_project(project)
    assert result["errors"] == ["korean_negative: missing/duplicate sample ID 0"]


def test_plural_speakers_and_component_sources_cannot_cross_splits(tmp_path):
    project, root, config, row = public_project(tmp_path)
    for pool, split, speakers in (("noise", "train", ["shared", "train_only"]),
                                  ("noise_validation", "validation", ["shared", "validation_only"])):
        entry = dict(row, sample_id=split, source_id="source:" + split, sha256="hash:" + split,
                     canonical_recording_id="recording:" + split, split=split,
                     canonical_speaker_ids=speakers, component_source_ids=["shared_component"])
        (root / config["manifests"][pool]).write_text(json.dumps(entry) + "\n", encoding="utf-8")
    result = data.audit_project(project, check_files=False)
    assert result["leaks"] == [
        {"key": "source", "id": "shared_component", "splits": ["train", "validation"]},
        {"key": "speaker", "id": "shared", "splits": ["train", "validation"]}]


def test_validation_accepts_reopenable_or_one_shot_iterators_without_losing_approval(tmp_path):
    _, root, config, row = public_project(tmp_path)
    approval = root / "review" / "approved.json"
    write_json(approval, dict(state="accepted", content_review="accepted", accepted_source_sha256=[row["sha256"]]))
    entry = dict(row, admission={"path": str(approval.relative_to(root)), "sha256": digest(approval)})
    assert data.validate_admission(root, (dict(entry, sample_id=str(i)) for i in range(2000)), config, "korean_negative") is None
    with pytest.raises(ValueError, match="Empty korean_negative"):
        data.validate_admission(root, iter(()), config, "korean_negative")
    write_json(approval, dict(state="accepted", content_review="accepted", accepted_source_sha256=[]))
    entry["admission"]["sha256"] = digest(approval)
    with pytest.raises(ValueError, match="does not cover"):
        data.validate_admission(root, iter([entry]), config, "korean_negative")


@pytest.mark.parametrize("raise_error", [False, True])
def test_temp_index_uses_selected_cache_and_removes_only_its_own_directory(tmp_path, monkeypatch, raise_error):
    selected = tmp_path / "selected_cache"
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(selected))
    protected = selected / "tmp" / "do_not_remove.txt"
    protected.parent.mkdir(parents=True)
    protected.write_text("other work", encoding="utf-8")
    owned = None
    try:
        with manifest_index(tmp_path / "project") as index:
            owned = Path(index.db.execute("PRAGMA database_list").fetchone()[2]).parent
            assert owned.parent == storage_paths(tmp_path / "project")["temp"]
            assert index.db.execute("PRAGMA cache_size").fetchone()[0] == -4096
            assert index.db.execute("PRAGMA temp_store").fetchone()[0] == 1
            if raise_error:
                raise RuntimeError("interrupted audit")
    except RuntimeError:
        assert raise_error
    assert owned is not None and not owned.exists()
    assert protected.read_text(encoding="utf-8") == "other work"


def test_failed_database_initialization_closes_and_removes_owned_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / "cache"))
    closed = []
    original_connect = sqlite3.connect

    class FailedConnection(sqlite3.Connection):
        def executescript(self, script):
            raise sqlite3.OperationalError("fixture initialization failure")
        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setattr(streaming.sqlite3, "connect", lambda *args, **kwargs:
                        original_connect(*args, factory=FailedConnection, **kwargs))
    with pytest.raises(sqlite3.OperationalError, match="initialization failure"):
        with manifest_index(tmp_path / "project"):
            pytest.fail("A failed database cannot yield a usable index")
    assert closed == [True]
    assert not list(storage_paths(tmp_path / "project")["temp"].glob("koww_manifest_*"))


def test_audit_python_memory_does_not_grow_with_window_count(tmp_path, monkeypatch):
    project, root, config, row = public_project(tmp_path)
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / "cache"))
    path = root / config["manifests"]["korean_negative"]
    peaks = []
    for count in (1000, 10000):
        write_windows(path, row, count)
        tracemalloc.start()
        result = data.audit_project(project)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        assert result["passed"] and result["pools"]["korean_negative"]["rows"] == count
        assert result["pools"]["korean_negative"]["unique_source_hours"] == count * 2 / 3600
    # The existing SHA verifier retains a constant 8 MiB read buffer.
    assert max(peaks) < 12 * 1024**2
    assert peaks[1] < peaks[0] + 2 * 1024**2


@pytest.mark.parametrize("layout", [None, "source_then_round_v1"])
def test_large_journal_python_memory_is_bounded_and_old_layout_is_supported(tmp_path, monkeypatch, layout):
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / "cache"))
    path = tmp_path / "augmentation.jsonl"
    count = 20000
    with path.open("w", encoding="utf-8") as stream:
        for i in range(count):
            stream.write(json.dumps(dict(feature_row=i, sample_id=f"window{i}", round_id=0,
                                         seed=i, pcm_sha256=f"{i:064x}")) + "\n")
    metadata = dict(feature_rows=count, original_count=count, rounds=1)
    if layout:
        metadata["journal_layout"] = layout
    tracemalloc.start()
    features._verify_journal(tmp_path, metadata)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 4 * 1024**2


def test_old_interleaved_round_journal_remains_valid(tmp_path):
    entries = []
    for round_id in range(3):
        for sample in ("A", "B"):
            entries.append(dict(feature_row=len(entries), sample_id=sample, round_id=round_id,
                                seed=round_id, pcm_sha256=sample + str(round_id)))
    (tmp_path / "augmentation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8")
    metadata = dict(feature_rows=6, original_count=2, rounds=3)
    features._verify_journal(tmp_path, metadata)
    with pytest.raises(ValueError, match="Missing rounds"):
        features._verify_journal(tmp_path, {**metadata, "journal_layout": "source_then_round_v1"})
