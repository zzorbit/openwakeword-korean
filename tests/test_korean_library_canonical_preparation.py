"""Storage migration preserves selection, partitions, manifests and resume."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import preparation
from korean_openwakeword.canonical_audio import canonicalize_prepared_source, verify_canonical_audio
from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project


def _index(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _provider(tmp_path, *, duplicates=False, count=6):
    folder = tmp_path / "provider"
    folder.mkdir()
    rows = []
    first = np.rint(np.random.default_rng(60).normal(0, 1200, 16000)).astype(np.int16)
    for i in range(count):
        samples = first if duplicates and i < 2 else np.random.default_rng(60+i).normal(0, 0.05, 16000)
        path = folder / (f"source{i:02d}" + (".flac" if i == 1 else ".wav"))
        sf.write(path, samples, 16000, subtype="PCM_16")
        rows.append({"path": path.name, "sha256": digest(path), "dataset_id": "fixture",
            "source_url": "https://example.org/fixture", "license": "fixture",
            "category": "environment", "category_basis": "procedural fixture",
            "language": "unknown", "speech": False, "canonical_source_id": "x"+str(i),
            "canonical_recording_id": "r"+str(i)})
    return folder, rows


def _word(tmp_path, label, *, flac=False):
    project = create_project(tmp_path / label, "보라야", "bora", positive_originals=1, updates=2)
    if flac:
        _, config = load_project(project)
        config["audio_storage_policy"] = "canonical_flac_only_v1"
        write_json(project, config)
    return project


def test_direct_flac_preparation_recovers_after_codec_commit_before_check_record(tmp_path, monkeypatch):
    folder, rows = _provider(tmp_path)
    project = _word(tmp_path, "word", flac=True)
    options = {"input_directory": folder, "source_index": _index(tmp_path / "index.jsonl", rows),
               "output": "prepared", "validation_fraction": 0, "preview_seconds": 1}
    writer = preparation.write_json
    def interrupt(path, value):
        if Path(path).parent.name == "checks":
            raise RuntimeError("fixture crash before check commit")
        return writer(path, value)
    monkeypatch.setattr(preparation, "write_json", interrupt)
    with pytest.raises(RuntimeError, match="before check commit"):
        preparation.prepare_public_audio(project, **options)
    generated = list((project.parent / "prepared/audio").glob("*.flac"))
    assert len(generated) == 1
    flac = generated[0]
    before, original = flac.stat(), digest(flac)
    assert flac.with_suffix(".flac.json").is_file()
    monkeypatch.setattr(preparation, "write_json", writer)
    result = preparation.prepare_public_audio(project, **options)
    assert result["train_source_count"] == 6 and result["rejected_files"] == 0
    assert digest(flac) == original and flac.stat().st_ino == before.st_ino
    assert all((folder / row["path"]).exists() for row in rows)
    for row in read_rows(result["artifacts"]["source_manifest"]["path"]):
        assert row["provider_sha256"] == row["sha256"]
        verify_canonical_audio(row)
    assert preparation.prepare_public_audio(project, **options)["resumed"]


@pytest.mark.parametrize("fraction,seconds", [(0, 1), (0, 3), (.25, 1), (.25, None)])
def test_normalized_duplicates_keep_legacy_partition_order_and_quota(tmp_path, fraction, seconds):
    folder, rows = _provider(tmp_path, duplicates=True, count=4)
    assert rows[0]["sha256"] != rows[1]["sha256"]
    canonical = canonicalize_prepared_source({"source": {"directory": str(folder)}, "rows": rows},
                                            tmp_path / "canonical")["prepared"]
    assert canonical["rows"][0]["logical_audio_sha256"] == canonical["rows"][1]["logical_audio_sha256"]
    outcomes, plans = [], []
    for label, directory, index_rows in (("legacy", folder, rows),
                                         ("canonical", Path(canonical["source"]["directory"]), canonical["rows"])):
        project = _word(tmp_path, label + "_word")
        options = {"input_directory": directory, "source_index": _index(tmp_path / (label+".jsonl"), index_rows),
                   "output": "prepared", "validation_fraction": fraction,
                   "target_hours": None if seconds is None else seconds / 3600,
                   "preview_seconds": 1, "seed": 20260928}
        try:
            result = preparation.prepare_public_audio(project, **options)
        except ValueError as error:
            outcomes.append(("error", str(error)))
        else:
            manifest = lambda key: [(r["canonical_source_id"], r["sample_id"], r["split"], r["duration_s"])
                                     for r in read_rows(result["artifacts"][key]["path"])]
            outcomes.append(("passed", manifest("source_manifest"), manifest("validation_reserve"),
                             manifest("unused_manifest"), result["train_hours"], result["validation_hours"]))
        plans.append(read_json(project.parent / "prepared/contract.json")["header_partition_plan"])
    assert plans[0] == plans[1]
    assert outcomes[0] == outcomes[1]


def test_unselected_canonical_rows_bind_physical_files_and_resume(tmp_path):
    folder, rows = _provider(tmp_path)
    converted = canonicalize_prepared_source({"source": {"directory": str(folder)}, "rows": rows},
                                            tmp_path / "canonical")["prepared"]
    project = _word(tmp_path, "word")
    options = {"input_directory": converted["source"]["directory"],
               "source_index": _index(tmp_path / "index.jsonl", converted["rows"]),
               "output": "prepared", "validation_fraction": 0, "target_hours": 1 / 3600,
               "preview_seconds": 1}
    result = preparation.prepare_public_audio(project, **options)
    assert result["train_source_count"] == 1 and result["unused_source_count"] == 5
    for row in read_rows(result["artifacts"]["unused_manifest"]["path"]):
        assert not row["technical_pass"] and not row["training_admitted"]
        assert row["sha256"] == digest(row["path"])
        assert row["sha256"] != row["logical_audio_sha256"]
        verify_canonical_audio(row)
    assert preparation.prepare_public_audio(project, **options)["resumed"]
