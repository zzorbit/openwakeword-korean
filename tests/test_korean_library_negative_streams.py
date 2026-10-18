"""Exact longer Negative streams reduce resets without fabricating exposure."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.negative_streams import pack_negative_streams, negative_scoring_budget
from korean_openwakeword.project import create_project
from korean_openwakeword.source_identity import partition_keys


def fixture_rows(tmp_path, lengths=(.4, .4, .4), bucket="environment"):
    project = create_project(tmp_path / "project", "보라야", "bora")
    values, rows = [], []
    for index, length in enumerate(lengths):
        samples = np.random.default_rng(200 + index).integers(-10000, 10000, round(16000 * length), dtype=np.int16)
        path = project.parent / "data" / f"source_{index}.wav"
        sf.write(path, samples, 16000, subtype="PCM_16")
        values.append(samples)
        rows.append({"sample_id": f"source_{index}", "canonical_source_id": f"fixture:source:{index}",
            "canonical_recording_id": f"fixture:recording:{index}", "path": str(path), "sha256": digest(path),
            "split": "validation", "kind": "negative", "calls": [], "duration_s": length,
            "category": "environment", "eval_bucket": bucket, "eval_type": "fixture_environment",
            "source_url": "https://example.test/isolated-original/" + str(index), "license": "isolated-test-fixture",
            "provenance": "public_dataset", "evaluation_content_approved": True,
            "individual_human_listening": False, "evaluation_content_review_scope": {"scope": "representative_fixture"}})
    admit(project, rows)
    return project, rows, values


def admit(project, rows):
    # Deliberate isolated admission fixture, never an assertion about real audio.
    path = project.parent / "review/isolated_evaluation_admission.json"
    write_json(path, {"kind": "studio_evaluation_noise_admission_v5", "state": "accepted_for_evaluation",
        "wakeword": "보라야", "evaluation_only": True, "training_admitted": False,
        "accepted_source_sha256": sorted({r["sha256"] for r in rows})})
    for row in rows:
        row["evaluation_admission"] = {"path": str(path), "sha256": digest(path)}


def test_exact_pcm_sequence_and_duration_have_no_gaps_gain_fades_or_repeats(tmp_path):
    project, rows, samples = fixture_rows(tmp_path)
    result = pack_negative_streams(project, rows, project.parent / "reports/negative_streams", seconds=1)
    assert result["stream_count"] == 2 and result["packed_originals"] == 3
    assert result["packed_seconds"] == pytest.approx(1.2)
    actual = []
    for stream in result["rows"]:
        values, rate = sf.read(stream["path"], dtype="int16")
        actual.extend(values)
        assert rate == 16000 and stream["kind"] == "negative" and stream["calls"] == []
        assert stream["duration_s"] == len(values) / rate
        assert stream["individual_stream_listening_claimed"] is False
    assert np.array_equal(np.asarray(actual, dtype=np.int16), np.concatenate(samples))
    assert all(Path(row["path"]).exists() for row in rows)


def test_all_original_identity_approval_and_offsets_survive_for_independent_final_audit(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    result = pack_negative_streams(project, rows, project.parent / "reports/negative_streams", seconds=10)
    stream = result["rows"][0]
    children = stream["evaluation_stream_components"]
    assert len(children) == 3
    for original, child in zip(rows, children):
        assert all(child[key] == value for key, value in original.items())
        assert child["original_read_frames"] == 6400
    assert [row["stream_start_s"] for row in children] == [0, .4, .8]
    inherited = set(partition_keys(stream))
    assert all(("source", row["canonical_source_id"]) in inherited and ("sha256", row["sha256"]) in inherited for row in rows)
    assert stream["canonical_source_id"].startswith("evaluation_stream:")


def test_one_long_original_and_multiple_intervals_of_a_recording_are_never_split(tmp_path):
    project, rows, _ = fixture_rows(tmp_path, lengths=(.8, .7))
    first = rows[0]
    parts = [{**first, "sample_id": "part_a", "start_s": 0, "end_s": .4},
             {**first, "sample_id": "part_b", "start_s": .4, "end_s": .8}, rows[1]]
    result = pack_negative_streams(project, parts, project.parent / "reports/negative_streams", seconds=.6)
    assert len(result["rows"]) == 2
    assert [row["duration_s"] for row in result["rows"]] == [.8, .7]
    assert [len(row["evaluation_stream_components"]) for row in result["rows"]] == [2, 1]


def test_supplemental_audio_is_kept_at_its_original_path_and_buckets_stay_apart(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    rows[1]["eval_bucket"] = "electronic"
    rows[2].pop("eval_bucket")
    original = rows[2]["path"]
    result = pack_negative_streams(project, rows, project.parent / "reports/negative_streams", seconds=10)
    assert result["stream_count"] == 2 and result["unchanged_supplemental_recordings"] == 1
    assert result["rows"][-1]["path"] == original
    assert {row["eval_bucket"] for row in result["rows"][:-1]} == {"environment", "electronic"}


def test_same_input_resume_is_immutable_and_changed_recipe_or_output_is_rejected(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    output = project.parent / "reports/negative_streams"
    first = pack_negative_streams(project, rows, output, seconds=1)
    manifest = Path(first["manifest"]["path"]).read_bytes()
    second = pack_negative_streams(project, list(reversed(rows)), output, seconds=1)
    assert first == second and Path(first["manifest"]["path"]).read_bytes() == manifest
    with pytest.raises(ValueError, match="different immutable"):
        pack_negative_streams(project, rows, output, seconds=2)
    with Path(first["rows"][0]["path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed bound"):
        pack_negative_streams(project, rows, output, seconds=1)


def test_unreviewed_sources_and_rir_filters_do_not_become_continuous_negative(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    rows[0]["evaluation_content_approved"] = False
    with pytest.raises(ValueError, match="reviewed and admitted"):
        pack_negative_streams(project, rows, project.parent / "reports/negative_streams")
    rows[0]["evaluation_content_approved"] = True
    rows[0]["category"] = "rir"
    with pytest.raises(ValueError, match="never RIRs"):
        pack_negative_streams(project, rows, project.parent / "reports/negative_streams")


def test_duplicate_or_overlapping_originals_are_rejected_without_inflating_hours(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    duplicate = {**rows[0], "sample_id": "renamed_alias", "canonical_source_id": "renamed", "canonical_recording_id": "renamed_parent"}
    with pytest.raises(ValueError, match="Repeated original PCM"):
        pack_negative_streams(project, rows + [duplicate], project.parent / "reports/negative_streams")
    rows[0]["parent_start_s"] = 0
    rows[1].update(canonical_recording_id=rows[0]["canonical_recording_id"], parent_start_s=.1)
    rows[1]["parent_source_id"] = rows[0]["canonical_recording_id"]
    with pytest.raises(ValueError, match="overlapping"):
        pack_negative_streams(project, rows, project.parent / "reports/negative_streams")


def test_noncanonical_pcm_and_changed_original_hash_are_not_silently_converted(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    sf.write(rows[0]["path"], np.ones(6400) * .1, 16000, subtype="FLOAT")
    rows[0]["sha256"] = digest(rows[0]["path"])
    admit(project, rows)
    with pytest.raises(ValueError, match="already be mono 16kHz PCM16"):
        pack_negative_streams(project, rows, project.parent / "reports/negative_streams")


def test_source_coverage_is_identical_after_stream_packing_and_not_multiplied_by_wrappers(tmp_path):
    from korean_openwakeword.evaluation_materials import evaluation_material_coverage
    project, rows, _ = fixture_rows(tmp_path)
    raw = evaluation_material_coverage(rows)
    packed = pack_negative_streams(project, rows, project.parent / "reports/negative_streams", seconds=1)
    actual = evaluation_material_coverage(packed["rows"])
    assert actual == raw
    assert actual["actual_total_hours"] == pytest.approx(1.2 / 3600)
    assert actual["unique_recordings_by_bucket"] == {"environment": 3}


def test_complete_wav_without_its_last_stream_journal_recovers_without_changing_pcm(tmp_path):
    project, rows, _ = fixture_rows(tmp_path)
    output = project.parent / "reports/negative_streams"
    first = pack_negative_streams(project, rows, output, seconds=1)
    source = first["rows"][0]
    path = output / "streams" / (source["sample_id"] + ".json")
    path.unlink()  # Isolated fixture emulates a crash after WAV rename.
    second = pack_negative_streams(project, rows, output, seconds=1)
    assert second == first and digest(source["path"]) == source["sha256"]
    assert path.is_file() and not list((output / "audio").glob("*.part"))


def test_default_runtime_budget_excludes_two_second_warmup_and_partial_frames(tmp_path):
    project, rows, _ = fixture_rows(tmp_path, lengths=(1.005, 1.005, 1.005))
    for row in rows:
        row.update(source_pool_scope="common_30h_only", evaluation_source_scope="common_30h_only",
            evaluation_material_catalog_id="fixture_catalog", normalized_frames=round(row["duration_s"] * 16000),
            positive_mix_eligible=True, evaluation_initialization_margin=False)
    budget = negative_scoring_budget(rows)
    assert budget["stream_count"] == 1
    assert budget["raw_seconds"] == pytest.approx(3.015)
    assert budget["eligible_seconds"] == pytest.approx(.96)
    assert budget["warmup_excluded_seconds"] == 2
    assert budget["partial_frame_excluded_seconds"] == pytest.approx(.055)
    actual = pack_negative_streams(project, rows, project.parent / "reports/unified", strict_common_pool=True)
    assert actual["duration_budget"]["eligible_seconds"] == budget["eligible_seconds"]
    assert actual["core_source_seconds"] == pytest.approx(3.015)
    assert actual["initialization_margin_seconds"] == 0
    assert actual["unchanged_supplemental_recordings"] == 0
    rows[0].pop("source_pool_scope")
    with pytest.raises(ValueError, match="pinned seven-bucket"):
        pack_negative_streams(project, rows, project.parent / "reports/forbidden", strict_common_pool=True)


def test_thirty_raw_hours_of_short_recordings_need_real_initialization_margin():
    # Metadata-only: 3,600 independent short recordings, not an artificial
    # 30-hour WAV or repeated audio pretending to create exposure.
    rows = [{"canonical_recording_id": f"original:{i:04d}", "canonical_source_id": f"source:{i:04d}",
             "sha256": f"{i:064x}", "eval_bucket": "environment", "category": "environment",
             "normalized_frames": 30 * 16000, "duration_s": 30} for i in range(3600)]
    raw = negative_scoring_budget(rows)
    assert raw["raw_seconds"] == 108000 and raw["stream_count"] == 180
    assert raw["eligible_seconds"] == 108000 - 360
    extra = {**rows[0], "canonical_recording_id": "unused:new", "canonical_source_id": "unused:new",
             "sha256": "f" * 64, "normalized_frames": round(362.08 * 16000), "duration_s": 362.08}
    result = negative_scoring_budget(rows + [extra])
    assert result["stream_count"] == 181
    assert result["eligible_seconds"] >= 108000
    assert result["raw_seconds"] - result["eligible_seconds"] == pytest.approx(362)
