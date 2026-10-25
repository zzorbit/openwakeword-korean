"""Storage evidence must not silently become a complete disk requirement."""
from copy import deepcopy
import json
import math

import pytest

from korean_openwakeword.storage_budget import commercial_storage_budget, storage_evidence
from korean_openwakeword.studio import StudioManager
from korean_openwakeword.studio_steps import preset_settings


def legacy_settings():
    settings = preset_settings("production")
    settings.pop("hard_samples_per_phrase", None)
    settings.pop("hard_generation_policy", None)
    settings.update(hard_originals=2000, train_references=100)
    return settings


def budget(**changes):
    settings = legacy_settings()
    settings.update(changes)
    return commercial_storage_budget(settings, base_archive_bytes=31592837906)


def by_id(result):
    return {row["id"]: row for row in result["rows"]}


def test_unknown_is_not_zero_and_reference_subtotal_is_not_total():
    result = budget()
    assert result["complete"] is False
    assert result["total_required_bytes"] is None
    assert result["peak_required_bytes"] is None
    unknown = [r for r in result["rows"] if r["classification"] == "unresolved"]
    assert len(unknown) >= 8
    assert all(r["bytes"] is None and "terms" not in r for r in unknown)
    known = [r for r in result["rows"] if r["bytes"] is not None]
    assert result["reference_subtotal_bytes"] == sum(r["bytes"] for r in known)
    assert {"published", "measured", "calculated", "reference"} <= {r["classification"] for r in known}
    assert result["removed_legacy_allowance_bytes"] == 33285996544
    assert all(r["id"] not in {"reserve", "base_working", "future_buffer"} for r in known)
    assert json.loads(json.dumps(result))["total_required_bytes"] is None


def test_baseline_file_formulas_and_shared_features():
    result = budget()
    rows = by_id(result)
    assert rows["negative_audio"]["bytes"] == 576000000000
    assert rows["negative_features"]["bytes"] == 55296000000
    assert rows["positive_features"]["bytes"] == 3072000000
    assert rows["hard_features"]["bytes"] == 12288000 + 512
    assert rows["tts_wavs"]["bytes"] == 171100170944
    assert rows["training_rirs"]["bytes"] == 8374902 + 1554203
    assert rows["fma_catalog"]["bytes"] == 260414445
    assert result["variables"]["tts_slots"] == 109024
    assert result["variables"]["reference_count"] == 140
    for row in result["rows"]:
        if row["bytes"] is not None:
            expected = math.ceil(sum(coef * (1 if key == "constant" else result["variables"][key])
                                     for key, coef in row["terms"].items()))
            assert row["bytes"] == expected


def test_site_audio_duplication_and_manual_final_example_are_explicit():
    result = budget()
    rows = by_id(result)
    assert rows["common_positive"]["bytes"] == 32000 * 167724 + 16000 * 199722
    assert rows["final_positive"]["bytes"] == 380000 * 167724
    assert rows["site_copies"]["bytes"] == sum(rows[key]["bytes"] for key in
        ("common_positive", "final_positive", "common_streams", "final_negative"))
    assert result["independent_final_scenario"]["status"] == "guide_example_not_frozen_protocol"
    assert result["independent_final_scenario"]["positive_sessions"] == 380000


def test_advanced_workload_updates_payload_not_common_protocol_or_approvals():
    settings = legacy_settings()
    before = deepcopy(settings)
    base = commercial_storage_budget(settings, base_archive_bytes=31592837906)
    assert settings == before
    small = budget(originals=20000, rounds=3, generic_hours=1000, korean_hours=200)
    rows = by_id(small)
    assert rows["negative_audio"]["bytes"] == 1200 * 115200000
    assert rows["negative_features"]["bytes"] == 1200 * 1800 * 6144
    assert rows["positive_features"]["bytes"] == 60000 * 6144
    assert small["variables"]["tts_slots"] == 29024
    assert rows["common_positive"] == by_id(base)["common_positive"]
    assert settings["final_locked"] is True


@pytest.mark.parametrize("changes", [
    {"originals": 0}, {"rounds": True}, {"hard_originals": -1},
    {"generic_hours": float("nan")}, {"korean_hours": float("inf")}, {"korean_hours": -1},
    {"material_acquisition_policy": "automatic_v2"}, {"validation_originals": 1},
])
def test_uncovered_or_invalid_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        budget(**changes)


def test_studio_exposes_evidence_without_starting_download_or_final(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Storage preview must be offline")
    monkeypatch.setattr("requests.get", forbidden)
    manager = StudioManager(tmp_path)
    recipes = manager.recipes()["presets"]
    assert [r["id"] for r in recipes] == ["check", "trial", "production"]
    assert "storage_budget" not in recipes[0]
    assert "storage_budget" not in recipes[1]
    result = recipes[-1]["storage_budget"]
    assert result == commercial_storage_budget(preset_settings("production"), base_archive_bytes=31592837906)
    assert recipes[-1]["estimated_workspace_bytes_scope"] == "legacy_partial_not_total"
    assert recipes[-1]["final_locked"] is True
    assert manager.registry["projects"] == {}


def test_evidence_totals_and_scope_are_internally_consistent():
    evidence = storage_evidence()
    assert sum(x["bytes"] for x in evidence["qwen_snapshots"]) == evidence["qwen_snapshot_bytes"]
    assert sum(x["bytes"] for x in evidence["common_source_components"]) == evidence["common_source_known_bytes"]
    assert evidence["checkpoint_reference"]["single_bytes"] * 9 == evidence["checkpoint_reference_bytes"]
    deployment = evidence["deployment_files_reference"]
    assert (deployment["three_trained_ONNX_bytes"] * deployment["ONNX_copies"]
            + deployment["frontend_pair_bytes"] * deployment["frontend_copies"]
            == evidence["deployment_files_reference_bytes"])
    bound = evidence["tts_completion_bound"]
    assert bound["bytes_per_completed_slot_bound"] == 2 * (327681 * 2 + 44) + (128000 * 2 + 44)
    assert "not extrapolated" in evidence["observed_fixtures"]["small_tts_scope"]


def test_saved_v4_budget_keeps_its_original_counts_and_new_v5_reduces_only_evaluation_payload():
    from korean_openwakeword.studio_steps import common_evaluation_policy
    current = legacy_settings()
    old = deepcopy(current)
    old["evaluation_policy"] = common_evaluation_policy(4)
    old["validation_originals"] = 5000
    previous = commercial_storage_budget(old, base_archive_bytes=31592837906)
    reduced = commercial_storage_budget(current, base_archive_bytes=31592837906)
    before, after = by_id(previous), by_id(reduced)
    assert before["common_positive"]["bytes"] == storage_evidence()["evaluation_audio"]["common_positive"]
    assert before["site_copies"]["bytes"] == storage_evidence()["evaluation_audio"]["site_copies"]
    assert previous["reference_subtotal_bytes"] == 1074538796779
    assert reduced["reference_subtotal_bytes"] == 1047345620779
    assert reduced["variables"]["tts_slots"] == previous["variables"]["tts_slots"] - 3000
    for key in before.keys() - {"common_positive", "site_copies", "tts_wavs", "scores_and_reports"}:
        assert before[key] == after[key], key
    assert "611.424" in before["scores_and_reports"]["detail"]
    assert "611.424" not in after["scores_and_reports"]["detail"]
    assert before["scores_and_reports"]["bytes"] is after["scores_and_reports"]["bytes"] is None


def test_unselected_phrase_count_is_unknown_not_zero_or_a_fixed_inventory():
    settings = preset_settings("production")
    before = deepcopy(settings)
    result = commercial_storage_budget(settings, base_archive_bytes=31592837906)
    rows = by_id(result)
    assert settings == before
    assert result["hard_generation"] == {
        "policy": "per_phrase_v1", "samples_per_phrase": 500,
        "phrase_count": None, "total_originals": None,
        "status": "pending_phrase_count", "included_in_reference_subtotal": False,
        "tts_slots_per_phrase": 500,
        "tts_audio_bytes_per_phrase": 500 * 1566856,
        "feature_bytes_per_phrase": 500 * 6144,
        "incremental_bytes_per_phrase": 500 * (1566856 + 6144),
    }
    assert result["variables"]["hard_originals"] is None
    assert result["variables"]["tts_slots"] == 100000 + 2000 + 5000 + 500
    assert result["variables"]["pilot_slots"] == 500
    assert result["tts_slots_scope"] == "excluding_unselected_hard_phrases"
    assert rows["hard_features"]["bytes"] is None
    assert rows["hard_features"]["classification"] == "unresolved"
    assert "terms" not in rows["hard_features"]
    assert rows["feature_headers"]["bytes"] == 512
    assert "유사어 제외" in rows["tts_wavs"]["label"]


def test_selected_phrase_count_adds_only_the_corresponding_audio_and_features():
    settings = preset_settings("production")
    base = commercial_storage_budget(settings, base_archive_bytes=31592837906)
    selected = commercial_storage_budget({**settings, "hard_phrase_count": 37}, base_archive_bytes=31592837906)
    assert selected["hard_generation"]["total_originals"] == 37 * 500
    assert selected["hard_generation"]["status"] == "calculated"
    assert selected["hard_generation"]["included_in_reference_subtotal"] is True
    assert selected["reference_subtotal_bytes"] - base["reference_subtotal_bytes"] == 37 * 500 * (1566856 + 6144)
    assert selected["variables"]["tts_slots"] - base["variables"]["tts_slots"] == 37 * 500
    assert by_id(selected)["hard_features"]["bytes"] == 37 * 500 * 6144 + 512
    assert "feature_headers" not in by_id(selected)
    assert selected["total_required_bytes"] is None
    assert selected["peak_required_bytes"] is None


@pytest.mark.parametrize("change", [
    {"hard_samples_per_phrase": 0}, {"hard_samples_per_phrase": True},
    {"hard_phrase_count": 0}, {"hard_phrase_count": -1}, {"hard_phrase_count": 1.5},
])
def test_invalid_per_phrase_storage_settings_are_rejected(change):
    settings = preset_settings("production")
    settings.update(change)
    with pytest.raises(ValueError):
        commercial_storage_budget(settings, base_archive_bytes=31592837906)
