"""A common benchmark is independent of training budgets and proves its scope."""
from collections import Counter
from importlib.resources import files
import json
import pytest

from korean_openwakeword.common import read_json
from korean_openwakeword.evaluation_coverage import assess_common_evaluation
from korean_openwakeword.studio_evaluation_data import evaluation_catalog, evaluation_preview
from korean_openwakeword.studio_steps import common_evaluation_policy, preset_settings


def test_expanded_catalog_preserves_every_previous_pin_and_has_real_source_coverage():
    prior = read_json(files("korean_openwakeword") / "resources/evaluation_noise_v3.json")
    current = evaluation_catalog()
    by_id = {r["id"]: r for r in current["assets"]}
    assert current["recipe_version"] == 4 and len(by_id) == len(current["assets"]) == 193
    assert all(by_id[r["id"]] == r for r in prior["assets"])
    assert len({r["canonical_recording_id"] for r in by_id.values()}) == 193
    assert current["size_bytes"] == sum(r["size_bytes"] for r in by_id.values()) == 109242565
    assert Counter(r["category"] for r in by_id.values()) == {"environment": 150, "music": 37, "competing_speech": 6}
    assert len([r for r in by_id.values() if r["eval_type"].startswith("music_")]) == 12
    assert all(r["source_url"].startswith("https://") and len(r["sha256"]) == 64 and r["license"] for r in by_id.values())
    previews = evaluation_preview()
    assert len(previews) == 4 and sum(r.get("file_count", 0) for r in previews) == 193
    assert all("_" not in label for r in previews for label in r["type_labels"])


def _scope_fixture(tmp_path):
    policy = {**common_evaluation_policy(1), "validation_originals": 200}
    rows = []
    for slot in range(200):
        base = {"sample_id": f"call_{slot}", "original_id": f"call_{slot}", "kind": "positive",
                "voice_family_id": "family_" + str(slot % 10)}
        rows.append({**base, "category": "clean", "snr_db": None})
        for category, source_count, type_count in (("environment", 100, 30), ("music", 12, 6), ("competing_speech", 12, 6)):
            for snr in policy["snrs"]:
                rows.append({**base, "sample_id": base["sample_id"] + f"_{category}_{snr}",
                    "category": category, "snr_db": snr, "noise_source_group_id": f"{category}:{slot % source_count}",
                    "noise_eval_type": ("music_" if category == "music" else category + "_") + str((slot % source_count) % type_count)})
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return policy, path


def test_coverage_requires_diversity_voice_samples_and_real_negative_time(tmp_path):
    policy, path = _scope_fixture(tmp_path)
    result = assess_common_evaluation(policy, {"models": [{"negative": {"hours": 20}}]}, path, {"fixture": True})
    assert result["coverage_targets_passed"] and not result["shortages"]
    assert result["positive_originals"] == 200 and result["synthetic_voice_families"] == 10
    assert result["categories"]["environment"]["used"]["sources"] == 100
    assert result["categories"]["music"]["used"]["music_sources"] == 12
    assert result["independent_Final_completed"] is False


def test_reusing_one_hour_for_three_models_is_not_three_hours_and_missing_evidence_fails(tmp_path):
    policy, path = _scope_fixture(tmp_path)
    result = assess_common_evaluation(policy, {"models": [{"negative": {"hours": 8}}] * 3}, path, None)
    assert result["negative_hours_per_candidate_minimum"] == 8
    assert not result["coverage_targets_passed"]
    assert any("8/20" in s for s in result["shortages"])
    assert any("분포 근거" in s for s in result["shortages"])
    result = assess_common_evaluation(policy, {"models": [{"negative": {"hours": 20}}, {}]}, path, {})
    assert result["negative_hours_per_candidate_minimum"] is None and not result["coverage_targets_passed"]


def test_snr_variants_do_not_inflate_original_or_voice_counts(tmp_path):
    policy, path = _scope_fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["original_id"] = "same_utterance"
        row["voice_family_id"] = "same_voice"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = assess_common_evaluation(policy, {"models": [{"negative": {"hours": 20}}]}, path, {"fixture": True})
    assert result["positive_originals"] == result["synthetic_voice_families"] == 1
    assert not result["coverage_targets_passed"]


def test_every_training_budget_uses_one_full_common_protocol():
    values = [preset_settings(key) for key in ("check", "trial", "production")]
    assert len({json.dumps(v["evaluation_policy"], sort_keys=True) for v in values}) == 1
    assert all(v["validation_originals"] == 5000 and v["validation_references"] == 30 for v in values)
    assert all(v["evaluation_policy"]["positive_sessions"] == 112000 for v in values)
    assert sum(values[0]["evaluation_policy"]["negative_hours_by_bucket"].values()) == 30
    assert [v["originals"] for v in values] == [1000, 20000, 100000]


def test_new_protocol_only_changes_the_approved_rir_subset_and_keeps_historical_policies():
    from korean_openwakeword.common import identity
    from korean_openwakeword.studio_steps import workflow_plan
    old = common_evaluation_policy(2)
    current = common_evaluation_policy(3)
    assert old["id"] == "common_development_v2"
    assert (old["positive_sessions"], old["reverb_originals"], old["reverb_positive_sessions"]) == (100000, 625, 20000)
    assert current["id"] == "common_development_v3"
    assert (current["positive_sessions"], current["reverb_originals"], current["reverb_positive_sessions"]) == (112000, 1000, 32000)
    changed = {key for key in old if current[key] != old[key]}
    assert changed == {"id", "positive_sessions", "reverb_originals", "reverb_positive_sessions"}
    assert set(current) - set(old) == {"reverb_selection"}
    assert current["reverb_selection"] == "metadata_axes"
    assert current["positive_sessions"] == current["validation_originals"] * 16 + current["reverb_originals"] * 16 * 2
    assert identity(old) != identity(current)
    # Hashes from the published 0.7.0 policies: old workflows keep exactly
    # their recorded contract, rather than silently gaining the new budget.
    assert identity(common_evaluation_policy(1)) == "cf2223feb7e8301f2e9b89c82f0d85dabd6455c528c8e262f990f5f65e7957e8"
    assert identity(old) == "b7cbaf6670eca4de1b21f8db3bc1d0f03e8aee6b27acc6e2eae6826efca4da29"
    for version in (1, 2, 3, 4):
        settings = preset_settings()
        policy = common_evaluation_policy(version)
        settings.update(evaluation_policy=policy, **{key: policy[key] for key in
                        ("validation_references", "validation_originals", "noise_sections")})
        frozen = json.dumps(settings, sort_keys=True)
        assert workflow_plan(settings)
        assert json.dumps(settings, sort_keys=True) == frozen
    assert "reverb_originals" not in common_evaluation_policy(1)
    assert common_evaluation_policy(1)["positive_sessions"] == 80000


@pytest.mark.parametrize("version", [0, 5, "3", None])
def test_unknown_policy_versions_are_rejected(version):
    with pytest.raises(ValueError, match="Unsupported"):
        common_evaluation_policy(version)


def test_unknown_or_modified_policy_cannot_be_reinterpreted_as_the_new_default():
    from korean_openwakeword.studio_steps import workflow_plan
    settings = preset_settings()
    settings["evaluation_policy"]["id"] = "common_development_future"
    with pytest.raises(ValueError, match="공통 평가"):
        workflow_plan(settings)
    settings["evaluation_policy"] = common_evaluation_policy(2)
    settings["evaluation_policy"]["reverb_originals"] = 1000
    with pytest.raises(ValueError, match="공통 평가"):
        workflow_plan(settings)


def _full_rir_coverage(tmp_path, monkeypatch, *, reverb_count=1000, mutation=None):
    """Exercise full-sized row membership without generating any audio."""
    import korean_openwakeword.evaluation_coverage as coverage_module
    import korean_openwakeword.evaluation_materials as material_module
    expected = [("clean", None)] + [(category, snr) for category in
                ("environment", "music", "competing_speech") for snr in (20, 15, 10, 5, 0)]
    def generate():
        for group, count in (("none", 5000), ("moderate", reverb_count), ("strong", reverb_count)):
            for slot in range(count):
                for category, snr in expected:
                    row = {"sample_id": f"{group}:{slot}:{category}:{snr}", "original_id": f"call:{slot}",
                           "kind": "positive", "voice_family_id": f"voice:{slot % 30}",
                           "reverb_condition": group, "category": category, "snr_db": snr}
                    if group != "none":
                        row.update(rir_sha256=f"{group}:rir:{slot % 40}", canonical_room_id=f"{group}:room:{slot % 10}")
                    if category != "clean":
                        row.update(noise_source_group_id=f"{category}:{slot % 120}",
                                   noise_eval_type=("music_" if category == "music" else category + "_") + str(slot % (30 if category == "environment" else 6)))
                    yield mutation(row) if mutation else row
    material_calls = []
    def material_coverage(rows, strict_common_pool=False):
        assert strict_common_pool is False  # This fixture exercises preserved v2/v3.
        material_calls.append(list(rows))
        return {"target_met": True, "shortages": []}
    monkeypatch.setattr(coverage_module, "iter_rows", lambda path: generate())
    monkeypatch.setattr(material_module, "evaluation_material_coverage", material_coverage)
    path = tmp_path / "bound_metadata_only_manifest.jsonl"
    path.write_text("metadata-only test fixture; generator is injected", encoding="utf-8")
    return lambda version=3: assess_common_evaluation(common_evaluation_policy(version),
            {"models": [{"negative": {"hours": 30}}]}, path, {"fixture": True}), material_calls


def test_full_new_protocol_requires_all_112000_conditions_and_1000_paired_originals(tmp_path, monkeypatch):
    assess, materials = _full_rir_coverage(tmp_path, monkeypatch)
    result = assess()
    assert result["coverage_targets_passed"] and not result["shortages"]
    assert result["positive_originals"] == 5000 and result["synthetic_voice_families"] == 30
    assert result["reverberation"]["originals_per_group"] == {"moderate": 1000, "strong": 1000}
    assert sum(row["samples"] for row in result["reverberation"]["condition_counts"]) == 112000
    assert len(materials) == 1  # The new version retains the real 30h material check.


def test_625_originals_remain_valid_for_old_protocol_but_do_not_satisfy_new_protocol(tmp_path, monkeypatch):
    assess, _ = _full_rir_coverage(tmp_path, monkeypatch, reverb_count=625)
    assert assess(2)["coverage_targets_passed"]
    result = assess(3)
    assert not result["coverage_targets_passed"]
    assert any("625/1000" in reason for reason in result["shortages"])


@pytest.mark.parametrize("defect,expected", [
    ("different_subsets", "서로 다른 호출어"),
    ("missing_dry", "잔향 없음 시험 누락"),
    ("duplicate_replaces_missing_cell", "원본별 16조건 대응 누락: strong")])
def test_paired_coverage_rejects_mismatches_that_leave_all_cell_totals_unchanged(tmp_path, monkeypatch, defect, expected):
    def mutate(row):
        if defect == "different_subsets" and row["reverb_condition"] == "strong":
            row["original_id"] = "call:" + str(1000 + int(row["original_id"].split(":")[1]))
        elif defect == "missing_dry" and row["reverb_condition"] == "none" and row["original_id"] == "call:0":
            row["original_id"] = "extra_dry_call"
        elif defect == "duplicate_replaces_missing_cell" and row["reverb_condition"] == "strong" and row["original_id"] == "call:0" and row["category"] == "environment" and row["snr_db"] == 5:
            row["original_id"] = "call:1"
        return row
    assess, _ = _full_rir_coverage(tmp_path, monkeypatch, mutation=mutate)
    result = assess()
    assert not result["coverage_targets_passed"]
    assert any(expected in reason for reason in result["shortages"])
