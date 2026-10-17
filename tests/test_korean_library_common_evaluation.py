"""A common benchmark is independent of training budgets and proves its scope."""
from collections import Counter
from importlib.resources import files
import json

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
    policy = {**common_evaluation_policy(), "validation_originals": 200}
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
    assert all(v["validation_originals"] == 5000 and v["validation_references"] == 10 for v in values)
    assert [v["originals"] for v in values] == [1000, 20000, 100000]
