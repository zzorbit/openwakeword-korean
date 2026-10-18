"""A new benchmark cannot silently append legacy clips or Train reserves."""
from pathlib import Path

import pytest

from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json
from korean_openwakeword import studio_data, studio_evaluation_data
from korean_openwakeword.studio_steps import common_evaluation_policy, preset_settings, workflow_plan
from test_korean_library_evaluation_pool_review import group_preparation
from test_korean_library_studio_evaluation_data import saved_answers


def _common_preparation(tmp_path, *, margin=False):
    project, prep, queue = group_preparation(tmp_path, groups=(5, 40, 40), rir_groups=(1, 2))
    sources = read_rows(queue["pool_manifest"])
    for row in sources:
        row["evaluation_source_scope"] = "common_30h_only"
        if row["category"] != "rir":
            row["eval_bucket"] = "environment"
            row["positive_mix_eligible"] = not (margin and row["sample_id"] == "source_0_4")
            row["evaluation_initialization_margin"] = not row["positive_mix_eligible"]
    # Test-only replacement happens before saved answers or any admission.
    Path(queue["pool_manifest"]).unlink()
    binding = studio_data._write_rows(Path(queue["pool_manifest"]), sources)
    record = read_json(queue["preparation"])
    record.update(kind="studio_evaluation_noise_preparation_v6", recipe_version=6,
                  evaluation_source_scope="common_30h_only")
    record["artifacts"]["source_manifest"] = binding
    write_json(queue["preparation"], record)
    prep = {**record, "preparation": {"path": queue["preparation"], "sha256": digest(queue["preparation"])}}
    queue["evaluation_source_scope"] = "common_30h_only"
    return project, prep, queue


def test_all_budgets_default_to_one_common_only_protocol_without_changing_old_policies():
    policies = [preset_settings(name)["evaluation_policy"] for name in ("check", "trial", "production")]
    assert policies[0] == policies[1] == policies[2] == common_evaluation_policy(4)
    current = policies[0]
    assert current["evaluation_materials_recipe"] == 6
    assert current["evaluation_source_scope"] == "common_30h_only"
    previous = common_evaluation_policy(3)
    assert current["positive_sessions"] == previous["positive_sessions"] == 112000
    assert {key for key in previous if previous[key] != current[key]} == {"id", "evaluation_materials_recipe"}
    assert identity(common_evaluation_policy(1)) == "cf2223feb7e8301f2e9b89c82f0d85dabd6455c528c8e262f990f5f65e7957e8"
    assert identity(common_evaluation_policy(2)) == "b7cbaf6670eca4de1b21f8db3bc1d0f03e8aee6b27acc6e2eae6826efca4da29"
    assert previous["evaluation_materials_recipe"] == 5 and "evaluation_source_scope" not in previous
    assert workflow_plan(preset_settings())


def test_new_recipe_does_not_load_old_catalog_or_show_legacy_reserve_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_evaluation_data, "evaluation_catalog", lambda: pytest.fail("legacy catalog is not part of recipe 6"))
    monkeypatch.setattr(studio_evaluation_data, "evaluation_preview", lambda: pytest.fail("legacy preview is not part of recipe 6"))
    plan = studio_data.material_recipe("check", tmp_path, common_evaluation_policy())
    assert plan["recipe_version"] == 6 and plan["evaluation_assets"] == []
    assert plan["evaluation_source_scope"] == "common_30h_only"
    assert len(plan["evaluation_sources"]) == 8  # Seven canonical buckets and RIR.
    assert all(row["id"].startswith("common_30h_") or row["id"] == "openslr28_common_rir_v1"
               for row in plan["evaluation_sources"])
    assert plan["known_download_bytes"] == sum(a["size_bytes"] for a in plan["assets"]) + plan["evaluation_download_bytes"]


def test_common_only_preparation_never_calls_supplement_download_and_preserves_v5(tmp_path, monkeypatch):
    from korean_openwakeword import evaluation_materials, evaluation_rir
    project, prep, queue = group_preparation(tmp_path, groups=(1,))
    rows = read_rows(queue["pool_manifest"])
    rows[0]["eval_bucket"] = "environment"
    rows[0]["positive_mix_eligible"] = True
    legacy_path = Path(queue["preparation"])
    legacy_bytes = legacy_path.read_bytes()
    calls = []
    def common_sources(*args, **kwargs):
        calls.append(kwargs)
        return {"source_rows": rows, "coverage": {"target_met": True}, "target_met": True}
    monkeypatch.setattr(evaluation_materials, "prepare_common_sources", common_sources)
    monkeypatch.setattr(evaluation_rir, "prepare_common_rirs", lambda *a, **k: {"source_rows": [], "coverage": {}})
    monkeypatch.setattr(studio_evaluation_data, "prepare_evaluation_materials", lambda *a, **k: pytest.fail("legacy download"))
    monkeypatch.setattr(studio_evaluation_data, "evaluation_catalog", lambda: pytest.fail("legacy catalog"))
    prepared = {"evaluation_materials_policy": common_evaluation_policy()}
    result = studio_evaluation_data.prepare_common_evaluation_materials(project, tmp_path / "workspace", prepared)
    assert result["recipe_version"] == 6 and result["source_count"] == 1
    assert calls == [{"strict_common_pool": True}]
    assert all(r["evaluation_source_scope"] == "common_30h_only" for r in read_rows(result["artifacts"]["source_manifest"]["path"]))
    assert "evaluation_v6_" in result["preparation"]["path"]
    assert legacy_path.read_bytes() == legacy_bytes
    assert studio_evaluation_data.prepare_common_evaluation_materials(project, tmp_path / "workspace", prepared)["resumed"]
    assert len(calls) == 1
    stored = read_json(result["preparation"]["path"])
    stored["evaluation_source_scope"] = "legacy_plus_common"
    write_json(result["preparation"]["path"], stored)
    with pytest.raises(ValueError, match="Frozen common"):
        studio_evaluation_data.prepare_common_evaluation_materials(project, tmp_path / "workspace", prepared)


@pytest.mark.parametrize("eligible_seconds", [30 * 3600, 30 * 3600 - .08])
def test_common_admission_excludes_old_reserve_and_keeps_margin_out_of_positive_mixtures(tmp_path, monkeypatch, eligible_seconds):
    from korean_openwakeword import evaluation_materials, negative_streams, preparation
    project, prep, queue = _common_preparation(tmp_path, margin=True)
    answers, _ = saved_answers(project, prep, queue)
    root = project.parent
    reserve = studio_data._write_rows(root / "review/legacy_train_reserve.jsonl", [{
        "sample_id": "must_not_be_published", "category": "competing_speech", "duration_s": 9,
        "canonical_source_id": "zeroth:legacy_reserve", "sha256": "0" * 64}])
    receipt = root / "review/train_admission.json"
    write_json(receipt, {"kind": "prepared_public_audio_admission_v1", "state": "accepted",
                        "source_manifests": {}, "manifests": {}, "evaluation_reserve": reserve})
    connected = []
    def connect(*args, **kwargs):
        connected.append(True)
        return {"fixture_connection": True}
    monkeypatch.setattr(preparation, "connect_materials", connect)
    monkeypatch.setattr(evaluation_materials, "evaluation_material_coverage", lambda rows, **kwargs: {"target_met": True, "shortages": []})
    monkeypatch.setattr(negative_streams, "negative_scoring_budget", lambda rows, **kwargs: {"eligible_seconds": eligible_seconds,
        "scope": "metadata-only routing fixture, not actual audio exposure"})
    packed = []
    def pack(project, rows, output, **kwargs):
        packed.append((rows, kwargs))
        return {"rows": rows, "fixture": True}
    monkeypatch.setattr(negative_streams, "pack_negative_streams", pack)
    result = studio_data.admit_materials(project, {"recipe_version": 6, "review_queues": [queue],
        "evaluation_source_scope": "common_30h_only", "evaluation_materials_policy": common_evaluation_policy()},
        {queue["id"]: answers}, admissions=[{"path": str(receipt), "sha256": digest(receipt)}])
    if eligible_seconds < 30 * 3600:
        assert result["state"] == "awaiting_additional_materials" and not connected and not packed
        assert result["negative_duration_budget"]["eligible_seconds"] == eligible_seconds
        assert any("30시간" in reason for reason in result["shortages"])
        assert not read_rows(root / read_json(project)["manifests"]["noise_validation"])
        return
    assert result["state"] == "connected" and connected == [True]
    noise = read_rows(result["validation_noise_manifest"]["path"])
    negatives = read_rows(result["negative_validation_manifest"]["path"])
    rirs = read_rows(result["validation_rir_manifest"]["path"])
    assert len(noise) == 4 and len(negatives) == 5 and len(rirs) == 80
    assert packed[0][1] == {"strict_common_pool": True}
    assert all(row.get("positive_mix_eligible") is not False for row in noise)
    assert sum(row.get("evaluation_initialization_margin", False) for row in negatives) == 1
    assert all(row["sample_id"] != "must_not_be_published" for row in noise + negatives)
    assert Path(reserve["path"]).exists() and read_rows(reserve["path"])[0]["sample_id"] == "must_not_be_published"
    assert "validation_rir_v6_" in result["validation_rir_manifest"]["path"]
    assert read_json(project)["manifests"]["noise_validation"]


@pytest.mark.parametrize("mutation", ["missing_scope", "missing_bucket", "inherited_old_pool"])
def test_rebound_legacy_rows_or_scope_cannot_be_approved_as_common_pool(tmp_path, mutation):
    project, prep, queue = _common_preparation(tmp_path)
    if mutation == "inherited_old_pool":
        record = read_json(queue["preparation"])
        record["inherited_admissions"] = [{"path": "legacy.json", "sha256": "0" * 64}]
        write_json(queue["preparation"], record)
    else:
        path = Path(queue["pool_manifest"])
        rows = read_rows(path)
        rows[0].pop("evaluation_source_scope" if mutation == "missing_scope" else "eval_bucket")
        path.unlink()
        binding = studio_data._write_rows(path, rows)
        record = read_json(queue["preparation"])
        record["artifacts"]["source_manifest"] = binding
        prep["artifacts"]["source_manifest"] = binding
        write_json(queue["preparation"], record)
    answers, _ = saved_answers(project, prep, queue)
    with pytest.raises(ValueError, match="scope|Legacy|Evaluation-only"):
        studio_evaluation_data.admit_evaluation_review(project, queue, answers, {"recipe_version": 6})


def test_existing_frozen_project_is_not_rewritten_for_new_corpus(tmp_path):
    project, prep, queue = _common_preparation(tmp_path)
    before = project.read_bytes()
    contract = project.parent / "features/started/contract.json"
    write_json(contract, {"fixture_already_started": True})
    with pytest.raises(ValueError, match="frozen"):
        studio_data.admit_materials(project, {"recipe_version": 6, "evaluation_source_scope": "common_30h_only",
            "evaluation_materials_policy": common_evaluation_policy(), "review_queues": [queue]})
    assert project.read_bytes() == before
