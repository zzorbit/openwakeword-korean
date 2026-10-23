"""Studio duration display uses saved metadata, never scans audio on polling."""
from copy import deepcopy

from korean_openwakeword.studio import StudioManager
from korean_openwakeword.studio_steps import common_evaluation_policy


def _describe(tmp_path, *, version=4, candidate=None, admitted=None):
    manager = object.__new__(StudioManager)
    manager._implementation_id = "fixture-implementation"
    manager._timing_receipts = {}
    manager.review_descriptors = lambda state: []
    state = {"id": "fixture", "wakeword": "보라야", "preset": "check", "status": "paused",
        "implementation_id": manager._implementation_id,
        "project": str(tmp_path / "not_opened/project.json"), "index": 0,
        "plan": [{"id": "data_review", "label": "검수", "kind": "review"}],
        "settings": {"evaluation_policy": common_evaluation_policy(version)},
        "results": {"data_download": {"result": candidate or {}},
                    "data_connect": {"result": admitted or {}}}}
    before = deepcopy(state)
    result = manager.describe(state)
    assert state == before
    assert not (tmp_path / "not_opened").exists()
    return result


def _material(core, margin, *, eligible=None):
    result = {"evaluation_material_coverage": {
        "core_coverage": {"actual_total_hours": core, "actual_hours_by_bucket": {"korean_read": 3}},
        "actual_total_hours": core + margin, "initialization_margin_hours": margin,
        "actual_hours_by_bucket": {"korean_read": 3 + margin}}}
    if eligible is not None:
        result["negative_duration_budget"] = {"eligible_seconds": eligible * 3600}
    return result


def test_candidate_and_admitted_core_margin_and_planned_time_are_separate(tmp_path):
    view = _describe(tmp_path, candidate=_material(30, .10, eligible=30.01),
                     admitted=_material(29.9, .08, eligible=29.88))
    materials = view["evaluation_materials"]
    assert materials["candidate"]["actual_total_hours"] == 30.1
    assert materials["duration"] == {
        "candidate": {"core_hours": 30, "initialization_margin_hours": .10, "eligible_hours": 30.01},
        "admitted": {"core_hours": 29.9, "initialization_margin_hours": .08, "eligible_hours": 29.88}}
    assert materials["candidate"]["core_coverage"]["actual_hours_by_bucket"]["korean_read"] == 3
    assert "results" not in view  # A duration plan is not an actual model result.


def test_unknown_budget_or_admission_is_not_reported_as_zero_or_approved(tmp_path):
    view = _describe(tmp_path, candidate=_material(30, .1))
    assert view["evaluation_materials"]["duration"] == {
        "candidate": {"core_hours": 30, "initialization_margin_hours": .1, "eligible_hours": None},
        "admitted": None}
    view = _describe(tmp_path, candidate={"evaluation_material_coverage": {"actual_total_hours": 30.1}})
    assert view["evaluation_materials"]["duration"]["candidate"] == {
        "core_hours": None, "initialization_margin_hours": None, "eligible_hours": None}


def test_old_policy_views_keep_their_original_duration_contract(tmp_path):
    result = _describe(tmp_path, version=3, candidate=_material(30, .1, eligible=30))
    assert set(result["evaluation_materials"]) == {"candidate", "admitted", "scope"}


def test_saved_candidate_budget_survives_preparation_and_studio_result_metadata(tmp_path, monkeypatch):
    from korean_openwakeword import evaluation_materials, evaluation_rir, studio_data, studio_evaluation_data
    budget = {"eligible_seconds": 108001.6, "eligible_hours": 108001.6 / 3600,
              "scope": "metadata fixture, not scored audio"}
    coverage = _material(30, .1)["evaluation_material_coverage"]
    source = {"sample_id": "metadata_only", "canonical_source_id": "fixture:metadata_only",
        "canonical_recording_id": "fixture:metadata_only", "sha256": "1" * 64,
        "original_sha256": "1" * 64, "path": str(tmp_path / "not_read.wav"),
        "category": "environment", "eval_type": "rain", "eval_bucket": "environment", "duration_s": 1}
    for module in (studio_data, studio_evaluation_data):
        monkeypatch.setattr(module, "load_project", lambda project: (tmp_path, {"wakeword": "보라야"}))
    monkeypatch.setattr(evaluation_materials, "evaluation_material_catalog", lambda: {"fixture_catalog": True})
    monkeypatch.setattr(evaluation_rir, "common_rir_catalog", lambda: {"fixture_rirs": True})
    monkeypatch.setattr(evaluation_materials, "prepare_common_sources", lambda *a, **k: {
        "source_rows": [source], "target_met": True, "coverage": coverage, "duration_budget": budget})
    monkeypatch.setattr(evaluation_rir, "prepare_common_rirs", lambda *a, **k: {"source_rows": [], "coverage": {}})
    monkeypatch.setattr(studio_evaluation_data, "assert_held_out", lambda *a, **k: {"fixture_only": True})
    prepared = {"evaluation_materials_policy": common_evaluation_policy()}
    # The waveform helper is intentionally mocked; this checks only the saved
    # metadata handoff used by UI polling, not audio quality or scored hours.
    receipt = studio_evaluation_data.prepare_common_evaluation_materials("fixture", tmp_path, prepared)
    assert receipt["negative_duration_budget"] == budget
    monkeypatch.setattr(studio_evaluation_data, "prepare_common_evaluation_materials", lambda *a, **k: receipt)
    result = studio_data.ensure_evaluation_materials("fixture", tmp_path, prepared)
    assert result["negative_duration_budget"] == budget
    assert result["evaluation_material_coverage"] == coverage
