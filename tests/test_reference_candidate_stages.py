"""The merged chapter prepares slots, never an unused baseline voice bank."""
from copy import deepcopy

import pytest

from korean_openwakeword import studio_steps as steps, tts
from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.project import create_project


@pytest.mark.parametrize("preset", ["check", "trial", "production"])
def test_candidate_preparation_never_loads_tts_or_generates_baseline_audio(tmp_path, monkeypatch, preset):
    settings = steps.preset_settings(preset)
    settings["reference_selection_policy"] = "candidate_pool_v3"
    project = create_project(tmp_path / "candidate", "보라야", "candidate",
        positive_originals=settings["originals"], augmentation_rounds=settings["rounds"],
        updates=settings["updates"], generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    root = project.parent
    approval_before = (root / "approval.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata preparation must not load a model or generate baseline voices")
    monkeypatch.setattr(tts, "_load_model", forbidden)
    monkeypatch.setattr(tts, "generate_references", forbidden)
    prepared = steps.run_stage(project, tmp_path, settings, "references")
    assert prepared["state"] == "completed" and prepared["result"]["generated"] == 0
    assert prepared["result"]["slot_count"] == settings["train_references"] + 30
    assert prepared["result"]["group_count"] == 20
    checked = steps.run_stage(project, tmp_path, settings, "references_review",
        {"results": {"references": prepared}})
    assert checked["state"] == "completed"
    assert checked["result"] == prepared["result"]
    assert checked["result"].get("state") != "accepted"
    assert not list((root / "data/references").rglob("*.wav"))
    assert not list((root / "data/references").rglob("references.jsonl"))
    assert (root / "approval.json").read_bytes() == approval_before
    bank = read_json(root / "voices.json")
    bank["families"][0]["description"] += " changed"
    write_json(root / "voices.json", bank)
    with pytest.raises(ValueError):
        steps.run_stage(project, tmp_path, settings, "references_review", {"results": {"references": prepared}})


def test_candidate_policy_requires_the_reviewed_call_and_technical_admission_contract():
    settings = steps.preset_settings()
    settings["reference_selection_policy"] = "candidate_pool_v3"
    steps._validate_settings(settings)
    for key, value in (("reference_selection_policy", "unknown"),
                       ("reference_admission_policy", None), ("pilot_review_policy", None)):
        changed = deepcopy(settings)
        changed[key] = value
        with pytest.raises(ValueError):
            steps._validate_settings(changed)
    labels = {stage["id"]: stage["label"] for stage in steps.workflow_plan(settings)}
    assert labels["references"] == "기준 목소리 후보 구성"
    assert labels["references_review"] == "기준 목소리 분류 확인"
