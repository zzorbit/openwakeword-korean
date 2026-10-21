"""Tiny procedural fixtures verify wiring, not full-scale corpus downloads."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword import studio_data as data
from korean_openwakeword import studio_steps as steps


@pytest.mark.parametrize("preset", ["check", "trial", "production"])
@pytest.mark.parametrize("policy", ["automatic_v1", "automatic_v2"])
def test_new_acquisition_has_explicit_offline_plan_and_keeps_legacy_recipe(preset, policy, monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: pytest.fail("offline preview"))
    legacy = data.material_recipe(preset)
    current = data.material_recipe(preset, material_acquisition_policy=policy)
    assert "material_acquisition_policy" not in legacy
    assert current["material_acquisition_policy"] == policy
    assert not current["shortages"]
    ids = {row["id"] for row in current["assets"]}
    components = current["automatic_source_plan"]["components"]
    if preset == "check":
        assert ids == {row["id"] for row in legacy["assets"]}
        assert components == []
    elif preset == "trial":
        assert "generic" in ids and "zeroth" not in ids
        assert {row["provider"] for row in components} == {"yodas2"}
        assert components[0]["target_hours"] == pytest.approx(2000 / 3)
        assert legacy["shortages"]
    else:
        assert "generic" not in ids and "zeroth" not in ids
        assert {row["provider"] for row in components} == {"audioset_unbalanced", "mls", "fma_full", "yodas2"}
        assert [row["target_hours"] for row in components] == [3000, 3000, 1000, 3000]
        assert legacy["shortages"]
        assert current["automatic_space_plan"]["minimum_total_bytes"] > 110 * 1024**3
    assert current["estimated_workspace_bytes"] == current["automatic_space_plan"]["estimated_total_bytes"]
    assert steps.preset_settings(preset)["material_acquisition_policy"] == "automatic_v2"


@pytest.mark.parametrize("policy", [None, "automatic_v1", "automatic_v2"])
def test_worker_preserves_saved_acquisition_policy(tmp_path, monkeypatch, policy):
    project = create_project(tmp_path / "bora", "보라야", "bora", positive_originals=1000,
                             augmentation_rounds=1, updates=10000, generic_hours=2000, korean_hours=1)
    captured = []
    monkeypatch.setattr(data, "prepare_materials", lambda *args, **kwargs: captured.append(kwargs) or {"review_queues": []})
    settings = steps.preset_settings("check")
    if policy is None:
        settings.pop("material_acquisition_policy")
    else:
        settings["material_acquisition_policy"] = policy
    steps.run_stage(project, tmp_path, settings, "data_download")
    assert captured[0].get("material_acquisition_policy") == policy
    assert ("material_acquisition_policy" in captured[0]) is (policy is not None)


def _write_index(directory, dataset, category, language, seed, seconds=2):
    directory.mkdir(parents=True, exist_ok=True)
    audio = directory / "source.wav"
    sf.write(audio, np.random.default_rng(seed).normal(0, .03, 16000 * seconds), 16000, subtype="PCM_16")
    row = {"path": audio.name, "sha256": digest(audio), "dataset_id": dataset,
           "source_url": "https://example.test/" + dataset, "license": "procedural-test-fixture",
           "category": category, "category_basis": "explicit procedural fixture", "language": language,
           "speech": category == "competing_speech", "canonical_source_id": "fixture:" + dataset}
    if category == "rir":
        row["canonical_room_id"] = "fixture:room:" + dataset
    index = directory.parent / (dataset + ".jsonl")
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return index


def test_production_uses_qualified_raw_sources_without_legacy_generic_or_zeroth(tmp_path, monkeypatch):
    project = create_project(tmp_path / "bora", "보라야", "bora", generic_hours=7 / 3600, korean_hours=2 / 3600)
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(data, "_download", lambda *args, **kwargs: pytest.fail("production has no bounded base archives"))
    monkeypatch.setattr(data, "ensure_evaluation_materials", lambda p, w, prepared, progress=None: prepared)
    mit = workspace / "shared/mit_fixture"
    mit_index = _write_index(mit, "mit_rir", "rir", "unknown", 9)
    monkeypatch.setattr(data, "_fetch_mit", lambda *args, **kwargs: mit)
    import korean_openwakeword.preparation as preparation
    real_index = preparation.create_source_index
    def source_index(directory, *, output, dataset, **kwargs):
        if dataset == "mit_rir":
            Path(output).write_text(mit_index.read_text(encoding="utf-8"), encoding="utf-8")
            return {"path": str(output), "sha256": digest(output)}
        return real_index(directory, output=output, dataset=dataset, **kwargs)
    monkeypatch.setattr(preparation, "create_source_index", source_index)
    calls = []
    def automatic(path, work, *, generic_hours, korean_hours, preset, progress, qualify_source,
                  material_acquisition_policy):
        assert material_acquisition_policy == "automatic_v2"
        calls.append((generic_hours, korean_hours, preset))
        sources = []
        for number, (name, component, category, language, pool, seconds) in enumerate((
            ("audioset", "audioset_unbalanced", "environment", "unknown", "generic_negative", 3),
            ("mls", "mls", "competing_speech", "en", "generic_negative", 3),
            ("fma", "fma_full", "music", "unknown", "generic_negative", 1),
            ("yodas2", "yodas2", "competing_speech", "ko", "korean_negative", 2))):
            directory = workspace / "shared" / name
            index = _write_index(directory, name, category, language, number, seconds)
            source = {"id": name, "directory": directory, "adapter": "manifest", "source_index": index,
                      "target_hours": None, "validation_fraction": 0, "pools": [pool], "component": component}
            source["prepared"] = qualify_source(source)
            sources.append(source)
        return {"sources": sources, "plan": {"fixture_only": True}, "receipts": [], "artifacts": [],
                "component_actual_hours": {}, "shortages": [], "actual_download_bytes": 0}
    from korean_openwakeword import automatic_sources
    monkeypatch.setattr(automatic_sources, "prepare_automatic_sources", automatic)
    settings = steps.preset_settings("production")
    settings.update(generic_hours=7 / 3600, korean_hours=2 / 3600)
    downloaded = steps.run_stage(project, workspace, settings, "data_download")
    assert downloaded["state"] == "completed"
    prepared = downloaded["result"]
    assert calls == [(7 / 3600, 2 / 3600, "full")]
    assert {row["id"] for row in prepared["preparations"]} == {"audioset", "mls", "fma", "yodas2", "mit_rir"}
    assert prepared["generic_feature_import"]["imported"] is False
    assert prepared["candidate_unique_hours"] == pytest.approx({"generic_negative": 7 / 3600, "korean_negative": 2 / 3600})
    assert not prepared["shortages"]
    assert prepared["human_listened_count"] == 0 and prepared["training_admitted"] is False
    assert prepared["automatic_source_preparation"]["actual_download_bytes"] == 0
    assert not read_rows(project.parent / load_project(project)[1]["manifests"]["noise"])
    assert all("noise" in q["pools"] for q in prepared["review_queues"] if q["id"] != "materials_mit_rir")
    assert {q["title"] for q in prepared["review_queues"] if q.get("automatic_component")} == {
        "AudioSet 일반 오디오 확인", "MLS 다국어 낭독 확인", "FMA 음악 확인", "YODAS2 한국어 음성 확인"}
    assert all(not row.get("training_admitted", False) for q in prepared["review_queues"] for row in read_rows(q["pool_manifest"]))
    with pytest.raises(ValueError, match="Saved representative"):
        data.admit_materials(project, prepared)
    context = {"results": {"data_download": downloaded}, "inputs": {"review_answers": {}, "review_results": {}}}
    assert steps.run_stage(project, workspace, settings, "data_review", context)["state"] == "waiting_review"
    for queue in prepared["review_queues"]:
        previews = read_rows(queue["queue"])
        answers = project.parent / "review" / (queue["id"] + ".json")
        # Explicit test dispositions exercise admission; no real-source human
        # listening is claimed by this procedural fixture.
        write_json(answers, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "fixture",
            "source_scope": "Human dispositions entered through the local review screen",
            "decisions": {row["sample_id"]: {"decision": "keep", "listened": True} for row in previews},
            "whole_pool_policy_approved": True, "source_manifest_sha256": digest(queue["pool_manifest"]),
            "representative_queue_sha256": digest(queue["queue"]), "queue_sha256": digest(queue["queue"]),
            "queued_audio_sha256": {row["sample_id"]: row["sha256"] for row in previews}})
        context["inputs"]["review_answers"][queue["id"]] = {"path": str(answers), "sha256": digest(answers)}
        context["inputs"]["review_results"][queue["id"]] = {"state": "review_saved", "admission_pending": True}
    assert steps.run_stage(project, workspace, settings, "data_review", context)["state"] == "completed"
    connected = steps.run_stage(project, workspace, settings, "data_connect", context)
    assert connected["state"] == "completed"
    actual = connected["result"]
    assert actual["admitted_unique_hours"] == pytest.approx(prepared["candidate_unique_hours"])
    assert actual["generic_feature_import"]["mode"] == "raw_generic"
    assert not actual["generic_feature_import"]["imported"]
    assert set(actual["pools_ready"]) == {"generic_negative", "korean_negative", "noise", "rir"}
    config = load_project(project)[1]
    negatives = read_rows(project.parent / config["manifests"]["generic_negative"]) + read_rows(project.parent / config["manifests"]["korean_negative"])
    noises = read_rows(project.parent / config["manifests"]["noise"])
    assert {row["sha256"] for row in noises} == {row["sha256"] for row in negatives}
    assert actual["Final_accessed"] is False
    assert actual["automatic_source_admission"]["target_met"]


def test_one_oversized_component_cannot_replace_other_admitted_source_targets(tmp_path):
    root = tmp_path
    preparation = root / "preparation.json"
    write_json(preparation, {"procedural_fixture": True})
    source = root / "sources.jsonl"
    source.write_text(json.dumps({"sha256": "a" * 64, "canonical_source_id": "fixture:as",
                                  "start_s": 0, "end_s": 8, "duration_s": 8}) + "\n", encoding="utf-8")
    admission = root / "admission.json"
    write_json(admission, {"request": {"preparation": {"sha256": digest(preparation)}},
                           "source_manifests": {"generic_negative": {"path": str(source), "sha256": digest(source)}}})
    prepared = {"automatic_source_plan": data._automatic_source_plan("full", 7 / 3600, 2 / 3600),
                "review_queues": [{"preparation": str(preparation), "automatic_component": "audioset_unbalanced"}]}
    coverage = data._automatic_admission_coverage(root, prepared, [str(admission)])
    assert coverage["admitted_unique_hours"]["audioset_unbalanced"] == 8 / 3600
    assert not coverage["target_met"]
    assert {item.split(":", 1)[0] for item in coverage["shortages"]} == {"mls", "fma_full", "yodas2"}


def test_automatic_space_shortage_stops_before_any_base_download(tmp_path, monkeypatch):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    monkeypatch.setattr(data.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 100 * 1024**3})())
    monkeypatch.setattr(data, "_download", lambda *args, **kwargs: pytest.fail("download before complete-workload preflight"))
    monkeypatch.setattr(data, "_fetch_mit", lambda *args, **kwargs: pytest.fail("RIR download before preflight"))
    with pytest.raises(ValueError, match="자료 준비 공간이 부족.*GiB.*부족"):
        data.prepare_materials(project, tmp_path / "workspace", preset="production", material_acquisition_policy="automatic_v1")
    report = read_json(project.parent / ".studio/materials/automatic_base_preflight.json")
    assert report["required_free_bytes"] > report["available_bytes"]
    assert not report["audio_download_started"]
