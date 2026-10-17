"""Final may reuse neither Train nor common-evaluation source material."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, read_rows, require_final_lock, write_json
from korean_openwakeword.data import audit_project
from korean_openwakeword.final_integrity import audit_independent_final, freeze_input_bindings
from korean_openwakeword.paired import prepare_evaluation
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.runtime import sample_bundle
from korean_openwakeword.source_identity import component_lineage


def rows(path, values):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


def audio_row(root, name, split, *, seed=1, category="environment", seconds=1):
    path = root / "data" / (name + ".wav")
    sf.write(path, np.random.default_rng(seed).normal(0, .03, round(16000 * seconds)), 16000, subtype="PCM_16")
    return {"sample_id": name, "canonical_source_id": "fixture:" + name,
            "path": str(path), "sha256": digest(path), "split": split,
            "category": category, "provenance": "public_dataset",
            "source_url": "https://example.test/" + name, "license": "procedural-test-fixture"}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("KOWW_CACHE_ROOT", str(tmp_path / "cache"))
    return create_project(tmp_path / "project", "춘식아", "final_fixture")


def connect(project, pool, values):
    root, config = load_project(project)
    rows(root / config["manifests"][pool], values)


@pytest.mark.parametrize("known_split,known_pool", [("train", "noise"), ("validation", "noise_validation")])
def test_canonical_recording_alias_cannot_hide_known_source(project, known_split, known_pool):
    root = project.parent
    known = audio_row(root, "known", known_split, seed=3)
    final = audio_row(root, "independent_name", "final", seed=4)
    final["canonical_recording_id"] = known["canonical_source_id"]
    final["dataset_id"] = "another-dataset-does-not-establish-independence"
    connect(project, known_pool, [known]); connect(project, "noise_final", [final])
    for report in (audit_project(project, check_files=False), audit_independent_final(project, check_files=False)):
        assert not report["passed"]
        assert any(leak["key"] == "source" for leak in report["leaks"])


@pytest.mark.parametrize("field,key", [("original_sha256", "sha256"), ("normalized_pcm_sha256", "pcm")])
def test_original_or_pcm_hash_survives_renaming_and_reencoding(project, field, key):
    root = project.parent
    common = audio_row(root, "common", "validation", seed=3)
    final = audio_row(root, "final", "final", seed=4)
    common[field] = final[field] = identity("same-unmodified-original")
    connect(project, "noise_validation", [common]); connect(project, "noise_final", [final])
    report = audit_independent_final(project, check_files=False)
    assert not report["passed"] and any(leak["key"] == key for leak in report["leaks"])


@pytest.mark.parametrize("field,key", [("voice_family_id", "family"), ("canonical_speaker_id", "speaker"),
                                      ("canonical_artist_id", "artist"), ("canonical_room_id", "room")])
def test_nested_final_component_lineage_is_compared_with_known_inputs(project, field, key):
    root = project.parent
    common = audio_row(root, "common", "validation", seed=3)
    common[field] = "provider:shared-identity"
    final = audio_row(root, "final", "final", seed=4)
    component = audio_row(root, "component", "final", seed=5)
    component[field] = common[field]
    final["components"] = [component_lineage(component, "background")]
    connect(project, "noise_validation", [common]); connect(project, "noise_final", [final])
    report = audit_independent_final(project, check_files=False)
    assert not report["passed"] and any(leak["key"] == key for leak in report["leaks"])


def test_distinct_wav_containers_with_identical_pcm_are_not_independent(project):
    root = project.parent
    common = audio_row(root, "common", "validation", seed=9)
    final = audio_row(root, "final", "final", seed=9)
    # Legal trailing RIFF metadata leaves decoded PCM unchanged but changes SHA.
    with Path(final["path"]).open("ab") as stream:
        stream.write(b"JUNK\x04\x00\x00\x00test")
    final["sha256"] = digest(final["path"])
    assert final["sha256"] != common["sha256"]
    connect(project, "noise_validation", [common]); connect(project, "noise_final", [final])
    report = audit_independent_final(project)
    assert not report["passed"] and any("PCM duplicates" in message for message in report["errors"])
    assert report["independent_STT_runs"] == 0


def test_genuinely_distinct_sources_pass_with_explicit_unknown_mapping_and_pcm_cache(project):
    root = project.parent
    connect(project, "noise", [audio_row(root, "train", "train", seed=9)])
    connect(project, "noise_validation", [audio_row(root, "common", "validation", seed=10)])
    connect(project, "noise_final", [audio_row(root, "final", "final", seed=11, category="competing_speech")])
    first = audit_independent_final(project)
    assert first["passed"] and first["pcm"]["known_same_length_sources_checked"] == 2
    assert first["unknown_metadata"]["speaker_identity"] == 1
    assert first["generic_precomputed_original_mapping_available"] is False
    assert first["fully_proven_independent"] is False
    second = audit_independent_final(project)
    assert second["passed"] and second["pcm"]["bytes_read"] == 0 and second["pcm"]["cache_hits"] == 3


def test_final_mixture_needs_original_background_lineage_not_only_different_wav_hash(project):
    final = audio_row(project.parent, "mixture", "final", seed=9)
    final.update(kind="positive", provenance="qwen_tts", voice_family_id="qwen:final",
                 human_reference_used=False, calls=[{"start_s": .2, "end_s": .8}], noise_source_id="unknown")
    report = audit_independent_final(project, extra_rows=[final], check_files=False)
    assert not report["passed"] and any("call and background" in message for message in report["errors"])


def test_final_rir_requires_original_lineage_when_applied(project):
    final = audio_row(project.parent, "mixture", "final", seed=9)
    final.update(rir_applied=True, rir_source_id="changed-name")
    report = audit_independent_final(project, extra_rows=[final], check_files=False)
    assert not report["passed"] and any("RIR" in message for message in report["errors"])


def test_wrong_final_split_and_missing_public_provenance_do_not_pass(project):
    final = audio_row(project.parent, "final", "validation", seed=9)
    assert not audit_independent_final(project, extra_rows=[final], check_files=False)["passed"]
    final["split"] = "final"
    final.pop("source_url")
    report = audit_independent_final(project, extra_rows=[final], check_files=False)
    assert not report["passed"] and any("provenance required" in message for message in report["errors"])


def test_a_final_wrapper_cannot_relabel_a_train_component_as_independent(project):
    component = audio_row(project.parent, "train_component", "train", seed=12)
    final = audio_row(project.parent, "final_wrapper", "final", seed=13)
    final["components"] = [component_lineage(component, "background")]
    report = audit_independent_final(project, extra_rows=[final], check_files=False)
    assert not report["passed"] and any("reserved final split" in message for message in report["errors"])


def _lock(project, tmp_path):
    root = project.parent
    bundle = tmp_path / "bundle"
    shutil.copytree(sample_bundle(), bundle)
    config = read_json(project)
    config["runtime"] = deepcopy(read_json(bundle / "manifest.json")["runtime"])
    write_json(project, config)
    report = root / "reports" / "fixture" / "report.json"
    write_json(report.parent / "clips.json", [])
    write_json(report, {"stage": "development"})
    bindings = freeze_input_bindings(project, report)
    write_json(root / "final_lock.json", {"schema_version": 2, "bundle": str(bundle),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"), "project_config_sha256": digest(project),
        "input_bindings": bindings})
    return bundle, bindings


def test_freeze_binds_train_common_and_external_manifest_but_leaves_empty_final_for_later_generation(project, tmp_path):
    root = project.parent
    external = root / "data" / "external_common.jsonl"
    rows(external, [])
    report = root / "reports" / "fixture" / "report.json"
    write_json(report.parent / "clips.json", [])
    write_json(report, {"evaluation_manifest": {"path": str(external), "sha256": digest(external)},
                        "manifest_sha256": digest(external)})
    bindings = freeze_input_bindings(project, report)
    assert "manifest:noise" in bindings and "manifest:noise_validation" in bindings
    assert "manifest:noise_final" not in bindings and "manifest:positive_final" not in bindings
    assert bindings["external_development_manifest"]["sha256"] == digest(external)
    final = audio_row(root, "final", "final", seed=9)
    connect(project, "noise_final", [final])
    assert "manifest:noise_final" in freeze_input_bindings(project, report)


def test_changed_common_input_after_freeze_cannot_open_final(project, tmp_path):
    _, bindings = _lock(project, tmp_path)
    assert require_final_lock(project)["schema_version"] == 2
    Path(bindings["manifest:noise_validation"]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed after freeze"):
        require_final_lock(project)


def test_frozen_external_common_clips_cannot_be_forgotten_when_manifest_was_not_connected(project, tmp_path):
    bundle, bindings = _lock(project, tmp_path)
    common = audio_row(project.parent, "external_common", "validation", seed=22)
    # A standalone --manifest Development run can live outside project manifests.
    clips = Path(bindings["development_clips"]["path"])
    write_json(clips, [common])
    lock = read_json(project.parent / "final_lock.json")
    lock["input_bindings"]["development_clips"]["sha256"] = digest(clips)
    write_json(project.parent / "final_lock.json", lock)
    final = audio_row(project.parent, "renamed_final", "final", seed=23)
    final["canonical_recording_id"] = common["canonical_source_id"]
    report = audit_independent_final(project, extra_rows=[final], check_files=False)
    assert not report["passed"] and any("source" in message for message in report["errors"])


def test_unchanged_manifest_cannot_hide_changed_frozen_model_file(project, tmp_path):
    bundle, _ = _lock(project, tmp_path)
    member = read_json(bundle / "manifest.json")["members"][0]
    with (bundle / member["path"]).open("ab") as stream:
        stream.write(b"not-the-frozen-model")
    with pytest.raises(ValueError, match="Missing or changed bound file"):
        require_final_lock(project)


def test_paired_outputs_carry_original_and_background_hashes_for_later_final_audit(project):
    root = project.parent
    positive = audio_row(root, "call", "validation", seed=99, category="clean", seconds=.6)
    positive.update(provenance="qwen_tts", text="춘식아", human_reference_used=False,
                    voice_family_id="qwen:validation-only", call_start_s=0, call_end_s=.6)
    backgrounds = [audio_row(root, category, "validation", seed=i + 30, category=category, seconds=5)
                   for i, category in enumerate(("environment", "music", "competing_speech"))]
    connect(project, "positive_validation", [positive]); connect(project, "noise_validation", backgrounds)
    result = prepare_evaluation(project, root / "reports" / "paired", originals=1, noise_sections=1,
                                noise_selection="balanced_sources")
    outputs = read_rows(result["manifest"])
    clean = next(row for row in outputs if row["category"] == "clean")
    mixed = next(row for row in outputs if row["category"] != "clean")
    assert clean["components"] == [component_lineage(positive, "call")]
    assert {component["role"] for component in mixed["components"]} == {"call", "background"}
    background = next(component for component in mixed["components"] if component["role"] == "background")
    assert background["sha256"] in {row["sha256"] for row in backgrounds}
    assert audit_project(project, extra_rows=outputs)["passed"]
