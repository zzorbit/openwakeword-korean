"""Held-out noise coverage without inflating the number of paired tests."""
from collections import Counter
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json
from korean_openwakeword.data import CATEGORIES
from korean_openwakeword.paired import _balanced_pool, _balanced_choice, _balanced_noise_window, _assignment_order, prepare_evaluation
from korean_openwakeword.audio import active_speech_mask, measured_snr, noise_scale
from korean_openwakeword.project import create_project


def rows(path, values):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")


def noise(category, subtype, number):
    key = f"{category}:{subtype}:{number}"
    return {"sample_id": key, "canonical_source_id": key, "sha256": identity(key),
            "path": key + ".wav", "category": category, "eval_type": subtype, "split": "validation"}


def synthetic_pool():
    return [noise(category, subtype, i) for category in CATEGORIES
            for subtype in ("type_a", "type_b", "type_c") for i in range(4)]


def test_balancing_types_then_sources_is_uniform_and_deterministic():
    pools = _balanced_pool(synthetic_pool())
    for category in CATEGORIES:
        selected = [_balanced_choice(pools, category, slot, 1234)[1] for slot in range(120)]
        assert set(Counter(g["eval_type"] for g in selected).values()) == {40}
        assert set(Counter(g["id"] for g in selected).values()) == {10}
        assert [g["id"] for g in selected] == [_balanced_choice(pools, category, slot, 1234)[1]["id"] for slot in range(120)]


def test_source_aliases_and_duplicate_intervals_cannot_gain_extra_votes():
    original = synthetic_pool()
    aliases = [{**original[0], "sample_id": f"duplicate{i}"} for i in range(20)]
    before = _balanced_pool(original)
    after = _balanced_pool(list(reversed(original + aliases)))
    for category in CATEGORIES:
        assert [_balanced_choice(before, category, i, 42)[1]["id"] for i in range(100)] == [
            _balanced_choice(after, category, i, 42)[1]["id"] for i in range(100)]
    aliased = next(g for gs in after["environment"].values() for g in gs if g["input_rows"] > 1)
    assert aliased["input_rows"] == 21 and len(aliased["members"]) == 1


def test_exact_same_audio_under_new_source_id_is_one_source():
    original = synthetic_pool()
    alias = {**original[0], "sample_id": "alias", "canonical_source_id": "other-source-name"}
    pools = _balanced_pool(original + [alias])
    assert sum(len(gs) for gs in pools["environment"].values()) == 12


def test_parent_recording_segments_are_not_independent_sources():
    original = synthetic_pool()
    first = original[0]
    first["canonical_recording_id"] = "real-parent-recording"
    segment = {**first, "sample_id": "second-segment", "canonical_source_id": "another-segment",
               "sha256": identity("different-segment-waveform"), "start_s": 2, "end_s": 5}
    pools = _balanced_pool(original + [segment])
    group = next(g for gs in pools["environment"].values() for g in gs if len(g["file_sha256"]) == 2)
    assert len(group["members"]) == 2
    assert sum(len(gs) for gs in pools["environment"].values()) == 12


def test_same_source_with_conflicting_type_metadata_is_rejected():
    original = synthetic_pool()
    with pytest.raises(ValueError, match="conflicting"):
        _balanced_pool(original + [{**original[0], "sample_id": "alias", "eval_type": "different_label"}])


def test_hash_assignment_breaks_periodic_voice_and_background_type_alignment():
    positives = [{"sample_id": f"validation_{i:06d}"} for i in range(100)]
    order = _assignment_order(positives, 20260928)
    assert order == _assignment_order(list(reversed(positives)), 20260928)
    assert sorted(order.values()) == list(range(100))
    # Alternating reference voices and two background types must not be
    # mechanically paired by their shared even/odd generation-slot cycle.
    assignments = Counter((i % 2, order[r["sample_id"]] % 2) for i, r in enumerate(positives))
    assert len(assignments) == 4 and all(n > 10 for n in assignments.values())


def test_early_single_knock_is_placed_once_with_zero_padding_and_same_snr_basis(tmp_path):
    path = tmp_path / "early_knock.wav"
    values = np.zeros(6 * 16000)
    t = np.arange(1600) / 16000
    values[1600:3200] = .3 * np.sin(2 * np.pi * 700 * t) * np.exp(-20 * t)
    sf.write(path, values, 16000, subtype="PCM_16")
    noise = {"source_id": "real_early_knock_fixture", "path": str(path), "sha256": digest(path), "category": "environment"}
    speech = np.r_[np.zeros(38400), .1 * np.sin(np.arange(9600) * .07), np.zeros(16000)]
    mask = active_speech_mask(speech, 2.4, 3.0)
    samples, offset, construction = _balanced_noise_window(tmp_path, noise, speech, mask, np.random.default_rng(10), (20, 15, 10, 5, 0), {})
    assert construction["placement"] == "zero_padded_event_alignment" and construction["offset_attempts"] == 16
    assert construction["repeat_count"] == 1 and construction["shift_samples"] > 0
    assert np.count_nonzero(samples) == np.count_nonzero(sf.read(path)[0])
    second, offset2, construction2 = _balanced_noise_window(tmp_path, noise, speech, mask, np.random.default_rng(10), (20, 15, 10, 5, 0), {})
    assert np.array_equal(samples, second) and offset == offset2 and construction == construction2
    for snr in (20, 15, 10, 5, 0):
        assert abs(measured_snr(speech, samples * noise_scale(speech, samples, mask, snr), mask) - snr) < 1e-8


@pytest.fixture
def project(tmp_path):
    project = create_project(tmp_path / "paired", "보라야", "bora")
    root, config = project.parent, read_json(project)
    positives = []
    for i in range(12):
        path = root / "data" / f"word{i}.wav"
        t = np.arange(9600) / 16000
        sf.write(path, .1 * np.sin(2 * np.pi * (210 + 11 * i) * t), 16000, subtype="PCM_16")
        positives.append({"sample_id": f"validation_{i:03d}", "source_id": f"qwen:validation:{i}",
            "path": str(path), "sha256": digest(path), "split": "validation", "provenance": "qwen_tts",
            "human_reference_used": False, "text": "보라야", "voice_family_id": f"validation_voice{i}",
            "call_start_s": 0, "call_end_s": .6})
    noises = []
    for category, types in (("environment", ("fan", "road", "impact")), ("music", ("instrumental", "electronic")),
                             ("competing_speech", ("korean_dialogue", "korean_radio"))):
        for subtype in types:
            for number in range(2):
                path = root / "data" / f"{subtype}{number}.wav"
                samples = np.random.default_rng(len(noises) + 500).normal(0, .04, 8 * 16000)
                sf.write(path, samples, 16000, subtype="PCM_16")
                noises.append({"sample_id": subtype + str(number), "canonical_source_id": subtype + ":" + str(number),
                    "path": str(path), "sha256": digest(path), "start_s": 0, "end_s": 8, "split": "validation",
                    "category": category, "eval_type": subtype, "provenance": "public_dataset",
                    "source_url": "https://example.test/procedural-paired-fixture", "license": "isolated-test-fixture"})
    rows(root / config["manifests"]["positive_validation"], positives)
    rows(root / config["manifests"]["noise_validation"], noises)
    return project


def test_real_paired_wavs_cover_sources_types_and_snr_without_extra_sessions(project):
    result = prepare_evaluation(project, project.parent / "reports/balanced", originals=12, noise_sections=1,
                                noise_selection="balanced_sources")
    assert result["audio_files"] == 12 * 16
    entries = read_rows(result["manifest"])
    coverage = read_json(result["noise_coverage"]["path"])
    assert coverage["originals_used"] == 12
    assert coverage["recipe"]["noise_selection"] == "balanced_sources_v1"
    assert all(c["samples"] == 12 for c in coverage["condition_counts"])
    for category, count in (("environment", 6), ("music", 4), ("competing_speech", 4)):
        actual = coverage["categories"][category]
        assert actual["effective_unique_sources"] == actual["used_sources"] == count
        assert actual["all_sources_used"] is True
        assert actual["source_selections"] == 12
        assert sum(t["source_selections"] for t in actual["types"]) == 12
        for snr in (20, 15, 10, 5, 0):
            assert len([r for r in entries if r["category"] == category and r["snr_db"] == snr]) == 12
    for original in {r["original_id"] for r in entries}:
        for category in CATEGORIES:
            paired = [r for r in entries if r["original_id"] == original and r["category"] == category]
            assert len({(r["noise_source_id"], r["noise_offset_s"], r["noise_eval_type"], r["noise_source_group_id"]) for r in paired}) == 1
            assert all(abs(r["measured_snr_db"] - r["snr_db"]) < 1e-8 for r in paired)


def test_audio_bits_are_invariant_to_manifest_order_and_duplicate_aliases(project):
    root, config = project.parent, read_json(project)
    first = prepare_evaluation(project, root / "reports/first", originals=6, noise_sections=1, noise_selection="balanced_sources")
    positives = root / config["manifests"]["positive_validation"]
    noises = root / config["manifests"]["noise_validation"]
    rows(positives, list(reversed(read_rows(positives))))
    noise_rows = read_rows(noises)
    rows(noises, list(reversed(noise_rows + [{**noise_rows[0], "sample_id": "duplicate_alias"}])))
    second = prepare_evaluation(project, root / "reports/second", originals=6, noise_sections=1, noise_selection="balanced_sources")
    assert {r["sample_id"]: r["sha256"] for r in read_rows(first["manifest"])} == {
        r["sample_id"]: r["sha256"] for r in read_rows(second["manifest"])}
    coverage = read_json(second["noise_coverage"]["path"])
    assert coverage["categories"]["environment"]["input_rows"] == 7
    assert coverage["categories"]["environment"]["effective_unique_sources"] == 6


def test_small_original_budget_reports_unused_sources_and_types_honestly(project):
    result = prepare_evaluation(project, project.parent / "reports/small", originals=1, noise_sections=1, noise_selection="balanced_sources")
    assert result["audio_files"] == 16
    coverage = read_json(result["noise_coverage"]["path"])
    for category in CATEGORIES:
        cell = coverage["categories"][category]
        assert cell["used_sources"] == cell["used_types"] == 1
        assert cell["unused_sources"] > 0 and cell["all_sources_used"] is False and cell["limitation"]


def test_legacy_default_keeps_previous_source_choice_and_waveform_bits(project):
    root = project.parent
    first = prepare_evaluation(project, root / "reports/default", originals=2, noise_sections=1)
    second = prepare_evaluation(project, root / "reports/explicit_legacy", originals=2, noise_sections=1, noise_selection="legacy")
    assert first["noise_selection_policy"] == "legacy" and "noise_coverage" not in first
    a, b = read_rows(first["manifest"]), read_rows(second["manifest"])
    assert {r["sample_id"]: r["sha256"] for r in a} == {r["sample_id"]: r["sha256"] for r in b}
    assert all(len({r["noise_source_id"] for r in a if r["category"] == category}) == 1 for category in CATEGORIES)


def test_balancing_never_bypasses_source_split_audit_or_final_lock(project):
    root, config = project.parent, read_json(project)
    item = read_rows(root / config["manifests"]["noise_validation"])[0]
    rows(root / config["manifests"]["noise"], [{**item, "sample_id": "train_alias", "split": "train"}])
    with pytest.raises(ValueError, match="Cross-split"):
        prepare_evaluation(project, root / "reports/leak", originals=1, noise_sections=1, noise_selection="balanced_sources")
    assert not (root / "reports/leak").exists()
    with pytest.raises(ValueError, match="Final is locked"):
        prepare_evaluation(project, root / "reports/final", split="final", noise_selection="balanced_sources")


def test_invalid_policy_does_not_materialize_audio(project):
    with pytest.raises(ValueError, match="Noise selection"):
        prepare_evaluation(project, project.parent / "reports/invalid", originals=1, noise_selection="invented")
    assert not (project.parent / "reports/invalid").exists()


def test_silent_source_is_not_dropped_or_relabeled_and_failure_is_recorded(project):
    root, config = project.parent, read_json(project)
    manifest = root / config["manifests"]["noise_validation"]
    noises = read_rows(manifest)
    for row in noises:
        if row["category"] == "environment":
            sf.write(row["path"], np.zeros(8 * 16000), 16000, subtype="PCM_16")
            row["sha256"] = digest(row["path"])
    # Give silent fixtures distinct container bytes so the test reaches the
    # window-QC diagnostic rather than an intentionally conflicting label alias.
    for i, row in enumerate(noises):
        if row["category"] == "environment":
            with open(row["path"], "ab") as stream:
                stream.write(bytes([i]))
            row["sha256"] = digest(row["path"])
    rows(manifest, noises)
    output = root / "reports/silent"
    with pytest.raises(ValueError, match="silent"):
        prepare_evaluation(project, output, originals=1, noise_sections=1, noise_selection="balanced_sources")
    failure = read_json(output / "noise_construction_failure.json")
    assert failure["source_dropped"] is failure["SNR_changed"] is failure["policy_lowered"] is False
    assert failure["category"] == "environment"
