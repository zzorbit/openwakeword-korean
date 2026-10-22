"""Fixed, source-separated RIR subset; no extra TTS or model claims."""
from collections import Counter
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows
from korean_openwakeword.audio import active_speech_mask, measured_snr, rms
from korean_openwakeword.evaluation_coverage import assess_common_evaluation
from korean_openwakeword.evaluation_rir import estimate_rt60_t20, render_rir_review_previews
from korean_openwakeword.paired import prepare_evaluation, reverb_subset
from test_korean_library_paired_diversity import delayed_call_with_uneven_background, project, rows


def rirs(project):
    root, config = project.parent, read_json(project)
    values = []
    for group, decay in (("moderate", 25), ("strong", 8)):
        for number in range(3):
            path = root / "data" / f"rir_{group}_{number}.wav"
            time = np.arange(8000) / 16000
            impulse = np.random.default_rng(200 + len(values)).normal(0, .03, len(time)) * np.exp(-decay * time)
            impulse[0] = .5
            sf.write(path, impulse, 16000, subtype="PCM_16")
            values.append({"sample_id": f"rir_{group}_{number}", "source_id": f"fixture:rir:{group}:{number}",
                "path": str(path), "sha256": digest(path), "split": "validation", "category": "rir",
                "provenance": "public_dataset", "source_url": "https://example.test/procedural-rir",
                "license": "isolated-test-fixture", "canonical_room_id": f"fixture_room_{group}_{number}",
                "reverb_group": group, "rt60_s": .3 if group == "moderate" else .9,
                "reverb_group_basis": "declared_test_fixture", "acoustic_source_type": "simulated"})
    rows(root / config["manifests"]["rir_validation"], values)
    return values


@pytest.mark.parametrize("count", [625, 1000])
def test_subset_is_fixed_voice_balanced_and_spreads_available_styles(count):
    values = [{"sample_id": f"call_{family}_{i}", "voice_family_id": f"voice_{family}",
               "style": ["neutral", "projected"][i % 2], "speed": ["slow", "fast"][(i // 2) % 2]}
              for family in range(10) for i in range(500)]
    chosen = reverb_subset(values, count, 42)
    assert len(chosen) == len({r["sample_id"] for r in chosen}) == count
    assert {r["voice_family_id"] for r in chosen} == {f"voice_{i}" for i in range(10)}
    assert max(Counter(r["voice_family_id"] for r in chosen).values()) - min(Counter(r["voice_family_id"] for r in chosen).values()) <= 1
    assert chosen == reverb_subset(list(reversed(values)), count, 42)
    assert len(Counter((r["voice_family_id"], r["style"], r["speed"]) for r in chosen)) == 40


def test_thousand_call_subset_balances_thirty_families_and_declared_styles():
    values = [{"sample_id": f"call_{slot:05d}", "voice_family_id": f"voice_{slot % 30:02d}",
               "axes": {"style": ["neutral", "projected"][(slot // 30) % 2],
                        "speed": ["slow", "fast"][(slot // 60) % 2],
                        "intonation": ["rising", "falling"][(slot // 120) % 2]}}
              for slot in range(5000)]
    chosen = reverb_subset(values, 1000, 42, selection="metadata_axes")
    counts = Counter(row["voice_family_id"] for row in chosen)
    assert set(counts.values()) == {33, 34} and len(counts) == 30
    assert len(chosen) == len({row["sample_id"] for row in chosen}) == 1000
    assert chosen == reverb_subset(values[2000:] + values[:2000], 1000, 42, selection="metadata_axes")
    assert {r["sample_id"] for r in reverb_subset(values, 625, 42, selection="metadata_axes")} <= {r["sample_id"] for r in chosen}
    assert chosen != reverb_subset(values, 1000, 42)  # Legacy ignores the nested TTS axes.
    styles = Counter((row["voice_family_id"], row["axes"]["style"], row["axes"]["speed"], row["axes"]["intonation"]) for row in chosen)
    assert len(styles) == 30 * 8 and max(styles.values()) - min(styles.values()) <= 1


def test_metadata_axes_sampling_falls_back_to_old_explicit_fields_and_records_recipe(project):
    from korean_openwakeword.paired import evaluation_recipe
    values = [{"sample_id": f"call_{slot}", "voice_family_id": f"voice_{slot % 2}",
               "style": ["neutral", "projected"][(slot // 2) % 2]} for slot in range(100)]
    assert reverb_subset(values, 24, 42, selection="metadata_axes") == reverb_subset(values, 24, 42)
    old = evaluation_recipe(project, originals=12, reverb_originals=3)
    new = evaluation_recipe(project, originals=12, reverb_originals=3, reverb_selection="metadata_axes")
    assert old["reverberation"]["selection_policy"] == "voice_then_style_stratified_fixed_subset_v1"
    assert new["reverberation"]["selection_policy"] == "voice_then_declared_axes_stratified_fixed_subset_v2"
    assert {key for key in old["reverberation"] if old["reverberation"][key] != new["reverberation"][key]} == {"selection_policy"}
    with pytest.raises(ValueError, match="Declared synthetic style axes"):
        reverb_subset([{**values[0], "axes": "not metadata"}], 1, 42, selection="metadata_axes")


@pytest.mark.parametrize("selection", ["legacy", "metadata_axes"])
def test_reverb_adds_only_fixed_subset_preserves_dry_audio_and_pairs_backgrounds(project, selection):
    root = project.parent
    rirs(project)
    baseline = prepare_evaluation(project, root / "reports/dry", originals=12, noise_sections=1,
                                  noise_selection="balanced_sources")
    result = prepare_evaluation(project, root / "reports/with_rir", originals=12, noise_sections=1,
                                noise_selection="balanced_sources", reverb_originals=3, reverb_selection=selection)
    dry = read_rows(baseline["manifest"])
    all_rows = read_rows(result["manifest"])
    assert len(all_rows) == 12 * 16 + 3 * 16 * 2
    assert {r["sample_id"]: r["sha256"] for r in dry} == {
        r["sample_id"]: r["sha256"] for r in all_rows if not r.get("rir_applied")}
    assert Counter(r.get("reverb_condition", "none") for r in all_rows) == {"none": 192, "moderate": 48, "strong": 48}
    originals_by_group = {group: {row["original_id"] for row in all_rows if row.get("reverb_condition", "none") == group}
                          for group in ("none", "moderate", "strong")}
    assert originals_by_group["moderate"] == originals_by_group["strong"]
    assert len(originals_by_group["moderate"]) == 3
    assert originals_by_group["moderate"] <= originals_by_group["none"]
    # Joint feasibility adds its context evidence to the construction record.
    # Compare wet pairs with their dry version from the same new evaluation;
    # the separate v1 run above still proves unchanged PCM for these already
    # feasible windows.
    by_dry = {(r["original_id"], r["category"], r.get("snr_db")): r
              for r in all_rows if not r.get("rir_applied")}
    for row in all_rows:
        if not row.get("rir_applied"):
            continue
        assert row["rir_tail_truncated"] is False and row["rir_acoustic_source_type"] == "simulated"
        assert any(c["role"] == "rir" and c["canonical_room_id"] for c in row["components"])
        values, rate = sf.read(row["path"])
        assert rate == 16000 and np.max(np.abs(values)) < .981
        if row["category"] != "clean":
            clean = by_dry[row["original_id"], row["category"], row["snr_db"]]
            assert (row["noise_source_id"], row["noise_offset_s"], row["noise_construction"]) == (
                clean["noise_source_id"], clean["noise_offset_s"], clean["noise_construction"])
            assert abs(row["measured_snr_db"] - row["snr_db"]) < 1e-8
            assert row["background_right_padding_samples"] == row["rir_frames"] - 1
    coverage = read_json(result["noise_coverage"]["path"])
    assert coverage["originals_used"] == 12 and coverage["reverberation"]["originals"] == 3
    assert coverage["reverberation"]["used_rooms"] == coverage["reverberation"]["used_rir_sources"] == 6


def test_joint_reverb_recipe_separates_versions_while_dry_only_stays_v1(project):
    from korean_openwakeword.paired import evaluation_recipe
    root = project.parent
    rirs(project)
    dry = evaluation_recipe(project, originals=1, noise_sections=1, noise_selection="balanced_sources")
    joint = evaluation_recipe(project, originals=1, noise_sections=1,
                              noise_selection="balanced_sources", reverb_originals=1)
    assert dry["noise_window_policy"] == "bounded_offsets_then_zero_padded_event_v1"
    assert joint["noise_window_policy"] == "bounded_offsets_then_zero_padded_event_joint_reverb_v2"
    assert joint["reverberation"]["background_feasibility_policy"] == "one_window_valid_for_dry_and_all_selected_rirs_v2"
    assert joint["snrs"] == dry["snrs"] and joint["seed"] == dry["seed"]
    assert joint["positive_manifest_sha256"] == dry["positive_manifest_sha256"]
    assert joint["noise_manifest_sha256"] == dry["noise_manifest_sha256"]
    # A completed historical v1 recipe must not match the new construction,
    # even if all selection/source inputs were otherwise the same.
    old = {**joint, "noise_window_policy": dry["noise_window_policy"],
           "reverberation": {k: v for k, v in joint["reverberation"].items()
                             if k != "background_feasibility_policy"}}
    write_path = root / "reports/old_recipe.json"
    write_path.write_text(json.dumps(old), encoding="utf-8")
    assert read_json(write_path) != joint


def test_actual_delayed_rir_uses_joint_background_and_keeps_every_condition_audible(project):
    """Persisted PCM, not a playback gain, fixes a dry-valid/wet-invalid window."""
    root, config = project.parent, read_json(project)
    noise, _, _, _ = delayed_call_with_uneven_background(root / "data")
    noises = read_rows(root / config["manifests"]["noise_validation"])
    environment = next(row for row in noises if row["category"] == "environment")
    environment.update(path=noise["path"], sha256=noise["sha256"], start_s=0, end_s=4.0)
    rows(root / config["manifests"]["noise_validation"],
         [environment] + [row for row in noises if row["category"] != "environment"])
    rir_rows = rirs(project)
    chosen = []
    for group, delay in (("moderate", 0), ("strong", 8000)):
        row = next(row for row in rir_rows if row["reverb_group"] == group)
        impulse = np.zeros(delay + 801)
        impulse[delay] = .5
        sf.write(row["path"], impulse, 16000, subtype="PCM_16")
        row.update(sha256=digest(row["path"]), direct_path_s=delay / 16000)
        chosen.append(row)
    rows(root / config["manifests"]["rir_validation"], chosen)
    baseline = prepare_evaluation(project, root / "reports/old_dry_window", originals=1,
                                  noise_sections=1, noise_selection="balanced_sources")
    result = prepare_evaluation(project, root / "reports/joint_reverb_window", originals=1,
                               noise_sections=1, noise_selection="balanced_sources", reverb_originals=1)
    again = prepare_evaluation(project, root / "reports/joint_reverb_repeat", originals=1,
                              noise_sections=1, noise_selection="balanced_sources", reverb_originals=1)
    old_rows, entries = read_rows(baseline["manifest"]), read_rows(result["manifest"])
    assert len(entries) == 48
    assert {row["sample_id"]: row["sha256"] for row in entries} == {
        row["sample_id"]: row["sha256"] for row in read_rows(again["manifest"])}
    dry = [row for row in entries if not row.get("rir_applied")]
    old_env = next(row for row in old_rows if row["category"] == "environment")
    new_env = next(row for row in dry if row["category"] == "environment")
    assert old_env["noise_source_id"] == new_env["noise_source_id"]
    assert old_env["noise_construction"]["offset_attempts"] == 1
    assert new_env["noise_construction"]["offset_attempts"] == 16
    assert new_env["noise_construction"]["placement"] == "zero_padded_event_alignment"
    assert new_env["sha256"] != old_env["sha256"]
    for condition in (None, "moderate", "strong"):
        selected = [row for row in entries if row.get("reverb_condition") == condition]
        clean = next(row for row in selected if row["category"] == "clean")
        clean_values, rate = sf.read(clean["path"])
        assert rate == 16000
        call = clean["calls"][0]
        active = active_speech_mask(clean_values, call["start_s"], call["end_s"])
        assert rms(clean_values, active) >= 10 ** (-35 / 20)
        for row in selected:
            values, _ = sf.read(row["path"])
            assert np.max(np.abs(values)) < .981
            assert row["common_headroom_gain"] == clean["common_headroom_gain"]
            if row["category"] == "clean":
                continue
            dry_row = next(item for item in dry if item["category"] == row["category"]
                           and item["snr_db"] == row["snr_db"])
            assert (row["noise_source_id"], row["noise_offset_s"], row["noise_construction"]) == (
                dry_row["noise_source_id"], dry_row["noise_offset_s"], dry_row["noise_construction"])
            assert abs(row["measured_snr_db"] - row["snr_db"]) < 1e-8
            # Quantized mixed PCM still contains the declared SNR. A modest
            # tolerance accounts for independent 16-bit rounding of each WAV.
            assert abs(measured_snr(clean_values, values - clean_values, active) - row["snr_db"]) < .03
            if row.get("rir_applied"):
                assert row["rir_tail_truncated"] is False
                assert len(values) == len(sf.read(dry_row["path"])[0]) + row["rir_frames"] - 1


def test_rir_missing_or_shared_with_training_never_becomes_a_clean_fallback(project):
    root, config = project.parent, read_json(project)
    with pytest.raises(ValueError, match="Missing held-out"):
        prepare_evaluation(project, root / "reports/missing", originals=2, noise_sections=1,
                           noise_selection="balanced_sources", reverb_originals=1)
    assert not (root / "reports/missing").exists()
    values = rirs(project)
    rows(root / config["manifests"]["rir"], [{**values[0], "sample_id": "training_alias", "split": "train"}])
    with pytest.raises(ValueError, match="Cross-split"):
        prepare_evaluation(project, root / "reports/leak_rir", originals=2, noise_sections=1,
                           noise_selection="balanced_sources", reverb_originals=1)
    assert not (root / "reports/leak_rir").exists()


def test_rir_needs_real_metadata_not_filenames_and_known_final_rooms_are_separate(project):
    root, config = project.parent, read_json(project)
    values = rirs(project)
    first = values[0]
    first.pop("canonical_room_id")
    rows(root / config["manifests"]["rir_validation"], values)
    with pytest.raises(ValueError, match="actual room"):
        prepare_evaluation(project, root / "reports/room_missing", originals=2, noise_sections=1,
                           noise_selection="balanced_sources", reverb_originals=1)
    values = rirs(project)
    final = {**values[0], "sample_id": "final_alias", "split": "final", "sha256": "0" * 64}
    rows(root / config["manifests"]["rir_final"], [final])
    with pytest.raises(ValueError, match="Cross-split"):
        prepare_evaluation(project, root / "reports/leak_room", originals=2, noise_sections=1,
                           noise_selection="balanced_sources", reverb_originals=1)


def test_final_stays_locked_and_new_project_rir_manifests_do_not_mutate_old_json(project):
    config = read_json(project)
    assert set(config["manifests"]) >= {"rir_validation", "rir_final"}
    with pytest.raises(ValueError, match="Final is locked"):
        prepare_evaluation(project, project.parent / "reports/final_rir", split="final", reverb_originals=1)


def test_t20_reports_computed_fit_not_a_measured_room_and_refuses_bad_decay():
    time = np.arange(32000) / 16000
    # Amplitude reaches -60dB after .9s; deterministic broadband tail.
    values = np.random.default_rng(700).normal(size=len(time)) * np.exp(-np.log(1000) * time / .9)
    estimate = estimate_rt60_t20(values, 16000)
    assert abs(estimate["rt60_s"] - .9) < .04 and estimate["fit_r_squared"] > .99
    assert "not a measured" in estimate["scope"]
    with pytest.raises(ValueError, match="silent"):
        estimate_rt60_t20(np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="milliseconds|stable"):
        estimate_rt60_t20(np.ones(16000), 16000)


def test_rir_previews_reuse_synthetic_test_signal_and_leave_approval_pending(project):
    source_rows = rirs(project)
    previews = render_rir_review_previews(project, source_rows, project.parent / "reports/rir_previews", count=6)
    assert len(previews) == 6
    assert set(row["source_type"] for row in previews) == {"simulated"}
    assert all(row["individual_human_listening"] is False and row["call_generation_performed"] is False for row in previews)
    assert all("호출어 발음 검수가 아닙니다" in row["description"] for row in previews)
    assert len({row["canonical_room_id"] for row in previews}) == 6


def test_coverage_checks_reverb_cells_and_uses_actual_rooms_and_sources(project):
    source_rows = rirs(project)
    result = prepare_evaluation(project, project.parent / "reports/coverage_reverb", originals=2, noise_sections=1,
                                noise_selection="balanced_sources", reverb_originals=1)
    policy = {"validation_originals": 2, "validation_references": 2, "noise_sections": 1,
              "snrs": [20, 15, 10, 5, 0], "reverb_originals": 1, "reverb_groups": ["moderate", "strong"],
              "rir_min_sources": 50, "rir_min_rooms": 20, "development_negative_target_hours": [30, 30]}
    scope = assess_common_evaluation(policy, {"models": [{"negative": {"hours": 30}}]}, result["manifest"], result["noise_coverage"]["path"])
    assert scope["reverberation"]["originals_per_group"] == {"moderate": 1, "strong": 1}
    assert all(row["samples"] == 2 for row in scope["condition_counts"])
    assert len(scope["reverberation"]["condition_counts"]) == 48
    assert any("잔향 원본 2/50" in row for row in scope["shortages"])
    assert any("잔향 공간 2/20" in row for row in scope["shortages"])


def test_new_thirty_validation_families_do_not_change_default_ten(tmp_path):
    from korean_openwakeword.project import create_project, voice_candidates
    assert Counter(row["split"] for row in voice_candidates()["families"]) == {"train": 100, "validation": 10, "final": 10}
    project = create_project(tmp_path / "thirty", "보라야", "bora", validation_voice_families=30)
    families = read_json(project.parent / "voices.json")["families"]
    assert Counter(row["split"] for row in families) == {"train": 100, "validation": 30, "final": 10}
    assert len({row["family_id"] for row in families}) == 140


def test_real_pinned_common_rir_catalog_records_simulated_rooms_and_fitted_decay():
    from korean_openwakeword.evaluation_rir import common_rir_catalog, common_rir_preview
    catalog = common_rir_catalog()
    assert len(catalog["assets"]) == 80
    assert len({row["canonical_room_id"] for row in catalog["assets"]}) == 20
    assert Counter(row["reverb_group"] for row in catalog["assets"]) == {"moderate": 40, "strong": 40}
    for row in catalog["assets"]:
        assert row["acoustic_source_type"] == "simulated"
        low, high = catalog["group_boundaries"][row["reverb_group"] + "_rt60_s"]
        assert low <= row["rt60_s"] < high
        assert row["rt60_estimation"]["fit_r_squared"] >= .95
        assert len(row["sha256"]) == 64 and row["provider_room_id"]
    preview = common_rir_preview()
    assert preview["simulated_room_count"] == preview["room_count"] == 20


@pytest.mark.parametrize("split,pool", [("train", "rir"), ("final", "rir_final")])
def test_common_rir_preparation_rejects_known_training_or_final_room_before_download(project, split, pool):
    from korean_openwakeword.evaluation_rir import common_rir_catalog, prepare_common_rirs
    root, config = project.parent, read_json(project)
    source = {**common_rir_catalog()["assets"][0], "sample_id": "known_room_alias", "split": split,
              "canonical_source_id": "renamed-original", "sha256": "0" * 64}
    rows(root / config["manifests"][pool], [source])
    with pytest.raises(ValueError, match="overlaps " + split):
        prepare_common_rirs(project, root / "reports/should_not_exist", archive=root / "not_downloaded.zip")
    assert not (root / "reports/should_not_exist").exists()


def test_interrupted_rir_preparation_can_resume_exact_pinned_files_without_receipt(tmp_path):
    from korean_openwakeword.evaluation_rir import _resumable_output
    from korean_openwakeword.common import identity
    destination = tmp_path / "interrupted"
    (destination / "audio").mkdir(parents=True)
    path = destination / "audio/example.wav"
    sf.write(path, np.r_[.5, np.zeros(500)], 16000, subtype="PCM_16")
    catalog = {"assets": [{"id": "example", "sha256": digest(path)}]}
    _resumable_output(destination, catalog)
    before = digest(path)
    _resumable_output(destination, catalog)
    assert digest(path) == before and read_json(destination / "preparation.json")["catalog_sha256"] == identity(catalog)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="altered"):
        _resumable_output(destination, catalog)
