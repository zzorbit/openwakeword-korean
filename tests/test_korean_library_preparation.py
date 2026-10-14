"""Public material preparation is resumable, source-separated and consent-bound."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from korean_openwakeword.audio import pcm16
from korean_openwakeword.common import digest, read_json, read_rows, write_json
from korean_openwakeword.data import audit_project, validate_admission
from korean_openwakeword.preparation import _convert, apply_prepared_review, connect_materials, create_source_index, prepare_public_audio
from korean_openwakeword.project import create_project, load_project


def _rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def material(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    folder = tmp_path / "provider"
    folder.mkdir()
    for n in range(5):
        sf.write(folder / f"source{n}.wav", np.random.default_rng(n).normal(0, .06, 48000), 24000, subtype="PCM_16")
    options = {"input_directory": folder, "output": "data/prepared/provider",
               "dataset_id": "fixture_provider", "source_url": "https://example.org/fixture",
               "license": "fixture only", "category": "environment",
               "speech": False,
               "category_basis": "provider fixture label", "validation_fraction": .2,
               "preview_seconds": 1, "representatives_per_stratum": 10}
    return project, folder, options


def _answers(project, receipt):
    queue = read_rows(receipt["artifacts"]["representative_queue"]["path"])
    path = project.parent / "review" / "pool_results.json"
    write_json(path, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "fixture",
                     "source_scope": "Human dispositions entered through the local review screen",
                     "decisions": {r["sample_id"]: {"decision": "keep", "listened": True} for r in queue},
                     "whole_pool_policy_approved": True,
                     "source_manifest_sha256": receipt["artifacts"]["source_manifest"]["sha256"],
                     "representative_queue_sha256": receipt["artifacts"]["representative_queue"]["sha256"],
                     "queue_sha256": receipt["artifacts"]["representative_queue"]["sha256"],
                     "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in queue}})
    return path, queue


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_bounded_conversion_matches_whole_polyphase_pcm(tmp_path, rate):
    source, target = tmp_path / "source.wav", tmp_path / "target.wav"
    samples = np.random.default_rng(rate).normal(0, .08, (rate * 3 + 13, 2))
    sf.write(source, samples, rate, subtype="PCM_16")
    _convert(source, target, block_frames=4096)
    decoded = sf.read(source, dtype="float64", always_2d=True)[0].mean(axis=1)
    from math import gcd
    divisor = gcd(rate, 16000)
    expected = pcm16(resample_poly(decoded, 16000 // divisor, rate // divisor))
    actual, actual_rate = sf.read(target, dtype="int16")
    assert actual_rate == 16000 and np.array_equal(actual, expected)


def test_prepare_retains_originals_qc_sources_and_separated_reserve(material):
    project, folder, options = material
    originals = {p.name: digest(p) for p in folder.iterdir()}
    receipt = prepare_public_audio(project, **options)
    assert receipt["train_source_count"] == 4 and receipt["validation_source_count"] == 1
    assert receipt["training_admitted"] is False and receipt["human_listened_count"] == 0
    assert receipt["independent_STT_runs"] == 0 and receipt["Final_accessed"] is False
    train = read_rows(receipt["artifacts"]["source_manifest"]["path"])
    reserve = read_rows(receipt["artifacts"]["validation_reserve"]["path"])
    assert {r["canonical_source_id"] for r in train}.isdisjoint(r["canonical_source_id"] for r in reserve)
    assert all(r["technical_pass"] and sf.info(r["path"]).subtype == "PCM_16" for r in train + reserve)
    assert {p.name: digest(p) for p in folder.iterdir()} == originals
    assert prepare_public_audio(project, **options)["resumed"] is True
    assert not list((project.parent / options["output"]).rglob("*.partial.wav"))


def test_target_is_whole_sources_not_duplication_and_reports_shortfall(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **{**options, "target_hours": 1})
    assert receipt["target_met"] is False and receipt["train_hours"] < 1
    assert receipt["train_source_count"] == 4


@pytest.mark.parametrize("change", ["source", "settings"])
def test_resume_refuses_changed_inputs_or_settings(material, change):
    project, folder, options = material
    prepare_public_audio(project, **options)
    if change == "source":
        sf.write(folder / "source0.wav", np.ones(24000) * .1, 24000)
    else:
        options["seed"] = 23
    with pytest.raises(ValueError, match="changed"):
        prepare_public_audio(project, **options)


def test_interrupted_conversion_reuses_verified_per_source_records(material, monkeypatch):
    project, _, options = material
    import korean_openwakeword.preparation as module
    convert = module._convert
    count = 0
    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise KeyboardInterrupt("fixture interruption")
        return convert(*args, **kwargs)
    monkeypatch.setattr(module, "_convert", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare_public_audio(project, **options)
    records = list((project.parent / options["output"] / "checks").glob("*.json"))
    assert len(records) == 2
    hashes = {p: digest(p) for p in records}
    monkeypatch.setattr(module, "_convert", convert)
    assert prepare_public_audio(project, **options)["train_source_count"] == 4
    assert all(digest(p) == h for p, h in hashes.items())


def test_missing_provider_metadata_refused_before_audio_decode(material, monkeypatch):
    project, _, options = material
    import korean_openwakeword.preparation as module
    monkeypatch.setattr(module, "_convert", lambda *a: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="license"):
        prepare_public_audio(project, **{**options, "license": None})


def test_final_metadata_and_known_final_source_refused_before_decode(material, monkeypatch):
    project, folder, options = material
    index = folder.parent / "index.jsonl"
    _rows(index, [{"path": p.name, "split": "final"} for p in folder.iterdir()])
    import korean_openwakeword.preparation as module
    monkeypatch.setattr(module, "_convert", lambda *a: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="Final"):
        prepare_public_audio(project, **{**options, "source_index": index})


def test_explicit_shared_speaker_keeps_all_utterances_in_one_partition(material):
    project, folder, options = material
    index = folder.parent / "index.jsonl"
    _rows(index, [{"path": p.name, "category": "competing_speech", "speech": True,
                   "language": "Korean", "canonical_speaker_id": "provider:one_person"} for p in folder.iterdir()])
    receipt = prepare_public_audio(project, **{**options, "source_index": index})
    assert receipt["validation_source_count"] == 0 and receipt["train_source_count"] == 5


def test_unknown_speakers_are_not_invented_and_train_only_is_supported(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **{**options, "category": "competing_speech", "language": "Korean", "speech": True})
    assert receipt["validation_source_count"] == 0
    rows = read_rows(receipt["artifacts"]["source_manifest"]["path"])
    assert all("canonical_speaker_id" not in r and r["unknown_speaker_group"] for r in rows)


def test_contradictory_declared_shared_room_splits_are_refused(material):
    project, folder, options = material
    index = folder.parent / "index.jsonl"
    _rows(index, [{"path": p.name, "canonical_room_id": "same_room",
                   "split": "train" if i else "validation"} for i, p in enumerate(folder.iterdir())])
    with pytest.raises(ValueError, match="contradictory"):
        prepare_public_audio(project, **{**options, "source_index": index})


def test_exact_duplicate_audio_counted_once(material):
    project, folder, options = material
    (folder / "copy.wav").write_bytes((folder / "source0.wav").read_bytes())
    receipt = prepare_public_audio(project, **{**options, "validation_fraction": 0})
    assert receipt["duplicates_excluded"] == 1 and receipt["train_source_count"] == 5
    assert receipt["train_hours"] == pytest.approx(10 / 3600)


def test_silence_and_clipping_are_rejected_without_destroying_originals(material):
    project, folder, options = material
    sf.write(folder / "silence.wav", np.zeros(1000), 16000)
    sf.write(folder / "clipped.wav", np.ones(1000), 16000)
    receipt = prepare_public_audio(project, **options)
    assert receipt["rejected_files"] == 2
    rejected = read_json(receipt["artifacts"]["rejections"]["path"])["sources"]
    assert all(r["original_preserved"] for r in rejected)


@pytest.mark.parametrize("field", ["whole_pool_policy_approved", "decision", "listened"])
def test_no_consent_unheard_or_held_review_blocks_admission(material, field):
    project, _, options = material
    receipt = prepare_public_audio(project, **options)
    path, queue = _answers(project, receipt)
    answers = read_json(path)
    if field == "whole_pool_policy_approved":
        answers[field] = False
    else:
        answers["decisions"][queue[0]["sample_id"]][field] = "review" if field == "decision" else False
    write_json(path, answers)
    with pytest.raises(ValueError):
        apply_prepared_review(project, preparation=receipt["preparation"], pools=["noise"])
    assert not (Path(receipt["preparation"]["path"]).parent / "admitted/evidence/admission.json").exists()


def test_saved_review_admits_only_train_and_connects_negative_windows_and_noise(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **{**options, "category": "competing_speech", "language": "Korean", "speech": True,
                                            "validation_fraction": 0})
    _answers(project, receipt)
    admitted = apply_prepared_review(project, preparation=receipt["preparation"],
                                    pools=["korean_negative", "noise"], connect=True)
    root, config = load_project(project)
    negative = read_rows(root / config["manifests"]["korean_negative"])
    noise = read_rows(root / config["manifests"]["noise"])
    assert len(negative) == 5 and all(r["window_policy"] == "fixed_2s_no_audio_copy" for r in negative)
    assert {r["sha256"] for r in negative} == {r["sha256"] for r in noise}
    validate_admission(root, negative, config, "korean_negative")
    assert audit_project(project)["passed"]
    assert admitted["connected"] and admitted["source_count"] == 5
    resumed = apply_prepared_review(project, preparation=receipt["preparation"],
                                    pools=["korean_negative", "noise"], connect=True)
    assert resumed["resumed"] and resumed["connection"]["resumed"]


def test_discard_excludes_exact_original_preserves_full_saved_evidence(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **options)
    path, queue = _answers(project, receipt)
    answers = read_json(path)
    answers["decisions"][queue[0]["sample_id"]]["decision"] = "discard"
    write_json(path, answers)
    before = digest(path)
    admitted = apply_prepared_review(project, preparation=receipt["preparation"], pools=["noise"])
    record = read_json(project.parent / admitted["admission"]["path"])
    assert record["full_frozen_source_count"] == 4 and record["source_count"] == 3
    assert record["excluded_source_ids"] == [queue[0]["canonical_source_id"]]
    assert record["human_answers"]["sha256"] == before and digest(path) == before
    hashes = {r["sha256"] for r in read_rows(admitted["manifests"]["noise"]["path"])}
    assert queue[0]["reviewed_source"]["sha256"] not in hashes
    assert apply_prepared_review(project, preparation=receipt["preparation"], pools=["noise"])["resumed"]


def test_resume_refuses_corrupted_prepared_audio(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **options)
    original = read_rows(receipt["artifacts"]["source_manifest"]["path"])[0]
    Path(original["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        prepare_public_audio(project, **options)


def test_wrong_queue_answers_and_tampered_sources_cannot_admit(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **options)
    path, _ = _answers(project, receipt)
    answers = read_json(path)
    answers["queue_sha256"] = "0" * 64
    write_json(path, answers)
    with pytest.raises(ValueError, match="bind"):
        apply_prepared_review(project, preparation=receipt["preparation"])


def test_empty_manifest_guard_does_not_overwrite_existing_inputs(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **options)
    _answers(project, receipt)
    root, config = load_project(project)
    manifest = root / config["manifests"]["noise"]
    _rows(manifest, read_rows(receipt["artifacts"]["source_manifest"]["path"])[:1])
    before = digest(manifest)
    with pytest.raises(ValueError, match="empty"):
        apply_prepared_review(project, preparation=receipt["preparation"], pools=["noise"], connect=True)
    assert digest(manifest) == before


def test_declared_korean_speech_required_for_korean_pool(material):
    project, _, options = material
    receipt = prepare_public_audio(project, **options)
    _answers(project, receipt)
    with pytest.raises(ValueError, match="eligible"):
        apply_prepared_review(project, preparation=receipt["preparation"], pools=["korean_negative"])


def test_zeroth_provider_adapter_never_invents_speakers_or_transcribes(tmp_path):
    folder = tmp_path / "zeroth"
    speaker = folder / "003"
    speaker.mkdir(parents=True)
    sf.write(speaker / "003_0001.wav", np.ones(5000) * .03, 16000)
    (speaker / "003.trans.txt").write_text("003_0001 공급자가 작성한 문장\n", encoding="utf-8")
    result = create_source_index(folder, output=tmp_path / "index.jsonl", dataset="zeroth",
                                 source_url="https://example.org/zeroth", license="fixture")
    row = read_rows(result["source_index"]["path"])[0]
    assert row["canonical_speaker_id"] == "zeroth:speaker:003" and row["language"] == "Korean"
    assert row["text"] == "공급자가 작성한 문장" and result["independent_STT_runs"] == 0


def test_provider_receipt_adapter_retains_license_category_and_actual_room_ids(material):
    _, folder, _ = material
    metadata = folder.parent / "provider.json"
    write_json(metadata, {"files": [{"path": str(p), "license": "per-file-license", "category": "rir",
                                     "canonical_room_id": "actual_measured_room"} for p in folder.iterdir()]})
    output = create_source_index(folder, output=folder.parent / "index.jsonl", metadata=metadata)
    rows = read_rows(output["source_index"]["path"])
    assert all(r["license"] == "per-file-license" and r["canonical_room_id"] == "actual_measured_room" for r in rows)


def test_multiple_dataset_admissions_merge_without_manual_manifest_editing(material):
    project, folder, options = material
    first = prepare_public_audio(project, **options)
    _answers(project, first)
    first_admission = apply_prepared_review(project, preparation=first["preparation"], pools=["noise"])
    second_folder = folder.parent / "music"
    second_folder.mkdir()
    for n in range(3):
        sf.write(second_folder / f"track{n}.wav", np.random.default_rng(100 + n).normal(0, .04, 16000), 16000)
    second = prepare_public_audio(project, **{**options, "input_directory": second_folder,
                                  "output": "data/prepared/music", "category": "music", "dataset_id": "fixture_music"})
    _answers(project, second)
    second_admission = apply_prepared_review(project, preparation=second["preparation"], pools=["noise"])
    connection = connect_materials(project, admissions=[first_admission["receipt"], second_admission["receipt"]], pools=["noise"])
    rows = read_rows(connection["manifests"]["noise"]["path"])
    assert len(rows) == 6 and {r["dataset_id"] for r in rows} == {"fixture_provider", "fixture_music"}
    assert connect_materials(project, admissions=[first_admission["receipt"], second_admission["receipt"]], pools=["noise"])["resumed"]


def test_interrupted_connection_resumes_exact_partial_publication(material, monkeypatch):
    project, _, options = material
    prepared = prepare_public_audio(project, **{**options, "category": "competing_speech", "language": "Korean", "speech": True, "validation_fraction": 0})
    _answers(project, prepared)
    admission = apply_prepared_review(project, preparation=prepared["preparation"], pools=["noise", "korean_negative"])
    import korean_openwakeword.preparation as module
    replace = module.os.replace
    count = 0
    def interrupt(src, dst):
        nonlocal count
        if Path(dst).name in ("noise.jsonl", "korean.jsonl") and "material_connections" not in str(dst):
            count += 1
            if count == 2:
                raise KeyboardInterrupt("fixture publication interruption")
        return replace(src, dst)
    monkeypatch.setattr(module.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        connect_materials(project, admissions=[admission["receipt"]], pools=["noise", "korean_negative"])
    monkeypatch.setattr(module.os, "replace", replace)
    result = connect_materials(project, admissions=[admission["receipt"]], pools=["noise", "korean_negative"])
    assert result["state"] == "connected" and all(Path(b["path"]).exists() for b in result["manifests"].values())


@pytest.mark.parametrize("container", ["runs/member1", "features/korean_negative"])
def test_existing_training_or_feature_contract_blocks_connection(material, container):
    project, _, options = material
    prepared = prepare_public_audio(project, **options)
    _answers(project, prepared)
    admission = apply_prepared_review(project, preparation=prepared["preparation"], pools=["noise"])
    write_json(project.parent / container / "contract.json", {"historical": "preserve"})
    with pytest.raises(ValueError, match="started"):
        connect_materials(project, admissions=[admission["receipt"]], pools=["noise"])


def test_disk_preflight_blocks_bulk_conversion(material, monkeypatch):
    project, _, options = material
    import korean_openwakeword.preparation as module
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: usage(100, 100, 0))
    monkeypatch.setattr(module, "_convert", lambda *a: pytest.fail("must not bulk-write"))
    with pytest.raises(ValueError, match="disk space"):
        prepare_public_audio(project, **options)


def test_source_index_escape_and_incomplete_mapping_are_refused(material):
    project, folder, options = material
    index = folder.parent / "index.jsonl"
    _rows(index, [{"path": "../outside.wav"}])
    with pytest.raises(ValueError, match="stay within"):
        prepare_public_audio(project, **{**options, "source_index": index})
    _rows(index, [{"path": "source0.wav"}])
    with pytest.raises(ValueError, match="exactly"):
        prepare_public_audio(project, **{**options, "source_index": index})


def test_final_frozen_project_never_prepares_or_connects(material):
    project, _, options = material
    write_json(project.parent / "final_lock.json", {"frozen": True})
    with pytest.raises(ValueError, match="Final"):
        prepare_public_audio(project, **options)
    with pytest.raises(ValueError, match="Final"):
        connect_materials(project, admissions=["not_opened.json"], pools=["noise"])


def test_declared_intervals_drive_hours_previews_and_windows(material):
    project, folder, options = material
    index = folder.parent / "intervals.jsonl"
    _rows(index, [{"path": p.name, "start_s": .5, "end_s": 1.0} for p in folder.iterdir()])
    prepared = prepare_public_audio(project, **{**options, "source_index": index, "validation_fraction": 0})
    assert prepared["train_hours"] == pytest.approx(2.5 / 3600)
    _, queue = _answers(project, prepared)
    assert all(r["preview_start_s"] == .5 and r["preview_end_s"] == 1 for r in queue)
    result = apply_prepared_review(project, preparation=prepared["preparation"], pools=["generic_negative"])
    windows = read_rows(result["manifests"]["generic_negative"]["path"])
    assert all(r["start_s"] == .5 and r["end_s"] == 1 for r in windows)


def test_out_of_bounds_interval_rejected_and_silent_interval_not_counted(material):
    project, folder, options = material
    mixed = folder / "interval.wav"
    sf.write(mixed, np.r_[np.ones(16000) * .05, np.zeros(16000)], 16000)
    index = folder.parent / "intervals.jsonl"
    _rows(index, [{"path": p.name, **({"start_s": 1.2, "end_s": 1.8} if p == mixed else
                                    {"start_s": -1, "end_s": .5} if p.name == "source0.wav" else {})}
                  for p in folder.iterdir()])
    prepared = prepare_public_audio(project, **{**options, "source_index": index, "validation_fraction": 0})
    assert prepared["rejected_files"] == 2 and prepared["train_source_count"] == 4


def test_provider_sha256_mismatch_is_refused_before_decode(material, monkeypatch):
    project, folder, options = material
    index = folder.parent / "checksums.jsonl"
    _rows(index, [{"path": p.name, "sha256": "0" * 64} for p in folder.iterdir()])
    import korean_openwakeword.preparation as module
    monkeypatch.setattr(module, "_convert", lambda *a: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="Provider source SHA256"):
        prepare_public_audio(project, **{**options, "source_index": index})


def test_all_declared_secondary_speakers_are_source_separated(material):
    project, folder, options = material
    index = folder.parent / "speakers.jsonl"
    _rows(index, [{"path": p.name, "speech": True, "category": "competing_speech",
                   "canonical_speaker_id": "primary:" + p.stem,
                   "canonical_speaker_ids": ["primary:" + p.stem, "shared:secondary"]} for p in folder.iterdir()])
    prepared = prepare_public_audio(project, **{**options, "source_index": index})
    assert prepared["validation_source_count"] == 0 and prepared["train_source_count"] == 5


def test_known_heldout_original_hash_alias_blocked_before_decode(material, monkeypatch):
    project, folder, options = material
    root, config = load_project(project)
    source = folder / "source0.wav"
    _rows(root / config["manifests"]["noise_validation"], [{"sample_id": "held", "split": "validation",
           "canonical_source_id": "different:alias", "original_sha256": digest(source), "sha256": "not-this-format"}])
    import korean_openwakeword.preparation as module
    monkeypatch.setattr(module, "_convert", lambda *a: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="Exact held-out"):
        prepare_public_audio(project, **options)


def test_cross_admission_exact_audio_aliases_are_deduplicated_after_identity_audit(material):
    project, folder, options = material
    first = prepare_public_audio(project, **{**options, "validation_fraction": 0})
    _answers(project, first)
    admitted1 = apply_prepared_review(project, preparation=first["preparation"], pools=["noise"])
    second = prepare_public_audio(project, **{**options, "dataset_id": "other_alias",
                   "output": "data/prepared/alias", "validation_fraction": 0})
    _answers(project, second)
    admitted2 = apply_prepared_review(project, preparation=second["preparation"], pools=["noise"])
    merged = connect_materials(project, admissions=[admitted1["receipt"], admitted2["receipt"]], pools=["noise"])
    assert merged["pool_rows"]["noise"] == 5


def test_existing_unowned_output_preserved_and_receipt_contract_checked(material):
    project, _, options = material
    out = project.parent / options["output"]
    out.mkdir(parents=True)
    marker = out / "keep.txt"
    marker.write_text("user-owned", encoding="utf-8")
    with pytest.raises(ValueError, match="dedicated empty"):
        prepare_public_audio(project, **options)
    assert marker.read_text(encoding="utf-8") == "user-owned"
    marker.unlink()
    prepared = prepare_public_audio(project, **options)
    path = Path(prepared["preparation"]["path"])
    damaged = read_json(path)
    damaged["contract"]["sha256"] = "0" * 64
    write_json(path, damaged)
    with pytest.raises(ValueError, match="actual contract"):
        prepare_public_audio(project, **options)


def test_completed_prepare_readonly_resume_does_not_require_new_bulk_space(material, monkeypatch):
    project, _, options = material
    prepare_public_audio(project, **options)
    import korean_openwakeword.preparation as module
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: pytest.fail("completed resume must not allocate"))
    assert prepare_public_audio(project, **options)["resumed"]


def test_unknown_speech_presence_is_preserved_not_labelled_non_speech(material):
    project, _, options = material
    prepared = prepare_public_audio(project, **{**options, "category": "music", "speech": None})
    rows = read_rows(prepared["artifacts"]["source_manifest"]["path"])
    assert all(r["speech"] is None and r["speech_presence"] == "unknown" and r["unknown_speaker_group"] for r in rows)
    assert prepared["validation_source_count"] == 0


def test_explicit_provider_license_exclusions_preserve_original_and_reason(material):
    project, folder, options = material
    index = folder.parent / "provider.jsonl"
    _rows(index, [{"path": p.name, **({"excluded": True, "exclusion_reason": "No published per-track license", "license": None}
                                    if p.name == "source0.wav" else {})} for p in folder.iterdir()])
    prepared = prepare_public_audio(project, **{**options, "source_index": index})
    assert prepared["excluded_provider_files"] == 1 and prepared["unique_audio_files"] == 4
    excluded = [r for r in read_json(prepared["artifacts"]["rejections"]["path"])["sources"] if r["state"] == "provider_excluded"]
    assert excluded[0]["original_preserved"] and "license" in excluded[0]["reason"]


def test_provider_metadata_binding_change_blocks_preparation(material):
    project, folder, options = material
    metadata = folder.parent / "provider_metadata.json"
    write_json(metadata, {"actual": "provider"})
    index = folder.parent / "index.jsonl"
    _rows(index, [{"path": p.name, "provider_metadata_bindings": [{"path": str(metadata), "sha256": digest(metadata)}]} for p in folder.iterdir()])
    write_json(metadata, {"changed": "after index"})
    with pytest.raises(ValueError, match="changed bound"):
        prepare_public_audio(project, **{**options, "source_index": index})


def test_rir_review_preview_contains_impulse_not_only_late_silent_tail(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    folder = tmp_path / "rir"
    folder.mkdir()
    samples = np.zeros(20 * 16000)
    samples[1600] = .5
    sf.write(folder / "rir.wav", samples, 16000)
    index = tmp_path / "index.jsonl"
    _rows(index, [{"path": "rir.wav", "canonical_room_id": "actual_room"}])
    prepared = prepare_public_audio(project, input_directory=folder, output="data/prepared/rir",
                   dataset_id="measured_rir", source_url="https://example.org/rir", license="fixture",
                   category="rir", category_basis="provider measured impulse response", source_index=index,
                   preview_seconds=1, validation_fraction=0)
    preview = read_rows(prepared["artifacts"]["representative_queue"]["path"])[0]
    values = sf.read(preview["path"])[0]
    assert np.max(values) > .07 and preview["preview_selection"] == "fixed_synthetic_chirp_with_rir"
    demonstration = preview["preview_construction"]
    assert demonstration["training_material"] is False and demonstration["scored_evaluation_material"] is False
    assert demonstration["rir_source"]["sha256"] == read_rows(prepared["artifacts"]["source_manifest"]["path"])[0]["sha256"]
    assert np.max(sf.read(folder / "rir.wav")[0]) == .5


def test_overlapping_parent_intervals_are_not_double_counted(material):
    project, folder, options = material
    index = folder.parent / "parents.jsonl"
    _rows(index, [{"path": p.name, "canonical_recording_id": "same:long_recording", "parent_start_s": n}
                 for n, p in enumerate(folder.iterdir())])
    prepared = prepare_public_audio(project, **{**options, "source_index": index})
    assert prepared["train_hours"] == pytest.approx(6 / 3600) and prepared["validation_source_count"] == 0


def test_window_manifest_generation_and_publication_are_lazy_and_immutable(tmp_path):
    from korean_openwakeword.preparation import _fixed_windows, _publish_rows
    from types import GeneratorType
    row = {"sample_id": "long_source", "duration_s": 360000}
    windows = _fixed_windows([row])
    assert isinstance(windows, GeneratorType)
    assert next(windows)["end_s"] == 2
    assert next(windows)["end_s"] == 4
    manifest = tmp_path / "rows.jsonl"
    _publish_rows(manifest, ({"slot": n} for n in range(10000)))
    before = digest(manifest)
    _publish_rows(manifest, ({"slot": n} for n in range(10000)))
    with pytest.raises(ValueError, match="Immutable"):
        _publish_rows(manifest, ({"slot": n} for n in range(9999)))
    assert digest(manifest) == before and not list(tmp_path.glob("*.tmp"))


def test_only_full_wakeword_in_actual_provider_transcript_is_excluded(material):
    project, folder, options = material
    index = folder.parent / "captions.jsonl"
    texts = {"source0.wav": {"text": "보라야 불을 켜", "transcript_origin": "provider_metadata_not_STT"},
             "source1.wav": {"text": "보라색 하늘", "transcript_origin": "provider_metadata_not_STT"},
             "source2.wav": {"text": "보라야", "category_basis": "class label not a transcript"}}
    _rows(index, [{"path": p.name, **texts.get(p.name, {})} for p in folder.iterdir()])
    prepared = prepare_public_audio(project, **{**options, "source_index": index, "validation_fraction": 0})
    assert prepared["provider_transcript_screened_sources"] == 2 and prepared["provider_transcript_excluded_sources"] == 1
    assert prepared["train_source_count"] == 4 and prepared["independent_STT_runs"] == 0


def test_target_hours_decodes_only_selected_sources_with_unselected_raw_manifest(material, monkeypatch):
    project, _, options = material
    import korean_openwakeword.preparation as module
    convert, calls = module._convert, []
    def track(*args, **kwargs):
        calls.append(str(args[0]))
        return convert(*args, **kwargs)
    monkeypatch.setattr(module, "_convert", track)
    prepared = prepare_public_audio(project, **{**options, "target_hours": 2 / 3600, "validation_fraction": 0})
    assert len(calls) == 1 and prepared["train_source_count"] == 1
    assert prepared["unselected_waveforms_not_decoded"] == 4
    unused = read_rows(prepared["artifacts"]["unused_manifest"]["path"])
    assert all(r["technical_pass"] is False and r["original_audio_opened_for_decode"] is False for r in unused)
    assert all(digest(r["path"]) == r["sha256"] for r in unused)


def test_target_hours_preserves_speaker_partition_but_subsets_within_train(material):
    project, folder, options = material
    index = folder.parent / "same_speaker.jsonl"
    _rows(index, [{"path": p.name, "speech": True, "canonical_speaker_id": "same:person"} for p in folder.iterdir()])
    prepared = prepare_public_audio(project, **{**options, "source_index": index, "target_hours": 2 / 3600})
    assert prepared["train_source_count"] == 1 and prepared["validation_source_count"] == 0
    unused = read_rows(prepared["artifacts"]["unused_manifest"]["path"])
    assert all(r["split"] == "train" for r in unused)


def test_failed_selected_source_is_replaced_before_training_quota_count(material, monkeypatch):
    project, _, options = material
    import korean_openwakeword.preparation as module
    convert, calls = module._convert, []
    def fail_first(*args, **kwargs):
        calls.append(str(args[0]))
        if len(calls) == 1:
            raise ValueError("fixture decoder failure")
        return convert(*args, **kwargs)
    monkeypatch.setattr(module, "_convert", fail_first)
    prepared = prepare_public_audio(project, **{**options, "target_hours": 2 / 3600, "validation_fraction": 0})
    assert len(calls) == 2 and prepared["train_source_count"] == 1 and prepared["target_met"]
    assert prepared["rejected_files"] == 1 and prepared["unused_source_count"] == 3


def test_normalized_duplicate_across_preassigned_train_reserve_stops(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    folder = tmp_path / "provider"
    folder.mkdir()
    values = np.random.default_rng(12).integers(-2000, 2000, 32000, dtype="int16")
    sf.write(folder / "train.wav", values, 16000, subtype="PCM_16")
    sf.write(folder / "reserve.flac", values, 16000, subtype="PCM_16")
    index = tmp_path / "index.jsonl"
    _rows(index, [{"path": "train.wav", "split": "train"}, {"path": "reserve.flac", "split": "validation"}])
    with pytest.raises(ValueError, match="Same normalized PCM"):
        prepare_public_audio(project, input_directory=folder, output="data/prepared/duplicate",
                 dataset_id="fixture", source_url="https://example.org/fixture", license="fixture",
                 category="environment", speech=False, category_basis="fixture", source_index=index)
