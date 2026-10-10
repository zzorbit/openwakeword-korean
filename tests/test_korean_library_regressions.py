"""New-word lifecycle regressions: evidence, incremental generation and sampling."""
import json
import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import bound_file, digest, identity, read_json, read_rows, write_json
from korean_openwakeword.project import create_project, load_project, validate_project
from korean_openwakeword.tts import generate, generate_references
from korean_openwakeword.data import audit_project, validate_admission
from korean_openwakeword.review import apply_review


class SyntheticFixture:
    """Exercise the Qwen API contract without a model download or human audio."""
    def generate_voice_design(self, **kwargs):
        t = np.arange(9600) / 16000
        return [.15 * np.sin(2 * np.pi * 250 * t) + np.random.normal(0, .002, len(t))], 16000


def test_reference_limits_can_be_extended(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    first = generate_references(project, limit=1, model=SyntheticFixture())
    second = generate_references(project, limit=2, model=SyntheticFixture())
    assert first["generated"] == 1 and second["generated"] == 2
    assert len(read_rows(project.parent / "data/references/train/references.jsonl")) == 2


def test_reference_rejection_revokes_future_reuse(tmp_path):
    from korean_openwakeword.reference_bank import import_references
    original = create_project(tmp_path / "original", "보라야", "bora")
    generated = generate_references(original, limit=1, model=SyntheticFixture())
    row = read_rows(generated["review_queue"])[0]
    apply_review(original, generated["review_queue"], {row["sample_id"]: dict(decision="keep", listened=True)}, "references")
    apply_review(original, generated["review_queue"], {row["sample_id"]: dict(decision="discard", listened=True)}, "references")
    assert row["voice_family_id"] not in read_json(original.parent / "approval.json")["accepted_reference_families"]
    other = create_project(tmp_path / "other", "다솜아", "dasom")
    with pytest.raises(ValueError):
        import_references(other, original)


def test_pilot_rejection_revokes_future_bulk_generation(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora", positive_originals=1)
    generated = generate(project, pilot=True, limit=1, method="design", model=SyntheticFixture())
    queue = project.parent / "data/pilot/train/candidates.jsonl"
    row = read_rows(queue)[0]
    apply_review(project, queue, {row["sample_id"]: dict(decision="keep", listened=True)}, "pilot")
    assert read_json(project.parent / "approval.json")["generation_policy_approved"]
    apply_review(project, queue, {row["sample_id"]: dict(decision="discard", listened=True)}, "pilot")
    assert not read_json(project.parent / "approval.json")["generation_policy_approved"]
    with pytest.raises(ValueError, match="Approve"):
        generate(project, method="design", model=SyntheticFixture())


def test_later_reference_review_does_not_break_positive_admission(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora", positive_originals=2)
    root, config = load_project(project)
    policy = read_json(root / "approval.json")
    policy.update(state="accepted", generation_policy_approved=True, generation_method="design",
                  tts_settings_id=identity(config["tts"]))
    write_json(root / "approval.json", policy)
    result = generate(project, method="design", model=SyntheticFixture())
    assert result["full_target_reached"]
    rows = read_rows(root / config["manifests"]["positive_train"])
    policy["accepted_reference_families"] = ["validation_voice_100"]
    write_json(root / "approval.json", policy)
    validate_admission(root, rows, config, "positive")
    assert rows[0]["admission"]["path"].startswith("review/evidence/")


def test_content_review_evidence_survives_another_queue(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    root = project.parent
    bindings = []
    for number in range(2):
        audio = root / "data" / f"noise{number}.wav"
        sf.write(audio, np.random.default_rng(number).normal(0, .03, 1600), 16000, subtype="PCM_16")
        row = dict(sample_id=str(number), path=str(audio.relative_to(root)), sha256=digest(audio),
                   provenance="public_dataset", source_url="https://example.org/test-fixture", license="fixture")
        queue = root / "review" / f"queue{number}.jsonl"
        queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
        result = apply_review(project, queue, {str(number): dict(decision="keep", listened=True)}, "content")
        bindings.append(result["admission"])
    assert bindings[0] != bindings[1]
    assert all(bound_file(root, binding).is_file() for binding in bindings)


def test_reviewed_reference_bank_reusable_across_three_projects(tmp_path):
    from korean_openwakeword.reference_bank import import_references
    original = create_project(tmp_path / "original", "보라야", "bora")
    generated = generate_references(original, limit=1, model=SyntheticFixture())
    row = read_rows(generated["review_queue"])[0]
    apply_review(original, generated["review_queue"], {row["sample_id"]: dict(decision="keep", listened=True)}, "references")
    second = create_project(tmp_path / "second", "다솜아", "dasom")
    assert import_references(second, original)["reused_families"] == 1
    third = create_project(tmp_path / "third", "아리아", "aria")
    assert import_references(third, second)["reused_families"] == 1
    copied = read_rows(third.parent / "data/references/train/references.jsonl")[0]
    assert copied["sha256"] == row["sha256"] and copied["human_reference_used"] is False


@pytest.mark.parametrize("key,value", [("clean_probability", 2), ("gain_db", [3, -6]), ("peak_limit", float("nan"))])
def test_bad_augmentation_rejected_before_heavy_work(tmp_path, key, value):
    project = create_project(tmp_path / "word", "보라야", "bora")
    config = read_json(project)
    config["augmentation"][key] = value
    with pytest.raises(ValueError):
        validate_project(config)


def test_wrong_heldout_noise_split_is_rejected(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    root, config = load_project(project)
    row = dict(sample_id="noise", source_id="public:noise", split="train", provenance="public_dataset",
               source_url="https://example.org/fixture", license="fixture", category="environment")
    (root / config["manifests"]["noise_validation"]).write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert not audit_project(project, check_files=False)["passed"]


def test_different_utterances_of_same_recording_cannot_cross_splits(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    root, config = load_project(project)
    for pool, split in (("noise", "train"), ("noise_validation", "validation")):
        row = dict(sample_id=split, source_id="video:" + split, canonical_recording_id="youtube:same-original",
                   split=split, provenance="public_dataset", source_url="https://example.org/fixture",
                   license="fixture", category="competing_speech")
        (root / config["manifests"][pool]).write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = audit_project(project, check_files=False)
    assert not report["passed"] and any(leak["key"] == "recording" for leak in report["leaks"])


def test_parent_offsets_remove_overlap_from_unique_hours():
    from korean_openwakeword.data import coverage_interval, union_hours
    intervals = {}
    for start in (0, 3):
        row = dict(source_id=f"utterance:{start}", canonical_recording_id="video:parent", parent_start_s=start)
        key, span = coverage_interval(row, 0, 6)
        intervals.setdefault(key, []).append(span)
    assert union_hours(intervals) == 9 / 3600


def test_mutated_final_config_cannot_open_generation(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    root = project.parent
    bundle = root / "bundle"
    write_json(bundle / "manifest.json", {"fixture": "lock integrity only"})
    write_json(root / "final_lock.json", {"project_config_sha256": digest(project), "bundle": "bundle",
                                         "bundle_manifest_sha256": digest(bundle / "manifest.json")})
    config = read_json(project)
    config["runtime"]["threshold"] = .8
    write_json(project, config)
    with pytest.raises(ValueError, match="changed after freeze"):
        generate_references(project, split="final", limit=1, model=SyntheticFixture())


def test_duplicate_negative_cannot_inflate_fp_hours(tmp_path):
    from korean_openwakeword.runtime import sample_bundle
    from korean_openwakeword.evaluate import evaluate
    project = create_project(tmp_path / "word", "춘식아", "chunsik")
    bundle = sample_bundle()
    audio = next(row for row in read_json(bundle / "audio_manifest.json") if row["kind"] == "negative")
    rows = [dict(audio, sample_id=f"duplicate_{i}", source_id=f"renamed:{i}", path=str(bundle / audio["path"])) for i in range(2)]
    manifest = project.parent / "data/eval/duplicated.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Repeated Negative"):
        evaluate(project, bundle, project.parent / "reports/duplicated", manifest=manifest, smoke=True)


def test_replay_keeps_human_final_condition_metadata():
    from korean_openwakeword.tune import replay
    row = dict(sample_id="protocol_fixture", source_id="fixture", kind="positive", category="clean",
               provenance="human", speaker_id="fictional_protocol_speaker", distance_m=3,
               microphone="fixture microphone", direction="side",
               calls=[dict(start_s=2.4, end_s=3.2)])
    result = replay([row], np.full((50, 1), .9), np.array([0, 50]), ["member"],
                    dict(threshold=.5, consecutive_frames=2, cooldown_s=1, warmup_s=2, call_postroll_s=.5))
    call = result[0]["results"]["member"]["calls"][0]
    assert (call["distance_m"], call["microphone"], call["direction"]) == (3, "fixture microphone", "side")


def test_precompiled_noise_sampler_preserves_pcm_and_metadata(tmp_path):
    from korean_openwakeword.features import augment, prepare_noise_index
    project = create_project(tmp_path / "word", "보라야", "bora")
    root, config = load_project(project)
    audio = root / "data" / "fixture.wav"
    t = np.arange(32000) / 16000
    sf.write(audio, .1 * np.sin(2 * np.pi * 230 * t), 16000, subtype="PCM_16")
    row = dict(sample_id="positive", source_id="qwen:fixture", path=str(audio.relative_to(root)),
               call_start_s=.2, call_end_s=1.0)
    noises = [dict(sample_id=category, source_id=category, category=category, language="Korean", path=row["path"])
              for category in config["augmentation"]["category_weights"]]
    index = prepare_noise_index(config["augmentation"], noises)
    for round_id in range(15):
        a, details_a = augment(root, row, "positive", round_id, config["augmentation"], noises, [])
        b, details_b = augment(root, row, "positive", round_id, config["augmentation"], noises, [], index)
        assert np.array_equal(a, b) and details_a == details_b
