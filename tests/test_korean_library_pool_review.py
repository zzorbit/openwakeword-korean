"""Pool consent cannot silently expand actual human listening or frozen hashes."""
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, write_json
from korean_openwakeword.data import validate_admission
from korean_openwakeword.pool_review import approve_pool
from korean_openwakeword.project import create_project, load_project


def _rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _bound(path):
    return {"path": str(path), "sha256": digest(path)}


@pytest.fixture
def pool(tmp_path):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    root = project.parent
    rows, representatives = [], []
    for dataset, category in (("audioset", "environment"), ("audioset", "competing_speech"), ("fma", "music"), ("mit", "rir")):
        for index in range(2):
            n = len(rows)
            path = root / "data" / f"original_{n}.wav"
            sf.write(path, np.random.default_rng(n).normal(0, .08, 4000), 16000, subtype="PCM_16")
            row = {"sample_id": f"source_{n}", "canonical_source_id": f"{dataset}:{n}",
                   "split": "train", "path": str(path), "sha256": digest(path),
                   "dataset_id": dataset, "category": category, "category_basis": "fixture publisher label",
                   "provenance": "public_dataset", "license": "test fixture only", "source_url": "https://example.org/fixture",
                   "technical_pass": True, "individual_human_listening": False, "speech": category == "competing_speech"}
            if category == "rir":
                row["canonical_room_id"] = f"mit:actual_room_{n}"
            rows.append(row)
            if index == 0:
                preview = root / "review" / f"preview_{n}.wav"
                sf.write(preview, sf.read(path)[0][:2000], 16000, subtype="PCM_16")
                representatives.append({"sample_id": f"preview_{n}", "path": str(preview), "sha256": digest(preview),
                                        "dataset_id": dataset, "category": category, "canonical_source_id": row["canonical_source_id"],
                                        "reviewed_source": _bound(path)})
    manifest, queue, answers = root / "review/pool.jsonl", root / "review/representatives.jsonl", root / "review/answers.json"
    _rows(manifest, rows); _rows(queue, representatives)
    decisions = {r["sample_id"]: {"listened": True, "decision": "keep"} for r in representatives}
    result = {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "2026-10-13T10:00:00+00:00",
              "source_scope": "Human dispositions entered through the local review screen", "decisions": decisions,
              "queue_sha256": digest(queue), "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in representatives},
              "source_manifest_sha256": digest(manifest), "representative_queue_sha256": digest(queue),
              "whole_pool_policy_approved": True}
    write_json(answers, result)
    return dict(project=project, root=root, rows=rows, representatives=representatives,
                manifest=manifest, queue=queue, answers=answers, output=root / "review/evidence/pool_admission.json")


def _approve(pool):
    return approve_pool(pool["project"], source_manifest=_bound(pool["manifest"]),
                        representative_queue=_bound(pool["queue"]), answers=_bound(pool["answers"]), output=pool["output"])


def _refresh(pool):
    _rows(pool["manifest"], pool["rows"]); _rows(pool["queue"], pool["representatives"])
    result = read_json(pool["answers"])
    result.update(source_manifest_sha256=digest(pool["manifest"]), representative_queue_sha256=digest(pool["queue"]),
                  queue_sha256=digest(pool["queue"]), queued_audio_sha256={r["sample_id"]: r["sha256"] for r in pool["representatives"]})
    result["decisions"] = {r["sample_id"]: result["decisions"].get(r["sample_id"], {"listened": True, "decision": "keep"}) for r in pool["representatives"]}
    write_json(pool["answers"], result)


def test_explicit_representative_pool_policy_preserves_exact_and_unheard_scope(pool):
    result = _approve(pool)
    record = read_json(pool["output"])
    assert result["source_count"] == 8 and result["representative_preview_count"] == 4
    assert set(record["accepted_source_sha256"]) == {r["sha256"] for r in pool["rows"]}
    assert len(record["representative_source_sha256"]) == 4
    assert len(record["unheard_source_policy_admitted_sha256"]) == 4
    assert record["individual_source_listening_claimed"] is False
    assert record["whole_sources_heard_claimed"] is False and record["wakeword_absence_certified"] is False
    assert len(record["technical_checks"]) == 8 and len(record["strata"]) == 4
    _, config = load_project(pool["project"])
    admitted = [{**r, "admission": result["admission"]} for r in pool["rows"]]
    validate_admission(pool["root"], admitted, config, "noise")
    with pytest.raises(ValueError, match="immutable"):
        _approve(pool)


@pytest.mark.parametrize("consent", [False, None, 1, "true"])
def test_no_implicit_or_truthy_pool_consent(pool, consent):
    answers = read_json(pool["answers"]); answers["whole_pool_policy_approved"] = consent
    write_json(pool["answers"], answers)
    with pytest.raises(ValueError, match="Explicit"):
        _approve(pool)
    assert not pool["output"].exists()


@pytest.mark.parametrize("field", ["source_manifest_sha256", "representative_queue_sha256", "queue_sha256", "queued_audio_sha256"])
def test_saved_consent_binds_frozen_source_queue_and_preview_hashes(pool, field):
    answers = read_json(pool["answers"]); answers[field] = {} if field == "queued_audio_sha256" else "0" * 64
    write_json(pool["answers"], answers)
    with pytest.raises(ValueError, match="bind"):
        _approve(pool)


def test_exact_source_tamper_is_rejected(pool):
    path = pool["rows"][-1]["path"]
    sf.write(path, np.zeros(1000), 16000)
    with pytest.raises(ValueError, match="changed bound file"):
        _approve(pool)


def test_frozen_manifest_hash_is_checked_before_any_approval(pool):
    manifest_binding = _bound(pool["manifest"])
    pool["rows"][0]["license"] = "changed after consent"
    _rows(pool["manifest"], pool["rows"])
    with pytest.raises(ValueError, match="changed bound file"):
        approve_pool(pool["project"], source_manifest=manifest_binding, representative_queue=_bound(pool["queue"]),
                     answers=_bound(pool["answers"]), output=pool["output"])


@pytest.mark.parametrize("decision,listened", [("keep", False), ("review", True), ("discard", True)])
def test_unheard_or_unresolved_representatives_block_the_whole_pool(pool, decision, listened):
    result = read_json(pool["answers"])
    result["decisions"][pool["representatives"][0]["sample_id"]] = {"decision": decision, "listened": listened}
    write_json(pool["answers"], result)
    with pytest.raises(ValueError, match="resolution"):
        _approve(pool)


def test_representative_each_dataset_and_category_required(pool):
    pool["representatives"] = pool["representatives"][:-1]
    _refresh(pool)
    with pytest.raises(ValueError, match="every declared"):
        _approve(pool)


@pytest.mark.parametrize("field,value", [("canonical_source_id", "impostor:recording"), ("dataset_id", "other_dataset"), ("category", "music")])
def test_preview_cannot_be_relabelled_or_rebind_canonical_identity(pool, field, value):
    pool["representatives"][0][field] = value
    _refresh(pool)
    with pytest.raises(ValueError, match="identity changed"):
        _approve(pool)


@pytest.mark.parametrize("split", ["validation", "final"])
def test_no_heldout_pool_admission_or_final_opening(pool, split, monkeypatch):
    pool["rows"][0]["split"] = split
    _refresh(pool)
    opened = []
    original = sf.SoundFile
    def track(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(sf, "SoundFile", track)
    with pytest.raises(ValueError, match="reserved"):
        _approve(pool)
    assert pool["rows"][0]["path"] not in opened


def test_existing_validation_recording_identity_blocks_train_pool(pool):
    root, config = load_project(pool["project"])
    heldout = {**pool["rows"][0], "sample_id": "heldout_recording", "split": "validation", "kind": "negative"}
    _rows(root / config["manifests"]["development"], [heldout])
    with pytest.raises(ValueError, match="audit failed"):
        _approve(pool)


def test_nonfinite_actual_samples_fail_even_with_passing_metadata(pool):
    row = pool["rows"][-1]
    sf.write(row["path"], np.full(4000, np.nan), 16000, subtype="FLOAT")
    row["sha256"] = digest(row["path"])
    _refresh(pool)
    with pytest.raises(ValueError, match="Nonfinite"):
        _approve(pool)


def test_pool_rows_cannot_claim_individual_listening(pool):
    pool["rows"][1]["individual_human_listening"] = True
    _refresh(pool)
    with pytest.raises(ValueError, match="must not claim"):
        _approve(pool)


def test_missing_rir_room_identity_rejected(pool):
    pool["rows"][-1].pop("canonical_room_id")
    _refresh(pool)
    with pytest.raises(ValueError, match="room identity"):
        _approve(pool)


def test_wrong_project_human_review_rejected(pool):
    result = read_json(pool["answers"]); result["wakeword"] = "춘식아"
    write_json(pool["answers"], result)
    with pytest.raises(ValueError, match="project's word"):
        _approve(pool)


def test_known_final_source_relabelled_train_is_rejected_before_audio_access(pool, monkeypatch):
    import korean_openwakeword.pool_review as module
    root, config = load_project(pool["project"])
    _rows(root / config["manifests"]["final"], [{**pool["rows"][0], "sample_id": "final_reserved", "split": "final", "kind": "negative"}])
    inspected = []
    original = module.bound_file
    def record(root, binding):
        inspected.append(str(binding["path"]))
        return original(root, binding)
    monkeypatch.setattr(module, "bound_file", record)
    with pytest.raises(ValueError, match="before audio access"):
        _approve(pool)
    assert pool["rows"][0]["path"] not in inspected


@pytest.fixture
def synthetic_pool(pool):
    root, config = load_project(pool["project"])
    reference_dir = root / "data/references/train"
    reference_dir.mkdir(parents=True)
    references = []
    for index in range(2):
        path = reference_dir / f"voice{index}.wav"
        sf.write(path, np.random.default_rng(50 + index).normal(0, .05, 8000), 16000, subtype="PCM_16")
        references.append({"sample_id": f"reference{index}", "voice_family_id": f"synthetic_train{index}",
                           "split": "train", "path": str(path), "sha256": digest(path), "human_reference_used": False,
                           "provenance": "qwen_synthetic_reference", "status": "accepted"})
    _rows(reference_dir / "references.jsonl", references)
    approval = read_json(root / "approval.json")
    approval["accepted_reference_families"] = [r["voice_family_id"] for r in references]
    write_json(root / "approval.json", approval)
    implementation = root / "fixture_generator.py"
    implementation.write_text("# Frozen test-only synthetic generation evidence.\n", encoding="utf-8")
    contract = {"kind": "qwen_hard_negative_generation_contract_v1", "word": "보라야", "split": "train",
                "human_reference_used": False, "method": "clone", "tts": config["tts"], "texts": ["돌아와", "보리야"],
                "references": references, "implementation": _bound(implementation)}
    contract_path = root / "review/hard_contract.json"
    write_json(contract_path, contract)
    for phrase_index, phrase in enumerate(contract["texts"]):
        for ref_index, reference in enumerate(references):
            for attempt in range(2):
                n = len(pool["rows"])
                audio = root / "data" / f"hard{n}.wav"
                sf.write(audio, np.random.default_rng(100 + n).normal(0, .05, 8000), 16000, subtype="PCM_16")
                row = {"sample_id": f"hard{n}", "canonical_source_id": f"qwen:hard:{n}", "split": "train",
                       "path": str(audio), "sha256": digest(audio), "dataset_id": "qwen_bora_hard_v1",
                       "category": "competing_speech", "category_basis": "declared non-call phrase",
                       "technical_pass": True, "individual_human_listening": False, "provenance": "qwen_tts",
                       "human_reference_used": False, "speech": True, "text": phrase,
                       "start_s": 0.0, "end_s": .5,
                       "voice_family_id": reference["voice_family_id"], "reference_sha256": reference["sha256"],
                       "generation_contract": _bound(contract_path), "generation_contract_id": identity(contract),
                       "tts_settings_id": identity(config["tts"])}
                pool["rows"].append(row)
                if attempt == 0:
                    pool["representatives"].append({"sample_id": f"hard_preview{n}", "path": str(audio), "sha256": row["sha256"],
                                                     "dataset_id": row["dataset_id"], "category": row["category"],
                                                     "canonical_source_id": row["canonical_source_id"], "reviewed_source": _bound(audio)})
    _refresh(pool)
    pool.update(contract=contract, contract_path=contract_path)
    return pool


def _refresh_contract(pool):
    write_json(pool["contract_path"], pool["contract"])
    for row in pool["rows"]:
        if row["provenance"] == "qwen_tts":
            row.update(generation_contract=_bound(pool["contract_path"]), generation_contract_id=identity(pool["contract"]))
    _refresh(pool)


@pytest.fixture
def per_phrase_pool(synthetic_pool):
    from korean_openwakeword.hard_tts import _combination, _seed
    pool = synthetic_pool
    contract = pool["contract"]
    contract.update(kind="qwen_hard_negative_generation_contract_v2", phrase_count=2,
                    samples_per_phrase=4, original_count=8, max_attempts_per_slot=3,
                    allocation="every_phrase_cycles_train_references_v1")
    _, config = load_project(pool["project"])
    slots = {}
    for slot in range(contract["original_count"]):
        phrase, reference = _combination(slot, contract["texts"], contract["references"])
        slots.setdefault((phrase, reference["voice_family_id"]), []).append((slot, reference))
    for row in pool["rows"]:
        if row["provenance"] == "qwen_tts":
            slot, reference = slots[(row["text"], row["voice_family_id"])].pop(0)
            row.update(slot=slot, attempt=0, seed=_seed(config, slot, 0, row["text"], reference))
    _refresh_contract(pool)
    return pool


def test_per_phrase_contract_admits_exact_reviewed_scope(per_phrase_pool):
    result = _approve(per_phrase_pool)
    assert result["source_count"] == 16
    record = read_json(per_phrase_pool["output"])
    assert len(record["synthetic_generation_contracts"]) == 1
    assert len(record["synthetic_phrase_family_strata"]) == 4


@pytest.mark.parametrize("field,value", [
    ("original_count", 7), ("phrase_count", 3), ("samples_per_phrase", 3),
    ("samples_per_phrase", True), ("allocation", "unbound"), ("max_attempts_per_slot", None),
])
def test_per_phrase_contract_rejects_changed_derived_budget(per_phrase_pool, field, value):
    per_phrase_pool["contract"][field] = value
    _refresh_contract(per_phrase_pool)
    with pytest.raises(ValueError, match="per-phrase count/allocation"):
        _approve(per_phrase_pool)
    assert not per_phrase_pool["output"].exists()


@pytest.mark.parametrize("field,value", [("slot", 0), ("slot", 8), ("attempt", 3), ("seed", -1)])
def test_per_phrase_rows_must_bind_their_declared_slot_pair_and_seed(per_phrase_pool, field, value):
    per_phrase_pool["rows"][-1][field] = value
    _refresh(per_phrase_pool)
    with pytest.raises(ValueError, match="per-phrase slot"):
        _approve(per_phrase_pool)


def test_per_phrase_pool_cannot_admit_two_accepted_attempts_for_one_slot(per_phrase_pool):
    from korean_openwakeword.hard_tts import _seed
    same_pair = [r for r in per_phrase_pool["rows"] if r["provenance"] == "qwen_tts"
                 and r["text"] == "보리야" and r["voice_family_id"] == "synthetic_train1"]
    first, duplicate = same_pair
    _, config = load_project(per_phrase_pool["project"])
    reference = per_phrase_pool["contract"]["references"][1]
    duplicate.update(slot=first["slot"], attempt=1,
                     seed=_seed(config, first["slot"], 1, duplicate["text"], reference))
    _refresh(per_phrase_pool)
    with pytest.raises(ValueError, match="duplicate accepted slot"):
        _approve(per_phrase_pool)


def test_mixed_public_and_qwen_hard_pool_keeps_generation_and_hearing_scope(synthetic_pool):
    result = _approve(synthetic_pool)
    record = read_json(synthetic_pool["output"])
    assert result["source_count"] == 16 and result["representative_preview_count"] == 8
    assert len(record["synthetic_generation_contracts"]) == 1
    assert len(record["synthetic_phrase_family_strata"]) == 4
    root, config = load_project(synthetic_pool["project"])
    hard = [{**r, "admission": result["admission"]} for r in synthetic_pool["rows"] if r["provenance"] == "qwen_tts"]
    validate_admission(root, hard, config, "hard_negative")


def test_each_synthetic_phrase_family_needs_its_actual_heard_representative(synthetic_pool):
    synthetic_pool["representatives"] = synthetic_pool["representatives"][:-1]
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="phrase/family"):
        _approve(synthetic_pool)


@pytest.mark.parametrize("field,value,match", [
    ("human_reference_used", True, "synthetic-only"),
    ("split", "final", "Train synthetic-only"),
    ("word", "춘식아", "this word"),
])
def test_synthetic_generation_contract_scope_cannot_change(synthetic_pool, field, value, match):
    synthetic_pool["contract"][field] = value
    _refresh_contract(synthetic_pool)
    with pytest.raises(ValueError, match=match):
        _approve(synthetic_pool)


def test_synthetic_human_reference_row_refused(synthetic_pool):
    synthetic_pool["rows"][-1]["human_reference_used"] = True
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="without human reference"):
        _approve(synthetic_pool)


def test_synthetic_changed_generator_code_refused(synthetic_pool):
    path = synthetic_pool["contract"]["implementation"]["path"]
    from pathlib import Path
    Path(path).write_text("# Replaced generator after the frozen contract.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed bound file"):
        _approve(synthetic_pool)


def test_synthetic_changed_tts_settings_refused(synthetic_pool):
    synthetic_pool["contract"]["tts"]["seed"] += 1
    _refresh_contract(synthetic_pool)
    with pytest.raises(ValueError, match="settings differ"):
        _approve(synthetic_pool)


def test_synthetic_undeclared_reference_sha_refused(synthetic_pool):
    synthetic_pool["rows"][-1]["reference_sha256"] = "0" * 64
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="not declared"):
        _approve(synthetic_pool)


def test_synthetic_final_reference_is_refused_before_its_audio_is_opened(synthetic_pool, monkeypatch):
    import korean_openwakeword.pool_review as module
    final_reference = synthetic_pool["contract"]["references"][0]
    final_reference["split"] = "final"
    _refresh_contract(synthetic_pool)
    inspected = []
    original = module.bound_file
    def record(root, binding):
        inspected.append(str(binding["path"]))
        return original(root, binding)
    monkeypatch.setattr(module, "bound_file", record)
    with pytest.raises(ValueError, match="approved Train family"):
        _approve(synthetic_pool)
    assert final_reference["path"] not in inspected


def test_long_synthetic_original_with_short_complete_phrase_interval_is_admitted(synthetic_pool):
    from korean_openwakeword.data import seek_audio
    row = synthetic_pool["rows"][-1]
    original = np.zeros(64000)
    original[35200:43200] = np.random.default_rng(777).normal(0, .05, 8000)
    sf.write(row["path"], original, 16000, subtype="PCM_16")
    row.update(sha256=digest(row["path"]), start_s=2.2, end_s=2.7)
    _refresh(synthetic_pool)
    result = _approve(synthetic_pool)
    assert result["source_count"] == 16
    record = read_json(synthetic_pool["output"])
    evidence = next(r for r in record["technical_checks"] if r["sha256"] == row["sha256"])
    assert evidence["frames"] / evidence["sample_rate"] == 4.0
    assert evidence["phrase_interval_s"] == [2.2, 2.7]
    samples, offset = seek_audio(synthetic_pool["root"], row, 2)
    expected = sf.read(row["path"], start=35200, stop=43200)[0]
    assert offset == 2.2 and np.array_equal(samples, expected)


@pytest.mark.parametrize("start,end", [
    (None, .5), (0, None), (float("inf"), .5), (0, float("inf")),
    (-.1, .5), (.5, .5), (.6, .4), (0, .51), (True, .5), (0, "0.5"),
])
def test_invalid_synthetic_phrase_interval_cannot_be_policy_admitted(synthetic_pool, start, end):
    synthetic_pool["rows"][-1].update(start_s=start, end_s=end)
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="complete phrase interval"):
        _approve(synthetic_pool)


def test_synthetic_phrase_longer_than_feature_fit_is_refused_even_inside_long_original(synthetic_pool):
    row = synthetic_pool["rows"][-1]
    sf.write(row["path"], np.random.default_rng(888).normal(0, .05, 64000), 16000, subtype="PCM_16")
    row.update(sha256=digest(row["path"]), start_s=1.0, end_s=2.85)
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="at most 1.84s"):
        _approve(synthetic_pool)


def test_synthetic_category_basis_can_come_from_verified_generation_contract(synthetic_pool):
    synthetic_pool["rows"][-1].pop("category_basis")
    _refresh(synthetic_pool)
    assert _approve(synthetic_pool)["source_count"] == 16


@pytest.mark.parametrize("per_phrase", [False, True])
def test_fresh_generated_references_review_pilot_and_hard_pool_need_no_manifest_rewrite(tmp_path, monkeypatch, per_phrase):
    """The public fresh-project flow keeps reference rows pending/immutable.

    A user review approves family IDs in approval.json. That is the authority
    used by both generators; pool admission must recognize the same evidence.
    No Qwen weights/downloads are needed for this procedural audio fixture.
    """
    from korean_openwakeword import tts
    from korean_openwakeword.common import read_rows
    from korean_openwakeword.hard_tts import generate_hard
    from korean_openwakeword.review import apply_review, apply_pool_review

    class ProceduralQwen:
        def __init__(self):
            self.calls = 0
        def _wave(self):
            self.calls += 1
            time = np.arange(16000) / 16000
            return [.1 * np.sin(2 * np.pi * (210 + self.calls * 13) * time)], 16000
        def generate_voice_design(self, **kwargs):
            return self._wave()
        def create_voice_clone_prompt(self, **kwargs):
            return kwargs["ref_audio"]
        def generate_voice_clone(self, **kwargs):
            return self._wave()

    monkeypatch.setattr(tts, "_seed", lambda seed: None)
    project = create_project(tmp_path / "fresh", "보라야", "bora", positive_originals=4)
    root = project.parent
    model = ProceduralQwen()
    references = tts.generate_references(project, "train", 2, model=model)
    reference_queue = references["review_queue"]
    original_bank_sha = digest(reference_queue)
    generated_rows = read_rows(reference_queue)
    assert {row["status"] for row in generated_rows} == {"pending_review"}
    apply_review(project, reference_queue, {r["sample_id"]: {"decision": "keep", "listened": True} for r in generated_rows}, "references")
    pilot = tts.generate(project, "train", 4, True, "clone", model=model)
    pilot_rows = read_rows(pilot["review_queue"])
    assert pilot["full_target_reached"]
    apply_review(project, pilot["review_queue"], {r["sample_id"]: {"decision": "keep", "listened": True} for r in pilot_rows}, "pilot")
    budget = {"samples_per_phrase": 2} if per_phrase else {"count": 4}
    candidates = generate_hard(project, ["보리야", "돌아와"], model=model, **budget)
    assert candidates["full_target_reached"]
    sources = [r for r in read_rows(candidates["candidates"]) if r["technical_pass"]]
    source_manifest = root / "review/fresh_sources.jsonl"
    queue = root / "review/fresh_representatives.jsonl"
    _rows(source_manifest, sources)
    previews = [{**r, "sample_id": "preview_" + r["sample_id"], "reviewed_source": {"path": r["path"], "sha256": r["sha256"]}} for r in sources]
    _rows(queue, previews)
    result = apply_pool_review(project, queue, source_manifest,
                              {r["sample_id"]: {"decision": "keep", "listened": True} for r in previews}, True)
    assert result["source_count"] == 4
    admission = read_json(root / result["admission"]["path"])
    assert len(admission["synthetic_phrase_family_strata"]) == 4
    assert digest(reference_queue) == original_bank_sha
    assert {r["status"] for r in read_rows(reference_queue)} == {"pending_review"}
    _, config = load_project(project)
    validate_admission(root, [{**r, "admission": result["admission"]} for r in sources], config, "hard_negative")


@pytest.mark.parametrize("status", ["pending_review", "accepted"])
def test_reference_row_status_cannot_replace_actual_accepted_family_ids(synthetic_pool, status):
    from korean_openwakeword.common import read_rows
    reference_path = synthetic_pool["root"] / "data/references/train/references.jsonl"
    references = read_rows(reference_path)
    for row in references:
        row["status"] = status
    _rows(reference_path, references)
    for row in synthetic_pool["contract"]["references"]:
        row["status"] = status
    _refresh_contract(synthetic_pool)
    approval_path = synthetic_pool["root"] / "approval.json"
    approval = read_json(approval_path)
    approval["accepted_reference_families"] = []
    write_json(approval_path, approval)
    with pytest.raises(ValueError, match="approved Train family"):
        _approve(synthetic_pool)
