"""The candidate pool is resumable and only the explicit chosen audio is used."""
from copy import deepcopy
import random

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import reference_candidates as candidates, tts
from korean_openwakeword.common import digest, read_json, snapshot_json, write_immutable_rows, write_json
from korean_openwakeword.reference_qc import _check_wave
from korean_openwakeword.reference_voice_review import overlay_reference_banks


def waveform(number):
    return (.15 * np.sin(np.arange(4800) * (number + 100) * 2 * np.pi / 16000)).astype(np.float32)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "voice-project"
    root.mkdir()
    project = root / "project.json"
    config = {"tts": {"design_model": "fixture-model", "reference_text": "오늘은 좋은 날입니다.",
                      "generation": {"temperature": .8}}}
    write_json(project, config)
    families, originals = [], []
    for split, count in (("train", 2), ("validation", 3)):
        for gender in candidates.GENDERS:
            for age in candidates.AGES:
                for number in range(count):
                    fid = f"{split}_{gender}_{age.replace(' ', '_')}_{number}"
                    families.append({"family_id": fid, "split": split, "gender": gender, "age_group": age})
                    audio = root / "original" / f"{fid}.wav"
                    audio.parent.mkdir(exist_ok=True)
                    sf.write(audio, waveform(len(originals)), 16000, subtype="FLOAT")
                    originals.append({"sample_id": fid, "voice_family_id": fid, "split": split,
                        "path": audio.relative_to(root).as_posix(), "sha256": digest(audio),
                        "text": config["tts"]["reference_text"], "model": config["tts"]["design_model"],
                        "description": f"Original {gender} {age}", "style": "neutral", "speed": "natural",
                        "intonation": "natural", "vocal_effort": "normal", "seed": 123,
                        "provenance": "qwen_synthetic_reference", "human_reference_used": False,
                        "voice_design": {"gender": {"value": gender}, "age_group": {"value": age}}})
    write_json(root / "voices.json", {"families": families})
    monkeypatch.setattr(candidates, "load_project", lambda project: (root, config))
    monkeypatch.setattr(tts, "_seed", lambda seed: None)
    saved = candidates.create_candidate_review(originals)
    class Model:
        def __init__(self):
            self.calls = []
            self.loads = 0
            self.fail_at = None

        def load(self, *args):
            self.loads += 1
            assert (root / ".studio/reference-candidates/jobs").is_dir()
            return self

        def generate_voice_design(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == self.fail_at:
                raise RuntimeError("interrupted inference")
            return [waveform(500 + len(self.calls))], 16000
    model = Model()
    return root, project, config, originals, saved, model


def batch(pool, *, group=None, rng_seed=1):
    root, project, config, originals, saved, model = pool
    key = group or candidates.active_candidate_group(saved)
    plan = candidates.plan_candidate_batch(project, saved, key, rng=random.Random(rng_seed))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / plan["batch_id"] / "audio"
    rows = candidates.generate_candidate_batch(project, output, plan["batch_id"], plan, [], load_model=model.load)
    candidates.ingest_candidate_batch(root, saved, plan, rows)
    return plan, rows, output


def choose(pool, row):
    root, _, _, _, saved, _ = pool
    for action in ("hear", "toggle"):
        candidates.candidate_action(root, saved, action, candidate_id=row["candidate_id"], sha256=row["sha256"])


def test_groups_have_fixed_order_and_training_and_evaluation_quotas(pool):
    *_, originals, saved, model = pool
    groups = candidates.candidate_groups(originals)
    assert len(groups) == 20
    assert [(row["gender"], row["age"]) for row in groups[:10]] == [
        (gender, age) for gender in candidates.GENDERS for age in candidates.AGES]
    assert [row["target"] for row in groups] == [2] * 10 + [3] * 10
    assert groups[0]["key"] == "train:man:child"
    assert groups[-1]["key"] == "validation:woman:older adult"
    for train_target in (10, 50):
        expanded = []
        for group in groups:
            original = next(row for row in originals if row["voice_family_id"] == group["families"][0])
            count = train_target if group["split"] == "train" else 3
            expanded.extend({**original, "voice_family_id": f"{group['key']}:{number}"} for number in range(count))
        assert [group["target"] for group in candidates.candidate_groups(expanded)][:10] == [train_target] * 10
    assert not saved["heard"] and not saved["selected"]


def test_plan_uses_all_five_age_valid_options_fresh_seeds_and_korean_model(pool):
    root, project, config, _, saved, model = pool
    plan, rows, output = batch(pool)
    assert len(rows) == len(model.calls) == 5 and model.loads == 1
    assert read_json(output / "plan.json") == plan
    assert read_json(output.parent / "progress.json")["completed"] == 5
    assert len({seed for item in plan["candidates"] for seed in item["attempt_seeds"]}) == 20
    for planned, row, request in zip(plan["candidates"], rows, model.calls):
        assert set(planned["voice_controls"]) == set(candidates.FIELDS)
        for field, oid in planned["voice_controls"].items():
            assert oid in {option["id"] for option in candidates.OPTIONS["ages"]["child"]["controls"][field]}
        assert request == {"language": "Korean", "text": planned["text"],
                           "instruct": planned["design_instruction"], **plan["generation"]}
        assert row["speed"] == planned["voice_controls"]["pace"]
        assert row["intonation"] == planned["voice_controls"]["intonation"]
        assert candidates.candidate_audio(root, saved, row["candidate_id"], row["sha256"]).is_file()
    assert not saved["heard"]  # Generation or displaying candidates never invents human listening.
    elder = candidates.OPTIONS["ages"]["older adult"]
    controls = {field: options[0]["id"] for field, options in elder["controls"].items()}
    assert "80" in candidates.candidate_instruction("man", "older adult", controls)


def test_interrupted_batch_resumes_same_plan_without_regenerating_completed_audio(pool):
    root, project, _, _, saved, model = pool
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved), rng=random.Random(2))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / plan["batch_id"] / "audio"
    model.fail_at = 3
    with pytest.raises(RuntimeError, match="interrupted"):
        candidates.generate_candidate_batch(project, output, "job", plan, [], load_model=model.load)
    assert read_json(output / "plan.json") == plan
    existing = {path.name: digest(path) for path in output.glob("*.wav")}
    assert len(existing) == 2
    model.fail_at = None
    rows = candidates.generate_candidate_batch(project, output, "job", plan, [], load_model=model.load)
    assert all(digest(output / name) == sha for name, sha in existing.items())
    assert len(model.calls) == 6 and len(rows) == 5
    loads = model.loads
    assert candidates.generate_candidate_batch(project, output, "job", plan, [],
        load_model=lambda *args: pytest.fail("Completed batch must not load a model")) == rows
    assert model.loads == loads
    candidates.ingest_candidate_batch(root, saved, plan, rows)
    revision = saved["revision"]
    candidates.ingest_candidate_batch(root, saved, plan, rows)
    assert saved["revision"] == revision  # Crash recovery can safely reconcile twice.


def test_zero_choices_allowed_selections_survive_more_and_exact_cap_is_enforced(pool):
    root, project, _, _, saved, _ = pool
    first, rows, _ = batch(pool)
    second, more, _ = batch(pool, rng_seed=3)  # No requirement to listen or choose before more.
    assert first["batch_id"] != second["batch_id"]
    assert [row["number"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["number"] for row in more] == [6, 7, 8, 9, 10]
    candidates.candidate_action(root, saved, "toggle", candidate_id=more[0]["candidate_id"], sha256=more[0]["sha256"])
    assert saved["selected"][first["group_key"]] == [more[0]["candidate_id"]] and not saved["heard"]
    candidates.candidate_action(root, saved, "toggle", candidate_id=more[0]["candidate_id"], sha256=more[0]["sha256"])
    choose(pool, rows[0])  # Past batch choices are still addressable.
    saved_path = root / ".studio/candidate-state.json"
    write_json(saved_path, saved)
    restored = read_json(saved_path)
    saved.clear()
    saved.update(restored)
    third, newest, _ = batch(pool, rng_seed=4)
    assert [row["number"] for row in newest] == [11, 12, 13, 14, 15]
    assert candidates.candidate_review_view(root, saved)["accepted"][0]["id"] == rows[0]["candidate_id"]
    choose(pool, newest[0])
    with pytest.raises(ValueError, match="최대 수량"):
        choose(pool, newest[1])
    with pytest.raises(ValueError, match="수량을 채웠"):
        candidates.plan_candidate_batch(project, saved, first["group_key"])
    candidates.candidate_action(root, saved, "approve")
    assert candidates.active_candidate_group(saved) == "train:man:teenager"
    with pytest.raises(ValueError, match="현재 검수"):
        candidates.candidate_action(root, saved, "toggle", group_key=first["group_key"],
            candidate_id=rows[0]["candidate_id"], sha256=rows[0]["sha256"])


def test_bad_plan_and_outside_output_fail_before_loading_model(pool):
    root, project, _, _, saved, _ = pool
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    bad = deepcopy(plan)
    bad["candidates"][0]["design_instruction"] += " Changed"
    load = lambda *args: pytest.fail("Invalid plan must fail before loading model")
    with pytest.raises(ValueError, match="설정 또는 난수"):
        candidates.generate_candidate_batch(project, candidates.candidate_jobs(root) / "bad", "bad", bad, [], load_model=load)
    with pytest.raises(ValueError, match="작업 폴더"):
        candidates.generate_candidate_batch(project, root / "outside", "bad", plan, [], load_model=load)
    bad = deepcopy(plan)
    bad["candidates"][0]["candidate_id"] = "../../outside"
    with pytest.raises(ValueError):
        candidates.generate_candidate_batch(project, candidates.candidate_jobs(root) / "bad", "bad", bad, [], load_model=load)


def test_original_decoded_pcm_duplicates_are_rejected_even_when_file_hash_differs(pool):
    root, project, _, originals, saved, model = pool
    original = originals[0]
    # Extra RIFF bytes change the file identity without changing decoded samples.
    with (root / original["path"]).open("ab") as handle:
        handle.write(b"preserved source metadata")
    original["sha256"] = digest(root / original["path"])
    saved.clear()
    saved.update(candidates.create_candidate_review(originals))
    _, metrics = _check_wave(root / original["path"])
    journal = root / "review/quality.jsonl"
    journal.parent.mkdir()
    write_immutable_rows(journal, [{"metrics": metrics}])
    receipt = root / "review/quality.json"
    write_json(receipt, {"quality_journal": {"path": "review/quality.jsonl", "sha256": digest(journal)}})
    write_json(root / "approval.json", {"reference_technical_admission": {
        "path": "review/quality.json", "sha256": digest(receipt)}})
    calls = []
    def duplicate(**kwargs):
        calls.append(kwargs)
        return [waveform(0)], 16000
    model.generate_voice_design = duplicate
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    output = candidates.candidate_jobs(root) / plan["batch_id"] / "audio"
    with pytest.raises(ValueError, match="기술 검사를 통과"):
        candidates.generate_candidate_batch(project, output, "duplicates", plan, [], load_model=model.load)
    assert len(calls) == 4
    assert not list(output.glob("*.wav"))
    assert read_json(output / "plan.json") == plan


def test_freeze_uses_exact_selected_waves_and_stable_slots_without_touching_sources(pool):
    root, _, _, originals, saved, _ = pool
    before = {row["path"]: digest(root / row["path"]) for row in originals}
    with pytest.raises(ValueError, match="모든"):
        candidates.freeze_candidate_review(root, saved, originals)
    for number, group in enumerate(saved["groups"]):
        _, rows, _ = batch(pool, rng_seed=number + 10)
        for row in rows[:group["target"]]:
            choose(pool, row)
        candidates.candidate_action(root, saved, "approve")
    manifest = candidates.freeze_candidate_review(root, saved, originals)
    binding = snapshot_json(root, manifest, "candidate-review")
    banks = {split: [{key: value for key, value in row.items() if key != "voice_design"}
                      for row in originals if row["split"] == split] for split in ("train", "validation")}
    result = overlay_reference_banks(root, banks, binding)
    for split, rows in result.items():
        assert len(rows) == (20 if split == "train" else 30)
        for row in rows:
            assert row["sha256"] == saved["candidates"][row["candidate_id"]]["sha256"]
            assert row["voice_family_id"] in saved["originals"]
            assert row["split"] == split
            assert row["description"] != next(source["description"] for source in originals if source["voice_family_id"] == row["voice_family_id"])
    assert before == {path: digest(root / path) for path in before}
    altered = deepcopy(manifest)
    a, b = altered["groups"][:2]
    a["families"], b["families"] = b["families"], a["families"]
    with pytest.raises(ValueError, match="분류 기록"):
        candidates.overlay_candidate_banks(root, banks, altered)
    altered = deepcopy(manifest)
    first = next(iter(altered["replacements"].values()))
    first["split"] = "validation"
    with pytest.raises(ValueError, match="출처 또는 검수"):
        candidates.overlay_candidate_banks(root, banks, altered)
    chosen = next(iter(manifest["replacements"].values()))
    with (root / chosen["path"]).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="changed bound file"):
        overlay_reference_banks(root, banks, binding)


@pytest.mark.parametrize("listen", [False, "mixed"])
def test_unheard_and_mixed_choices_freeze_with_honest_listening_and_unchanged_source_guards(pool, listen):
    root, _, _, originals, saved, _ = pool
    original_hashes = {row["path"]: digest(root / row["path"]) for row in originals}
    for number, group in enumerate(saved["groups"]):
        _, rows, _ = batch(pool, rng_seed=number + 80)
        for index, row in enumerate(rows[:group["target"]]):
            if listen == "mixed" and index == 0:
                candidates.candidate_action(root, saved, "hear", candidate_id=row["candidate_id"], sha256=row["sha256"])
            candidates.candidate_action(root, saved, "toggle", candidate_id=row["candidate_id"], sha256=row["sha256"])
        candidates.candidate_action(root, saved, "approve")
    expected_heard = deepcopy(saved["heard"])
    assert len(expected_heard) == (20 if listen else 0)
    manifest = candidates.freeze_candidate_review(root, saved, originals)
    assert manifest["listening_policy"] == candidates.OPTIONAL_LISTENING_POLICY
    assert manifest["heard"] == saved["heard"] == expected_heard
    banks = {split: [row for row in originals if row["split"] == split] for split in ("train", "validation")}
    selected = candidates.overlay_candidate_banks(root, banks, manifest)
    assert [len(selected[split]) for split in banks] == [20, 30]
    assert original_hashes == {path: digest(root / path) for path in original_hashes}
    first = next(iter(manifest["replacements"].values()))
    bad_heard = deepcopy(manifest)
    bad_heard["heard"][first["candidate_id"]] = "0" * 64
    with pytest.raises(ValueError, match="출처 또는 검수"):
        candidates.overlay_candidate_banks(root, banks, bad_heard)
    unknown = {**manifest, "listening_policy": "unknown-policy"}
    with pytest.raises(ValueError, match="청취 정책"):
        candidates.overlay_candidate_banks(root, banks, unknown)
    strict = deepcopy(manifest)
    strict.pop("listening_policy")
    with pytest.raises(ValueError, match="출처 또는 검수"):
        candidates.overlay_candidate_banks(root, banks, strict)
    # A real old, unmarked frozen selection remains valid only with all of its
    # original listening evidence. A marker never synthesizes that evidence.
    strict["heard"] = {row["candidate_id"]: row["sha256"] for row in strict["replacements"].values()}
    assert candidates.overlay_candidate_banks(root, banks, strict) == selected
    strict_binding = snapshot_json(root, strict, "legacy-strict-selection")
    write_json(root / strict_binding["path"], {**strict, "listening_policy": candidates.OPTIONAL_LISTENING_POLICY, "heard": {}})
    with pytest.raises(ValueError, match="changed bound file"):
        overlay_reference_banks(root, banks, strict_binding)
    strict["heard"].pop(first["candidate_id"])
    with pytest.raises(ValueError, match="출처 또는 검수"):
        candidates.overlay_candidate_banks(root, banks, strict)
    chosen = root / first["path"]
    chosen.write_bytes(chosen.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed bound file"):
        candidates.overlay_candidate_banks(root, banks, manifest)
