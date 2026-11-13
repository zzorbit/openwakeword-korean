"""Generate one review sample per non-call phrase before bulk synthesis."""
from collections import Counter
from .common import append_row, bound_file, digest, identity, read_json, read_rows, write_immutable_rows, write_json, writer_lock
from .task_progress import emit as report_progress, generation_attempt
from .project import load_project
from .storage import project_storage


HARD_SAMPLE_REVIEW_POLICY = "sample_before_generation_v2"


@project_storage
def generate_hard_pilot(project, phrases, samples_per_phrase, model, *, revision=None, pause_requested=None):
    from . import clone_reference, tts
    from .hard_tts import (MAX_ATTEMPTS, _RECOVERY_ATTEMPTS, _attempt_rounds, _candidate_path, _exhausted_details,
                           _phrases, _short_clone_context, _store_candidate, _synthesise, _validate_short_clone_attempt,
                           _verify_prepared_reference,
                           check_generation_pause, hard_phrase_scope)

    check_generation_pause(pause_requested)
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; create a separate project for new Hard candidates")
    phrases = _phrases(phrases, config["wakeword"])
    approval_path = root / "approval.json"
    approval = read_json(approval_path)
    if (approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True
            or approval.get("human_reference_used") is not False or approval.get("wakeword") != config["wakeword"]
            or approval.get("tts_settings_id") != identity(config["tts"]) or approval.get("generation_method") != "clone"):
        raise ValueError("Approve the standalone Positive pilot and synthetic clone policy before Hard generation")
    references = [row for row in read_rows(root / "data/references/train/references.jsonl")
                  if row.get("status") == "accepted" or row.get("voice_family_id") in approval.get("accepted_reference_families", [])]
    if not references or len({row.get("voice_family_id") for row in references}) != len(references):
        raise ValueError("A nonempty distinct approved Train synthetic reference bank is required")
    for reference in references:
        if (reference.get("human_reference_used") is not False or reference.get("provenance") != "qwen_synthetic_reference"
                or reference.get("split") != "train" or not reference.get("voice_family_id") or not reference.get("text")):
            raise ValueError("Hard generation accepts only approved Qwen synthetic Train references")
        bound_file(root, reference)

    output = hard_phrase_scope(root / "data/generated", revision) / "hard_pilot"
    contract_path = output / "contract.json"
    journal = output / "candidates.jsonl"
    contract = {"kind": HARD_SAMPLE_REVIEW_POLICY, "word": config["wakeword"], "phrases": phrases,
                "samples_per_phrase": samples_per_phrase, "references": references, "tts": config["tts"],
                "implementation_sha256": digest(__file__), "technical_qc_sha256": digest(tts.__file__),
                "short_clone_policy": tts.SHORT_CLONE_POLICY,
                "positive_policy": {"path": "approval.json", "sha256": digest(approval_path)}}
    with writer_lock(output):
        rows = read_rows(journal)
        expected = contract
        if contract_path.exists():
            from .call_generation_history import compatible_hard_contract
            saved = read_json(contract_path)
            compatible = {key: value for key, value in expected.items() if key != "short_clone_policy"}
            if saved != expected and not compatible_hard_contract(saved, compatible, pilot=True):
                raise ValueError("Hard pilot settings, phrase list or references changed")
            contract = saved
        elif rows:
            raise ValueError("Hard pilot journal has no generation contract")
        else:
            write_json(contract_path, contract)
        binding = {"path": contract_path.relative_to(root).as_posix(), "sha256": digest(contract_path)}
        update, boundary, recovery = _short_clone_context(root, output, contract, expected, rows)
        passed, attempts, hashes, prepared = {}, Counter(), set(), set()
        for number, row in enumerate(rows):
            check_generation_pause(pause_requested)
            slot, attempt = row.get("slot"), row.get("attempt")
            if (type(slot) is not int or not 0 <= slot < len(phrases) or type(attempt) is not int
                    or attempt != attempts[slot] or slot in passed):
                raise ValueError("Invalid Hard pilot attempt journal")
            historical = number < boundary
            ceiling = _RECOVERY_ATTEMPTS if not historical and slot in recovery else MAX_ATTEMPTS
            if not 0 <= attempt < ceiling:
                raise ValueError("Invalid Hard pilot attempt budget")
            sample_id = f"hard_pilot_{slot:06d}_attempt{attempt}"
            reference = references[slot % len(references)]
            family = reference["voice_family_id"]
            seed = int(identity([config["tts"]["seed"], HARD_SAMPLE_REVIEW_POLICY, phrases[slot], family, attempt])[:8], 16)
            path = _candidate_path(root, output, row, sample_id)
            _validate_short_clone_attempt(row, historical=historical, update=update)
            if not historical:
                _verify_prepared_reference(root, row, reference, prepared)
            if (row.get("sample_id") != sample_id or row.get("text") != phrases[slot]
                    or row.get("voice_family_id") != family or row.get("reference_sha256") != reference["sha256"]
                    or row.get("generation_contract") != binding
                    or row.get("seed") != seed
                    or row.get("technical_pass") is not row.get("qc", {}).get("passed")):
                raise ValueError("Hard pilot candidate changed")
            if row["technical_pass"]:
                if row["sha256"] in hashes:
                    raise ValueError("Hard pilot passed duplicate audio")
                passed[slot] = row
            attempts[slot] += 1
            if path is not None:
                hashes.add(row["sha256"])

        prompts = {}
        dataset_id = "qwen_" + config["project_id"] + "_hard_v1"
        for slot, attempt in _attempt_rounds(len(phrases), passed, attempts, recovery):
            check_generation_pause(pause_requested)
            if "short_clone_policy" not in contract and update is None:
                update, _, _ = _short_clone_context(root, output, contract, expected, rows, create=True)
            phrase = phrases[slot]
            reference = references[slot % len(references)]
            family = reference["voice_family_id"]
            if family not in prompts:
                report_progress("tts_prompt_preparation", "기준 목소리 복제 정보 준비", eta_eligible=False)
                prompt_path, preparation = clone_reference.prepare_clone_reference(root, reference, compact_internal_silence=True)
                prompt = model.create_voice_clone_prompt(ref_audio=str(prompt_path), ref_text=reference["text"],
                                                        x_vector_only_mode=False)
                prompts[family] = prompt, preparation
            prompt, preparation = prompts[family]
            check_generation_pause(pause_requested)
            sample_id = f"hard_pilot_{slot:06d}_attempt{attempt}"
            seed = int(identity([config["tts"]["seed"], HARD_SAMPLE_REVIEW_POLICY, phrase, family, attempt])[:8], 16)
            tts._seed(seed)
            with generation_attempt() as timing:
                qc, converted, token_limit = _synthesise(model, phrase, prompt, config["tts"]["generation"],
                                                        retry=attempt > 0, expected_reuses=1)
            path, sha, bounds = _store_candidate(root, output, sample_id, converted, qc, hashes)
            source_id = "qwen:" + config["project_id"] + ":" + sample_id
            row = {"sample_id": sample_id, "slot": slot, "attempt": attempt, "text": phrase,
                       "source_id": source_id, "canonical_source_id": source_id, "dataset_id": dataset_id,
                       "category": "competing_speech", "split": "train", "speech": True, "provenance": "qwen_tts",
                       "human_reference_used": False, "individual_human_listening": False, "training_admitted": False,
                       "voice_family_id": family, "reference_sha256": reference["sha256"],
                       "generation_contract": binding, "generation_method": "clone", "seed": seed,
                       "short_clone_policy": tts.SHORT_CLONE_POLICY, "short_clone_update": update,
                       "generation_token_limit": token_limit, "clone_reference_preparation": preparation,
                       "path": path, "sha256": sha, "qc": qc,
                       "start_s": bounds[0] if bounds else None, "end_s": bounds[1] if bounds else None,
                       "technical_pass": qc["passed"]}
            append_row(journal, row)
            if timing is not None:
                timing.committed(qc["passed"])
            rows.append(row)
            attempts[slot] += 1
            if qc["passed"]:
                passed[slot] = row
            print(f"TTS Hard review samples: {len(passed)}/{len(phrases)}", flush=True)
        check_generation_pause(pause_requested)
        report_progress("tts_finalizing", "유사어 검수 목록 준비", eta_eligible=False)
        accepted = [passed[slot] for slot in sorted(passed)]
        exhausted = [slot for slot in range(len(phrases)) if slot not in passed]
        return {"target_originals": len(phrases), "technical_pass": len(accepted), "attempts": sum(attempts.values()),
                "full_target_reached": len(accepted) == len(phrases), "generation_contract": binding,
                "exhausted_slots": exhausted, "exhausted_slot_details": _exhausted_details(exhausted, rows,
                    lambda slot: (phrases[slot], references[slot % len(references)]), recovery)}, accepted


def publish_hard_pilot_review(root, workspace, summary, rows, samples_per_phrase):
    sources = workspace / "hard_pilot_sources.jsonl"
    queue = workspace / "hard_pilot_review.jsonl"
    selection_path = workspace / "hard_pilot_selection.json"
    previews = [{**row, "sample_id": "preview_" + row["sample_id"],
                 "reviewed_source": {"path": row["path"], "sha256": row["sha256"]},
                 "review_group": row["text"] + " · " + row["voice_family_id"],
                 "review_instruction": "문구가 명확하고 호출어로 들리지 않으면 보존하세요."} for row in rows]
    write_immutable_rows(sources, rows)
    write_immutable_rows(queue, previews)
    source_binding = {"path": sources.relative_to(root).as_posix(), "sha256": digest(sources)}
    queue_binding = {"path": queue.relative_to(root).as_posix(), "sha256": digest(queue)}
    selection = {"kind": "hard_phrase_representative_selection_v1", "policy": HARD_SAMPLE_REVIEW_POLICY,
                 "source_manifest": source_binding, "representative_queue": queue_binding,
                 "generation_contract": summary["generation_contract"], "samples_per_phrase": samples_per_phrase,
                 "representatives": [{"dataset_id": row["dataset_id"], "text": row["text"],
                     "voice_family_id": row["voice_family_id"], "source_sha256": row["sha256"],
                     "preview_sample_id": row["sample_id"]} for row in previews]}
    if selection_path.exists():
        if read_json(selection_path) != selection:
            raise ValueError("Frozen Hard pilot selection changed")
    else:
        write_json(selection_path, selection)
    return {"id": "hard", "kind": "content", "queue": str(queue), "queue_sha256": queue_binding["sha256"],
            "count": len(previews), "pool_manifest": str(sources), "pool_sha256": source_binding["sha256"],
            "synthetic_review_policy": HARD_SAMPLE_REVIEW_POLICY,
            "representative_selection": {"path": selection_path.relative_to(root).as_posix(), "sha256": digest(selection_path)},
            "description": "문구마다 대표 음원 한 개를 확인하세요. 보존한 문구만 전체 생성합니다."}


def saved_hard_sample_scope(root, config, descriptor, context):
    from .pool_review import hard_sample_consent

    answer = context.get("inputs", {}).get("review_answers", {}).get("hard")
    if not answer:
        raise ValueError("Save the Hard pronunciation sample review before full generation")
    return hard_sample_consent(root, config, descriptor["representative_selection"],
        {"path": descriptor["queue"], "sha256": descriptor["queue_sha256"]}, read_json(bound_file(root, answer)))
