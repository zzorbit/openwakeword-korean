"""One pronunciation sample per admitted voice, with frozen-call reuse."""
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import json
import os

from .common import bound_file, digest, identity, iter_rows, read_json, read_rows, resolved, snapshot_json, write_immutable_rows, write_json, writer_lock
from .project import load_project
from .reference_qc import _human_answers
from .storage import project_storage


TRAIN_PILOT_POLICY = "one_per_training_voice_v1"
COMBINED_PILOT_POLICY = "one_per_training_and_validation_voice_v2"
POSTGEN_PILOT_POLICY = "one_per_voice_after_generation_v3"
PREPOST_PILOT_POLICY = "one_per_voice_before_and_after_generation_v4"
REVIEW_GROUPS = {"train": "학습용", "validation": "공통 평가용"}


def _passing_slots(root, config, split, references, directory, scope=None):
    """Validate the saved generation contract and every recorded attempt."""
    from .tts import call_generation_contract, call_journal_context, validate_call_rows
    expected = call_generation_contract(config, split, references, True, "clone", scope)
    rows, context = call_journal_context(root, directory, expected)
    passing, _ = validate_call_rows(root, config, split, references, rows, context, scope=scope)
    return rows, passing


def _admitted_references(root, approval, answers, split, count):
    journal = root / "data/references" / split / "references.jsonl"
    references = [row for row in read_rows(journal) if row.get("status") == "accepted"
                  or row.get("voice_family_id") in approval.get("accepted_reference_families", [])]
    ids = [row.get("voice_family_id") for row in references]
    if not references or None in ids or len(set(ids)) != len(ids):
        raise ValueError(f"A nonempty, unique admitted {split.title()} reference bank is required")
    target = len(references) if count is None else count
    if type(target) is not int or target != len(references):
        raise ValueError(f"Pilot count must match every selected admitted {split.title()} reference")
    for row in references:
        if (row.get("provenance") != "qwen_synthetic_reference" or row.get("human_reference_used") is not False
                or row.get("split") != split):
            raise ValueError("Pilot references must be this split's Qwen synthetic voices")
        if answers.get((row["sample_id"], row["sha256"])) in ("discard", "review"):
            raise ValueError("A selected pilot reference has a human discard or pending review")
        bound_file(root, row)
    return references


def _prepare_split(project, root, config, split, references, generate):
    from .tts import _exhausted_slots, CLONE_ATTEMPTS
    directory = root / "data/pilot" / split
    rows, passing = _passing_slots(root, config, split, references, directory)
    target = len(references)
    reused = sum(slot in passing for slot in range(target))
    if reused != target:
        target_key = "positive_originals" if split == "train" else "validation_originals"
        if config["targets"][target_key] < target:
            raise ValueError(f"The positive generation target must cover every {split.title()} reference before pilot generation")
        generate(project, split, target)
        rows, passing = _passing_slots(root, config, split, references, directory)
    selected = [passing[slot] for slot in range(target) if slot in passing]
    counts = {"attempts": len(rows), "technical_pass": len(selected), "target": target,
              "reused_existing_count": reused}
    bindings = {key: _binding(root, path) for key, path in (
        ("references_journal", root / "data/references" / split / "references.jsonl"),
        ("candidate_journal", directory / "candidates.jsonl"),
        ("generation_contract", directory / "contract.json"),
    )}
    missing = [references[slot]["voice_family_id"] for slot in range(target) if slot not in passing]
    exhausted = _exhausted_slots(rows, references, target, passing, CLONE_ATTEMPTS, split)
    return selected, counts, bindings, missing, exhausted


def _binding(root, path):
    return {"path": str(path.relative_to(root)), "sha256": digest(path)}


def combined_review_queue(root, binding, train_count, validation_count):
    """Load the bound queue only if both intended voice banks are represented."""
    queue = bound_file(root, binding)
    rows = read_rows(queue)
    expected = {("train", REVIEW_GROUPS["train"]): train_count,
                ("validation", REVIEW_GROUPS["validation"]): validation_count}
    groups = Counter((row.get("split"), row.get("review_group")) for row in rows)
    if (groups != expected or len({row["sample_id"] for row in rows}) != len(rows)
            or len({row["voice_family_id"] for row in rows}) != len(rows)
            or any(row.get("qc", {}).get("passed") is not True for row in rows)):
        raise ValueError("호출어 발음 검수 목록에 학습용·공통 평가용 기준 목소리가 모두 포함되어야 합니다.")
    return queue


def combined_approval_matches(root, descriptor, approval):
    """Keep an approval of a different queue from admitting these voices."""
    report = root / "review/pilot_results.json"
    return (approval.get("accepted_pilot_items") == descriptor["count"]
            and report.is_file() and approval.get("pilot_review_sha256") == digest(report)
            and read_json(report).get("queue_sha256") == descriptor["queue_sha256"])


def prepare_pilot(project, train_count=None, *, validation_count=None, tts_backend="auto", tts_cpu_threads=4):
    """Publish one successful call per admitted reference in each requested bank.

    With no Validation count, retain the Train-only queue used by older Studio
    workflows and CLI callers. Both policies reuse passing slots without
    rewriting old journals, summaries, or review evidence.
    """
    if tts_backend not in ("auto", "eager") or type(tts_cpu_threads) is not int or tts_cpu_threads < 1:
        raise ValueError("Use TTS backend auto/eager and a positive CPU thread count")
    root, config = load_project(project)
    approval = read_json(root / "approval.json")
    answers = _human_answers(root, approval)
    counts = {"train": train_count}
    if validation_count is not None:
        counts["validation"] = validation_count
    banks = {split: _admitted_references(root, approval, answers, split, count)
             for split, count in counts.items()}
    families = [row["voice_family_id"] for references in banks.values() for row in references]
    if len(set(families)) != len(families):
        raise ValueError("Train and Validation pilot reference banks must be disjoint")
    if validation_count is not None and ({row["sha256"] for row in banks["train"]}
                                         & {row["sha256"] for row in banks["validation"]}):
        raise ValueError("Train and Validation pilot reference audio must be disjoint")

    if validation_count is None:
        def generate(path, split, limit):
            from .tts_runtime import generate_with_runtime
            return generate_with_runtime(path, split, limit, True, "clone",
                                         backend=tts_backend, cpu_threads=tts_cpu_threads)
    else:
        model = None

        def generate(path, split, limit):
            nonlocal model
            from .tts import generate as generate_calls
            from .tts_runtime import CloneRuntime
            if model is None:
                model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, tts_backend, tts_cpu_threads)
            return generate_calls(path, split, limit, True, "clone", model=model)

    selected, split_counts, split_bindings, missing_families, exhausted_slots = [], {}, {}, [], []
    for split, references in banks.items():
        rows, split_counts[split], split_bindings[split], missing, exhausted = _prepare_split(
            project, root, config, split, references, generate)
        if validation_count is not None:
            rows = [{**row, "review_group": REVIEW_GROUPS[split]} for row in rows]
        selected.extend(rows)
        missing_families.extend(missing)
        exhausted_slots.extend(exhausted)
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Pilot review samples must be unique across partitions")

    directory = root / "data/pilot"
    if validation_count is None:
        queue = directory / "train" / ("review_one_per_voice_" + identity(selected) + ".jsonl")
        bindings = split_bindings["train"]
    else:
        queue = directory / ("review_train_validation_" + identity(selected) + ".jsonl")
        bindings = split_bindings
    write_immutable_rows(queue, selected)
    bindings["review_queue"] = _binding(root, queue)
    totals = {key: sum(count[key] for count in split_counts.values()) for key in split_counts["train"]}
    summary = {**totals, "pilot": True, "human_listened": 0, "policy_admitted": 0,
               "independent_STT_runs": 0, "review_queue": str(queue),
               "full_target_reached": len(selected) == len(families),
               "selected_reference_count": len(families), "selected_reference_families": families,
               "missing_reference_families": missing_families, "exhausted_slots": exhausted_slots,
               "distribution": {"voice_family": dict(Counter(row["voice_family_id"] for row in selected))},
               "bindings": bindings}
    if validation_count is not None:
        summary["split_counts"] = split_counts
    return summary


def _preflight_banks(root, settings):
    approval = read_json(root / "approval.json")
    answers = _human_answers(root, approval)
    banks = {split: _admitted_references(root, approval, answers, split, settings[f"{split}_references"])
             for split in REVIEW_GROUPS}
    reference_review = os.environ.get("KOWW_STUDIO_REFERENCE_REVIEW")
    if reference_review:
        from .reference_voice_review import overlay_reference_banks
        banks = overlay_reference_banks(root, banks, json.loads(reference_review))
    if ({row["voice_family_id"] for row in banks["train"]} & {row["voice_family_id"] for row in banks["validation"]}
            or {row["sha256"] for row in banks["train"]} & {row["sha256"] for row in banks["validation"]}):
        raise ValueError("Train and Validation reference banks must be disjoint")
    return banks


def _preflight_state(root, config, banks, prior_rounds):
    from .review import PLAY_STARTED_POLICY, validate_started_decision
    references = {row["voice_family_id"]: row for bank in banks.values() for row in bank}
    pending = set(references)
    approved, heard_hashes, reference_hashes = {}, set(), {row["sha256"] for row in references.values()}
    for index, evidence in enumerate(prior_rounds):
        queue = read_rows(bound_file(root, evidence["queue"]))
        report = read_json(bound_file(root, evidence["report"]))
        if (not pending or len(queue) != len(pending)
                or {row.get("voice_family_id") for row in queue} != pending):
            raise ValueError("사전 검수 기록이 해당 차수의 기준 목소리와 일치하지 않습니다.")
        audio = {row["sample_id"]: row["sha256"] for row in queue}
        decisions = report.get("decisions", {})
        playback_policy = report.get("autosave_policy")
        if playback_policy not in (None, PLAY_STARTED_POLICY):
            raise ValueError("지원하지 않는 호출어 사전 검수 판정 방식입니다.")
        if (report.get("kind") != "pilot_listening_review" or report.get("wakeword") != config["wakeword"]
                or report.get("queue_sha256") != evidence["queue"]["sha256"]
                or report.get("source_scope") != "Human dispositions entered through the local review screen"
                or not report.get("saved_at") or report.get("queued_audio_sha256") != audio
                or len(audio) != len(queue) or set(decisions) != set(audio)):
            raise ValueError("사전 검수 음원과 저장된 판정이 일치하지 않습니다.")
        if playback_policy == PLAY_STARTED_POLICY:
            for answer in decisions.values():
                validate_started_decision(answer)
            if (report.get("playback_started_count") != sum(answer["playback_started"] for answer in decisions.values())
                    or report.get("playback_completed_count") != sum(answer["playback_completed"] for answer in decisions.values())):
                raise ValueError("사전 검수의 재생 시작·완료 집계가 실제 판정과 다릅니다.")
        for row in queue:
            answer = decisions[row["sample_id"]]
            family = row["voice_family_id"]
            original = references[family]
            reference = row["preflight_reference"]
            if (row.get("pronunciation_policy") != PREPOST_PILOT_POLICY or row.get("preflight_round") != index
                    or row.get("text") != config["wakeword"] or row.get("tts_settings_id") != identity(config["tts"])
                    or row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False
                    or row.get("generation_method") != "clone" or row.get("qc", {}).get("passed") is not True
                    or row.get("split") != original["split"] or row.get("voice_reference_sha256") != reference["sha256"]
                    or reference.get("voice_family_id") != family or reference.get("split") != original["split"]
                    or reference.get("provenance") != "qwen_synthetic_reference"
                    or reference.get("human_reference_used") is not False
                    or (reference != original and reference.get("replaces_reference_sha256") != original["sha256"])
                    or (index < 2 and reference != original) or (index >= 2 and reference == original)
                    or playback_policy is None and answer.get("listened") is not True
                    or answer.get("decision") not in ("keep", "discard")):
                raise ValueError("기준 목소리, 사전 검수 음원, 또는 저장된 판정이 일치하지 않습니다.")
            bound_file(root, reference)
            bound_file(root, row)
            if row["sha256"] in heard_hashes:
                raise ValueError("사전 검수 차수 사이에서 동일한 음원이 발견됐습니다.")
            # Exclude every reviewed candidate from regeneration, including an
            # explicitly discarded, unplayed item. This is an exclusion set;
            # actual listening facts remain the report's separate booleans.
            heard_hashes.add(row["sha256"])
            reference_hashes.add(reference["sha256"])
            if answer["decision"] == "keep":
                approved[family] = reference
                pending.remove(family)
    return approved, pending, heard_hashes, reference_hashes


def _alternate_reference(root, config, original, round_key, load_design_model, reference_hashes):
    import soundfile as sf
    from .clone_reference import prepare_generated_reference, verify_generated_reference, GeneratedReferenceQualityError
    from .tts import _reference_instruction, _seed

    family = original["voice_family_id"]
    token = identity([round_key, family, original["sha256"]])[:16]
    directory = root / "data/call_pronunciation/references"
    directory.mkdir(parents=True, exist_ok=True)
    audio, record = directory / f"{token}.wav", directory / f"{token}.json"
    if record.exists():
        reference = read_json(record)
        preparation = reference.get("reference_preparation")
        expected_path = preparation.get("prepared", {}).get("path") if preparation else audio.relative_to(root).as_posix()
        if (reference.get("voice_family_id") != family or reference.get("split") != original["split"]
                or reference.get("replaces_reference_sha256") != original["sha256"]
                or reference.get("path") != expected_path
                or reference.get("sha256") in reference_hashes):
            raise ValueError("교체 기준 목소리의 출처가 변경되었습니다.")
        bound_file(root, reference)
        if preparation:
            verified = verify_generated_reference(root, preparation)
            if digest(verified) != reference["sha256"]:
                raise ValueError("교체 기준 목소리와 원본 보존 기록이 다릅니다.")
        return reference

    instruction = (original["design_instruction"] if original.get("candidate_id") and original.get("design_instruction")
                   else _reference_instruction(original))
    for attempt in range(3):
        seed = int(identity([config["tts"]["seed"], token, attempt])[:8], 16)
        raw = directory / "raw" / f"{token}_attempt{attempt}.wav"
        base, recovery = raw, 0
        # An interrupted file without its raw receipt is preserved as an
        # orphan, never silently re-attested as a finished model generation.
        while raw.exists() and not raw.with_name(raw.name + ".raw.json").exists():
            recovery += 1
            raw = base.with_name(base.stem + f"_recovery{recovery}.wav")
        candidate = directory / (raw.stem + ".wav")
        if not raw.exists():
            _seed(seed)
            model = load_design_model()
            wavs, rate = model.generate_voice_design(text=original["text"], language="Korean",
                instruct=instruction, **config["tts"]["generation"])
            raw.parent.mkdir(parents=True, exist_ok=True)
            sf.write(raw, wavs[0], rate, subtype="FLOAT")
        generation = {"kind": "call_preflight_replacement", "model": config["tts"]["design_model"],
                      "seed": seed, "text": original["text"], "instruction": instruction,
                      "tts_settings_id": identity(config["tts"]), "attempt": attempt,
                      "replaces_reference_sha256": original["sha256"]}
        try:
            preparation = prepare_generated_reference(root,
                {"path": raw.relative_to(root).as_posix(), "sha256": digest(raw), "generation": generation}, candidate)
        except GeneratedReferenceQualityError:
            continue
        sha = preparation["prepared"]["sha256"]
        if sha not in reference_hashes:
            reference = {**original, "sample_id": f"{family}_{token}", "path": preparation["prepared"]["path"],
                         "sha256": sha, "seed": seed, "replaces_reference_sha256": original["sha256"],
                         "reference_preparation": preparation}
            write_json(record, reference)
            return reference
    raise ValueError("교체 기준 목소리의 기술 검사를 통과하지 못했습니다.")


@project_storage
def prepare_prepost_pilot(project, settings, *, prior_rounds=()):
    """Prepare only pending voices; each reviewed round is immutable evidence."""
    from .tts import _load_model, call_generation_location, generate
    from .tts_runtime import CloneRuntime

    root, config = load_project(project)
    directory = root / "data/call_pronunciation"
    with writer_lock(directory):
        banks = _preflight_banks(root, settings)
        _, pending, heard_hashes, reference_hashes = _preflight_state(root, config, banks, prior_rounds)
        if not pending:
            raise ValueError("모든 기준 목소리의 사전 검수가 이미 완료되었습니다.")
        round_index = len(prior_rounds)
        round_key = identity(list(prior_rounds))
        if round_index >= 2:
            design = None

            def load_design_model():
                nonlocal design
                if design is None:
                    design = _load_model(config["tts"]["design_model"], config["tts"], root)
                return design

            for split, bank in banks.items():
                for index, original in enumerate(bank):
                    if original["voice_family_id"] in pending:
                        reference = _alternate_reference(root, config, original, round_key, load_design_model, reference_hashes)
                        bank[index] = reference
                        reference_hashes.add(reference["sha256"])
            design = None
        model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, "auto", config["training"]["cpu_threads"])
        selected, split_counts, attempts, exhausted = [], {}, 0, []
        previous_split_complete = True
        for split, group in REVIEW_GROUPS.items():
            voice_numbers = {row["voice_family_id"]: index for index, row in enumerate(banks[split], 1)}
            references = [row for row in banks[split] if row["voice_family_id"] in pending]
            if not references:
                continue
            if not previous_split_complete:
                split_counts[split] = {"target": len(references), "technical_pass": 0}
                continue
            scope = {"kind": "preflight", "revision": round_key, "references": references,
                     "excluded_sha256": sorted(heard_hashes)}
            output, _ = call_generation_location(root, split, scope)
            summary = generate(project, split, len(references), True, "clone", model=model, staged_scope=scope)
            attempts += summary["attempts"]
            exhausted.extend(summary.get("exhausted_slots", []))
            _, passing = _passing_slots(root, config, split, references, output, scope)
            split_counts[split] = {"target": len(references), "technical_pass": len(passing)}
            previous_split_complete = len(passing) == len(references)
            for slot, row in sorted(passing.items()):
                selected.append({**row, "review_group": group, "pronunciation_policy": PREPOST_PILOT_POLICY,
                                 "voice_number": voice_numbers[row["voice_family_id"]],
                                 "preflight_round": round_index, "preflight_reference": references[slot]})
                heard_hashes.add(row["sha256"])
        result = {"attempts": attempts, "target": len(pending), "technical_pass": len(selected), "pilot": True,
                  "independent_STT_runs": 0, "full_target_reached": len(selected) == len(pending),
                  "split_counts": split_counts, "exhausted_slots": exhausted}
        if result["full_target_reached"]:
            queue = directory / f"preflight_{round_key[:16]}.jsonl"
            write_immutable_rows(queue, selected)
            result.update(review_queue=str(queue), bindings={"review_queue": _binding(root, queue)})
        return result


def stage_positives_for_review(project, settings, *, prior_rounds=None):
    """Freeze the full technical-pass corpus and select one call per voice."""
    from .tts import call_generation_location, generate
    from .tts_runtime import CloneRuntime

    root, config = load_project(project)
    banks = _preflight_banks(root, settings)
    preflight = prior_rounds is not None
    if preflight:
        chosen, pending, heard_hashes, _ = _preflight_state(root, config, banks, prior_rounds)
        if pending:
            raise ValueError("모든 기준 목소리의 사전 검수가 완료되어야 본생성을 시작합니다.")
        banks = {split: [chosen[row["voice_family_id"]] for row in bank] for split, bank in banks.items()}
        revision = identity(list(prior_rounds))
    targets = {"train": settings["originals"], "validation": settings["validation_originals"]}
    if any(targets[split] < len(banks[split]) for split in REVIEW_GROUPS):
        raise ValueError("The generation targets must cover every voice in each reference bank")
    if (targets["train"] > config["targets"]["positive_originals"]
            or targets["validation"] > config["targets"]["validation_originals"]):
        raise ValueError("Staged generation cannot exceed the project's frozen targets")
    model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, "auto", config["training"]["cpu_threads"])
    outputs, sources, selected = [], {}, []
    source_hashes = set()
    for split, group in REVIEW_GROUPS.items():
        target = targets[split]
        # The existing pilot synthesis mode keeps the full target out of the
        # train/evaluation manifests until its actual pronunciation review.
        scope = None
        directory = root / "data/pilot" / split
        if preflight:
            scope = {"kind": "production", "revision": revision, "references": banks[split],
                     "excluded_sha256": sorted(heard_hashes | source_hashes)}
            directory, _ = call_generation_location(root, split, scope)
        options = {"staged_scope": scope} if preflight else {}
        summary = generate(project, split, target, True, "clone", model=model, **options)
        if summary.get("full_target_reached") is not True:
            return summary
        _, passing = _passing_slots(root, config, split, banks[split], directory, scope)
        if set(passing) != set(range(target)):
            raise ValueError("Staged call count differs from the completed generation target")
        rows = [passing[slot] for slot in range(target)]
        hashes = {row["sha256"] for row in rows}
        if (len(hashes) != target or source_hashes & hashes or (preflight and hashes & heard_hashes)
                or any(row.get("admission") for row in rows)):
            raise ValueError("Staged calls must be unique across splits and awaiting admission")
        source_hashes.update(hashes)
        manifest = directory / "postgeneration_sources.jsonl"
        write_immutable_rows(manifest, rows)
        sources[split] = _binding(root, manifest)
        representatives = [{**passing[slot], "review_group": group} for slot in range(len(banks[split]))]
        if preflight:
            for number, representative in enumerate(representatives, 1):
                representative["pronunciation_policy"] = PREPOST_PILOT_POLICY
                representative["voice_number"] = number
        selected.extend(representatives)
        outputs.append(summary)
    queue = root / "data/pilot" / "postgeneration_review.jsonl"
    if preflight:
        queue = root / "data/call_pronunciation" / f"production_{revision[:16]}.jsonl"
    write_immutable_rows(queue, selected)
    return {"outputs": outputs, "train_originals": targets["train"], "validation_originals": targets["validation"],
            "independent_STT_runs": 0, "full_target_reached": True, "queue": str(queue),
            "queue_binding": _binding(root, queue), "preview_sources": sources,
            **({"review_kind": "positive", "preflight_reviews": list(prior_rounds)} if preflight else {})}


def _postgeneration_admission(root, config, generated):
    """Bind actual listened dispositions to this completed corpus."""
    approval = read_json(root / "approval.json")
    queue = bound_file(root, generated["queue_binding"])
    queue_rows = read_rows(queue)
    prepost = generated.get("review_kind") == "positive"
    report_path = root / "review" / ("positive_results.json" if prepost else "pilot_results.json")
    if not queue_rows or not report_path.is_file():
        raise ValueError("The generated call corpus still needs approved pronunciation review")
    if not prepost and (approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True
            or approval.get("wakeword") != config["wakeword"] or approval.get("human_reference_used") is not False
            or approval.get("tts_settings_id") != identity(config["tts"]) or approval.get("generation_method") != "clone"
            or approval.get("accepted_pilot_items") != len(queue_rows) or approval.get("pilot_review_sha256") != digest(report_path)):
        raise ValueError("The generated call corpus still needs approved pronunciation review")
    report = read_json(report_path)
    decisions = report.get("decisions", {})
    reviewed = {row["sample_id"]: row["sha256"] for row in queue_rows}
    if ((prepost and report.get("kind") != "positive_listening_review")
            or report.get("queue_sha256") != generated["queue_binding"]["sha256"]
            or report.get("wakeword") != config["wakeword"]
            or report.get("source_scope") != "Human dispositions entered through the local review screen"
            or not report.get("saved_at") or report.get("queued_audio_sha256") != reviewed
            or set(decisions) != set(reviewed)
            or any(answer.get("decision") != "keep" or answer.get("listened") is not True for answer in decisions.values())):
        raise ValueError("Saved pronunciation review does not approve this generated call corpus")
    if prepost:
        banks = _preflight_banks(root, {
            f"{split}_references": sum(row["split"] == split for row in queue_rows) for split in REVIEW_GROUPS
        })
        accepted, pending, preflight_hashes, _ = _preflight_state(root, config, banks, generated["preflight_reviews"])
        if (pending or len(accepted) != len(queue_rows)
                or {row["voice_family_id"] for row in queue_rows} != set(accepted)
                or preflight_hashes & set(reviewed.values())
                or any(row.get("pronunciation_policy") != PREPOST_PILOT_POLICY
                       or row.get("voice_reference_sha256") != accepted[row["voice_family_id"]]["sha256"]
                       for row in queue_rows)):
            raise ValueError("본검수 목록이 사전 검수에서 승인한 기준 목소리와 일치하지 않습니다.")
        approval.update(state="accepted", generation_policy_approved=True, human_reference_used=False,
            wakeword=config["wakeword"], tts_settings_id=identity(config["tts"]), generation_method="clone",
            positive_review_sha256=digest(report_path), call_preflight_reviews=generated["preflight_reviews"],
            approval_scope="Accepted reference revisions and independent representative pronunciation review")
    evidence = {**approval, "postgeneration_review_queue": generated["queue_binding"],
                "postgeneration_sources": generated["preview_sources"]}
    return evidence, reviewed


def _admitted_calls(root, generated, split, admission, reviewed):
    source = bound_file(root, generated["preview_sources"][split])
    for row in iter_rows(source):
        yield {**row, "admission": admission, "individual_human_listening": reviewed.get(row["sample_id"]) == row["sha256"]}


def admit_postgeneration_positives(project, generated):
    """Publish both approved splits idempotently after authentic review."""
    root, config = load_project(project)
    with writer_lock(root / "data/pilot"):
        evidence, reviewed = _postgeneration_admission(root, config, generated)
        manifests = {split: resolved(root, config["manifests"]["positive_" + split]) for split in REVIEW_GROUPS}
        unmatched = set(reviewed)
        for split in manifests:
            source = bound_file(root, generated["preview_sources"][split])
            for row in iter_rows(source):
                bound_file(root, row)
                if reviewed.get(row["sample_id"]) == row["sha256"]:
                    unmatched.discard(row["sample_id"])
        if unmatched:
            raise ValueError("The generated corpus is missing a reviewed representative")
        admission = snapshot_json(root, evidence, "generation_policy")
        for split, destination in manifests.items():
            write_immutable_rows(destination, _admitted_calls(root, generated, split, admission, reviewed), replace_empty=True)
        if generated.get("review_kind") == "positive":
            policy = {key: value for key, value in evidence.items()
                      if key not in ("postgeneration_review_queue", "postgeneration_sources")}
            if read_json(root / "approval.json") != policy:
                write_json(root / "approval.json", policy)
    return list(manifests.values())


def matches_postgeneration_publication(project, generated, path):
    """Recognize an already-published split if the review worker was interrupted."""
    root, config = load_project(project)
    for split in REVIEW_GROUPS:
        if path != resolved(root, config["manifests"]["positive_" + split]):
            continue
        evidence, reviewed = _postgeneration_admission(root, config, generated)
        receipt = root / "review/evidence" / f"generation_policy_{identity(evidence)}.json"
        if not receipt.is_file() or read_json(receipt) != evidence:
            return False
        admission = {"path": receipt.relative_to(root).as_posix(), "sha256": digest(receipt)}
        expected = _admitted_calls(root, generated, split, admission, reviewed)
        return all(actual == desired for actual, desired in zip_longest(iter_rows(path), expected))
    return False
