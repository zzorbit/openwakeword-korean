"""Selective, hash-bound repair of rejected post-generation voice families.

Original corpora and human reports remain immutable inputs. A repair reuses
the existing clone generator and retry limits, then publishes a new effective
corpus only after every replacement passes the unchanged technical checks.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .common import (bound_file, digest, identity, read_json, read_rows, resolved,
                     snapshot_json, write_immutable_rows, write_json, writer_lock)
from .pilot import PREPOST_PILOT_POLICY, REVIEW_GROUPS, _admitted_references, _binding, _passing_slots, _preflight_state
from .project import load_project
from .reference_qc import _human_answers
from .storage import project_storage

POLICY = "selective_postgeneration_repair_v1"


def _immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable post-generation repair evidence changed")
    else:
        write_json(path, value)


def _directory(root, request):
    return root / "data/call_pronunciation/repairs" / request["plan"]["sha256"]


def _read_plan(root, config, request, reference_review=None):
    if request.get("policy") != POLICY or not isinstance(request.get("plan"), dict):
        raise ValueError("Unsupported post-generation repair request")
    plan = read_json(bound_file(root, request["plan"]))
    if (plan.get("policy") != POLICY or plan.get("project_sha256") != digest(root / "project.json")
            or plan.get("tts_settings_id") != identity(config["tts"])
            or reference_review is not None and reference_review != plan.get("reference_review")):
        raise ValueError("Repair request no longer matches the frozen project")
    expected = _request(request["plan"], plan)
    if request != expected:
        raise ValueError("Repair request summary differs from its bound plan")
    return plan


def _request(binding, plan):
    return {"policy": POLICY, "plan": binding, "round": plan["round"],
            "target": sum(item["count"] for item in plan["families"]),
            "pending_review_count": len(plan["families"]),
            "carried_review_count": plan["review_count"] - len(plan["families"]),
            "affected_families": [{key: item[key] for key in ("split", "voice_family_id", "reference_sha256", "count")}
                                  for item in plan["families"]]}


def _reference_banks(root, counts, reference_review):
    """Apply the frozen selected-reference overlay without process-global env."""
    from .reference_voice_review import overlay_reference_banks
    approval = read_json(root / "approval.json")
    answers = _human_answers(root, approval)
    banks = {split: _admitted_references(root, approval, answers, split, counts[split]) for split in REVIEW_GROUPS}
    banks = overlay_reference_banks(root, banks, reference_review)
    if ({row["voice_family_id"] for row in banks["train"]} & {row["voice_family_id"] for row in banks["validation"]}
            or {row["sha256"] for row in banks["train"]} & {row["sha256"] for row in banks["validation"]}):
        raise ValueError("Train and Validation reference banks must remain disjoint")
    return banks


def _generated(root, config, generated, *, verify_audio=True, reference_review=None):
    if generated.get("review_kind") != "positive" or generated.get("full_target_reached") is not True:
        raise ValueError("Only a complete two-phase call corpus can be repaired")
    queue_path = bound_file(root, generated["queue_binding"])
    if resolved(root, generated["queue"]) != queue_path:
        raise ValueError("Generated queue path differs from its binding")
    queue = read_rows(queue_path)
    counts = Counter(row.get("split") for row in queue)
    banks = _reference_banks(root, counts, reference_review)
    accepted, pending, heard, reference_hashes = _preflight_state(root, config, banks, generated["preflight_reviews"])
    ordered = [(split, reference["voice_family_id"]) for split, bank in banks.items() for reference in bank]
    if (pending or not queue or [(row.get("split"), row.get("voice_family_id")) for row in queue] != ordered
            or set(accepted) != {family for _, family in ordered}):
        raise ValueError("Post-generation representatives differ from approved reference families")
    sources, ids, hashes = {}, set(), set()
    for split, bank in banks.items():
        rows = read_rows(bound_file(root, generated["preview_sources"][split]))
        target = generated[f"{split}_originals"]
        maximum = config["targets"]["positive_originals" if split == "train" else "validation_originals"]
        if type(target) is not int or target < len(bank) or target > maximum or len(rows) != target:
            raise ValueError("Repair source counts differ from the frozen generation target")
        for position, row in enumerate(rows):
            family = bank[position % len(bank)]["voice_family_id"]
            if (row.get("sample_id") in ids or row.get("sha256") in hashes or not row.get("sample_id") or not row.get("sha256")
                    or row.get("split") != split or row.get("voice_family_id") != family
                    or row.get("voice_reference_sha256") != accepted[family]["sha256"]
                    or row.get("text") != config["wakeword"] or row.get("tts_settings_id") != identity(config["tts"])
                    or row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False
                    or row.get("generation_method") != "clone" or row.get("qc", {}).get("passed") is not True
                    or row.get("admission") or row.get("individual_human_listening") is not False):
                raise ValueError("Repair source identity, reference, uniqueness or technical admission changed")
            if verify_audio:
                bound_file(root, row)
            ids.add(row["sample_id"]); hashes.add(row["sha256"])
        sources[split] = rows
    if hashes & heard:
        raise ValueError("Preflight samples cannot be reused as production calls")
    by_id = {row["sample_id"]: row for rows in sources.values() for row in rows}
    queue_ids = set()
    for row in queue:
        source = by_id.get(row.get("sample_id"))
        if (row.get("sample_id") in queue_ids or source is None or any(row.get(key) != value for key, value in source.items())
                or row.get("pronunciation_policy") != PREPOST_PILOT_POLICY or row.get("review_group") != REVIEW_GROUPS[row["split"]]):
            raise ValueError("A post-generation representative is not an exact member of its source corpus")
        queue_ids.add(row["sample_id"])
    return queue, sources, accepted, set(heard) | set(reference_hashes)


def _report(root, config, generated, queue, binding):
    report = read_json(bound_file(root, binding))
    decisions = report.get("decisions", {})
    if (report.get("kind") != "positive_listening_review" or report.get("wakeword") != config["wakeword"]
            or report.get("queue_sha256") != generated["queue_binding"]["sha256"]
            or report.get("queued_audio_sha256") != {row["sample_id"]: row["sha256"] for row in queue}
            or report.get("source_scope") != "Human dispositions entered through the local review screen"
            or not report.get("saved_at") or set(decisions) != {row["sample_id"] for row in queue}
            or any(not isinstance(answer, dict) or answer.get("decision") not in ("keep", "discard")
                   or answer.get("listened") is not True for answer in decisions.values())):
        raise ValueError("Repair requires the actual complete, hash-bound post-generation review")
    return report


def _ancestor_hashes(root, generated):
    hashes, seen = set(), set()
    while True:
        for binding in generated["preview_sources"].values():
            hashes.update(row["sha256"] for row in read_rows(bound_file(root, binding)))
        hashes.update(row["sha256"] for row in read_rows(bound_file(root, generated["queue_binding"])))
        repair = generated.get("postgeneration_repair")
        if not repair:
            return hashes
        key = repair["evidence"]["sha256"]
        if key in seen:
            raise ValueError("Cyclic repair ancestry")
        seen.add(key)
        evidence = read_json(bound_file(root, repair["evidence"]))
        generated = read_json(bound_file(root, evidence["parent_generated"]))


def _rejected_families(queue, sources, accepted, report):
    result = []
    for row in queue:
        if report["decisions"][row["sample_id"]]["decision"] == "discard":
            family, split = row["voice_family_id"], row["split"]
            result.append({"split": split, "voice_family_id": family, "reference_sha256": accepted[family]["sha256"],
                           "reference": accepted[family], "count": sum(source["voice_family_id"] == family for source in sources[split])})
    return result


def _plan_inputs(root, config, plan, *, verify_audio):
    parent = read_json(bound_file(root, plan["parent_generated"]))
    queue, sources, accepted, excluded = _generated(root, config, parent, verify_audio=verify_audio,
                                                   reference_review=plan.get("reference_review"))
    report = _report(root, config, parent, queue, plan["parent_report"])
    previous = parent.get("postgeneration_repair")
    expected_round = (read_json(bound_file(root, previous["plan"]))["round"] if previous else 0) + 1
    if (plan.get("families") != _rejected_families(queue, sources, accepted, report) or not plan["families"]
            or plan.get("review_count") != len(queue) or plan.get("round") != expected_round
            or plan.get("excluded_sha256") != sorted(excluded | _ancestor_hashes(root, parent))):
        raise ValueError("Repair plan changed the rejected families, exact quotas, references or exclusions")
    return parent, queue, sources, report


@project_storage
def prepare_postgeneration_repair(project, generated, report_binding, *, reference_review=None):
    """Freeze the held report and only the rejected families' exact quotas."""
    root, config = load_project(project)
    if reference_review is None and generated.get("postgeneration_repair"):
        reference_review = read_json(bound_file(root, generated["postgeneration_repair"]["plan"])).get("reference_review")
    queue, sources, accepted, excluded = _generated(root, config, generated, reference_review=reference_review)
    report = _report(root, config, generated, queue, report_binding)
    families = _rejected_families(queue, sources, accepted, report)
    if not families:
        raise ValueError("No explicitly unsuitable post-generation voice needs repair")
    previous = generated.get("postgeneration_repair")
    round_number = (read_json(bound_file(root, previous["plan"]))["round"] if previous else 0) + 1
    plan = {"policy": POLICY, "round": round_number, "project_sha256": digest(project),
            "tts_settings_id": identity(config["tts"]), "reference_review": reference_review, "review_count": len(queue), "families": families,
            "parent_generated": snapshot_json(root, generated, "postgeneration_repair_parent_generated"),
            "parent_report": snapshot_json(root, report, "postgeneration_repair_parent_report"),
            "excluded_sha256": sorted(excluded | _ancestor_hashes(root, generated))}
    return _request(snapshot_json(root, plan, "postgeneration_repair_plan"), plan)


def _runtime(config, root):
    from .tts_runtime import CloneRuntime
    return CloneRuntime(config["tts"]["clone_model"], config["tts"], root, "auto", config["training"]["cpu_threads"])


def _replacements(project, root, config, request, plan, *, generate=False):
    from .tts import call_generation_location, generate as synthesize
    excluded, replacements, scopes, summaries = set(plan["excluded_sha256"]), {}, [], []
    model, completed = None, 0
    directory = _directory(root, request)
    for item in plan["families"]:
        split, family, target = item["split"], item["voice_family_id"], item["count"]
        scope = {"kind": "postgeneration_repair", "revision": request["plan"]["sha256"],
                 "references": [item["reference"]], "excluded_sha256": sorted(excluded)}
        output, _ = call_generation_location(root, split, scope)
        rows, passing = _passing_slots(root, config, split, [item["reference"]], output, scope)
        if generate:
            scope_binding = snapshot_json(root, scope, "postgeneration_repair_scope")
            progress = {"plan": request["plan"], "done": completed, "total": request["target"], "phase": "generating",
                        "current": {"journal": (output / "candidates.jsonl").relative_to(root).as_posix(), "target": target}}
            write_json(directory / "progress.json", progress)
            if set(passing) != set(range(target)):
                if model is None:
                    model = _runtime(config, root)
                summary = synthesize(project, split, target, True, "clone", model=model, staged_scope=scope)
                rows, passing = _passing_slots(root, config, split, [item["reference"]], output, scope)
            else:
                summary = {"target": target, "technical_pass": target, "attempts": len(rows), "full_target_reached": True, "reused": True}
            summaries.append(summary); scopes.append(scope_binding)
            if summary.get("full_target_reached") is not True or set(passing) != set(range(target)):
                write_json(directory / "progress.json", {**progress, "done": completed + len(passing), "phase": "blocked", "current": None})
                return None, scopes, summaries
        elif set(passing) != set(range(target)):
            raise ValueError("Repair completion has missing or unverified generation slots")
        else:
            scope_path = root / "review/evidence" / f"postgeneration_repair_scope_{identity(scope)}.json"
            if read_json(scope_path) != scope:
                raise ValueError("Repair scope evidence changed")
            scopes.append({"path": scope_path.relative_to(root).as_posix(), "sha256": digest(scope_path)})
        selected = [passing[slot] for slot in range(target)]
        if (any(row.get("voice_family_id") != family or row.get("voice_reference_sha256") != item["reference_sha256"]
                or row.get("qc", {}).get("passed") is not True or row.get("admission") for row in selected)
                or len({row["sha256"] for row in selected}) != target or {row["sha256"] for row in selected} & excluded):
            raise ValueError("Repair output reused an excluded waveform or changed its exact reference")
        replacements[(split, family)] = selected
        excluded.update(row["sha256"] for row in rows if row.get("sha256")); completed += target
    return replacements, scopes, summaries


def _compose(parent_queue, parent_sources, replacements, report):
    sources = {}
    for split, rows in parent_sources.items():
        positions = Counter(); composed = []
        for row in rows:
            key = (split, row["voice_family_id"])
            if key in replacements:
                row = replacements[key][positions[key]]; positions[key] += 1
            composed.append(row)
        sources[split] = composed
    queue, carried, pending = [], {}, []
    for row in parent_queue:
        key = (row["split"], row["voice_family_id"])
        if key in replacements:
            replacement = {**replacements[key][0], "review_group": row["review_group"],
                           "pronunciation_policy": PREPOST_PILOT_POLICY, "voice_number": row["voice_number"]}
            queue.append(replacement); pending.append(replacement["sample_id"])
        else:
            queue.append(row); carried[row["sample_id"]] = deepcopy(report["decisions"][row["sample_id"]])
    all_rows = [row for rows in sources.values() for row in rows]
    if len({row["sample_id"] for row in all_rows}) != len(all_rows) or len({row["sha256"] for row in all_rows}) != len(all_rows):
        raise ValueError("Effective repaired corpus contains duplicate IDs or waveforms")
    return queue, sources, carried, pending


def _completed(root, config, request, plan):
    result = read_json(_directory(root, request) / "completed.json")
    metadata = result.get("postgeneration_repair", {})
    if metadata.get("policy") != POLICY or metadata.get("plan") != request["plan"]:
        raise ValueError("Repair completion belongs to another plan")
    evidence = read_json(bound_file(root, metadata["evidence"]))
    parent, parent_queue, parent_sources, report = _plan_inputs(root, config, plan, verify_audio=False)
    replacements, scopes, _ = _replacements(root / "project.json", root, config, request, plan)
    queue, sources, carried, pending = _compose(parent_queue, parent_sources, replacements, report)
    if (evidence.get("policy") != POLICY or evidence.get("plan") != request["plan"]
            or evidence.get("parent_generated") != plan["parent_generated"] or evidence.get("parent_report") != plan["parent_report"]
            or evidence.get("scopes") != scopes or evidence.get("carried_decisions") != carried or evidence.get("pending_ids") != pending
            or result != {**evidence.get("generated", {}), "postgeneration_repair": metadata}
            or read_rows(bound_file(root, result["queue_binding"])) != queue
            or any(read_rows(bound_file(root, result["preview_sources"][split])) != rows for split, rows in sources.items())):
        raise ValueError("Repair completion, carried decisions or effective corpus changed")
    _generated(root, config, result, verify_audio=False, reference_review=plan.get("reference_review"))
    return result, evidence


@project_storage
def generate_postgeneration_repair(project, settings, request, *, reference_review=None):
    """Resume one planned round; publish nothing until every bad family passes."""
    root, config = load_project(project)
    plan = _read_plan(root, config, request, reference_review)
    directory = _directory(root, request)
    with writer_lock(directory):
        if (directory / "completed.json").exists():
            return _completed(root, config, request, plan)[0]
        parent, queue, sources, report = _plan_inputs(root, config, plan, verify_audio=True)
        if settings["validation_originals"] != parent["validation_originals"] or settings["originals"] != parent["train_originals"]:
            raise ValueError("Repair settings changed the frozen corpus target")
        replacements, scopes, outputs = _replacements(project, root, config, request, plan, generate=True)
        if replacements is None:
            return {"full_target_reached": False, "target": request["target"], "technical_pass": postgeneration_repair_progress(project, request)["done"],
                    "outputs": outputs, "repair_request": request, "exhausted_slots": [slot for output in outputs for slot in output.get("exhausted_slots", [])]}
        new_queue, new_sources, carried, pending = _compose(queue, sources, replacements, report)
        bindings = {}
        for split, rows in new_sources.items():
            path = directory / f"effective_{split}.jsonl"; write_immutable_rows(path, rows); bindings[split] = _binding(root, path)
        path = directory / "review_queue.jsonl"; write_immutable_rows(path, new_queue)
        generated = {**{key: value for key, value in parent.items() if key != "postgeneration_repair"},
                     "outputs": outputs, "queue": str(path), "queue_binding": _binding(root, path), "preview_sources": bindings,
                     "full_target_reached": True}
        evidence = {"policy": POLICY, "plan": request["plan"], "parent_generated": plan["parent_generated"],
                    "parent_report": plan["parent_report"], "generated": generated, "scopes": scopes,
                    "carried_decisions": carried, "pending_ids": pending}
        result = {**generated, "postgeneration_repair": {"policy": POLICY, "plan": request["plan"],
                    "evidence": snapshot_json(root, evidence, "postgeneration_repair_completion")}}
        _immutable_json(directory / "completed.json", result)
        write_json(directory / "progress.json", {"plan": request["plan"], "done": request["target"], "total": request["target"], "phase": "completed", "current": None})
        return _completed(root, config, request, plan)[0]


def resolve_postgeneration_repair(project, original_generated, rounds, *, reference_review=None):
    """Resolve only completed, linked rounds; a partial last round stays pending."""
    root, config = load_project(project)
    generated = original_generated
    for position, entry in enumerate(rounds):
        request = entry.get("request", entry)
        plan = _read_plan(root, config, request, reference_review)
        if read_json(bound_file(root, plan["parent_generated"])) != generated:
            raise ValueError("Repair rounds are not a consecutive bound corpus chain")
        if not (_directory(root, request) / "completed.json").exists():
            if position != len(rounds) - 1:
                raise ValueError("A later repair cannot skip an incomplete round")
            break
        generated = _completed(root, config, request, plan)[0]
    return generated


def postgeneration_repair_review(project, generated, *, reference_review=None):
    """Return factual carried answers only for exact unchanged queue members."""
    root, config = load_project(project)
    metadata = generated.get("postgeneration_repair")
    if not metadata:
        return {"carried_decisions": {}, "pending_ids": [row["sample_id"] for row in read_rows(bound_file(root, generated["queue_binding"]))]}
    plan = read_json(bound_file(root, metadata["plan"]))
    request = _request(metadata["plan"], plan); _read_plan(root, config, request, reference_review)
    completed, evidence = _completed(root, config, request, plan)
    if completed != generated:
        raise ValueError("Review descriptor differs from the completed repair")
    return {"carried_decisions": deepcopy(evidence["carried_decisions"]), "pending_ids": list(evidence["pending_ids"]),
            "plan": metadata["plan"], "evidence": metadata["evidence"]}


def postgeneration_repair_progress(project, request):
    """Small UI telemetry: never scans the complete effective call corpus."""
    root, _ = load_project(project)
    path = _directory(root, request) / "progress.json"
    if not path.exists():
        return {"done": 0, "total": request["target"], "phase": "pending"}
    progress = read_json(path)
    if progress.get("plan") != request["plan"] or progress.get("total") != request["target"]:
        raise ValueError("Repair progress belongs to another plan")
    current = progress.get("current")
    additional = 0
    if current:
        journal = resolved(root, current["journal"])
        if not journal.is_relative_to(root):
            raise ValueError("Repair progress journal is outside the project")
        additional = len({row["slot"] for row in read_rows(journal) if row.get("qc", {}).get("passed") is True
                          and type(row.get("slot")) is int and 0 <= row["slot"] < current["target"]})
    return {"done": min(request["target"], progress["done"] + additional), "total": request["target"], "phase": progress["phase"]}
