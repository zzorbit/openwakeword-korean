"""Explicit representative review of a frozen public Train source pool.

This optional path does not change the existing per-source listening review.
It admits exact source hashes under an explicitly accepted pool policy, while
recording which previews were actually heard and which originals were not.
"""
from collections import Counter
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock
from .data import CATEGORIES, audit_project, source_key, validate_hard_negative
from .project import load_project


def _binding(root, entry):
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise ValueError("Evidence needs its frozen path and SHA256")
    path = bound_file(root, entry)
    return path, {"path": str(path), "sha256": entry["sha256"]}


def _stratum(row):
    dataset = row.get("dataset_id")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("Every pool source needs a declared dataset_id")
    category = row.get("category")
    if category not in (*CATEGORIES, "rir"):
        raise ValueError("Every pool source needs a supported category")
    return dataset, category


def _technical_check(path, category, *, start_s=0, end_s=None):
    """Inspect actual decoded samples in bounded blocks, rather than trust a flag."""
    count, energy, peak, clipped, peak_frame = 0, 0.0, 0.0, 0, 0
    with sf.SoundFile(path) as audio:
        if audio.frames < 1 or audio.samplerate < 1 or audio.channels != 1:
            raise ValueError("Pool sources must be nonempty mono audio")
        rate, frames = audio.samplerate, audio.frames
        duration = frames / rate
        end_s = duration if end_s is None else end_s
        if (type(start_s) not in (int, float) or type(end_s) not in (int, float)
                or not math.isfinite(start_s) or not math.isfinite(end_s)
                or not 0 <= start_s < end_s <= duration + 1e-9):
            raise ValueError("Technical source interval must be finite and inside audio")
        begin, finish = round(start_s * rate), min(frames, round(end_s * rate))
        if finish <= begin:
            raise ValueError("Technical source interval contains no samples")
        audio.seek(begin)
        for samples in audio.blocks(blocksize=65536, frames=finish - begin, dtype="float64"):
            if not np.isfinite(samples).all():
                raise ValueError("Nonfinite source audio cannot be policy-admitted")
            local_peak = float(np.max(np.abs(samples)))
            if local_peak > peak:
                peak_frame = begin + count + int(np.argmax(np.abs(samples)))
                peak = local_peak
            count += len(samples)
            energy += float(np.dot(samples, samples))
            clipped += int(np.count_nonzero(np.abs(samples) >= 32760 / 32768))
    rms = math.sqrt(energy / count)
    if not math.isfinite(rms) or rms <= 1e-8 or peak > 1.0:
        raise ValueError("Silent or out-of-range source audio cannot be policy-admitted")
    fraction = clipped / count
    # Impulse responses are filters, not ordinary microphone recordings; their
    # isolated impulse peak is not assessed as speech clipping.
    if category != "rir" and fraction >= .01:
        raise ValueError("Excessively clipped source audio cannot be policy-admitted")
    return {"finite": True, "rms": rms, "peak": peak, "clipped_fraction": fraction,
            "frames": frames, "sample_rate": rate, "channels": 1,
            "inspected_frames": count, "inspected_interval_s": [start_s, min(end_s, duration)],
            "peak_frame": peak_frame,
            "kind": "impulse_response" if category == "rir" else "recording"}


def _synthetic_contract(root, row, config, contracts):
    binding = row.get("generation_contract")
    path, frozen_binding = _binding(root, binding)
    key = (str(path), frozen_binding["sha256"])
    if key not in contracts:
        contract = read_json(path)
        if contract.get("kind") != "qwen_hard_negative_generation_contract_v1" or contract.get("word") != config["wakeword"]:
            raise ValueError("Synthetic pool needs this word's Hard Negative generation contract")
        if contract.get("split") != "train" or contract.get("human_reference_used") is not False or contract.get("method") != "clone":
            raise ValueError("Synthetic Hard Negative contract must use Train synthetic-only clone references")
        if identity(contract.get("tts")) != identity(config["tts"]):
            raise ValueError("Synthetic generation settings differ from the frozen project settings")
        _binding(root, contract.get("implementation"))
        references = contract.get("references")
        if not isinstance(references, list) or not references or not isinstance(contract.get("texts"), list) or not contract["texts"]:
            raise ValueError("Synthetic contract needs actual references and declared phrases")
        train_refs = read_rows(root / "data/references/train/references.jsonl")
        allowed = {(r["voice_family_id"], r["sha256"]): r for r in train_refs
                   if r.get("split") == "train" and r.get("status") == "accepted" and r.get("human_reference_used") is False}
        accepted_families = set(read_json(root / "approval.json").get("accepted_reference_families", []))
        selected = {}
        for reference in references:
            family = reference.get("voice_family_id")
            # Check Train identity before reading any held-out reference audio.
            if reference.get("split") != "train" or reference.get("human_reference_used") is not False or family not in accepted_families:
                raise ValueError("Synthetic reference must be an approved Train family without human reference")
            original = allowed.get((family, reference.get("sha256")))
            if original is None or original.get("provenance") != "qwen_synthetic_reference" or resolved(root, reference["path"]) != resolved(root, original["path"]):
                raise ValueError("Synthetic reference does not bind this project's accepted Train reference")
            bound_file(root, reference)
            selected[(family, reference["sha256"])] = reference
        # A preserved generation contract can include phrases subsequently
        # excluded by the user. Validate its declaration's shape here and the
        # actual selected row against the current non-call policy below.
        if any(not isinstance(text, str) or not text.strip() for text in contract["texts"]):
            raise ValueError("Synthetic contract phrases must be nonempty strings")
        contracts[key] = (contract, selected, frozen_binding)
    contract, selected, _ = contracts[key]
    validate_hard_negative(row, config)
    if row.get("generation_contract_id") != identity(contract) or row.get("tts_settings_id") != identity(contract["tts"]):
        raise ValueError("Synthetic row generation contract/settings identity changed")
    if row["text"] not in contract["texts"] or (row["voice_family_id"], row.get("reference_sha256")) not in selected:
        raise ValueError("Synthetic phrase or reference was not declared in its generation contract")
    return row["dataset_id"], row["text"], row["voice_family_id"]


def _selected_pool(root, sources, previews, result, excluded_hard_phrases):
    """Narrow saved consent without editing its source, queue or dispositions."""
    if (isinstance(excluded_hard_phrases, (str, bytes))
            or not isinstance(excluded_hard_phrases, (list, tuple, set, frozenset))
            or any(not isinstance(text, str) or not text.strip() or text != text.strip()
                   for text in excluded_hard_phrases)):
        raise ValueError("excluded_hard_phrases must contain exact nonempty phrase strings")
    excluded = set(excluded_hard_phrases)
    if len(excluded) != len(excluded_hard_phrases):
        raise ValueError("excluded_hard_phrases must not repeat a phrase")
    decisions = result.get("decisions")
    preview_ids = [row.get("sample_id") for row in previews]
    if (not all(preview_ids) or len(set(preview_ids)) != len(previews)
            or not isinstance(decisions, dict) or set(decisions) != set(preview_ids)
            or any(not isinstance(value, dict) for value in decisions.values())):
        raise ValueError("Every unique queued preview needs its saved human disposition")
    if result.get("queued_audio_sha256") != {row["sample_id"]: row["sha256"] for row in previews}:
        raise ValueError("Saved answers do not bind the exact queued preview audio")
    if not excluded:
        return sources, previews, [], []

    source_by_binding, excluded_sources, selected_sources = {}, [], []
    seen_samples, seen_canonical = set(), set()
    for row in sources:
        sample, canonical = row.get("sample_id"), row.get("canonical_source_id")
        if (not sample or sample in seen_samples or not canonical or canonical != source_key(row)
                or canonical in seen_canonical):
            raise ValueError("Pool sources need unique sample and canonical source IDs")
        seen_samples.add(sample); seen_canonical.add(canonical)
        if row.get("split") != "train":
            raise ValueError("Pool policy only admits Train; Final/Validation stay reserved")
        key = (str(resolved(root, row["path"])), row["sha256"])
        if key in source_by_binding:
            raise ValueError("Pool source bindings must be unique")
        source_by_binding[key] = row
        if row.get("text") in excluded:
            if row.get("provenance") != "qwen_tts" or row.get("category") != "competing_speech":
                raise ValueError("Phrase exclusions only narrow Qwen Hard Negative sources")
            excluded_sources.append(row)
        else:
            selected_sources.append(row)
    if {row["text"] for row in excluded_sources} != excluded:
        raise ValueError("Every excluded Hard Negative phrase must exist in the frozen source pool")
    if not selected_sources:
        raise ValueError("Phrase exclusions cannot leave an empty source pool")

    selected_previews, excluded_previews, evidenced_phrases = [], [], set()
    for preview in previews:
        binding = preview.get("reviewed_source")
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("Evidence needs its frozen path and SHA256")
        source = source_by_binding.get((str(resolved(root, binding["path"])), binding["sha256"]))
        if source is None:
            raise ValueError("Representative does not bind an exact source in the frozen pool")
        if (preview.get("canonical_source_id") != source["canonical_source_id"]
                or _stratum(preview) != _stratum(source)):
            raise ValueError("Representative canonical source or dataset/category identity changed")
        if source.get("text") in excluded:
            decision = decisions[preview["sample_id"]]
            if decision.get("decision") != "discard" or decision.get("listened") is not True:
                raise ValueError("Excluded Hard Negative representatives require actual listened/discard answers")
            evidenced_phrases.add(source["text"])
            excluded_previews.append({
                "sample_id": preview["sample_id"], "sha256": preview["sha256"],
                "reviewed_source_sha256": source["sha256"],
                "canonical_source_id": source["canonical_source_id"],
                "text": source["text"], "voice_family_id": source.get("voice_family_id"),
                "saved_human_disposition": dict(decision),
            })
        else:
            selected_previews.append(preview)
    if evidenced_phrases != excluded:
        raise ValueError("Every excluded phrase needs a saved listened/discard representative")
    return selected_sources, selected_previews, excluded_sources, excluded_previews


def _exclude_public_sources(root, sources, previews, result, excluded_source_ids):
    """Narrow public consent using saved, actually heard discard answers."""
    if (isinstance(excluded_source_ids, (str, bytes))
            or not isinstance(excluded_source_ids, (list, tuple, set, frozenset))
            or any(not isinstance(s, str) or not s.strip() or s != s.strip()
                   for s in excluded_source_ids)):
        raise ValueError("excluded_source_ids needs exact nonempty canonical IDs")
    excluded = set(excluded_source_ids)
    if len(excluded) != len(excluded_source_ids):
        raise ValueError("excluded_source_ids must not repeat an ID")
    if not excluded:
        return sources, previews, [], []
    by_id = {r["canonical_source_id"]: r for r in sources}
    if len(by_id) != len(sources) or not excluded <= set(by_id):
        raise ValueError("Excluded source must exist uniquely in the frozen pool")
    if any(by_id[s].get("provenance") != "public_dataset" for s in excluded):
        raise ValueError("Source exclusions only narrow public dataset sources")
    kept, removed, evidenced = [], [], set()
    for preview in previews:
        source_id = preview.get("canonical_source_id")
        if source_id not in by_id:
            raise ValueError("Representative source identity changed")
        source = by_id[source_id]
        binding = preview.get("reviewed_source", {})
        if (resolved(root, binding.get("path", "")) != resolved(root, source["path"])
                or binding.get("sha256") != source["sha256"]):
            raise ValueError("Representative source binding changed")
        if source_id in excluded:
            answer = result["decisions"][preview["sample_id"]]
            if answer.get("decision") != "discard" or answer.get("listened") is not True:
                raise ValueError("Source exclusions require actually listened/discard answers")
            evidenced.add(source_id)
            removed.append({"sample_id": preview["sample_id"], "sha256": preview["sha256"],
                            "canonical_source_id": source_id,
                            "reviewed_source_sha256": source["sha256"],
                            "saved_human_disposition": dict(answer)})
        else:
            kept.append(preview)
    if evidenced != excluded:
        raise ValueError("Every excluded source needs an actually listened/discard representative")
    selected = [r for r in sources if r["canonical_source_id"] not in excluded]
    if not selected:
        raise ValueError("Source exclusions cannot leave an empty pool")
    return selected, kept, [by_id[s] for s in sorted(excluded)], removed


def approve_pool(project, *, source_manifest, representative_queue, answers, output,
                 excluded_hard_phrases=(), excluded_source_ids=()):
    """Create an immutable admission after explicit, hash-bound human consent.

    ``source_manifest``, ``representative_queue`` and ``answers`` are bindings
    containing ``path`` and ``sha256``; relative paths use the project root.
    Each public source declares dataset_id/category/canonical_source_id,
    Train split, provenance, source_url/license, category_basis and a prior
    technical_pass=True. Actual samples, hashes and cross-split identities are
    verified again here. A RIR also declares canonical_room_id.

    Answers use the existing content_listening_review shape, with additional
    source_manifest_sha256, representative_queue_sha256 and the explicit
    boolean whole_pool_policy_approved=True. A frozen representative queue
    must cover every dataset_id/category stratum. Every queued preview must
    have an actual listened/keep answer, and reviewed_source must bind the
    exact pool original. Unresolved discard/review answers reject admission.
    Explicit excluded_hard_phrases can narrow Qwen Hard Negative admission:
    their existing listened/discard answers remain unchanged, all originals
    of those phrases are excluded, and the full frozen evidence stays bound.

    Qwen-only Hard Negative rows additionally bind the generation contract,
    implementation, exact approved Train references, settings and phrase/family.
    Every synthetic phrase/family combination also needs a heard representative.

    This function never infers consent from playback or a default checkbox,
    and never claims all originals were heard or wakeword-free.
    """
    root, config = load_project(project)
    destination = resolved(root, output)
    if not destination.is_relative_to(root):
        raise ValueError("Admission output must stay inside the project")
    if destination.exists():
        raise ValueError("Pool admission output is immutable; choose a new path")
    manifest_path, manifest_binding = _binding(root, source_manifest)
    queue_path, queue_binding = _binding(root, representative_queue)
    answers_path, answers_binding = _binding(root, answers)
    result = read_json(answers_path)
    if result.get("whole_pool_policy_approved") is not True:
        raise ValueError("Explicit whole_pool_policy_approved=True human consent is required")
    if result.get("source_manifest_sha256") != manifest_binding["sha256"]:
        raise ValueError("Human pool consent does not bind this frozen source manifest")
    if result.get("representative_queue_sha256") != queue_binding["sha256"] or result.get("queue_sha256") != queue_binding["sha256"]:
        raise ValueError("Human pool consent does not bind this frozen representative queue")
    if result.get("kind") != "content_listening_review" or result.get("wakeword") != config["wakeword"]:
        raise ValueError("Content listening evidence is not for this project's word")
    if result.get("source_scope") != "Human dispositions entered through the local review screen" or not result.get("saved_at"):
        raise ValueError("Saved human review evidence is required")

    sources, previews = read_rows(manifest_path), read_rows(queue_path)
    if not sources or not previews:
        raise ValueError("Source pool and representative queue must be nonempty")
    full_source_count, full_preview_count = len(sources), len(previews)
    sources, previews, excluded_sources, excluded_previews = _selected_pool(
        root, sources, previews, result, excluded_hard_phrases)
    sources, previews, public_exclusions, public_previews = _exclude_public_sources(
        root, sources, previews, result, excluded_source_ids)
    excluded_sources += public_exclusions
    excluded_previews += public_previews
    source_ids, sample_ids, source_hashes, strata = set(), set(), set(), Counter()
    originals, technical, contracts = {}, [], {}
    synthetic_strata = set()
    for row in sources:
        sid = row.get("sample_id")
        canonical = row.get("canonical_source_id")
        if not sid or sid in sample_ids or not canonical or canonical != source_key(row) or canonical in source_ids:
            raise ValueError("Pool sources need unique sample and canonical source IDs")
        sample_ids.add(sid); source_ids.add(canonical)
        if row.get("split") != "train":
            raise ValueError("Pool policy only admits Train; Final/Validation stay reserved")
        if row.get("provenance") not in ("public_dataset", "qwen_tts"):
            raise ValueError("Pool supports public sources or Qwen-only Hard Negative")
        if row["provenance"] == "public_dataset" and (not row.get("source_url") or not row.get("license")):
            raise ValueError("Public pool needs source URL, license and public provenance")
        if row.get("technical_pass") is not True or row["provenance"] == "public_dataset" and not row.get("category_basis"):
            raise ValueError("Declared provider category basis and successful technical checks are required")
        if any(row.get(k) is True for k in ("listened", "human_listened", "individual_human_listening", "human_listening_completed")):
            raise ValueError("Pool policy rows must not claim individual human listening")
        stratum = _stratum(row)
        if stratum[1] == "rir" and not row.get("canonical_room_id"):
            raise ValueError("RIR pool needs its actual canonical room identity")
        if stratum[1] == "competing_speech" and row.get("speech") is not True:
            raise ValueError("Competing speech must be declared speech")
        if row["provenance"] == "qwen_tts":
            if stratum[1] != "competing_speech":
                raise ValueError("Synthetic Hard Negative category is competing_speech")
            validate_hard_negative(row, config)
        strata[stratum] += 1

    # Metadata audit precedes opening any source/preview/reference audio. A
    # held-out source cannot bypass Final locking by relabelling its split.
    preview_audit_rows = [{**r, "sample_id": f"__representative_{i}", "split": "train"}
                          for i, r in enumerate(previews)]
    preflight = audit_project(project, check_files=False, extra_rows=sources + preview_audit_rows)
    if not preflight["passed"]:
        raise ValueError("Pool source/split audit failed before audio access: " + str(preflight["errors"][:5]))
    for row in sources:
        if row["provenance"] == "qwen_tts":
            synthetic_strata.add(_synthetic_contract(root, row, config, contracts))
        stratum = _stratum(row)
        path = bound_file(root, row)
        key = (str(path), row["sha256"])
        if row["sha256"] in source_hashes:
            raise ValueError("Exact duplicate source audio must be deduplicated before pool consent")
        source_hashes.add(row["sha256"])
        originals[key] = row
        checks = _technical_check(path, stratum[1], **({"start_s": row.get("start_s", 0), "end_s": row.get("end_s")}
                                 if row["provenance"] == "public_dataset" else {}))
        if row["provenance"] == "qwen_tts":
            start, end = row.get("start_s"), row.get("end_s")
            duration = checks["frames"] / checks["sample_rate"]
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= duration or end - start > 1.84 + 1e-9):
                raise ValueError("Synthetic Hard Negative needs a finite complete phrase interval within the original and at most 1.84s")
            checks["phrase_interval_s"] = [start, end]
        technical.append({"canonical_source_id": row["canonical_source_id"], "sha256": row["sha256"], **checks})

    decisions = result["decisions"]
    covered, covered_synthetic, heard_sources, heard_previews = set(), set(), set(), []
    for preview in previews:
        decision = decisions[preview["sample_id"]]
        if decision.get("decision") != "keep" or decision.get("listened") is not True:
            raise ValueError("Unheard, discarded or review-held representatives need resolution before pool admission")
        bound_file(root, preview)
        binding = preview.get("reviewed_source")
        source_path, source_binding = _binding(root, binding)
        original = originals.get((str(source_path), source_binding["sha256"]))
        if original is None:
            raise ValueError("Representative does not bind an exact source in the frozen pool")
        if preview.get("canonical_source_id") != original["canonical_source_id"] or _stratum(preview) != _stratum(original):
            raise ValueError("Representative canonical source or dataset/category identity changed")
        covered.add(_stratum(original)); heard_sources.add(original["sha256"])
        if original["provenance"] == "qwen_tts":
            covered_synthetic.add((original["dataset_id"], original["text"], original["voice_family_id"]))
        heard_previews.append({"sample_id": preview["sample_id"], "sha256": preview["sha256"],
                              "reviewed_source_sha256": original["sha256"],
                              "canonical_source_id": original["canonical_source_id"],
                              "dataset_id": original["dataset_id"], "category": original["category"]})
    if covered != set(strata):
        raise ValueError("Actual listened representatives must cover every declared dataset/category stratum")
    if covered_synthetic != synthetic_strata:
        raise ValueError("Actual listened representatives must cover every synthetic phrase/family combination")

    audit = audit_project(project, extra_rows=sources)
    if not audit["passed"]:
        raise ValueError("Pool source/split audit failed: " + str(audit["errors"][:5]))
    # Recheck frozen evidence before publishing. A completed download alone is
    # never substituted for the representative answers or explicit policy vote.
    for binding in (manifest_binding, queue_binding, answers_binding):
        bound_file(root, binding)
    record = {
        "kind": "representative_source_pool_admission_v1", "state": "accepted",
        "content_review": "accepted", "wakeword": config["wakeword"], "created_at": now(),
        "whole_pool_policy_approved": True,
        "source_manifest": manifest_binding, "representative_queue": queue_binding,
        "human_answers": answers_binding, "accepted_source_sha256": sorted(source_hashes),
        "source_count": len(sources), "representative_preview_count": len(previews),
        "full_frozen_source_count": full_source_count, "full_frozen_preview_count": full_preview_count,
        "excluded_hard_phrases": sorted(set(excluded_hard_phrases)),
        "excluded_source_ids": sorted(set(excluded_source_ids)),
        "excluded_source_count": len(excluded_sources),
        "excluded_source_sha256": sorted(row["sha256"] for row in excluded_sources),
        "excluded_sources": [{"sample_id": row["sample_id"], "canonical_source_id": row["canonical_source_id"],
                              "path": row["path"], "sha256": row["sha256"], "text": row.get("text"),
                              "voice_family_id": row.get("voice_family_id"), "training_admitted": False}
                             for row in excluded_sources],
        "excluded_listened_discard_previews": excluded_previews,
        "actual_listened_previews": heard_previews,
        "representative_source_sha256": sorted(heard_sources),
        "unheard_source_policy_admitted_sha256": sorted(source_hashes - heard_sources),
        "individual_source_listening_claimed": False, "whole_sources_heard_claimed": False,
        "wakeword_absence_certified": False, "independent_STT_runs": 0,
        "technical_checks": technical,
        "strata": [{"dataset_id": d, "category": c, "source_count": count}
                   for (d, c), count in sorted(strata.items())],
        "synthetic_generation_contracts": [entry[2] for entry in contracts.values()],
        "synthetic_phrase_family_strata": [{"dataset_id": d, "text": t, "voice_family_id": f}
                                          for d, t, f in sorted(synthetic_strata)],
        "scope": "Explicit approval of accepted hashes from the exact frozen Train pool under public provider-label/provenance/technical/split policy or verified Qwen-only Hard Negative generation policy. Explicitly excluded phrases and their saved discard previews remain excluded; source/queue/answer evidence was not rewritten. Only the listed representative previews were heard; other originals were policy-admitted, not individually heard or certified wakeword-free.",
    }
    with writer_lock(destination.parent):
        if destination.exists():
            raise ValueError("Pool admission output is immutable; choose a new path")
        write_json(destination, record)
    return {"state": "saved", "source_count": len(sources), "representative_preview_count": len(previews),
            "admission": {"path": destination.relative_to(root).as_posix(), "sha256": digest(destination)},
            "scope": record["scope"]}
