"""Resumable Qwen non-call candidates; generation never grants admission.

The approved synthetic reference bank is reused, but each new phrase/voice
combination still needs representative content approval before training.
Technical energy bounds do not certify pronunciation or transcribe audio.
"""
from collections import Counter
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf

from . import tts
from .audio import pcm16
from .common import append_row, bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock
from .data import validate_hard_negative
from .hard_phrase_policy import protected_call_terms, validate_noncall_text
from .project import load_project
from .storage import project_storage

MAX_ORIGINALS = 100_000
MAX_ATTEMPTS = 3


def _phrases(values, wakeword):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Provide at least one distinct Korean non-call phrase")
    phrases = list(values)
    for phrase in phrases:
        if not isinstance(phrase, str):
            raise ValueError("Hard phrases must be standalone Hangul strings")
        validate_noncall_text(phrase, wakeword)
        if not re.fullmatch(r"[가-힣]{1,16}", phrase):
            raise ValueError("Hard phrases must be standalone Hangul without spaces")
    if len(set(phrases)) != len(phrases):
        raise ValueError("Hard phrases must be distinct")
    return phrases


def _combination(slot, phrases, references):
    """Visit the entire Cartesian product while balancing both margins.

    For four phrases/four voices each 16-slot block visits every pair once.
    Each four-slot sub-block also visits all four phrases and all four voices.
    """
    phrase_index = slot % len(phrases)
    voice_index = (slot + slot // len(phrases)) % len(references)
    # The diagonal schedule above covers the complete product when the two
    # dimensions are equal. A simple product schedule handles other banks.
    if len(phrases) != len(references):
        voice_index = (slot // len(phrases)) % len(references)
    return phrases[phrase_index], references[voice_index]


def _seed(config, slot, attempt, phrase, reference):
    return int(identity([config["tts"]["seed"], "hard_negative", slot, attempt,
                         phrase, reference["voice_family_id"], reference["sha256"]])[:8], 16)


def _validate_existing(root, output, rows, contract, config, phrases, references, count):
    passed, ids, combinations, hashes = {}, set(), set(), set()
    contract_binding = {"path": (output / "contract.json").relative_to(root).as_posix(),
                        "sha256": digest(output / "contract.json")}
    attempts = Counter()
    for row in rows:
        slot, attempt = row.get("slot"), row.get("attempt")
        if type(slot) is not int or not 0 <= slot < count or type(attempt) is not int or not 0 <= attempt < MAX_ATTEMPTS:
            raise ValueError("Invalid Hard candidate slot/attempt")
        phrase, reference = _combination(slot, phrases, references)
        sample_id = f"hard_train_{slot:06d}_attempt{attempt}"
        path = bound_file(root, row)
        if path.parent != output or path.name != sample_id + ".wav":
            raise ValueError("Hard candidate path escaped its generation directory")
        if sample_id in ids or (slot, attempt) in combinations or attempt != attempts[slot]:
            raise ValueError("Duplicate or noncontiguous Hard attempt journal")
        ids.add(sample_id); combinations.add((slot, attempt)); attempts[slot] += 1
        if (row.get("sample_id") != sample_id or row.get("text") != phrase or
            row.get("voice_family_id") != reference["voice_family_id"] or
            row.get("reference_sha256") != reference["sha256"] or row.get("split") != "train" or
            row.get("generation_contract") != contract_binding or
            row.get("generation_contract_id") != identity(contract) or
            row.get("tts_settings_id") != identity(config["tts"]) or
            row.get("seed") != _seed(config, slot, attempt, phrase, reference) or
            row.get("individual_human_listening") is not False or row.get("admission") is not None):
            raise ValueError("Hard candidate contract/provenance changed")
        validate_hard_negative(row, config)
        qc = row.get("qc", {})
        if row.get("technical_pass") is not qc.get("passed"):
            raise ValueError("Hard technical acceptance fields disagree")
        if qc.get("passed") is True:
            info = sf.info(path)
            start, end = row.get("start_s"), row.get("end_s")
            if (slot in passed or row["sha256"] in hashes or info.samplerate != 16000 or info.channels != 1 or
                type(start) not in (int, float) or type(end) not in (int, float) or
                not 0 <= start < end <= info.duration + 1e-6 or end - start > 1.840001):
                raise ValueError("Hard candidate duplicates or incomplete phrase interval")
            persisted_qc, _ = tts.technical_qc(sf.read(path, dtype="float64")[0], 16000)
            if not persisted_qc["passed"]:
                raise ValueError("Stored Hard candidate no longer passes technical QC")
            passed[slot] = row
        hashes.add(row["sha256"])
    return passed, attempts, hashes


@project_storage
def generate_hard(project, phrases, count=None, model=None, *, samples_per_phrase=None):
    """Create candidate-only originals, independently of Hard batch draws.

    New callers specify ``samples_per_phrase``; the total is the number of
    distinct phrases multiplied by that value. Each phrase cycles through
    the approved Train voices, covering every voice once when the per-phrase
    target equals the reference-bank size. Legacy ``count`` calls retain
    their total-budget semantics and 100,000-original limit (default 100).
    Both modes are frozen on first execution and cannot be mixed. The caller
    may inject the qualified lazy Qwen runtime; approval checks precede any
    loading. ``candidates.jsonl`` is the authoritative per-attempt journal;
    generation never publishes training admission or listening decisions.
    """
    root, config = load_project(project)
    if samples_per_phrase is not None:
        if count is not None:
            raise ValueError("Choose samples_per_phrase or a legacy total count, not both")
        if type(samples_per_phrase) is not int or samples_per_phrase < 1:
            raise ValueError("Hard samples_per_phrase must be a positive integer")
    else:
        count = 100 if count is None else count
        if type(count) is not int or not 1 <= count <= MAX_ORIGINALS:
            raise ValueError(f"Hard original count must be an integer from 1 to {MAX_ORIGINALS}")
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; create a separate project for new Hard candidates")
    phrases = _phrases(phrases, config["wakeword"])
    if samples_per_phrase is not None:
        count = len(phrases) * samples_per_phrase
    approval = read_json(root / "approval.json")
    if (approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True or
        approval.get("human_reference_used") is not False or approval.get("wakeword") != config["wakeword"] or
        approval.get("tts_settings_id") != identity(config["tts"]) or approval.get("generation_method") != "clone"):
        raise ValueError("Approve the standalone Positive pilot and synthetic clone policy before Hard generation")
    references = [r for r in read_rows(root / "data" / "references" / "train" / "references.jsonl")
                  if r.get("status") == "accepted" or r.get("voice_family_id") in approval.get("accepted_reference_families", [])]
    if not references or len({r.get("voice_family_id") for r in references}) != len(references):
        raise ValueError("A nonempty distinct approved Train synthetic reference bank is required")
    for reference in references:
        if (reference.get("human_reference_used") is not False or reference.get("provenance") != "qwen_synthetic_reference" or
            reference.get("split") != "train" or not reference.get("voice_family_id") or not reference.get("text")):
            raise ValueError("Hard generation accepts only approved Qwen synthetic Train references")
        bound_file(root, reference)
    output = root / "data" / "generated" / "hard"
    dataset_id = "qwen_" + config["project_id"] + "_hard_v1"
    contract = {"kind": "qwen_hard_negative_generation_contract_v1", "word": config["wakeword"],
        "split": "train", "human_reference_used": False, "method": "clone", "tts": config["tts"],
        "texts": phrases, "references": references, "original_count": count,
        "max_attempts_per_slot": MAX_ATTEMPTS, "text_profile": "standalone_nonwake_phrase",
        "phrase_policy": {"kind": "exclude_complete_call_and_vocative_base_v1",
                          "protected_terms": protected_call_terms(config["wakeword"])},
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "technical_qc_implementation": {"path": str(Path(tts.__file__).resolve()), "sha256": digest(tts.__file__)},
        "positive_policy": {"path": "approval.json", "sha256": digest(root / "approval.json")},
        "acceptance": "technical_candidates_only_content_approval_pending"}
    if samples_per_phrase is not None:
        contract.update(kind="qwen_hard_negative_generation_contract_v2",
                        samples_per_phrase=samples_per_phrase,
                        phrase_count=len(phrases),
                        allocation="every_phrase_cycles_train_references_v1")
    with writer_lock(output):
        contract_path = output / "contract.json"
        if contract_path.exists():
            if read_json(contract_path) != contract:
                raise ValueError("Hard settings/reference/code/count changed; use a versioned output/project")
        else:
            if read_rows(output / "candidates.jsonl"):
                raise ValueError("Hard candidate journal exists without its frozen contract")
            write_json(contract_path, contract)
        contract_binding = {"path": contract_path.relative_to(root).as_posix(), "sha256": digest(contract_path)}
        rows = read_rows(output / "candidates.jsonl")
        passed, attempts, hashes = _validate_existing(root, output, rows, contract, config, phrases, references, count)
        pending = [slot for slot in range(count) if slot not in passed and attempts[slot] < MAX_ATTEMPTS]
        if model is None and pending:
            model = tts._load_model(config["tts"]["clone_model"], config["tts"], root)
        prompts = {}
        for slot in pending:
            phrase, reference = _combination(slot, phrases, references)
            family = reference["voice_family_id"]
            prompt_key = (family, reference["sha256"])
            if prompt_key not in prompts:
                prompts[prompt_key] = model.create_voice_clone_prompt(ref_audio=str(resolved(root, reference["path"])),
                    ref_text=reference["text"], x_vector_only_mode=False)
            for attempt in range(attempts[slot], MAX_ATTEMPTS):
                sample_id = f"hard_train_{slot:06d}_attempt{attempt}"
                seed = _seed(config, slot, attempt, phrase, reference)
                tts._seed(seed)
                started = time.perf_counter()
                wavs, rate = model.generate_voice_clone(text=phrase, language="Korean",
                    voice_clone_prompt=prompts[prompt_key], **config["tts"]["generation"])
                if (len(wavs) != 1 or type(rate) not in (int, float) or not np.isfinite(rate) or
                    not 8000 <= rate <= 192000 or int(rate) != rate):
                    raise ValueError("Qwen must return exactly one waveform and a valid sample rate")
                qc, converted = tts.technical_qc(wavs[0], int(rate))
                if converted is None:
                    raise ValueError("Empty/nonfinite/nonmono Hard waveform cannot be persisted")
                samples, bounds = converted
                path = output / (sample_id + ".wav")
                # Unjournaled files can be left by a prior interruption; the
                # deterministic same slot/attempt regenerates their content.
                sf.write(path, pcm16(np.clip(samples, -1, 1)), 16000, subtype="PCM_16")
                sha = digest(path)
                if sha in hashes:
                    qc["passed"] = False; qc["reasons"].append("exact_duplicate")
                row = {"sample_id": sample_id, "slot": slot, "attempt": attempt,
                    "source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                    "canonical_source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                    "dataset_id": dataset_id, "category": "competing_speech", "split": "train",
                    "category_basis": "Declared standalone non-call Qwen text and synthetic-only generation contract; representative pronunciation review pending",
                    "path": path.relative_to(root).as_posix(), "sha256": sha, "text": phrase,
                    "speech": True, "provenance": "qwen_tts", "human_reference_used": False,
                    "voice_family_id": family, "reference_sha256": reference["sha256"],
                    "start_s": bounds[0] if bounds else None, "end_s": bounds[1] if bounds else None,
                    "phrase_start_s": bounds[0] if bounds else None, "phrase_end_s": bounds[1] if bounds else None,
                    "qc": qc, "technical_pass": qc["passed"], "status": "pending_content_review" if qc["passed"] else "quarantined_technical_failure",
                    "axes": {"voice_family": family, "text_profile": "standalone_nonwake_phrase"},
                    "text_profile": "standalone_nonwake_phrase", "generation_method": "clone",
                    "generation_contract": contract_binding, "generation_contract_id": identity(contract),
                    "tts_settings_id": identity(config["tts"]), "seed": seed,
                    "generation_seconds": time.perf_counter() - started, "created_at": now(),
                    "individual_human_listening": False, "content_review": "pending", "training_admitted": False}
                append_row(output / "candidates.jsonl", row)
                rows.append(row); hashes.add(sha); attempts[slot] += 1
                if qc["passed"]:
                    passed[slot] = row
                    break
            print(f"TTS Hard: technical {len(passed)}/{count}; attempts {len(rows)}; content approval pending", flush=True)
        accepted = sorted(passed.values(), key=lambda r: r["slot"])
        summary = {"kind": "qwen_hard_negative_candidate_summary_v1", "target_originals": count,
            "attempts": len(rows), "technical_pass": len(accepted), "full_target_reached": len(accepted) == count,
            "exhausted_slots": [s for s in range(count) if s not in passed and attempts[s] >= MAX_ATTEMPTS],
            "batch_hard_draws_per_update": config["batch"].get("hard_negative", 0),
            "phrase_distribution": dict(Counter(r["text"] for r in accepted)),
            "voice_distribution": dict(Counter(r["voice_family_id"] for r in accepted)),
            "combination_distribution": dict(Counter(r["text"] + "|" + r["voice_family_id"] for r in accepted)),
            "generation_contract": contract_binding, "generation_contract_id": identity(contract),
            "candidates": str(output / "candidates.jsonl"), "attempt_log": str(output / "candidates.jsonl"),
            "individual_human_listened": 0, "policy_admitted": 0, "independent_STT_runs": 0,
            "actual_training_manifest_published": False, "content_review": "pending"}
        if samples_per_phrase is not None:
            summary.update(generation_mode="per_phrase", samples_per_phrase=samples_per_phrase,
                           phrase_count=len(phrases))
        write_json(output / "summary.json", summary)
        return summary
