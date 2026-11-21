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

from . import clone_reference, tts
from .common import append_row, bound_file, digest, identity, now, read_json, read_rows, write_json, writer_lock
from .task_progress import emit as report_progress, generation_attempt
from .data import validate_hard_negative
from .hard_phrase_policy import protected_call_terms, validate_noncall_text
from .project import load_project
from .storage import project_storage
from .tts_runtime import MIXED_VOICE_BATCH_SIZE

MAX_ORIGINALS = 100_000
MAX_ATTEMPTS = 3
_RECOVERY_ATTEMPTS = 2 * MAX_ATTEMPTS
_SHORT_CLONE_UPDATE = "hard_short_clone_update_v1"
_SHORT_CLONE_RECEIPT = "short_clone_update.json"
_LENGTH_FAILURES = {"duration_outlier", "does_not_fit_full_call_feature_window"}
_MIXED_VOICE_BATCH_POLICY = "mixed_voice_cuda4_v1"
_MIXED_VOICE_NARROW_POLICY = "mixed_voice_cuda4_then2_v2"
_MIXED_VOICE_POLICIES = (_MIXED_VOICE_BATCH_POLICY, _MIXED_VOICE_NARROW_POLICY)


def hard_phrase_scope(base, revision):
    if revision is None:
        return base
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{32}", revision):
        raise ValueError("Invalid Hard phrase revision")
    return base / "hard_phrase_revisions" / revision


class HardGenerationPaused(Exception):
    """Stop before another synthesis attempt while preserving the journal."""


def check_generation_pause(pause_requested):
    if pause_requested is not None and pause_requested():
        raise HardGenerationPaused


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


def _batch_slots(contract, first, size=MIXED_VOICE_BATCH_SIZE):
    phrases = len(contract["texts"])
    policy = contract.get("hard_batch_policy")
    if (policy not in _MIXED_VOICE_POLICIES or size not in (2, MIXED_VOICE_BATCH_SIZE)
            or size == 2 and policy != _MIXED_VOICE_NARROW_POLICY
            or len(contract["references"]) < size or first // phrases % size
            or first + (size - 1) * phrases >= contract["original_count"]):
        return None
    return [first + index * phrases for index in range(size)]


def _batch_identity(contract, contract_id, slots):
    seeds = [_seed(contract, slot, 0, *_combination(slot, contract["texts"], contract["references"]))
             for slot in slots]
    return {"policy": contract["hard_batch_policy"], "slots": slots,
            "seed": int(identity([contract_id, seeds, slots])[:8], 16)}


def _validate_batch_attempt(row, contract, contract_id):
    batch = row.get("cuda_batch")
    slots = batch.get("slots") if isinstance(batch, dict) else None
    if (row.get("attempt") != 0 or not isinstance(slots, list) or len(slots) not in (2, MIXED_VOICE_BATCH_SIZE)
            or any(type(slot) is not int or slot < 0 for slot in slots)
            or row.get("slot") not in slots or slots != _batch_slots(contract, slots[0], len(slots))
            or batch != _batch_identity(contract, contract_id, slots)):
        raise ValueError("Hard CUDA batch identity changed")
    ending = row.get("batch_termination")
    if not isinstance(ending, dict):
        raise ValueError("Hard CUDA batch ending evidence is missing")
    stop, tokens, limit = ending.get("termination"), ending.get("generated_tokens"), ending.get("token_limit")
    if (type(limit) is not int or limit != row.get("generation_token_limit")
            or type(tokens) is not int or not 1 <= tokens <= limit
            or stop not in ("eos", "token_limit", "without_eos")
            or (stop == "token_limit") != (tokens == limit and stop != "eos")
            or stop != "eos" and row.get("technical_pass") is not False):
        raise ValueError("Hard CUDA batch ending evidence changed")


def _recover_hard_batch(root, output, contract, contract_id, binding):
    pending = output / "pending_cuda_batch.json"
    if not pending.exists():
        return
    intent = read_json(pending)
    rows = intent.get("rows") if isinstance(intent, dict) else None
    if (not isinstance(rows, list) or len(rows) not in (2, MIXED_VOICE_BATCH_SIZE) or type(intent.get("offset")) is not int
            or intent.get("contract_id") != contract_id or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("Pending Hard CUDA batch does not belong to this generation contract")
    for row in rows:
        _validate_batch_attempt(row, contract, contract_id)
        slot = row["slot"]
        phrase, reference = _combination(slot, contract["texts"], contract["references"])
        sample_id = f"hard_train_{slot:06d}_attempt0"
        if (row.get("generation_contract_id") != contract_id or row.get("generation_contract") != binding
                or row.get("sample_id") != sample_id or row.get("text") != phrase
                or row.get("voice_family_id") != reference["voice_family_id"]
                or row.get("reference_sha256") != reference["sha256"]
                or row.get("seed") != _seed(contract, slot, 0, phrase, reference)
                or type(row.get("technical_pass")) is not bool or not isinstance(row.get("qc"), dict)
                or row["technical_pass"] is not row["qc"].get("passed")):
            raise ValueError("Pending Hard CUDA batch contains a changed candidate")
        _validate_short_clone_attempt(row, historical=False, update=None)
        _candidate_path(root, output, row, sample_id)
    batch = rows[0]["cuda_batch"]
    if [row["slot"] for row in rows] != batch["slots"] or any(row["cuda_batch"] != batch for row in rows):
        raise ValueError("Pending Hard CUDA batch order changed")
    tts._finish_call_batch(output, intent)


def _legacy_recovery_slots(rows):
    counts, passed, length_failure = Counter(), set(), set()
    for row in rows:
        slot, qc = row["slot"], row["qc"]
        counts[slot] += 1
        if qc["passed"]:
            passed.add(slot)
        elif _LENGTH_FAILURES.intersection(qc.get("reasons", ())):
            length_failure.add(slot)
    return sorted(slot for slot in length_failure - passed if counts[slot] == MAX_ATTEMPTS)


def _short_clone_update_record(contract_binding, expected, journal, boundary, recovery_slots):
    pilot = "implementation_sha256" in expected
    return {"kind": _SHORT_CLONE_UPDATE, "generation_contract": contract_binding,
            "short_clone_policy": tts.SHORT_CLONE_POLICY,
            "implementation_sha256": (expected["implementation_sha256"] if pilot else expected["implementation"]["sha256"]),
            "technical_qc_sha256": (expected["technical_qc_sha256"] if pilot else expected["technical_qc_implementation"]["sha256"]),
            "previous_row_count": boundary, "previous_journal_prefix_sha256": tts._journal_prefix(journal, boundary),
            "recovery_slots": recovery_slots, "max_recovery_attempts_per_slot": _RECOVERY_ATTEMPTS}


def _short_clone_context(root, output, contract, expected, rows, *, create=False):
    """Bind new synthesis to the exact old journal prefix without replacing it."""
    path = output / _SHORT_CLONE_RECEIPT
    if "short_clone_policy" in contract:
        if path.exists():
            raise ValueError("Unexpected short-clone update for a current Hard contract")
        return None, 0, frozenset()
    journal = output / "candidates.jsonl"
    contract_path = output / "contract.json"
    contract_binding = {"path": contract_path.relative_to(root).as_posix(), "sha256": digest(contract_path)}
    saved = read_json(path) if path.exists() else None
    if saved is not None and not isinstance(saved, dict):
        raise ValueError("Invalid Hard synthesis update receipt")
    boundary = saved.get("previous_row_count") if saved is not None else len(rows)
    if type(boundary) is not int or not 0 <= boundary <= len(rows):
        raise ValueError("Invalid Hard synthesis update journal boundary")
    recovery_slots = _legacy_recovery_slots(rows[:boundary])
    if saved is not None or create:
        receipt = _short_clone_update_record(contract_binding, expected, journal, boundary, recovery_slots)
        if saved is None:
            write_json(path, receipt)
        elif saved != receipt:
            from .call_generation_history import compatible_short_clone_hard_release

            implementation, qc = saved.get("implementation_sha256"), saved.get("technical_qc_sha256")
            if (not compatible_short_clone_hard_release(implementation, qc, pilot="implementation_sha256" in expected)
                    or saved != {**receipt, "implementation_sha256": implementation, "technical_qc_sha256": qc}):
                raise ValueError("Hard synthesis update or original journal prefix changed")
        binding = {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
    else:
        binding = None
    return binding, boundary, frozenset(recovery_slots)


def _current_hard_contract(contract, *, pilot):
    if pilot:
        from . import hard_pilot
        return {**contract, "implementation_sha256": hard_pilot._implementation_sha256(),
                "technical_qc_sha256": tts._technical_qc_sha256()}
    return {**contract,
            "implementation": {"path": str(Path(__file__).resolve()), "sha256": _implementation_sha256()},
            "technical_qc_implementation": {"path": str(Path(tts.__file__).resolve()), "sha256": tts._technical_qc_sha256()}}


def _implementation_sha256():
    from .studio import _compute_dispatch_compatible_digest

    return _compute_dispatch_compatible_digest("hard_tts.py", digest(__file__))


def hard_generation_recovery_available(output, *, pilot=False):
    """Report whether a previously exhausted generation has unfinished recovery slots."""
    from .call_generation_history import compatible_hard_contract

    contract_path = output / "contract.json"
    if not contract_path.exists():
        return False
    contract = read_json(contract_path)
    if "short_clone_policy" in contract:
        return False
    current = _current_hard_contract(contract, pilot=pilot)
    if not compatible_hard_contract(contract, current, pilot=pilot):
        return False
    rows = read_rows(output / "candidates.jsonl")
    generated = output.parents[2] if output.parents[1].name == "hard_phrase_revisions" else output.parent
    _, _, recovered = _short_clone_context(generated.parents[1], output, contract, current, rows)
    counts = Counter(row["slot"] for row in rows)
    passed = {row["slot"] for row in rows if row.get("technical_pass") is True}
    return any(slot not in passed and counts[slot] < _RECOVERY_ATTEMPTS for slot in recovered)


def short_clone_admission_context(root, output, contract, *, pilot=False):
    """Verify the immutable update once and index its exact appended attempts."""
    if "hard_batch_policy" in contract and (pilot or contract["hard_batch_policy"] not in _MIXED_VOICE_POLICIES):
        raise ValueError("Invalid Hard CUDA batch policy")
    if "short_clone_policy" in contract:
        if contract["short_clone_policy"] != tts.SHORT_CLONE_POLICY or (output / _SHORT_CLONE_RECEIPT).exists():
            raise ValueError("Invalid Hard short-clone generation policy")
        return None, {}, frozenset(), set()
    if not (output / _SHORT_CLONE_RECEIPT).exists():
        return None, {}, frozenset(), set()
    from .call_generation_history import compatible_hard_contract

    current = _current_hard_contract(contract, pilot=pilot)
    if not compatible_hard_contract(contract, current, pilot=pilot):
        raise ValueError("Hard synthesis update has no audited original generation contract")
    rows = read_rows(output / "candidates.jsonl")
    update, boundary, recovery = _short_clone_context(root, output, contract, current, rows)
    return update, {(row["slot"], row["attempt"]): row for row in rows[boundary:]}, recovery, set()


def verify_short_clone_admission(root, row, reference, contract, context):
    update, appended, recovery, verified = context
    if "cuda_batch" in row or "batch_termination" in row:
        _validate_batch_attempt(row, contract, identity(contract))
    current_contract = contract.get("short_clone_policy") == tts.SHORT_CLONE_POLICY
    if current_contract or "short_clone_policy" in row:
        _validate_short_clone_attempt(row, historical=False, update=update)
        if not current_contract and (update is None or appended.get((row.get("slot"), row.get("attempt"))) != row):
            raise ValueError("Hard candidate is not in its bound synthesis update journal")
        _verify_prepared_reference(root, row, reference, verified)
    else:
        _validate_short_clone_attempt(row, historical=True, update=None)
    if update is not None and row.get("short_clone_update") == update and row.get("slot") in recovery:
        return _RECOVERY_ATTEMPTS
    return MAX_ATTEMPTS


def _attempt_rounds(count, passed, attempts, recovery):
    """Try all unfinished slots at each attempt level before retrying any slot."""
    for attempt in range(_RECOVERY_ATTEMPTS if recovery else MAX_ATTEMPTS):
        slots = range(count) if attempt < MAX_ATTEMPTS else sorted(recovery)
        yield from ((slot, attempt) for slot in slots if slot not in passed and attempts[slot] == attempt)


def _exhausted_details(slots, rows, phrase_reference, recovery):
    failures = {slot: [] for slot in slots}
    for row in rows:
        if row["slot"] in failures:
            failures[row["slot"]].append({"sample_id": row["sample_id"], "qc": row["qc"]})
    details = []
    for slot in slots:
        phrase, reference = phrase_reference(slot)
        details.append({"slot": slot, "label": f"{phrase} · {slot + 1}번", "text": phrase,
                        "voice_family_id": reference["voice_family_id"], "attempts": len(failures[slot]),
                        "maximum_attempts": _RECOVERY_ATTEMPTS if slot in recovery else MAX_ATTEMPTS,
                        "failures": failures[slot]})
    return details


def _validate_short_clone_attempt(row, *, historical, update):
    if historical:
        if any(key in row for key in ("short_clone_policy", "short_clone_update", "generation_token_limit",
                                      "clone_reference_preparation", "candidate_pcm_qc")):
            raise ValueError("Unexpected short-clone evidence in original Hard journal")
    else:
        limit = row.get("generation_token_limit")
        if (row.get("short_clone_policy") != tts.SHORT_CLONE_POLICY or row.get("short_clone_update") != update
                or not isinstance(row.get("clone_reference_preparation"), dict)
                or (limit is not None and (type(limit) is not int or limit < 1))):
            raise ValueError("Hard attempt does not bind its short-clone policy and update")
        if "candidate_pcm_qc" in row and (row["candidate_pcm_qc"] != tts.CANDIDATE_PCM_QC or not row.get("path")):
            raise ValueError("Hard attempt has invalid saved PCM quality evidence")


def _verify_prepared_reference(root, row, reference, verified):
    preparation = row["clone_reference_preparation"]
    key = reference["sha256"], identity(preparation)
    if key not in verified:
        clone_reference.verify_clone_reference(root, reference, preparation)
        verified.add(key)


def _candidate_path(root, output, row, sample_id):
    if (row.get("path") is None and row.get("sha256") is None and row.get("technical_pass") is False
            and row.get("qc", {}).get("reasons") == ["empty_nonfinite_or_nonmono"]):
        return None
    path = bound_file(root, row)
    if path != output / (sample_id + ".wav"):
        raise ValueError("Hard candidate path escaped its generation directory")
    return path


def _synthesise(model, phrase, prompt, settings, *, retry, expected_reuses):
    termination = {}
    wavs, rate, token_limit = tts.generate_short_clone(model, text=phrase, language="Korean",
        voice_clone_prompt=prompt, retry=retry, expected_reuses=expected_reuses, termination=termination, **settings)
    if ((wavs is not None and len(wavs) > 1) or type(rate) not in (int, float) or not np.isfinite(rate) or
            not 8000 <= rate <= 192000 or int(rate) != rate):
        raise ValueError("Qwen must return exactly one waveform and a valid sample rate")
    waveform = wavs[0] if wavs is not None and len(wavs) else []
    _, qc, converted = tts.check_short_clone_audio(waveform, int(rate), token_limit, termination)
    return qc, converted, token_limit


def _mixed_voice_waveforms(result, size=MIXED_VOICE_BATCH_SIZE):
    wavs, rate, limit, endings = result
    if (len(wavs) != size or len(endings) != size
            or type(rate) not in (int, float) or not np.isfinite(rate) or not 8000 <= rate <= 192000 or int(rate) != rate
            or type(limit) is not int or limit < 1
            or any(not isinstance(ending, dict) or ending.get("token_limit") != limit for ending in endings)):
        raise ValueError("Qwen Hard CUDA batch must return one bounded waveform per voice")
    return wavs, int(rate), limit, endings


def _store_candidate(root, output, sample_id, converted, qc, hashes):
    if converted is None:
        return None, None, None
    samples, bounds = tts.encode_candidate_pcm16(converted, qc)
    path = output / (sample_id + ".wav")
    if path.exists():
        # Only an attempt without a journal row may reach this path.
        orphan = output / "interrupted_audio" / f"{sample_id}_{digest(path)}.wav"
        orphan.parent.mkdir(exist_ok=True)
        if not orphan.exists():
            path.rename(orphan)
    sf.write(path, samples, 16000, subtype="PCM_16")
    sha = digest(path)
    if sha in hashes:
        qc["passed"] = False
        qc["reasons"].append("exact_duplicate")
    hashes.add(sha)
    return path.relative_to(root).as_posix(), sha, bounds


def _validate_saved_candidate(path, row):
    """Keep original rows and require the stored samples to pass before resuming."""
    with sf.SoundFile(path) as audio:
        start, end = row.get("start_s"), row.get("end_s")
        if (audio.samplerate != 16000 or audio.channels != 1 or audio.subtype != "PCM_16"
                or type(start) not in (int, float) or type(end) not in (int, float)
                or not 0 <= start < end <= audio.frames / audio.samplerate + 1e-6 or end - start > 1.840001):
            raise ValueError("Hard candidate has invalid saved PCM or an incomplete phrase interval")
        persisted_qc, converted = tts.technical_qc(audio.read(dtype="float64"), audio.samplerate)
    if not persisted_qc["passed"]:
        raise ValueError("Stored Hard candidate no longer passes technical QC")
    if row.get("candidate_pcm_qc") == tts.CANDIDATE_PCM_QC and converted[1] != [start, end]:
        raise ValueError("Hard candidate saved PCM phrase interval changed")


def _validate_existing(root, output, rows, contract, config, phrases, references, count, pause_requested, update, boundary, recovery):
    passed, hashes, prepared = {}, set(), set()
    batches = {}
    contract_id, settings_id = identity(contract), identity(config["tts"])
    contract_binding = {"path": (output / "contract.json").relative_to(root).as_posix(),
                        "sha256": digest(output / "contract.json")}
    attempts = Counter()
    for number, row in enumerate(rows):
        check_generation_pause(pause_requested)
        slot, attempt = row.get("slot"), row.get("attempt")
        historical = number < boundary
        if type(slot) is not int or not 0 <= slot < count or type(attempt) is not int:
            raise ValueError("Invalid Hard candidate slot/attempt")
        ceiling = _RECOVERY_ATTEMPTS if not historical and slot in recovery else MAX_ATTEMPTS
        if not 0 <= attempt < ceiling:
            raise ValueError("Invalid Hard candidate slot/attempt")
        phrase, reference = _combination(slot, phrases, references)
        sample_id = f"hard_train_{slot:06d}_attempt{attempt}"
        if slot in passed or attempt != attempts[slot]:
            raise ValueError("Duplicate or noncontiguous Hard attempt journal")
        attempts[slot] += 1
        path = _candidate_path(root, output, row, sample_id)
        _validate_short_clone_attempt(row, historical=historical, update=update)
        if "cuda_batch" in row or "batch_termination" in row:
            _validate_batch_attempt(row, contract, contract_id)
            batches.setdefault(tuple(row["cuda_batch"]["slots"]), []).append(slot)
        if not historical:
            _verify_prepared_reference(root, row, reference, prepared)
        if (row.get("sample_id") != sample_id or row.get("text") != phrase or
            row.get("voice_family_id") != reference["voice_family_id"] or
            row.get("reference_sha256") != reference["sha256"] or row.get("split") != "train" or
            row.get("generation_contract") != contract_binding or
            row.get("generation_contract_id") != contract_id or
            row.get("tts_settings_id") != settings_id or
            row.get("seed") != _seed(config, slot, attempt, phrase, reference) or
            row.get("individual_human_listening") is not False or row.get("admission") is not None):
            raise ValueError("Hard candidate contract/provenance changed")
        validate_hard_negative(row, config)
        qc = row.get("qc", {})
        if row.get("technical_pass") is not qc.get("passed"):
            raise ValueError("Hard technical acceptance fields disagree")
        if qc.get("passed") is True:
            if row["sha256"] in hashes:
                raise ValueError("Hard attempt passed duplicate audio")
            _validate_saved_candidate(path, row)
            passed[slot] = row
        if path is not None:
            hashes.add(row["sha256"])
    if any(list(slots) != published for slots, published in batches.items()):
        raise ValueError("Hard CUDA batch journal contains an incomplete or reordered group")
    return passed, attempts, hashes


@project_storage
def generate_hard(project, phrases, count=None, model=None, *, samples_per_phrase=None,
                  revision=None, pause_requested=None):
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
    New attempts record quality validation of stored PCM. Earlier accepted
    attempts keep their records and must still pass QC from the saved audio.
    A new per-phrase scope can group four different voices for its first
    CUDA attempt. Each failure retains its existing individual retry seed.
    """
    check_generation_pause(pause_requested)
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
    output = hard_phrase_scope(root / "data/generated", revision) / "hard"
    dataset_id = "qwen_" + config["project_id"] + "_hard_v1"
    contract = {"kind": "qwen_hard_negative_generation_contract_v1", "word": config["wakeword"],
        "split": "train", "human_reference_used": False, "method": "clone", "tts": config["tts"],
        "texts": phrases, "references": references, "original_count": count,
        "max_attempts_per_slot": MAX_ATTEMPTS, "text_profile": "standalone_nonwake_phrase",
        "short_clone_policy": tts.SHORT_CLONE_POLICY,
        "phrase_policy": {"kind": "exclude_complete_call_and_vocative_base_v1",
                          "protected_terms": protected_call_terms(config["wakeword"])},
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": _implementation_sha256()},
        "technical_qc_implementation": {"path": str(Path(tts.__file__).resolve()), "sha256": tts._technical_qc_sha256()},
        "positive_policy": {"path": "approval.json", "sha256": digest(root / "approval.json")},
        "acceptance": "technical_candidates_only_content_approval_pending"}
    if samples_per_phrase is not None:
        contract.update(kind="qwen_hard_negative_generation_contract_v2",
                        samples_per_phrase=samples_per_phrase,
                        phrase_count=len(phrases),
                        allocation="every_phrase_cycles_train_references_v1")
        if min(samples_per_phrase, len(references)) >= MIXED_VOICE_BATCH_SIZE:
            contract["hard_batch_policy"] = _MIXED_VOICE_NARROW_POLICY
    with writer_lock(output):
        expected = contract
        contract_path = output / "contract.json"
        if contract_path.exists():
            from .call_generation_history import compatible_hard_contract
            saved = read_json(contract_path)
            # An earlier frozen scope must keep its original scalar schedule.
            if isinstance(saved, dict) and "hard_batch_policy" not in saved:
                expected = {key: value for key, value in expected.items() if key != "hard_batch_policy"}
            elif (expected.get("hard_batch_policy") == _MIXED_VOICE_NARROW_POLICY
                    and isinstance(saved, dict) and saved.get("hard_batch_policy") == _MIXED_VOICE_BATCH_POLICY):
                expected = {**expected, "hard_batch_policy": _MIXED_VOICE_BATCH_POLICY}
            if not compatible_hard_contract(saved, expected):
                raise ValueError("Hard settings/reference/code/count changed; use a versioned output/project")
            contract = saved
        else:
            if read_rows(output / "candidates.jsonl"):
                raise ValueError("Hard candidate journal exists without its frozen contract")
            write_json(contract_path, contract)
        contract_binding = {"path": contract_path.relative_to(root).as_posix(), "sha256": digest(contract_path)}
        contract_id, settings_id = identity(contract), identity(config["tts"])
        _recover_hard_batch(root, output, contract, contract_id, contract_binding)
        journal = output / "candidates.jsonl"
        rows = read_rows(journal)
        update, boundary, recovery = _short_clone_context(root, output, contract, expected, rows)
        passed, attempts, hashes = _validate_existing(root, output, rows, contract, config, phrases, references,
                                                      count, pause_requested, update, boundary, recovery)
        pending = [slot for slot in range(count) if slot not in passed
                   and attempts[slot] < (_RECOVERY_ATTEMPTS if slot in recovery else MAX_ATTEMPTS)]
        if pending and "short_clone_policy" not in contract and update is None:
            update, _, _ = _short_clone_context(root, output, contract, expected, rows, create=True)
        if model is None and pending:
            report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
            model = tts._load_model(config["tts"]["clone_model"], config["tts"], root)
        prompts = {}
        remaining = Counter((phrase, reference["voice_family_id"])
                            for phrase, reference in (_combination(slot, phrases, references) for slot in pending))
        generate_batch = getattr(model, "generate_short_clone_mixed_voice_batch", None)
        batch_available = getattr(model, "mixed_voice_batch_available", None)
        pair_available = getattr(model, "mixed_voice_pair_available", None)
        synthesis_settings = config["tts"]["generation"]

        def prepare_prompt(reference):
            prompt_key = reference["voice_family_id"], reference["sha256"]
            if prompt_key not in prompts:
                check_generation_pause(pause_requested)
                report_progress("tts_prompt_preparation", "기준 목소리 복제 정보 준비", eta_eligible=False)
                prompt_path, preparation = clone_reference.prepare_clone_reference(root, reference, compact_internal_silence=True)
                prompt = model.create_voice_clone_prompt(ref_audio=str(prompt_path), ref_text=reference["text"],
                                                        x_vector_only_mode=False)
                prompts[prompt_key] = prompt, preparation
            return prompts[prompt_key]

        def record_attempt(slot, attempt, phrase, reference, preparation, qc, converted, token_limit, started):
            family = reference["voice_family_id"]
            sample_id = f"hard_train_{slot:06d}_attempt{attempt}"
            path, sha, bounds = _store_candidate(root, output, sample_id, converted, qc, hashes)
            row = {"sample_id": sample_id, "slot": slot, "attempt": attempt,
                    "source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                    "canonical_source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                    "dataset_id": dataset_id, "category": "competing_speech", "split": "train",
                    "category_basis": "Declared standalone non-call Qwen text and synthetic-only generation contract; representative pronunciation review pending",
                    "path": path, "sha256": sha, "text": phrase,
                    "speech": True, "provenance": "qwen_tts", "human_reference_used": False,
                    "voice_family_id": family, "reference_sha256": reference["sha256"],
                    "start_s": bounds[0] if bounds else None, "end_s": bounds[1] if bounds else None,
                    "phrase_start_s": bounds[0] if bounds else None, "phrase_end_s": bounds[1] if bounds else None,
                    "qc": qc, "technical_pass": qc["passed"], "status": "pending_content_review" if qc["passed"] else "quarantined_technical_failure",
                    "axes": {"voice_family": family, "text_profile": "standalone_nonwake_phrase"},
                    "text_profile": "standalone_nonwake_phrase", "generation_method": "clone",
                    "generation_contract": contract_binding, "generation_contract_id": contract_id,
                    "short_clone_policy": tts.SHORT_CLONE_POLICY, "short_clone_update": update,
                    "generation_token_limit": token_limit, "clone_reference_preparation": preparation,
                    "tts_settings_id": settings_id, "seed": _seed(config, slot, attempt, phrase, reference),
                    "generation_seconds": time.perf_counter() - started, "created_at": now(),
                    "individual_human_listening": False, "content_review": "pending", "training_admitted": False}
            if path is not None:
                row["candidate_pcm_qc"] = tts.CANDIDATE_PCM_QC
            return row

        def track(row):
            slot = row["slot"]
            rows.append(row)
            attempts[slot] += 1
            if row["technical_pass"]:
                passed[slot] = row
            if row["technical_pass"] or attempts[slot] == (_RECOVERY_ATTEMPTS if slot in recovery else MAX_ATTEMPTS):
                remaining[row["text"], row["voice_family_id"]] -= 1

        def progress():
            print(f"TTS Hard: technical {len(passed)}/{count}; attempts {len(rows)}; content approval pending", flush=True)

        def run_batch(slots, phrase):
            voices = [_combination(slot, phrases, references)[1] for slot in slots]
            prepared = [prepare_prompt(reference) for reference in voices]
            check_generation_pause(pause_requested)
            batch = _batch_identity(contract, contract_id, slots)
            started = time.perf_counter()
            with generation_attempt() as timing:
                result = generate_batch(batch_seed=batch["seed"],
                    voice_clone_prompts=[prompt for prompt, _ in prepared], text=phrase, language="Korean",
                    **synthesis_settings)
            if result is None:
                return False
            wavs, rate, token_limit, endings = _mixed_voice_waveforms(result, len(slots))
            synthesis_share = (time.perf_counter() - started) / len(slots)
            batch_rows = []
            candidates = zip(slots, voices, prepared, wavs, endings, strict=True)
            for slot, reference, (_, preparation), waveform, ending in candidates:
                started = time.perf_counter() - synthesis_share
                _, qc, converted = tts.check_short_clone_audio(waveform, rate, token_limit, ending)
                row = record_attempt(slot, 0, phrase, reference, preparation, qc, converted, token_limit, started)
                row.update(cuda_batch=batch, batch_termination=ending)
                _validate_batch_attempt(row, contract, contract_id)
                batch_rows.append(row)
            intent = {"contract_id": contract_id, "offset": journal.stat().st_size if journal.exists() else 0,
                      "rows": batch_rows}
            write_json(output / "pending_cuda_batch.json", intent)
            tts._finish_call_batch(output, intent)
            for row in batch_rows:
                track(row)
            if timing is not None:
                timing.committed(sum(row["technical_pass"] for row in batch_rows))
            progress()
            return True

        for slot, attempt in _attempt_rounds(count, passed, attempts, recovery):
            check_generation_pause(pause_requested)
            phrase, reference = _combination(slot, phrases, references)
            slots = _batch_slots(contract, slot) if attempt == 0 and callable(generate_batch) else None
            if (slots and not any(attempts[current] for current in slots)
                    and (not callable(batch_available) or batch_available()) and run_batch(slots, phrase)):
                continue
            pair = _batch_slots(contract, slot, 2) if attempt == 0 and callable(pair_available) else None
            if pair and not any(attempts[current] for current in pair) and pair_available() and run_batch(pair, phrase):
                continue
            prompt, preparation = prepare_prompt(reference)
            check_generation_pause(pause_requested)
            tts._seed(_seed(config, slot, attempt, phrase, reference))
            started = time.perf_counter()
            with generation_attempt() as timing:
                qc, converted, token_limit = _synthesise(model, phrase, prompt, synthesis_settings,
                    retry=attempt > 0, expected_reuses=remaining[phrase, reference["voice_family_id"]])
            row = record_attempt(slot, attempt, phrase, reference, preparation, qc, converted, token_limit, started)
            append_row(journal, row)
            if timing is not None:
                timing.committed(qc["passed"])
            track(row)
            progress()
        check_generation_pause(pause_requested)
        report_progress("tts_finalizing", "유사어 음원 목록·검사 결과 저장", eta_eligible=False)
        accepted = sorted(passed.values(), key=lambda r: r["slot"])
        exhausted = [slot for slot in range(count) if slot not in passed]
        summary = {"kind": "qwen_hard_negative_candidate_summary_v1", "target_originals": count,
            "attempts": len(rows), "technical_pass": len(accepted), "full_target_reached": len(accepted) == count,
            "exhausted_slots": exhausted,
            "exhausted_slot_details": _exhausted_details(exhausted, rows,
                lambda slot: _combination(slot, phrases, references), recovery),
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
