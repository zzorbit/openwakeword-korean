"""Qwen standalone generation with synthetic-only references and bounded QC.

No STT model is imported. Pronunciation approval is a pilot/policy decision;
production technical acceptance never masquerades as individual human listening.
"""
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
import os
import random
import time
import warnings
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import pcm16, active_speech_mask, mask_intervals
from .common import append_row, bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, write_immutable_rows, writer_lock, snapshot_json, require_final_lock
from .task_progress import emit as report_progress, generation_attempt
from . import clone_reference
from .project import load_project
from .storage import project_storage, local_tts_snapshot, storage_environment

# Legacy projects without an explicit sentence keep their original generation
# input. New projects record the shorter default in tts.reference_text.
REFERENCE_TEXT = "오늘은 창밖으로 따뜻한 햇살이 들어옵니다. 천천히 숨을 고르고 자연스러운 목소리로 한국어 문장을 읽습니다."
STYLE_OPTIONS = (
    ("neutral", "natural relaxed speech"), ("friendly", "warm friendly calling"),
    ("urgent", "clear urgent calling"), ("projected", "strong projected vocal style as if calling to someone far away; no echo or distance simulation"),
)

# This is an audited transition from the exact previous generator, not a
# general exception for journals produced by unknown implementations.
LEGACY_CALL_IMPLEMENTATION = "a8a62455fee84260915d489aa2bccb0249ad874c9bfb2496678a5c725292473e"
CALL_RECOVERY_POLICY = "prepared_clone_bounded_recovery_v1"
ORIGINAL_ATTEMPTS = 3
CLONE_ATTEMPTS = 6
_MAX_CALL_DURATION_S = 8
SHORT_CLONE_POLICY = "qwen_short_clone_v1"
_QWEN_CODEC_RATE = 24000
_QWEN_CODEC_FRAME_SAMPLES = 1920
_REFERENCE_PREPARATION_PREVIOUS_IMPLEMENTATIONS = {
    LEGACY_CALL_IMPLEMENTATION, "9f39f255e364f0186f8e866bfb317dbef90d6c01f3481147c04637b3d0585aee"}


def _generation_evidence_sha256():
    from .call_generation_history import SCHEDULING_ONLY_TTS_DIGESTS

    current = _technical_qc_sha256()
    return SCHEDULING_ONLY_TTS_DIGESTS.get(current, current)


def _technical_qc_sha256():
    from .studio import _compute_dispatch_compatible_digest

    return _compute_dispatch_compatible_digest("tts.py", digest(__file__))


def _clone_reference_evidence_sha256():
    from .call_generation_history import CONDITIONING_ONLY_REFERENCE_DIGESTS

    current = digest(clone_reference.__file__)
    return CONDITIONING_ONLY_REFERENCE_DIGESTS.get(current, current)


def technical_qc(samples, rate):
    samples = np.asarray(samples, dtype=np.float64)
    reasons = []
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        return {"passed": False, "reasons": ["empty_nonfinite_or_nonmono"]}, None
    duration = len(samples) / rate
    if np.max(np.abs(samples)) > .999:
        reasons.append("clipping")
    level = float(np.sqrt(np.mean(samples ** 2)))
    if level < .001:
        reasons.append("near_silence")
    if not .25 <= duration <= _MAX_CALL_DURATION_S:
        reasons.append("duration_outlier")
    d = math.gcd(rate, 16000)
    mono = resample_poly(samples, 16000 // d, rate // d) if rate != 16000 else samples
    if np.max(np.abs(mono)) > 1:
        reasons.append("resampling_clipping")
    bounds = None
    if not reasons:
        try:
            mask = active_speech_mask(mono, 0, len(mono) / 16000)
            ranges = mask_intervals(mask)
            bounds = [max(0, ranges[0][0] - .04), min(len(mono) / 16000, ranges[-1][1] + .04)]
            if bounds[1] - bounds[0] > 1.84:
                reasons.append("does_not_fit_full_call_feature_window")
        except ValueError:
            reasons.append("no_active_audio")
    return {"passed": not reasons, "reasons": reasons, "duration_s": duration, "rms": level,
            "boundary_method": "energy_envelope_candidate_no_semantic_STT", "pronunciation_listened": False}, (mono, bounds)


def _load_model(model_path, config, project_root):
    with storage_environment(project_root):
        import torch
        from qwen_tts import Qwen3TTSModel
        local_path = local_tts_snapshot(model_path, config, project_root)
        requested = config["device"]
        backend = os.environ.get("KOWW_STUDIO_TORCH_BACKEND")
        if backend == "cpu":
            requested = "cpu"
        elif backend == "cu128" and not requested.startswith("cuda"):
            requested = "auto"
        automatic = requested == "auto"
        devices = [requested]
        if automatic:
            devices = [] if backend == "cu128" else ["cpu"]
            if torch.cuda.is_available():
                current = torch.cuda.current_device()
                others = [index for index in range(torch.cuda.device_count()) if index != current]
                devices[:0] = [f"cuda:{index}" for index in [current, *others]]
            if not devices:
                raise ValueError("선택한 NVIDIA GPU를 사용할 수 없습니다. CUDA 지원 드라이버와 학습 도구를 확인하세요.")

        def load(device, dtype):
            return Qwen3TTSModel.from_pretrained(local_path, device_map=device, dtype=dtype,
                attn_implementation=config["attention"])

        for device in devices:
            if not device.startswith("cuda"):
                return load(device, torch.float32)
            # The BF16 capability check is scoped to the device receiving the model.
            with torch.cuda.device(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
                try:
                    return load(device, dtype)
                except torch.cuda.OutOfMemoryError:
                    if not automatic:
                        raise
                # The failed loader's traceback can retain tensors until the
                # except block has exited. Release the allocator cache afterwards.
                torch.cuda.empty_cache()
            warnings.warn(f"TTS model did not fit on {device}; trying the next available device.", RuntimeWarning)
        raise ValueError("선택한 GPU의 메모리가 부족해 음성 모델을 불러오지 못했습니다.")


def _short_clone_token_limit(model, configured_max_new_tokens, *, text=None, retry=False):
    """Bound short Qwen clones using the verified codec frame size."""
    if type(model).__module__ != "qwen_tts.inference.qwen3_tts_model":
        return None
    try:
        if version("qwen-tts") != "0.1.1" or version("transformers") != "4.57.3":
            return None
    except PackageNotFoundError:
        return None
    if configured_max_new_tokens is None:
        configured_max_new_tokens = (getattr(model, "generate_defaults", None) or {}).get("max_new_tokens", 2048)
    if type(configured_max_new_tokens) is not int or configured_max_new_tokens < 3:
        return None
    core = model.model
    tokenizer = core.speech_tokenizer
    generation = core.talker.generation_config
    if (type(tokenizer.model).__module__ != "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
            or generation.forced_eos_token_id is not None or generation.num_beams != 1):
        return None
    sample_rate = tokenizer.get_output_sample_rate()
    samples_per_frame = tokenizer.get_decode_upsample_rate()
    if (sample_rate, samples_per_frame) != (_QWEN_CODEC_RATE, _QWEN_CODEC_FRAME_SAMPLES):
        return None
    # Allow for natural cadence and padding around the speech window. A retry
    # adds room for a later natural stop without reverting to the raw budget.
    characters = sum(char.isalnum() for char in text) if isinstance(text, str) else 0
    maximum_frames = _MAX_CALL_DURATION_S * sample_rate // samples_per_frame
    frames = min(maximum_frames, 40 + 2 * characters + (12 if retry else 0))
    return min(configured_max_new_tokens, frames + 2)


def generate_short_clone(model, *, expected_reuses=None, retry=False, **kwargs):
    """Use the same codec budget whether a clone runs through the runtime or directly."""
    generate = getattr(model, "generate_short_clone", None)
    if generate is not None:
        return generate(expected_reuses=expected_reuses, retry=retry, **kwargs)
    configured = kwargs.get("max_new_tokens")
    limit = _short_clone_token_limit(model, configured, text=kwargs.get("text"), retry=retry)
    if limit is not None:
        kwargs["max_new_tokens"] = limit
    wavs, rate = model.generate_voice_clone(**kwargs)
    return wavs, rate, limit


def short_clone_hit_limit(waveform, rate, token_limit):
    """Reject the codec boundary when the waveform cannot prove a natural stop."""
    # Qwen returns decoded audio and omits the stop marker. The prefill does
    # not emit audio, so a full output can contain max_new_tokens - 1 frames.
    # Clone reference slicing can round the output boundary by one sample.
    return (type(token_limit) is int and rate == _QWEN_CODEC_RATE
            and np.size(waveform) >= (token_limit - 1) * _QWEN_CODEC_FRAME_SAMPLES - 1)


def _seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generation_journal(output, contract, filename, changed_message):
    """Load an existing frozen journal without replacing or inventing its contract.

    The caller holds the output writer lock. A journal without its original
    contract cannot prove which settings produced the saved waveforms.
    """
    contract_path = output / "contract.json"
    existing_contract = contract_path.exists()
    if existing_contract and read_json(contract_path) != contract:
        raise ValueError(changed_message)
    rows = read_rows(output / filename)
    if not existing_contract:
        if rows:
            raise ValueError("Generation journal exists without its frozen contract")
        write_json(contract_path, contract)
    return rows


def _reference_instruction(family):
    return (family["description"]
            + f" {family.get('style', 'neutral')} delivery, {family.get('speed', 'natural')} rate,"
            + f" {family.get('intonation', 'natural')} intonation, {family.get('vocal_effort', 'normal')} vocal effort."
            + " Projected means vocal style, not room echo or microphone distance. Say only the provided Korean text once. No extra speech, laughter or echo.")


def _reference_generation_journal(root, output, contract):
    """Keep exact old reference rows/contracts; bind only newly appended rows."""
    path, journal = output / "contract.json", output / "references.jsonl"
    if not path.exists() or read_json(path) == contract:
        return _generation_journal(output, contract, "references.jsonl",
            "Reference generation settings changed; version the bank"), None
    previous = read_json(path)
    core = {key: contract[key] for key in ("tts", "families", "text")}
    if (set(previous) != {*core, "implementation_sha256"}
            or previous.get("implementation_sha256") not in _REFERENCE_PREPARATION_PREVIOUS_IMPLEMENTATIONS
            or any(previous.get(key) != value for key, value in core.items())):
        raise ValueError("Reference generation settings changed; version the bank")
    rows = read_rows(journal)
    bridge_path = output / "reference_generation_update.json"
    source = {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
    if bridge_path.exists():
        bridge = read_json(bridge_path)
        count = bridge.get("preserved_rows")
        if (type(count) is not int or not 0 <= count <= len(rows)
                or bridge.get("kind") != "generated_reference_preparation_update"
                or bridge.get("source_contract") != source or bridge.get("target_contract") != contract
                or bridge.get("preserved_prefix_sha256") != _journal_prefix(journal, count)):
            raise ValueError("Reference generation compatibility evidence changed")
    else:
        if any("reference_preparation" in row for row in rows):
            raise ValueError("New reference rows exist without their preparation compatibility evidence")
        bridge = {"kind": "generated_reference_preparation_update", "source_contract": source,
                  "target_contract": contract, "preserved_rows": len(rows),
                  "preserved_prefix_sha256": _journal_prefix(journal, len(rows))}
        write_json(bridge_path, bridge)
    binding = {"path": bridge_path.relative_to(root).as_posix(), "sha256": digest(bridge_path)}
    if any(row.get("reference_generation_update") != binding for row in rows[bridge["preserved_rows"]:]):
        raise ValueError("New reference rows do not match their preparation compatibility evidence")
    return rows, binding


@project_storage
def generate_references(project, split="train", limit=None, model=None):
    root, config = load_project(project)
    reference_text = config["tts"].get("reference_text", REFERENCE_TEXT)
    if split not in ("train", "validation", "final") or limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Use a valid split and a positive reference limit")
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final reference families stay reserved until the model/runtime is frozen")
    if split == "final":
        require_final_lock(project)
    families = [f for f in read_json(root / "voices.json")["families"] if f["split"] == split]
    selected = families[:limit] if limit is not None else families
    output = root / "data" / "references" / split
    with writer_lock(output):
        reference_contract = {"tts": config["tts"], "families": families, "text": reference_text, "implementation_sha256": _generation_evidence_sha256(),
            "reference_preparation": {"policy": clone_reference.GENERATED_POLICY, "rules": dict(clone_reference.RULES),
                                      "implementation_sha256": _clone_reference_evidence_sha256()}}
        rows, generation_update = _reference_generation_journal(root, output, reference_contract)
        completed = {r["voice_family_id"]: r for r in rows}
        family_by_id = {family["family_id"]: family for family in families}
        for row in completed.values():
            if digest(resolved(root, row["path"])) != row["sha256"]:
                raise ValueError("Synthetic reference changed")
            if "reference_preparation" in row:
                prepared = clone_reference.verify_generated_reference(root, row["reference_preparation"])
                family = family_by_id.get(row["voice_family_id"])
                expected_generation = {"kind": "synthetic_reference", "model": config["tts"]["design_model"],
                    "seed": int(identity([config["tts"]["seed"], row["voice_family_id"]])[:8], 16),
                    "text": reference_text, "instruction": _reference_instruction(family) if family else None,
                    "family_id": row["voice_family_id"], "tts_settings_id": identity(config["tts"])}
                if (not prepared.samefile(resolved(root, row["path"]))
                        or row["reference_preparation"]["prepared"]["sha256"] != row["sha256"]
                        or row["reference_preparation"]["generation"] != expected_generation):
                    raise ValueError("Synthetic reference preparation changed")
        for family in selected:
            if family["family_id"] in completed:
                continue
            seed = int(identity([config["tts"]["seed"], family["family_id"]])[:8], 16)
            path = output / (family["family_id"] + ".wav")
            raw = output / "raw" / (family["family_id"] + ".wav")
            raw.parent.mkdir(exist_ok=True)
            raw_base, recovery = raw, 0
            while raw.exists() and not raw.with_name(raw.name + ".raw.json").exists():
                recovery += 1
                raw = raw_base.with_name(raw_base.stem + f"_recovery{recovery}.wav")
            if not raw.exists():
                if model is None:
                    report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
                    model = _load_model(config["tts"]["design_model"], config["tts"], root)
                _seed(seed)
                with generation_attempt() as timing:
                    wavs, rate = model.generate_voice_design(text=reference_text, language="Korean",
                        instruct=_reference_instruction(family), **config["tts"]["generation"])
                if len(wavs) != 1:
                    raise ValueError("Invalid synthetic reference waveform count")
                sf.write(raw, wavs[0], rate, subtype="FLOAT")
            else:
                timing = None  # Resumed raw audio is not a new speed sample.
            preparation = clone_reference.prepare_generated_reference(root,
                {"path": raw.relative_to(root).as_posix(), "sha256": digest(raw), "generation": {
                    "kind": "synthetic_reference", "model": config["tts"]["design_model"], "seed": seed,
                    "text": reference_text, "instruction": _reference_instruction(family),
                    "family_id": family["family_id"], "tts_settings_id": identity(config["tts"]) }}, path)
            row = {"sample_id": family["family_id"], "source_id": "qwen:" + family["family_id"],
                "voice_family_id": family["family_id"], "split": split, "path": str(path.relative_to(root)),
                "sha256": digest(path), "text": reference_text, "provenance": "qwen_synthetic_reference",
                "human_reference_used": False, "status": "pending_review", "model": config["tts"]["design_model"],
                "seed": seed, "reference_preparation": preparation,
                **({"reference_generation_update": generation_update} if generation_update else {}),
                "description": family["description"], "style": family.get("style", "neutral"), "speed": family.get("speed", "natural"),
                "intonation": family.get("intonation", "natural"), "vocal_effort": family.get("vocal_effort", "normal"), "call_length": "reference_style", "created_at": now(), "axes_scope": "requested reference delivery; confirm actual acoustics in the pilot"}
            append_row(output / "references.jsonl", row)
            if timing is not None:
                timing.committed(True)
            completed[family["family_id"]] = row
        return {"generated": len(completed), "review_queue": str(output / "references.jsonl"),
                "purpose": "synthetic reference voices only; not positive training utterances"}


def call_generation_location(root, split, staged_scope):
    scope_id = identity(staged_scope)[:16]
    return root / "data/call_pronunciation" / scope_id / split, scope_id


def call_generation_contract(config, split, references, pilot, method, staged_scope=None):
    contract = {"word": config["wakeword"], "split": split, "tts": config["tts"], "method": method,
                "references": references, "pilot": pilot, "implementation_sha256": _generation_evidence_sha256()}
    if staged_scope is not None:
        contract["staged_scope"] = staged_scope
    if method == "clone":
        contract["clone_input_policy"] = {"policy": CALL_RECOVERY_POLICY,
            "reference_policy": clone_reference.POLICY,
            "reference_rules": dict(clone_reference.RULES),
            "reference_implementation_sha256": _clone_reference_evidence_sha256(),
            "original_attempts": ORIGINAL_ATTEMPTS, "maximum_attempts": CLONE_ATTEMPTS}
    return contract


def _journal_prefix(path, count):
    """Hash exact old JSONL bytes; never normalize or replace old history."""
    value = hashlib.sha256()
    if count:
        if not path.exists():
            raise ValueError("Saved generation journal prefix is missing")
        with path.open("rb") as journal:
            for _ in range(count):
                line = journal.readline()
                if not line:
                    raise ValueError("Saved generation journal prefix is missing")
                value.update(line)
    return value.hexdigest()


def call_journal_context(root, output, expected, *, create=False):
    """Validate a current journal or the single exact audited legacy bridge.

    Legacy contract.json and its journal prefix remain byte-for-byte intact.
    Only appended attempts use the new contract and its immutable receipt.
    Reading a complete old bank creates no update receipt or derived audio.
    """
    path, journal = output / "contract.json", output / "candidates.jsonl"
    rows = read_rows(journal)
    if not path.exists():
        if rows:
            raise ValueError("Generation journal exists without its frozen contract")
        if create:
            write_json(path, expected)
        return rows, {"current_id": identity(expected), "legacy_count": 0, "update": None}
    actual = read_json(path)
    from .call_generation_history import preserved_context
    preserved = preserved_context(root, output, expected, actual, create=create)
    if preserved is not None:
        return preserved
    if actual == expected:
        if (output / "generation_update.json").exists():
            raise ValueError("Unexpected generation update attached to a current contract")
        return rows, {"current_id": identity(expected), "legacy_count": 0, "update": None}
    legacy = {key: value for key, value in expected.items() if key != "clone_input_policy"}
    legacy["implementation_sha256"] = LEGACY_CALL_IMPLEMENTATION
    if expected.get("method") != "clone" or actual != legacy:
        raise ValueError("TTS model/reference/settings changed; use an audited compatible update or versioned output")
    old_id, new_id = identity(actual), identity(expected)
    update_path = output / "generation_update.json"
    if update_path.exists():
        update = read_json(update_path)
        count = update.get("legacy_row_count")
        if type(count) is not int or count < 0 or count > len(rows):
            raise ValueError("Invalid generation update journal boundary")
        expected_update = {"kind": CALL_RECOVERY_POLICY, "source_contract": {
            "path": str(path.relative_to(root)), "sha256": digest(path)},
            "source_contract_id": old_id, "target_contract": expected,
            "target_contract_id": new_id, "legacy_row_count": count,
            "legacy_journal_prefix_sha256": _journal_prefix(journal, count)}
        if update != expected_update:
            raise ValueError("Generation recovery receipt or its original journal prefix changed")
        binding = {"path": str(update_path.relative_to(root)), "sha256": digest(update_path)}
    else:
        count, binding = len(rows), None
        if any(row.get("generation_contract_id") != old_id for row in rows):
            raise ValueError("Legacy generation rows have no matching update receipt")
        if create:
            update = {"kind": CALL_RECOVERY_POLICY, "source_contract": {
                "path": str(path.relative_to(root)), "sha256": digest(path)},
                "source_contract_id": old_id, "target_contract": expected,
                "target_contract_id": new_id, "legacy_row_count": count,
                "legacy_journal_prefix_sha256": _journal_prefix(journal, count)}
            write_json(update_path, update)
            binding = {"path": str(update_path.relative_to(root)), "sha256": digest(update_path)}
    return rows, {"current_id": new_id, "legacy_id": old_id, "legacy_count": count, "update": binding}


def validate_call_rows(root, config, split, references, rows, context, *, scope=None, method="clone"):
    """Check both historical and newly prepared attempts without inventing QC."""
    tts_id, samples, passing, counts, verified = identity(config["tts"]), set(), {}, Counter(), set()
    scope_id = identity(scope)[:16] if scope is not None else ""
    implementation_sha = _generation_evidence_sha256() if method == "clone" else None
    production = scope is not None and scope.get("kind") == "production"
    old_prompt_attempts = Counter()
    for index, row in enumerate(rows):
        slot = row.get("slot")
        if type(slot) is not int or slot < 0 or not references:
            raise ValueError("Invalid generated call slot")
        reference = references[slot % len(references)]
        family = reference.get("voice_family_id", reference.get("family_id"))
        historical = index < context["legacy_count"]
        expected_id = context.get("legacy_id") if historical else context["current_id"]
        expected_implementation, expected_update = implementation_sha, context["update"]
        for epoch in context.get("epochs", []):
            if index < epoch["stop"]:
                historical, expected_id = epoch["historical"], epoch["current_id"]
                expected_implementation, expected_update = epoch["implementation_sha256"], epoch["update"]
                break
        if (row.get("sample_id") in samples or row.get("voice_family_id") != family
                or row.get("split") != split or row.get("text") != config["wakeword"]
                or row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False
                or row.get("generation_method") != method or row.get("generation_contract_id") != expected_id
                or row.get("tts_settings_id") != tts_id
                or (scope is not None and row.get("voice_reference_sha256") != reference["sha256"])):
            raise ValueError("Generated call identity, reference assignment or generation provenance changed")
        attempt = counts[slot]
        # Legacy journals have deterministic attempt IDs but no explicit seed.
        sample_id = f"{split}_{slot:06d}_attempt{attempt}"
        if scope_id:
            sample_id = scope_id + "_" + sample_id
        if row["sample_id"] != sample_id or slot in passing:
            raise ValueError("Generated call attempt sequence changed")
        if historical and attempt >= ORIGINAL_ATTEMPTS:
            raise ValueError("Legacy generation retry budget changed")
        if method == "clone" and not historical:
            seed_parts = [config["tts"]["seed"], split, slot, attempt] + ([scope_id] if scope_id else [])
            preparation = row.get("clone_reference_preparation")
            policy = preparation.get("policy") if isinstance(preparation, dict) else None
            compacted = policy == clone_reference.INTERNAL_SILENCE_POLICY
            extended = attempt >= CLONE_ATTEMPTS
            if (compacted and not production or extended and (not compacted or attempt >= 2 * CLONE_ATTEMPTS
                    or old_prompt_attempts[slot] != CLONE_ATTEMPTS)
                    or row.get("attempt") != attempt
                    or row.get("seed") != int(identity(seed_parts)[:8], 16)
                    or row.get("clone_input_policy") != CALL_RECOVERY_POLICY
                    or row.get("retry_phase") != ("conditioned_recovery" if extended else
                                                  "recovery" if attempt >= ORIGINAL_ATTEMPTS else "initial")
                    or row.get("individual_human_listening") is not False
                    or row.get("generation_implementation_sha256") != expected_implementation
                    or row.get("generation_update") != expected_update):
                raise ValueError("Prepared generation attempt policy or version changed")
            if policy == clone_reference.POLICY and not extended:
                old_prompt_attempts[slot] += 1
            key = identity([reference, preparation])
            if key not in verified:
                clone_reference.verify_clone_reference(root, reference, preparation)
                verified.add(key)
            if row.get("short_clone_policy") is not None:
                limit = row.get("generation_token_limit")
                if (row["short_clone_policy"] != SHORT_CLONE_POLICY or type(limit) is not int or limit < 3
                        or ("generation_length_limit" in row["qc"]["reasons"] and row["qc"]["passed"] is not False)):
                    raise ValueError("Short-clone generation evidence changed")
        if row.get("generation_error"):
            if (historical or method != "clone" or row["generation_error"] != {"kind": "invalid_waveform", "audio_saved": False}
                    or row.get("path") is not None or row.get("sha256") is not None
                    or row.get("qc") != {"passed": False, "reasons": ["empty_nonfinite_or_nonmono"]}
                    or row.get("call_start_s") is not None or row.get("call_end_s") is not None):
                raise ValueError("Invalid waveform failure record was changed or admitted")
        else:
            if not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
                raise ValueError("Generated call waveform binding is missing")
            bound_file(root, row)
        samples.add(row["sample_id"])
        counts[slot] += 1
        if row.get("qc", {}).get("passed") is True:
            passing[slot] = row
    return passing, counts


def _exhausted_slots(rows, references, target, passing, limit, split, *, recovery_slots=()):
    result, by_slot = [], {}
    for row in rows:
        by_slot.setdefault(row["slot"], []).append(row)
    for slot in range(target):
        attempts = by_slot.get(slot, [])
        slot_limit = 2 * limit if slot in recovery_slots else limit
        if slot in passing or len(attempts) < slot_limit:
            continue
        reference = references[slot % len(references)]
        group = "학습용" if split == "train" else "공통 평가용" if split == "validation" else "최종 평가용"
        gender = {"man": "남성", "woman": "여성"}.get(reference.get("gender"), "")
        age = {"child": "어린이", "teen": "청소년", "teenager": "청소년", "young-adult": "청년", "young adult": "청년",
               "young_adult": "청년", "middle-aged": "중년", "older adult": "노년", "elderly": "노년", "senior": "노년"}.get(reference.get("age_group"), "")
        family = reference.get("voice_family_id", reference.get("family_id"))
        peers = [ref for ref in references if ref.get("gender") == reference.get("gender")
                 and ref.get("age_group") == reference.get("age_group")]
        number = next((index for index, ref in enumerate(peers, 1) if ref == reference), slot + 1)
        label = " · ".join(item for item in (group, gender, age, f"{number}번") if item)
        result.append({"split": split, "slot": slot, "label": label,
            "voice_family_id": family,
            "attempts": len(attempts), "maximum_attempts": slot_limit,
            "reference_sha256": reference.get("sha256"), "automatic_retry_available": False,
            "failures": [{"sample_id": row["sample_id"], "qc": row["qc"]} for row in attempts]})
    return result


@project_storage
def generate(project, split="train", limit=None, pilot=False, method="clone", model=None, *,
             staged_scope=None, attempt_ceiling=None, excluded_sha256=()):
    root, config = load_project(project)
    if split not in ("train", "validation", "final") or method not in ("clone", "design") or limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Use a valid split/method and a positive generation limit")
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final generation stays locked until model/runtime freeze")
    if split == "final":
        require_final_lock(project)
    approval = read_json(root / "approval.json")
    tts_settings_id = identity(config["tts"])
    if not pilot and split in ("train", "validation") and approval.get("call_preflight_reviews"):
        raise ValueError("Use the Studio pronunciation workflow for its reviewed Train and Validation calls")
    if not pilot and (approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True or
                      approval.get("wakeword") != config["wakeword"] or approval.get("tts_settings_id") != tts_settings_id or approval.get("generation_method") != method):
        raise ValueError("Approve the new-word standalone pilot before bulk generation")
    admission = snapshot_json(root, approval, "generation_policy") if not pilot else None
    if staged_scope is not None and (not pilot or method != "clone" or split == "final"):
        raise ValueError("Scoped pronunciation generation is only for staged Train or Validation calls")
    scope_kind = staged_scope.get("kind") if isinstance(staged_scope, dict) else None
    production = scope_kind == "production"
    if attempt_ceiling is not None or excluded_sha256:
        if scope_kind not in ("preflight", "production"):
            raise ValueError("Retry scheduling is only for staged preflight or production calls")
        if attempt_ceiling is not None and (type(attempt_ceiling) is not int or not 1 <= attempt_ceiling <= CLONE_ATTEMPTS):
            raise ValueError("Invalid staged preflight attempt ceiling")
    if not isinstance(excluded_sha256, (tuple, list, set, frozenset)) or any(
            type(sha) is not str or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha)
            for sha in excluded_sha256):
        raise ValueError("Invalid staged preflight audio exclusion")
    references = (staged_scope["references"] if staged_scope is not None
                  else read_rows(root / "data" / "references" / split / "references.jsonl"))
    if method == "clone":
        if staged_scope is None:
            references = [r for r in references if r.get("status") == "accepted"
                          or r["voice_family_id"] in approval.get("accepted_reference_families", [])]
        if not references:
            raise ValueError("No approved synthetic references; run references and review/import their provenance first")
        for row in references:
            if row.get("human_reference_used") is not False or row.get("provenance") != "qwen_synthetic_reference" or row.get("split") != split:
                raise ValueError("Only Qwen-created synthetic references from this split are allowed")
            if digest(resolved(root, row["path"])) != row["sha256"]:
                raise ValueError("Reference audio changed")
    else:
        references = [f for f in read_json(root / "voices.json")["families"] if f["split"] == split]
    target_key = "positive_originals" if split == "train" else "validation_originals" if split == "validation" else "final_originals"
    target = min(config["targets"][target_key], limit) if limit is not None else config["targets"][target_key]
    if pilot and limit is None:
        target = min(12, len(references))
    output = root / "data" / ("pilot" if pilot else "generated") / split
    scope_id = ""
    if staged_scope is not None:
        output, scope_id = call_generation_location(root, split, staged_scope)
    settings = call_generation_contract(config, split, references, pilot, method, staged_scope)
    with writer_lock(output):
        if method == "clone":
            rows, context = call_journal_context(root, output, settings)
            by_slot, attempts_by_slot = validate_call_rows(root, config, split, references, rows, context, scope=staged_scope)
        else:
            rows = _generation_journal(output, settings, "candidates.jsonl",
                                       "TTS model/reference/settings changed; use a new project or versioned output")
            by_slot = {r["slot"]: r for r in rows if r["qc"]["passed"]}
            attempts_by_slot = Counter(row["slot"] for row in rows)
            for row in rows:
                if digest(resolved(root, row["path"])) != row["sha256"]:
                    raise ValueError("Generated candidate changed")
        initial_row_count = len(rows)
        pending = [i for i in range(target) if i not in by_slot]
        attempt_limit = CLONE_ATTEMPTS if method == "clone" else ORIGINAL_ATTEMPTS
        # An intermediate sweep does not change the frozen per-slot budget.
        ceiling = attempt_limit if attempt_ceiling is None else attempt_ceiling
        runnable = [slot for slot in pending if attempts_by_slot[slot] < ceiling]
        prepared = {}
        recovery_slots = set()
        if production and attempt_ceiling is None:
            # Re-open a spent slot only when all six attempts used the earlier,
            # unconditioned prompt. Persisted rows preserve that boundary.
            old_prompt_attempts = Counter(row["slot"] for row in rows
                if row.get("attempt", CLONE_ATTEMPTS) < CLONE_ATTEMPTS
                and (row.get("clone_reference_preparation") or {}).get("policy") == clone_reference.POLICY)
            recovery_slots = {slot for slot in pending if old_prompt_attempts[slot] == CLONE_ATTEMPTS}
        if method == "clone" and (runnable or recovery_slots):
            # Prepare/verify before model loading, and only for unfinished slots.
            report_progress("tts_prompt_preparation", "기준 목소리 복제 입력 확인", eta_eligible=False)
            for slot in [*runnable, *sorted(recovery_slots)]:
                reference = references[slot % len(references)]
                key = (reference["voice_family_id"], reference["sha256"])
                if key not in prepared:
                    if production:
                        prepared[key] = clone_reference.prepare_clone_reference(root, reference, compact_internal_silence=True)
                    else:
                        prepared[key] = clone_reference.prepare_clone_reference(root, reference)
            recovery_slots = {slot for slot in recovery_slots if prepared[
                (references[slot % len(references)]["voice_family_id"], references[slot % len(references)]["sha256"])
            ][1]["policy"] == clone_reference.INTERNAL_SILENCE_POLICY}
            runnable.extend(slot for slot in sorted(recovery_slots) if attempts_by_slot[slot] < 2 * attempt_limit)
            if runnable:
                rows, context = call_journal_context(root, output, settings, create=True)
        if model is None and runnable:
            report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
            model = _load_model(config["tts"]["clone_model" if method == "clone" else "design_model"], config["tts"], root)
        prompts = {}
        existing_hashes = {r["sha256"] for r in rows if r.get("sha256")}
        if staged_scope is not None:
            existing_hashes.update(staged_scope.get("excluded_sha256", ()))
        existing_hashes.update(excluded_sha256)
        # These immutable settings and retry counts are independent of output.
        generation_contract_id = context["current_id"] if method == "clone" else identity(settings)
        remaining = (Counter(slot % len(references) for slot in runnable)
                     if method == "clone" else {})
        for slot in runnable:
            reference_index = slot % len(references)
            reference = references[reference_index]
            family = reference.get("voice_family_id", reference.get("family_id"))
            slot_ceiling = 2 * ceiling if slot in recovery_slots else ceiling
            for attempt in range(attempts_by_slot[slot], slot_ceiling):
                sample_id = f"{split}_{slot:06d}_attempt{attempt}"
                seed_parts = [config["tts"]["seed"], split, slot, attempt]
                if scope_id:
                    sample_id = scope_id + "_" + sample_id
                    seed_parts.append(scope_id)
                seed = int(identity(seed_parts)[:8], 16)
                _seed(seed)
                started = time.perf_counter()
                token_limit = None
                if method == "clone":
                    prompt_key = (family, reference["sha256"])
                    if prompt_key not in prompts:
                        report_progress("tts_prompt_preparation", "기준 목소리 복제 정보 준비", eta_eligible=False)
                        prompt_path, preparation = prepared[prompt_key]
                        prompts[prompt_key] = model.create_voice_clone_prompt(ref_audio=str(prompt_path),
                            ref_text=reference["text"], x_vector_only_mode=False)
                    with generation_attempt() as timing:
                        generation = dict(text=config["wakeword"], language="Korean",
                                          voice_clone_prompt=prompts[prompt_key], **config["tts"]["generation"])
                        wavs, rate, token_limit = generate_short_clone(model, retry=attempt > 0,
                            expected_reuses=remaining[reference_index], **generation)
                    axes = {k: reference.get(k, "reference_style") for k in ("style", "speed", "intonation", "vocal_effort", "call_length")}
                else:
                    style, instruction = STYLE_OPTIONS[slot % len(STYLE_OPTIONS)]
                    speed = ("slow", "natural", "quick")[(slot // len(references)) % 3]
                    with generation_attempt() as timing:
                        wavs, rate = model.generate_voice_design(text=config["wakeword"], language="Korean",
                            instruct=reference["description"] + f" {instruction}. {speed} speaking rate. Say only the provided word once. No other words, breaths, laughs or background.",
                            **config["tts"]["generation"])
                    axes = {"style": style, "speed": speed, "intonation": "natural variation", "vocal_effort": style, "call_length": speed}
                waveform = wavs[0] if wavs is not None and len(wavs) else []
                qc, converted = technical_qc(waveform, rate)
                if converted is not None and short_clone_hit_limit(waveform, rate, token_limit):
                    qc["passed"] = False
                    qc["reasons"].append("generation_length_limit")
                if converted is None and method != "clone":
                    raise ValueError("Nonfinite/empty TTS output; cannot persist or silently accept")
                path, sha, bounds = None, None, None
                if converted is not None:
                    samples, bounds = converted
                    path = output / (sample_id + ".wav")
                    if np.max(np.abs(samples)) > 1:
                        samples = np.clip(samples, -1, 1)  # quarantined QC-fail only, never training
                    if production and path.exists():
                        # An interruption can leave a WAV without its journal row.
                        # Keep that uncommitted output before retrying its exact seed.
                        orphan = output / "interrupted_audio" / f"{sample_id}_{digest(path)}.wav"
                        orphan.parent.mkdir(exist_ok=True)
                        if not orphan.exists():
                            path.rename(orphan)
                        elif digest(path) != digest(orphan):
                            raise ValueError("An interrupted generation WAV differs from its archived copy")
                    sf.write(path, pcm16(samples), 16000, subtype="PCM_16")
                    sha = digest(path)
                    if sha in existing_hashes:
                        qc["passed"] = False; qc["reasons"].append("exact_duplicate")
                row = {"sample_id": sample_id, "slot": slot, "source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                    "path": str(path.relative_to(root)) if path is not None else None, "sha256": sha, "split": split,
                    "voice_family_id": family, "text": config["wakeword"], "provenance": "qwen_tts",
                    "human_reference_used": False, "call_start_s": bounds[0] if bounds else None,
                    "call_end_s": bounds[1] if bounds else None, "qc": qc, "axes": {"voice_family": family, **axes, "text_profile": "standalone"},
                    "generation_contract_id": generation_contract_id, "generation_seconds": time.perf_counter() - started,
                    "generation_method": method, "tts_settings_id": tts_settings_id,
                    "individual_human_listening": False, "created_at": now()}
                if method == "clone":
                    row.update(attempt=attempt, seed=seed, clone_input_policy=CALL_RECOVERY_POLICY,
                        retry_phase=("conditioned_recovery" if attempt >= CLONE_ATTEMPTS else
                                     "recovery" if attempt >= ORIGINAL_ATTEMPTS else "initial"),
                        generation_implementation_sha256=settings["implementation_sha256"],
                        generation_update=context["update"],
                        clone_reference_preparation=prepared[(family, reference["sha256"])][1])
                    if token_limit is not None:
                        row.update(generation_token_limit=token_limit, short_clone_policy=SHORT_CLONE_POLICY)
                    if converted is None:
                        row["generation_error"] = {"kind": "invalid_waveform", "audio_saved": False}
                if staged_scope is not None:
                    row["voice_reference_sha256"] = reference["sha256"]
                if not pilot and qc["passed"]:
                    row["admission"] = admission
                append_row(output / "candidates.jsonl", row)
                if timing is not None:
                    timing.committed(qc["passed"])
                rows.append(row)
                if sha:
                    existing_hashes.add(sha)
                attempts_by_slot[slot] += 1
                if qc["passed"]:
                    by_slot[slot] = row
                    break
            if remaining:
                remaining[reference_index] -= 1
            print(f"TTS {split}: accepted technical {len(by_slot)}/{target}; attempts {len(rows)}", flush=True)
        report_progress("tts_finalizing", "생성 음원 목록·검사 결과 저장", eta_eligible=False)
        passed = sorted(by_slot.values(), key=lambda r: r["slot"])
        # Atomically publish the currently accepted candidates, including partial
        # runs. The summary records whether the selected target was reached.
        if not pilot:
            dest = resolved(root, config["manifests"]["positive_" + split])
            temporary = dest.with_suffix(".tmp")
            temporary.write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in passed), encoding="utf-8")
            temporary.replace(dest)
        summary = {"attempts": len(rows), "technical_pass": len(passed), "target": target,
            "pilot": pilot, "human_listened": 0, "policy_admitted": 0 if pilot else len(passed),
            "independent_STT_runs": 0, "distribution": {axis: dict(Counter(r["axes"][axis] for r in passed)) for axis in ("voice_family", "style", "speed", "intonation", "vocal_effort", "call_length", "text_profile")},
            "review_queue": str(output / "candidates.jsonl"), "full_target_reached": len(passed) == target}
        if any(row.get("generation_error") for row in rows):
            # Failure-only attempts are audit facts, not playable audio. Keep
            # real QC-failed WAVs available to the standalone CLI reviewer.
            reviewable = [row for row in rows if row.get("path") and row.get("sha256")]
            queue = output / ("reviewable_" + identity(reviewable)[:16] + ".jsonl")
            write_immutable_rows(queue, reviewable)
            summary["review_queue"] = str(queue)
            summary["attempt_journal"] = str(output / "candidates.jsonl")
        summary["exhausted_slots"] = _exhausted_slots(rows, references, target, by_slot, attempt_limit, split,
                                                     recovery_slots=recovery_slots)
        if method == "clone":
            summary["clone_input_policy"] = CALL_RECOVERY_POLICY
            summary["maximum_attempts_per_slot"] = attempt_limit
        if method == "clone" and ("legacy_id" in context or "epochs" in context):
            # Historical summaries are evidence too. A recovery publishes a
            # separate current-version summary; a fully reused old bank writes
            # neither a receipt nor any summary.
            if len(rows) > initial_row_count:
                write_json(output / ("summary_recovery_" + context["current_id"][:16] + ".json"), summary)
        else:
            write_json(output / "summary.json", summary)
        return summary
