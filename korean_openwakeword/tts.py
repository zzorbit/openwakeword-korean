"""Qwen standalone generation with synthetic-only references and bounded QC.

No STT model is imported. Pronunciation approval is a pilot/policy decision;
production technical acceptance never masquerades as individual human listening.
"""
from collections import Counter
import math
import random
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import pcm16, active_speech_mask, mask_intervals
from .common import append_row, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock, snapshot_json, require_final_lock
from .project import load_project
from .storage import project_storage, local_tts_snapshot, storage_environment

REFERENCE_TEXT = "오늘은 창밖으로 따뜻한 햇살이 들어옵니다. 천천히 숨을 고르고 자연스러운 목소리로 한국어 문장을 읽습니다."
STYLE_OPTIONS = (
    ("neutral", "natural relaxed speech"), ("friendly", "warm friendly calling"),
    ("urgent", "clear urgent calling"), ("projected", "strong projected vocal style as if calling to someone far away; no echo or distance simulation"),
)


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
    if not .25 <= duration <= 8:
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
        device = config["device"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        local_path = local_tts_snapshot(model_path, config, project_root)
        return Qwen3TTSModel.from_pretrained(local_path, device_map=device,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation=config["attention"])


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


@project_storage
def generate_references(project, split="train", limit=None, model=None):
    root, config = load_project(project)
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
        reference_contract = {"tts": config["tts"], "families": families, "text": REFERENCE_TEXT, "implementation_sha256": digest(__file__)}
        rows = _generation_journal(output, reference_contract, "references.jsonl",
                                   "Reference generation settings changed; version the bank")
        completed = {r["voice_family_id"]: r for r in rows}
        for row in completed.values():
            if digest(resolved(root, row["path"])) != row["sha256"]:
                raise ValueError("Synthetic reference changed")
        if model is None and any(f["family_id"] not in completed for f in selected):
            model = _load_model(config["tts"]["design_model"], config["tts"], root)
        for family in selected:
            if family["family_id"] in completed:
                continue
            seed = int(identity([config["tts"]["seed"], family["family_id"]])[:8], 16)
            _seed(seed)
            wavs, rate = model.generate_voice_design(text=REFERENCE_TEXT, language="Korean",
                instruct=family["description"] + f" {family.get('style', 'neutral')} delivery, {family.get('speed', 'natural')} rate, {family.get('intonation', 'natural')} intonation, {family.get('vocal_effort', 'normal')} vocal effort. Projected means vocal style, not room echo or microphone distance. Say only the provided Korean text once. No extra speech, laughter or echo.",
                **config["tts"]["generation"])
            samples = np.asarray(wavs[0], dtype=float)
            if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
                raise ValueError("Invalid synthetic reference waveform")
            path = output / (family["family_id"] + ".wav")
            sf.write(path, samples, rate, subtype="FLOAT")
            row = {"sample_id": family["family_id"], "source_id": "qwen:" + family["family_id"],
                "voice_family_id": family["family_id"], "split": split, "path": str(path.relative_to(root)),
                "sha256": digest(path), "text": REFERENCE_TEXT, "provenance": "qwen_synthetic_reference",
                "human_reference_used": False, "status": "pending_review", "model": config["tts"]["design_model"],
                "seed": seed, "description": family["description"], "style": family.get("style", "neutral"), "speed": family.get("speed", "natural"),
                "intonation": family.get("intonation", "natural"), "vocal_effort": family.get("vocal_effort", "normal"), "call_length": "reference_style", "created_at": now(), "axes_scope": "requested reference delivery; confirm actual acoustics in the pilot"}
            append_row(output / "references.jsonl", row)
            completed[family["family_id"]] = row
        return {"generated": len(completed), "review_queue": str(output / "references.jsonl"),
                "purpose": "synthetic reference voices only; not positive training utterances"}


@project_storage
def generate(project, split="train", limit=None, pilot=False, method="clone", model=None):
    root, config = load_project(project)
    if split not in ("train", "validation", "final") or method not in ("clone", "design") or limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Use a valid split/method and a positive generation limit")
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final generation stays locked until model/runtime freeze")
    if split == "final":
        require_final_lock(project)
    approval = read_json(root / "approval.json")
    tts_settings_id = identity(config["tts"])
    if not pilot and (approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True or
                      approval.get("wakeword") != config["wakeword"] or approval.get("tts_settings_id") != tts_settings_id or approval.get("generation_method") != method):
        raise ValueError("Approve the new-word standalone pilot before bulk generation")
    admission = snapshot_json(root, approval, "generation_policy") if not pilot else None
    references = read_rows(root / "data" / "references" / split / "references.jsonl")
    if method == "clone":
        references = [r for r in references if r.get("status") == "accepted" or r["voice_family_id"] in approval.get("accepted_reference_families", [])]
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
    settings = {"word": config["wakeword"], "split": split, "tts": config["tts"], "method": method,
                "references": references, "pilot": pilot, "implementation_sha256": digest(__file__)}
    with writer_lock(output):
        rows = _generation_journal(output, settings, "candidates.jsonl",
                                   "TTS model/reference/settings changed; use a new project or versioned output")
        by_slot = {r["slot"]: r for r in rows if r["qc"]["passed"]}
        attempts_by_slot = Counter(row["slot"] for row in rows)
        for row in rows:
            if digest(resolved(root, row["path"])) != row["sha256"]:
                raise ValueError("Generated candidate changed")
        pending = [i for i in range(target) if i not in by_slot]
        if model is None and pending:
            model = _load_model(config["tts"]["clone_model" if method == "clone" else "design_model"], config["tts"], root)
        prompts = {}
        existing_hashes = {r["sha256"] for r in rows}
        # These immutable settings and retry counts are independent of output.
        generation_contract_id = identity(settings)
        for slot in pending:
            reference = references[slot % len(references)]
            family = reference.get("voice_family_id", reference.get("family_id"))
            for attempt in range(attempts_by_slot[slot], 3):
                sample_id = f"{split}_{slot:06d}_attempt{attempt}"
                seed = int(identity([config["tts"]["seed"], split, slot, attempt])[:8], 16)
                _seed(seed)
                started = time.perf_counter()
                if method == "clone":
                    prompt_key = (family, reference["sha256"])
                    if prompt_key not in prompts:
                        prompts[prompt_key] = model.create_voice_clone_prompt(ref_audio=str(resolved(root, reference["path"])),
                            ref_text=reference["text"], x_vector_only_mode=False)
                    wavs, rate = model.generate_voice_clone(text=config["wakeword"], language="Korean",
                        voice_clone_prompt=prompts[prompt_key], **config["tts"]["generation"])
                    axes = {k: reference.get(k, "reference_style") for k in ("style", "speed", "intonation", "vocal_effort", "call_length")}
                else:
                    style, instruction = STYLE_OPTIONS[slot % len(STYLE_OPTIONS)]
                    speed = ("slow", "natural", "quick")[(slot // len(references)) % 3]
                    wavs, rate = model.generate_voice_design(text=config["wakeword"], language="Korean",
                        instruct=reference["description"] + f" {instruction}. {speed} speaking rate. Say only the provided word once. No other words, breaths, laughs or background.",
                        **config["tts"]["generation"])
                    axes = {"style": style, "speed": speed, "intonation": "natural variation", "vocal_effort": style, "call_length": speed}
                qc, converted = technical_qc(wavs[0], rate)
                if converted is None:
                    raise ValueError("Nonfinite/empty TTS output; cannot persist or silently accept")
                samples, bounds = converted
                path = output / (sample_id + ".wav")
                if np.max(np.abs(samples)) > 1:
                    samples = np.clip(samples, -1, 1)  # quarantined QC-fail only, never training
                sf.write(path, pcm16(samples), 16000, subtype="PCM_16")
                sha = digest(path)
                if sha in existing_hashes:
                    qc["passed"] = False; qc["reasons"].append("exact_duplicate")
                row = {"sample_id": sample_id, "slot": slot, "source_id": "qwen:" + config["project_id"] + ":" + sample_id,
                    "path": str(path.relative_to(root)), "sha256": sha, "split": split,
                    "voice_family_id": family, "text": config["wakeword"], "provenance": "qwen_tts",
                    "human_reference_used": False, "call_start_s": bounds[0] if bounds else None,
                    "call_end_s": bounds[1] if bounds else None, "qc": qc, "axes": {"voice_family": family, **axes, "text_profile": "standalone"},
                    "generation_contract_id": generation_contract_id, "generation_seconds": time.perf_counter() - started,
                    "generation_method": method, "tts_settings_id": tts_settings_id,
                    "individual_human_listening": False, "created_at": now()}
                if not pilot and qc["passed"]:
                    row["admission"] = admission
                append_row(output / "candidates.jsonl", row)
                rows.append(row)
                existing_hashes.add(sha)
                attempts_by_slot[slot] += 1
                if qc["passed"]:
                    by_slot[slot] = row
                    break
            print(f"TTS {split}: accepted technical {len(by_slot)}/{target}; attempts {len(rows)}", flush=True)
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
        write_json(output / "summary.json", summary)
        return summary
