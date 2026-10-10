"""Controlled evaluation audio: same word, same noise section, changing SNR."""
from pathlib import Path
import json
import math
import numpy as np
import soundfile as sf

from .audio import active_speech_mask, pcm16, noise_scale, measured_snr, rms, common_headroom_gain
from .common import digest, identity, read_rows, resolved, require_final_lock
from .data import audit_project, rows_for, seek_audio, source_key, CATEGORIES
from .project import load_project
from .storage import project_storage


@project_storage
def prepare_evaluation(project, output, split="validation", originals=100, noise_sections=5, snrs=(20, 15, 10, 5, 0)):
    root, config = load_project(project)
    if type(originals) is not int or originals < 1 or type(noise_sections) is not int or noise_sections < 1 or not snrs or any(not math.isfinite(s) for s in snrs):
        raise ValueError("Declare positive counts and finite SNR conditions")
    if split not in ("validation", "final"):
        raise ValueError("Paired evaluation only uses held-out synthetic voices")
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final is locked until model/runtime freeze")
    if split == "final":
        require_final_lock(project)
    _, positives = rows_for(root, config, "positive_" + split)
    noise_manifest = resolved(root, config["manifests"]["noise_" + split])
    noises = read_rows(noise_manifest)
    if not positives or not noises:
        raise ValueError("Held-out positive originals and source-separated evaluation noise are required")
    if any(n["split"] != split for n in noises):
        raise ValueError("Evaluation noise has the wrong split")
    audit = audit_project(project, include_final_files=split == "final")
    if not audit["passed"]:
        raise ValueError("Cross-split audit failed")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation audio versions are immutable")
    (output / "audio").mkdir(parents=True, exist_ok=True)
    entries = []
    for order, row in enumerate(positives[:originals]):
        source, _ = seek_audio(root, {**row, "start_s": row["call_start_s"], "end_s": row["call_end_s"]}, row["call_end_s"] - row["call_start_s"])
        # An explicit, recorded test construction gain; the same WAV is listened to and scored.
        mask0 = active_speech_mask(source, 0, len(source) / 16000)
        gain = min(.98 / max(np.max(np.abs(source)), 1e-9), 10 ** (-20 / 20) / max(rms(source, mask0), 1e-9))
        source = source * gain
        lead, tail = 2.4, 1.0
        speech = np.r_[np.zeros(round(lead * 16000)), source, np.zeros(round(tail * 16000))]
        mask = active_speech_mask(speech, lead, lead + len(source) / 16000)
        common = {"split": split, "kind": "positive", "provenance": "qwen_tts", "text": config["wakeword"],
            "human_reference_used": False, "voice_family_id": row["voice_family_id"], "original_id": row["sample_id"],
            "original_order": order + 1, "calls": [{"start_s": lead, "end_s": lead + len(source) / 16000,
                "call_id": row["sample_id"], "original_utterance_id": row["sample_id"]}],
            "source_id": source_key(row), "source_generation_gain": gain, "language": "Korean"}
        # One shared headroom gain for all the original's paired variants and clean baseline.
        variants = []
        for category in CATEGORIES:
            candidates = [n for n in noises if n["category"] == category]
            if not candidates:
                raise ValueError("Missing held-out noise category: " + category)
            for section in range(noise_sections):
                rng = np.random.default_rng(int(identity([config["augmentation"]["seed"], row["sample_id"], category, section])[:16], 16))
                noise = candidates[section % len(candidates)]
                samples, offset = seek_audio(root, noise, len(speech) / 16000, rng)
                if len(samples) < len(speech):
                    samples = np.tile(samples, int(np.ceil(len(speech) / len(samples))))
                samples = samples[:len(speech)]
                for snr in snrs:
                    scaled = samples * noise_scale(speech, samples, mask, snr)
                    variants.append((category, section, snr, noise, offset, scaled))
        headroom = common_headroom_gain(speech, [v[-1] for v in variants])
        if rms(speech * headroom, mask) < 10 ** (-35 / 20):
            raise ValueError("Paired word becomes too quiet after common headroom; choose another noise section, do not add playback-only gain")
        base = output / "audio" / f"original_{order+1:04d}_clean.wav"
        sf.write(base, pcm16(speech * headroom), 16000, subtype="PCM_16")
        entries.append({**common, "sample_id": f"original_{order+1:04d}_clean", "category": "clean", "snr_db": None,
            "path": str(base), "sha256": digest(base), "common_headroom_gain": headroom, "component_source_ids": [source_key(row)]})
        for category, section, snr, noise, offset, scaled in variants:
            name = f"original_{order+1:04d}_{category}_section{section+1}_snr{snr}"
            path = output / "audio" / (name + ".wav")
            sf.write(path, pcm16((speech + scaled) * headroom), 16000, subtype="PCM_16")
            entries.append({**common, "sample_id": name, "category": category, "snr_db": snr,
                "noise_section": section + 1, "noise_source_id": source_key(noise), "noise_offset_s": offset,
                "measured_snr_db": measured_snr(speech, scaled, mask), "common_headroom_gain": headroom,
                "path": str(path), "sha256": digest(path), "component_source_ids": [source_key(row), source_key(noise)]})
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in entries), encoding="utf-8")
    return {"originals": min(originals, len(positives)), "audio_files": len(entries), "manifest": str(manifest),
            "manifest_sha256": digest(manifest), "next": "Combine these positive entries with intact held-out Negative recordings and evaluate the exact WAVs."}
