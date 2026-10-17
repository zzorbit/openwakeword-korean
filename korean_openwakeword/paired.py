"""Controlled evaluation audio: same word, same noise section, changing SNR."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import numpy as np
import soundfile as sf

from .audio import active_speech_mask, pcm16, noise_scale, measured_snr, rms, common_headroom_gain
from .common import digest, identity, read_rows, resolved, require_final_lock, write_json
from .data import audit_project, rows_for, seek_audio, source_key, CATEGORIES
from .project import load_project
from .storage import project_storage
from .source_identity import component_lineage


def evaluation_recipe(project, split="validation", originals=100, noise_sections=5,
                      snrs=(20, 15, 10, 5, 0), noise_selection="legacy"):
    """Hash-bound construction inputs; old outputs never acquire a new policy."""
    if noise_selection not in ("legacy", "balanced_sources"):
        raise ValueError("Noise selection must be legacy or balanced_sources")
    root, config = load_project(project)
    return {"kind": "paired_evaluation_recipe_v1", "split": split,
            "noise_selection": "balanced_sources_v1" if noise_selection == "balanced_sources" else "legacy",
            "noise_window_policy": "bounded_offsets_then_zero_padded_event_v1" if noise_selection == "balanced_sources" else "legacy",
            "source_assignment_policy": "seed_sample_id_permutation_v1" if noise_selection == "balanced_sources" else "legacy",
            "requested_originals": originals, "noise_sections": noise_sections, "snrs": list(snrs),
            "seed": config["augmentation"]["seed"],
            "positive_manifest_sha256": digest(resolved(root, config["manifests"]["positive_" + split])),
            "noise_manifest_sha256": digest(resolved(root, config["manifests"]["noise_" + split]))}


def _noise_type(row):
    value = row.get("eval_type", row.get("subtype", row["category"]))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Evaluation noise type must be a nonempty declared string")
    return value.strip()


def _balanced_pool(noises, *, required_categories=CATEGORIES):
    """Deduplicate source aliases before balancing types and source groups.

    A canonical recording (where declared), canonical source, or exact source
    file SHA joins aliases into one group. Multiple intervals do not create
    extra source votes. Labels are declared metadata, never inferred classes.
    """
    parent = list(range(len(noises)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen = {}
    for i, row in enumerate(noises):
        sid, sha = source_key(row), row.get("sha256")
        if not sid or not sha:
            raise ValueError("Evaluation noise needs canonical source identity and exact file hash")
        recording = row.get("canonical_recording_id", row.get("parent_source_id"))
        keys = [("source", sid), ("sha256", sha)]
        if recording:
            keys.append(("recording", recording))
        for key in keys:
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
    aliases = defaultdict(list)
    for i, row in enumerate(noises):
        aliases[find(i)].append(row)
    pools = {category: defaultdict(list) for category in CATEGORIES}
    for rows in aliases.values():
        labels = {(r["category"], _noise_type(r)) for r in rows}
        if len(labels) != 1:
            raise ValueError("The same evaluation source has conflicting category/type labels")
        category, subtype = next(iter(labels))
        if category not in pools:
            raise ValueError("Unsupported evaluation noise category: " + category)
        intervals = {}
        for row in sorted(rows, key=lambda r: (str(source_key(r)), str(r["path"]), str(r.get("sample_id", "")))):
            key = (row["sha256"], row.get("start_s", 0), row.get("end_s"))
            intervals.setdefault(key, row)
        members = sorted(intervals.values(), key=lambda r: (r["sha256"], r.get("start_s", 0),
                          float("inf") if r.get("end_s") is None else r["end_s"]))
        hashes = sorted({r["sha256"] for r in rows})
        canonical = sorted({str(source_key(r)) for r in rows})
        group = {"id": identity(hashes), "category": category, "eval_type": subtype, "members": members,
                 "canonical_source_ids": canonical, "file_sha256": hashes, "input_rows": len(rows)}
        pools[category][subtype].append(group)
    for category in CATEGORIES:
        if not pools[category] and category in required_categories:
            raise ValueError("Missing held-out noise category: " + category)
        for groups in pools[category].values():
            groups.sort(key=lambda g: g["id"])
    return pools


def _balanced_choice(pools, category, slot, seed):
    types = sorted(pools[category])
    shift = int(identity([seed, category, "type_order"])[:8], 16) % len(types)
    subtype = types[(slot + shift) % len(types)]
    groups = pools[category][subtype]
    visit = slot // len(types)
    source_shift = int(identity([seed, category, subtype, "source_order"])[:8], 16) % len(groups)
    group = groups[(visit + source_shift) % len(groups)]
    member = group["members"][(visit // len(groups)) % len(group["members"])]
    return member, group


def _assignment_order(positives, seed):
    shuffled = sorted(positives, key=lambda r: identity([seed, r["sample_id"], "noise_assignment_v1"]))
    return {r["sample_id"]: i for i, r in enumerate(shuffled)}


def _source_peak_time(root, row, cache):
    key = (row["sha256"], row.get("start_s", 0), row.get("end_s"))
    if key in cache:
        return cache[key]
    with sf.SoundFile(resolved(root, row["path"])) as stream:
        lo = round(float(row.get("start_s", 0)) * stream.samplerate)
        hi = min(stream.frames, round(float(row.get("end_s", stream.frames / stream.samplerate)) * stream.samplerate))
        stream.seek(lo)
        position, peak, location = lo, 0.0, lo
        while position < hi:
            values = stream.read(min(65536, hi - position), dtype="float64", always_2d=True).mean(axis=1)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError("Empty/nonfinite evaluation source during bounded event search")
            index = int(np.argmax(np.abs(values)))
            if abs(values[index]) > peak:
                peak, location = float(abs(values[index])), position + index
            position += len(values)
        if peak <= 1e-8:
            raise ValueError("Selected evaluation source is silent; no background may be silently dropped")
        cache[key] = location / stream.samplerate
    return cache[key]


def _placed_noise(values, length, shift=0):
    output = np.zeros(length, dtype=np.float64)
    source_start, destination_start = max(0, -shift), max(0, shift)
    count = max(0, min(len(values) - source_start, length - destination_start))
    output[destination_start:destination_start + count] = values[source_start:source_start + count]
    return output, count, source_start, destination_start


def _balanced_noise_window(root, noise, speech, mask, rng, snrs, peak_cache):
    """Keep the chosen source, SNR and call level; never repeat an impact.

    Try at most sixteen original contiguous offsets. If an early/short event
    cannot overlap the call, retain one original excerpt and place it with
    explicit zero padding. No circular join, source replacement, SNR change or
    playback-only gain is allowed. The same construction serves every SNR.
    """
    seconds = len(speech) / 16000
    failed = []
    def usable(samples):
        if rms(samples, mask) <= 1e-8:
            return False, "no_noise_energy_in_call_region"
        scaled = samples * noise_scale(speech, samples, mask, min(snrs))
        gain = common_headroom_gain(speech, [scaled])
        if rms(speech * gain, mask) < 10 ** (-35 / 20):
            return False, "call_below_minus35dB_after_common_headroom"
        return True, None
    for attempt in range(1, 17):
        values, offset = seek_audio(root, noise, seconds, rng)
        samples, count, source_start, destination_start = _placed_noise(values, len(speech))
        accepted, reason = usable(samples)
        if accepted:
            return samples, offset, {"policy": "bounded_offsets_then_zero_padded_event_v1", "offset_attempts": attempt,
                "placement": "original_contiguous" if count == len(speech) else "right_zero_padding",
                "source_read_start_s": offset, "source_read_samples": len(values), "retained_samples": count,
                "destination_start_sample": destination_start, "source_start_sample": source_start,
                "repeat_count": 1, "shift_samples": 0, "rejected_offsets": failed}
        failed.append({"offset_s": offset, "reason": reason})
    peak = _source_peak_time(root, noise, peak_cache)
    active_center = int(np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2])
    start = max(float(noise.get("start_s", 0)), peak - active_center / 16000)
    values, offset = seek_audio(root, {**noise, "start_s": start}, seconds)
    shift = active_center - round((peak - offset) * 16000)
    samples, count, source_start, destination_start = _placed_noise(values, len(speech), shift)
    accepted, reason = usable(samples)
    if not accepted:
        raise ValueError(f"Evaluation source {source_key(noise)} cannot form an audible paired condition after 16 offsets "
                         f"and one zero-padded event placement: {reason}; source, SNR and threshold were not changed")
    return samples, offset, {"policy": "bounded_offsets_then_zero_padded_event_v1", "offset_attempts": 16,
        "placement": "zero_padded_event_alignment", "source_peak_time_s": peak,
        "source_read_start_s": offset, "source_read_samples": len(values), "retained_samples": count,
        "destination_start_sample": destination_start, "source_start_sample": source_start,
        "repeat_count": 1, "shift_samples": shift, "rejected_offsets": failed}


def _coverage(pools, selections, entries, recipe, manifest):
    used = Counter((r["category"], r["eval_type"], r["source_group_id"]) for r in selections)
    conditions = Counter((r["category"], str(r.get("snr_db"))) for r in entries)
    result = {"kind": "paired_noise_coverage_v1", "recipe": recipe,
              "manifest": {"path": str(manifest), "sha256": digest(manifest)},
              "originals_used": sum(r["category"] == "clean" for r in entries),
              "condition_counts": [{"category": c, "snr_db": None if s == "None" else float(s), "samples": n}
                                   for (c, s), n in sorted(conditions.items())], "categories": {},
              "scope": "Held-out declared source/type coverage, not proof of real-population performance or independent acoustic diversity",
              "counting": "One source selection is one original/category/section; its SNR variants are not independent source recordings"}
    for category, by_type in pools.items():
        sources, types = [], []
        for subtype, groups in sorted(by_type.items()):
            count = sum(used[category, subtype, g["id"]] for g in groups)
            used_count = sum(used[category, subtype, g["id"]] > 0 for g in groups)
            types.append({"eval_type": subtype, "available_sources": len(groups), "used_sources": used_count,
                          "source_selections": count, "samples_per_snr": {str(s): count for s in recipe["snrs"]}})
            for group in groups:
                count0 = used[category, subtype, group["id"]]
                sources.append({"source_group_id": group["id"], "eval_type": subtype,
                    "canonical_source_ids": group["canonical_source_ids"], "file_sha256": group["file_sha256"],
                    "source_selections": count0, "samples_per_snr": {str(s): count0 for s in recipe["snrs"]}})
        available = sum(len(g) for g in by_type.values())
        used_count = sum(r["source_selections"] > 0 for r in sources)
        result["categories"][category] = {"input_rows": sum(g["input_rows"] for gs in by_type.values() for g in gs),
            "effective_unique_sources": available, "used_sources": used_count, "unused_sources": available - used_count,
            "available_types": len(types), "used_types": sum(t["source_selections"] > 0 for t in types),
            "source_selections": sum(t["source_selections"] for t in types), "types": types, "sources": sources,
            "all_sources_used": available == used_count,
            "limitation": None if available == used_count else "Limited original/section slots and balanced type allocation leave some available sources unused"}
    return result


@project_storage
def prepare_evaluation(project, output, split="validation", originals=100, noise_sections=5, snrs=(20, 15, 10, 5, 0), noise_selection="legacy"):
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
    final_audit = None
    if split == "final":
        from .final_integrity import audit_independent_final
        final_audit = audit_independent_final(project)
        if not final_audit["passed"]:
            raise ValueError("Independent Final source audit failed: " + str(final_audit["errors"][:3]))
    recipe = evaluation_recipe(project, split, originals, noise_sections, snrs, noise_selection)
    pools = _balanced_pool(noises) if noise_selection == "balanced_sources" else None
    selection_order = {}
    if pools is not None:
        positives = sorted(positives, key=lambda r: (str(r["sample_id"]), r["sha256"]))
        # Keep playback order stable while avoiding a periodic correlation
        # between synthetic voice/style slots and background-type cycles.
        selection_order = _assignment_order(positives[:originals], config["augmentation"]["seed"])
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation audio versions are immutable")
    (output / "audio").mkdir(parents=True, exist_ok=True)
    write_json(output / "recipe.json", recipe)
    entries, selections, peak_cache = [], [], {}
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
        call_lineage = component_lineage(row, "call")
        # One shared headroom gain for all the original's paired variants and clean baseline.
        variants = []
        for category in CATEGORIES:
            candidates = [n for n in noises if n["category"] == category]
            if not candidates:
                raise ValueError("Missing held-out noise category: " + category)
            for section in range(noise_sections):
                rng = np.random.default_rng(int(identity([config["augmentation"]["seed"], row["sample_id"], category, section])[:16], 16))
                if pools is None:
                    noise, source_group = candidates[section % len(candidates)], None
                else:
                    noise, source_group = _balanced_choice(pools, category, selection_order[row["sample_id"]] * noise_sections + section, config["augmentation"]["seed"])
                    selections.append({"category": category, "eval_type": source_group["eval_type"], "source_group_id": source_group["id"]})
                construction = None
                if pools is not None:
                    try:
                        samples, offset, construction = _balanced_noise_window(root, noise, speech, mask, rng, snrs, peak_cache)
                    except ValueError as error:
                        write_json(output / "noise_construction_failure.json", {"source_id": source_key(noise),
                            "source_sha256": noise["sha256"], "category": category, "eval_type": _noise_type(noise),
                            "original_id": row["sample_id"], "error": str(error), "recipe": recipe,
                            "source_dropped": False, "SNR_changed": False, "policy_lowered": False})
                        raise
                else:
                    samples, offset = seek_audio(root, noise, len(speech) / 16000, rng)
                    if len(samples) < len(speech):
                        samples = np.tile(samples, int(np.ceil(len(speech) / len(samples))))
                    samples = samples[:len(speech)]
                for snr in snrs:
                    scaled = samples * noise_scale(speech, samples, mask, snr)
                    variants.append((category, section, snr, noise, offset, scaled, source_group, construction))
        headroom = common_headroom_gain(speech, [v[5] for v in variants])
        if rms(speech * headroom, mask) < 10 ** (-35 / 20):
            if pools is not None:
                culprit = max(variants, key=lambda v: np.max(np.abs(speech + v[5])))
                write_json(output / "noise_construction_failure.json", {"source_id": source_key(culprit[3]),
                    "source_sha256": culprit[3]["sha256"], "category": culprit[0], "eval_type": _noise_type(culprit[3]),
                    "original_id": row["sample_id"], "error": "Combined common headroom makes the call quieter than -35 dB RMS",
                    "common_headroom_gain": headroom, "construction": culprit[7], "recipe": recipe,
                    "source_dropped": False, "SNR_changed": False, "policy_lowered": False})
            raise ValueError("Paired word becomes too quiet after common headroom; choose another noise section, do not add playback-only gain")
        base = output / "audio" / f"original_{order+1:04d}_clean.wav"
        sf.write(base, pcm16(speech * headroom), 16000, subtype="PCM_16")
        entries.append({**common, "sample_id": f"original_{order+1:04d}_clean", "category": "clean", "snr_db": None,
            "path": str(base), "sha256": digest(base), "common_headroom_gain": headroom,
            "component_source_ids": [source_key(row)], "components": [call_lineage]})
        for category, section, snr, noise, offset, scaled, source_group, construction in variants:
            name = f"original_{order+1:04d}_{category}_section{section+1}_snr{snr}"
            path = output / "audio" / (name + ".wav")
            sf.write(path, pcm16((speech + scaled) * headroom), 16000, subtype="PCM_16")
            entries.append({**common, "sample_id": name, "category": category, "snr_db": snr,
                "noise_section": section + 1, "noise_source_id": source_key(noise), "noise_offset_s": offset,
                "measured_snr_db": measured_snr(speech, scaled, mask), "common_headroom_gain": headroom,
                "path": str(path), "sha256": digest(path), "component_source_ids": [source_key(row), source_key(noise)],
                "components": [call_lineage, component_lineage(noise, "background")]})
            if source_group is not None:
                entries[-1].update(noise_eval_type=source_group["eval_type"], noise_eval_label=noise.get("eval_type_label"),
                                   noise_source_group_id=source_group["id"],
                                   noise_selection_policy=recipe["noise_selection"], noise_construction=construction)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in entries), encoding="utf-8")
    result = {"originals": min(originals, len(positives)), "audio_files": len(entries), "manifest": str(manifest),
              "manifest_sha256": digest(manifest), "recipe": {"path": str(output / "recipe.json"), "sha256": digest(output / "recipe.json")},
              "noise_selection_policy": recipe["noise_selection"],
              "next": "Combine these positive entries with intact held-out Negative recordings and evaluate the exact WAVs."}
    if pools is not None:
        write_json(output / "noise_coverage.json", _coverage(pools, selections, entries, recipe, manifest))
        result["noise_coverage"] = {"path": str(output / "noise_coverage.json"), "sha256": digest(output / "noise_coverage.json")}
    if final_audit is not None:
        write_json(output / "independent_source_audit.json", final_audit)
        result["independent_source_audit"] = {"path": str(output / "independent_source_audit.json"),
                                               "sha256": digest(output / "independent_source_audit.json")}
    return result
