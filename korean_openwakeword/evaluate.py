"""One-pass whole-audio scoring for singles and their mean; truthful scope.

The same frame scores can be reused for declared Development gate experiments.
Final has one frozen gate and never grants tuning authority.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

from .common import bound_file, digest, now, read_json, read_rows, resolved, write_json, writer_lock
from .data import CATEGORIES, audit_project, source_key
from .events import EventDetector, PositiveTracker, positive_statistics, poisson_rate_interval
from .project import load_project
from .runtime import WakeWordDetector, verify_bundle, bundle_candidates, deployment_candidate
from .storage import project_storage


def summarize(clip_rows, config, candidates=None):
    summaries = []
    kinds = {row["id"]: row["kind"] for row in candidates or []}
    for model_id in clip_rows[0]["results"] if clip_rows else []:
        positives, negatives, cells = [], [], defaultdict(list)
        for row in clip_rows:
            result = row["results"][model_id]
            if row["kind"] == "positive":
                for call in result["calls"]:
                    positives.append(call)
                    cells[(row["category"], row.get("snr_db"))].append(call)
            else:
                negatives.append((row, result))
        primary = [c for row in clip_rows if row["kind"] == "positive" and (row["category"] == "clean" or row.get("snr_db") in (20, 15, 10, 5)) for c in row["results"][model_id]["calls"]]
        clean = [c for row in clip_rows if row["category"] == "clean" and row["kind"] == "positive" for c in row["results"][model_id]["calls"]]
        grouped = {f"{cat}:{snr}": positive_statistics(calls) for (cat, snr), calls in cells.items()}
        noise_recalls = [v["recall"] for k, v in grouped.items() if k.split(":")[0] in CATEGORIES and k.split(":")[1] in ("20", "15", "10", "5", "20.0", "15.0", "10.0", "5.0")]
        fa = sum(r["false_activations"] for _, r in negatives)
        hours = sum(row["eligible_seconds"] for row, _ in negatives) / 3600
        by_negative = {}
        for category in CATEGORIES:
            selected = [(row, r) for row, r in negatives if row["category"] == category]
            by_negative[category] = poisson_rate_interval(sum(r["false_activations"] for _, r in selected), sum(row["eligible_seconds"] for row, _ in selected) / 3600)
        summaries.append({"id": model_id, "kind": kinds.get(model_id, "ensemble" if model_id == "mean" else "single"),
            "primary": positive_statistics(primary), "all_positive": positive_statistics(positives),
            "clean": positive_statistics(clean), "noise_cell_mean": float(np.mean(noise_recalls)) if noise_recalls else None,
            "noise_cells_measured": len(noise_recalls), "conditions": grouped,
            "negative": poisson_rate_interval(fa, hours), "negative_by_category": by_negative})
        summary = summaries[-1]
        summary["positive_by_provenance"] = {p: positive_statistics([c for c in positives if c["provenance"] == p]) for p in sorted({c["provenance"] for c in positives})}
        human = [c for c in positives if c["provenance"] == "human"]
        for dimension in ("speaker_id", "category", "distance_m", "direction", "microphone"):
            summary["human_by_" + dimension] = {str(v): positive_statistics([c for c in human if str(c.get(dimension)) == str(v)]) for v in sorted({str(c.get(dimension)) for c in human})}
    singles = sorted([r for r in summaries if r["kind"] == "single"], key=lambda r: (-(r["primary"]["recall"] if r["primary"]["recall"] is not None else -1), r["id"]))
    for i, row in enumerate(singles):
        row["rank"] = 1 + sum(other["primary"]["recall"] is not None and row["primary"]["recall"] is not None and other["primary"]["recall"] > row["primary"]["recall"] for other in singles)
    return singles + [r for r in summaries if r["kind"] == "ensemble"]


def adoption(summaries, config, candidate=None):
    if candidate is not None:
        selected = next((r for r in summaries if r["id"] == candidate), None)
        if selected is None:
            raise ValueError(f"Unknown evaluated candidate: {candidate}")
    else:
        selected = next((r for r in summaries if r["kind"] == "ensemble"),
                        summaries[0] if len(summaries) == 1 else None)
    errors = []
    if selected is None:
        return {"passed": False, "reasons": ["No evaluated candidate"]}
    targets = config["evaluation_targets"]
    if selected["clean"]["recall"] is None or selected["clean"]["recall"] < targets["clean_recall"]:
        errors.append("Clean recall below target or unmeasured")
    for category in CATEGORIES:
        for snr, key in ((10, "snr10_recall"), (5, "snr5_recall")):
            cell = selected["conditions"].get(f"{category}:{snr}", selected["conditions"].get(f"{category}:{float(snr)}"))
            if not cell or cell["recall"] is None or cell["recall"] < targets[key]:
                errors.append(f"{category} SNR{snr} below target or unmeasured")
    negative = selected["negative"]
    if negative["hours"] < config["targets"]["development_negative_hours"][0]:
        errors.append(f"Development FP duration below {config['targets']['development_negative_hours'][0]:g}h")
    if negative["fa_per_hour"] is None or negative["fa_per_hour"] > targets["fa_per_hour"]:
        errors.append("FA/hour target failed or unmeasured")
    latency = selected["primary"].get("latency_p95_s")
    if latency is None or latency > targets["latency_p95_s"]:
        errors.append("Latency p95 target failed or unmeasured")
    return {"passed": not errors, "candidate": selected["id"], "reasons": errors,
            "scope": "Development observed values; not a population guarantee or independent Final"}


def candidate_adoptions(summaries, config):
    """Apply identical declared gates independently to every measured candidate."""
    return {row["id"]: adoption(summaries, config, row["id"]) for row in summaries}


@project_storage
def evaluate(project, bundle, output, stage="development", manifest=None, smoke=False, candidate=None):
    root, config = load_project(project)
    directory = Path(bundle).resolve()
    package = verify_bundle(directory)
    candidates = bundle_candidates(package)
    deployed = deployment_candidate(package)
    candidate = deployed["id"] if candidate is None else candidate
    if candidate not in {row["id"] for row in candidates}:
        raise ValueError(f"Unknown bundle candidate: {candidate}")
    if package["wakeword"] != config["wakeword"]:
        raise ValueError("Candidate word differs from this project")
    if stage not in ("development", "final"):
        raise ValueError("Unknown evaluation stage")
    if stage == "final":
        if candidate != deployed["id"]:
            raise ValueError("Final evaluates the frozen deployed candidate; no candidate selection")
        lock = read_json(root / "final_lock.json")
        if lock["bundle_manifest_sha256"] != digest(directory / "manifest.json") or lock["project_config_sha256"] != digest(project):
            raise ValueError("Final model/runtime/project changed after freeze")
        if smoke:
            raise ValueError("Final cannot be a smoke run")
    manifest = Path(manifest).resolve() if manifest else resolved(root, config["manifests"][stage])
    sources = read_rows(manifest)
    audit = audit_project(project, include_final_files=stage == "final", extra_rows=sources)
    if not audit["passed"]:
        raise ValueError("Source/family split audit failed: " + str(audit["errors"][:3]))
    if not sources:
        raise ValueError("Empty evaluation manifest")
    if len({r["sample_id"] for r in sources}) != len(sources):
        raise ValueError("Repeated evaluation IDs")
    if stage == "development" and any(r["kind"] == "positive" and r.get("provenance") == "human" for r in sources):
        raise ValueError("Actual human call recordings are Final-only; do not select models/runtime with them")
    negative_sources, negative_hashes = set(), set()
    for row in sources:
        if row["split"] != ("validation" if stage == "development" else "final"):
            raise ValueError("Evaluation stage/split mismatch")
        if row.get("kind") not in ("positive", "negative"):
            raise ValueError("Evaluation kind required")
        if row.get("kind") == "positive" and not row.get("calls"):
            raise ValueError("Positive evaluation needs explicit call boundaries")
        if row["kind"] == "negative":
            if row.get("calls"):
                raise ValueError("Negative evaluation cannot declare calls")
            key = source_key(row)
            if key in negative_sources or row["sha256"] in negative_hashes:
                raise ValueError("Repeated Negative originals cannot inflate independent evaluation hours")
            negative_sources.add(key); negative_hashes.add(row["sha256"])
        bound_file(root, row)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation outputs are immutable; choose a new directory")
    predictor = WakeWordDetector(directory)
    runtime = package["runtime"]
    model_ids = [row["id"] for row in candidates]
    clips, cached_scores, offsets = [], [], [0]
    with writer_lock(output):
        for index, row in enumerate(sources):
            path = bound_file(root, row)
            predictor.reset()
            gates = {k: EventDetector(runtime["threshold"], runtime["consecutive_frames"], runtime["cooldown_s"]) for k in model_ids}
            session = {"session_id": row["sample_id"], "source_id": source_key(row), "calls": row.get("calls", []),
                "provenance": row.get("provenance", "qwen_tts" if row["kind"] == "positive" else "public_dataset"),
                "voice_family_id": row.get("voice_family_id"), "category": row["category"], "speaker_id": row.get("speaker_id")}
            session.update({key: row[key] for key in ("distance_m", "direction", "microphone") if key in row})
            for call_index, call in enumerate(session["calls"]):
                call.setdefault("call_id", row["sample_id"] + f":call{call_index}")
                call.setdefault("original_utterance_id", row.get("original_id", row["sample_id"]))
            trackers = {k: PositiveTracker(session, runtime["call_postroll_s"]) for k in model_ids}
            events = {k: [] for k in model_ids}
            max_scores = dict.fromkeys(model_ids, 0.0)
            frames = 0
            frame_scores = []
            with sf.SoundFile(path) as audio:
                if audio.samplerate != 16000 or audio.channels != 1 or audio.subtype != "PCM_16":
                    raise ValueError("Scored WAV must already be mono16kHz PCM16; no playback-only gain")
                for call in session["calls"]:
                    if not 0 <= call["start_s"] < call["end_s"] <= audio.frames / 16000:
                        raise ValueError("Call boundary exceeds evaluation file")
                    if call["start_s"] < runtime["warmup_s"]:
                        raise ValueError("Positive call starts during excluded warmup; fix the test audio")
                while True:
                    samples = audio.read(1280, dtype="int16")
                    if len(samples) < 1280:
                        break
                    scores = predictor.score_members(samples)
                    if len(scores) == 3:
                        scores = np.r_[scores, np.float32((np.float64(scores[0]) + np.float64(scores[1]) + np.float64(scores[2])) / 3)]
                    frame_scores.append(scores)
                    timestamp = (frames + 1) * .08
                    eligible = frames * .08 + 1e-9 >= runtime["warmup_s"]
                    for key, score in zip(model_ids, scores):
                        decision = gates[key].update(float(score), timestamp, eligible)
                        max_scores[key] = max(max_scores[key], float(score))
                        if decision["event"]:
                            events[key].append(timestamp)
                        if eligible and row["kind"] == "positive":
                            trackers[key].update(timestamp, float(score), decision)
                    frames += 1
            results = {}
            for key in model_ids:
                calls = trackers[key].rows
                results[key] = {"calls": calls, "success": all(c["success"] for c in calls) if calls else None,
                    "events_s": events[key], "false_activations": len(events[key]) if row["kind"] == "negative" else trackers[key].outside_call_events,
                    "max_score": max((c["max_wake_score"] or 0 for c in calls), default=max_scores[key]),
                    "latency_s": calls[0]["latency_s"] if len(calls) == 1 else None,
                    "activity": gates[key].activity_statistics()}
            clips.append({**row, "path": str(path), "duration_s": frames * .08,
                "eligible_seconds": max(0, frames * .08 - runtime["warmup_s"]), "results": results})
            cached_scores.append(np.asarray(frame_scores, dtype=np.float32).reshape(-1, len(model_ids)))
            offsets.append(offsets[-1] + frames)
            print(f"Evaluate {stage}: {index+1}/{len(sources)}", flush=True)
        clean_by_original = {r.get("original_id", r["sample_id"]): r for r in clips if r["category"] == "clean"}
        for row in clips:
            clean = clean_by_original.get(row.get("original_id", row["sample_id"]))
            for key, value in row["results"].items():
                value["clean_score_drop"] = clean["results"][key]["max_score"] - value["max_score"] if clean else None
        summaries = summarize(clips, config, candidates)
        report = {"schema_version": 1, "wakeword": config["wakeword"], "stage": stage, "created_at": now(),
            "smoke_test_only": smoke or any(m.get("smoke_test_only") for m in package["members"]),
            "bundle_manifest_sha256": digest(directory / "manifest.json"), "manifest_sha256": digest(manifest),
            "runtime": runtime, "sessions": len(clips), "models": summaries,
            "candidate_definitions": candidates, "selected_candidate": candidate,
            "deployment_candidate": deployed, "candidate_adoptions": candidate_adoptions(summaries, config),
            "adoption": adoption(summaries, config, candidate),
            "scope": "Supplied held-out manifest; synthetic and real human calls remain separate",
            "main_noise_snrs": [20, 15, 10, 5], "snr0_is_diagnostic": True}
        for summary in summaries:
            summary["positive_by_provenance"] = {p: positive_statistics([call for r in clips if r["kind"] == "positive" and r.get("provenance", "qwen_tts") == p for call in r["results"][summary["id"]]["calls"]]) for p in sorted({r.get("provenance", "qwen_tts") for r in clips if r["kind"] == "positive"})}
        if stage == "final":
            report["final_negative_duration_passed"] = all(s["negative"]["hours"] >= 100 for s in summaries)
            report["real_human_calls_present"] = any(r.get("provenance") == "human" for r in clips)
            report["real_population_verified"] = False
        np.savez_compressed(output / "scores.npz", scores=np.concatenate(cached_scores), offsets=np.asarray(offsets, dtype=np.int64), model_ids=np.asarray(model_ids))
        report["score_cache_sha256"] = digest(output / "scores.npz")
        write_json(output / "clips.json", clips)
        write_json(output / "report.json", report)
        write_json(output / "receipt.json", {"report_sha256": digest(output / "report.json"),
            "clips_sha256": digest(output / "clips.json"), "bundle_sha256": digest(directory / "manifest.json"), "completed_at": now()})
    return report
