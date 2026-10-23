"""One-pass whole-audio scoring for singles and their mean; truthful scope.

The same frame scores can be reused for declared Development gate experiments.
Final has one frozen gate and never grants tuning authority.
"""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock, require_final_lock
from .data import CATEGORIES, audit_project, source_key
from .events import positive_statistics, poisson_rate_interval
from .project import load_project
from .runtime import WakeWordDetector, verify_bundle, bundle_candidates, deployment_candidate
from .storage import project_storage
from .evaluation_workers import EvaluationJournal, score_sessions, worker_plan
from .streaming import VerifiedAudioFiles


def reverb_condition(row):
    """Missing metadata is the historical unmodified/dry condition."""
    return row.get("reverb_condition") or "none"


def _score_drop_statistics(values):
    values = [value for value in values if value is not None]
    return {"samples": len(values), "mean": float(np.mean(values)) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def annotate_clean_score_drops(clips):
    """Compare noise against the same room and reverb against original dry audio."""
    clean_by_original = {(row.get("original_id", row["sample_id"]), reverb_condition(row)): row
                         for row in clips if row["category"] == "clean"}
    for row in clips:
        original_id = row.get("original_id", row["sample_id"])
        clean = clean_by_original.get((original_id, reverb_condition(row)))
        dry_clean = clean_by_original.get((original_id, "none"))
        for key, value in row["results"].items():
            value["clean_score_drop"] = clean["results"][key]["max_score"] - value["max_score"] if clean else None
            value["dry_clean_score_drop"] = dry_clean["results"][key]["max_score"] - value["max_score"] if dry_clean else None
    return clips


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
                    if reverb_condition(row) == "none":
                        cells[(row["category"], row.get("snr_db"))].append(call)
            else:
                negatives.append((row, result))
        primary = [c for row in clip_rows if row["kind"] == "positive" and (row["category"] == "clean" or row.get("snr_db") in (20, 15, 10, 5)) for c in row["results"][model_id]["calls"]]
        clean = [c for row in clip_rows if row["category"] == "clean" and row["kind"] == "positive" and reverb_condition(row) == "none" for c in row["results"][model_id]["calls"]]
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
        summary["reverb_conditions"] = {}
        for condition in sorted({reverb_condition(row) for row in clip_rows if row["kind"] == "positive"}):
            selected_rows = [row for row in clip_rows if row["kind"] == "positive" and reverb_condition(row) == condition]
            selected_calls = [call for row in selected_rows for call in row["results"][model_id]["calls"]]
            condition_cells = defaultdict(list)
            for row in selected_rows:
                condition_cells[(row["category"], row.get("snr_db"))].extend(row["results"][model_id]["calls"])
            condition_statistics = {f"{cat}:{snr}": positive_statistics(calls)
                                    for (cat, snr), calls in condition_cells.items()}
            noise_statistics = [stats["recall"] for (cat, snr), calls in condition_cells.items()
                                if cat in CATEGORIES and snr in (20, 15, 10, 5)
                                for stats in [condition_statistics[f"{cat}:{snr}"]]]
            summary["reverb_conditions"][condition] = {
                "all_positive": positive_statistics(selected_calls),
                "clean": positive_statistics([call for row in selected_rows if row["category"] == "clean"
                                              for call in row["results"][model_id]["calls"]]),
                "conditions": condition_statistics,
                "noise_cell_mean": float(np.mean(noise_statistics)) if noise_statistics else None,
                "noise_cells_measured": len(noise_statistics),
                "noise_score_drop_from_same_reverb_clean": _score_drop_statistics(
                    [row["results"][model_id].get("clean_score_drop") for row in selected_rows if row["category"] != "clean"]),
                "clean_score_drop_from_dry_clean": _score_drop_statistics(
                    [row["results"][model_id].get("dry_clean_score_drop") for row in selected_rows if row["category"] == "clean"])}
        summary["positive_by_reverb_condition"] = {condition: value["all_positive"]
                                                   for condition, value in summary["reverb_conditions"].items()}
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


def evaluation_binding(project, bundle, manifest, stage="development", candidate=None, smoke=False):
    """Identity for resumable work; scoring still verifies all actual inputs.

    Reading this identity does not certify WAVs, model assets or a checkpoint.
    The evaluate entry point performs those checks before reusing any score.
    """
    directory = Path(bundle).resolve()
    package = read_json(directory / "manifest.json")
    model_ids = [row["id"] for row in bundle_candidates(package)]
    candidate = deployment_candidate(package)["id"] if candidate is None else candidate
    return identity({"project_sha256": digest(project), "bundle_sha256": digest(directory / "manifest.json"),
                     "manifest_sha256": digest(manifest), "stage": stage, "candidate": candidate,
                     "smoke": smoke, "model_ids": model_ids, "scoring_protocol": "reference_cpu_session_v1"})


@project_storage
def evaluate(project, bundle, output, stage="development", manifest=None, smoke=False, candidate=None, workers="auto", resume=False):
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
        lock = require_final_lock(project)
        if lock["bundle_manifest_sha256"] != digest(directory / "manifest.json") or lock["project_config_sha256"] != digest(project):
            raise ValueError("Final model/runtime/project changed after freeze")
        if smoke:
            raise ValueError("Final cannot be a smoke run")
    manifest = Path(manifest).resolve() if manifest else resolved(root, config["manifests"][stage])
    sources = read_rows(manifest)
    # Reuse this operation's full hash/header checks from the split audit in
    # the format/offset pass below. No evidence is reused across evaluations.
    verified_files = VerifiedAudioFiles()
    audit = audit_project(project, include_final_files=stage == "final", extra_rows=sources,
                          verified_files=verified_files)
    if not audit["passed"]:
        raise ValueError("Source/family split audit failed: " + str(audit["errors"][:3]))
    if not sources:
        raise ValueError("Empty evaluation manifest")
    final_audit = None
    if stage == "final":
        from .final_integrity import audit_independent_final
        final_audit = audit_independent_final(project, extra_rows=sources)
        if not final_audit["passed"]:
            raise ValueError("Independent Final source audit failed: " + str(final_audit["errors"][:3]))
    if len({r["sample_id"] for r in sources}) != len(sources):
        raise ValueError("Repeated evaluation IDs")
    if stage == "development" and any(r["kind"] == "positive" and (r.get("provenance") == "human"
            or any(c.get("role") == "call" and c.get("provenance") == "human"
                   for c in r.get("components", []) if isinstance(c, dict))) for r in sources):
        raise ValueError("Actual human call recordings are Final-only; do not select models/runtime with them")
    negative_sources, negative_hashes = set(), set()
    paths, offsets = [], [0]
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
        path, info = verified_files.inspect(root, row)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
            raise ValueError("Scored WAV must already be mono16kHz PCM16; no playback-only gain")
        paths.append(path)
        offsets.append(offsets[-1] + info.frames // 1280)
    verified_files.clear()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        if not resume or (output / "receipt.json").exists():
            raise ValueError("Evaluation outputs are immutable; choose a new directory (or resume an incomplete run explicitly)")
    elif resume:
        raise ValueError("No resumable evaluation checkpoint; choose a new output directory")
    runtime = package["runtime"]
    model_ids = [row["id"] for row in candidates]
    execution = worker_plan(workers, len(sources))
    binding = evaluation_binding(project, directory, manifest, stage, candidate, smoke)
    started = time.perf_counter()
    with writer_lock(output):
        journal = EvaluationJournal(output, binding, offsets, model_ids, resume=resume)
        try:
            execution["resumed_sessions"] = journal.resumed_sessions
            items = ((index, sources[index], paths[index]) for index in range(journal.resumed_sessions, len(sources)))
            for index, clip, scores in score_sessions(directory, model_ids, items, execution["workers"],
                                                       detector_factory=WakeWordDetector):
                journal.add(index, clip, scores)
                if index < 9 or (index + 1) % 100 == 0 or index + 1 == len(sources):
                    print(f"Evaluate {stage}: {index + 1}/{len(sources)} (CPU workers={execution['workers']})", flush=True)
            journal.publish_scores()
            clips = journal.clips
        finally:
            journal.close()
        execution["scoring_and_cache_seconds"] = time.perf_counter() - started
        execution["frame_scores_storage"] = "project-owned disk memmap; ordered resumable checkpoints every 32 sessions"
        annotate_clean_score_drops(clips)
        summaries = summarize(clips, config, candidates)
        report = {"schema_version": 1, "wakeword": config["wakeword"], "stage": stage, "created_at": now(),
            "smoke_test_only": smoke or any(m.get("smoke_test_only") for m in package["members"]),
            "bundle_manifest_sha256": digest(directory / "manifest.json"), "manifest_sha256": digest(manifest),
            "evaluation_manifest": {"path": str(manifest), "sha256": digest(manifest)},
            "evaluation_binding": binding,
            "runtime": runtime, "sessions": len(clips), "models": summaries, "execution": execution,
            "candidate_definitions": candidates, "selected_candidate": candidate,
            "deployment_candidate": deployed, "candidate_adoptions": candidate_adoptions(summaries, config),
            "adoption": adoption(summaries, config, candidate),
            "scope": "Supplied held-out manifest; synthetic and real human calls remain separate",
            "main_noise_snrs": [20, 15, 10, 5], "snr0_is_diagnostic": True}
        for summary in summaries:
            summary["positive_by_provenance"] = {p: positive_statistics([call for r in clips if r["kind"] == "positive" and r.get("provenance", "qwen_tts") == p for call in r["results"][summary["id"]]["calls"]]) for p in sorted({r.get("provenance", "qwen_tts") for r in clips if r["kind"] == "positive"})}
        if stage == "final":
            report["independent_source_audit"] = final_audit
            report["final_lock_policy"] = lock.get("independent_final_policy", "legacy_lock_new_source_audit_applied")
            report["legacy_lock_without_input_bindings"] = not bool(lock.get("input_bindings"))
            report["final_negative_duration_passed"] = all(s["negative"]["hours"] >= 100 for s in summaries)
            report["real_human_calls_present"] = any(r.get("provenance") == "human" for r in clips)
            report["real_population_verified"] = False
        report["score_cache_sha256"] = digest(output / "scores.npz")
        journal.publish_clips()
        write_json(output / "report.json", report)
        write_json(output / "receipt.json", {"report_sha256": digest(output / "report.json"),
            "clips_sha256": digest(output / "clips.json"), "bundle_sha256": digest(directory / "manifest.json"), "completed_at": now()})
        journal.discard_completed_work()
    return report
