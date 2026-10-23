"""Development-only gate comparison. Reuse exact scores; never rerun the network."""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
from .common import digest, now, read_json, write_json, writer_lock
from .project import load_project
from .events import EventDetector, PositiveTracker
from .evaluate import summarize, adoption, adoption_criteria, candidate_adoptions, annotate_clean_score_drops
from .runtime import verify_bundle, deployment_candidate
from .storage import project_storage
from .cache_analysis import statistic_projection, write_clip_array


def iter_replay(clips, scores, offsets, model_ids, runtime, columns=None):
    """Replay one record at a time; the network and source metadata stay intact."""
    columns = list(range(len(model_ids))) if columns is None else columns
    for index, source in enumerate(clips):
        row = dict(source)
        session = {"session_id": row["sample_id"], "source_id": row.get("source_id", row["sample_id"]), "calls": deepcopy(row.get("calls", [])), "provenance": row.get("provenance", "qwen_tts"), "voice_family_id": row.get("voice_family_id"), "category": row["category"], "speaker_id": row.get("speaker_id")}
        session.update({key: row[key] for key in ("distance_m", "direction", "microphone") if key in row})
        for j, call in enumerate(session["calls"]):
            call.setdefault("call_id", f"{row['sample_id']}:call{j}")
            call.setdefault("original_utterance_id", row.get("original_id", row["sample_id"]))
        if "calls" in row:
            row["calls"] = session["calls"]
        row["results"] = {}
        frames = scores[offsets[index]:offsets[index + 1]]
        for key, column in zip(model_ids, columns):
            gate = EventDetector(runtime["threshold"], runtime["consecutive_frames"], runtime["cooldown_s"])
            tracker = PositiveTracker(session, runtime["call_postroll_s"])
            events = []
            for f, score in enumerate(frames[:, column]):
                eligible = f * .08 + 1e-9 >= runtime["warmup_s"]
                decision = gate.update(float(score), (f + 1) * .08, eligible)
                if decision["event"]:
                    events.append((f + 1) * .08)
                if eligible and row["kind"] == "positive":
                    tracker.update((f + 1) * .08, float(score), decision)
            calls = tracker.rows
            row["results"][key] = {"calls": calls, "success": all(c["success"] for c in calls) if calls else None,
                "events_s": events, "false_activations": len(events) if row["kind"] == "negative" else tracker.outside_call_events,
                "max_score": max((c["max_wake_score"] or 0 for c in calls), default=float(np.max(frames[:, column])) if len(frames) else 0),
                "latency_s": calls[0]["latency_s"] if len(calls) == 1 else None, "activity": gate.activity_statistics()}
        yield row


def replay(clips, scores, offsets, model_ids, runtime):
    """Legacy in-memory API for small callers; large paths use iter_replay."""
    return annotate_clean_score_drops(list(iter_replay(clips, scores, offsets, model_ids, runtime)))


def _analysis_rows(clips, scores, offsets, model_ids, runtime, columns=None):
    # Only call outcomes and grouping fields are retained. Full nested source
    # provenance, component inventories and event diagnostics remain on disk.
    rows = [statistic_projection(row) for row in
            iter_replay(clips, scores, offsets, model_ids, runtime, columns)]
    return annotate_clean_score_drops(rows)


def _clean_scores(rows):
    return {(row.get("original_id", row["sample_id"]), row.get("reverb_condition") or "none"):
            {key: value["max_score"] for key, value in row["results"].items()}
            for row in rows if row["category"] == "clean"}


def _annotated_replay(clips, scores, offsets, ids, runtime, clean):
    for row in iter_replay(clips, scores, offsets, ids, runtime):
        original = row.get("original_id", row["sample_id"])
        room = clean.get((original, row.get("reverb_condition") or "none"))
        dry = clean.get((original, "none"))
        for key, value in row["results"].items():
            value["clean_score_drop"] = room[key] - value["max_score"] if room else None
            value["dry_clean_score_drop"] = dry[key] - value["max_score"] if dry else None
        yield row


def _parameters(thresholds, consecutive):
    if (not thresholds or not consecutive
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t < 1 for t in thresholds)
            or any(type(n) is not int or n < 1 for n in consecutive)):
        raise ValueError("Declare finite thresholds in (0,1) and positive frame counts")
    return list(dict.fromkeys(thresholds)), list(dict.fromkeys(consecutive))


def _output_path(output, bundle, evaluation):
    output, bundle, evaluation = map(lambda value: Path(value).resolve(), (output, bundle, evaluation))
    if (output == bundle or output == evaluation or output.is_relative_to(bundle)
            or output.is_relative_to(evaluation) or bundle.is_relative_to(output)
            or evaluation.is_relative_to(output)):
        raise ValueError("Analysis output must not modify or contain the source bundle/evaluation")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output is immutable; choose a new directory")
    return output


def _source_bindings(project, bundle, evaluation):
    return {"source_bundle_manifest_sha256": digest(Path(bundle) / "manifest.json"),
            "source_report_sha256": digest(Path(evaluation) / "report.json"),
            "source_clips_sha256": digest(Path(evaluation) / "clips.json"),
            "source_score_cache_sha256": digest(Path(evaluation) / "scores.npz"),
            "project_config_sha256": digest(project)}


@project_storage
def compare_cached(project, bundle, evaluation, output, thresholds, consecutive=(2,), candidate=None):
    """Compare explicit settings without selecting/deploying a model or runtime.

    Call this from a background job. Validation streams the original archive;
    unchanged settings reuse the frozen summary, and new settings replay only
    the requested candidate's exact cached score column.
    """
    from .selection import verified_evaluation
    thresholds, consecutive = _parameters(thresholds, consecutive)
    root, config, package, report, clips, scores, offsets, candidates = verified_evaluation(project, bundle, evaluation)
    if (root / "final_lock.json").exists():
        raise ValueError("Runtime is frozen; start a versioned experiment")
    candidate = deployment_candidate(package)["id"] if candidate is None else candidate
    chosen = next((row for row in candidates if row["id"] == candidate), None)
    if chosen is None:
        raise ValueError(f"Unknown evaluated candidate: {candidate}")
    output = _output_path(output, bundle, evaluation)
    ids = [row["id"] for row in candidates]
    column = ids.index(candidate)
    bindings = _source_bindings(project, bundle, evaluation)
    comparisons = []
    with writer_lock(output):
        for threshold in thresholds:
            for count in consecutive:
                runtime = dict(package["runtime"], threshold=threshold, consecutive_frames=count)
                if runtime == package["runtime"]:
                    summaries = [deepcopy(next(row for row in report["models"] if row["id"] == candidate))]
                else:
                    rows = _analysis_rows(clips, scores, offsets, [candidate], runtime, [column])
                    summaries = summarize(rows, config, [chosen])
                    del rows
                decision = adoption(summaries, config, candidate)
                comparisons.append({"runtime": runtime, "adoption": decision,
                    "candidate_adoptions": candidate_adoptions(summaries, config), "models": summaries})
                print(f"Cache comparison: {len(comparisons)}/{len(thresholds) * len(consecutive)} candidate={candidate} threshold={threshold} frames={count}", flush=True)
        result = {"schema_version": 1, "stage": "development", "created_at": now(),
            "candidate": chosen, "adoption_criteria": adoption_criteria(config), **bindings,
            "comparisons": comparisons, "selection_applied": False, "network_rerun": False,
            "scope": "Explicit cached-score comparisons only; original model, project and evaluation are unchanged"}
        write_json(output / "comparison.json", result)
        write_json(output / "receipt.json", {"comparison_sha256": digest(output / "comparison.json"),
            **bindings, "completed_at": now(), "selection_applied": False})
    return {"comparison": str(output / "comparison.json"), "receipt": str(output / "receipt.json"),
            "candidate": candidate, "comparisons": comparisons,
            "network_rerun": False, "selection_applied": False}


def tune(project, bundle, evaluation, output, thresholds, consecutive=(2,), candidate=None):
    from .selection import verified_evaluation, copy_candidate_bundle, restrict_evaluation
    root, config, package, report, clips, scores, offsets, candidates = verified_evaluation(project, bundle, evaluation)
    if (root / "final_lock.json").exists():
        raise ValueError("Runtime is frozen; start a versioned experiment")
    evaluation, bundle, output = map(lambda p: Path(p).resolve(), (evaluation, bundle, output))
    candidate = deployment_candidate(package)["id"] if candidate is None else candidate
    chosen = next((row for row in candidates if row["id"] == candidate), None)
    if chosen is None:
        raise ValueError(f"Unknown evaluated candidate: {candidate}")
    thresholds, consecutive = _parameters(thresholds, consecutive)
    output = _output_path(output, bundle, evaluation)
    ids = [row["id"] for row in candidates]
    comparisons, best = [], None
    with writer_lock(output):
        for threshold in thresholds:
            for count in consecutive:
                runtime = dict(package["runtime"], threshold=threshold, consecutive_frames=count)
                rows = _analysis_rows(clips, scores, offsets, ids, runtime)
                summaries = summarize(rows, config, candidates)
                decision = adoption(summaries, config, candidate)
                selected = next(s for s in summaries if s["id"] == candidate)
                key = (decision["passed"], selected["clean"]["recall"] or 0, selected["noise_cell_mean"] or 0, -(selected["negative"]["fa_per_hour"] if selected["negative"]["fa_per_hour"] is not None else math.inf))
                comparisons.append({"runtime": runtime, "adoption": decision,
                                    "candidate_adoptions": candidate_adoptions(summaries, config), "models": summaries})
                if best is None or key > best[0]:
                    best = (key, runtime, _clean_scores(rows), summaries, decision)
                del rows
                print(f"Development gate: threshold={threshold} frames={count} adoption={decision['passed']}", flush=True)
        write_json(output / "comparison.json", {"stage": "development", "created_at": now(), "selected_candidate": chosen,
                    "adoption_criteria": adoption_criteria(config),
                    "source_report_sha256": digest(evaluation / "report.json"), "source_score_cache_sha256": digest(evaluation / "scores.npz"),
                    "comparisons": comparisons, "selection_order": ["all admission targets passed", "clean recall", "balanced noise recall", "lower FA/hour"],
                    "scope": "Declared gates for one explicit/default deployed candidate; no network retraining and no Final tuning"})
        selected_bundle = output / "bundle"
        package.update(runtime=best[1], gate_selection_sha256=digest(output / "comparison.json"), stage="development_gate_selected")
        evidence = {"stage": "development", "candidate": chosen, "selection": "declared_gate_comparison",
                    "source_bundle_manifest_sha256": digest(bundle / "manifest.json"),
                    "source_report_sha256": digest(evaluation / "report.json"),
                    "source_score_cache_sha256": digest(evaluation / "scores.npz"),
                    "comparison_sha256": digest(output / "comparison.json"), "runtime": best[1],
                    "adoption": best[4], "adoption_criteria": adoption_criteria(config),
                    "Final_used_for_selection": False, "network_rerun": False}
        copy_candidate_bundle(bundle, selected_bundle, package, chosen, evidence=evidence)
        selected_report = dict(report, runtime=best[1], models=best[3], adoption=best[4])
        selected_rows = _annotated_replay(clips, scores, offsets, ids, best[1], best[2])
        selected_report, selected_clips, selected_scores = restrict_evaluation(selected_report, selected_rows, scores, candidates, chosen, config)
        selected_report.update(bundle_manifest_sha256=digest(selected_bundle / "manifest.json"),
                               gate_selection_sha256=digest(output / "comparison.json"), created_at=now())
        selected_dir = output / "evaluation"
        write_clip_array(selected_dir / "clips.json", selected_clips)
        np.savez_compressed(selected_dir / "scores.npz", scores=selected_scores, offsets=offsets,
                            model_ids=np.asarray([row["id"] for row in selected_report["candidate_definitions"]]))
        selected_report["score_cache_sha256"] = digest(selected_dir / "scores.npz")
        write_json(selected_dir / "report.json", selected_report)
        write_json(selected_dir / "receipt.json", {"report_sha256": digest(selected_dir / "report.json"), "clips_sha256": digest(selected_dir / "clips.json"), "bundle_sha256": digest(selected_bundle / "manifest.json"), "completed_at": now()})
    return {"bundle": str(selected_bundle), "evaluation": str(selected_dir), "candidate": candidate,
            "runtime": best[1], "adoption": best[4], "smoke_test_only": report["smoke_test_only"],
            "network_rerun": False}


@project_storage
def select_cached_configuration(project, bundle, evaluation, output, candidate,
                                threshold, consecutive, comparison=None):
    """Export exactly the user's chosen candidate/settings to a new directory.

    A comparison can authorize only a setting it actually measured. No result
    is chosen by rank here, and the project runtime is never updated implicitly.
    """
    from .selection import select_candidate
    _parameters([threshold], [consecutive])
    package = verify_bundle(bundle)
    if comparison is not None:
        path = Path(comparison).resolve()
        if path.is_dir():
            path = path / "comparison.json"
        evidence, receipt = read_json(path), read_json(path.parent / "receipt.json")
        bindings = _source_bindings(project, bundle, evaluation)
        if (receipt.get("comparison_sha256") != digest(path)
                or evidence.get("stage") != "development"
                or evidence.get("selection_applied") is not False
                or any(evidence.get(key) != value or receipt.get(key) != value
                       for key, value in bindings.items())):
            raise ValueError("Cached comparison inputs or completion receipt changed")
        if (evidence.get("candidate", {}).get("id") != candidate
                or not any(item["runtime"]["threshold"] == threshold
                    and item["runtime"]["consecutive_frames"] == consecutive
                    for item in evidence.get("comparisons", []))):
            raise ValueError("Choose a candidate and setting measured by this comparison")
    elif (threshold != package["runtime"]["threshold"]
            or consecutive != package["runtime"]["consecutive_frames"]):
        raise ValueError("Compare changed detection settings before selecting them")
    if (threshold == package["runtime"]["threshold"]
            and consecutive == package["runtime"]["consecutive_frames"]):
        return select_candidate(project, bundle, evaluation, output, candidate)
    # There is exactly one explicit configuration, so the legacy tune helper
    # cannot silently choose another threshold or consecutive-frame count.
    return tune(project, bundle, evaluation, output, [threshold], [consecutive], candidate)


def select_runtime(project, bundle):
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Runtime is frozen; start a versioned experiment")
    package = verify_bundle(bundle)
    if package["wakeword"] != config["wakeword"] or not (package.get("gate_selection_sha256") or package.get("candidate_selection_evidence")):
        raise ValueError("Use this word's Development-selected bundle")
    config["runtime"] = package["runtime"]
    from .project import validate_project
    validate_project(config)
    write_json(project, config)
    return {"runtime": config["runtime"], "Final_frozen": False}
