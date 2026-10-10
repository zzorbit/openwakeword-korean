"""Development-only gate comparison. Reuse exact scores; never rerun the network."""
from copy import deepcopy
from pathlib import Path
import math
import shutil
import numpy as np
from .common import digest, now, read_json, write_json, writer_lock
from .project import load_project
from .events import EventDetector, PositiveTracker, positive_statistics
from .evaluate import summarize, adoption
from .runtime import verify_bundle


def replay(clips, scores, offsets, model_ids, runtime):
    result = deepcopy(clips)
    for index, row in enumerate(result):
        session = {"session_id": row["sample_id"], "source_id": row.get("source_id", row["sample_id"]), "calls": row.get("calls", []), "provenance": row.get("provenance", "qwen_tts"), "voice_family_id": row.get("voice_family_id"), "category": row["category"], "speaker_id": row.get("speaker_id")}
        session.update({key: row[key] for key in ("distance_m", "direction", "microphone") if key in row})
        for j, call in enumerate(session["calls"]):
            call.setdefault("call_id", f"{row['sample_id']}:call{j}")
            call.setdefault("original_utterance_id", row.get("original_id", row["sample_id"]))
        row["results"] = {}
        frames = scores[offsets[index]:offsets[index + 1]]
        for column, key in enumerate(model_ids):
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
    clean = {r.get("original_id", r["sample_id"]): r for r in result if r["category"] == "clean"}
    for row in result:
        baseline = clean.get(row.get("original_id", row["sample_id"]))
        for key, value in row["results"].items():
            value["clean_score_drop"] = baseline["results"][key]["max_score"] - value["max_score"] if baseline else None
    return result


def tune(project, bundle, evaluation, output, thresholds, consecutive=(2,)):
    _, config = load_project(project)
    evaluation, bundle, output = map(lambda p: Path(p).resolve(), (evaluation, bundle, output))
    package, report, receipt = verify_bundle(bundle), read_json(evaluation / "report.json"), read_json(evaluation / "receipt.json")
    if report["stage"] != "development" or report["wakeword"] != config["wakeword"]:
        raise ValueError("Only this word's Development scores can select a gate")
    if digest(bundle / "manifest.json") != report["bundle_manifest_sha256"] or digest(evaluation / "report.json") != receipt["report_sha256"] or digest(evaluation / "clips.json") != receipt["clips_sha256"] or digest(evaluation / "scores.npz") != report["score_cache_sha256"]:
        raise ValueError("Evaluation/model/score cache changed")
    if not thresholds or any(not math.isfinite(t) or not 0 < t < 1 for t in thresholds) or any(type(n) is not int or n < 1 for n in consecutive):
        raise ValueError("Declare finite thresholds in (0,1) and positive frame counts")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Tuning output is immutable")
    clips = read_json(evaluation / "clips.json")
    with np.load(evaluation / "scores.npz", allow_pickle=False) as cache:
        scores, offsets, ids = cache["scores"], cache["offsets"], cache["model_ids"].tolist()
    comparisons, best = [], None
    with writer_lock(output):
        for threshold in dict.fromkeys(thresholds):
            for count in dict.fromkeys(consecutive):
                runtime = dict(package["runtime"], threshold=threshold, consecutive_frames=count)
                rows = replay(clips, scores, offsets, ids, runtime)
                summaries = summarize(rows, config)
                decision = adoption(summaries, config)
                candidate = next((s for s in summaries if s["id"] == "mean"), summaries[0])
                key = (decision["passed"], candidate["clean"]["recall"] or 0, candidate["noise_cell_mean"] or 0, -(candidate["negative"]["fa_per_hour"] if candidate["negative"]["fa_per_hour"] is not None else math.inf))
                comparisons.append({"runtime": runtime, "adoption": decision, "models": summaries})
                if best is None or key > best[0]:
                    best = (key, runtime, rows, summaries, decision)
                print(f"Development gate: threshold={threshold} frames={count} adoption={decision['passed']}", flush=True)
        write_json(output / "comparison.json", {"stage": "development", "created_at": now(), "source_report_sha256": digest(evaluation / "report.json"), "source_score_cache_sha256": digest(evaluation / "scores.npz"), "comparisons": comparisons, "selection_order": ["all admission targets passed", "clean recall", "balanced noise recall", "lower FA/hour"], "scope": "Declared gates only; no network retraining and no Final tuning"})
        selected_bundle = output / "bundle"
        shutil.copytree(bundle, selected_bundle)
        package.update(runtime=best[1], gate_selection_sha256=digest(output / "comparison.json"), stage="development_gate_selected")
        write_json(selected_bundle / "manifest.json", package)
        selected_report = dict(report, runtime=best[1], models=best[3], adoption=best[4], bundle_manifest_sha256=digest(selected_bundle / "manifest.json"), gate_selection_sha256=digest(output / "comparison.json"), created_at=now())
        selected_dir = output / "evaluation"
        write_json(selected_dir / "clips.json", best[2]); write_json(selected_dir / "report.json", selected_report)
        shutil.copyfile(evaluation / "scores.npz", selected_dir / "scores.npz")
        write_json(selected_dir / "receipt.json", {"report_sha256": digest(selected_dir / "report.json"), "clips_sha256": digest(selected_dir / "clips.json"), "bundle_sha256": digest(selected_bundle / "manifest.json"), "completed_at": now()})
    return {"bundle": str(selected_bundle), "evaluation": str(selected_dir), "runtime": best[1], "adoption": best[4], "smoke_test_only": report["smoke_test_only"]}


def select_runtime(project, bundle):
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Runtime is frozen; start a versioned experiment")
    package = verify_bundle(bundle)
    if package["wakeword"] != config["wakeword"] or not package.get("gate_selection_sha256"):
        raise ValueError("Use this word's Development-selected bundle")
    config["runtime"] = package["runtime"]
    from .project import validate_project
    validate_project(config)
    write_json(project, config)
    return {"runtime": config["runtime"], "Final_frozen": False}
