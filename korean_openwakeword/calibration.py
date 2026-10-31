"""Automatic Development gate recommendations from immutable frame scores.

Only detector decisions are searched. The acoustic network, source reports and
model files are never changed. Point estimates suffice for ranking; expensive
bootstrap intervals are not recomputed hundreds of times for identical calls.
"""
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from .common import append_row, digest, identity, iter_rows, now, read_json, write_json, writer_lock
from .data import CATEGORIES
from .evaluate import adoption, adoption_criteria
from .project import load_project
from .storage import project_storage


METHOD = "joint_cached_hard_conditions_v2"
COARSE_THRESHOLDS = tuple([i / 20 for i in range(1, 20)] + [.975, .99, .995, .999])
CONSECUTIVE_COUNTS = (1, 2, 3, 4, 5, 6)
HARD_CONDITIONS = tuple(
    [(reverb, category, snr) for reverb in ("none", "moderate", "strong")
     for category in CATEGORIES for snr in (10, 5)]
    + [(reverb, "clean", None) for reverb in ("moderate", "strong")])
SEARCH_POLICY = {
    "method": METHOD,
    "starting_threshold": .5,
    "starting_consecutive_frames": 1,
    "coarse_thresholds": list(COARSE_THRESHOLDS),
    "consecutive_frames": list(CONSECUTIVE_COUNTS),
    "refinement": "Two rounds per consecutive count: three interior quartile probes between that count's immediate tested neighbours of its best threshold (open-range boundaries 0.000001/0.999999)",
    "refinement_rounds": 2,
    "hard_conditions": [{"reverb_condition": reverb, "category": category, "snr_db": snr}
                        for reverb, category, snr in HARD_CONDITIONS],
    "objective": "Equal-weight mean of all 20 hard-condition recalls; dry Clean and SNR0 are excluded; missing conditions prevent a recommendation",
    "selection_order": ["clean/false-activation/latency/duration protections and complete hard-condition coverage passed", "clean protection passed", "false-activation protection passed", "lower false activations/hour when its protection is missed", "latency protection passed", "negative duration protection passed", "complete hard-condition coverage", "higher hard-condition mean recall", "higher worst hard-condition recall", "lower false activations/hour", "lower primary latency p95", "fewer consecutive frames", "lower threshold"],
    "no_feasible_candidate": "Show the conservative tradeoff separately; preserve the original runtime for microphone testing instead of silently applying the tradeoff",
    "scope": "Best among this declared finite grid; not a global optimum or independent Final result",
}


def _point_statistics(success, latency):
    count = len(success)
    values = latency[success]
    return {"calls": count, "successes": int(np.count_nonzero(success)),
            "recall": float(np.mean(success)) if count else None,
            "latency_p50_s": float(np.quantile(values, .5)) if len(values) else None,
            "latency_p95_s": float(np.quantile(values, .95)) if len(values) else None}


class PreparedScores:
    """Small metadata plus one vectorized score pass per detector setting.

    Rolling minima encode consecutive qualification. Event extraction obeys
    the same rising-edge, warmup and cooldown rules as EventDetector. Calls
    normally have one event window; overlapping/multiple calls use the exact
    earliest-unmatched assignment rule rather than counting one event twice.
    """
    def __init__(self, clips, offsets, runtime):
        self.rows = [{key: deepcopy(row[key]) for key in
                     ("sample_id", "kind", "category", "snr_db", "reverb_condition", "calls", "eligible_seconds")
                     if key in row} for row in clips]
        self.offsets = np.asarray(offsets, dtype=np.int64)
        lengths = np.diff(self.offsets)
        self.session = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        self.local = np.arange(int(self.offsets[-1]), dtype=np.int64) - self.offsets[self.session]
        self.eligible = self.local * .08 + 1e-9 >= runtime["warmup_s"]
        self.first_eligible = self.offsets[:-1] + max(0, int(math.ceil((runtime["warmup_s"] - 1e-9) / .08)))
        self.first_eligible = self.first_eligible[self.first_eligible < self.offsets[1:]]
        self.runtime = runtime
        self.negative = np.asarray([row["kind"] == "negative" for row in self.rows], dtype=bool)
        self.negative_hours = sum(row["eligible_seconds"] for row in self.rows if row["kind"] == "negative") / 3600
        call_session, call_start, call_end, self.calls = [], [], [], []
        self.multiple = {}
        for index, row in enumerate(self.rows):
            if row["kind"] != "positive":
                continue
            positions = []
            for call in row.get("calls", []):
                positions.append(len(self.calls))
                self.calls.append(call)
                call_session.append(index)
                call_start.append(self.offsets[index] + max(0, int(math.ceil((call["start_s"] - 1e-9) / .08 - 1))))
                call_end.append(min(self.offsets[index + 1] - 1, self.offsets[index] + int(math.floor((call["end_s"] + runtime["call_postroll_s"] + 1e-9) / .08 - 1))))
            if len(positions) > 1:
                self.multiple[index] = positions
        self.call_session = np.asarray(call_session, dtype=np.int64)
        self.call_start = np.asarray(call_start, dtype=np.int64)
        self.call_end = np.asarray(call_end, dtype=np.int64)
        self.end_seconds = np.asarray([call["end_s"] for call in self.calls], dtype=np.float64)
        self.clean, self.primary = [], []
        cells, hard_cells = defaultdict(list), defaultdict(list)
        for position, session in enumerate(self.call_session):
            row = self.rows[session]
            dry = not row.get("reverb_condition") or row.get("reverb_condition") == "none"
            self.clean.append(row["category"] == "clean" and dry)
            self.primary.append(row["category"] == "clean" or row.get("snr_db") in (20, 15, 10, 5))
            if dry:
                cells[f"{row['category']}:{row.get('snr_db')}"].append(position)
            hard_cells[(row.get("reverb_condition") or "none", row["category"], row.get("snr_db"))].append(position)
        self.clean = np.asarray(self.clean, dtype=bool)
        self.primary = np.asarray(self.primary, dtype=bool)
        self.cells = {key: np.asarray(value, dtype=np.int64) for key, value in cells.items()}
        self.hard_cells = {key: np.asarray(hard_cells[key], dtype=np.int64) for key in HARD_CONDITIONS}

    def rolling_scores(self, values, count):
        # float64 comparison is intentional: EventDetector compares float(score)
        # with a Python threshold. Rounding the threshold to float32 can change
        # an equality decision and is not an acceptable optimization.
        values = np.asarray(values, dtype=np.float64)
        result = values.copy()
        for shift in range(1, count):
            np.minimum(result[shift:], values[:-shift], out=result[shift:])
        result[self.local < count - 1] = -math.inf
        return result

    def events(self, rolling, threshold):
        qualified = rolling >= threshold
        starts = qualified.copy()
        starts[1:] &= ~qualified[:-1]
        starts[self.offsets[:-1][np.diff(self.offsets) > 0]] = qualified[self.offsets[:-1][np.diff(self.offsets) > 0]]
        starts[self.first_eligible] = qualified[self.first_eligible]
        starts &= self.eligible
        indices = np.flatnonzero(starts)
        if self.runtime["cooldown_s"] > 0 and len(indices):
            keep = np.ones(len(indices), dtype=bool)
            prior_session, prior_frame = -1, 0
            for position, frame in enumerate(indices):
                session = self.session[frame]
                if session != prior_session or (frame - prior_frame) * .08 + 1e-9 >= self.runtime["cooldown_s"]:
                    prior_session, prior_frame = session, frame
                else:
                    keep[position] = False
            indices = indices[keep]
        return indices

    def summary(self, values, threshold, count, candidate, rolling=None):
        indices = self.events(self.rolling_scores(values, count) if rolling is None else rolling, threshold)
        success = np.zeros(len(self.calls), dtype=bool)
        latency = np.full(len(self.calls), np.nan, dtype=np.float64)
        if len(indices) and len(self.calls):
            positions = np.searchsorted(indices, self.call_start)
            valid = positions < len(indices)
            matched = indices[np.minimum(positions, len(indices) - 1)]
            valid &= matched <= self.call_end
            valid &= self.session[matched] == self.call_session
            success[valid] = True
            latency[valid] = (self.local[matched[valid]] + 1) * .08 - self.end_seconds[valid]
            for session, calls in self.multiple.items():
                success[calls] = False
                latency[calls] = np.nan
                next_call, active = 0, []
                lower, upper = np.searchsorted(indices, self.offsets[session:session + 2])
                for frame in indices[lower:upper]:
                    timestamp = (self.local[frame] + 1) * .08
                    while next_call < len(calls) and self.calls[calls[next_call]]["start_s"] <= timestamp + 1e-9:
                        active.append(calls[next_call])
                        next_call += 1
                    active = [position for position in active if timestamp <= self.calls[position]["end_s"] + self.runtime["call_postroll_s"] + 1e-9]
                    chosen = next((position for position in active if not success[position]), None)
                    if chosen is not None:
                        success[chosen], latency[chosen] = True, timestamp - self.calls[chosen]["end_s"]
        conditions = {key: _point_statistics(success[positions], latency[positions]) for key, positions in self.cells.items()}
        noise = [stats["recall"] for key, stats in conditions.items()
                 if key.split(":")[0] in CATEGORIES and key.split(":")[1] in ("20", "15", "10", "5", "20.0", "15.0", "10.0", "5.0")]
        hard = {}
        for (reverb, category, snr), positions in self.hard_cells.items():
            key = f"{reverb}:{category}:{snr if snr is not None else 'none'}"
            # The objective needs recall only. Repeating latency quantiles for
            # every hard cell would add work without affecting any decision.
            successes = int(np.count_nonzero(success[positions]))
            hard[key] = {"calls": len(positions), "successes": successes,
                         "recall": successes / len(positions) if len(positions) else None,
                         "reverb_condition": reverb, "category": category, "snr_db": snr}
        missing = [key for key, stats in hard.items() if stats["recall"] is None]
        recalls = [stats["recall"] for stats in hard.values() if stats["recall"] is not None]
        hard_summary = {"conditions": hard, "cell_mean": float(np.mean(recalls)) if not missing else None,
                        "worst_recall": min(recalls) if not missing else None,
                        "measured_cells": len(recalls), "expected_cells": len(HARD_CONDITIONS),
                        "complete": not missing, "missing_conditions": missing}
        false = int(np.count_nonzero(self.negative[self.session[indices]]))
        return {"id": candidate["id"], "kind": candidate["kind"],
                "clean": _point_statistics(success[self.clean], latency[self.clean]),
                "primary": _point_statistics(success[self.primary], latency[self.primary]),
                "all_positive": _point_statistics(success, latency),
                "conditions": conditions, "noise_cell_mean": float(np.mean(noise)) if noise else None,
                "noise_cells_measured": len(noise),
                "hard_conditions": hard_summary,
                "negative": {"events": false, "hours": self.negative_hours,
                             "fa_per_hour": false / self.negative_hours if self.negative_hours else None}}


def _rank(item):
    summary, runtime = item["metrics"], item["runtime"]
    checks = item["constraint_checks"]
    hard = summary["hard_conditions"]
    false, latency = summary["negative"]["fa_per_hour"], summary["primary"]["latency_p95_s"]
    # Only protected, fully measured candidates can become recommendations.
    # A no-pass tradeoff stays conservative and is never applied automatically.
    false_failure = 0 if checks["false_activations"] else (false if false is not None else math.inf)
    return (all(checks.values()),
            checks["clean"], checks["false_activations"], -false_failure, checks["latency"], checks["negative_duration"],
            checks["hard_conditions"],
            hard["cell_mean"] if hard["cell_mean"] is not None else -1,
            hard["worst_recall"] if hard["worst_recall"] is not None else -1,
            -false if false is not None else -math.inf,
            -latency if latency is not None else -math.inf,
            -runtime["consecutive_frames"], -runtime["threshold"])


def _constraint_checks(metrics, targets):
    """Keep no-pass fallbacks conservative instead of rewarding always-high gates."""
    clean, false, latency = metrics["clean"]["recall"], metrics["negative"]["fa_per_hour"], metrics["primary"]["latency_p95_s"]
    return {"clean": clean is not None and clean >= targets["clean_recall"],
            "false_activations": false is not None and false <= targets["fa_per_hour"],
            "latency": latency is not None and latency <= targets["latency_p95_s"],
            "negative_duration": metrics["negative"]["hours"] >= targets["minimum_negative_hours"],
            "hard_conditions": metrics["hard_conditions"]["complete"]}


def _recommendation_reasons(checks, targets):
    reasons = {"clean": "Clean recall below target or unmeasured",
               "false_activations": "FA/hour target failed or unmeasured",
               "latency": "Latency p95 target failed or unmeasured",
               "negative_duration": f"Development FP duration below {targets['minimum_negative_hours']:g}h",
               "hard_conditions": "Hard noise/reverberation conditions incomplete"}
    return [reason for key, reason in reasons.items() if not checks[key]]


def _bindings(project, bundle, evaluation):
    return {"project_config_sha256": digest(project),
            "source_bundle_manifest_sha256": digest(Path(bundle) / "manifest.json"),
            "source_report_sha256": digest(Path(evaluation) / "report.json"),
            "source_clips_sha256": digest(Path(evaluation) / "clips.json"),
            "source_score_cache_sha256": digest(Path(evaluation) / "scores.npz")}


def read_recommendation(project, bundle, evaluation, output, verify_inputs=True):
    """Validate completed recommendations without trusting an unbound UI path."""
    directory = Path(output)
    result, receipt = read_json(directory / "recommendation.json"), read_json(directory / "receipt.json")
    if verify_inputs:
        bindings = _bindings(project, bundle, evaluation)
    else:
        # Display-only callers already validated the frozen evaluation and
        # invalidate their caches on source file identity changes. Do not hash
        # hundreds of megabytes of clips at every browser status poll.
        report = read_json(Path(evaluation) / "report.json")
        completion = read_json(Path(evaluation) / "receipt.json")
        bindings = {"project_config_sha256": digest(project),
                    "source_bundle_manifest_sha256": digest(Path(bundle) / "manifest.json"),
                    "source_report_sha256": digest(Path(evaluation) / "report.json"),
                    "source_clips_sha256": completion.get("clips_sha256"),
                    "source_score_cache_sha256": report.get("score_cache_sha256")}
        if (completion.get("report_sha256") != bindings["source_report_sha256"]
                or completion.get("bundle_sha256") != bindings["source_bundle_manifest_sha256"]):
            raise ValueError("Automatic detection recommendation source completion receipt changed")
    if (result.get("schema_version") != 1 or result.get("method") != METHOD
            or result.get("bindings") != bindings or receipt.get("bindings") != bindings
            or receipt.get("recommendation_sha256") != digest(directory / "recommendation.json")
            or result.get("search_policy") != SEARCH_POLICY
            or result.get("stage") != "development" or result.get("Final_used_for_selection") is not False):
        raise ValueError("Automatic detection recommendation inputs or receipt changed")
    return result


def load_recommendation(project, bundle, evaluation, output, verify_inputs=True):
    """Load the immutable sidecar; cached servers may skip repeated input hashes.

    Callers using verify_inputs=False must separately invalidate their cache
    when any source file identity changes. The published sidecar itself is
    still checked against its completion receipt on every call.
    """
    return read_recommendation(project, bundle, evaluation, output, verify_inputs=verify_inputs)


@project_storage
def recommend(project, bundle, evaluation, output, progress_callback=None):
    """Recommend one jointly searched gate for every single and mean candidate.

    Interrupted searches resume only matching, completed comparison records.
    A completed directory is immutable. New source evidence requires a new
    output directory. GPU cache recommendations describe that exact numerical
    protocol; they do not certify CPU microphone/deployment decisions.
    """
    from .selection import verified_evaluation
    root, config = load_project(project)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; start a versioned experiment before automatic recommendation")
    bundle, evaluation, output = map(lambda value: Path(value).resolve(), (bundle, evaluation, output))
    if any(output == source or output.is_relative_to(source) or source.is_relative_to(output) for source in (bundle, evaluation)):
        raise ValueError("Recommendation output must not modify or contain the source bundle/evaluation")
    bindings = _bindings(project, bundle, evaluation)
    if (output / "receipt.json").exists():
        result = read_recommendation(project, bundle, evaluation, output)
        return {"recommendation": str(output / "recommendation.json"), "receipt": str(output / "receipt.json"),
                "candidates": result["candidates"], "network_rerun": False, "selection_applied": False}
    root, config, package, report, clips, scores, offsets, candidates = verified_evaluation(project, bundle, evaluation)
    prepared = PreparedScores(clips, offsets, package["runtime"])
    if _bindings(project, bundle, evaluation) != bindings:
        raise ValueError("Automatic detection recommendation inputs changed during verification")
    protocol = report.get("execution", {}).get("scoring_protocol", "reference_cpu_session_v1")
    cpu = report.get("execution", {}).get("backend", "cpu") == "cpu" and protocol == "reference_cpu_session_v1"
    declaration = {"bindings": bindings, "search_policy": SEARCH_POLICY,
                   "candidates": candidates, "adoption_criteria": adoption_criteria(config)}
    journal = output / "comparisons.jsonl"
    comparisons = defaultdict(dict)
    with writer_lock(output):
        declaration_path = output / "search.json"
        if declaration_path.exists():
            if read_json(declaration_path) != declaration:
                raise ValueError("Resumable detection search inputs or policy changed; use a new directory")
        elif any(path.name != ".writer.lock" for path in output.iterdir()):
            raise ValueError("Recommendation output is not a matching resumable search")
        else:
            write_json(declaration_path, declaration)
        for row in iter_rows(journal):
            if (row.get("candidate") not in {candidate["id"] for candidate in candidates}
                    or row.get("checksum") != identity({key: value for key, value in row.items() if key != "checksum"})):
                raise ValueError("Resumable detection comparison record changed")
            entry = row["entry"]
            comparisons[row["candidate"]][(entry["runtime"]["threshold"], entry["runtime"]["consecutive_frames"])] = entry
        selected = []
        total_coarse = len(candidates) * len(COARSE_THRESHOLDS) * len(CONSECUTIVE_COUNTS)
        completed = 0
        for column, candidate in enumerate(candidates):
            cache = comparisons[candidate["id"]]
            # Replay the declared query sequence on resume. Loaded records may
            # include a partly finished refinement; those future probes must
            # not influence an earlier round when reconstructing its bounds.
            visited = set()
            values = scores[:, column]
            rolling = {count: prepared.rolling_scores(values, count) for count in CONSECUTIVE_COUNTS}
            def compare(threshold, count):
                nonlocal completed
                key = (threshold, count)
                if key not in cache:
                    runtime = dict(package["runtime"], threshold=threshold, consecutive_frames=count)
                    metrics = prepared.summary(values, threshold, count, candidate, rolling=rolling[count])
                    entry = {"runtime": runtime, "metrics": metrics, "adoption": adoption([metrics], config, candidate["id"]),
                             "constraint_checks": _constraint_checks(metrics, declaration["adoption_criteria"])}
                    row = {"candidate": candidate["id"], "entry": entry}
                    append_row(journal, dict(row, checksum=identity(row)))
                    cache[key] = entry
                visited.add(key)
                completed += 1
                if progress_callback:
                    progress_callback({"phase": "runtime_recommendation", "candidate": candidate["id"],
                                       "done": completed, "completed": completed,
                                       "total": total_coarse + len(candidates) * (1 + SEARCH_POLICY["refinement_rounds"] * 3 * len(CONSECUTIVE_COUNTS)),
                                       "unit": "조합", "label": "모델별 감지 설정 비교",
                                       "threshold": threshold, "consecutive_frames": count})
                return cache[key]
            baseline = compare(.5, 1)
            for threshold in COARSE_THRESHOLDS:
                for count in CONSECUTIVE_COUNTS:
                    compare(threshold, count)
            for _ in range(SEARCH_POLICY["refinement_rounds"]):
                for count in CONSECUTIVE_COUNTS:
                    local = {threshold: cache[(threshold, frames)] for threshold, frames in visited if frames == count}
                    best = max(local.values(), key=_rank)
                    tested = sorted(local)
                    index = tested.index(best["runtime"]["threshold"])
                    low = tested[index - 1] if index > 0 else .000001
                    high = tested[index + 1] if index + 1 < len(tested) else .999999
                    for threshold in np.linspace(low, high, 5)[1:-1]:
                        compare(float(threshold), count)
            best = deepcopy(max((cache[key] for key in visited), key=_rank))
            available = all(best["constraint_checks"].values())
            best.update(id=candidate["id"], kind=candidate["kind"],
                        passed=best["adoption"]["passed"],
                        status="recommended" if available else "no_feasible_candidate",
                        recommendation_available=available,
                        protection_passed=all(value for key, value in best["constraint_checks"].items() if key != "hard_conditions"),
                        recommendation_reasons=_recommendation_reasons(best["constraint_checks"], declaration["adoption_criteria"]),
                        original_runtime=deepcopy(package["runtime"]),
                        baseline=baseline, tested_combinations=len(visited),
                        source_scoring_protocol=protocol, deployment_backend_verified=cpu,
                        verification_scope="reference_cpu_scores" if cpu else "cached_score_protocol_only")
            selected.append(best)
            print(f"Detection settings: {candidate['id']} threshold={best['runtime']['threshold']:g} frames={best['runtime']['consecutive_frames']} recommendation_available={available} targets_passed={best['passed']}", flush=True)
            del rolling
        if _bindings(project, bundle, evaluation) != bindings:
            raise ValueError("Automatic detection recommendation inputs changed during search")
        result = {"schema_version": 1, "method": METHOD, "stage": "development", "created_at": now(),
                  "bindings": bindings, "search_policy": SEARCH_POLICY,
                  "adoption_criteria": adoption_criteria(config), "candidates": selected,
                  "selection_applied": False, "network_rerun": False, "Final_used_for_selection": False}
        write_json(output / "recommendation.json", result)
        write_json(output / "receipt.json", {"schema_version": 1, "bindings": bindings,
                   "recommendation_sha256": digest(output / "recommendation.json"), "completed_at": now()})
    return {"recommendation": str(output / "recommendation.json"), "receipt": str(output / "receipt.json"),
            "candidates": selected, "network_rerun": False, "selection_applied": False}
