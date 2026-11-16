"""Read optional Studio telemetry and estimate time from matching progress samples."""

import math
import statistics
import time


def read_optional_progress(path, scope, view, snapshots, *, read_json, max_bytes=None):
    """Read display-only JSON; retain same-task data through a sharing denial."""
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            snapshots.pop(path, None)
            return {}
        record = read_json(path)
    except PermissionError:
        view["progress_pending"] = True
        previous_scope, previous = snapshots.get(path, (None, {}))
        return previous if previous_scope == scope else {}
    except (OSError, ValueError):
        snapshots.pop(path, None)
        return {}
    if not isinstance(record, dict):
        snapshots.pop(path, None)
        return {}
    snapshots[path] = scope, record
    return record

def estimate_progress_eta(state, job, progress, rates_by_project):
    """Use the worker's measured rate, or a stable window for older producers."""
    project = state["id"]

    def valid_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    done, total = progress.get("done"), progress.get("total")
    counter = progress.get("rate_done", done)
    phase = progress.get("rate_phase", progress.get("phase") or progress.get("label"))
    excluded = {"retry", "download_many", "feature_validation", "device_selection",
                "model_initialization", "gpu_cache_loading", "preparation_validation",
                "preparation_finalizing", "validation", "finalizing"}
    if (state.get("status") != "running" or progress.get("eta_eligible") is False
            or phase in excluded or progress.get("unit") == "완료 단계"
            or not all(valid_number(value) for value in (done, total, counter))
            or total <= 0 or not 0 <= done < total or counter < 0):
        rates_by_project.pop(project, None)
        return None
    if "worker_rate" in progress:
        rates_by_project.pop(project, None)
        rate = progress["worker_rate"]
        if not isinstance(rate, dict):
            return None
        speed, measured_at = rate.get("per_second"), rate.get("measured_at")
        if (not valid_number(speed) or speed <= 0 or not valid_number(measured_at)
                or not -1 <= time.time() - measured_at <= 60):
            return None
        estimate = (total - done) / speed
        return estimate if math.isfinite(estimate) else None
    stamp = time.monotonic()
    scope = (job.get("task_id"), job.get("stage"), phase, total, progress.get("unit"))
    window = rates_by_project.get(project)
    if (not isinstance(window, dict) or window.get("scope") != scope
            or counter < window["points"][-1][1] or stamp < window["points"][-1][0]
            or stamp - window["points"][-1][0] > 90):
        window = rates_by_project[project] = {"scope": scope, "points": [(stamp, counter)]}
        return None
    points = window["points"]
    if counter > points[-1][1]:
        if points[-1][1] == 0:
            # A long wait for the first output is startup, not per-item cost.
            points[:] = [(stamp, counter)]
        else:
            points.append((stamp, counter))
            points[:] = [point for point in points[-8:] if stamp - point[0] <= 90]
    if len(points) < 4 or points[-1][0] - points[0][0] < 15 or stamp - points[-1][0] > 30:
        return None
    rates = [(right[1] - left[1]) / (right[0] - left[0])
             for left, right in zip(points, points[1:]) if right[0] > left[0]]
    if len(rates) < 3 or min(rates) <= 0 or max(rates) > 2 * min(rates):
        return None
    estimate = (total - done) / statistics.median(rates)
    return estimate if math.isfinite(estimate) and estimate >= 0 else None
