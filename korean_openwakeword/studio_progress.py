"""Read Studio progress and project published preparation onto download rows."""

from functools import lru_cache
from importlib.resources import files
import math
from pathlib import Path
import re
import statistics
import time

from .common import identity, read_json


_BASE_DOWNLOAD_PREPARATIONS = {"zeroth": "zeroth", "audioset": "audioset",
                               "official_music": "fma", "mit_rir": "mit_rir"}
_BASE_ARTIFACTS = {"source_manifest": "train_sources.jsonl", "validation_reserve": "validation_reserve.jsonl",
                   "unused_manifest": "unused_sources.jsonl", "representative_queue": "representatives.jsonl",
                   "rejections": "rejected.json"}
_COMMON_DOWNLOAD_GROUPS = {"musan", "seoul", "fsd50k"}


def _metadata(read, path):
    try:
        value = read(path)
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _artifact_exists(binding, path):
    if not isinstance(binding, dict):
        return False
    checksum = binding.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        return False
    try:
        return Path(binding["path"]).resolve() == path and path.is_file()
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def _prepared_artifacts(receipt, directory, names):
    artifacts = receipt.get("artifacts")
    return isinstance(artifacts, dict) and all(
        _artifact_exists(artifacts.get(key), directory / filename) for key, filename in names.items())


def _base_download_prepared(directory, wakeword, read_metadata):
    receipt = _metadata(read_metadata, directory / "preparation_receipt.json")
    return (receipt.get("kind") == "public_audio_preparation_v1"
            and receipt.get("state") == "awaiting_human_review" and receipt.get("wakeword") == wakeword
            and receipt.get("output_directory") == str(directory) and receipt.get("target_met") is True
            and _artifact_exists(receipt.get("contract"), directory / "contract.json")
            and _prepared_artifacts(receipt, directory, _BASE_ARTIFACTS))


@lru_cache(maxsize=2)
def _download_evaluation_catalog(version):
    # Same static catalog identity as prepare_common_evaluation_materials.
    resources = files("korean_openwakeword") / "resources"
    material = read_json(resources / "evaluation_materials_v1.json")
    rir = read_json(resources / "evaluation_rirs_v1.json")
    scope = ({"evaluation_source_scope": "common_30h_only"} if version == 6
             else read_json(resources / "evaluation_backgrounds_v4.json"))
    catalog_id = identity([material, rir, scope])[:16]
    return catalog_id, identity(rir), rir["archive"], len(rir["assets"])


def _evaluation_download_prepared(directory, wakeword, version, catalog_id, read_metadata):
    receipt = _metadata(read_metadata, directory / "preparation.json")
    return (receipt.get("kind") == f"studio_evaluation_noise_preparation_v{version}"
            and receipt.get("catalog_id") == catalog_id and receipt.get("recipe_version") == version
            and receipt.get("state") == "awaiting_human_review" and receipt.get("wakeword") == wakeword
            and (version != 6 or receipt.get("evaluation_source_scope") == "common_30h_only")
            and _prepared_artifacts(receipt, directory, {"source_manifest": "evaluation_sources.jsonl",
                                                        "representative_queue": "representatives.jsonl"}))


def _rir_download_prepared(directory, catalog_sha256, archive_pin, source_count, read_metadata):
    receipt = _metadata(read_metadata, directory / "receipt.json")
    archive, coverage = receipt.get("archive"), receipt.get("coverage")
    return (receipt.get("kind") == "common_rir_preparation_v1" and receipt.get("catalog_sha256") == catalog_sha256
            and isinstance(archive, dict) and archive.get("sha256") == archive_pin["sha256"]
            and archive.get("bytes") == archive_pin["size_bytes"]
            and isinstance(coverage, dict) and coverage.get("rir_sources") == source_count
            and _artifact_exists(receipt.get("manifest"), directory / "rir_validation.jsonl"))


def prepared_download_groups(root, wakeword, files, evaluation_policy, progress, *, read_metadata):
    """Return groups whose verified transfers and preparation have both finished.

    ``progress`` must belong to the selected project's current download task.
    Receipts are display-only evidence; the worker owns full artifact validation.
    """
    rows_by_group = {}
    for row in files:
        group = row.get("group") or row.get("id")
        if group:
            rows_by_group.setdefault(group, []).append(row)
    transferred = {group for group, rows in rows_by_group.items()
                   if all(row.get("state") in ("complete", "reused") for row in rows)}
    prepared = set()
    for group in transferred:
        if group.startswith("automatic:") and all(
            row.get("unit") == "hours" and _valid_hours(row.get("prepared_hours"), row.get("total"))
            for row in rows_by_group[group]
        ):
            prepared.add(group)

    materials = Path(root).resolve() / ".studio" / "materials"
    version = (evaluation_policy or {}).get("evaluation_materials_recipe")
    zeroth_train = any(row.get("id") == "base:zeroth" for row in rows_by_group.get("zeroth", ()))
    for group in transferred.intersection(_BASE_DOWNLOAD_PREPARATIONS):
        if group != "zeroth" or zeroth_train:
            if _base_download_prepared(materials / _BASE_DOWNLOAD_PREPARATIONS[group], wakeword, read_metadata):
                prepared.add(group)

    if version in (5, 6) and transferred.intersection(_COMMON_DOWNLOAD_GROUPS | {"zeroth", "common_rir"}):
        catalog_id, rir_sha256, archive_pin, source_count = _download_evaluation_catalog(version)
        directory = materials / f"evaluation_v{version}_{catalog_id}"
        common_prepared = _evaluation_download_prepared(directory, wakeword, version, catalog_id, read_metadata)
        # The evaluation workflow starts the RIR check only after the joint set
        # of 30-hour background audio has returned successfully, including refill.
        common_prepared = common_prepared or (
            progress.get("stage") == "data_download"
            and progress.get("label") == "공통 잔향 원본 archive 준비·무결성 확인")
        if common_prepared:
            prepared.update(transferred.intersection(_COMMON_DOWNLOAD_GROUPS))
            if "zeroth" in transferred and not zeroth_train:
                prepared.add("zeroth")
        else:
            # Training and evaluation share one Zeroth archive and one UI row.
            prepared.discard("zeroth")
        if "common_rir" in transferred and _rir_download_prepared(
            directory / "rir", rir_sha256, archive_pin, source_count, read_metadata
        ):
            prepared.add("common_rir")
    return sorted(prepared)


def _valid_hours(actual, target):
    return (type(actual) in (int, float) and type(target) in (int, float)
            and math.isfinite(actual) and math.isfinite(target) and actual >= target > 0)


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
