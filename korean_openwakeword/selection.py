"""Explicit Development candidate selection without rerunning audio or networks."""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np

from .common import bound_file, digest, now, read_json, write_json, writer_lock
from .evaluate import adoption, adoption_criteria, candidate_adoptions
from .project import load_project
from .runtime import bundle_candidates, deployment_candidate, verify_bundle
from .storage import project_storage
from .cache_analysis import FrozenClipRows, RestrictedClipRows, write_clip_array


def verified_evaluation(project, bundle, evaluation):
    """Read immutable Development evidence and its exact bound score columns."""
    root, config = load_project(project)
    bundle, evaluation = Path(bundle).resolve(), Path(evaluation).resolve()
    package = verify_bundle(bundle)
    report = read_json(evaluation / "report.json")
    receipt = read_json(evaluation / "receipt.json")
    if package["wakeword"] != config["wakeword"] or report.get("wakeword") != config["wakeword"]:
        raise ValueError("Candidate/evaluation word differs from this project")
    if report.get("stage") != "development":
        raise ValueError("Only completed Development evidence can select a candidate")
    expected = {"report_sha256": digest(evaluation / "report.json"),
                "clips_sha256": digest(evaluation / "clips.json"),
                "bundle_sha256": digest(bundle / "manifest.json")}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Development completion receipt changed")
    if report.get("bundle_manifest_sha256") != expected["bundle_sha256"]:
        raise ValueError("Development evaluation belongs to a different bundle")
    if report.get("runtime") != package["runtime"]:
        raise ValueError("Development runtime does not match this bundle")
    if report.get("score_cache_sha256") != digest(evaluation / "scores.npz"):
        raise ValueError("Development score cache changed")
    candidates = bundle_candidates(package)
    if report.get("candidate_definitions", candidates) != candidates:
        raise ValueError("Candidate IDs do not bind the actual model members")
    if report.get("deployment_candidate", deployment_candidate(package)) != deployment_candidate(package):
        raise ValueError("Development deployment identity differs from its source bundle")
    if report.get("adoption_criteria", adoption_criteria(config)) != adoption_criteria(config):
        raise ValueError("Development adoption criteria differ from the project; preserve the evaluated targets")
    count = report.get("sessions")
    if type(count) is not int or count < 0:
        raise ValueError("Development report needs its exact session count")
    clips = FrozenClipRows(evaluation / "clips.json", count)
    with np.load(evaluation / "scores.npz", allow_pickle=False) as cache:
        scores, offsets = cache["scores"], cache["offsets"]
        ids = cache["model_ids"].tolist()
    if ids != [row["id"] for row in candidates] or scores.ndim != 2 or scores.shape[1] != len(ids):
        raise ValueError("Cached score columns do not match candidate identities")
    if (scores.dtype != np.float32
            or offsets.dtype.kind not in "iu" or offsets.ndim != 1
            or len(offsets) != len(clips) + 1 or offsets[0] != 0
            or offsets[-1] != len(scores) or np.any(np.diff(offsets) < 0)):
        raise ValueError("Invalid cached Development scores or offsets")
    for start in range(0, len(scores), 65536):
        block = scores[start:start + 65536]
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError("Invalid cached Development scores or offsets")
    if len({row["id"] for row in report["models"]}) != len(ids) or {row["id"] for row in report["models"]} != set(ids):
        raise ValueError("Development candidate summaries are incomplete")
    for index, row in enumerate(clips):
        if row.get("provenance") == "human" and row.get("kind") == "positive":
            raise ValueError("Actual human calls are Final-only, not candidate-selection data")
        if set(row.get("results", {})) != set(ids):
            raise ValueError("Development clips do not match cached candidate columns")
        duration = int(offsets[index + 1] - offsets[index]) * .08
        eligible = max(0, duration - package["runtime"]["warmup_s"])
        if (not np.isclose(row.get("duration_s", -1), duration, atol=1e-8, rtol=0)
                or not np.isclose(row.get("eligible_seconds", -1), eligible, atol=1e-8, rtol=0)):
            raise ValueError("Development duration does not match actual cached frames")
    return root, config, package, report, clips, scores, offsets, candidates


def copy_candidate_bundle(source, output, package, candidate, *, evidence=None):
    """Copy only the deployed heads and shared frontend; preserve model bytes."""
    source, output = Path(source), Path(output)
    result = deepcopy(package)
    wanted = {(entry["path"], entry["sha256"]) for entry in candidate["members"]}
    result["members"] = [m for m in package["members"] if (m["path"], m["sha256"]) in wanted]
    if len(result["members"]) != len(wanted):
        raise ValueError("Selected candidate members are missing from source bundle")
    result["aggregation"] = candidate["aggregation"]
    # Old selection evidence belongs to the old manifest; the new evidence is explicit.
    result.pop("candidate_selection_evidence", None)
    result.pop("candidate_selection_sha256", None)
    result["selected_candidate"] = deployment_candidate(result)
    for entry in result["members"] + result["frontend"]:
        destination = output / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bound_file(source, entry), destination)
        if digest(destination) != entry["sha256"]:
            raise ValueError("Copied candidate asset hash differs")
    notices = [path for pattern in ("LICENSE*", "NOTICE*", "THIRD_PARTY_NOTICES*", "COPYING*")
               for path in source.glob(pattern) if path.is_file()]
    licenses = source / "licenses"
    if licenses.is_dir():
        notices.extend(path for path in licenses.rglob("*") if path.is_file())
    for path in notices:
        if not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Candidate license files must remain inside the portable bundle")
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    if evidence is not None:
        write_json(output / "selection.json", evidence)
        result["candidate_selection_evidence"] = {"path": "selection.json", "sha256": digest(output / "selection.json")}
        result["candidate_selection_sha256"] = digest(output / "selection.json")
    write_json(output / "manifest.json", result)
    return verify_bundle(output)


def restrict_evaluation(report, clips, scores, candidates, chosen, config):
    """Reuse exact score bits and statistics for the heads actually packaged."""
    keep = candidates if chosen["kind"] == "ensemble" else [chosen]
    ids = [row["id"] for row in keep]
    columns = [next(i for i, row in enumerate(candidates) if row["id"] == key) for key in ids]
    rows = RestrictedClipRows(clips, ids)
    result = deepcopy(report)
    result["models"] = [row for row in result["models"] if row["id"] in ids]
    result["candidate_definitions"] = deepcopy(keep)
    result["selected_candidate"] = chosen["id"]
    result["deployment_candidate"] = deepcopy(chosen)
    result["candidate_adoptions"] = candidate_adoptions(result["models"], config)
    result["adoption"] = adoption(result["models"], config, chosen["id"])
    result["adoption_criteria"] = adoption_criteria(config)
    return result, rows, np.ascontiguousarray(scores[:, columns])


@project_storage
def select_candidate(project, bundle, evaluation, output, candidate):
    """Package an explicitly chosen single or mean from completed Development."""
    root, config, package, report, clips, scores, offsets, candidates = verified_evaluation(project, bundle, evaluation)
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; start a versioned experiment before candidate selection")
    chosen = next((row for row in candidates if row["id"] == candidate), None)
    if chosen is None:
        raise ValueError(f"Unknown evaluated candidate: {candidate}")
    output, bundle, evaluation = Path(output).resolve(), Path(bundle).resolve(), Path(evaluation).resolve()
    if output == bundle or output == evaluation or output.is_relative_to(bundle) or output.is_relative_to(evaluation):
        raise ValueError("Selection output must not modify the source bundle/evaluation")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Candidate selection output is immutable; choose a new directory")
    selected_report, selected_clips, selected_scores = restrict_evaluation(report, clips, scores, candidates, chosen, config)
    evidence = {"schema_version": 1, "created_at": now(), "stage": "development",
                "selection": "explicit_candidate_parameter", "candidate": deepcopy(chosen),
                "source_bundle_manifest_sha256": digest(bundle / "manifest.json"),
                "source_report_sha256": digest(evaluation / "report.json"),
                "source_clips_sha256": digest(evaluation / "clips.json"),
                "source_score_cache_sha256": digest(evaluation / "scores.npz"),
                "project_config_sha256": digest(project), "runtime": deepcopy(package["runtime"]),
                "adoption": selected_report["adoption"], "adoption_criteria": selected_report["adoption_criteria"],
                "source_models": deepcopy(report["models"]),
                "Final_used_for_selection": False, "network_rerun": False}
    with writer_lock(output):
        write_json(output / "selection.json", evidence)
        selected_bundle = copy_candidate_bundle(bundle, output / "bundle", package, chosen, evidence=evidence)
        selected_bundle["stage"] = "development_candidate_selected"
        write_json(output / "bundle" / "manifest.json", selected_bundle)
        selected_report.update(created_at=now(), bundle_manifest_sha256=digest(output / "bundle" / "manifest.json"),
                               source_report_sha256=evidence["source_report_sha256"],
                               candidate_selection_sha256=digest(output / "selection.json"))
        selected_dir = output / "evaluation"
        selected_dir.mkdir()
        np.savez_compressed(selected_dir / "scores.npz", scores=selected_scores, offsets=offsets,
                            model_ids=np.asarray([row["id"] for row in selected_report["candidate_definitions"]]))
        selected_report["score_cache_sha256"] = digest(selected_dir / "scores.npz")
        write_clip_array(selected_dir / "clips.json", selected_clips)
        write_json(selected_dir / "report.json", selected_report)
        completion = {"report_sha256": digest(selected_dir / "report.json"),
                      "clips_sha256": digest(selected_dir / "clips.json"),
                      "bundle_sha256": digest(output / "bundle" / "manifest.json"), "completed_at": now()}
        write_json(selected_dir / "receipt.json", completion)
        write_json(output / "receipt.json", {**completion, "selection_sha256": digest(output / "selection.json"),
                                            "score_cache_sha256": selected_report["score_cache_sha256"]})
    return {"bundle": str(output / "bundle"), "evaluation": str(selected_dir), "candidate": chosen["id"],
            "adoption": selected_report["adoption"], "runtime": package["runtime"],
            "smoke_test_only": report.get("smoke_test_only", False), "network_rerun": False}
