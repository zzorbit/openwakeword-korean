"""Portable candidate bundles and hash-bound Final locks."""
from contextlib import contextmanager, suppress
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .common import _retry_windows_sharing, digest, now, read_json, resolved, write_json
from .project import load_project, validate_project
from .runtime import verify_bundle, bundle_candidates, deployment_candidate


def _require_empty_bundle_destination(output, occupied_error):
    try:
        if not any(output.iterdir()):
            return
    except FileNotFoundError:
        return
    except NotADirectoryError:
        pass
    raise ValueError(occupied_error)


def _read_json_snapshot(path):
    raw = _retry_windows_sharing(Path(path).read_bytes, read_access=True)
    return json.loads(raw.decode("utf-8-sig")), sha256(raw).hexdigest()


def _publish_bundle(staged, output, occupied_error):
    try:
        # Windows requires an absent rename target. Only an empty placeholder can be removed.
        with suppress(FileNotFoundError):
            _retry_windows_sharing(output.rmdir)
        _retry_windows_sharing(lambda: staged.rename(output))
    except OSError:
        # A concurrent publisher may have installed a complete bundle since the initial check.
        _require_empty_bundle_destination(output, occupied_error)
        raise


@contextmanager
def _staged_bundle_output(output, *, occupied_error="Bundle destination must be empty"):
    """Stage on the destination filesystem; publish a complete directory by rename."""
    _require_empty_bundle_destination(output, occupied_error)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".bundle-") as temporary:
        staged = Path(temporary) / "bundle"
        yield staged
        _publish_bundle(staged, output, occupied_error)


def package_run(project, run, output):
    project, run, output = (Path(path).resolve() for path in (project, run, output))
    config, project_sha256 = _read_json_snapshot(project)
    config = validate_project(config)
    with _staged_bundle_output(output) as staged:
        result = _stage_bundle(project.parent, config, run, staged, project_sha256)
        if digest(project) != project_sha256:
            raise ValueError("Project configuration changed during packaging")
    return result


def _stage_bundle(root, config, run, staged, project_sha256):
    members = []
    for i, seed in enumerate(config["training"]["seeds"], 1):
        folder = run / f"member{i}_seed{seed}"
        receipt = read_json(folder / "receipt.json")
        contract = read_json(folder / "contract.json")
        if contract["config"] != config:
            raise ValueError("Run/config binding changed")
        if receipt["state"] not in ("trained_not_final_evaluated", "smoke_trained_not_performance_validated"):
            raise ValueError("Uncompleted training member")
        destination = staged / "models" / f"member{i}_seed{seed}.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "model.onnx", destination)
        members.append({"path": destination.relative_to(staged).as_posix(),
                        "sha256": receipt["model_sha256"], "seed": seed, "smoke_test_only": contract["smoke_test_only"]})
    frontend = []
    from .frontend import PINNED_ASSETS
    for filename, expected in PINNED_ASSETS.items():
        source = resolved(root, config["frontend_dir"]) / filename
        destination = staged / "frontend" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        frontend.append({"path": "frontend/" + filename, "sha256": expected})
    manifest = {"schema_version": 1, "wakeword": config["wakeword"], "id": run.name,
        "members": members, "frontend": frontend,
        "aggregation": "single" if len(members) == 1 else "ordered_float64_mean_to_float32",
        "runtime": config["runtime"], "training": config["training"], "rounds": config["augmentation_rounds"],
        "created_at": now(), "stage": "candidate_not_final_validated", "project_config_sha256": project_sha256}
    manifest["selected_candidate"] = deployment_candidate(manifest)
    write_json(staged / "manifest.json", manifest)
    return verify_bundle(staged)


def freeze(project, bundle, development_report, output=None):
    root, config = load_project(project)
    bundle, development_report = Path(bundle).resolve(), Path(development_report).resolve()
    manifest = verify_bundle(bundle)
    report = read_json(development_report)
    completion = read_json(development_report.parent / "receipt.json")
    if completion.get("report_sha256") != digest(development_report) or completion.get("clips_sha256") != digest(development_report.parent / "clips.json") or completion.get("bundle_sha256") != digest(bundle / "manifest.json"):
        raise ValueError("Development completion receipt no longer binds this report/model")
    if report.get("stage") != "development" or report.get("bundle_manifest_sha256") != digest(bundle / "manifest.json"):
        raise ValueError("Freeze requires the exact candidate's Development report")
    if report.get("smoke_test_only") or any(m.get("smoke_test_only") for m in manifest["members"]):
        raise ValueError("A functional smoke model/report cannot authorize Final")
    if report.get("wakeword") != config["wakeword"] or manifest["runtime"] != config["runtime"]:
        raise ValueError("Word/runtime differs from the evaluated project setting")
    if report.get("runtime") != manifest["runtime"]:
        raise ValueError("Development runtime differs from the deployed bundle")
    deployed = deployment_candidate(manifest)
    if report.get("selected_candidate", report.get("adoption", {}).get("candidate")) != deployed["id"]:
        raise ValueError("Selected evaluation candidate differs from deployed members/aggregation; use select-candidate")
    if report.get("candidate_definitions", bundle_candidates(manifest)) != bundle_candidates(manifest):
        raise ValueError("Development candidate member/hash binding differs from deployed bundle")
    if report.get("deployment_candidate", deployed) != deployed:
        raise ValueError("Development deployment identity differs from deployed bundle")
    from .evaluate import adoption
    decision = adoption(report["models"], config, deployed["id"])
    if report.get("adoption", {}).get("candidate") != deployed["id"] or decision != report["adoption"]:
        raise ValueError("Development adoption does not describe the deployed candidate")
    # Evidence may be under target. Preserve the explicit judgement rather than fabricate a pass.
    if not decision["passed"]:
        raise ValueError("Selected configuration does not pass declared Development adoption criteria")
    from .final_integrity import freeze_input_bindings
    lock = {"schema_version": 2, "frozen_at": now(), "wakeword": config["wakeword"],
        "bundle": str(bundle), "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "development_report": str(development_report), "development_report_sha256": digest(development_report),
        "project_config_sha256": digest(project), "runtime": config["runtime"],
        "selected_candidate": deployed,
        "input_bindings": freeze_input_bindings(project, development_report),
        "independent_final_policy": "known_train_common_lineage_and_exact_pcm_v1",
        "generic_precomputed_original_mapping_available": False,
        "Final_used_for_selection": False}
    output = Path(output) if output else root / "final_lock.json"
    if output.exists():
        raise ValueError("Final lock is immutable; create a new versioned experiment")
    write_json(output, lock)
    return lock
