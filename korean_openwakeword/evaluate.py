"""One-pass whole-audio scoring for singles and their mean; truthful scope.

The same frame scores can be reused for declared Development gate experiments.
Final has one frozen gate and never grants tuning authority.
"""
from collections import defaultdict
from pathlib import Path
import math
import json
import time
import numpy as np
import soundfile as sf

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


def evaluation_binding(project, bundle, manifest, stage="development", candidate=None, smoke=False,
                       *, backend="cpu", gpu_batch_size=128, boundary_margin=.001, cpu_prefix=None):
    """Identity for resumable work; scoring still verifies all actual inputs.

    Reading this identity does not certify WAVs, model assets or a checkpoint.
    The evaluate entry point performs those checks before reusing any score.
    """
    directory = Path(bundle).resolve()
    package = read_json(directory / "manifest.json")
    model_ids = [row["id"] for row in bundle_candidates(package)]
    candidate = deployment_candidate(package)["id"] if candidate is None else candidate
    payload = {"project_sha256": digest(project), "bundle_sha256": digest(directory / "manifest.json"),
                     "manifest_sha256": digest(manifest), "stage": stage, "candidate": candidate,
                     "smoke": smoke, "model_ids": model_ids, "scoring_protocol": "reference_cpu_session_v1"}
    if backend == "cuda":
        payload.update(scoring_protocol="cuda_batch_fp32_cpu_boundary_v1", gpu_batch_size=gpu_batch_size,
                       boundary_margin=boundary_margin, cpu_prefix=cpu_prefix)
    elif backend != "cpu":
        raise ValueError("Evaluation backend must be cpu or cuda")
    return identity(payload)


def validate_cpu_prefix_import(project, bundle, manifest, output, stage, candidate, smoke, backend,
                               gpu_batch_size, boundary_margin, import_path, *, sources=None):
    """Verify an explicit, immutable CPU prefix imported into a CUDA run.

    This verifies prior score lineage, not every still-unscored WAV. A caller
    may defer file checks only after this succeeds and must verify each remaining
    session's actual hashed PCM, format and bounds before scoring/publishing it.
    Initial target cache whole-file hashes are historical evidence: CUDA progress
    changes their tails, so only the imported CPU prefix remains invariant.
    """
    import hashlib
    import io
    import math
    import zipfile
    import numpy as np
    from pathlib import Path
    from .common import bound_file, digest, identity, read_json, read_rows
    from .runtime import bundle_candidates
    from .evaluate import evaluation_binding

    def require(condition, message):
        if not condition:
            raise ValueError("Invalid CPU score-prefix import: " + message)

    def file_identity(path):
        value = Path(path).stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    project, bundle = Path(project).resolve(), Path(bundle).resolve()
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    import_path = Path(import_path).resolve()
    root = project.parent
    require(stage == "development" and not smoke and backend == "cuda", "only explicit non-smoke Development CUDA imports are allowed")
    require(import_path == output / "cpu_prefix_import.json", "import receipt must stay in the target evaluation directory")
    import_before = file_identity(import_path)
    receipt = read_json(import_path)
    require(receipt.get("schema_version") == 1 and receipt.get("kind") == "explicit_cpu_score_prefix_import_v1", "unknown receipt schema")
    require(receipt.get("user_approved_numerical_differences") is True, "CUDA numerical differences were not approved")
    require(receipt.get("target_backend") == "cuda", "wrong target backend")
    require(receipt.get("gpu_batch_size") == gpu_batch_size and receipt.get("boundary_margin") == boundary_margin,
            "CUDA batch or boundary policy changed")
    require(type(gpu_batch_size) is int and gpu_batch_size > 0 and math.isfinite(boundary_margin) and boundary_margin > 0,
            "invalid CUDA parameters")
    require(Path(receipt.get("project", "")).resolve() == project, "project path changed")
    require(receipt.get("project_config_sha256") == digest(project), "project configuration changed")
    require(receipt.get("bundle_manifest_sha256") == digest(bundle / "manifest.json"), "model/runtime bundle changed")
    require(receipt.get("source_scoring_protocol") == "reference_cpu_session_v1", "source was not the reference CPU protocol")
    stable_files = {import_path: import_before}

    def checked(entry):
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("sha256"), str), "missing bound file")
        path = resolved(root, entry["path"])
        before = file_identity(path)
        result = bound_file(root, entry)
        require(file_identity(path) == before, "source proof changed during verification: " + str(path))
        stable_files[path] = before
        return Path(result).resolve()

    require(checked(receipt.get("evaluation_manifest")) == manifest, "evaluation manifest changed")
    prefix = receipt.get("prefix")
    keys = {"source_binding", "committed_sessions", "frame_prefix_sha256", "journal_prefix_sha256"}
    require(isinstance(prefix, dict) and set(prefix) == keys, "prefix identity must have exactly four canonical fields")
    count = prefix["committed_sessions"]
    require(type(count) is int and count > 0, "invalid imported session count")
    source_directory = Path(receipt.get("source_evaluation_directory", "")).resolve()
    source_work = Path(receipt.get("source_work_directory", "")).resolve()
    require(source_directory.is_relative_to(root) and source_directory != output and source_work == source_directory / "_evaluation_work",
            "original CPU work directory is not a separate project output")
    require(output.is_relative_to(root) and Path(receipt.get("target_evaluation_directory", "")).resolve() == output,
            "wrong CUDA output directory")
    source_checkpoint = checked(receipt.get("source_checkpoint"))
    source_frames = checked(receipt.get("source_frames"))
    source_journal = checked(receipt.get("source_journal"))
    require(source_checkpoint == source_work / "checkpoint.json" and source_frames == source_work / "frames.npy"
            and source_journal == source_work / "clips.jsonl", "CPU work files do not match their declared directory")
    checkpoint = read_json(source_checkpoint)
    require(checkpoint.get("schema_version") == 1 and checkpoint.get("binding") == prefix["source_binding"], "CPU checkpoint binding changed")
    require(all(checkpoint.get(key) == prefix[key] for key in keys - {"source_binding"}), "CPU checkpoint prefix changed")
    require(type(checkpoint.get("journal_bytes")) is int and checkpoint["journal_bytes"] > 0, "invalid CPU journal boundary")
    require(evaluation_binding(project, bundle, manifest, stage, candidate, smoke) == prefix["source_binding"],
            "original CPU model, runtime or inputs changed")
    expected_binding = evaluation_binding(project, bundle, manifest, stage, candidate, smoke, backend="cuda",
        gpu_batch_size=gpu_batch_size, boundary_margin=boundary_margin, cpu_prefix=prefix)
    require(receipt.get("target_binding") == expected_binding and expected_binding != prefix["source_binding"], "wrong CUDA protocol binding")
    request_path, owner_path = checked(receipt.get("source_request")), checked(receipt.get("source_owner"))
    require(request_path.is_relative_to(root / ".studio/jobs") and request_path.name == "request.json"
            and owner_path == request_path.parent / "owner.json", "CPU owner/request are outside their original task")
    request, owner = read_json(request_path), read_json(owner_path)
    require(request.get("stage") == "evaluation" and request.get("config_sha256") == receipt["project_config_sha256"]
            and Path(request.get("project", "")).resolve() == project, "CPU request belongs to another experiment")
    require(owner.get("task_id") == request.get("task_id") and owner.get("request_sha256") == receipt["source_request"]["sha256"],
            "CPU owner does not match the original request")
    require(request.get("implementation_id") == receipt.get("source_implementation_id"), "CPU implementation identity changed")
    original_results = request.get("context", {}).get("results", {})
    require(Path(original_results.get("package", {}).get("result", {}).get("bundle", "")).resolve() == bundle
            and Path(original_results.get("development", {}).get("result", {}).get("manifest", "")).resolve() == manifest,
            "CPU request used different evaluated inputs")
    # Verify the archived producer code against its exact implementation ID.
    # The original producer creates its journal only after the full preflight;
    # this proof does not declare remaining files verified in this new run.
    proof = receipt.get("old_preflight_completion_proof", {})
    require(proof.get("source_checkpoint_binding") == prefix["source_binding"], "missing completed preflight lineage")
    require(proof.get("source_request") == receipt["source_request"], "preflight proof refers to another CPU request")
    backup_path = checked(receipt.get("cpu_prefix_backup"))
    full_backup_path = checked(receipt.get("full_score_backup"))
    require(receipt["full_score_backup"]["sha256"] == receipt["source_frames"]["sha256"], "CPU full frame backup changed")
    with zipfile.ZipFile(backup_path) as archive:
        archived_code = {name.removeprefix("code_before/"): hashlib.sha256(archive.read(name)).hexdigest()
                         for name in archive.namelist() if name.startswith("code_before/") and not name.endswith("/")}
        require(archived_code and identity(archived_code) == receipt["source_implementation_id"], "archived CPU implementation changed")
        require(archived_code.get("evaluate.py") == proof.get("source_evaluate_module_sha256"), "archived preflight producer changed")
        archived_checkpoint = json.loads(archive.read("checkpoint.json"))
        require(archived_checkpoint == checkpoint, "CPU checkpoint backup changed")
        with archive.open("clips_prefix.jsonl") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
            require(checksum.hexdigest() == prefix["journal_prefix_sha256"], "CPU clip-prefix backup changed")
        archived_scores = np.load(io.BytesIO(archive.read("frames_prefix.npy")), allow_pickle=False)
        require(archived_scores.dtype == np.float32 and archived_scores.ndim == 2
                and hashlib.sha256(archived_scores.tobytes()).hexdigest() == prefix["frame_prefix_sha256"], "CPU score-prefix backup changed")
    sources = read_rows(manifest) if sources is None else sources
    require(count < len(sources), "CUDA must have remaining sessions to score")
    remaining, frames, rows = checkpoint["journal_bytes"], 0, 0
    clip_hash = hashlib.sha256()
    with source_journal.open("rb") as stream:
        while remaining:
            line = stream.readline(remaining)
            require(bool(line) and line.endswith(b"\n"), "CPU journal prefix is incomplete")
            remaining -= len(line); clip_hash.update(line)
            require(rows < count, "CPU journal prefix has too many rows")
            scored, original = json.loads(line), sources[rows]
            require(all(scored.get(key) == original.get(key) for key in ("sample_id", "sha256", "kind", "category", "split")),
                    "CPU prefix input order or labels changed")
            require(Path(scored["path"]).resolve() == Path(original["path"]).resolve(), "CPU prefix audio path changed")
            require(len(scored.get("calls", [])) == len(original.get("calls", [])), "CPU call count changed")
            require(all(all(after.get(key) == value for key, value in before.items())
                        for before, after in zip(original.get("calls", []), scored.get("calls", []))), "CPU call boundaries changed")
            duration = scored.get("duration_s")
            require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0, "CPU duration is invalid")
            n_frames = round(duration / .08)
            require(abs(duration - n_frames * .08) < 1e-7, "CPU duration is not an 80ms frame count")
            frames += n_frames; rows += 1
    require(rows == count and clip_hash.hexdigest() == prefix["journal_prefix_sha256"], "CPU ordered clip prefix changed")
    package = read_json(bundle / "manifest.json")
    width = len(bundle_candidates(package))
    require(archived_scores.shape == (frames, width), "CPU prefix backup frame shape changed")
    target_work = output / "_evaluation_work"
    target_journal, target_frames = target_work / "clips.jsonl", target_work / "frames.npy"
    target_before = {path: file_identity(path) for path in (target_journal, target_frames, target_work / "checkpoint.json")}
    target_checkpoint = read_json(target_work / "checkpoint.json")
    require(target_checkpoint.get("binding") == expected_binding and target_checkpoint.get("schema_version") == 1,
            "target journal has the wrong CUDA protocol")
    require(target_checkpoint.get("shape") == checkpoint.get("shape") and target_checkpoint.get("committed_sessions", -1) >= count,
            "target journal lost the imported prefix")
    with target_journal.open("rb") as stream:
        checksum, left = hashlib.sha256(), checkpoint["journal_bytes"]
        while left:
            block = stream.read(min(1024 * 1024, left))
            require(bool(block), "CUDA target clip prefix was truncated")
            checksum.update(block); left -= len(block)
    require(checksum.hexdigest() == prefix["journal_prefix_sha256"], "CUDA target changed imported CPU clip metadata")
    for path in (source_frames, target_frames):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            require(values.dtype == np.float32 and values.shape == tuple(checkpoint["shape"])
                    and values.ndim == 2 and values.shape[1] == width and frames <= len(values), "CPU/target score cache shape or precision changed")
            checksum = hashlib.sha256()
            for begin in range(0, frames, 65536):
                checksum.update(values[begin:min(begin + 65536, frames)].tobytes())
            require(checksum.hexdigest() == prefix["frame_prefix_sha256"], "CPU/target imported float32 score bits changed")
        finally:
            values._mmap.close()
    for path, expected in {**stable_files, **target_before}.items():
        require(file_identity(path) == expected, "import proof or cache changed during verification: " + str(path))
    return receipt


def _verified_gpu_items(root, sources, paths, offsets, start):
    """Verify each remaining input immediately before consuming its PCM."""
    for index in range(start, len(sources)):
        row, path = sources[index], paths[index]
        before = path.stat()
        bound_file(root, row)
        info = sf.info(path)
        duration = info.frames / info.samplerate
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames == 0:
            raise ValueError("Scored WAV must already be nonempty mono16kHz PCM16")
        if info.frames // 1280 != offsets[index + 1] - offsets[index]:
            raise ValueError("Evaluation WAV frame count changed after planning")
        start_s, end_s = float(row.get("start_s", 0)), float(row.get("end_s", duration))
        if not 0 <= start_s < end_s <= duration + 1e-5:
            raise ValueError("Evaluation source interval is invalid")
        for call in row.get("calls", []):
            if not 0 <= float(call["start_s"]) < float(call["end_s"]) <= duration + 1e-5:
                raise ValueError("Evaluation call boundaries are invalid")
        yield index, row, path
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Evaluation audio changed during GPU consumption")


def _gpu_sessions_with_environment(environment, *args, **kwargs):
    from .gpu_evaluation import evaluation_runtime_environment, score_gpu_sessions
    with evaluation_runtime_environment(environment):
        yield from score_gpu_sessions(*args, **kwargs)


@project_storage
def evaluate(project, bundle, output, stage="development", manifest=None, smoke=False, candidate=None, workers="auto", resume=False,
             *, backend="auto", gpu_batch_size=128, boundary_margin=.001, cpu_prefix_import=None,
             numerical_differences_approved=None):
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
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("Evaluation backend must be auto, cpu or cuda")
    if backend == "cuda" and stage != "development":
        raise ValueError("CUDA evaluation is qualified for Development only; independent Final remains on its frozen reference protocol")
    if backend == "cuda" and numerical_differences_approved is False:
        raise ValueError("CUDA FP32 requires explicit approval of small numerical score differences")
    manifest = Path(manifest).resolve() if manifest else resolved(root, config["manifests"][stage])
    output = Path(output).resolve()
    resume_backend = "cuda" if cpu_prefix_import else None
    if resume and not cpu_prefix_import:
        checkpoint_path = output / "_evaluation_work/checkpoint.json"
        if not checkpoint_path.is_file():
            raise ValueError("No resumable evaluation checkpoint")
        saved_binding = read_json(checkpoint_path).get("binding")
        for saved_backend in ("cpu", "cuda"):
            if saved_binding == evaluation_binding(project, directory, manifest, stage, candidate, smoke,
                    backend=saved_backend, gpu_batch_size=gpu_batch_size, boundary_margin=boundary_margin):
                resume_backend = saved_backend
                break
        if resume_backend is None:
            raise ValueError("Evaluation resume model, inputs or scoring protocol changed")
    from .evaluation_backend import resolve_evaluation_backend
    selection = resolve_evaluation_backend(backend, stage=stage, bundle=directory,
        resume_backend=resume_backend, project_root=root)
    backend = selection["backend"]
    print("Evaluation device: " + ("GPU" if backend == "cuda" else "CPU") + " · " + selection["reason"], flush=True)
    if backend == "cuda":
        if stage != "development":
            raise ValueError("CUDA evaluation is qualified for Development only; independent Final remains on its frozen reference protocol")
        if numerical_differences_approved is False:
            raise ValueError("CUDA FP32 requires explicit approval of small numerical score differences")
        if type(gpu_batch_size) is not int or not 1 <= gpu_batch_size <= 128:
            raise ValueError("GPU batch size must be an integer from 1 to 128")
        if isinstance(boundary_margin, bool) or not isinstance(boundary_margin, (int, float)) or not math.isfinite(boundary_margin) or not 0 < boundary_margin < 1:
            raise ValueError("GPU boundary margin must be positive and finite below one")
    if cpu_prefix_import and (backend != "cuda" or not resume):
        raise ValueError("CPU prefix import requires explicit CUDA checkpoint resume")
    if stage == "final":
        if candidate != deployed["id"]:
            raise ValueError("Final evaluates the frozen deployed candidate; no candidate selection")
        lock = require_final_lock(project)
        if lock["bundle_manifest_sha256"] != digest(directory / "manifest.json") or lock["project_config_sha256"] != digest(project):
            raise ValueError("Final model/runtime/project changed after freeze")
        if smoke:
            raise ValueError("Final cannot be a smoke run")
    sources = read_rows(manifest)
    imported = validate_cpu_prefix_import(project, directory, manifest, output, stage, candidate, smoke,
        backend, gpu_batch_size, boundary_margin, cpu_prefix_import, sources=sources) if cpu_prefix_import else None
    # Reuse this operation's full hash/header checks from the split audit in
    # the format/offset pass below. No evidence is reused across evaluations.
    verified_files = VerifiedAudioFiles()
    audit = audit_project(project, check_files=not bool(imported), include_final_files=stage == "final", extra_rows=sources,
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
        if imported:
            # The immutable CPU prefix was checked before it was scored. Remaining
            # WAVs are hashed at their actual point of GPU consumption below.
            path = resolved(root, row["path"])
            info = sf.info(path)
        else:
            path, info = verified_files.inspect(root, row)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
            raise ValueError("Scored WAV must already be mono16kHz PCM16; no playback-only gain")
        paths.append(path)
        offsets.append(offsets[-1] + info.frames // 1280)
    verified_files.clear()
    if output.exists() and any(output.iterdir()):
        if not resume or (output / "receipt.json").exists():
            raise ValueError("Evaluation outputs are immutable; choose a new directory (or resume an incomplete run explicitly)")
    elif resume:
        raise ValueError("No resumable evaluation checkpoint; choose a new output directory")
    runtime = package["runtime"]
    model_ids = [row["id"] for row in candidates]
    execution = worker_plan(workers, len(sources))
    binding = evaluation_binding(project, directory, manifest, stage, candidate, smoke, backend=backend,
        gpu_batch_size=gpu_batch_size, boundary_margin=boundary_margin,
        cpu_prefix=imported["prefix"] if imported else None)
    execution["backend"] = backend
    execution["backend_selection"] = selection
    if backend == "cuda":
        execution["gpu"] = {}
        execution["input_validation"] = ("Prior CPU prefix verified by immutable import proof; current metadata audit and header plan; remaining WAVs fully hashed and checked before consumption"
                                         if imported else "Full input audit plus exact-use WAV checks")
        execution["cpu_prefix_import"] = {"path": str(Path(cpu_prefix_import).resolve()), "sha256": digest(cpu_prefix_import),
            "prefix": imported["prefix"]} if imported else None
        execution["scoring_protocol"] = "cuda_batch_fp32_cpu_boundary_v1"
    started = time.perf_counter()
    with writer_lock(output):
        journal = EvaluationJournal(output, binding, offsets, model_ids, resume=resume)
        try:
            execution["resumed_sessions"] = journal.resumed_sessions
            items = ((index, sources[index], paths[index]) for index in range(journal.resumed_sessions, len(sources)))
            if backend == "cuda":
                items = _verified_gpu_items(root, sources, paths, offsets, journal.resumed_sessions)
                scored = _gpu_sessions_with_environment(selection["runtime_environment"], directory, model_ids, items, batch_size=gpu_batch_size,
                    boundary_margin=boundary_margin,
                    qualification={"user_approved_numerical_differences": True,
                                   "cpu_prefix_import_sha256": digest(cpu_prefix_import) if imported else None},
                    stats=execution["gpu"])
            else:
                scored = score_sessions(directory, model_ids, items, execution["workers"], detector_factory=WakeWordDetector)
            for index, clip, scores in scored:
                journal.add(index, clip, scores)
                if index < 9 or (index + 1) % 100 == 0 or index + 1 == len(sources):
                    label = f"CUDA batch={gpu_batch_size}, CPU rechecked={execution['gpu'].get('cpu_rechecked_sessions', 0)}" if backend == "cuda" else f"CPU workers={execution['workers']}"
                    print(f"Evaluate {stage}: {index + 1}/{len(sources)} ({label})", flush=True)
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
