"""Build result viewers: portable copied audio or verified Studio references."""
from importlib.resources import files
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import os
import re
import uuid

from .common import digest, read_json, write_json


_STUDIO_RESULT_AUDIO = ContextVar("studio_result_audio_project", default=None)


@contextmanager
def studio_result_audio(project):
    """Opt one Studio results invocation into server-verified audio references."""
    project = Path(project)
    if not project.is_absolute() or project.name != "project.json" or not project.is_file():
        raise ValueError("Studio result audio needs an existing absolute project.json")
    project = project.resolve()
    if project.name != "project.json":
        raise ValueError("Studio result audio needs an existing absolute project.json")
    token = _STUDIO_RESULT_AUDIO.set(project)
    try:
        yield
    finally:
        _STUDIO_RESULT_AUDIO.reset(token)


def _file_identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _copy_verified_audio(source, destination, expected, sources, destinations):
    """Copy and hash in one source pass, then verify the independent copy.

    Cache only this export's exact file identities. A repeated clip can share
    its verified destination, but a changed source or destination is checked
    again. Failed temporary copies remain for diagnosis and are not published.
    """
    source_key, destination_key = str(source), str(destination)
    before = _file_identity(source)
    if destination.exists():
        if sources.get((source_key, expected)) != before:
            if digest(source) != expected or _file_identity(source) != before:
                raise ValueError("Scored WAV changed; cannot attach old scores to new audio")
            sources[source_key, expected] = before
        copied_before = _file_identity(destination)
        if destinations.get((destination_key, expected)) != copied_before:
            if digest(destination) != expected or _file_identity(destination) != copied_before:
                raise ValueError("Copied WAV changed")
            destinations[destination_key, expected] = copied_before
        return
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    checksum = hashlib.sha256()
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            checksum.update(block)
            outgoing.write(block)
    if checksum.hexdigest() != expected or _file_identity(source) != before:
        raise ValueError("Scored WAV changed; cannot attach old scores to new audio")
    copied_before = _file_identity(temporary)
    if digest(temporary) != expected or _file_identity(temporary) != copied_before:
        raise ValueError("Copied WAV changed")
    os.replace(temporary, destination)
    sources[source_key, expected] = before
    destinations[destination_key, expected] = _file_identity(destination)


def export_site(evaluation, output, runtime_recommendations=None, *, audio_mode=None):
    """Copy scored WAVs by default; Studio can explicitly bind lazy references.

    Reference exports are served only by Studio, which verifies the selected
    scored WAV's hash before playback. They are not portable static exports.
    """
    project = _STUDIO_RESULT_AUDIO.get()
    audio_mode = ("studio_reference_v1" if project is not None else "copy") if audio_mode is None else audio_mode
    if audio_mode not in ("copy", "studio_reference_v1"):
        raise ValueError("Unsupported result audio storage mode")
    reference_audio = audio_mode == "studio_reference_v1"
    if reference_audio and project is None:
        raise ValueError("Studio audio references require studio_result_audio(project)")
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    if reference_audio and not evaluation.is_relative_to(project.parent):
        raise ValueError("Studio evaluation does not belong to the selected project")
    reference_inputs = {name: _file_identity(evaluation / name) for name in
                        ("receipt.json", "report.json", "clips.json")} if reference_audio else {}
    receipt = read_json(evaluation / "receipt.json")
    for name in ("report.json", "clips.json"):
        if digest(evaluation / name) != receipt[name.split(".")[0] + "_sha256"]:
            raise ValueError("Completed evaluation report/clip binding changed")
    reference_receipt = digest(evaluation / "receipt.json") if reference_audio else None
    if reference_audio and any(_file_identity(evaluation / name) != binding for name, binding in reference_inputs.items()):
        raise ValueError("Completed evaluation changed during Studio reference export")
    report = read_json(evaluation / "report.json")
    if runtime_recommendations is not None:
        expected = {"source_report_sha256": receipt["report_sha256"],
                    "source_clips_sha256": receipt["clips_sha256"],
                    "source_bundle_manifest_sha256": report.get("bundle_manifest_sha256"),
                    "source_score_cache_sha256": report.get("score_cache_sha256")}
        bindings = runtime_recommendations.get("bindings", {}) if isinstance(runtime_recommendations, dict) else {}
        if any(value is None or bindings.get(key) != value for key, value in expected.items()):
            raise ValueError("Detection recommendations belong to a different evaluation")
        # This is a new display document. The frozen scoring report is never
        # overwritten or presented as having used these newly selected gates.
        report["runtime_recommendations"] = runtime_recommendations
    if output.exists() and any(output.iterdir()):
        raise ValueError("Site exports are immutable")
    clips = None if (evaluation / "clips.json").stat().st_size > 4 * 1024 ** 2 else read_json(evaluation / "clips.json")
    (output / "audio" if not reference_audio else output).mkdir(parents=True, exist_ok=True)
    references = {}

    def reference_path(row):
        expected, source = row["sha256"], Path(row["path"])
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or not source.is_absolute():
            raise ValueError("Studio audio reference needs an absolute scored WAV and SHA256")
        # Keep publication metadata-only: do not open, stat, or resolve every
        # WAV. The server checks the bound source at the moment it is played.
        references.setdefault(expected, {"path": os.path.abspath(source), "sha256": expected})
        return "audio/" + expected + ".wav"

    def provenance_audio_storage():
        if not reference_audio:
            return {}
        if digest(evaluation / "receipt.json") != reference_receipt or any(
                _file_identity(evaluation / name) != binding for name, binding in reference_inputs.items()):
            raise ValueError("Completed evaluation changed during Studio reference export")
        sidecar = output / "audio-references.json"
        write_json(sidecar, {"schema_version": 1, "kind": "studio_scored_audio_references_v1",
            "project": str(project), "evaluation": str(evaluation),
            "evaluation_receipt_sha256": reference_receipt,
            "report_sha256": receipt["report_sha256"], "clips_sha256": receipt["clips_sha256"], "audio": references})
        return {"audio_storage": {"mode": "studio_reference_v1",
                "references": {"path": sidecar.name, "sha256": digest(sidecar)}}}
    # A large evaluation must not become a single browser script. Keep full
    # diagnostics in the evaluation archive and load display details by page.
    if reference_audio or clips is None or len(clips) > 1000:
        from .paged_site import iter_clip_array, write_paged_viewer
        audio = {}
        verified_sources, verified_destinations = {}, {}
        def display_rows():
            rows = iter_clip_array(evaluation / "clips.json") if clips is None else clips
            for row in rows:
                if reference_audio:
                    path = reference_path(row)
                else:
                    source = Path(row["path"])
                    name = row["sha256"] + ".wav"
                    _copy_verified_audio(source, output / "audio" / name, row["sha256"],
                                         verified_sources, verified_destinations)
                    path = "audio/" + name
                audio[row["sample_id"]] = row["sha256"]
                yield {**row, "path": path}
        viewer = write_paged_viewer(report, display_rows(), output)
        resources = files("korean_openwakeword") / "resources"
        for source, destination in (("results_paged.html", "index.html"), ("results_paged.js", "results_paged.js"), ("results_ui.js", "results_ui.js"),
                                    ("results_scope.js", "results_scope.js"), ("results.css", "results.css"), ("audio_listening.css", "audio_listening.css"), ("listening_player.js", "listening_player.js")):
            (output / destination).write_bytes((resources / source).read_bytes())
        write_json(output / "provenance.json", {"evaluation_receipt_sha256": digest(evaluation / "receipt.json"),
            "audio": audio, "models": report["models"], "stage": report["stage"],
            "same_scored_WAVs": True, "new_inference_or_training": False,
            "viewer": {"schema_version": 1, **viewer}, **provenance_audio_storage()})
        return {"site": str(output / "index.html"), "audio_files": viewer["audio_files"], "stage": report["stage"]}
    verified_sources, verified_destinations = {}, {}
    for row in clips:
        source = Path(row["path"])
        name = row["sha256"] + ".wav"
        dest = output / "audio" / name
        _copy_verified_audio(source, dest, row["sha256"], verified_sources, verified_destinations)
        row["path"] = "audio/" + name
    root = files("korean_openwakeword") / "resources"
    for name in ("results.html", "results_scope.js", "results_ui.js", "results.js", "results.css", "audio_listening.css", "listening_player.js"):
        (output / ("index.html" if name == "results.html" else name)).write_bytes((root / name).read_bytes())
    # Avoid an additional full JSON string for large evaluation manifests.
    with (output / "results-data.js").open("w", encoding="utf-8") as stream:
        stream.write('window.KOWW_RESULTS={"report": ')
        stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
        stream.write(', "clips": [')
        for index, row in enumerate(clips):
            if index:
                stream.write(", ")
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"))
        stream.write("]};\n")
    write_json(output / "provenance.json", {"evaluation_receipt_sha256": digest(evaluation / "receipt.json"),
        "audio": {r["sample_id"]: r["sha256"] for r in clips}, "models": report["models"],
        "stage": report["stage"], "same_scored_WAVs": True, "new_inference_or_training": False,
        **provenance_audio_storage()})
    return {"site": str(output / "index.html"), "audio_files": len(clips), "stage": report["stage"]}
