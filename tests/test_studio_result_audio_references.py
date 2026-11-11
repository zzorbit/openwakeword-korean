"""Studio result references: immutable scored evidence, no model or real data."""
from contextlib import contextmanager
from pathlib import Path
import json
import wave

import pytest

from korean_openwakeword import studio, studio_steps, studio_worker, site as site_module
from korean_openwakeword.common import digest, read_json, write_json
from test_korean_library_studio import manager, http_server, create, worker_request


def reference_site(manager, *, project_override=None, audio_override=None):
    view = create(manager)
    state = manager._load(view["id"])
    root = Path(state["project"]).parent
    evaluation, site = root / "reports/evaluation", root / "reports/site"
    evaluation.mkdir(parents=True)
    site.mkdir(parents=True)
    audio = audio_override or root / "data/scored.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 80)
    sha = digest(audio)
    write_json(evaluation / "report.json", {"stage": "development", "models": []})
    write_json(evaluation / "clips.json", [{"path": str(audio), "sha256": sha}])
    write_json(evaluation / "receipt.json", {"report_sha256": digest(evaluation / "report.json"),
        "clips_sha256": digest(evaluation / "clips.json")})
    receipt = read_json(evaluation / "receipt.json")
    references = {"schema_version": 1, "kind": "studio_scored_audio_references_v1",
        "project": str(project_override or Path(state["project"])), "evaluation": str(evaluation),
        "evaluation_receipt_sha256": digest(evaluation / "receipt.json"), **receipt,
        "audio": {sha: {"path": str(audio), "sha256": sha}}}
    write_json(site / "audio-references.json", references)
    write_json(site / "provenance.json", {"evaluation_receipt_sha256": references["evaluation_receipt_sha256"],
        "audio_storage": {"mode": "studio_reference_v1", "references": {
            "path": "audio-references.json", "sha256": digest(site / "audio-references.json")}}})
    (site / "index.html").write_text("scored reference site", encoding="utf-8")
    outputs = {"evaluation": {"state": "completed", "result": {"evaluation": str(evaluation)}},
               "results": {"state": "completed", "result": {"site": str(site / "index.html")}}}
    history = []
    for stage, paths in (("evaluation", [evaluation / n for n in ("receipt.json", "report.json", "clips.json")]),
                         ("results", [site / "provenance.json"])):
        result_path = root / ".studio/jobs" / stage / "result.json"
        write_json(result_path, {"result": outputs[stage], "artifact_bindings": [
            {"path": str(path), "sha256": digest(path)} for path in paths]})
        history.append({"stage": stage, "result": str(result_path), "sha256": digest(result_path)})
    state.update(plan=[{"id": "evaluation"}, {"id": "results"}], index=2, status="completed",
                 results=outputs, history=history)
    manager._save(state)
    return state, site, evaluation, audio, sha


def test_result_reference_streams_exact_original_without_copy(manager, http_server):
    state, site, _, audio, sha = reference_site(manager)
    status, body, headers = http_server(f"/results/{state['id']}/audio/{sha}.wav")
    assert status == 200 and body == audio.read_bytes()
    assert "audio" in headers["Content-Type"]
    assert not (site / "audio" / (sha + ".wav")).exists()
    assert http_server(f"/results/{state['id']}/")[0] == 200


@pytest.mark.parametrize("name", ["audio-references.json", "%61udio-references.json", "AUDIO-REFERENCES.JSON",
    "audio-references.json%3A%3A%24DATA", "audio-references.json.", "audio-references.json%20"])
def test_private_reference_index_is_never_served(manager, http_server, name):
    state, _, _, audio, _ = reference_site(manager)
    status, body, _ = http_server(f"/results/{state['id']}/{name}")
    assert status == 403 and str(audio).encode() not in body


@pytest.mark.parametrize("suffix", ["audio/not-a-hash.wav", "audio/../audio-references.json", "../secret.wav"])
def test_result_reference_rejects_unregistered_and_traversal_names(manager, http_server, suffix):
    state, _, _, audio, _ = reference_site(manager)
    status, body, _ = http_server(f"/results/{state['id']}/{suffix}")
    assert status in (400, 403) and str(audio).encode() not in body


def test_result_reference_rejects_query_path_injection(manager, http_server):
    state, _, _, audio, sha = reference_site(manager)
    status, body, _ = http_server(f"/results/{state['id']}/audio/{sha}.wav?path=other.wav")
    assert status == 400 and str(audio).encode() not in body


@pytest.mark.parametrize("mutation", ["audio", "audio-missing", "sidecar", "sidecar-missing", "provenance-and-sidecar", "evaluation-receipt"])
def test_changed_or_deleted_reference_evidence_is_rejected_after_cache(manager, http_server, mutation):
    state, site, evaluation, audio, sha = reference_site(manager)
    route = f"/results/{state['id']}/audio/{sha}.wav"
    # Prime the same metadata cache without racing Windows' HTTP stream close.
    assert manager.result_audio_reference(state, site, f"audio/{sha}.wav") == audio
    if mutation == "audio":
        audio.write_bytes(audio.read_bytes()[:-2] + b"\x02\x00")
    elif mutation == "audio-missing":
        audio.unlink()
    elif mutation == "sidecar-missing":
        (site / "audio-references.json").unlink()
    elif mutation in ("sidecar", "provenance-and-sidecar"):
        refs = read_json(site / "audio-references.json")
        refs["audio"][sha]["path"] = str(audio.parent / "unscored.wav")
        write_json(site / "audio-references.json", refs)
        if mutation == "provenance-and-sidecar":
            provenance = read_json(site / "provenance.json")
            provenance["audio_storage"]["references"]["sha256"] = digest(site / "audio-references.json")
            write_json(site / "provenance.json", provenance)
    else:
        write_json(evaluation / "receipt.json", {"report_sha256": "0" * 64, "clips_sha256": "0" * 64})
    status, body, _ = http_server(route)
    assert status == 400 and str(audio).encode() not in body


def test_reference_requires_completed_result_receipt(manager):
    state, site, _, _, sha = reference_site(manager)
    state["history"] = [row for row in state["history"] if row["stage"] != "results"]
    with pytest.raises(ValueError, match="결과 음원 참조"):
        manager.result_audio_reference(state, site, f"audio/{sha}.wav")


def test_hash_bound_reference_from_other_project_is_rejected(manager, tmp_path):
    state, site, _, _, sha = reference_site(manager, project_override=tmp_path / "foreign/project.json")
    with pytest.raises(ValueError, match="결과 음원 참조"):
        manager.result_audio_reference(state, site, f"audio/{sha}.wav")


def test_original_outside_selected_workspace_is_rejected(manager, tmp_path):
    state, site, _, _, sha = reference_site(manager, audio_override=tmp_path / "outside.wav")
    with pytest.raises(ValueError, match="결과 음원 참조"):
        manager.result_audio_reference(state, site, f"audio/{sha}.wav")


def test_reference_cache_never_reparses_large_metadata_or_opens_clip_manifest_per_play(manager, monkeypatch):
    state, site, evaluation, audio, sha = reference_site(manager)
    name = f"audio/{sha}.wav"
    assert manager.result_audio_reference(state, site, name) == audio
    original_read, original_digest = studio.read_json, studio.digest
    reads, hashes = [], []
    monkeypatch.setattr(studio, "read_json", lambda path: reads.append(Path(path)) or original_read(path))
    monkeypatch.setattr(studio, "digest", lambda path: hashes.append(Path(path)) or original_digest(path))
    assert manager.result_audio_reference(state, site, name) == audio
    assert reads == [] and hashes == [audio]
    assert evaluation / "clips.json" not in hashes


def test_reference_evidence_change_during_verified_read_is_rejected(manager, monkeypatch):
    state, site, _, _, sha = reference_site(manager)
    original_read = studio.read_json
    def changed(path):
        value = original_read(path)
        if Path(path) == site / "audio-references.json":
            write_json(path, {**value, "untrusted_change": True})
        return value
    monkeypatch.setattr(studio, "read_json", changed)
    with pytest.raises(ValueError, match="결과 음원 참조"):
        manager.result_audio_reference(state, site, f"audio/{sha}.wav")


def test_legacy_copied_result_audio_still_streams_without_reference_receipts(manager, http_server):
    state, site, _, audio, sha = reference_site(manager)
    write_json(site / "provenance.json", {"audio": {"fixture": sha}})
    copied = site / "audio" / (sha + ".wav")
    copied.parent.mkdir(); copied.write_bytes(audio.read_bytes())
    state["history"] = []
    manager._save(state)
    assert http_server(f"/results/{state['id']}/audio/{sha}.wav")[1] == audio.read_bytes()


@pytest.mark.parametrize("stage,expected", [("results", True), ("frontend", False), ("features", False)])
def test_worker_scopes_reference_context_to_results_only(tmp_path, monkeypatch, stage, expected):
    request_path, request = worker_request(tmp_path, stage)
    active, entered = [], []
    @contextmanager
    def context(project):
        entered.append(project); active.append(project)
        try:
            yield
        finally:
            active.pop()
    monkeypatch.setattr(site_module, "studio_result_audio", context, raising=False)
    def run(*args):
        assert bool(active) is expected
        return {"state": "completed"}
    monkeypatch.setattr(studio_steps, "run_stage", run)
    assert studio_worker._execute(request_path) == 0
    assert entered == ([request["project"]] if expected else []) and active == []


def test_worker_reference_context_is_reset_after_result_export_failure(tmp_path, monkeypatch):
    request_path, _ = worker_request(tmp_path, "results")
    active = []
    @contextmanager
    def context(project):
        active.append(project)
        try:
            yield
        finally:
            active.pop()
    monkeypatch.setattr(site_module, "studio_result_audio", context, raising=False)
    def fail(*args):
        assert active
        raise ValueError("fixture failure")
    monkeypatch.setattr(studio_steps, "run_stage", fail)
    assert studio_worker._execute(request_path) == 1 and active == []
