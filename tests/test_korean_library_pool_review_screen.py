"""The local screen must separate preview listening from whole-pool consent."""
import json
import numpy as np
import pytest
import soundfile as sf
from korean_openwakeword.common import digest, read_json
from korean_openwakeword.project import create_project
from korean_openwakeword.review import apply_pool_review

@pytest.fixture
def screen(tmp_path):
    project = create_project(tmp_path / "bora", "보라야", "bora")
    path = project.parent / "review/preview.wav"
    sf.write(path, np.random.default_rng(10).normal(0, .05, 4000), 16000)
    queue = project.parent / "review/queue.jsonl"
    queue.write_text(json.dumps({"sample_id": "one", "path": str(path), "sha256": digest(path)}) + "\n", encoding="utf-8")
    manifest = project.parent / "review/pool.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    return project, queue, manifest

def test_save_never_infers_whole_pool_consent(screen):
    project, queue, manifest = screen
    with pytest.raises(ValueError, match="동의"):
        apply_pool_review(project, queue, manifest, {"one": {"decision": "keep", "listened": True}}, False)
    assert not (project.parent / "review/pool_results.json").exists()

def test_save_never_turns_unheard_preview_into_listening(screen):
    project, queue, manifest = screen
    with pytest.raises(ValueError, match="Unheard"):
        apply_pool_review(project, queue, manifest, {"one": {"decision": "keep", "listened": False}}, True)

def test_exception_is_saved_but_not_admitted(screen, monkeypatch):
    project, queue, manifest = screen
    def no_approval(*a, **kw):
        pytest.fail("Exception must not approve the pool")
    monkeypatch.setattr("korean_openwakeword.pool_review.approve_pool", no_approval)
    result = apply_pool_review(project, queue, manifest, {"one": {"decision": "review", "listened": True}}, True)
    assert result["state"] == "content_exceptions_pending"
    assert not (project.parent / "review/pool_save_receipt.json").exists()

def test_positive_save_binds_actual_answers_and_frozen_inputs(screen, monkeypatch):
    project, queue, manifest = screen
    def approval(p, **kw):
        assert p == project
        assert kw["source_manifest"]["sha256"] == digest(manifest)
        assert kw["representative_queue"]["sha256"] == digest(queue)
        answers = read_json(kw["answers"]["path"])
        assert answers["whole_pool_policy_approved"] is True
        assert answers["decisions"]["one"]["listened"] is True
        return {"state": "saved", "admission": {"path": "immutable", "sha256": "fixture"}}
    monkeypatch.setattr("korean_openwakeword.pool_review.approve_pool", approval)
    result = apply_pool_review(project, queue, manifest, {"one": {"decision": "keep", "listened": True}}, True)
    assert read_json(project.parent / "review/pool_save_receipt.json") == result
