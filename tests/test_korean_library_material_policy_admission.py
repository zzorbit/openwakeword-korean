"""Technical library admission keeps exact hashes and human review boundaries."""
import sys
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import data, negative_streams
from korean_openwakeword.common import digest, write_json
from korean_openwakeword.streaming import manifest_index


def technical_record(**extra):
    return {"state": "accepted", "wakeword": "보라야",
        "type": "library_fixed_catalog_technical_validation", "policy_id": "library_fixed_sources_v1",
        "content_review": "library_fixed_sources_technical_validation", "human_listening": False,
        "human_listened_count": 0, "human_answers": None,
        "library_evidence": {"path": "fixture_evidence.json", "sha256": "e" * 64},
        "source_manifest": {"path": "fixture_sources.jsonl", "sha256": "s" * 64}, **extra}


@pytest.fixture
def evidence_calls(monkeypatch):
    calls = []
    module = ModuleType("korean_openwakeword.material_policy")
    module.validate_library_policy_evidence = lambda root, artifact, role=None: calls.append((root, artifact, role))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def source(root, number=0, split="train"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"source_{number}.wav"
    sf.write(path, .1 * np.sin(np.arange(6400) / (20 + number)), 16000, subtype="PCM_16")
    return {"sample_id": f"source_{number}", "canonical_source_id": f"fixture:source:{number}",
        "canonical_recording_id": f"fixture:recording:{number}", "path": str(path), "sha256": digest(path),
        "split": split, "category": "environment", "speech": False, "duration_s": .4,
        "provenance": "public_dataset", "source_url": "https://example.test/fixture", "license": "fixture"}


def bind_record(root, record):
    path = root / "admission.json"
    write_json(path, record)
    return {"path": str(path), "sha256": digest(path)}


@pytest.mark.parametrize("role", ["generic_negative", "korean_negative", "noise", "rir"])
def test_train_library_policy_is_validated_once_per_receipt_with_exact_membership(tmp_path, evidence_calls, role):
    row = source(tmp_path)
    record = technical_record(accepted_source_sha256=[row["sha256"]])
    row["admission"] = bind_record(tmp_path, record)
    data.validate_admission(tmp_path, (dict(row, sample_id=str(i)) for i in range(300)), {"wakeword": "보라야"}, role)
    assert len(evidence_calls) == 1 and evidence_calls[0][2] == role
    metadata = evidence_calls[0][1]
    for key in ("type", "policy_id", "library_evidence", "human_listening", "human_listened_count", "human_answers", "source_manifest"):
        assert metadata[key] == record[key]
    unknown = source(tmp_path, 1)
    unknown["admission"] = row["admission"]
    with pytest.raises(ValueError, match="does not cover this exact source"):
        data.validate_admission(tmp_path, [unknown], {"wakeword": "보라야"}, role)


@pytest.mark.parametrize("role", ["positive", "hard_negative"])
def test_library_policy_cannot_replace_synthetic_or_hard_review(tmp_path, evidence_calls, role):
    row = source(tmp_path)
    row.update(provenance="qwen_tts", human_reference_used=False, voice_family_id="fixture_train",
               text="보라야" if role == "positive" else "보람아", speech=True)
    row["admission"] = bind_record(tmp_path, technical_record(accepted_source_sha256=[row["sha256"]],
        generation_policy_approved=True, human_reference_used=False))
    with pytest.raises(ValueError, match="cannot admit synthetic or Hard Negative"):
        data.validate_admission(tmp_path, [row], {"wakeword": "보라야"}, role)
    assert evidence_calls == []


def test_library_policy_does_not_bypass_train_split_or_file_integrity(tmp_path, evidence_calls):
    row = source(tmp_path)
    row["admission"] = bind_record(tmp_path, technical_record(accepted_source_sha256=[row["sha256"]]))
    with pytest.raises(ValueError, match="train split"):
        data.validate_admission(tmp_path, [dict(row, split="validation")], {"wakeword": "보라야"}, "noise")
    with open(row["path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed bound"):
        data.validate_admission(tmp_path, [row], {"wakeword": "보라야"}, "noise")
    assert evidence_calls == []


def test_human_admission_still_uses_existing_content_review_without_library_helper(tmp_path, evidence_calls):
    row = source(tmp_path)
    row["admission"] = bind_record(tmp_path, {"state": "accepted", "content_review": "accepted",
        "accepted_source_sha256": [row["sha256"]]})
    data.validate_admission(tmp_path, [row], {"wakeword": "보라야"}, "noise")
    assert evidence_calls == []


def test_disk_cached_admission_preserves_policy_evidence_without_reloading(tmp_path):
    record = technical_record(accepted_source_sha256=["a" * 64])
    with manifest_index(tmp_path) as index:
        key = ("admission.json", "f" * 64)
        first = index.admission(key, lambda: record)
        index.admission_metadata.clear()
        second = index.admission(key, lambda: pytest.fail("SQL cache must retain policy metadata"))
        assert second == first
        assert all(second[k] == record[k] for k in ("type", "policy_id", "library_evidence",
            "human_listening", "human_listened_count", "human_answers", "source_manifest"))
        assert index.admits(key, "a" * 64) and not index.admits(key, "b" * 64)


def evaluation_rows(root):
    rows = [source(root, i, "validation") for i in range(2)]
    binding = bind_record(root, technical_record(state="accepted_for_evaluation", evaluation_only=True,
        training_admitted=False, accepted_source_sha256=[r["sha256"] for r in rows]))
    return [{**r, "kind": "negative", "calls": [], "eval_bucket": "environment",
             "evaluation_content_approved": True, "evaluation_admission": binding} for r in rows]


def test_evaluation_library_policy_is_validated_once_and_retains_exact_membership(tmp_path, evidence_calls):
    rows = evaluation_rows(tmp_path)
    buckets, passthrough = negative_streams._inventory(tmp_path, rows, "보라야")
    assert len(buckets["environment"]) == 2 and passthrough == []
    assert len(evidence_calls) == 1 and evidence_calls[0][2] == "evaluation"
    unknown = source(tmp_path, 2, "validation")
    unknown.update({k: rows[0][k] for k in ("kind", "calls", "eval_bucket",
        "evaluation_content_approved", "evaluation_admission")})
    with pytest.raises(ValueError, match="does not cover this exact original"):
        negative_streams._inventory(tmp_path, [unknown], "보라야")


def test_evaluation_library_policy_cannot_bypass_scope_or_evidence_failure(tmp_path, monkeypatch, evidence_calls):
    rows = evaluation_rows(tmp_path)
    with pytest.raises(ValueError, match="held-out non-call"):
        negative_streams._inventory(tmp_path, [{**rows[0], "split": "train"}], "보라야")
    assert evidence_calls == []

    def reject(*args, **kwargs):
        raise ValueError("Changed fixed catalog evidence")

    monkeypatch.setattr(sys.modules["korean_openwakeword.material_policy"], "validate_library_policy_evidence", reject)
    with pytest.raises(ValueError, match="Changed fixed catalog evidence"):
        negative_streams._inventory(tmp_path, rows, "보라야")
