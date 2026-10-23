"""One-operation file verification reuses no cross-run trust or score state."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import common, data, evaluate, streaming
from korean_openwakeword.common import digest, identity, read_json
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.runtime import bundle_candidates, deployment_candidate, sample_bundle
from korean_openwakeword.streaming import VerifiedAudioFiles


def _audio(tmp_path):
    path = tmp_path / "wave.wav"
    sf.write(path, .1 * np.sin(np.arange(1600) / 25), 16000, subtype="PCM_16")
    return path, {"path": str(path), "sha256": digest(path)}


def _count_io(monkeypatch, paths):
    expected = {Path(path).resolve() for path in paths}
    hashes, headers = Counter(), Counter()
    original_digest, original_info = common.digest, sf.info
    def measured_digest(path):
        actual = Path(path).resolve()
        if actual in expected:
            hashes[actual] += 1
        return original_digest(path)
    def measured_info(path):
        actual = Path(path).resolve()
        if actual in expected:
            headers[actual] += 1
        return original_info(path)
    monkeypatch.setattr(common, "digest", measured_digest)
    monkeypatch.setattr(sf, "info", measured_info)
    return hashes, headers


def test_audio_checks_are_reused_only_within_one_operation(monkeypatch, tmp_path):
    path, row = _audio(tmp_path)
    hashes, headers = _count_io(monkeypatch, [path])
    first = VerifiedAudioFiles()
    actual, facts = first.inspect(tmp_path, row)
    assert actual == path and facts.frames == 1600 and facts.samplerate == 16000
    assert facts.channels == 1 and facts.subtype == "PCM_16"
    assert first.inspect(tmp_path, row) == (actual, facts)
    assert hashes[path] == headers[path] == 1
    # A separate operation hashes actual bytes again, even with unchanged stat.
    assert VerifiedAudioFiles().inspect(tmp_path, row) == (actual, facts)
    assert hashes[path] == headers[path] == 2
    first.clear()
    first.inspect(tmp_path, row)
    assert hashes[path] == headers[path] == 3


def test_touch_revalidates_and_changed_expected_hash_cannot_reuse_facts(monkeypatch, tmp_path):
    path, row = _audio(tmp_path)
    hashes, headers = _count_io(monkeypatch, [path])
    verified = VerifiedAudioFiles()
    verified.inspect(tmp_path, row)
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
    verified.inspect(tmp_path, row)
    assert hashes[path] == headers[path] == 2
    with pytest.raises(ValueError, match="Missing or changed"):
        verified.inspect(tmp_path, {**row, "sha256": "0" * 64})
    assert hashes[path] == 3 and headers[path] == 2


def test_replacement_with_same_size_and_timestamp_still_requires_hash(monkeypatch, tmp_path):
    path, row = _audio(tmp_path)
    hashes, _ = _count_io(monkeypatch, [path])
    verified = VerifiedAudioFiles()
    verified.inspect(tmp_path, row)
    before = path.stat()
    replacement = path.with_name("replacement.wav")
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    replacement.write_bytes(payload)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="Missing or changed"):
        verified.inspect(tmp_path, row)
    assert hashes[path] == 2


def test_change_during_header_read_is_not_cached(monkeypatch, tmp_path):
    path, row = _audio(tmp_path)
    original = sf.info
    def changed(path):
        info = original(path)
        before = Path(path).stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        return info
    monkeypatch.setattr(sf, "info", changed)
    verified = VerifiedAudioFiles()
    with pytest.raises(ValueError, match="changed during hash/header"):
        verified.inspect(tmp_path, row)
    assert not verified._files


def _evaluation_inputs(tmp_path):
    bundle = sample_bundle()
    project = create_project(tmp_path / "word", "춘식아", "chunsik")
    root, config = load_project(project)
    audio = read_json(bundle / "audio_manifest.json")
    selected = [next(row for row in audio if row["kind"] == kind) for kind in ("positive", "negative")]
    for number, row in enumerate(selected):
        source = bundle / row["path"]
        target = root / "data" / f"test_{number}.wav"
        shutil.copyfile(source, target)
        row["path"] = str(target)
    manifest = root / "data/eval/input.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    return project, root, bundle, manifest, selected


def test_evaluation_preflight_hashes_each_audio_once_and_keeps_exact_scores(monkeypatch, tmp_path):
    project, root, bundle, manifest, rows = _evaluation_inputs(tmp_path)
    files = [Path(row["path"]) for row in rows]
    hashes, headers = _count_io(monkeypatch, files)
    cached = evaluate.evaluate(project, bundle, root / "reports/cached", manifest=manifest, workers=1, smoke=True)
    assert all(hashes[path] == headers[path] == 1 for path in files)
    package = read_json(bundle / "manifest.json")
    # Preflight changes must not change the existing resumable score identity.
    expected = identity({"project_sha256": digest(project), "bundle_sha256": digest(bundle / "manifest.json"),
        "manifest_sha256": digest(manifest), "stage": "development", "candidate": deployment_candidate(package)["id"],
        "smoke": True, "model_ids": [row["id"] for row in bundle_candidates(package)],
        "scoring_protocol": "reference_cpu_session_v1"})
    assert cached["evaluation_binding"] == expected
    class Uncached(VerifiedAudioFiles):
        def inspect(self, root, row):
            self.clear()
            return super().inspect(root, row)
    monkeypatch.setattr(evaluate, "VerifiedAudioFiles", Uncached)
    reference = evaluate.evaluate(project, bundle, root / "reports/uncached", manifest=manifest, workers=1, smoke=True)
    assert all(hashes[path] == headers[path] == 3 for path in files)
    assert reference["evaluation_binding"] == cached["evaluation_binding"]
    assert reference["models"] == cached["models"]
    assert read_json(root / "reports/cached/clips.json") == read_json(root / "reports/uncached/clips.json")
    with np.load(root / "reports/cached/scores.npz") as first, np.load(root / "reports/uncached/scores.npz") as second:
        assert set(first.files) == set(second.files)
        assert all(np.array_equal(first[key], second[key]) for key in first.files)


def test_audio_mutation_between_audit_and_format_pass_is_rejected(monkeypatch, tmp_path):
    project, root, bundle, manifest, rows = _evaluation_inputs(tmp_path)
    original = data.audit_project
    target = Path(rows[0]["path"])
    def mutate_after_audit(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result["passed"]
        with target.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        before = target.stat()
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        return result
    monkeypatch.setattr(evaluate, "audit_project", mutate_after_audit)
    with pytest.raises(ValueError, match="Missing or changed"):
        evaluate.evaluate(project, bundle, root / "reports/refused", manifest=manifest, workers=1, smoke=True)
    assert not (root / "reports/refused").exists()


def test_cached_facts_do_not_skip_required_pcm_format_checks(tmp_path):
    project, root, bundle, manifest, rows = _evaluation_inputs(tmp_path)
    row = rows[0]
    values, rate = sf.read(row["path"])
    sf.write(row["path"], np.column_stack([values, values]), rate, subtype="PCM_16")
    row["sha256"] = digest(row["path"])
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mono16kHz PCM16"):
        evaluate.evaluate(project, bundle, root / "reports/stereo", manifest=manifest, workers=1, smoke=True)
    assert not (root / "reports/stereo").exists()
