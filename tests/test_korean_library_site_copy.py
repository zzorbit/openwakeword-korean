"""Independent, exact scored WAV exports without a redundant source read."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword import site


def _evaluation(tmp_path, copies=1):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    source = directory / "scored.wav"
    sf.write(source, np.linspace(-.2, .2, 16000), 16000, subtype="PCM_16")
    clips = [{"sample_id": f"call_{index}", "path": str(source), "sha256": digest(source),
              "kind": "positive", "category": "clean", "results": {"single": {"success": True, "max_score": .91}}}
             for index in range(copies)]
    report = {"stage": "development", "wakeword": "<보라야>", "models": [{"id": "single"}]}
    write_json(directory / "clips.json", clips)
    write_json(directory / "report.json", report)
    write_json(directory / "receipt.json", {"clips_sha256": digest(directory / "clips.json"),
               "report_sha256": digest(directory / "report.json")})
    return directory, source, report, clips


def test_export_reads_source_once_and_independent_destination_once_for_repeated_clips(tmp_path, monkeypatch):
    evaluation, source, report, clips = _evaluation(tmp_path, copies=3)
    output = tmp_path / "site"
    expected = clips[0]["sha256"]
    opened = []
    original_open = Path.open

    def counted(path, mode="r", *args, **kwargs):
        if "r" in mode and "b" in mode and (path == source or path.name.startswith(expected)):
            opened.append(path)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", counted)
        result = site.export_site(evaluation, output)
    assert len(opened) == 2
    assert opened[0] == source and opened[1].suffix == ".tmp"
    destination = output / "audio" / (expected + ".wav")
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino
    assert result["audio_files"] == 3
    payload = (output / "results-data.js").read_text(encoding="utf-8")
    exported = json.loads(payload.removeprefix("window.KOWW_RESULTS=").removesuffix(";\n"))
    assert exported["report"] == report
    assert [row["results"] for row in exported["clips"]] == [row["results"] for row in clips]
    expected_rows = [{**row, "path": "audio/" + expected + ".wav"} for row in clips]
    old_serialization = "window.KOWW_RESULTS=" + json.dumps({"report": report, "clips": expected_rows},
        ensure_ascii=False, allow_nan=False).replace("<", "\\u003c") + ";\n"
    assert payload == old_serialization
    assert read_json(output / "provenance.json")["same_scored_WAVs"] is True
    assert not list((output / "audio").glob("*.tmp"))
    # The export has independent file contents and survives source removal.
    data = destination.read_bytes()
    source.unlink()
    assert destination.read_bytes() == data


def test_corrupt_source_is_preserved_and_never_published(tmp_path):
    evaluation, source, _, clips = _evaluation(tmp_path)
    source.write_bytes(source.read_bytes() + b"changed")
    before = source.read_bytes()
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="Scored WAV changed"):
        site.export_site(evaluation, output)
    assert source.read_bytes() == before
    assert not (output / "audio" / (clips[0]["sha256"] + ".wav")).exists()
    assert len(list((output / "audio").glob("*.tmp"))) == 1
    assert not (output / "index.html").exists()


def test_another_source_claiming_same_hash_is_still_verified(tmp_path):
    evaluation, source, _, clips = _evaluation(tmp_path, copies=2)
    other = evaluation / "other.wav"
    other.write_bytes(source.read_bytes() + b"not the scored bytes")
    clips[1]["path"] = str(other)
    write_json(evaluation / "clips.json", clips)
    receipt = read_json(evaluation / "receipt.json")
    receipt["clips_sha256"] = digest(evaluation / "clips.json")
    write_json(evaluation / "receipt.json", receipt)
    with pytest.raises(ValueError, match="Scored WAV changed"):
        site.export_site(evaluation, tmp_path / "site")
    assert digest(source) == clips[0]["sha256"]
    assert other.read_bytes().endswith(b"not the scored bytes")
    assert not (tmp_path / "site/index.html").exists()


def test_corrupt_temporary_copy_is_retained_without_touching_source_or_publishing(tmp_path, monkeypatch):
    evaluation, source, _, clips = _evaluation(tmp_path)
    before = source.read_bytes()
    original_digest = site.digest

    def corrupt_copy(path):
        if Path(path).suffix == ".tmp":
            Path(path).write_bytes(b"damaged destination")
        return original_digest(path)

    monkeypatch.setattr(site, "digest", corrupt_copy)
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="Copied WAV changed"):
        site.export_site(evaluation, output)
    assert source.read_bytes() == before
    assert not (output / "audio" / (clips[0]["sha256"] + ".wav")).exists()
    assert len(list((output / "audio").glob("*.tmp"))) == 1
    assert not (output / "index.html").exists()


@pytest.mark.parametrize("changed_file,error", [("source", "Scored WAV changed"), ("destination", "Copied WAV changed")])
def test_duplicate_reuse_checks_file_identity_again(tmp_path, changed_file, error):
    _, source, _, clips = _evaluation(tmp_path)
    destination = tmp_path / "copy.wav"
    sources, destinations = {}, {}
    site._copy_verified_audio(source, destination, clips[0]["sha256"], sources, destinations)
    target = source if changed_file == "source" else destination
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=error):
        site._copy_verified_audio(source, destination, clips[0]["sha256"], sources, destinations)
    if changed_file == "destination":
        assert digest(source) == clips[0]["sha256"]


def test_copy_refuses_source_identity_change_during_stream_without_publishing(tmp_path, monkeypatch):
    _, source, _, clips = _evaluation(tmp_path)
    destination = tmp_path / "copy.wav"
    before = source.read_bytes()
    original_identity = site._file_identity
    calls = 0

    def changing_identity(path):
        nonlocal calls
        identity = original_identity(path)
        if path == source:
            calls += 1
            if calls == 2:
                return (*identity[:-1], identity[-1] + 1)
        return identity

    monkeypatch.setattr(site, "_file_identity", changing_identity)
    with pytest.raises(ValueError, match="Scored WAV changed"):
        site._copy_verified_audio(source, destination, clips[0]["sha256"], {}, {})
    assert source.read_bytes() == before
    assert not destination.exists()
    assert len(list(tmp_path.glob("copy.wav.*.tmp"))) == 1


def test_existing_export_remains_immutable(tmp_path):
    evaluation, _, _, _ = _evaluation(tmp_path)
    output = tmp_path / "site"
    site.export_site(evaluation, output)
    protected = {path: digest(path) for path in output.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="immutable"):
        site.export_site(evaluation, output)
    assert {path: digest(path) for path in protected} == protected
