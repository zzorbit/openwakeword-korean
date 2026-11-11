"""Studio references publish display metadata without touching scored WAVs."""
import json
from pathlib import Path
import wave

import pytest

from korean_openwakeword import site
from korean_openwakeword.common import digest, read_json, write_json


def _fixture(tmp_path, count=2):
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir(parents=True)
    source = evaluation / "scored.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)
    sha256 = digest(source)
    rows = [{"sample_id": f"sample_{i}", "path": str(source), "sha256": sha256,
             "kind": "positive", "category": "clean", "duration_s": 1.0,
             "components": [{"path": str(source), "private_provenance": "archive only"}], "original_path": str(source),
             "results": {"single": {"success": True, "max_score": .91}}} for i in range(count)]
    write_json(evaluation / "report.json", {"stage": "development", "wakeword": "보라야", "models": [{"id": "single"}]})
    write_json(evaluation / "clips.json", rows)
    write_json(evaluation / "receipt.json", {"report_sha256": digest(evaluation / "report.json"), "clips_sha256": digest(evaluation / "clips.json")})
    project = tmp_path / "project.json"
    write_json(project, {"wakeword": "보라야"})
    return evaluation, source, sha256, rows, project


def _data(output):
    return json.loads((output / "results-data.js").read_text(encoding="utf-8").removeprefix("window.KOWW_RESULTS=").removesuffix(";\n"))


@pytest.mark.parametrize("count", [2, 1001, 48000])
def test_reference_export_never_reads_or_copies_audio_and_preserves_every_display_item(tmp_path, monkeypatch, count):
    evaluation, source, sha256, rows, project = _fixture(tmp_path, count)
    original = {name: digest(evaluation / name) for name in ("receipt.json", "report.json", "clips.json")}
    output = tmp_path / "site"
    original_open = Path.open

    def no_audio_reads(path, *args, **kwargs):
        assert path != source, "Studio publication must not open the scored audio"
        return original_open(path, *args, **kwargs)

    def no_copy(*args, **kwargs):
        pytest.fail("Studio reference publication must not copy audio")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", no_audio_reads)
        patch.setattr(site, "_copy_verified_audio", no_copy)
        with site.studio_result_audio(project):
            result = site.export_site(evaluation, output)
    assert result["audio_files"] == count
    assert not (output / "audio").exists()
    assert not list(output.rglob("*.wav"))
    provenance = read_json(output / "provenance.json")
    assert provenance["audio_storage"] == {"mode": "studio_reference_v1", "references": {
        "path": "audio-references.json", "sha256": digest(output / "audio-references.json")}}
    sidecar = read_json(output / "audio-references.json")
    assert sidecar == {"schema_version": 1, "kind": "studio_scored_audio_references_v1",
        "project": str(project.resolve()), "evaluation": str(evaluation.resolve()),
        "evaluation_receipt_sha256": original["receipt.json"], "report_sha256": original["report.json"],
        "clips_sha256": original["clips.json"], "audio": {sha256: {"path": str(source), "sha256": sha256}}}
    payload = _data(output)
    assert payload["viewer"]["page_size"] == 128
    index = read_json(output / "viewer-index.json")
    assert len(index["clips"]) == count
    exported = [row for page in sorted((output / "clips").glob("page-*.json")) for row in read_json(page)["clips"]]
    assert [row["sample_id"] for row in exported] == [row["sample_id"] for row in rows]
    assert all(row["path"] == f"audio/{sha256}.wav" and row["sha256"] == sha256 for row in exported)
    assert all(row["results"] == rows[i]["results"] for i, row in enumerate(exported))
    assert all("components" not in row and "original_path" not in row for row in exported)
    # Private absolute source paths stay in the hash-bound server sidecar.
    for asset in [output / "results-data.js", *sorted((output / "clips").glob("*.json"))]:
        assert str(source) not in asset.read_text(encoding="utf-8")
    assert {name: digest(evaluation / name) for name in original} == original
    assert digest(source) == sha256


def test_default_and_explicit_copy_remain_portable_even_inside_studio_context(tmp_path):
    evaluation, source, sha256, _, project = _fixture(tmp_path)
    default = tmp_path / "default"
    site.export_site(evaluation, default)
    explicit = tmp_path / "explicit"
    with site.studio_result_audio(project):
        site.export_site(evaluation, explicit, audio_mode="copy")
    for output in (default, explicit):
        assert (output / "audio" / f"{sha256}.wav").read_bytes() == source.read_bytes()
        assert "audio_storage" not in read_json(output / "provenance.json")
        assert not (output / "audio-references.json").exists()
    source.unlink()
    assert digest(default / "audio" / f"{sha256}.wav") == sha256


def test_context_is_explicit_scoped_and_restored_after_nested_failure(tmp_path):
    evaluation, _, _, _, project = _fixture(tmp_path)
    other_evaluation, _, _, _, other = _fixture(tmp_path / "other")
    with site.studio_result_audio(project):
        with pytest.raises(RuntimeError):
            with site.studio_result_audio(other):
                site.export_site(other_evaluation, tmp_path / "inner")
                raise RuntimeError("caller failed")
        site.export_site(evaluation, tmp_path / "outer")
    site.export_site(evaluation, tmp_path / "after")
    assert read_json(tmp_path / "inner/audio-references.json")["project"] == str(other.resolve())
    assert read_json(tmp_path / "outer/audio-references.json")["project"] == str(project.resolve())
    assert not (tmp_path / "after/audio-references.json").exists()
    with pytest.raises(ValueError, match="require studio_result_audio"):
        site.export_site(evaluation, tmp_path / "unscoped", audio_mode="studio_reference_v1")
    for invalid in ("../project.json", tmp_path / "missing/project.json", tmp_path / "other.txt"):
        with pytest.raises(ValueError, match="absolute project.json"):
            with site.studio_result_audio(invalid):
                pytest.fail("invalid context entered")
    with pytest.raises(ValueError, match="Unsupported"):
        site.export_site(evaluation, tmp_path / "invalid", audio_mode="symlink")


@pytest.mark.parametrize("name", ["report.json", "clips.json"])
def test_reference_mode_keeps_completed_evaluation_hash_checks(tmp_path, name):
    evaluation, _, _, _, project = _fixture(tmp_path)
    target = evaluation / name
    target.write_bytes(target.read_bytes() + b"\n")
    with site.studio_result_audio(project), pytest.raises(ValueError, match="report/clip binding changed"):
        site.export_site(evaluation, tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_reference_mode_never_overwrites_an_existing_export(tmp_path):
    evaluation, _, _, _, project = _fixture(tmp_path)
    output = tmp_path / "site"
    with site.studio_result_audio(project):
        site.export_site(evaluation, output)
    original = {path: digest(path) for path in output.rglob("*") if path.is_file()}
    with site.studio_result_audio(project), pytest.raises(ValueError, match="immutable"):
        site.export_site(evaluation, output)
    assert {path: digest(path) for path in original} == original


def test_reference_context_refuses_another_projects_evaluation_before_publication(tmp_path):
    evaluation, _, _, _, project = _fixture(tmp_path / "owner")
    _, _, _, _, other = _fixture(tmp_path / "other")
    output = tmp_path / "site"
    with site.studio_result_audio(other), pytest.raises(ValueError, match="does not belong"):
        site.export_site(evaluation, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["report.json", "clips.json", "receipt.json"])
def test_reference_export_detects_metadata_change_immediately_after_hashing(tmp_path, monkeypatch, name):
    evaluation, _, _, _, project = _fixture(tmp_path)
    original_digest = site.digest
    changed = False

    def change_after_hash(path):
        nonlocal changed
        result = original_digest(path)
        if Path(path) == evaluation / name and not changed:
            changed = True
            Path(path).write_bytes(Path(path).read_bytes() + b"\n")
        return result

    monkeypatch.setattr(site, "digest", change_after_hash)
    with site.studio_result_audio(project), pytest.raises(ValueError, match="changed during Studio"):
        site.export_site(evaluation, tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_reference_export_rechecks_metadata_before_publishing_provenance(tmp_path, monkeypatch):
    evaluation, _, _, _, project = _fixture(tmp_path)
    original_files = site.files

    def change_during_display_export(package):
        path = evaluation / "report.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return original_files(package)

    monkeypatch.setattr(site, "files", change_during_display_export)
    with site.studio_result_audio(project), pytest.raises(ValueError, match="changed during Studio"):
        site.export_site(evaluation, tmp_path / "site")
    assert not (tmp_path / "site/provenance.json").exists()
    assert not (tmp_path / "site/audio-references.json").exists()
