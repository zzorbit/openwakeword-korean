"""Small real WAV/FLAC files exercise the publisher's actual metadata layouts."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.commercial_sources import _PROVIDERS
from korean_openwakeword.common import digest, identity, read_json, write_json
from korean_openwakeword.source_waveform_index import build_waveform_index


def make_source(tmp_path, dataset, *, texts=None, duration=".25"):
    base, audio = tmp_path / "download", tmp_path / "audio"
    audio.mkdir(parents=True)
    if dataset == "mls":
        selection = {"language": "german", "parts": ["10087_10388_000"]}
        prefix = "data/mls_german/train/"
        originals = {prefix + "audio/10087_10388_000.tar.gz": b"original fixture archive"}
        metadata = {
            prefix + "segments.txt": f"10087_10388_000005\thttp://www.archive.org/download/book/chapter.mp3\t12.00\t{12+float(duration)}\n",
            prefix + "transcripts.txt": "10087_10388_000005\tEin deutscher Beispielsatz.\n",
            prefix + "audio_filenames.txt": "10087_10388_000.tar.gz\n",
            "README.md": "Original MLS provider description\n",
        }
        target = audio / "10087_10388_000005.flac"
    else:
        selection = {"subset": "ko000", "parts": [0]}
        prefix = "data/ko000/"
        originals = {prefix + "audio/00000000.tar.gz": b"original fixture archive"}
        metadata = {
            prefix + "duration/00000000.txt": f"YowNukCDSr4 {duration}\n",
            prefix + "text/00000000.json": json.dumps(texts if texts is not None else [
                {"audio_id": "YowNukCDSr4", "text": {"YowNukCDSr4-00000-00000000-00000020": "한국어 예제입니다."}}
            ], ensure_ascii=False),
            "README.md": "Original YODAS2 provider description\n",
        }
        target = audio / "YowNukCDSr4.wav"
    sf.write(target, np.sin(np.arange(4000) * .1) * .1, 16000)
    provider = copy.deepcopy(_PROVIDERS[dataset])
    def asset(path, contents):
        destination = base / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        contents = contents.encode("utf-8") if isinstance(contents, str) else contents
        destination.write_bytes(contents)
        return {"id": path, "path": path, "filename": Path(path).name,
            "source_url": f"https://huggingface.co/datasets/{provider['repo']}/resolve/{provider['revision']}/{path}",
            "size_bytes": len(contents), "checksum_algorithm": "sha256", "sha256": digest(destination),
            "upstream_revision": provider["revision"], "license": provider["license"]}
    resolution = {"schema_version": 1, "kind": "commercial_source_resolution_v1", "source_id": dataset,
        "provider": provider, "selection": selection,
        "assets": [asset(p, content) for p, content in originals.items()],
        "metadata_assets": [asset(p, content) for p, content in metadata.items()],
        "content_approved": False, "prepared_hours": 0, "admitted_hours": 0, "resolved_at": "fixture"}
    resolution["total_bytes"] = sum(a["size_bytes"] for a in resolution["assets"] + resolution["metadata_assets"])
    resolution["resolution_sha256"] = identity({k: v for k, v in resolution.items() if k != "resolved_at"})
    resolution_path = base / "source_resolution.json"
    write_json(resolution_path, resolution)
    receipt = {"kind": "commercial_source_download_v1", "state": "downloaded", "source_id": dataset,
        "resolution_sha256": resolution["resolution_sha256"],
        "source_resolution": {"path": str(resolution_path), "sha256": digest(resolution_path)},
        "files": [{"path": str(base / a["path"]), "source_path": a["path"], "bytes": a["size_bytes"],
                   "sha256": digest(base / a["path"])} for a in resolution["assets"] + resolution["metadata_assets"]]}
    receipt_path = base / "download_receipt.json"
    write_json(receipt_path, receipt)
    return audio, resolution_path, receipt_path


def test_mls_keeps_actual_book_speaker_parent_time_and_original_caption(tmp_path):
    audio, resolution, receipt = make_source(tmp_path, "mls")
    result = build_waveform_index(audio, dataset="mls", metadata=receipt)
    row, = result["rows"]
    assert row["path"] == "10087_10388_000005.flac"
    assert row["language"] == "German"
    assert row["canonical_speaker_id"] == "mls:german:speaker:10087"
    assert row["canonical_book_id"] == "mls:german:book:10388"
    assert row["provider_book_id"] == "10388"
    assert row["canonical_recording_id"] == "librivox:recording:http://www.archive.org/download/book/chapter.mp3"
    assert (row["parent_start_s"], row["parent_end_s"]) == (12, 12.25)
    assert row["text"] == "Ein deutscher Beispielsatz."
    assert row["sha256"] == digest(audio / row["path"])
    assert row["provider_index_adapter"] == "mls"
    assert len(row["provider_metadata_bindings"]) == 3
    assert len(result["provider_metadata_bindings"]) == 6
    assert result["independent_STT_runs"] == 0 and result["content_approved"] is False
    assert build_waveform_index(audio, dataset="mls", metadata=resolution)["rows"][0]["text"] == row["text"]


def test_yodas2_uses_provider_video_not_youtube_or_invented_speaker(tmp_path):
    audio, _, receipt = make_source(tmp_path, "yodas2")
    result = build_waveform_index(audio, dataset="yodas2", metadata=receipt)
    row, = result["rows"]
    assert row["canonical_source_id"] == "yodas:provider-video:YowNukCDSr4"
    assert row["canonical_recording_id"] == row["canonical_source_id"]
    assert row["language"] == "Korean" and row["split"] == "train"
    assert row["provider_index_adapter"] == "yodas2"
    assert len(row["provider_metadata_bindings"]) == 3
    assert len(result["provider_metadata_bindings"]) == 5
    assert not row.get("canonical_speaker_id") and not row.get("original_audio_url")
    assert row["provider_utterances"] == [{"utterance_id": "YowNukCDSr4-00000-00000000-00000020",
        "start_s": 0, "end_s": .2, "text": "한국어 예제입니다."}]
    assert any("not YouTube" in text for text in result["limitations"])


@pytest.mark.parametrize("dataset", ["mls", "yodas2"])
def test_changed_original_metadata_is_rejected_even_with_existing_audio(tmp_path, dataset):
    audio, resolution, receipt = make_source(tmp_path, dataset)
    record = read_json(resolution)
    file = resolution.parent / record["metadata_assets"][0]["path"]
    original = file.read_bytes()
    file.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="checksum"):
        build_waveform_index(audio, dataset=dataset, metadata=receipt)


def test_stale_resolution_or_receipt_binding_cannot_be_used(tmp_path):
    audio, resolution, receipt = make_source(tmp_path, "mls")
    changed = read_json(resolution)
    changed["selection"]["language"] = "french"
    write_json(resolution, changed)
    with pytest.raises(ValueError, match="resolution"):
        build_waveform_index(audio, dataset="mls", metadata=receipt)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "escape", "wrong_size"])
def test_download_receipt_requires_exact_file_bindings(tmp_path, mutation):
    audio, _, receipt = make_source(tmp_path, "yodas2")
    record = read_json(receipt)
    if mutation == "missing":
        record["files"].pop()
    elif mutation == "duplicate":
        record["files"][-1] = record["files"][0]
    elif mutation == "escape":
        record["files"][0]["path"] = str(tmp_path / "outside.tar.gz")
    else:
        record["files"][0]["bytes"] += 1
    write_json(receipt, record)
    with pytest.raises(ValueError):
        build_waveform_index(audio, dataset="yodas2", metadata=receipt)


@pytest.mark.parametrize("dataset", ["mls", "yodas2"])
def test_every_discovered_audio_file_needs_its_own_provider_metadata(tmp_path, dataset):
    audio, _, receipt = make_source(tmp_path, dataset)
    sf.write(audio / "unexpected.wav", np.zeros(40), 16000)
    with pytest.raises(ValueError):
        build_waveform_index(audio, dataset=dataset, metadata=receipt)


def test_duplicate_audio_format_cannot_increase_original_count(tmp_path):
    audio, _, receipt = make_source(tmp_path, "mls")
    sf.write(audio / "10087_10388_000005.wav", np.zeros(4000), 16000)
    with pytest.raises(ValueError, match="repeat"):
        build_waveform_index(audio, dataset="mls", metadata=receipt)


def test_audio_header_duration_must_match_provider_duration(tmp_path):
    audio, _, receipt = make_source(tmp_path, "yodas2", duration="3.25")
    with pytest.raises(ValueError, match="duration"):
        build_waveform_index(audio, dataset="yodas2", metadata=receipt)


@pytest.mark.parametrize("texts", [
    [{"audio_id": "../../bad", "text": {"../../bad-0-0-20": "not safe"}}],
    [{"audio_id": "YowNukCDSr4", "text": {"another-0-0-20": "wrong parent"}}],
    [{"audio_id": "YowNukCDSr4", "text": {"YowNukCDSr4-0-30-20": "reversed boundary"}}],
    [{"audio_id": "YowNukCDSr4", "text": {"YowNukCDSr4-0-0-20": ""}}],
])
def test_invalid_yodas_video_or_utterance_is_rejected(tmp_path, texts):
    audio, _, receipt = make_source(tmp_path, "yodas2", texts=texts)
    with pytest.raises(ValueError):
        build_waveform_index(audio, dataset="yodas2", metadata=receipt)


def test_missing_selected_source_waveform_is_not_silently_ignored(tmp_path):
    audio, _, receipt = make_source(tmp_path, "yodas2")
    next(audio.iterdir()).unlink()
    with pytest.raises(ValueError, match="audio"):
        build_waveform_index(audio, dataset="yodas2", metadata=receipt)


def test_provider_provenance_overrides_cannot_replace_pinned_license(tmp_path):
    audio, _, receipt = make_source(tmp_path, "mls")
    with pytest.raises(ValueError, match="License"):
        build_waveform_index(audio, dataset="mls", metadata=receipt, license="anything")


def test_metadata_only_receipt_does_not_count_as_waveform_download_evidence(tmp_path):
    audio, _, receipt = make_source(tmp_path, "mls")
    record = read_json(receipt)
    record.update(state="metadata_downloaded", metadata_only=True)
    write_json(receipt, record)
    with pytest.raises(ValueError, match="metadata-only"):
        build_waveform_index(audio, dataset="mls", metadata=receipt)


def test_yodas_original_out_of_bounds_annotations_are_preserved_and_reported(tmp_path):
    text = [{"audio_id": "YowNukCDSr4", "text": {
        "YowNukCDSr4-00000-00000000-00000300": "제공자 경계가 실제 영상보다 깁니다."
    }}]
    audio, _, receipt = make_source(tmp_path, "yodas2", texts=text)
    result = build_waveform_index(audio, dataset="yodas2", metadata=receipt)
    row, = result["rows"]
    assert row["duration_s"] == .25
    assert row["provider_utterances"][0]["end_s"] == 3
    assert row["provider_annotation_status"] == "timing_out_of_bounds"
    assert row["provider_annotation_issues"][0]["overrun_s"] == 2.75
    assert result["annotation_quality"]["videos_with_out_of_bounds_annotations"] == 1
    assert result["annotation_quality"]["out_of_bounds_utterances"] == 1
    assert result["annotation_quality"]["timestamps_modified"] is False
    assert result["content_approved"] is False


def test_yodas_empty_provider_transcript_is_explicitly_excluded_without_stt(tmp_path):
    audio, _, receipt = make_source(tmp_path, "yodas2", texts=[{"audio_id": "YowNukCDSr4", "text": {}}])
    result = build_waveform_index(audio, dataset="yodas2", metadata=receipt)
    row, = result["rows"]
    assert row["excluded"] is True
    assert row["provider_annotation_status"] == "missing_provider_transcript"
    assert row["speech"] is None and row["speech_presence"] == "unknown"
    assert row["text"] == "" and row["provider_utterances"] == []
    assert result["annotation_quality"]["sources_without_provider_transcripts"] == 1
    assert result["independent_STT_runs"] == 0
