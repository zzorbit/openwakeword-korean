"""Network-free invariants for bounded FMA download/conversion overlap."""
from pathlib import Path
import threading

import pytest

from korean_openwakeword.canonical_audio import canonicalize_prepared_source, pcm_identity
from korean_openwakeword.common import digest, read_json, read_rows
from korean_openwakeword.fma_download import download_fma
from korean_openwakeword.fma_streaming import (prepare_fma_streaming_source,
                                              verify_fma_streaming_source)
from korean_openwakeword.preparation import create_source_index

# Reuse the existing tiny original ZIP/member fixture; it has no network access
# and deliberately makes provider durations differ from actual audio headers.
from test_korean_library_fma_download import source  # noqa: F401


def _prepare(directory, factory, **kwargs):
    return prepare_fma_streaming_source(directory, target_hours=.02,
        max_download_bytes=8*1024**2, max_extract_bytes=16*1024**2,
        archive_factory=factory, **kwargs)


def test_same_playlist_pcm_logical_ids_and_legacy_provider_order(source, tmp_path):
    factory, state = source
    legacy = download_fma(tmp_path / "legacy", hours=.02,
        max_download_bytes=8*1024**2, archive_factory=factory)
    audio = Path(legacy["audio_directory"])
    index = tmp_path / "legacy_index.jsonl"
    create_source_index(audio, output=index, dataset="fma",
        metadata=legacy["metadata"]["path"], source_url="https://github.com/mdeff/fma")
    before = canonicalize_prepared_source({"source": {"directory": str(audio), "adapter": "fma"},
        "rows": read_rows(index)}, tmp_path / "legacy_flac")["prepared"]
    state["song_calls"].clear()
    after = _prepare(tmp_path / "stream", factory)
    assert len(after["rows"]) == 2
    assert len(state["song_calls"]) == 2  # No unused reserve prefetch.
    assert not list((tmp_path / "stream" / "tracks").rglob("*.mp3"))
    for original, canonical in zip(before["rows"], after["rows"], strict=True):
        def semantics(row):
            value = {key: item for key, item in row.items()
                     if key not in ("path", "provider_metadata_bindings", "canonical_audio")}
            value["provider_metadata_hashes"] = [r["sha256"] for r in row["provider_metadata_bindings"]]
            value["canonical_policy"] = row["canonical_audio"]["policy"]
            value["original_info"] = row["canonical_audio"]["original_info"]
            return value
        assert semantics(canonical) == semantics(original)
        assert pcm_identity(Path(before["source"]["directory"]) / original["path"]) == pcm_identity(
            Path(after["source"]["directory"]) / canonical["path"])
    plan = read_json(after["receipts"][0]["path"])
    assert after["acquired_original_peak_bytes"] <= sum(
        row["original_bytes"] for row in plan["selection"]["selected"] + plan["selection"]["reserve"])
    assert after["fma_acquisition_policy"] == "per_track_flac_prefetch_v1"


def test_complete_reuse_is_local_and_tampering_fails_closed(source, tmp_path):
    factory, state = source
    output = tmp_path / "stream"
    first = _prepare(output, factory)
    receipt_bytes = Path(first["canonical_receipt"]["path"]).read_bytes()
    state["song_calls"].clear()
    def forbidden(*args, **kwargs):
        pytest.fail("Completed streaming cache touched the network")
    second = _prepare(output, forbidden)
    assert second["resumed"] and state["song_calls"] == []
    assert receipt_bytes == Path(first["canonical_receipt"]["path"]).read_bytes()
    audio = Path(second["source"]["directory"]) / second["rows"][0]["path"]
    audio.write_bytes(audio.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_fma_streaming_source(output)


def test_prefetch_overlaps_conversion_and_is_single_download_worker(source, tmp_path, monkeypatch):
    factory, state = source
    import korean_openwakeword.fma_streaming as streaming
    started = threading.Event()
    original = streaming.canonicalize_prepared_source
    original_stage = streaming.stage_canonical_source
    stage_calls = 0
    worker_names = set()
    def instrumented_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls > 1:
            started.set()
        return original_stage(*args, **kwargs)
    class Instrumented(factory):
        def extract(self, member, target):
            if member.endswith(".mp3"):
                worker_names.add(threading.current_thread().name)
            return super().extract(member, target)
    def convert(prepared, directory):
        assert started.wait(30), "Next song was not prefetched during current conversion"
        return original(prepared, directory)
    monkeypatch.setattr(streaming, "canonicalize_prepared_source", convert)
    monkeypatch.setattr(streaming, "stage_canonical_source", instrumented_stage)
    _prepare(tmp_path / "stream", Instrumented)
    assert len(worker_names) == 1
    assert next(iter(worker_names)).startswith("fma-range")


def test_interrupted_conversion_keeps_durable_download_and_resumes(source, tmp_path, monkeypatch):
    factory, state = source
    import korean_openwakeword.fma_streaming as streaming
    original = streaming.canonicalize_prepared_source
    calls = 0
    def interrupted(prepared, directory):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("fixture conversion interruption")
        return original(prepared, directory)
    monkeypatch.setattr(streaming, "canonicalize_prepared_source", interrupted)
    output = tmp_path / "stream"
    with pytest.raises(KeyboardInterrupt):
        _prepare(output, factory)
    assert len(state["song_calls"]) == 2
    first_receipt = next(output.rglob("canonical_receipt.json"))
    first_hash = digest(first_receipt)
    monkeypatch.setattr(streaming, "canonicalize_prepared_source", original)
    state["song_calls"].clear()
    resumed = _prepare(output, factory)
    assert state["song_calls"] == []
    assert digest(first_receipt) == first_hash
    assert len(resumed["rows"]) == 2


def test_preexisting_original_is_preserved_and_old_namespace_is_refused(source, tmp_path):
    factory, _ = source
    output = tmp_path / "legacy"
    download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)
    before = {str(p): digest(p) for p in output.rglob("*.mp3")}
    with pytest.raises(ValueError, match="new policy namespace"):
        _prepare(output, factory)
    assert {str(p): digest(p) for p in output.rglob("*.mp3")} == before


def test_small_storage_bound_stops_before_conversion_without_deleting_raw(source, tmp_path):
    factory, _ = source
    with pytest.raises(ValueError, match="storage cap"):
        prepare_fma_streaming_source(tmp_path / "stream", target_hours=.02,
            max_download_bytes=8*1024**2, max_extract_bytes=100,
            archive_factory=factory)
    assert not list((tmp_path / "stream").rglob("*.mp3"))
    assert not list((tmp_path / "stream").rglob("canonical_receipt.json"))


def test_producer_failure_joins_worker_and_keeps_completed_current_song(source, tmp_path):
    factory, state = source
    failed = False
    class Interrupted(factory):
        def extract(self, member, target):
            nonlocal failed
            if member.endswith(".mp3") and len(state["song_calls"]) == 1 and not failed:
                failed = True
                raise ValueError("fixture second-song transfer failed")
            return super().extract(member, target)
    output = tmp_path / "stream"
    with pytest.raises(ValueError, match="second-song"):
        _prepare(output, Interrupted)
    assert len(list(output.rglob("canonical_receipt.json"))) == 1
    first = state["song_calls"][0]
    state["song_calls"].clear()
    result = _prepare(output, factory)
    assert len(result["rows"]) == 2
    assert first not in state["song_calls"]


def test_shared_original_hardlink_is_never_retired(source, tmp_path):
    import os
    factory, _ = source
    shared = tmp_path / "shared_original.mp3"
    shared_target = []
    class Shared(factory):
        def extract(self, member, target):
            result = super().extract(member, target)
            if member.endswith(".mp3") and not shared.exists():
                os.link(target, shared)
                shared_target.append(Path(target))
            return result
    result = _prepare(tmp_path / "stream", Shared)
    assert shared_target[0].exists()
    assert digest(shared) == digest(shared_target[0])
    assert result["retired_original_bytes"] < sum(read_json(Path(p["path"]))["prepared"]["fma_track"]["bytes"]
        for p in result["receipts"] if Path(p["path"]).name == "canonical_receipt.json")


def test_per_track_receipt_and_incomplete_archive_pin_tampering_is_rejected(source, tmp_path, monkeypatch):
    factory, state = source
    import korean_openwakeword.fma_streaming as streaming
    original = streaming.canonicalize_prepared_source
    def fail(*args):
        raise ValueError("fixture stop before conversion")
    monkeypatch.setattr(streaming, "canonicalize_prepared_source", fail)
    output = tmp_path / "stream"
    with pytest.raises(ValueError, match="stop before conversion"):
        _prepare(output, factory)
    state["etag"] = '"changed-source"'
    monkeypatch.setattr(streaming, "canonicalize_prepared_source", original)
    with pytest.raises(ValueError, match="ETag"):
        _prepare(output, factory)
    state["etag"] = '"source1"'
    completed = _prepare(output, factory)
    song_receipt = next(output.rglob("song_receipt.json"))
    song_receipt.write_bytes(song_receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_fma_streaming_source(output)
