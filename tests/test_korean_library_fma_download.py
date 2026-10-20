"""Playlist, original-file verification, quotas and resume are distinct from admission."""
import csv
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json
from korean_openwakeword.fma_download import _lease, _remaining_bytes, download_fma


@pytest.fixture
def source(tmp_path):
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerows([["", "artist", "artist", "track", "track", "track", "track"],
        ["", "id", "name", "license", "duration", "url", "genre_top"],
        ["track_id", "", "", "", "", "", ""]])
    contents, entries = {}, []
    for track in range(1, 5):
        writer.writerow([track, track, f"Artist {track}", "Attribution 4.0", 100,
                         f"https://example.org/track{track}", "Rock"])
        audio = io.BytesIO()
        sf.write(audio, np.random.default_rng(track).normal(0, .04, 8000 * 40), 8000,
                 format="WAV", subtype="PCM_16")
        member = f"fma_full/000/{track:06d}.mp3"
        contents[member] = audio.getvalue()
        entry = zipfile.ZipInfo(member)
        entry.file_size = entry.compress_size = len(contents[member])
        entry.header_offset = track * 1000000
        entry.CRC = __import__("zlib").crc32(contents[member])
        entry.compress_type = zipfile.ZIP_STORED
        entries.append(entry)
    contents["fma_metadata/tracks.csv"] = text.getvalue().encode()
    state = {"song_calls": [], "fail_once": False, "etag": '"source1"', "instances": []}
    class Archive:
        def __init__(self, url, expected_size, *, max_download_bytes, expected_etag=None):
            if expected_etag is not None and expected_etag != state["etag"]:
                raise ValueError("ETag changed")
            self.url, self.size, self.limit = url, expected_size, max_download_bytes
            self.entries, self.downloaded_bytes = entries, 0
            state["instances"].append(self)
        def binding(self):
            return {"url": self.url, "size_bytes": self.size, "etag": state["etag"],
                    "index_sha256": "a"*64, "index_bytes": 100}
        def extract(self, member, target):
            target = Path(target)
            is_song = member.endswith(".mp3")
            if is_song:
                state["song_calls"].append(member)
                if state["fail_once"]:
                    state["fail_once"] = False
                    raise KeyboardInterrupt("interrupted fixture")
            target.parent.mkdir(parents=True, exist_ok=True)
            receipt = target.with_name(target.name + ".fixture.json")
            if target.exists():
                if not receipt.exists() or digest(target) != read_json(receipt)["sha256"]:
                    raise ValueError("Changed original")
            else:
                if self.downloaded_bytes + len(contents[member]) > self.limit:
                    raise ValueError("Fixture transfer budget")
                target.write_bytes(contents[member])
                self.downloaded_bytes += len(contents[member])
                from korean_openwakeword.common import write_json
                write_json(receipt, {"sha256": digest(target)})
            return {"path": str(target), "sha256": digest(target), "bytes": target.stat().st_size}
        def close(self): pass
    return Archive, state


def test_plan_only_freezes_playlist_without_fetching_songs_and_changes_are_refused(source, tmp_path):
    factory, state = source
    output = tmp_path / "selection"
    result = download_fma(output, hours=.02, plan_only=True, archive_factory=factory)
    assert result["state"] == "planned" and state["song_calls"] == []
    before = (output / "selection_plan.json").read_bytes()
    assert download_fma(output, hours=.02, plan_only=True, archive_factory=factory)["plan"] == result["plan"]
    assert (output / "selection_plan.json").read_bytes() == before
    with pytest.raises(ValueError, match="settings"):
        download_fma(output, hours=.03, plan_only=True, archive_factory=factory)
    assert state["song_calls"] == []


def test_actual_headers_use_fixed_reserves_and_complete_receipts_stay_immutable(source, tmp_path):
    factory, state = source
    output = tmp_path / "selection"
    result = download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)
    assert result["track_count"] == 2  # One provider100s main track needs a reserve after the actual40s header.
    assert result["actual_header_hours"] == pytest.approx(80/3600)
    assert result["provider_hours"] == pytest.approx(200/3600)
    assert result["reserve_song_payload_bytes"] > 0
    assert result["estimated_max_song_transfer_bytes"] > result["estimated_song_transfer_bytes"]
    assert not result["audio_reencoded"] and not result["content_approved"]
    assert not result["whole_archive_sha1_verified"] and not result["training_admitted"]
    before = (output / "download_receipt.json").read_bytes()
    state["song_calls"].clear()
    second = download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)
    assert second["reused"] and second["downloaded_bytes_this_run"] == 0
    assert len(state["song_calls"]) == 2  # Every reused original is still verified.
    assert (output / "download_receipt.json").read_bytes() == before


def test_interrupted_run_resumes_same_plan_and_releases_output_lease(source, tmp_path):
    factory, state = source
    output = tmp_path / "selection"
    state["fail_once"] = True
    with pytest.raises(KeyboardInterrupt):
        download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)
    plan = digest(output / "selection_plan.json")
    assert download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)["state"] == "downloaded"
    assert digest(output / "selection_plan.json") == plan


def test_budget_rejected_before_any_song_and_changed_source_stays_blocked(source, tmp_path):
    factory, state = source
    output = tmp_path / "selection"
    download_fma(output, hours=.02, plan_only=True, archive_factory=factory)
    with pytest.raises(ValueError, match="budget"):
        download_fma(output, hours=.02, max_download_bytes=100, archive_factory=factory)
    assert state["song_calls"] == []
    state["etag"] = '"changed"'
    with pytest.raises(ValueError, match="ETag"):
        download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)


def test_changed_original_is_not_reused_or_silently_redownloaded(source, tmp_path):
    factory, _ = source
    output = tmp_path / "selection"
    download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)
    original = next((output / "audio").rglob("*.mp3"))
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed original"):
        download_fma(output, hours=.02, max_download_bytes=8*1024**2, archive_factory=factory)
    assert original.read_bytes() == b"changed"


def test_selective_download_lease_cannot_have_two_writers_and_recovers(source, tmp_path):
    with _lease(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with _lease(tmp_path):
                pytest.fail("Second writer entered")
    with _lease(tmp_path):
        assert True  # OS lock released; persistent lock filename does not block resume.


def test_budget_counts_only_committed_partial_bytes_not_an_uncommitted_tail(tmp_path):
    from korean_openwakeword.common import write_json
    row = {"archive_member": "fma_full/000/000001.mp3", "compressed_bytes": 100,
           "original_bytes": 110, "crc32": "12345678"}
    target = tmp_path / row["archive_member"]
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".compressed.part").write_bytes(b"x" * 90)
    write_json(target.with_name(target.name + ".range-part.json"), {
        "kind": "range_zip_partial_v1", "request": {"member": row["archive_member"],
        "compressed_bytes": 100, "crc32": "12345678"}, "compressed_bytes": 40})
    assert _remaining_bytes([row], tmp_path) == 60 + 65536
