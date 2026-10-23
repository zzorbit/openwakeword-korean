"""The pinned MIT source has 270 WAVs, not 271 provider entries."""
import hashlib
import io
from pathlib import Path
import struct
import wave

import pytest

from korean_openwakeword import studio_data as module
from korean_openwakeword.common import digest, read_json, write_json


PINNED_REVISION = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"
API_URL = ("https://huggingface.co/api/datasets/davidscripka/"
           f"MIT_environmental_impulse_responses/tree/{PINNED_REVISION}/16khz?limit=1000")
RESOLVE_ROOT = ("https://huggingface.co/datasets/davidscripka/"
                f"MIT_environmental_impulse_responses/resolve/{PINNED_REVISION}/")


def payload(number):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack("<8h", number + 1, 0, 0, 0, 0, 0, 0, 0))
    return stream.getvalue()


def inventory(count=270):
    return [{"type": "file", "path": f"16khz/room_{number:04d}.wav",
             "size": len(payload(number)),
             "lfs": {"oid": hashlib.sha256(payload(number)).hexdigest()}}
            for number in range(count)]


class Response:
    def __init__(self, rows, links=None):
        self.rows, self.links = rows, links or {}
    def raise_for_status(self):
        pass
    def json(self):
        return self.rows


def mock_downloads(monkeypatch, rows, directory):
    expected = {RESOLVE_ROOT + row["path"]: row for row in rows}
    calls = []
    def checked(url, target, checksum, size, *, progress=None):
        assert url in expected
        row = expected[url]
        assert Path(target) == directory / row["path"]
        assert checksum == row["lfs"]["oid"] and size == row["size"]
        number = int(Path(row["path"]).stem.split("_")[1])
        body = payload(number)
        assert len(body) == size and hashlib.sha256(body).hexdigest() == checksum
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
        assert digest(target) == checksum
        if progress:
            progress({"phase": "hash", "done": size, "total": size, "unit": "bytes"})
        calls.append((url, str(target), checksum, size))
        return {"path": str(target), "sha256": checksum, "bytes": size}
    monkeypatch.setattr(module, "download_checked", checked)
    return calls


def test_all_recipe_sizes_use_the_same_actual_mit_count_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("Offline recipe"))
    assert module.MIT_REV == PINNED_REVISION
    for preset in ("check", "test", "full"):
        recipe = module.material_recipe(preset, workspace=tmp_path)
        assert recipe["additional_download"]["files"] == 270
        assert recipe["additional_download"]["upstream_revision"] == PINNED_REVISION


def test_exact_270_wavs_pass_pinned_checks_and_resume_metadata_without_network(tmp_path, monkeypatch):
    rows = inventory()
    # Provider listing entries that are not WAV audio never count as RIRs.
    provider_rows = rows + [{"type": "directory", "path": "16khz/other"},
                            {"type": "file", "path": "16khz/README.md", "size": 1}]
    fetches = []
    def get(url, **kwargs):
        assert url == API_URL and kwargs == {"timeout": (30, 120)}
        fetches.append(url)
        return Response(provider_rows)
    monkeypatch.setattr(module.requests, "get", get)
    calls = mock_downloads(monkeypatch, rows, tmp_path / "mit_rir")
    batches = []
    real_download_many = module.download_many
    def download_batch(items, worker, **kwargs):
        batches.append(list(items))
        return real_download_many(items, worker, **kwargs)
    monkeypatch.setattr(module, "download_many", download_batch)
    progress = []
    result = module._fetch_mit(tmp_path, progress=progress.append)
    assert result == tmp_path / "mit_rir/16khz"
    assert len(calls) == 270 and fetches == [API_URL]
    assert len(list(result.glob("*.wav"))) == 270
    assert progress[-1]["done"] == progress[-1]["total"] == 270
    assert [item["id"] for item in batches[0]] == [row["path"] for row in rows]
    assert any((event.get("current_event") or {}).get("phase") == "hash" for event in progress)
    manifest = tmp_path / "mit_rir/provider_tree.json"
    before = manifest.read_bytes()
    assert read_json(manifest)["entries"] == provider_rows
    file_hashes = {path.name: digest(path) for path in result.glob("*.wav")}
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("Use pinned metadata cache"))
    assert module._fetch_mit(tmp_path) == result
    assert sorted(calls[:270]) == sorted(calls[270:])  # Existing WAVs still reach checksum verification.
    assert batches[0] == batches[1]
    assert manifest.read_bytes() == before
    assert {path.name: digest(path) for path in result.glob("*.wav")} == file_hashes


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_incomplete_extra_or_duplicate_wavs_fail_before_audio_download(tmp_path, monkeypatch, change):
    rows = inventory()
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows = inventory(271)
    else:
        rows[-1] = rows[0].copy()
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response(rows))
    monkeypatch.setattr(module, "download_checked", lambda *a, **k: pytest.fail("Invalid inventory downloads no audio"))
    with pytest.raises(ValueError, match="exactly 270 WAVs"):
        module._fetch_mit(tmp_path)
    assert not list((tmp_path / "mit_rir").rglob("*.wav"))


def test_changed_cached_revision_fails_without_overwriting_metadata_or_network(tmp_path, monkeypatch):
    manifest = tmp_path / "mit_rir/provider_tree.json"
    write_json(manifest, {"source_url": API_URL.replace(PINNED_REVISION, "old-revision"),
                          "entries": inventory(), "retrieved_at": "fixture"})
    before = manifest.read_bytes()
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("Do not replace changed cache"))
    monkeypatch.setattr(module, "download_checked", lambda *a, **k: pytest.fail("No audio after revision mismatch"))
    with pytest.raises(ValueError, match="MIT cached provider revision changed"):
        module._fetch_mit(tmp_path)
    assert manifest.read_bytes() == before


def test_mit_batch_checks_all_missing_bytes_before_audio_and_counts_partial_resume(tmp_path, monkeypatch):
    rows = inventory()
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response(rows))
    monkeypatch.setattr(module, "download_many", lambda *args, **kwargs: pytest.fail("Space preflight must finish first"))
    expected = sum(row["size"] for row in rows)
    usage_type = type(module.shutil.disk_usage(tmp_path))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: usage_type(expected * 2, expected, expected - 1))
    with pytest.raises(ValueError, match="space for the pinned MIT"):
        module._fetch_mit(tmp_path)
    target = tmp_path / "mit_rir" / rows[0]["path"]
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(b"x")
    batches = []
    monkeypatch.setattr(module, "download_many", lambda items, worker, **kwargs: batches.append(list(items)))
    assert module._fetch_mit(tmp_path) == tmp_path / "mit_rir/16khz"
    assert len(batches) == 1 and len(batches[0]) == 270


def test_paginated_inventory_fails_before_cache_commit_or_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response(
        inventory(), {"next": {"url": "https://example.org/unexpected-next"}}))
    monkeypatch.setattr(module, "download_checked", lambda *a, **k: pytest.fail("Incomplete page downloads no audio"))
    with pytest.raises(ValueError, match="unexpectedly paginated"):
        module._fetch_mit(tmp_path)
    assert not (tmp_path / "mit_rir/provider_tree.json").exists()


@pytest.mark.parametrize("change", ["missing_sha", "invalid_sha", "oversized", "zero", "negative", "bool", "invalid_last_sha"])
def test_real_file_hash_and_size_guards_remain_enabled(tmp_path, monkeypatch, change):
    rows = inventory()
    if change == "missing_sha":
        rows[0].pop("lfs")
    elif change == "invalid_sha":
        rows[0]["lfs"]["oid"] = "not-a-real-sha256"
    elif change == "invalid_last_sha":
        rows[-1]["lfs"]["oid"] = "not-a-real-sha256"
    elif change in ("zero", "negative", "bool"):
        rows[0]["size"] = {"zero": 0, "negative": -1, "bool": True}[change]
    else:
        rows[0]["size"] = 8 * 1024**2 + 1
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: Response(rows))
    monkeypatch.setattr(module, "download_checked", lambda *a, **k: pytest.fail("Invalid first WAV downloads no audio"))
    with pytest.raises(ValueError, match="bounded LFS SHA256"):
        module._fetch_mit(tmp_path)
