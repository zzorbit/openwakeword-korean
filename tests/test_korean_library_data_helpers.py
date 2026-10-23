"""Transport and manifest consolidation preserves the existing caller contracts."""
import hashlib
import json
import os
import pytest

from korean_openwakeword import automatic_sources, commercial_sources, downloads, fma_download, studio_data
from korean_openwakeword.common import digest, read_json


def _asset(payload, algorithm):
    prefix = f"blob {len(payload)}\0".encode() if algorithm == "git_sha1" else b""
    checksum = hashlib.sha256(payload).hexdigest() if algorithm == "sha256" else hashlib.sha1(prefix + payload).hexdigest()
    return {"id": "fixture", "filename": "source.bin", "source_url": "https://example.org/source",
            "size_bytes": len(payload), "checksum_algorithm": algorithm, algorithm: checksum,
            "license": "fixture", "upstream_revision": "fixed"}


class _Response:
    def __init__(self, payload, offset=0, *, interrupted=False):
        self.payload, self.offset, self.interrupted = payload, offset, interrupted
        self.status_code = 206 if offset else 200
        self.headers = {"Content-Range": f"bytes {offset}-{len(payload)-1}/{len(payload)}"} if offset else {}

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass

    def iter_content(self, block):
        if self.interrupted:
            yield self.payload[self.offset:self.offset+4]
            raise RuntimeError("interrupted fixture")
        yield self.payload[self.offset:]


@pytest.mark.parametrize("algorithm", ["sha256", "sha1", "git_sha1"])
@pytest.mark.parametrize("resume", [False, True])
def test_pinned_download_bytes_receipt_and_progress_parity(tmp_path, monkeypatch, algorithm, resume):
    payload = "고정 원본 음원 fixture".encode()
    asset = _asset(payload, algorithm)
    if resume:
        (tmp_path / "source.bin.part").write_bytes(payload[:4])
    calls, progress = [], []
    def get(url, *, headers, **kwargs):
        offset = 4 if resume else 0
        assert url == asset["source_url"]
        assert headers == {"Accept-Encoding": "identity", **({"Range": "bytes=4-"} if offset else {})}
        calls.append(url)
        return _Response(payload, offset)
    monkeypatch.setattr(downloads.requests, "get", get)
    monkeypatch.setattr(downloads, "now", lambda: "fixed-time")
    monkeypatch.setattr(studio_data, "now", lambda: "fixed-time")
    path = studio_data._download(asset, tmp_path, progress.append)
    expected = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                "source_url": asset["source_url"], "completed_at": "fixed-time", "reused": False,
                "content_approved": False, "license": "fixture", "upstream_revision": "fixed"}
    if algorithm != "sha256":
        expected.update(verified_upstream_algorithm=algorithm, verified_upstream_checksum=asset[algorithm])
    assert path.read_bytes() == payload
    assert read_json(path.with_name("source.bin.receipt.json")) == expected
    assert commercial_sources._verify_cached_original(path, asset) == expected["sha256"]
    assert not path.with_name("source.bin.part").exists()
    assert all(row["label"].startswith(("자료 다운로드:", "자료 무결성 검사:", "자료 확인 완료:")) for row in progress)
    studio_data._download(asset, tmp_path)
    assert len(calls) == 1
    assert read_json(path.with_name("source.bin.receipt.json")) == {**expected, "reused": True}


@pytest.mark.parametrize("algorithm", ["sha256", "sha1", "git_sha1"])
def test_ancestor_original_and_receipt_are_not_rewritten(tmp_path, monkeypatch, algorithm):
    payload = b"shared original fixture"
    asset = _asset(payload, algorithm)
    original = tmp_path / "shared/source.bin"
    original.parent.mkdir()
    original.write_bytes(payload)
    source_receipt = original.with_name("source.bin.receipt.json")
    source_receipt.write_bytes(b'{"historical":true}\n')
    before = {path: (digest(path), path.stat().st_mtime_ns) for path in (original, source_receipt)}
    monkeypatch.setattr(downloads.requests, "get", lambda *a, **k: pytest.fail("Shared original must not download"))
    directory = original.parent / "downloads"
    assert studio_data._download(asset, directory) == original
    assert {path: (digest(path), path.stat().st_mtime_ns) for path in before} == before
    receipt = read_json(directory / "source.bin.reused.receipt.json")
    assert receipt["reused"] is True and receipt["path"] == str(original)


@pytest.mark.parametrize("algorithm", ["sha256", "sha1", "git_sha1"])
def test_interrupted_download_keeps_partial_and_resumes_same_original(tmp_path, monkeypatch, algorithm):
    payload = b"same original after interruption"
    asset = _asset(payload, algorithm)
    monkeypatch.setattr(downloads.requests, "get", lambda *a, **k: _Response(payload, interrupted=True))
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        studio_data._download(asset, tmp_path)
    assert (tmp_path / "source.bin.part").read_bytes() == payload[:4]
    assert not (tmp_path / "source.bin").exists()
    assert not (tmp_path / "source.bin.receipt.json").exists()
    def resume(url, *, headers, **kwargs):
        assert headers["Range"] == "bytes=4-"
        return _Response(payload, 4)
    monkeypatch.setattr(downloads.requests, "get", resume)
    assert studio_data._download(asset, tmp_path).read_bytes() == payload


@pytest.mark.parametrize("algorithm", ["sha256", "sha1", "git_sha1"])
def test_changed_original_is_preserved_and_cannot_create_success_receipt(tmp_path, monkeypatch, algorithm):
    payload = b"good fixture"
    asset = _asset(payload, algorithm)
    original = tmp_path / "source.bin"
    original.write_bytes(b"bad! fixture")
    receipt = original.with_name("source.bin.receipt.json")
    receipt.write_bytes(b'{"historical":true}\n')
    before = (original.read_bytes(), receipt.read_bytes())
    monkeypatch.setattr(downloads.requests, "get", lambda *a, **k: pytest.fail("Do not replace corrupted originals"))
    with pytest.raises(ValueError):
        studio_data._download(asset, tmp_path)
    assert (original.read_bytes(), receipt.read_bytes()) == before
    with pytest.raises(ValueError, match="publisher checksum"):
        commercial_sources._verify_cached_original(original, asset)


_WRITERS = [automatic_sources._rows, studio_data._write_rows, fma_download._publish_rows]
_ROWS = [{"이름": "보라야", "id": 2, "nested": [True, None]}, {"id": 1, "value": 0.125}]


def _legacy_bytes(writer, rows):
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    return text.replace("\n", "\n" if writer is fma_download._publish_rows else os.linesep).encode()


@pytest.mark.parametrize("writer", _WRITERS)
def test_manifest_bytes_order_hash_and_existing_file_identity_match_legacy(tmp_path, writer):
    target = tmp_path / "source.jsonl"
    writer(target, (row for row in _ROWS))
    assert target.read_bytes() == _legacy_bytes(writer, _ROWS)
    before = (digest(target), target.stat().st_mtime_ns)
    writer(target, iter(_ROWS))
    assert (digest(target), target.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="changed|differs"):
        writer(target, reversed(_ROWS))
    assert (digest(target), target.stat().st_mtime_ns) == before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("writer", _WRITERS)
def test_manifest_newline_comparison_keeps_each_existing_contract(tmp_path, writer):
    target = tmp_path / "source.jsonl"
    target.write_bytes(_legacy_bytes(fma_download._publish_rows, _ROWS).replace(b"\n", b"\r\n"))
    before = target.read_bytes()
    if writer is fma_download._publish_rows:
        with pytest.raises(ValueError, match="changed"):
            writer(target, iter(_ROWS))
    else:
        writer(target, iter(_ROWS))
    assert target.read_bytes() == before


@pytest.mark.parametrize("writer", _WRITERS)
def test_manifest_empty_placeholder_policy_is_unchanged(tmp_path, writer):
    target = tmp_path / "source.jsonl"
    target.touch()
    if writer is studio_data._write_rows:
        writer(target, iter(_ROWS))
        assert target.read_bytes() == _legacy_bytes(writer, _ROWS)
    else:
        with pytest.raises(ValueError, match="changed"):
            writer(target, iter(_ROWS))
        assert target.read_bytes() == b""


@pytest.mark.parametrize("writer", _WRITERS)
@pytest.mark.parametrize("existing", [False, True])
def test_manifest_interruption_and_invalid_row_never_publish_partial(tmp_path, writer, existing):
    target = tmp_path / "source.jsonl"
    if existing:
        target.write_bytes(_legacy_bytes(writer, _ROWS))
    before = target.read_bytes() if existing else None
    def interrupted():
        yield _ROWS[0]
        raise RuntimeError("interrupted manifest fixture")
    with pytest.raises(RuntimeError, match="interrupted manifest"):
        writer(target, interrupted())
    with pytest.raises(ValueError, match="JSON"):
        writer(target, [_ROWS[0], {"invalid": float("nan")}])
    assert (target.read_bytes() if target.exists() else None) == before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("writer", _WRITERS)
def test_empty_manifest_serialization_is_unchanged(tmp_path, writer):
    target = tmp_path / "source.jsonl"
    writer(target, iter(()))
    assert target.read_bytes() == b""
