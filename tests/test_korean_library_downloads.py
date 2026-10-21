"""HTTP resume/corruption handling without downloading the production corpus."""
import hashlib
import pytest
from korean_openwakeword.downloads import download_checked, dataset_catalog


class Response:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, block):
        yield self.content


@pytest.mark.parametrize("resume", [False, True])
def test_download_and_reuse(tmp_path, monkeypatch, resume):
    payload = b"the pinned public dataset"
    sha = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "public.tar.gz"
    if resume:
        path.with_name(path.name + ".part").write_bytes(payload[:4])
    def get(url, headers, **kwargs):
        if resume:
            assert headers["Range"] == "bytes=4-"
            return Response(payload[4:], 206, {"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})
        return Response(payload)
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", get)
    first = download_checked("https://example.org/fixture", path, sha, len(payload))
    assert path.read_bytes() == payload and not first["content_approved"]
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: pytest.fail("Reused asset must not redownload"))
    assert download_checked("https://example.org/fixture", path, sha, len(payload))["reused"]


def test_server_ignoring_range_restarts_safely(tmp_path, monkeypatch):
    payload = b"correct full response"
    output = tmp_path / "asset"
    output.with_name("asset.part").write_bytes(payload[:3])
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: Response(payload))
    download_checked("https://example.org/fixture", output, hashlib.sha256(payload).hexdigest(), len(payload))
    assert output.read_bytes() == payload


def test_bad_resume_range_preserves_partial(tmp_path, monkeypatch):
    output = tmp_path / "asset"
    partial = output.with_name("asset.part")
    partial.write_bytes(b"abc")
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: Response(b"xyz", 206, {"Content-Range": "bytes 1-3/6"}))
    with pytest.raises(ValueError, match="range"):
        download_checked("https://example.org/fixture", output, "0" * 64, 6)
    assert partial.read_bytes() == b"abc" and not output.exists()


def test_complete_partial_finalized_without_http(tmp_path, monkeypatch):
    output = tmp_path / "asset"
    payload = b"complete previous transfer"
    output.with_name("asset.part").write_bytes(payload)
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: pytest.fail("Full partial needs only verification"))
    download_checked("https://example.org/fixture", output, hashlib.sha256(payload).hexdigest(), len(payload))
    assert output.read_bytes() == payload


def test_corrupt_bytes_never_published(tmp_path, monkeypatch):
    output = tmp_path / "asset"
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: Response(b"corrupt"))
    with pytest.raises(ValueError, match="altered"):
        download_checked("https://example.org/fixture", output, "0" * 64, 7)
    assert not output.exists()


@pytest.mark.parametrize("resume", [False, True])
def test_oversized_response_stops_before_writing_outside_the_pin(tmp_path, monkeypatch, resume):
    output = tmp_path / "asset"
    partial = output.with_name("asset.part")
    if resume:
        partial.write_bytes(b"ab")
    class ChunkedResponse(Response):
        def iter_content(self, block):
            yield b"cd" if resume else b"ab"
            yield b"efghi"  # Would exceed the pinned total of six bytes.
            pytest.fail("An oversized response must not be consumed further")
    response = ChunkedResponse(b"", 206 if resume else 200,
        {"Content-Range": "bytes 2-5/6"} if resume else {})
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: response)
    with pytest.raises(ValueError, match="exceeds.*pinned"):
        download_checked("https://example.org/fixture", output, "0" * 64, 6)
    assert not output.exists()
    assert partial.read_bytes() == (b"abcd" if resume else b"ab")


@pytest.mark.parametrize("size", [True, 0, -1, 1.5, "6"])
def test_invalid_size_pin_fails_before_creating_a_download_folder(tmp_path, size):
    output = tmp_path / "not_created" / "asset"
    with pytest.raises(ValueError, match="size"):
        download_checked("https://example.org/fixture", output, "0" * 64, size)
    assert not output.parent.exists()


def test_cached_hash_cannot_override_the_pinned_size(tmp_path, monkeypatch):
    output = tmp_path / "asset"
    output.write_bytes(b"correct bytes")
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: pytest.fail("Invalid cached size must stop"))
    with pytest.raises(ValueError, match="Existing download"):
        download_checked("https://example.org/fixture", output, sha, output.stat().st_size + 1)
    assert output.read_bytes() == b"correct bytes"


def test_catalog_contains_actual_pins_no_local_paths():
    catalog = dataset_catalog()
    assert len(catalog["assets"]) == 34
    for row in catalog["assets"].values():
        assert len(row["sha256"]) == 64 and row["size_bytes"] > 0
        assert row["source_url"].startswith("https://") and "path" not in row
