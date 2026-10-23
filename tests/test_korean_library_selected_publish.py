"""Selective acquisition must never replace an original published concurrently."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from korean_openwakeword import selected_zip as module
from korean_openwakeword.common import digest
from test_korean_library_selected_zip import fixture, Network


@pytest.mark.parametrize("from_local_archive", [False, True])
def test_acquire_preserves_a_concurrent_original_and_verified_temporary(
        tmp_path, monkeypatch, from_local_archive):
    group, bodies, originals = fixture()
    network = Network(bodies)
    destination, transaction = tmp_path / "originals", tmp_path / "transaction"
    local, source_sha = None, None
    if from_local_archive:
        source = tmp_path / "existing-publisher.zip"
        source.write_bytes(bodies[group["volumes"][0]["source_url"]])
        source_sha = digest(source)
        local = {group["volumes"][0]["id"]: source}
    publish = module._publish_original
    attempts = []
    foreign = b"another process published this original first"
    def concurrent_publish(temporary, output):
        temporary, output = Path(temporary), Path(output)
        assert not output.exists()
        assert temporary.read_bytes() == originals["audio/first.wav"]
        # Precisely the old exists()/replace() gap, with a real filesystem
        # original appearing immediately before the atomic publication call.
        output.write_bytes(foreign)
        attempts.append((temporary, output))
        return publish(temporary, output)
    monkeypatch.setattr(module, "_publish_original", concurrent_publish)
    with pytest.raises((FileExistsError, ValueError)):
        module.acquire_selected(group, destination, transaction,
            session_factory=network.session, local_volumes=local, workers=1)
    assert len(attempts) == 1
    temporary, output = attempts[0]
    assert output.read_bytes() == foreign
    assert temporary.read_bytes() == originals["audio/first.wav"]
    assert not list(transaction.glob("receipt_*.json"))
    assert not list((transaction / "chunks").glob("*.json"))
    if from_local_archive:
        assert digest(source) == source_sha and not network.requests
    else:
        assert list((transaction / "ranges").glob("*.part"))
    assert all(response.closed for response in network.responses)
    assert all(session.closed for session in network.sessions)


def test_native_publish_succeeds_and_retires_only_its_temporary(tmp_path):
    temporary, output = tmp_path / "verified.part", tmp_path / "original.wav"
    temporary.write_bytes(b"verified original bytes")
    module._publish_original(temporary, output)
    assert output.read_bytes() == b"verified original bytes" and not temporary.exists()


def test_posix_link_publication_is_atomic_and_never_overwrites(tmp_path, monkeypatch):
    temporary, output = tmp_path / "verified.part", tmp_path / "original.wav"
    temporary.write_bytes(b"verified original bytes")
    output.write_bytes(b"existing original")
    def forbidden_rename(*args):
        pytest.fail("POSIX no-clobber publication must not fall back to rename/replace")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", link=os.link,
                                                    rename=forbidden_rename, replace=forbidden_rename))
    with pytest.raises(FileExistsError):
        module._publish_original(temporary, output)
    assert output.read_bytes() == b"existing original"
    assert temporary.read_bytes() == b"verified original bytes"
    output.unlink()
    module._publish_original(temporary, output)
    assert output.read_bytes() == b"verified original bytes" and not temporary.exists()


def test_unsupported_link_filesystem_fails_without_overwrite_fallback(tmp_path, monkeypatch):
    temporary, output = tmp_path / "verified.part", tmp_path / "original.wav"
    temporary.write_bytes(b"verified original bytes")
    def unavailable_link(*args):
        raise OSError("Filesystem does not support hard links")
    def forbidden_rename(*args):
        pytest.fail("Unsupported hard links must fail closed, without an overwrite-capable fallback")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", link=unavailable_link,
                                                    rename=forbidden_rename, replace=forbidden_rename))
    with pytest.raises(OSError, match="does not support"):
        module._publish_original(temporary, output)
    assert not output.exists() and temporary.read_bytes() == b"verified original bytes"
