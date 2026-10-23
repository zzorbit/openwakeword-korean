"""Exact original-byte preservation and two-phase compact-cache recovery."""
import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest
import requests

from korean_openwakeword import compact_sources as module
from korean_openwakeword import commercial_sources as sources
from korean_openwakeword.common import digest, identity, read_json


def archive_bytes(members, *, compressed=False):
    """Reusable tiny TAR fixture; caller may supply real FLAC/WAV bytes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz" if compressed else "w") as target:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            target.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def make_resolution(archives, *, source_id="audioset_unbalanced"):
    """Publisher-shaped resolution plus URL-to-body map, no network needed."""
    provider = sources._PROVIDERS[source_id]
    payloads = {}
    def asset(path, body):
        url = sources._resolve_url(provider, path)
        payloads[url] = body
        return {"path": path, "filename": Path(path).name, "source_url": url,
                "size_bytes": len(body), "checksum_algorithm": "sha256",
                "sha256": hashlib.sha256(body).hexdigest(),
                "upstream_revision": provider["revision"], "license": provider["license"]}
    assets = [asset(f"data/unbal_train{index:03d}.tar", body) for index, body in enumerate(archives)]
    metadata = [asset("data/ontology.json", b"[]"), asset("data/unbalanced_train_segments.csv", b"fixture\n")]
    resolution = {"kind": "commercial_source_resolution_v1", "source_id": source_id,
        "provider": provider, "selection": {"parts": list(range(len(assets)))},
        "assets": assets, "metadata_assets": metadata,
        "total_bytes": sum(a["size_bytes"] for a in assets + metadata), "resolved_at": "fixture"}
    resolution["resolution_sha256"] = identity({k: v for k, v in resolution.items() if k != "resolved_at"})
    sources._validate_resolution(resolution)
    return resolution, payloads


class Transport:
    def __init__(self, payloads):
        self.payloads, self.calls = payloads, []
        self.interrupt_once = None
        self.interrupt_exception = requests.ConnectionError
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        raw = self.payloads[url]
        offset = int(kwargs.get("headers", {}).get("Range", "bytes=0-").split("=")[-1].split("-")[0])
        interrupt = self.interrupt_once == url
        interrupt_exception = self.interrupt_exception
        if interrupt: self.interrupt_once = None
        class Response:
            status_code = 206 if offset else 200
            headers = {"Content-Range": f"bytes {offset}-{len(raw)-1}/{len(raw)}"} if offset else {}
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def raise_for_status(self): pass
            def iter_content(self, size):
                if interrupt:
                    yield raw[offset:offset + len(raw)//2]
                    raise interrupt_exception("fixture interrupted transfer")
                for start in range(offset, len(raw), size): yield raw[start:start+size]
        return Response()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw = {"corpus/a.flac": b"original-one"*23, "corpus/b.wav": b"original-two"*17}
    resolution, payloads = make_resolution([archive_bytes(raw)])
    transport = Transport(payloads)
    monkeypatch.setattr(requests, "get", transport.get)
    return tmp_path / "compact", resolution, raw, transport


def prepare(setup, **overrides):
    directory, resolution, _, _ = setup
    return module.prepare_compact_source(resolution, directory,
        max_download_bytes=overrides.get("download", resolution["total_bytes"]),
        max_extract_bytes=overrides.get("retained", 1024**2))


def test_exact_members_and_stable_resume_without_archives_or_network(setup, monkeypatch):
    directory, resolution, raw, transport = setup
    result = prepare(setup)
    archive = directory / "originals" / resolution["assets"][0]["path"]
    assert not archive.exists()
    assert result["retired_archive_bytes"] == resolution["assets"][0]["size_bytes"]
    assert result["original_storage_bytes"] == sum(map(len, raw.values())) + sum(a["size_bytes"] for a in resolution["metadata_assets"])
    for name, data in raw.items():
        assert (Path(result["audio_directory"]) / "part_0000" / name).read_bytes() == data
    assert all(Path(binding["path"]).is_file() and digest(binding["path"]) == binding["sha256"] for binding in result["receipts"])
    before = digest(result["compact_receipt"]["path"])
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Completed compact cache never downloads again"))
    resumed = prepare(setup, retained=2*1024**2)
    assert resumed["resumed"] and digest(resumed["compact_receipt"]["path"]) == before
    assert module.verify_compact_source(directory)["original_storage_bytes"] == result["original_storage_bytes"]


@pytest.mark.parametrize("after_unlink", [False, True])
def test_crash_before_or_after_owned_archive_retirement_recovers_without_download(setup, monkeypatch, after_unlink):
    directory, resolution, raw, transport = setup
    real = module._retire_archive
    def stop(archive, part, request, originals):
        assert Path(part["extraction"]["path"]).is_file()
        assert (directory / "parts/part_0000.json").is_file()
        if after_unlink: real(archive, part, request, originals)
        raise RuntimeError("fixture crash after durable member preservation")
    monkeypatch.setattr(module, "_retire_archive", stop)
    with pytest.raises(RuntimeError, match="fixture crash"):
        prepare(setup)
    archive = directory / "originals" / resolution["assets"][0]["path"]
    assert archive.exists() is (not after_unlink)
    monkeypatch.setattr(module, "_retire_archive", real)
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Existing members/metadata resume locally"))
    result = prepare(setup, retained=3*1024**2)
    assert not archive.exists() and not result["resumed"]
    assert module.verify_compact_source(directory)["resumed"]


def test_crash_after_extraction_before_part_commit_keeps_archive_and_fixed_extraction_allowance(setup, monkeypatch):
    directory, resolution, _, transport = setup
    real = module._durable_json
    def stop(path, record):
        if Path(path).parent.name == "parts": raise RuntimeError("fixture before part commit")
        return real(path, record)
    monkeypatch.setattr(module, "_durable_json", stop)
    with pytest.raises(RuntimeError, match="part commit"): prepare(setup)
    archive = directory / "originals" / resolution["assets"][0]["path"]
    assert archive.is_file()
    extraction = directory / "audio/part_0000/extraction_receipt.json"
    before = digest(extraction)
    monkeypatch.setattr(module, "_durable_json", real)
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Partial preservation reuses verified archive"))
    prepare(setup, retained=4*1024**2)
    assert digest(extraction) == before and not archive.exists()


def test_interrupted_transfer_uses_original_range_resume_before_retirement(setup):
    directory, resolution, raw, transport = setup
    transport.interrupt_once = resolution["assets"][0]["source_url"]
    # A process shutdown remains a durable stop. A temporary connection error
    # is now automatically recovered and is covered separately below.
    transport.interrupt_exception = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt): prepare(setup)
    assert not list((directory / "parts").glob("*.json"))
    archive = directory / "originals" / resolution["assets"][0]["path"]
    partial = archive.with_name(archive.name + ".part")
    assert partial.is_file() and not archive.exists()
    assert partial.stat().st_size == resolution["assets"][0]["size_bytes"] // 2
    result = prepare(setup)
    assert any("Range" in kwargs.get("headers", {}) for _, kwargs in transport.calls)
    assert result["retired_archive_bytes"] > 0
    assert all((Path(result["audio_directory"]) / "part_0000" / name).read_bytes() == body for name, body in raw.items())
    assert module.verify_compact_source(directory)["resumed"]


def test_temporary_connection_recovers_before_verified_original_retirement(setup, monkeypatch):
    directory, resolution, raw, transport = setup
    url = resolution["assets"][0]["source_url"]
    transport.interrupt_once = url
    original_get, resumed = transport.get, []
    def observed_get(request_url, **kwargs):
        if request_url == url and "Range" in kwargs.get("headers", {}):
            # Neither a part commit nor archive retirement may be inferred
            # merely from the existence of an interrupted transfer.
            assert not list((directory / "parts").glob("*.json"))
            partial = directory / "originals" / (resolution["assets"][0]["path"] + ".part")
            assert partial.stat().st_size == resolution["assets"][0]["size_bytes"] // 2
            resumed.append(kwargs["headers"]["Range"])
        return original_get(request_url, **kwargs)
    monkeypatch.setattr(requests, "get", observed_get)
    result = prepare(setup)
    assert resumed == [f"bytes={resolution['assets'][0]['size_bytes']//2}-"]
    assert result["retired_archive_bytes"] == resolution["assets"][0]["size_bytes"]
    assert all((Path(result["audio_directory"]) / "part_0000" / name).read_bytes() == body for name, body in raw.items())
    assert module.verify_compact_source(directory)["resumed"]


@pytest.mark.parametrize("target", ["raw", "metadata", "part", "resolution"])
def test_completed_raw_metadata_or_evidence_tampering_fails_without_network(setup, monkeypatch, target):
    directory, resolution, raw, _ = setup
    prepare(setup)
    targets = {"raw": directory / "audio/part_0000" / next(iter(raw)),
        "metadata": directory / "originals" / resolution["metadata_assets"][0]["path"],
        "part": directory / "parts/part_0000.json", "resolution": directory / "originals/source_resolution.json"}
    targets[target].write_bytes(b"changed")
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Tampered source must not be silently replaced"))
    with pytest.raises((ValueError, KeyError)): prepare(setup)


def test_preexisting_archive_is_preserved_and_accounted(setup):
    directory, resolution, raw, transport = setup
    asset = resolution["assets"][0]
    archive = directory / "originals" / asset["path"]
    archive.parent.mkdir(parents=True); archive.write_bytes(transport.payloads[asset["source_url"]])
    before = digest(archive)
    result = prepare(setup)
    assert archive.is_file() and digest(archive) == before and result["retired_archive_bytes"] == 0
    assert result["original_storage_bytes"] == asset["size_bytes"] + sum(map(len, raw.values())) + sum(a["size_bytes"] for a in resolution["metadata_assets"])


def test_two_archives_are_materialized_and_retired_one_at_a_time(tmp_path, monkeypatch):
    resolution, payloads = make_resolution([archive_bytes({"a.flac": b"a"}), archive_bytes({"b.flac": b"b"})])
    transport = Transport(payloads); directory = tmp_path / "compact"
    real_get = transport.get
    def observed(url, **kwargs):
        if url == resolution["assets"][1]["source_url"]:
            assert not (directory / "originals" / resolution["assets"][0]["path"]).exists()
        return real_get(url, **kwargs)
    monkeypatch.setattr(requests, "get", observed)
    result = module.prepare_compact_source(resolution, directory, max_download_bytes=resolution["total_bytes"], max_extract_bytes=100000)
    assert result["temporary_archive_peak_bytes"] == max(a["size_bytes"] for a in resolution["assets"])
    assert result["retired_archive_bytes"] == sum(a["size_bytes"] for a in resolution["assets"])


def test_unsafe_member_fails_before_any_archive_retirement(tmp_path, monkeypatch):
    resolution, payloads = make_resolution([archive_bytes({"../outside.wav": b"bad"})])
    transport = Transport(payloads); monkeypatch.setattr(requests, "get", transport.get)
    directory = tmp_path / "compact"
    with pytest.raises(ValueError, match="escapes|Unsafe"):
        module.prepare_compact_source(resolution, directory, max_download_bytes=resolution["total_bytes"], max_extract_bytes=100000)
    assert (directory / "originals" / resolution["assets"][0]["path"]).is_file()
    assert not (tmp_path / "outside.wav").exists()


def test_budget_failure_precedes_download_or_retirement(setup):
    _, resolution, _, transport = setup
    with pytest.raises(ValueError, match="download byte cap"):
        prepare(setup, download=resolution["total_bytes"]-1)
    assert not transport.calls


def test_mismatched_archive_pin_preserves_partial_and_never_retires(setup):
    directory, resolution, _, transport = setup
    url = resolution["assets"][0]["source_url"]
    transport.payloads[url] = b"broken" + transport.payloads[url][6:]
    with pytest.raises(ValueError, match="Incomplete/altered"):
        prepare(setup)
    assert (directory / "originals" / (resolution["assets"][0]["path"] + ".part")).is_file()
    assert not (directory / "compact_receipt.json").exists()


@pytest.mark.parametrize("algorithm", ["sha1", "git_sha1"])
def test_original_sha1_metadata_pins_work_without_studio_specific_id(setup, algorithm):
    _, resolution, _, transport = setup
    for asset in resolution["metadata_assets"]:
        body = transport.payloads[asset["source_url"]]
        asset.pop("sha256")
        asset["checksum_algorithm"] = algorithm
        asset[algorithm] = hashlib.sha1((f"blob {len(body)}\0".encode() if algorithm == "git_sha1" else b"") + body).hexdigest()
    resolution["resolution_sha256"] = identity({k: v for k, v in resolution.items() if k not in ("resolved_at", "resolution_sha256")})
    result = prepare(setup)
    assert module.verify_compact_source(Path(result["compact_receipt"]["path"]).parent)["resumed"]


def test_archive_appearing_after_request_is_preserved_as_external(setup, monkeypatch):
    directory, resolution, _, transport = setup
    real = module._durable_json
    archive = directory / "originals" / resolution["assets"][0]["path"]
    def insert(path, record):
        real(path, record)
        if Path(path).name == "compact_request.json":
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(transport.payloads[resolution["assets"][0]["source_url"]])
    monkeypatch.setattr(module, "_durable_json", insert)
    result = prepare(setup)
    assert archive.is_file() and result["retired_archive_bytes"] == 0
    assert read_json(directory / "compact_request.json")["owned_archives"][resolution["assets"][0]["path"]] is True
    assert read_json(directory / "downloads/part_0000.json")["retire_eligible"] is False


def test_crash_before_acquisition_commit_never_guesses_archive_ownership(setup, monkeypatch):
    directory, resolution, _, _ = setup
    real = module._durable_json
    def stop(path, record):
        if Path(path).parent.name == "downloads":
            raise RuntimeError("fixture before acquisition commit")
        real(path, record)
    monkeypatch.setattr(module, "_durable_json", stop)
    with pytest.raises(RuntimeError, match="acquisition commit"): prepare(setup)
    archive = directory / "originals" / resolution["assets"][0]["path"]
    assert archive.is_file()
    monkeypatch.setattr(module, "_durable_json", real)
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Verified local archive must be preserved and reused"))
    result = prepare(setup)
    assert archive.is_file() and result["retired_archive_bytes"] == 0


def test_identical_external_replacement_after_commit_is_preserved(setup, monkeypatch):
    directory, _, _, _ = setup
    real = module._retire_archive
    def replace(archive, part, request, originals):
        body = archive.read_bytes()
        archive.replace(archive.with_name("preserved-before-replacement.tar"))
        archive.write_bytes(body)
        real(archive, part, request, originals)
    monkeypatch.setattr(module, "_retire_archive", replace)
    with pytest.raises(ValueError, match="replaced"): prepare(setup)
    assert (directory / "originals/data/unbal_train000.tar").is_file()
    assert (directory / "originals/data/preserved-before-replacement.tar").is_file()
    assert not (directory / "compact_receipt.json").exists()


def test_incomplete_preservation_checks_every_raw_before_retirement(setup, monkeypatch):
    directory, resolution, raw, _ = setup
    real = module._retire_archive
    def stop(*args): raise RuntimeError("fixture before retirement")
    monkeypatch.setattr(module, "_retire_archive", stop)
    with pytest.raises(RuntimeError): prepare(setup)
    (directory / "audio/part_0000" / next(iter(raw))).write_bytes(b"corrupt")
    monkeypatch.setattr(module, "_retire_archive", real)
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Corrupt raw source fails closed"))
    with pytest.raises(ValueError, match="missing or changed"): prepare(setup)
    assert (directory / "originals" / resolution["assets"][0]["path"]).is_file()


def test_new_archive_becoming_shared_before_acquisition_is_preserved(setup, monkeypatch):
    directory, resolution, _, _ = setup
    real = module._asset_file
    external = directory.parent / "external-original.tar"
    def materialize(asset, originals, progress=None):
        record = real(asset, originals, progress)
        if asset in resolution["assets"]:
            os.link(record["path"], external)
        return record
    monkeypatch.setattr(module, "_asset_file", materialize)
    result = prepare(setup)
    assert external.is_file() and (directory / "originals/data/unbal_train000.tar").is_file()
    assert result["retired_archive_bytes"] == 0


def test_internal_directory_symlink_is_not_followed(tmp_path):
    root = tmp_path / "compact"; root.mkdir()
    actual = root / "actual"; actual.mkdir()
    try:
        (root / "linked").symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create test symlinks")
    with pytest.raises(ValueError, match="link"):
        module._inside(root, "linked/a.wav")


def test_windows_junction_in_cache_or_member_path_is_rejected(tmp_path, monkeypatch):
    # Model a Windows directory junction without creating a path to another
    # drive or depending on this account's link-creation privileges.
    monkeypatch.setattr(Path, "is_junction", lambda path: path.name == "junction", raising=False)
    root = tmp_path / "compact"
    with pytest.raises(ValueError, match="links"):
        module._directory(root / "junction/cache")
    with pytest.raises(ValueError, match="link"):
        module._inside(root, "junction/a.wav")


@pytest.mark.parametrize("protected", [".codex", ".git"])
def test_resolved_cache_target_is_checked_for_protected_directories(tmp_path, monkeypatch, protected):
    target = tmp_path / "compact"
    real = Path.resolve
    def resolved(path, *args, **kwargs):
        return tmp_path / protected / "data" if path == target else real(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolved)
    with pytest.raises(ValueError, match="Resolved compact originals"):
        module._directory(target)
