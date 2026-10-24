"""Global queue concurrency and call-scoped verification, without networking."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import threading

import pytest

from korean_openwakeword import material_downloads as module
from korean_openwakeword import evaluation_rir, studio_data
from korean_openwakeword.downloads import verify_checksum


def checksum(data, algorithm="sha256"):
    if algorithm == "git_sha1":
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return getattr(hashlib, algorithm)(data).hexdigest()


def test_reentering_an_owner_cannot_reuse_a_previous_calls_verification(tmp_path):
    path = tmp_path / "original.bin"
    data = b"same bytes must be verified again for another call"
    path.write_bytes(data)
    expected = checksum(data)
    owner = module.MaterialDownloads(tmp_path)
    with owner:
        verify_checksum(path, "sha256", expected, verification_cache=owner.verification_cache)
        assert owner._remember(path, {"algorithm": "sha256", "expected": expected, "size_bytes": len(data)})
        assert module.borrowed_download(path, "sha256", expected, len(data)) == path
    with owner:
        assert module.borrowed_download(path, "sha256", expected, len(data)) is None
        assert not owner.verification_cache.matches(path, "sha256", expected)


def asset(name, host, data):
    return {"id": name, "filename": name + ".bin", "source_url": f"https://{host}/{name}",
            "size_bytes": len(data), "checksum_algorithm": "sha256", "sha256": checksum(data)}


def isolated_plan(monkeypatch, workspace, *, base, mit=(), common=(), payloads):
    """Keep the real planner and queue; replace only provider discovery/work."""
    monkeypatch.setenv("KOWW_DOWNLOAD_WORKERS", "3")
    monkeypatch.setattr("korean_openwakeword.downloads.requests.get", lambda *a, **k: pytest.fail("No network in queue fixtures"))
    mit_dir = workspace / "shared" / "MIT"
    requests = [{"id": item["filename"], "request": (item["source_url"], mit_dir / item["filename"], item["sha256"], item["size_bytes"])} for item in mit]
    monkeypatch.setattr(studio_data, "_mit_download_plan", lambda shared: (mit_dir, requests))
    recipe = {"recipe_version": 4, "evaluation_assets": []}
    if common:
        recipe.update(recipe_version=6, common_evaluation_catalog={"archives": list(common)})
        rir_data = b"common RIR archive fixture"
        pin = asset("common-rir", "rir.example", rir_data)
        payloads[pin["filename"]] = rir_data
        monkeypatch.setattr(evaluation_rir, "common_rir_catalog", lambda: {"archive": pin})
    return list(base), recipe, {"targets": {"generic_hours": 1, "korean_hours": 1}}


def complete_fixture(owner, item, report, payloads):
    """Return a pinned worker result after real local checksum validation."""
    path = Path(item["target"])
    data = payloads[path.name]
    reused = path.exists()
    if reused:
        assert path.read_bytes() == data
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        report({"phase": "download", "done": len(data), "total": len(data), "unit": "bytes",
                "network_bytes": len(data), "network_active": False})
    verify_checksum(path, item["algorithm"], item["expected"], progress=report,
                    verification_cache=owner.verification_cache)
    return {"path": str(path), "reused": reused}


@pytest.mark.parametrize("algorithm", ["sha256", "sha1", "git_sha1", "md5"])
def test_verified_borrowing_is_bound_to_context_and_full_pins(tmp_path, algorithm):
    path = tmp_path / "existing.bin"
    data = b"existing verified original"
    path.write_bytes(data)
    expected = checksum(data, algorithm)
    owner = module.MaterialDownloads(tmp_path)
    verify_checksum(path, algorithm, expected, verification_cache=owner.verification_cache)
    owner._remember(path, {"algorithm": algorithm, "expected": expected, "size_bytes": len(data)})
    assert module.borrowed_download(path, algorithm, expected, len(data)) is None
    with owner:
        assert module.borrowed_download(path, algorithm, expected, len(data)) == path.resolve()
        assert module.borrowed_download(path, algorithm, "0" * len(expected), len(data)) is None
        assert module.borrowed_download(path, algorithm, expected, len(data) + 1) is None
        assert module.borrowed_download(path, "unverified-algorithm", expected, len(data)) is None
        other = tmp_path / "other.bin"
        other.write_bytes(data)
        assert module.borrowed_download(other, algorithm, expected, len(data)) is None
        with module.MaterialDownloads(tmp_path):
            assert module.borrowed_download(path, algorithm, expected, len(data)) is None
        assert module.borrowed_download(path, algorithm, expected, len(data)) == path.resolve()
    assert module.borrowed_download(path, algorithm, expected, len(data)) is None
    with module.MaterialDownloads(tmp_path):
        assert module.borrowed_download(path, algorithm, expected, len(data)) is None


@pytest.mark.parametrize("mutation", ["content", "metadata", "replace", "remove"])
def test_a_changed_file_identity_invalidates_a_previously_verified_borrow(tmp_path, mutation):
    path, data = tmp_path / "original.bin", b"original pinned data"
    path.write_bytes(data)
    expected = checksum(data)
    with module.MaterialDownloads(tmp_path) as owner:
        verify_checksum(path, "sha256", expected, verification_cache=owner.verification_cache)
        owner._remember(path, {"algorithm": "sha256", "expected": expected, "size_bytes": len(data)})
        assert module.borrowed_download(path, "sha256", expected, len(data)) == path.resolve()
        if mutation == "content":
            path.write_bytes(data + b" changed")
        elif mutation == "metadata":
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        elif mutation == "replace":
            replacement = tmp_path / "replacement.bin"
            replacement.write_bytes(data)
            replacement.replace(path)
        else:
            path.unlink()
        assert module.borrowed_download(path, "sha256", expected, len(data)) is None


def test_unverified_file_cannot_be_promoted_into_call_scoped_verification(tmp_path):
    path, data = tmp_path / "not-verified.bin", b"unverified bytes"
    path.write_bytes(data)
    expected = checksum(data)
    with module.MaterialDownloads(tmp_path) as owner:
        owner._remember(path, {"algorithm": "sha256", "expected": expected, "size_bytes": len(data)})
        assert module.borrowed_download(path, "sha256", expected, len(data)) is None


def test_global_queue_starts_other_hosts_before_a_slow_training_host_finishes(tmp_path, monkeypatch):
    payloads = {f"{name}.bin": (name + " original bytes").encode() for name in ("a1", "a2", "a3", "m1", "m2", "c1", "c2")}
    base = [asset(name, "slow.example", payloads[name + ".bin"]) for name in ("a1", "a2", "a3")]
    mit = [asset(name, "mit.example", payloads[name + ".bin"]) for name in ("m1", "m2")]
    common = [asset(name, "common.example", payloads[name + ".bin"]) for name in ("c1", "c2")]
    arguments = isolated_plan(monkeypatch, tmp_path, base=base, mit=mit, common=common, payloads=payloads)
    before_inputs = deepcopy(arguments)
    cached = tmp_path / "shared" / "downloads" / "a2.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payloads[cached.name])
    receipt = cached.with_name(cached.name + ".receipt.json")
    receipt.write_bytes(b"old immutable cache receipt")
    cached_before = (cached.read_bytes(), cached.stat().st_mtime_ns, receipt.read_bytes())
    release_slow, common_started = threading.Event(), threading.Event()
    initial = threading.Barrier(3)
    lock = threading.Lock()
    starts, finishes, queue_order, queued_paths, returned_order, progress = [], [], [], [], [], []
    real_download_many = module.download_many

    def observe_queue(items, worker, **kwargs):
        queue_order.extend(item["id"] for item in items)
        queued_paths.extend(Path(item["target"]).name for item in items)
        results = real_download_many(items, worker, **kwargs)
        returned_order.extend(Path(row["path"]).name for row in results)
        return results

    def worker(owner, item, report):
        with lock:
            starts.append(item["id"])
            first_batch = len(starts) <= 3
        if first_batch:
            initial.wait(timeout=10)
        if item["id"].startswith("common:"):
            common_started.set()
        if item["source_url"].startswith("https://slow.example/"):
            assert release_slow.wait(timeout=10)
        result = complete_fixture(owner, item, report, payloads)
        with lock:
            finishes.append(item["id"])
        return result

    monkeypatch.setattr(module, "download_many", observe_queue)
    monkeypatch.setattr(module.MaterialDownloads, "_acquire", worker)
    owner = module.MaterialDownloads(tmp_path, progress.append)
    with owner, ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(owner.prepare, *arguments)
        try:
            assert common_started.wait(timeout=10), "The common corpus was blocked behind all Train files"
            assert set(starts[:3]) == {"base:a1", "mit:m1.bin", "common:c1"}
            assert not any(item.startswith("base:") for item in finishes)
        finally:
            release_slow.set()
        future.result(timeout=10)
        # Queue output order follows submission, despite deliberately delayed
        # completion of its first file. The input provider lists remain intact.
        assert queue_order == ["base:a1", "mit:m1.bin", "common:c1", "common:rir_archive",
                               "base:a2", "mit:m2.bin", "common:c2", "base:a3"]
        assert returned_order == queued_paths
        assert list(owner.base_paths) == [row["id"] for row in base]
        for row in base:
            path = owner.base_paths[row["id"]]
            assert path.read_bytes() == payloads[row["filename"]]
            assert module.borrowed_download(path, "sha256", row["sha256"], row["size_bytes"]) == path.resolve()
    assert arguments == before_inputs
    assert cached_before == (cached.read_bytes(), cached.stat().st_mtime_ns, receipt.read_bytes())
    assert any(record.get("completed") for record in progress)
    final = next(record for record in reversed(progress) if record.get("completed"))
    reused = next(row for row in final["files"] if row["id"] == "base:a2")
    assert reused["state"] == "reused" and reused["network_bytes"] == 0


def test_completed_early_file_changed_before_queue_drains_is_not_borrowed(tmp_path, monkeypatch):
    payloads = {"fast.bin": b"first verified source", "slow.bin": b"later verified source"}
    base = [asset("fast", "first.example", payloads["fast.bin"])]
    mit = [asset("slow", "second.example", payloads["slow.bin"])]
    arguments = isolated_plan(monkeypatch, tmp_path, base=base, mit=mit, payloads=payloads)
    verified = threading.Event()

    def worker(owner, item, report):
        if item["id"] == "base:fast":
            result = complete_fixture(owner, item, report, payloads)
            verified.set()
            return result
        assert verified.wait(timeout=10)
        fast = tmp_path / "shared" / "downloads" / "fast.bin"
        fast.write_bytes(b"changed while another source was downloading")
        return complete_fixture(owner, item, report, payloads)

    monkeypatch.setattr(module.MaterialDownloads, "_acquire", worker)
    with module.MaterialDownloads(tmp_path) as owner:
        with pytest.raises(ValueError, match="changed before preparation"):
            owner.prepare(*arguments)
        fast = tmp_path / "shared" / "downloads" / "fast.bin"
        assert fast.read_bytes() == b"changed while another source was downloading"
        assert module.borrowed_download(fast, "sha256", base[0]["sha256"], base[0]["size_bytes"]) is None
