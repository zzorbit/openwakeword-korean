"""Commercial source routing checks use tiny publisher-shaped fixtures."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from korean_openwakeword import commercial_sources as sources
from korean_openwakeword.common import identity


def publisher_file(path, content=b"audio"):
    checksum = hashlib.sha256(content).hexdigest()
    return {"type": "file", "path": path, "size": len(content),
            "lfs": {"oid": checksum, "size": len(content)}}


class Response:
    def __init__(self, rows, next_url=None):
        self.rows = rows
        self.links = {"next": {"url": next_url}} if next_url else {}

    def raise_for_status(self):
        pass

    def json(self):
        return self.rows


class Publisher:
    def __init__(self, trees):
        self.trees = trees
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        for suffix, response in self.trees.items():
            if url.endswith(suffix):
                return response
        raise AssertionError("Unexpected publisher request: " + url)


def yodas_publisher():
    return Publisher({
        "/data/ko000/audio?limit=1000": Response([publisher_file("data/ko000/audio/00000000.tar.gz")]),
        "/data/ko000/duration?limit=1000": Response([publisher_file("data/ko000/duration/00000000.txt")]),
        "/data/ko000/text?limit=1000": Response([publisher_file("data/ko000/text/00000000.json")]),
        "?limit=1000": Response([publisher_file("README.md")]),
    })


def yodas_resolution():
    return sources.resolve_commercial_source("yodas2", selection={"subset": "ko000", "parts": [0]},
                                             session=yodas_publisher())


def reseal(record):
    record["resolution_sha256"] = identity({k: v for k, v in record.items()
                                           if k not in ("resolved_at", "resolution_sha256")})


def test_plan_scales_pools_without_changing_batch_or_supply():
    plan = sources.commercial_negative_plan()
    assert plan["generic_hours"] == 7000
    assert plan["korean_hours"] == 3000
    assert plan["total_hours"] == 10000
    assert (plan["id"], plan["version"]) == ("commercial_negative_v2", 2)
    assert sum(c["target_hours"] for c in plan["components"] if c["pool"] == "generic_negative") == 7000
    assert [c["target_hours"] for c in plan["components"]] == [3000, 3000, 1000, 3000]
    assert [c["sources"][0]["id"] for c in plan["components"]] == ["audioset_unbalanced", "mls", "fma_full", "yodas2"]
    assert plan["content_approved"] is False
    assert plan["batch_changed"] is False
    scaled = sources.commercial_negative_plan(3500, 1)
    assert [c["target_hours"] for c in scaled["components"]] == [1500, 1500, 500, 1]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "7000"])
def test_invalid_hour_budgets_rejected(bad):
    with pytest.raises(ValueError):
        sources.commercial_negative_plan(bad, 3000)


def test_catalog_has_public_origins_but_no_preparation_claim():
    catalog = sources.commercial_source_catalog()
    assert {s["id"] for s in catalog["sources"]} == {"audioset_unbalanced", "mls", "yodas2", "fma_large", "fma_full"}
    assert all(s["availability"] == "public_source_requires_preparation" for s in catalog["sources"])
    assert not any(s["content_approved"] for s in catalog["sources"])


def test_yodas_resolution_binds_audio_duration_text_and_readme():
    publisher = yodas_publisher()
    record = sources.resolve_commercial_source("yodas2", selection={"subset": "ko000", "parts": [0]},
                                               session=publisher)
    assert len(record["assets"]) == 1
    assert len(record["metadata_assets"]) == 3
    assert record["total_bytes"] == 20
    assert record["admitted_hours"] == record["prepared_hours"] == 0
    assert all("/" + sources._PROVIDERS["yodas2"]["revision"] + "/" in a["source_url"]
               for a in record["assets"] + record["metadata_assets"])
    assert sources._validate_resolution(record)
    assert len(publisher.calls) == 4


def test_audio_resolution_uses_unbalanced_only_and_keeps_provider_labels():
    publisher = Publisher({
        "/data?limit=1000": Response([publisher_file(p) for p in (
            "data/unbal_train000.tar", "data/unbalanced_train_segments.csv", "data/ontology.json")]),
        "?limit=1000": Response([publisher_file("README.md")]),
    })
    record = sources.resolve_commercial_source("audioset_unbalanced", selection={"parts": [0]}, session=publisher)
    assert record["assets"][0]["path"] == "data/unbal_train000.tar"
    assert len(record["metadata_assets"]) == 3


def test_mls_resolution_follows_pagination_and_preserves_source_metadata():
    provider = sources._PROVIDERS["mls"]
    base = "data/mls_german/train"
    next_url = f"https://huggingface.co/api/datasets/{provider['repo']}/tree/{provider['revision']}/{base}/audio?cursor=next"
    publisher = Publisher({
        "/audio?limit=1000": Response([], next_url),
        "/audio?cursor=next": Response([publisher_file(base + "/audio/10087_10388_000.tar.gz")]),
        "/train?limit=1000": Response([publisher_file(base + "/" + f)
            for f in ("segments.txt", "transcripts.txt", "audio_filenames.txt")]),
        "?limit=1000": Response([publisher_file("README.md")]),
    })
    record = sources.resolve_commercial_source("mls", selection={"language": "german", "parts": ["10087_10388_000"]},
                                               session=publisher)
    assert len(record["metadata_assets"]) == 4
    assert len(publisher.calls) == 4


@pytest.mark.parametrize("link", ["https://evil.example/next", "http://huggingface.co/next"])
def test_remote_pagination_cannot_leave_the_pinned_provider(link):
    publisher = Publisher({"/data/ko000/audio?limit=1000": Response([], link)})
    with pytest.raises(ValueError, match="pagination"):
        sources.resolve_commercial_source("yodas2", selection={"subset": "ko000", "parts": [0]}, session=publisher)


def test_publisher_inventory_rejects_repeated_paths():
    publisher = yodas_publisher()
    entry = publisher_file("data/ko000/audio/00000000.tar.gz")
    publisher.trees["/data/ko000/audio?limit=1000"] = Response([entry, entry])
    with pytest.raises(ValueError, match="repeats"):
        sources.resolve_commercial_source("yodas2", selection={"subset": "ko000", "parts": [0]}, session=publisher)


@pytest.mark.parametrize("source_id,selection", [
    ("unknown", {"parts": [0]}), ("yodas2", {"subset": "../../secrets", "parts": [0]}),
    ("yodas2", {"subset": "ko000", "parts": []}), ("yodas2", {"subset": "ko000", "parts": [0, 0]}),
    ("audioset_unbalanced", {"parts": [870]}), ("audioset_unbalanced", {"parts": [True]}),
    ("mls", {"language": "english", "parts": ["1_2_0"]}),
    ("mls", {"language": "german", "parts": ["../../elsewhere"]}),
    ("fma_large", {"parts": ["all"]}),
    ("fma_large", {"parts": ["full"]}), ("fma_full", {"parts": ["large"]}),
])
def test_source_resolution_needs_valid_explicit_parts(source_id, selection):
    with pytest.raises(ValueError):
        sources.resolve_commercial_source(source_id, selection=selection)


def test_fma_keeps_original_publisher_sha1_and_large_archive_size():
    record = sources.resolve_commercial_source("fma_large", selection={"parts": ["large"]})
    assert record["assets"][0]["size_bytes"] == 100306112191
    assert record["assets"][0]["sha1"] == "497109f4dd721066b5ce5e5f250ec604dc78939e"
    assert record["metadata_assets"][0]["filename"] == "fma_metadata.zip"
    assert sources._validate_resolution(record)


def test_full_route_pins_original_archive_without_substituting_the_large_subset():
    record = sources.resolve_commercial_source("fma_full", selection={"parts": ["full"]})
    asset = record["assets"][0]
    assert asset["path"] == "fma_full.zip"
    assert asset["source_url"] == "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
    assert asset["size_bytes"] == 943642698636
    assert asset["sha1"] == "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab"
    assert asset["checksum_algorithm"] == "sha1"
    assert record["metadata_assets"][0]["filename"] == "fma_metadata.zip"
    assert record["total_bytes"] == 944001111077
    assert record["admitted_hours"] == record["prepared_hours"] == 0
    assert sources._validate_resolution(record)


@pytest.mark.parametrize("mutation", ["variant", "missing_audio", "missing_metadata", "checksum_algorithm"])
def test_fma_full_resolution_rejects_variant_and_file_coverage_tampering(mutation):
    record = sources.resolve_commercial_source("fma_full", selection={"parts": ["full"]})
    if mutation == "variant":
        record["selection"]["parts"] = ["large"]
    elif mutation == "missing_audio":
        record["assets"] = []
    elif mutation == "missing_metadata":
        record["metadata_assets"] = []
    else:
        record["assets"][0].update(checksum_algorithm="sha256", sha256="0" * 64)
    record["total_bytes"] = sum(a["size_bytes"] for a in record["assets"] + record["metadata_assets"])
    reseal(record)
    with pytest.raises(ValueError, match="FMA"):
        sources._validate_resolution(record)


def test_full_archive_small_budget_is_refused_before_network_or_output_creation(tmp_path):
    record = sources.resolve_commercial_source("fma_full", selection={"parts": ["full"]})
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="exceed max_download_bytes"):
        sources.download_commercial_source(record, destination, max_download_bytes=2 * 1024**3)
    assert not destination.exists()


def test_download_rejects_budget_before_any_network_or_directory_creation(tmp_path):
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="max_download_bytes"):
        sources.download_commercial_source(yodas_resolution(), destination, max_download_bytes=19)
    assert not destination.exists()


def test_download_rejects_codex_storage(tmp_path):
    with pytest.raises(ValueError, match="Codex"):
        sources.download_commercial_source(yodas_resolution(), tmp_path / ".codex" / "data", max_download_bytes=20)


def test_download_rejects_low_free_space(monkeypatch, tmp_path):
    monkeypatch.setattr(sources.shutil, "disk_usage", lambda _: SimpleNamespace(free=19))
    with pytest.raises(ValueError, match="Insufficient"):
        sources.download_commercial_source(yodas_resolution(), tmp_path, max_download_bytes=20)
    assert not (tmp_path / "source_resolution.json").exists()


def test_resolved_asset_tampering_is_rejected_even_with_a_new_local_hash(tmp_path):
    record = yodas_resolution()
    record["assets"][0]["source_url"] = "https://evil.example/audio"
    reseal(record)
    with pytest.raises(ValueError, match="URL"):
        sources.download_commercial_source(record, tmp_path, max_download_bytes=20)
    record = yodas_resolution()
    record["assets"][0]["path"] = "../escaped.wav"
    reseal(record)
    with pytest.raises(ValueError, match="escapes"):
        sources.download_commercial_source(record, tmp_path, max_download_bytes=20)


def test_download_calls_existing_hash_verifier_and_never_marks_admitted(monkeypatch, tmp_path):
    calls = []

    def download(url, target, checksum, size):
        calls.append((url, checksum, size))
        target = Path(target)
        reused = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"audio")
        assert hashlib.sha256(target.read_bytes()).hexdigest() == checksum
        return {"path": str(target), "sha256": checksum, "bytes": size, "reused": reused}

    monkeypatch.setattr(sources, "download_checked", download)
    record = yodas_resolution()
    receipt = sources.download_commercial_source(record, tmp_path, max_download_bytes=20)
    assert len(calls) == 4
    assert receipt["state"] == "downloaded"
    assert receipt["bytes"] == 20
    assert receipt["prepared_hours"] == receipt["admitted_hours"] == 0
    assert receipt["content_approved"] is False
    assert receipt["Final_accessed"] is False
    assert json.loads((tmp_path / "source_resolution.json").read_text(encoding="utf-8"))["selection"] == record["selection"]
    original_receipt = (tmp_path / "download_receipt.json").read_bytes()
    resumed = sources.download_commercial_source(record, tmp_path, max_download_bytes=20)
    assert resumed == receipt
    assert (tmp_path / "download_receipt.json").read_bytes() == original_receipt
    assert len(calls) == 4  # Verify existing bytes directly; do not rewrite per-file receipts.
    assert json.loads((tmp_path / "download_progress.json").read_text(encoding="utf-8"))["state"] == "verified_reuse"


def test_a_download_directory_cannot_change_selections(monkeypatch, tmp_path):
    (tmp_path / "source_resolution.json").write_text('{"resolution_sha256": "different"}')
    with pytest.raises(ValueError, match="resolved commercial source|different source selection"):
        sources.download_commercial_source(yodas_resolution(), tmp_path, max_download_bytes=20)


class BytesResponse:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code = content, status
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.mark.parametrize("source_id,part", [("fma_large", "large"), ("fma_full", "full")])
def test_original_fma_sha1_downloader_returns_receipts_and_resumes(monkeypatch, tmp_path, source_id, part):
    content = b"audio"
    sha1 = hashlib.sha1(content).hexdigest()
    filename = source_id + ".zip"
    monkeypatch.setattr(sources, "_FMA_FILES", {filename: (5, sha1), "fma_metadata.zip": (5, sha1)})
    from korean_openwakeword import studio_data
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs.get("headers", {}))
        if "Range" in kwargs.get("headers", {}):
            assert kwargs["headers"]["Range"] == "bytes=2-"
            return BytesResponse(b"dio", status=206, headers={"Content-Range": "bytes 2-4/5"})
        return BytesResponse(content)

    monkeypatch.setattr(studio_data.requests, "get", get)
    (tmp_path / (filename + ".part")).write_bytes(b"au")
    record = sources.resolve_commercial_source(source_id, selection={"parts": [part]})
    receipt = sources.download_commercial_source(record, tmp_path, max_download_bytes=10)
    assert receipt["bytes"] == 10
    assert len(receipt["files"]) == 2
    assert all(row["verified_upstream_algorithm"] == "sha1" for row in receipt["files"])
    assert (tmp_path / filename).read_bytes() == content
    assert receipt["content_approved"] is False
    assert calls[0]["Range"] == "bytes=2-"
    assert (tmp_path / "download_receipt.json").exists()
    original_receipt = (tmp_path / "download_receipt.json").read_bytes()
    assert sources.download_commercial_source(record, tmp_path, max_download_bytes=10) == receipt
    assert (tmp_path / "download_receipt.json").read_bytes() == original_receipt
    assert len(calls) == 2


def test_fma_full_metadata_only_never_requests_the_near_terabyte_archive(monkeypatch, tmp_path):
    from korean_openwakeword import studio_data
    content = b"audio"
    sha1 = hashlib.sha1(content).hexdigest()
    monkeypatch.setattr(sources, "_FMA_FILES", {"fma_full.zip": (943642698636, "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab"),
                                              "fma_metadata.zip": (5, sha1)})
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        assert url.endswith("/fma_metadata.zip")
        return BytesResponse(content)
    monkeypatch.setattr(studio_data.requests, "get", get)
    record = sources.resolve_commercial_source("fma_full", selection={"parts": ["full"]})
    receipt = sources.download_commercial_source(record, tmp_path, max_download_bytes=5, metadata_only=True)
    assert len(calls) == 1
    assert receipt["bytes"] == 5
    assert receipt["state"] == "metadata_downloaded" and receipt["downloaded_audio_files"] == 0
    assert not (tmp_path / "fma_full.zip").exists()
    assert not receipt["content_approved"]


def test_git_blob_readme_checksum_is_verified(monkeypatch, tmp_path):
    publisher = yodas_publisher()
    blob_sha = hashlib.sha1(b"blob 5\0audio").hexdigest()
    publisher.trees["?limit=1000"] = Response([{"type": "file", "path": "README.md", "size": 5, "oid": blob_sha}])
    record = sources.resolve_commercial_source("yodas2", selection={"subset": "ko000", "parts": [0]}, session=publisher)
    from korean_openwakeword import studio_data
    monkeypatch.setattr(studio_data.requests, "get", lambda *a, **kw: BytesResponse(b"audio"))
    receipt = sources.download_commercial_source(record, tmp_path, max_download_bytes=20)
    assert receipt["files"][-1]["verified_upstream_algorithm"] == "git_sha1"
    assert receipt["files"][-1]["verified_upstream_checksum"] == blob_sha


def test_metadata_only_download_has_separate_receipt_and_fetches_no_audio(monkeypatch, tmp_path):
    paths = []

    def download(url, target, checksum, size):
        paths.append(str(target))
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"audio")
        return {"path": str(target), "sha256": checksum, "bytes": size, "reused": False}

    monkeypatch.setattr(sources, "download_checked", download)
    record = yodas_resolution()
    receipt = sources.download_commercial_source(record, tmp_path, max_download_bytes=15, metadata_only=True)
    assert len(paths) == 3
    assert not any(path.endswith("tar.gz") for path in paths)
    assert receipt["state"] == "metadata_downloaded"
    assert receipt["downloaded_audio_files"] == 0
    assert (tmp_path / "metadata_download_receipt.json").exists()
    assert not (tmp_path / "download_receipt.json").exists()
    original_receipt = (tmp_path / "metadata_download_receipt.json").read_bytes()
    assert sources.download_commercial_source(record, tmp_path, max_download_bytes=15, metadata_only=True) == receipt
    assert (tmp_path / "metadata_download_receipt.json").read_bytes() == original_receipt
    assert len(paths) == 3


@pytest.mark.parametrize("mutation", ["file", "receipt_hash", "receipt_bytes", "source_binding", "approval"])
def test_completed_cache_or_receipt_changes_are_rejected(monkeypatch, tmp_path, mutation):
    def download(url, target, checksum, size):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"audio")
        return {"path": str(target), "sha256": checksum, "bytes": size, "reused": False}

    monkeypatch.setattr(sources, "download_checked", download)
    record = yodas_resolution()
    result = sources.download_commercial_source(record, tmp_path, max_download_bytes=20)
    receipt_path = tmp_path / "download_receipt.json"
    if mutation == "file":
        Path(result["files"][0]["path"]).write_bytes(b"other")
    else:
        if mutation == "receipt_hash":
            result["files"][0]["sha256"] = "0" * 64
        elif mutation == "receipt_bytes":
            result["files"][0]["bytes"] += 1
        elif mutation == "source_binding":
            result["source_resolution"]["sha256"] = "0" * 64
        else:
            result["content_approved"] = True
        receipt_path.write_text(json.dumps(result), encoding="utf-8")
    before = receipt_path.read_bytes()
    with pytest.raises(ValueError):
        sources.download_commercial_source(record, tmp_path, max_download_bytes=20)
    assert receipt_path.read_bytes() == before
