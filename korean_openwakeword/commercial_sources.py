"""Public waveform source routes for the commercial-size Negative budget.

Targets are preparation budgets, never downloaded/admitted hours.  Source
resolution reads only publisher metadata. Downloads require explicit archive
parts and a byte budget; they do not extract, classify, admit or train audio.
"""
from __future__ import annotations

import math
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.parse import quote, urlparse

import requests

from .common import digest, identity, now, read_json, write_json
from .downloads import download_checked


_PROVIDERS = {
    "audioset_unbalanced": {
        "name": "AudioSet unbalanced", "repo": "agkphysics/AudioSet",
        "revision": "5a2fa42a1506470d275a47ff8e1fdac5b364e6ef",
        "source_url": "https://huggingface.co/datasets/agkphysics/AudioSet",
        "license": "Provider CC BY 4.0; original YouTube audio rights must also be checked",
        "description": "생활·환경·말소리·음악이 포함된 혼합 음원",
    },
    "mls": {
        "name": "Multilingual LibriSpeech", "repo": "facebook/multilingual_librispeech",
        "revision": "2e83e61823b4c47dcbcb1980bb88601274127609",
        "source_url": "https://www.openslr.org/94/", "license": "CC BY 4.0",
        "description": "공개 오디오북의 여러 언어·화자 낭독 음성",
    },
    "yodas2": {
        "name": "ESPnet YODAS2", "repo": "espnet/yodas2",
        "revision": "c9674490249665d658f527e2684848377108d82c",
        "source_url": "https://huggingface.co/datasets/espnet/yodas2", "license": "CC BY 3.0",
        "description": "영상 단위 웹 음성; 대화·방송 등 세부 장르는 별도 확인",
    },
    "fma_large": {
        "name": "Free Music Archive large", "revision": "607364e7d8263f890d23c838802f29637600611f",
        "source_url": "https://github.com/mdeff/fma",
        "license": "Per-track Creative Commons; metadata/code license is not the audio license",
        "description": "다양한 장르의 30초 음악; 곡별 이용조건 확인 후 선별",
    },
    "fma_full": {
        "name": "Free Music Archive full", "revision": "607364e7d8263f890d23c838802f29637600611f",
        "source_url": "https://github.com/mdeff/fma",
        "license": "Per-track Creative Commons; metadata/code license is not the audio license",
        "description": "음악·보컬·연주곡의 전체 길이 원음; 곡별 이용조건 확인 후 선별",
    },
}
_LANGUAGES = {"dutch", "french", "german", "italian", "polish", "portuguese", "spanish"}
_FMA_FILES = {
    "fma_large.zip": (100306112191, "497109f4dd721066b5ce5e5f250ec604dc78939e"),
    "fma_full.zip": (943642698636, "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab"),
    "fma_metadata.zip": (358412441, "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"),
}
_FMA_VARIANTS = {"fma_large": ("large", "fma_large.zip"), "fma_full": ("full", "fma_full.zip")}


def _hours(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Preparation hours must be positive finite numbers")
    return float(value)


def commercial_source_catalog():
    """Return reproducible public origins, without network access or a supply claim."""
    return {"schema_version": 1, "sources": [
        {"id": key, **value, "availability": "public_source_requires_preparation",
         "content_approved": False} for key, value in _PROVIDERS.items()],
        "limitations": [
            "Archive access does not establish permission for every original recording.",
            "YODAS2 has no verified genre, channel or speaker identity for each video.",
            "Original hours are measured after preparation and deduplication, not from target hours.",
            "Legacy 2,000h features cannot be added to raw hours without source-overlap evidence.",
        ]}


def commercial_negative_plan(generic_hours=7000, korean_hours=3000):
    """Scale each pool independently, leaving the selected training batch unchanged."""
    generic_hours, korean_hours = _hours(generic_hours), _hours(korean_hours)
    specs = (
        ("mixed", "generic_negative", "생활·환경·말소리 혼합 음원", 3000, "audioset_unbalanced"),
        ("multilingual_read", "generic_negative", "다국어 낭독 음성", 3000, "mls"),
        ("music", "generic_negative", "음악·보컬·연주곡", 1000, "fma_full"),
        ("korean_web_speech", "korean_negative", "한국어 음성", 3000, "yodas2"),
    )
    components = []
    for key, pool, name, baseline, provider in specs:
        source = _PROVIDERS[provider]
        component = {"id": key, "pool": pool, "name": name,
            "description": source["description"],
            "target_hours": baseline * (generic_hours if pool == "generic_negative" else korean_hours)
                            / (7000 if pool == "generic_negative" else 3000),
            "sources": [{"id": provider, "name": source["name"], "source_url": source["source_url"],
                "availability": "public_source_requires_preparation", "description": source["description"]}]}
        if provider == "fma_full":
            component["description"] = "필요한 원곡만 선택 다운로드 · 원본 MP3 보존"
            component["sources"][0].update(download_mode="selective", description=component["description"])
        components.append(component)
    return {"id": "commercial_negative_v3", "version": 3,
        "generic_hours": generic_hours, "korean_hours": korean_hours,
        "total_hours": generic_hours + korean_hours, "components": components,
        "scope": "Preparation targets, not downloaded, verified, listened or admitted hours.",
        "batch_changed": False, "content_approved": False}


def _safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("A relative publisher path is required")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")) or ":" in value:
        raise ValueError("Publisher path escapes its source")
    return value


def _resolve_url(provider, path):
    path = _safe_relative(path)
    return f"https://huggingface.co/datasets/{provider['repo']}/resolve/{provider['revision']}/{quote(path, safe='/')}"


def _tree(provider, path, session):
    """Follow all pinned HF pages, restricting continuation links to the same tree."""
    prefix = f"https://huggingface.co/api/datasets/{provider['repo']}/tree/{provider['revision']}"
    url = prefix + ("/" + quote(_safe_relative(path), safe="/") if path else "") + "?limit=1000"
    rows, pages, seen_paths = [], set(), set()
    while url:
        parsed = urlparse(url)
        tree_path = urlparse(prefix).path
        if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
                or not (parsed.path == tree_path or parsed.path.startswith(tree_path + "/"))):
            raise ValueError("Publisher pagination escapes the pinned tree")
        if url in pages or len(pages) >= 256:
            raise ValueError("Publisher pagination loop or inventory limit exceeded")
        pages.add(url)
        response = session.get(url, timeout=(30, 120))
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list) or len(page) > 1000:
            raise ValueError("Invalid bounded publisher tree response")
        for row in page:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in seen_paths:
                raise ValueError("Publisher inventory repeats or omits a path")
            seen_paths.add(row["path"])
        rows.extend(page)
        url = response.links.get("next", {}).get("url")
    return rows


def _asset(row, provider):
    if row.get("type") != "file":
        raise ValueError("A publisher file is required")
    path, size = _safe_relative(row.get("path")), row.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 1024**4:
        raise ValueError("Publisher file needs a bounded positive byte size")
    lfs = row.get("lfs") or {}
    checksum = lfs.get("oid")
    if re.fullmatch("[0-9a-f]{64}", checksum or ""):
        if lfs.get("size") != size:
            raise ValueError("Publisher LFS size disagrees with the tree")
        algorithm = "sha256"
    else:
        checksum, algorithm = row.get("oid"), "git_sha1"
        if not re.fullmatch("[0-9a-f]{40}", checksum or ""):
            raise ValueError("Publisher file lacks a fixed checksum")
    return {"id": path, "path": path, "filename": PurePosixPath(path).name,
        "source_url": _resolve_url(provider, path), "size_bytes": size,
        "checksum_algorithm": algorithm, algorithm: checksum,
        "upstream_revision": provider["revision"], "license": provider["license"]}


def _pick(rows, paths, provider):
    by_path = {row.get("path"): row for row in rows if row.get("type") == "file"}
    if len(set(paths)) != len(paths) or any(path not in by_path for path in paths):
        raise ValueError("Selected publisher file is missing or repeated")
    return [_asset(by_path[path], provider) for path in paths]


def resolve_commercial_source(source_id, *, selection, session=None):
    """Resolve explicit archive parts plus their metadata; no audio is fetched.

    AudioSet: {"parts": [0]}; MLS: {"language": "german",
    "parts": ["10087_10388_000"]}; YODAS2: {"subset": "ko000",
    "parts": [0]}; FMA large/full: {"parts": ["large"]}/["full"], or ["metadata"].
    """
    if source_id not in _PROVIDERS or not isinstance(selection, dict):
        raise ValueError("Choose a known commercial source and explicit selection")
    provider = _PROVIDERS[source_id]
    parts = selection.get("parts")
    if not isinstance(parts, list) or not parts or len(parts) > 1000 or len(set(map(str, parts))) != len(parts):
        raise ValueError("Select 1–1,000 distinct source archive parts")
    session = session or requests
    metadata, assets = [], []
    if source_id == "audioset_unbalanced":
        if any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= 869 for p in parts):
            raise ValueError("AudioSet part indices must be 0–869")
        rows = _tree(provider, "data", session)
        assets = _pick(rows, [f"data/unbal_train{p:03d}.tar" for p in parts], provider)
        metadata = _pick(rows, ["data/unbalanced_train_segments.csv", "data/ontology.json"], provider)
    elif source_id == "mls":
        language = selection.get("language")
        if language not in _LANGUAGES or any(not isinstance(p, str) or not re.fullmatch(r"\d+_\d+_\d+", p) for p in parts):
            raise ValueError("Select a supported MLS language and actual speaker/book part IDs")
        base = f"data/mls_{language}/train"
        assets = _pick(_tree(provider, base + "/audio", session),
                       [f"{base}/audio/{p}.tar.gz" for p in parts], provider)
        metadata = _pick(_tree(provider, base, session),
                         [base + "/segments.txt", base + "/transcripts.txt", base + "/audio_filenames.txt"], provider)
    elif source_id == "yodas2":
        subset = selection.get("subset", "")
        if not isinstance(subset, str) or not re.fullmatch(r"[a-z]{2,3}\d{3}", subset):
            raise ValueError("A publisher YODAS2 language subset is required")
        if any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < 100000000 for p in parts):
            raise ValueError("YODAS2 archive indices must be nonnegative integers")
        base = "data/" + subset
        assets = _pick(_tree(provider, base + "/audio", session),
                       [f"{base}/audio/{p:08d}.tar.gz" for p in parts], provider)
        for directory, extension in (("duration", "txt"), ("text", "json")):
            metadata.extend(_pick(_tree(provider, base + "/" + directory, session),
                                 [f"{base}/{directory}/{p:08d}.{extension}" for p in parts], provider))
    else:
        variant, archive = _FMA_VARIANTS[source_id]
        if any(p not in (variant, "metadata") for p in parts):
            raise ValueError(f"FMA {variant} parts must be {variant} and/or metadata")
        for name in ([archive] if variant in parts else []):
            size, checksum = _FMA_FILES[name]
            assets.append({"id": name, "path": name, "filename": name,
                "source_url": "https://os.unil.cloud.switch.ch/fma/" + name,
                "size_bytes": size, "checksum_algorithm": "sha1", "sha1": checksum,
                "upstream_revision": provider["revision"], "license": provider["license"]})
        name = "fma_metadata.zip"
        size, checksum = _FMA_FILES[name]
        metadata.append({"id": name, "path": name, "filename": name,
            "source_url": "https://os.unil.cloud.switch.ch/fma/" + name,
            "size_bytes": size, "checksum_algorithm": "sha1", "sha1": checksum,
            "upstream_revision": provider["revision"], "license": provider["license"]})
    if source_id not in _FMA_VARIANTS:
        metadata += _pick(_tree(provider, "", session), ["README.md"], provider)
    result = {"schema_version": 1, "kind": "commercial_source_resolution_v1", "source_id": source_id,
        "provider": dict(provider), "selection": selection, "assets": assets, "metadata_assets": metadata,
        "total_bytes": sum(a["size_bytes"] for a in assets + metadata),
        "content_approved": False, "prepared_hours": 0, "admitted_hours": 0,
        "resolved_at": now(), "Final_accessed": False}
    result["resolution_sha256"] = identity({k: v for k, v in result.items() if k != "resolved_at"})
    return result


def _validate_resolution(record):
    if not isinstance(record, dict) or record.get("kind") != "commercial_source_resolution_v1":
        raise ValueError("A resolved commercial source is required")
    expected = identity({k: v for k, v in record.items() if k not in ("resolved_at", "resolution_sha256")})
    if record.get("resolution_sha256") != expected:
        raise ValueError("Resolved publisher metadata changed")
    provider = _PROVIDERS.get(record.get("source_id"))
    if provider is None or record.get("provider") != provider:
        raise ValueError("Source provider or revision is not pinned")
    all_assets = record.get("assets", []) + record.get("metadata_assets", [])
    if not all_assets or len({a.get("path") for a in all_assets}) != len(all_assets):
        raise ValueError("Resolved files must be nonempty and unique")
    if record["source_id"] in _FMA_VARIANTS:
        variant, archive = _FMA_VARIANTS[record["source_id"]]
        parts = record.get("selection", {}).get("parts")
        if (not isinstance(parts, list) or not parts or len(set(map(str, parts))) != len(parts)
                or any(p not in (variant, "metadata") for p in parts)):
            raise ValueError("FMA selection does not match its pinned variant")
        expected_audio = [archive] if variant in parts else []
        if ([a.get("path") for a in record.get("assets", [])] != expected_audio
                or [a.get("path") for a in record.get("metadata_assets", [])] != ["fma_metadata.zip"]):
            raise ValueError("FMA source file coverage does not match the selected variant")
    for asset in all_assets:
        path = _safe_relative(asset.get("path"))
        expected_url = ("https://os.unil.cloud.switch.ch/fma/" + path if record["source_id"] in _FMA_VARIANTS
                        else _resolve_url(provider, path))
        if asset.get("source_url") != expected_url or asset.get("upstream_revision") != provider["revision"]:
            raise ValueError("Asset URL or revision escapes its pinned provider")
        if record["source_id"] in _FMA_VARIANTS:
            if (path not in _FMA_FILES or (asset.get("size_bytes"), asset.get("sha1")) != _FMA_FILES[path]
                    or asset.get("checksum_algorithm") != "sha1"):
                raise ValueError("FMA asset differs from original publisher pins")
        size = asset.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 1024**4:
            raise ValueError("Asset size must be bounded")
        algorithm = asset.get("checksum_algorithm")
        checksum = asset.get(algorithm) if isinstance(algorithm, str) else None
        if algorithm not in ("sha256", "sha1", "git_sha1") or not isinstance(checksum, str):
            raise ValueError("Asset needs a supported original checksum")
        if not re.fullmatch("[0-9a-f]{64}" if algorithm == "sha256" else "[0-9a-f]{40}", checksum):
            raise ValueError("Asset checksum is malformed")
    if record.get("total_bytes") != sum(a["size_bytes"] for a in all_assets):
        raise ValueError("Source byte total disagrees with resolved assets")
    return all_assets


def download_commercial_source(record, output, *, max_download_bytes, progress=None, metadata_only=False):
    """Fetch resolved parts, or only their metadata, within an explicit byte cap."""
    assets = _validate_resolution(record)
    if not isinstance(metadata_only, bool):
        raise ValueError("metadata_only must be a boolean")
    if metadata_only:
        assets = record["metadata_assets"]
    selected_bytes = sum(asset["size_bytes"] for asset in assets)
    if isinstance(max_download_bytes, bool) or not isinstance(max_download_bytes, int) or max_download_bytes <= 0:
        raise ValueError("An explicit positive integer max_download_bytes is required")
    if selected_bytes > max_download_bytes:
        raise ValueError("Selected source parts exceed max_download_bytes; no download started")
    output = Path(output).resolve()
    if any(part.lower() == ".codex" for part in output.parts):
        raise ValueError("Codex state is not a training-data download directory")
    output.mkdir(parents=True, exist_ok=True)
    required = 0
    for asset in assets:
        target = output / asset["path"]
        if not target.resolve().is_relative_to(output):
            raise ValueError("Download path escapes selected storage")
        partial = target.with_name(target.name + ".part")
        already = target.stat().st_size if target.is_file() else partial.stat().st_size if partial.is_file() else 0
        if already > asset["size_bytes"]:
            raise ValueError("Existing source file exceeds pinned size")
        required += asset["size_bytes"] - already
    if shutil.disk_usage(output).free < required:
        raise ValueError("Insufficient storage for selected original parts; choose fewer parts or another data drive")
    resolution_path = output / "source_resolution.json"
    if resolution_path.exists():
        saved_resolution = read_json(resolution_path)
        _validate_resolution(saved_resolution)
        if saved_resolution.get("resolution_sha256") != record["resolution_sha256"]:
            raise ValueError("Download directory belongs to a different source selection")
    else:
        write_json(resolution_path, record)
    receipt_path = output / ("metadata_download_receipt.json" if metadata_only else "download_receipt.json")
    previous = read_json(receipt_path) if receipt_path.exists() else None
    if previous is not None:
        expected_state = "metadata_downloaded" if metadata_only else "downloaded"
        expected_binding = {"path": str(resolution_path), "sha256": digest(resolution_path)}
        if (not isinstance(previous, dict) or previous.get("kind") != "commercial_source_download_v1"
                or previous.get("state") != expected_state or previous.get("metadata_only", False) != metadata_only
                or previous.get("source_id") != record["source_id"]
                or previous.get("resolution_sha256") != record["resolution_sha256"]
                or previous.get("source_resolution") != expected_binding
                or previous.get("bytes") != selected_bytes or previous.get("content_approved") is not False
                or previous.get("prepared_hours") != 0 or previous.get("admitted_hours") != 0
                or previous.get("Final_accessed") is not False
                or previous.get("downloaded_audio_files") != (0 if metadata_only else len(record["assets"]))):
            raise ValueError("Existing completed source receipt changed or belongs to different inputs")
        old_files = previous.get("files")
        if (not isinstance(old_files, list) or len(old_files) != len(assets)
                or any(not isinstance(row, dict) for row in old_files)
                or {row.get("source_path") for row in old_files} != {a["path"] for a in assets}):
            raise ValueError("Existing completed source receipt file coverage changed")
        previous_files = {row["source_path"]: row for row in old_files}
        for asset in assets:
            row = previous_files[asset["path"]]
            expected_path = (output / asset["path"]).resolve()
            if (Path(row.get("path", "")).resolve() != expected_path or row.get("bytes") != asset["size_bytes"]
                    or row.get("upstream_revision") != asset["upstream_revision"]
                    or row.get("source_url") != asset["source_url"]
                    or row.get("license") != asset["license"] or row.get("content_approved") is not False
                    or not isinstance(row.get("sha256"), str)
                    or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
                raise ValueError("Existing completed source receipt has changed file bindings")
            if not expected_path.is_file() or expected_path.stat().st_size != asset["size_bytes"]:
                raise ValueError("Existing completed original file is missing or changed")
            actual_sha256 = _verify_cached_original(expected_path, asset)
            if actual_sha256 != row["sha256"]:
                raise ValueError("Existing completed original file is missing or changed")
        write_json(output / ("metadata_download_progress.json" if metadata_only else "download_progress.json"),
            {"state": "verified_reuse", "completed_files": len(assets), "total_files": len(assets),
             "content_approved": False})
        # Source indexes can bind this file's SHA. Only the observation changes;
        # successful cache verification preserves the completed receipt bytes.
        if progress:
            progress({"label": "기존 원본 자료 해시 확인 완료", "done": len(assets), "total": len(assets), "unit": "files"})
        return previous
    receipts = []
    for index, asset in enumerate(assets):
        target = output / asset["path"]
        if progress:
            progress({"label": "원본 자료 다운로드: " + asset["path"], "done": index, "total": len(assets), "unit": "files"})
        if asset["checksum_algorithm"] == "sha256":
            receipt = download_checked(asset["source_url"], target, asset["sha256"], asset["size_bytes"])
        else:
            # Reuse the existing resumable original SHA1/Git-blob downloader.
            from .studio_data import _download
            downloaded_path = _download(asset, target.parent, progress=progress)
            if downloaded_path.parent != target.parent:
                # The legacy downloader can reuse an ancestor's verified cache.
                receipt_path = target.parent / (downloaded_path.name + ".reused.receipt.json")
                receipt = read_json(receipt_path)
                shutil.copyfile(downloaded_path, target)
                if digest(target) != receipt["sha256"]:
                    raise ValueError("Copied source cache differs from verified original")
                receipt = {**receipt, "path": str(target), "copied_from": str(downloaded_path)}
            else:
                receipt = read_json(downloaded_path.with_name(downloaded_path.name + ".receipt.json"))
        receipts.append({**receipt, "source_path": asset["path"], "upstream_revision": asset["upstream_revision"],
                         "source_url": asset["source_url"], "license": asset["license"], "content_approved": False})
        write_json(output / ("metadata_download_progress.json" if metadata_only else "download_progress.json"), {"state": "downloading", "files": receipts,
            "completed_files": len(receipts), "total_files": len(assets), "content_approved": False})
    receipt = {"schema_version": 1, "kind": "commercial_source_download_v1",
        "state": "metadata_downloaded" if metadata_only else "downloaded", "metadata_only": metadata_only,
        "source_id": record["source_id"], "resolution_sha256": record["resolution_sha256"],
        "source_resolution": {"path": str(resolution_path), "sha256": digest(resolution_path)},
        "files": receipts, "bytes": sum(a["bytes"] for a in receipts), "completed_at": now(),
        "downloaded_audio_files": 0 if metadata_only else len(record["assets"]),
        "prepared_hours": 0, "admitted_hours": 0, "content_approved": False, "Final_accessed": False}
    write_json(receipt_path, receipt)
    if progress:
        progress({"label": "선택한 원본 자료 다운로드 완료", "done": len(assets), "total": len(assets), "unit": "files"})
    return receipt


def _verify_cached_original(path, asset):
    """Check receipt SHA256 and the publisher checksum with one sequential read."""
    algorithm = asset["checksum_algorithm"]
    sha256 = hashlib.sha256()
    original = hashlib.sha1() if algorithm != "sha256" else None
    if algorithm == "git_sha1":
        original.update(f"blob {asset['size_bytes']}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            if original is not None:
                original.update(chunk)
    checksum = sha256.hexdigest() if original is None else original.hexdigest()
    if checksum != asset[algorithm]:
        raise ValueError("Cached original differs from the publisher checksum")
    return sha256.hexdigest()
