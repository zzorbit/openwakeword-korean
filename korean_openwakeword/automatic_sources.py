"""Resumable raw Negative supply for the web workflow.

Publisher hours are never admission. Each frozen part follows the existing
checksum download, safe extraction and provider-index adapters. An optional
qualification callback supplies real post-QC Train hours and requests further
parts naturally when technical rejects leave a shortfall.
"""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import re
import shutil

import requests
import soundfile as sf

from .common import digest, identity, now, read_json, read_rows, write_json, writer_lock
from .budgets import DEFAULT_GENERIC_HOURS, DEFAULT_KOREAN_HOURS, generic_source_hours
from .project import load_project


POLICY_VERSION = 1
SEED = 20261021
_GIB = 1024**3
_RESERVE = 8 * _GIB
_PROVIDERS = ("audioset_unbalanced", "mls", "fma_full", "yodas2")
_LANGUAGES = ("dutch", "french", "german", "italian", "polish", "portuguese", "spanish")
_KOREAN = ("ko000", "ko100", "ko101", "ko102", "ko103")


def _hours(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite nonnegative hours")
    return float(value)


def _preset(value):
    value = {"trial": "test", "production": "full", "commercial": "full"}.get(value, value)
    if value not in ("check", "test", "full"):
        raise ValueError("Choose check, test or full automatic materials")
    return value


def automatic_source_policy(generic_hours=DEFAULT_GENERIC_HOURS, korean_hours=DEFAULT_KOREAN_HOURS, preset="full", *,
                            material_acquisition_policy="automatic_v1", fma_acquisition_policy=None):
    """Offline description; no source files or provider metadata are opened."""
    if material_acquisition_policy not in ("automatic_v1", "automatic_v2", "automatic_v3"):
        raise ValueError("Unsupported automatic material acquisition policy")
    if fma_acquisition_policy not in (None, "per_track_flac_prefetch_v1"):
        raise ValueError("Unsupported FMA acquisition policy")
    if fma_acquisition_policy and material_acquisition_policy != "automatic_v3":
        raise ValueError("Per-track FMA requires canonical FLAC storage")
    generic, korean = _hours(generic_hours, "Generic"), _hours(korean_hours, "Korean")
    preset = _preset(preset)
    targets = generic_source_hours(generic) if preset == "full" else {}
    if preset != "check" and korean:
        targets["yodas2"] = korean
    result = {"version": POLICY_VERSION, "preset": preset, "component_targets": targets,
            "generic_mode": "raw_waveforms" if preset == "full" else "publisher_2000h_features",
            "korean_mode": "existing_zeroth_1h" if preset == "check" else "yodas2_raw_waveforms",
            "post_qc_refill": True, "validation_fraction": 0,
            "content_approved": False, "training_admitted": False, "Final_accessed": False}
    if material_acquisition_policy == "automatic_v2":
        result.update(version=2, storage_policy="verified_extracted_only_v1")
    elif material_acquisition_policy == "automatic_v3":
        result.update(version=3, storage_policy="canonical_flac_only_v1")
    if fma_acquisition_policy:
        result["fma_acquisition_policy"] = fma_acquisition_policy
    return result


def _space_plan(policy, config):
    """Known downstream payload plus explicitly estimated original storage."""
    seconds = sum(policy["component_targets"].values()) * 3600
    normalized = math.ceil(seconds * 16000 * 2)
    negative_features = math.ceil(seconds / 2) * 16 * 96 * 4
    originals = config.get("targets", {}).get("positive_originals", 0)
    rounds = config.get("augmentation_rounds", 1)
    positive_features = math.ceil(originals * rounds * 16 * 96 * 4)
    # Raw codecs and compression vary. These are forecasts, not publisher
    # verified sizes or promises; resolved part sizes remain the hard gate.
    canonical = policy.get("storage_policy") == "canonical_flac_only_v1"
    compact = policy.get("storage_policy") in ("verified_extracted_only_v1", "canonical_flac_only_v1")
    per_hour = ({"audioset_unbalanced": 500_000_000, "mls": 140_000_000,
                 "fma_full": 125_000_000, "yodas2": 180_000_000} if compact else
                {"audioset_unbalanced": 1_000_000_000, "mls": 280_000_000,
                 "fma_full": 125_000_000, "yodas2": 300_000_000})
    source_forecast = math.ceil(sum(hours * per_hour[key]
        for key, hours in policy["component_targets"].items()))
    known = normalized + negative_features + positive_features
    result = {"normalized_pcm16_bytes": normalized,
        "negative_feature_payload_bytes": negative_features,
        "positive_feature_payload_bytes": positive_features,
        "minimum_known_bytes": known, "future_buffer_bytes": _RESERVE,
        "estimated_original_archive_bytes": source_forecast,
        "estimated_total_bytes": known + source_forecast + _RESERVE,
        "estimate_scope": "Original/archive sizes are planning forecasts; TTS WAV, evaluation, optimizer/checkpoint and codec variation need additional room"}
    if compact:
        temporary = _RESERVE if policy["component_targets"] else 0
        download_per_hour = {"audioset_unbalanced": 500_000_000, "mls": 140_000_000,
                             "fma_full": 125_000_000, "yodas2": 140_000_000}
        result.update(storage_policy=policy["storage_policy"],
            estimated_retained_source_bytes=source_forecast,
            source_forecast_bytes_per_hour=per_hour,
            download_forecast_bytes_per_hour=download_per_hour,
            estimated_download_bytes=math.ceil(sum(hours * download_per_hour[key]
                for key, hours in policy["component_targets"].items())),
            transient_archive_bytes=temporary,
            estimated_total_bytes=known + source_forecast + _RESERVE + temporary,
            estimate_scope="Retained original-audio and transfer bytes are separate forecasts. One archive at a time is temporary; actual selected archive/extraction size is checked before use. Existing archives and old receipts are not retired. TTS/evaluation WAVs and checkpoints need additional room.")
    if canonical:
        # Only features are an exact hard lower bound. FLAC compression depends
        # on the source; the PCM figure is a reference budget, never a measured
        # FLAC estimate. Old requests retain their batch acquisition contract.
        fma_peak = math.ceil(policy["component_targets"].get("fma_full", 0) * 125_000_000)
        streaming = policy.get("fma_acquisition_policy") == "per_track_flac_prefetch_v1"
        if streaming:
            # A planning allowance, not a hard track-size upper bound. Exact
            # selected-track and conversion workspace checks run before I/O.
            fma_peak = 0
        transient = max(fma_peak, _RESERVE if policy["component_targets"] else 0)
        known_features = negative_features + positive_features
        result.update(storage_policy=policy["storage_policy"],
            minimum_known_bytes=known_features,
            estimated_retained_source_bytes=0,
            estimated_original_archive_bytes=0,
            eliminated_retained_original_forecast_bytes=source_forecast,
            estimated_canonical_audio_bytes=normalized,
            canonical_flac_estimated_bytes=None,
            transient_archive_bytes=0,
            transient_source_working_bytes=_RESERVE if policy["component_targets"] else 0,
            transient_source_bytes=transient,
            estimated_total_bytes=known_features + normalized + _RESERVE + transient,
            estimate_scope="PCM16 audio is an uncompressed comparison budget, not measured FLAC size or a minimum. Only feature payload is a known lower bound. Verified canonical FLAC replaces newly acquired raw audio; transfer bytes are unchanged. The complete unchanged FMA playlist remains a temporary raw batch. Existing sources, TTS/evaluation audio and checkpoints are separate.")
        if streaming:
            result.update(fma_acquisition_policy=policy["fma_acquisition_policy"],
                estimate_scope="PCM16 audio is a reference budget, not measured FLAC size or a minimum. FMA converts each verified track while at most one next track is prefetched; the working allowance is checked against actual selected track sizes. Existing sources, TTS/evaluation audio and checkpoints are separate.")
    return result


def automatic_storage_plan(config=None, *, generic_hours=DEFAULT_GENERIC_HOURS, korean_hours=DEFAULT_KOREAN_HOURS, preset="full",
                           material_acquisition_policy="automatic_v1", fma_acquisition_policy=None):
    """Offline known payload lower bound and separately labelled forecasts."""
    if fma_acquisition_policy is None:
        fma_acquisition_policy = (config or {}).get("fma_acquisition_policy")
    return _space_plan(automatic_source_policy(generic_hours, korean_hours, preset,
        material_acquisition_policy=material_acquisition_policy,
        fma_acquisition_policy=fma_acquisition_policy), config or {})


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def _frozen(path, record):
    # Round-trip tuples and other JSON-compatible containers before comparison.
    record = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Automatic source request changed; preserve it and use a new project")
    else:
        write_json(path, record)
    return record


def _rows(path, rows):
    data = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise ValueError("Frozen automatic source index changed")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(data, encoding="utf-8")
        os.replace(temporary, path)
    return _binding(path)


def _interleave(groups):
    groups = deque(deque(group) for group in groups if group)
    result = []
    while groups:
        group = groups.popleft()
        result.append(group.popleft())
        if group:
            groups.append(group)
    return result


class _TreeSession:
    """Cache immutable provider JSON pages, never archive bodies."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.total = 0

    def get(self, url, **kwargs):
        path = self.directory / (identity(url) + ".json")
        if path.exists():
            saved = read_json(path)
            if saved.get("url") != url or saved.get("body_sha256") != identity(saved.get("body")):
                raise ValueError("Pinned provider metadata cache changed")
        else:
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                pieces, count = [], 0
                for block in response.iter_content(65536):
                    count += len(block)
                    self.total += len(block)
                    if count > 8 * 1024**2 or self.total > 256 * 1024**2:
                        raise ValueError("Automatic provider metadata exceeds its bounded transfer limit")
                    pieces.append(block)
                body = json.loads(b"".join(pieces))
                saved = {"url": url, "body": body, "body_sha256": identity(body),
                         "links": response.links}
                write_json(path, saved)
        class Response:
            links = saved["links"]
            def raise_for_status(self): pass
            def json(self): return saved["body"]
        return Response()


class PublicSourceBackend:
    """Existing public APIs composed into bounded, explicit source parts.

    ``inventory(component)`` returns frozen candidate selections.
    ``prepare(candidate, directory, *, target_hours, max_download_bytes,
    max_extract_bytes, excluded_track_ids, progress)`` returns ``source`` plus
    its provider index rows and immutable receipt bindings. A test backend may
    implement these two methods without network access.
    """
    def __init__(self, directory, *, storage_policy="retain_archives", fma_acquisition_policy=None):
        if storage_policy not in ("retain_archives", "verified_extracted_only_v1", "canonical_flac_only_v1"):
            raise ValueError("Unsupported source archive storage policy")
        self.directory = Path(directory)
        self.storage_policy = storage_policy
        if fma_acquisition_policy not in (None, "per_track_flac_prefetch_v1"):
            raise ValueError("Unsupported FMA acquisition policy")
        if fma_acquisition_policy and storage_policy != "canonical_flac_only_v1":
            raise ValueError("Per-track FMA requires canonical FLAC storage")
        self.fma_acquisition_policy = fma_acquisition_policy
        self.session = _TreeSession(self.directory / "provider_pages")

    def inventory(self, component):
        from .commercial_sources import _PROVIDERS as providers, _tree
        provider = providers[component]
        def candidates(rows, selection):
            result = []
            for row, choice in selection(rows):
                result.append({"component": component, "selection": choice,
                    "key": identity([component, choice])[:24], "archive_bytes": row["size"]})
            return sorted(result, key=lambda row: identity([SEED, row["key"]]))
        if component == "audioset_unbalanced":
            rows = _tree(provider, "data", self.session)
            return candidates(rows, lambda rows: [(row, {"parts": [int(match[1])]})
                for row in rows if row.get("type") == "file"
                and (match := re.fullmatch(r"data/unbal_train(\d{3})\.tar", row["path"]))])
        if component == "yodas2":
            groups = []
            for subset in _KOREAN:
                rows = _tree(provider, f"data/{subset}/audio", self.session)
                groups.append(candidates(rows, lambda rows, subset=subset: [
                    (row, {"subset": subset, "parts": [int(Path(row["path"]).name.split(".")[0])]})
                    for row in rows if row.get("type") == "file" and row["path"].endswith(".tar.gz")]))
            return _interleave(groups)
        if component == "mls":
            groups = []
            for language in _LANGUAGES:
                rows = [row for row in _tree(provider, f"data/mls_{language}/train/audio", self.session)
                        if row.get("type") == "file" and re.fullmatch(r"\d+_\d+_\d+\.tar\.gz", Path(row["path"]).name)]
                rows.sort(key=lambda row: identity([SEED, language, row["path"]]))
                batches = []
                # Shared whole-language transcripts are amortized across books.
                for start in range(0, len(rows), 16):
                    batch = rows[start:start+16]
                    choice = {"language": language,
                              "parts": [Path(row["path"]).name.removesuffix(".tar.gz") for row in batch]}
                    batches.append({"component": component, "selection": choice,
                        "key": identity([component, choice])[:24], "archive_bytes": sum(row["size"] for row in batch)})
                groups.append(batches)
            return _interleave(groups)
        if component == "fma_full":
            # Each supplement excludes all previously selected track IDs.
            return [{"component": component, "selection": {"attempt": attempt},
                     "key": f"fma_supplement_{attempt:04d}", "archive_bytes": None}
                    for attempt in range(1000)]
        raise ValueError("Unknown automatic Negative component")

    def _share_metadata(self, resolution, destination):
        """Hard-link already verified metadata on the same selected drive."""
        shared = self.directory / "shared_metadata"
        shared.mkdir(exist_ok=True)
        for asset in resolution["metadata_assets"]:
            algorithm = asset["checksum_algorithm"]
            cached = shared / (asset[algorithm] + "_" + asset["filename"])
            target = destination / asset["path"]
            if target.exists() or not cached.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(cached, target)
            except OSError:
                shutil.copyfile(cached, target)

    def prepare(self, candidate, directory, *, target_hours, max_download_bytes,
                max_extract_bytes, excluded_track_ids=(), progress=None):
        if candidate["component"] == "fma_full" and self.fma_acquisition_policy:
            from .fma_streaming import prepare_fma_streaming_source
            destination = Path(directory) / self.fma_acquisition_policy / identity([
                target_hours, sorted(set(excluded_track_ids)), SEED, "allow-derivatives"])
            return prepare_fma_streaming_source(destination, target_hours=target_hours,
                seed=SEED, max_download_bytes=max_download_bytes, max_extract_bytes=max_extract_bytes,
                excluded_track_ids=excluded_track_ids, progress=progress)
        if self.storage_policy == "canonical_flac_only_v1":
            from .canonical_sources import prepare_canonical_source
            from .canonical_audio import canonicalize_prepared_source
            request = {"candidate": candidate, "target_hours": target_hours,
                       "excluded_track_ids": sorted(set(excluded_track_ids)), "seed": SEED,
                       "storage_policy": self.storage_policy}
            return prepare_canonical_source(directory, request=request,
                acquire=lambda: self._prepare(candidate, directory, target_hours=target_hours,
                    max_download_bytes=max_download_bytes, max_extract_bytes=max_extract_bytes,
                    excluded_track_ids=excluded_track_ids, progress=progress),
                canonicalize=canonicalize_prepared_source,
                max_retained_bytes=max_extract_bytes, progress=progress)
        return self._prepare(candidate, directory, target_hours=target_hours,
            max_download_bytes=max_download_bytes, max_extract_bytes=max_extract_bytes,
            excluded_track_ids=excluded_track_ids, progress=progress)

    def _prepare(self, candidate, directory, *, target_hours, max_download_bytes,
                 max_extract_bytes, excluded_track_ids=(), progress=None):
        from .commercial_sources import resolve_commercial_source, download_commercial_source
        from .studio_data import extract_archive
        from .preparation import create_source_index
        component = candidate["component"]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if component == "fma_full":
            from .fma_download import download_fma
            # Frozen selections with different hour targets/exclusions must
            # never collide in the shared cross-project cache.
            directory = directory / identity([target_hours, sorted(set(excluded_track_ids)), SEED,
                                               "allow-derivatives"])
            directory.mkdir(parents=True, exist_ok=True)
            exclusions = directory / "excluded_tracks.json"
            _frozen(exclusions, {"track_ids": sorted(set(excluded_track_ids))})
            preview = download_fma(directory / "originals", hours=target_hours, seed=SEED,
                max_download_bytes=max_download_bytes, exclusions=exclusions, plan_only=True, progress=progress)
            playlist = read_json(preview["plan"]["path"])["selection"]
            worst_originals = sum(row["original_bytes"] for row in playlist["selected"] + playlist.get("reserve", []))
            metadata_size = Path(preview["metadata_csv"]).stat().st_size
            if worst_originals + metadata_size > max_extract_bytes:
                raise ValueError("선택할 FMA 원곡·보충곡의 실제 원본 크기가 남은 저장 한도를 초과합니다. 원곡 다운로드는 시작하지 않았습니다.")
            preview_bytes = preview["metadata_transfer_bytes_this_run"]
            result = download_fma(directory / "originals", hours=target_hours, seed=SEED,
                max_download_bytes=max_download_bytes-preview_bytes, exclusions=exclusions, progress=progress)
            audio, metadata = Path(result["audio_directory"]), Path(result["metadata"]["path"])
            source = {"directory": str(audio), "adapter": "fma", "metadata": str(metadata),
                      "source_url": "https://github.com/mdeff/fma"}
            receipts = [result["receipt"]]
            reserved_bytes = result.get("downloaded_bytes_this_run", 0) + preview_bytes
            original_storage = result["original_mp3_bytes"] + metadata.stat().st_size
            if original_storage > max_extract_bytes:
                raise ValueError("선택한 FMA 원본이 자동 자료 저장 용량 한도를 초과했습니다. 기존 원본은 보존됩니다.")
        else:
            resolution = resolve_commercial_source(component, selection=candidate["selection"], session=self.session)
            original = directory / "originals"
            original.mkdir(exist_ok=True)
            self._share_metadata(resolution, original)
            if self.storage_policy in ("verified_extracted_only_v1", "canonical_flac_only_v1"):
                from .compact_sources import prepare_compact_source
                result = prepare_compact_source(resolution, directory,
                    max_download_bytes=max_download_bytes, max_extract_bytes=max_extract_bytes,
                    progress=progress)
                audio = Path(result["audio_directory"])
                receipts = result["receipts"]
                reserved_bytes = result["reserved_download_bytes"]
                original_storage = result["original_storage_bytes"]
            else:
                if resolution["total_bytes"] >= max_extract_bytes:
                    raise ValueError("다음 자료 묶음의 원본 다운로드 크기가 남은 자동 자료 저장 한도를 초과합니다.")
                # The legacy cache retains its exact archives and receipt layout.
                result = download_commercial_source(resolution, original,
                    max_download_bytes=max_download_bytes, progress=progress)
                audio = directory / "audio"
                extraction_receipts = []
                expanded = 0
                for index, asset in enumerate(resolution["assets"]):
                    extracted = extract_archive(original / asset["path"], audio / f"part_{index:04d}",
                        max_bytes=max_extract_bytes-resolution["total_bytes"]-expanded)
                    expanded += extracted["bytes"]
                    extraction_receipts.append(_binding(Path(extracted["output_directory"]) / "extraction_receipt.json")
                        if "output_directory" in extracted else _binding(audio / f"part_{index:04d}" / "extraction_receipt.json"))
                receipts = [_binding(original / "download_receipt.json"), *extraction_receipts]
                reserved_bytes = result["bytes"]
                original_storage = result["bytes"] + expanded
            shared = self.directory / "shared_metadata"
            for asset in resolution["metadata_assets"]:
                cached = shared / (asset[asset["checksum_algorithm"]] + "_" + asset["filename"])
                target = original / asset["path"]
                if not cached.exists():
                    try:
                        os.link(target, cached)
                    except OSError:
                        shutil.copyfile(target, cached)
            provider = resolution["provider"]
            source = {"directory": str(audio), "adapter": "audioset" if component == "audioset_unbalanced" else component,
                      "source_url": provider["source_url"], "license": provider["license"]}
            if component == "audioset_unbalanced":
                source.update(metadata=str(original / "data/unbalanced_train_segments.csv"),
                              ontology=str(original / "data/ontology.json"))
            else:
                source["metadata"] = str(original / "source_resolution.json")
        index = directory / "provider_index.jsonl"
        # Provider adapters preserve exact identities/transcripts, without STT.
        create_source_index(audio, output=index, dataset=source["adapter"],
            metadata=source.get("metadata"), ontology=source.get("ontology"),
            source_url=source.get("source_url"), license=source.get("license"))
        prepared = {"source": source, "rows": read_rows(index), "receipts": receipts,
                "reserved_download_bytes": reserved_bytes,
                "original_storage_bytes": original_storage,
                "artifacts": [str(index), *(receipt["path"] for receipt in receipts)]}
        if self.storage_policy in ("verified_extracted_only_v1", "canonical_flac_only_v1") and component != "fma_full":
            prepared.update(retired_archive_bytes=result["retired_archive_bytes"],
                            temporary_archive_peak_bytes=result["temporary_archive_peak_bytes"],
                            compact_directory=str(directory), compact_receipt=result["compact_receipt"])
        return prepared


def _validate_rows(source, rows, selected, seen_hashes, seen_sources, diagnostics=None,
                   storage_counter=None, max_copy_bytes=None):
    directory = Path(source["directory"]).resolve()
    selected.mkdir(parents=True, exist_ok=True)
    result, hashes, sources = [], set(seen_hashes), set(seen_sources)
    for supplied in rows:
        row = dict(supplied)
        if row.get("excluded") is True:
            continue
        relative = Path(row["path"])
        original = (directory / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not original.is_relative_to(directory):
            raise ValueError("Automatic provider row escapes its original audio directory")
        canonical = row.get("canonical_source_id")
        if not isinstance(canonical, str) or not canonical or row.get("split") in ("validation", "final"):
            raise ValueError("Automatic Train rows need actual source identities and cannot consume held-out audio")
        if not original.is_file() or original.is_symlink():
            raise ValueError("Automatic original audio is missing or a link")
        checksum = digest(original)
        if row.get("sha256") and row["sha256"] != checksum:
            raise ValueError("Automatic original waveform changed after provider indexing")
        original_identity = row.get("original_source_sha256", checksum)
        if original_identity in hashes or canonical in sources:
            continue
        try:
            header = sf.info(original)
        except (RuntimeError, sf.LibsndfileError) as error:
            if diagnostics is not None:
                diagnostics.append({"path": relative.as_posix(), "reason": "invalid_audio_header",
                                    "detail": str(error)})
            continue
        if header.frames < 1 or header.samplerate < 1 or header.channels < 1:
            if diagnostics is not None:
                diagnostics.append({"path": relative.as_posix(), "reason": "empty_audio_header"})
            continue
        destination = selected / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != checksum:
                raise ValueError("Frozen selected automatic source changed")
            # A transfer may stop after a same-volume link fallback copied the
            # file but before source.json committed its storage accounting.
            # Recount that independent copy on retry; a hardlink consumes no
            # second audio payload and remains excluded from the copy cost.
            if storage_counter is not None and not os.path.samefile(original, destination):
                copy_bytes = original.stat().st_size
                if max_copy_bytes is not None and storage_counter["bytes"] + copy_bytes > max_copy_bytes:
                    raise ValueError("원본 연결에 필요한 복사본이 자동 자료 누적 저장 한도를 초과합니다.")
                storage_counter["bytes"] += copy_bytes
        else:
            try:
                os.link(original, destination)
            except OSError:
                if storage_counter is not None and max_copy_bytes is not None:
                    if storage_counter["bytes"] + original.stat().st_size > max_copy_bytes:
                        raise ValueError("원본 연결에 필요한 복사본이 자동 자료 누적 저장 한도를 초과합니다.")
                shutil.copyfile(original, destination)
                if digest(destination) != checksum:
                    raise ValueError("Copied automatic source hash mismatch")
                if storage_counter is not None:
                    storage_counter["bytes"] += original.stat().st_size
        original_info = row.get("canonical_audio", {}).get("original_info", {})
        duration = (original_info["input_frames"] / original_info["input_sample_rate"]
                    if original_info else header.frames / header.samplerate)
        row.update(path=relative.as_posix(), sha256=checksum,
                   duration_s=duration, split="train")
        result.append(row)
        hashes.add(original_identity)
        sources.add(canonical)
    return result


def prepare_automatic_sources(project, workspace, *, generic_hours, korean_hours,
                              preset, progress=None, backend=None, qualify_source=None,
                              material_acquisition_policy="automatic_v1"):
    """Supply Train originals until real qualified component quotas are reached.

    ``qualify_source(source)`` returns the existing ``prepare_public_audio``
    record. Its post-QC ``train_hours`` controls refill, and that same record is
    returned as ``source['prepared']`` so the caller must not repeat the QC.
    Without a callback only header hours are reported; nothing is approved.
    Source volume caps are frozen in project state, not an unlimited promise
    of network retries. Provider downloaders retain their per-file byte caps.
    """
    root, config = load_project(project)
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    if any(part.lower() in (".codex", ".git") for part in workspace.parts):
        raise ValueError("Automatic materials must use the selected data workspace")
    if (root / "final_lock.json").exists():
        raise ValueError("Final-locked projects cannot request new training materials")
    policy = automatic_source_policy(generic_hours, korean_hours, preset,
        material_acquisition_policy=material_acquisition_policy,
        fma_acquisition_policy=config.get("fma_acquisition_policy"))
    canonical = material_acquisition_policy == "automatic_v3"
    compact = material_acquisition_policy in ("automatic_v2", "automatic_v3")
    if not policy["component_targets"]:
        return {"sources": [], "plan": policy, "receipts": [], "artifacts": [],
                "component_actual_hours": {}, "shortages": [], "actual_download_bytes": 0}
    workspace.mkdir(parents=True, exist_ok=True)
    namespace = {"automatic_v1": "automatic_sources_v1", "automatic_v2": "automatic_sources_v2",
                 "automatic_v3": "automatic_sources_v3"}[material_acquisition_policy]
    state = root / ".studio/materials" / namespace
    state.mkdir(parents=True, exist_ok=True)
    default_backend = workspace / "shared" / namespace
    request_path = state / "request.json"
    existing = read_json(request_path) if request_path.exists() else None
    limits = config.get("automatic_data_limits", {})
    available = max(0, shutil.disk_usage(workspace).free - _RESERVE)
    requested = {"kind": "automatic_negative_supply_" + material_acquisition_policy.removeprefix("automatic_"), "policy": policy,
        "wakeword": config["wakeword"], "project_id": config["project_id"],
        "workspace": str(workspace), "seed": SEED}
    from .commercial_sources import _PROVIDERS as providers
    requested["providers"] = {key: providers[key]["revision"] for key in _PROVIDERS}
    space = _space_plan(policy, config)
    requested["space_plan"] = space
    if existing:
        requested["limits"] = existing["limits"]
    else:
        requested["limits"] = {"max_download_bytes": limits.get("max_download_bytes", 4 * 1024**4 if canonical else min(4 * 1024**4, available)),
                               "max_workspace_bytes": limits.get("max_workspace_bytes", available),
                               "reserve_free_bytes": _RESERVE}
    for key in ("max_download_bytes", "max_workspace_bytes"):
        if type(requested["limits"][key]) is not int or requested["limits"][key] <= 0:
            raise ValueError("Automatic material byte budgets need positive integers and free selected-drive space")
    with writer_lock(state):
        if not existing:
            free = shutil.disk_usage(workspace).free
            minimum = space["minimum_known_bytes"] + space["future_buffer_bytes"] + space.get("transient_archive_bytes", 0)
            write_json(state / "space_preflight.json", {**space, "available_bytes": free,
                "max_workspace_bytes": requested["limits"]["max_workspace_bytes"],
                "audio_download_started": False, "created_at": now()})
            if free < minimum or requested["limits"]["max_workspace_bytes"] < space["minimum_known_bytes"]:
                raise ValueError(f"자동 자료 준비를 시작할 공간이 부족합니다. {'feature·여유 공간' if canonical else '정규화 음원·feature'}만 최소 {minimum:,}바이트가 필요하지만 "
                    f"선택 드라이브 여유는 {free:,}바이트, 설정한 저장 한도는 {requested['limits']['max_workspace_bytes']:,}바이트입니다. "
                    f"원본·압축 자료 예상 {space['estimated_original_archive_bytes']:,}바이트는 별도입니다. 목표량을 줄이지 않고 중단했습니다.")
        _frozen(request_path, requested)
        backend = backend or PublicSourceBackend(default_backend,
            **({"storage_policy": policy["storage_policy"]} if compact else {}),
            **({"fma_acquisition_policy": policy["fma_acquisition_policy"]}
               if policy.get("fma_acquisition_policy") else {}))
        sources, receipts, artifacts, shortages = [], [], [str(request_path)], []
        actual = {component: 0.0 for component in policy["component_targets"]}
        seen_hashes, seen_sources, seen_tracks = set(), set(), set()
        reserved_bytes, source_storage_bytes = 0, 0
        retired_archive_bytes, temporary_archive_peak_bytes = 0, 0
        canonical_storage_bytes, retired_original_bytes, acquired_original_peak_bytes = 0, 0, 0
        for component, target in policy["component_targets"].items():
            inventory_path = state / (component + "_inventory.json")
            inventory = read_json(inventory_path) if inventory_path.exists() else backend.inventory(component)
            _frozen(inventory_path, inventory)
            artifacts.append(str(inventory_path))
            if (not isinstance(inventory, list) or len(inventory) > 100000
                    or len({row.get("key") for row in inventory}) != len(inventory)):
                raise ValueError("Automatic source inventory is ambiguous or oversized")
            for candidate in inventory:
                if actual[component] + 1e-9 >= target:
                    break
                key = candidate.get("key")
                if candidate.get("component") != component or not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key):
                    raise ValueError("Automatic source candidate is unsafe or belongs to another component")
                part = state / "parts" / key
                part.mkdir(parents=True, exist_ok=True)
                record_path = part / "source.json"
                remaining_budget = requested["limits"]["max_download_bytes"] - reserved_bytes
                if progress:
                    progress({"label": "학습용 비호출 자료 자동 준비: " + component,
                              "done": actual[component], "total": target, "unit": "hours", "part": key})
                if record_path.exists():
                    record = read_json(record_path)
                    if record.get("candidate") != candidate:
                        raise ValueError("Frozen automatic source selection changed")
                    canonical_binding = record.get("canonical_receipt") if canonical else None
                    if canonical_binding:
                        from .canonical_sources import verify_canonical_source
                        canonical_receipt = Path(canonical_binding["path"]).resolve()
                        if (not canonical_receipt.is_relative_to(default_backend.resolve())
                                or not canonical_receipt.is_file()
                                or digest(canonical_receipt) != canonical_binding["sha256"]):
                            raise ValueError("Frozen canonical source receipt changed or escaped its cache")
                        if read_json(canonical_receipt).get("kind") == "fma_streaming_source_v1":
                            from .fma_streaming import verify_fma_streaming_source
                            verify_fma_streaming_source(canonical_receipt.parent)
                        else:
                            verify_canonical_source(canonical_receipt.parent)
                    compact_binding = record.get("compact_receipt") if compact and not canonical else None
                    if compact_binding:
                        from .compact_sources import verify_compact_source
                        compact_receipt = Path(compact_binding["path"]).resolve()
                        if (not compact_receipt.is_relative_to(default_backend.resolve())
                                or not compact_receipt.is_file()
                                or digest(compact_receipt) != compact_binding["sha256"]):
                            raise ValueError("Frozen compact source receipt changed or escaped its cache")
                        verify_compact_source(compact_receipt.parent)
                    source, rows = record["source"], read_rows(record["source"]["source_index"])
                    if digest(source["source_index"]) != record["source_index_sha256"]:
                        raise ValueError("Frozen automatic source index changed")
                    # Verify selected originals before trusting persisted hours.
                    for row in rows:
                        if digest(Path(source["directory"]) / row["path"]) != row["sha256"]:
                            raise ValueError("Previously selected automatic waveform changed")
                    part_receipts, part_artifacts = record["receipts"], record["artifacts"]
                    part_bytes = record["reserved_download_bytes"]
                    part_storage = record["original_storage_bytes"]
                else:
                    archive_bytes = candidate.get("archive_bytes")
                    if (remaining_budget <= 0 or type(archive_bytes) is int and archive_bytes > remaining_budget):
                        shortages.append(f"{component}: automatic source byte budget is exhausted before the next part")
                        break
                    free = shutil.disk_usage(workspace).free
                    # v1 retains the whole archive selection. Compact sources
                    # materialize only one archive at a time; its downloader
                    # and safe extractor gate the actual resolved peak. Do not
                    # reserve three times an MLS multi-archive batch here.
                    minimum = (_GIB if compact else max(_GIB, (archive_bytes or 0) * 3)) + _RESERVE
                    if free < minimum:
                        shortages.append(f"{component}: next part needs at least {minimum} free bytes on the selected drive; {free} available")
                        break
                    source_room = (requested["limits"]["max_workspace_bytes"]
                                   - space["minimum_known_bytes"] - source_storage_bytes)
                    if source_room <= 0 or (not compact and type(archive_bytes) is int and archive_bytes * 2 > source_room):
                        shortages.append(f"{component}: 자동 자료 누적 저장 한도 부족. 원본·압축·추출 자료에 남은 {max(0, source_room):,}바이트로 다음 묶음을 준비할 수 없습니다.")
                        break
                    prepared = backend.prepare(candidate, default_backend / "parts" / key,
                        target_hours=target-actual[component], max_download_bytes=remaining_budget,
                        max_extract_bytes=min(source_room, max(1, free-_RESERVE)),
                        excluded_track_ids=sorted(seen_tracks), progress=progress)
                    source = dict(prepared["source"])
                    diagnostics = []
                    copy_storage = {"bytes": 0}
                    rows = _validate_rows(source, prepared["rows"], part / "selected", seen_hashes, seen_sources,
                        diagnostics, copy_storage,
                        max(0, source_room-prepared.get("original_storage_bytes", prepared["reserved_download_bytes"])))
                    source.update(id="automatic_" + key, directory=str(part / "selected"),
                                  target_hours=None, validation_fraction=0,
                                  pools=["korean_negative"] if component == "yodas2" else ["generic_negative"],
                                  component=component, component_id=component, provider=component)
                    source["source_index"] = str(part / "source_index.jsonl")
                    _rows(Path(source["source_index"]), rows)
                    part_bytes = prepared["reserved_download_bytes"]
                    part_storage = prepared.get("original_storage_bytes", part_bytes) + copy_storage["bytes"]
                    if type(part_storage) is not int or part_storage < 0 or part_storage > source_room:
                        raise ValueError("Automatic backend exceeded the cumulative workspace byte cap")
                    if type(part_bytes) is not int or part_bytes < 0 or part_bytes > remaining_budget:
                        raise ValueError("Automatic backend exceeded its frozen source byte budget")
                    part_receipts, part_artifacts = prepared.get("receipts", []), prepared.get("artifacts", [])
                    record = {"candidate": candidate, "source": source,
                        "source_index_sha256": digest(source["source_index"]),
                        "reserved_download_bytes": part_bytes, "receipts": part_receipts,
                        "original_storage_bytes": part_storage, "header_rejections": diagnostics,
                        "artifacts": part_artifacts}
                    if compact:
                        record.update(retired_archive_bytes=prepared.get("retired_archive_bytes", 0),
                            temporary_archive_peak_bytes=prepared.get("temporary_archive_peak_bytes", 0))
                        if canonical:
                            record.update(canonical_storage_bytes=prepared.get("canonical_storage_bytes", 0),
                                retired_original_bytes=prepared.get("retired_original_bytes", 0),
                                acquired_original_peak_bytes=prepared.get("acquired_original_peak_bytes", 0))
                            if prepared.get("canonical_receipt"):
                                record["canonical_receipt"] = prepared["canonical_receipt"]
                        elif prepared.get("compact_receipt"):
                            record["compact_receipt"] = prepared["compact_receipt"]
                    _frozen(record_path, record)
                reserved_bytes += part_bytes
                source_storage_bytes += part_storage
                if compact:
                    retired_archive_bytes += record.get("retired_archive_bytes", 0)
                    temporary_archive_peak_bytes = max(temporary_archive_peak_bytes,
                        record.get("temporary_archive_peak_bytes", 0))
                if canonical:
                    canonical_storage_bytes += record.get("canonical_storage_bytes", 0)
                    retired_original_bytes += record.get("retired_original_bytes", 0)
                    acquired_original_peak_bytes = max(acquired_original_peak_bytes,
                        record.get("acquired_original_peak_bytes", 0))
                if reserved_bytes > requested["limits"]["max_download_bytes"]:
                    raise ValueError("Persisted automatic sources exceed their frozen byte budget")
                if source_storage_bytes + space["minimum_known_bytes"] > requested["limits"]["max_workspace_bytes"]:
                    raise ValueError("Persisted automatic sources exceed their cumulative workspace cap")
                receipts.extend(part_receipts)
                artifacts.extend([str(record_path), source["source_index"], *part_artifacts])
                if not rows:
                    continue
                try:
                    qualification = qualify_source(source) if qualify_source else None
                except ValueError as error:
                    # Only the preparation API's explicit all-technical-rejects
                    # outcome is a zero-hour part. Integrity, space, metadata
                    # and arbitrary callback failures must still stop closed.
                    if str(error) != "All sources failed technical checks; see checks records":
                        raise
                    qualification = {"train_hours": 0, "target_met": False,
                                     "no_usable_sources": True, "reason": str(error)}
                    failed = part / "qualification_failed.json"
                    _frozen(failed, {"source_index": _binding(source["source_index"]),
                                    "qualification": qualification, "content_approved": False})
                    artifacts.append(str(failed))
                hours = _hours(qualification["train_hours"] if qualification else sum(row["duration_s"] for row in rows)/3600,
                               "Qualified Train")
                header_hours = sum(row["duration_s"] for row in rows)/3600
                if hours > header_hours + 1e-6:
                    raise ValueError("Qualified Train hours exceed their unique source audio")
                if qualification:
                    source["prepared"] = qualification
                if hours > 0:
                    sources.append(source)
                actual[component] += hours
                seen_hashes.update(row.get("original_source_sha256", row["sha256"]) for row in rows)
                seen_sources.update(row["canonical_source_id"] for row in rows)
                seen_tracks.update(int(row["track_id"]) for row in rows if row.get("track_id") is not None)
                write_json(state / "progress.json", {"component_actual_hours": actual,
                    "reserved_source_bytes": reserved_bytes, "source_storage_bytes": source_storage_bytes,
                    "content_approved": False, "updated_at": now()})
                if progress:
                    progress({"label": "학습용 비호출 자료 기술 검사 완료: " + component,
                              "done": actual[component], "total": target, "unit": "hours", "part": key})
            if actual[component] + 1e-9 < target:
                shortages.append(f"{component}: {actual[component]:g} unique {'post-QC Train' if qualify_source else 'header'} hours prepared, {target:g} required")
        result = {"sources": sources, "plan": {**policy, "request": _binding(request_path)},
            "receipts": receipts, "artifacts": list(dict.fromkeys(artifacts)),
            "component_actual_hours": actual, "shortages": list(dict.fromkeys(shortages)),
            "actual_download_bytes": reserved_bytes, "download_byte_scope": "Conservative pinned source byte reservation, not a network-retry traffic measurement",
            "source_storage_bytes": source_storage_bytes, "space_plan": space,
            "hours_scope": "post_technical_QC_Train" if qualify_source else "provider_index_and_audio_headers_only",
            "content_approved": False, "training_admitted": False, "Final_accessed": False}
        if compact:
            result.update(storage_policy=policy["storage_policy"], retired_archive_bytes=retired_archive_bytes,
                          temporary_archive_peak_bytes=temporary_archive_peak_bytes)
        if canonical:
            result.update(canonical_storage_bytes=canonical_storage_bytes,
                retired_original_bytes=retired_original_bytes,
                acquired_original_peak_bytes=acquired_original_peak_bytes)
        write_json(state / "result.json", result)
        return result
