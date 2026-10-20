"""Index MLS/YODAS2 waveforms from hash-bound publisher metadata, without STT.

This translates already extracted audio.  It neither downloads/extracts audio
nor admits it, and never treats a provider video ID as a YouTube or speaker ID.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import soundfile as sf

from .commercial_sources import _validate_resolution
from .common import digest, read_json


_AUDIO = frozenset({".wav", ".flac", ".ogg", ".opus", ".mp3", ".aiff", ".aif", ".au"})
_LANGUAGE = {"german": "German", "dutch": "Dutch", "french": "French", "spanish": "Spanish",
             "italian": "Italian", "portuguese": "Portuguese", "polish": "Polish"}
_MLS_ID = re.compile(r"(\d+)_(\d+)_(\d{6})")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _binding(path):
    return {"path": str(path), "sha256": digest(path)}


def _checked_file(base, asset):
    path = (base / asset["path"]).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size != asset["size_bytes"]:
        raise ValueError("Missing, escaped or size-changed original provider metadata")
    algorithm = asset.get("checksum_algorithm")
    expected = asset.get(algorithm)
    if algorithm not in ("sha256", "sha1", "git_sha1") or not re.fullmatch(
            r"[0-9a-f]{64}" if algorithm == "sha256" else r"[0-9a-f]{40}", expected or ""):
        raise ValueError("Provider metadata needs an original checksum")
    sha256 = hashlib.sha256()
    other = hashlib.sha1() if algorithm != "sha256" else None
    if algorithm == "git_sha1":
        other.update(f"blob {asset['size_bytes']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            if other is not None:
                other.update(chunk)
    actual = sha256.hexdigest()
    if (actual if other is None else other.hexdigest()) != expected:
        raise ValueError("Changed original provider metadata checksum")
    return path, {"path": str(path), "sha256": actual}


def _load_resolution(metadata, dataset):
    if metadata is None:
        raise ValueError("Use source_resolution.json or its bound download receipt as metadata")
    supplied = Path(metadata).resolve()
    if not supplied.is_file():
        raise ValueError("Source metadata receipt is missing")
    record = read_json(supplied)
    if not isinstance(record, dict):
        raise ValueError("Source metadata receipt must be an object")
    receipt = None
    if record.get("kind") == "commercial_source_resolution_v1":
        resolution_path, resolution = supplied, record
    elif record.get("kind") in ("commercial_source_download_v1", "commercial_source_metadata_download_v1"):
        receipt = record
        if record.get("state") != "downloaded" or record.get("metadata_only") is True:
            raise ValueError("Source indexing needs an original-audio download receipt, not metadata-only downloads")
        binding = record.get("source_resolution")
        if not isinstance(binding, dict):
            raise ValueError("Source receipt needs its original resolution binding")
        resolution_path = Path(binding.get("path", ""))
        if not resolution_path.is_absolute():
            resolution_path = supplied.parent / resolution_path
        resolution_path = resolution_path.resolve()
        if resolution_path != supplied.parent / "source_resolution.json" or not resolution_path.is_file():
            raise ValueError("Source resolution binding escapes the receipt directory")
        if binding.get("sha256") != digest(resolution_path):
            raise ValueError("Missing or changed source resolution binding")
        resolution = read_json(resolution_path)
        if record.get("resolution_sha256") != resolution.get("resolution_sha256"):
            raise ValueError("Source receipt belongs to another resolution")
    else:
        raise ValueError("Use a commercial source resolution or completed download receipt")
    _validate_resolution(resolution)
    if resolution.get("source_id") != dataset:
        raise ValueError("Source resolution does not match the selected dataset adapter")
    base = resolution_path.parent
    files, bindings = {}, [_binding(resolution_path)]
    for asset in resolution["metadata_assets"]:
        path, binding = _checked_file(base, asset)
        files[asset["path"]] = path
        bindings.append(binding)
    if receipt is not None:
        expected = resolution["assets"] + resolution["metadata_assets"]
        if receipt["kind"] == "commercial_source_metadata_download_v1":
            expected = resolution["metadata_assets"]
        entries = receipt.get("files")
        if not isinstance(entries, list) or len(entries) != len(expected):
            raise ValueError("Source receipt file coverage is incomplete")
        by_source = {entry.get("source_path"): entry for entry in entries}
        if len(by_source) != len(entries) or set(by_source) != {a["path"] for a in expected}:
            raise ValueError("Source receipt repeats or omits provider files")
        metadata_hashes = {str(path): b["sha256"] for path, b in zip(files.values(), bindings[1:])}
        for asset in expected:
            entry = by_source[asset["path"]]
            path = Path(entry.get("path", "")).resolve()
            if (path != (base / asset["path"]).resolve() or not path.is_relative_to(base)
                    or not path.is_file() or path.stat().st_size != asset["size_bytes"]
                    or entry.get("bytes") != asset["size_bytes"]):
                raise ValueError("Downloaded source file binding is missing, escaped or stale")
            if str(path) in metadata_hashes and entry.get("sha256") != metadata_hashes[str(path)]:
                raise ValueError("Source receipt metadata hash differs from the actual file")
        bindings.append(_binding(supplied))
    return resolution, files, bindings


def _audio_files(directory):
    if not directory.is_dir():
        raise ValueError("Input must be an extracted source audio directory")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in _AUDIO)
    if not paths or any(not p.resolve().is_relative_to(directory) for p in paths):
        raise ValueError("Source audio is missing or escapes the selected directory")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Provider audio IDs repeat across files or formats")
    return paths


def _audio_facts(path, declared_duration):
    info = sf.info(path)
    if info.frames < 1 or info.samplerate < 1:
        raise ValueError("Source audio header is empty or invalid")
    actual = info.frames / info.samplerate
    if not math.isfinite(declared_duration) or declared_duration <= 0 or abs(actual - declared_duration) > .1:
        raise ValueError("Source audio duration differs from original provider metadata")
    return {"sha256": digest(path), "duration_s": actual,
            "provider_duration_s": declared_duration, "provider_audio_header_checked": True}


def _metadata_subset(bindings, lookup, *paths):
    """Bind only the shared resolution and this source's actual caption metadata."""
    return [bindings[0], *(lookup[str(path)] for path in paths)]


def _base_row(path, directory, resolution, bindings, dataset_id, source_url, license, language):
    provider = resolution["provider"]
    if source_url is not None and source_url != provider["source_url"]:
        raise ValueError("Source URL override differs from the pinned provider")
    if license is not None and license != provider["license"]:
        raise ValueError("License override differs from the pinned provider")
    if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id.strip()):
        raise ValueError("Dataset label must be a nonempty string")
    return {"path": path.relative_to(directory).as_posix(), "dataset_id": dataset_id or resolution["source_id"],
        "source_url": provider["source_url"], "license": provider["license"], "upstream_revision": provider["revision"],
        "provider_index_adapter": resolution["source_id"],
        "category": "competing_speech", "speech": True, "language": language,
        "speech_presence": "provider_transcript_present", "transcript_origin": "provider_metadata_not_STT",
        "provider_metadata_bindings": bindings, "content_approved": False, "independent_STT_runs": 0}


def _mls(directory, paths, resolution, files, bindings, dataset_id, source_url, license):
    language = resolution["selection"].get("language")
    if language not in _LANGUAGE:
        raise ValueError("MLS resolution needs a supported publisher language")
    prefix = f"data/mls_{language}/train/"
    try:
        segments, transcripts, filenames = (files[prefix + key] for key in (
            "segments.txt", "transcripts.txt", "audio_filenames.txt"))
    except KeyError:
        raise ValueError("MLS requires original segments, transcripts and archive filename metadata") from None
    compact_bindings = _metadata_subset(bindings, {b["path"]: b for b in bindings}, segments, transcripts)
    selected_parts = set(resolution["selection"]["parts"])
    with filenames.open(encoding="utf-8-sig") as stream:
        declared_parts = {line.strip() for line in stream if line.strip()}
    if not {p + ".tar.gz" for p in selected_parts}.issubset(declared_parts):
        raise ValueError("Selected MLS archive part is absent from the provider filename list")
    by_id, rows = {}, []
    for path in paths:
        match = _MLS_ID.fullmatch(path.stem)
        if match is None:
            raise ValueError("MLS waveform filename needs the actual speaker_book_utterance ID")
        speaker, book, segment = match.groups()
        part = f"{speaker}_{book}_{segment[:-3]}"
        if part not in selected_parts:
            raise ValueError("MLS waveform belongs to an unselected provider archive part")
        by_id[path.stem] = (path, speaker, book)
    selected_segments = {}
    with segments.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4 or _MLS_ID.fullmatch(fields[0]) is None:
                raise ValueError("MLS segments need four original tab-separated fields")
            clip, url, start, end = fields
            match = _MLS_ID.fullmatch(clip)
            part = f"{match[1]}_{match[2]}_{match[3][:-3]}"
            if part not in selected_parts:
                continue
            if clip in selected_segments:
                raise ValueError("MLS segments repeat a selected utterance ID")
            start, end = float(start), float(end)
            if (urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc
                    or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end):
                raise ValueError("MLS segment needs an original URL and finite parent interval")
            selected_segments[clip] = (url, start, end)
    if set(selected_segments) != set(by_id):
        raise ValueError("Extracted MLS audio must cover all selected provider utterances exactly")
    captions = {}
    with transcripts.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2:
                raise ValueError("MLS transcript needs an original tab-separated utterance and text")
            if fields[0] not in by_id:
                continue
            if fields[0] in captions or not fields[1].strip():
                raise ValueError("MLS selected transcripts must be unique and nonempty")
            captions[fields[0]] = fields[1]
    if set(captions) != set(by_id):
        raise ValueError("An MLS waveform is missing its original transcript")
    for clip, (path, speaker, book) in sorted(by_id.items()):
        url, start, end = selected_segments[clip]
        row = _base_row(path, directory, resolution, compact_bindings, dataset_id, source_url, license, _LANGUAGE[language])
        row.update(_audio_facts(path, end - start))
        row.update(canonical_source_id=f"mls:{language}:utterance:{clip}",
            canonical_recording_id="librivox:recording:" + url,
            canonical_speaker_id=f"mls:{language}:speaker:{speaker}", canonical_book_id=f"mls:{language}:book:{book}",
            provider_utterance_id=clip, provider_speaker_id=speaker, provider_book_id=book,
            original_audio_url=url, parent_start_s=start, parent_end_s=end,
            text=captions[clip], category_basis="MLS original train speech corpus, segment/transcript and language metadata")
        rows.append(row)
    return rows


def _json_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("YODAS2 metadata contains duplicate JSON keys")
        result[key] = value
    return result


def _yodas2(directory, paths, resolution, files, bindings, dataset_id, source_url, license):
    subset = resolution["selection"].get("subset", "")
    if re.fullmatch(r"[a-z]{2,3}\d{3}", subset) is None:
        raise ValueError("YODAS2 resolution needs its actual language subset")
    code = re.sub(r"\d+$", "", subset)
    language = {"ko": "Korean", "en": "English", "fr": "French", "de": "German", "es": "Spanish",
                "ja": "Japanese", "it": "Italian", "pt": "Portuguese"}.get(code, code)
    durations, captions, metadata_by_video, annotation_issues = {}, {}, {}, {}
    binding_lookup = {binding["path"]: binding for binding in bindings}
    for part in resolution["selection"]["parts"]:
        base = f"data/{subset}/"
        try:
            duration_path = files[base + f"duration/{part:08d}.txt"]
            text_path = files[base + f"text/{part:08d}.json"]
        except KeyError:
            raise ValueError("YODAS2 needs matching original duration/text metadata for each selected archive") from None
        compact_bindings = _metadata_subset(bindings, binding_lookup, duration_path, text_path)
        with duration_path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 2 or _VIDEO_ID.fullmatch(fields[0]) is None or fields[0] in durations:
                    raise ValueError("YODAS2 duration IDs must be unique valid provider IDs")
                duration = float(fields[1])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("YODAS2 duration must be positive and finite")
                durations[fields[0]] = duration
        with text_path.open(encoding="utf-8-sig") as stream:
            entries = json.load(stream, object_pairs_hook=_json_unique)
        if not isinstance(entries, list):
            raise ValueError("YODAS2 text metadata must contain its original list of video records")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("YODAS2 text video record must be an object")
            video, texts = entry.get("audio_id"), entry.get("text")
            if (not isinstance(video, str) or _VIDEO_ID.fullmatch(video) is None
                    or video in captions or not isinstance(texts, dict)):
                raise ValueError("YODAS2 text IDs must be unique with a supplied utterance mapping")
            utterances, issues = [], []
            for utterance_id, text in texts.items():
                fields = utterance_id.rsplit("-", 3)
                if (len(fields) != 4 or fields[0] != video or any(not p.isdigit() for p in fields[1:])
                        or not isinstance(text, str) or not text.strip()):
                    raise ValueError("YODAS2 utterance needs its actual provider ID and nonempty text")
                start, end = int(fields[-2]) / 100, int(fields[-1]) / 100
                if not 0 <= start < end or video not in durations:
                    raise ValueError("YODAS2 supplied utterance has an invalid parent interval")
                if end > durations[video] + .1:
                    issues.append({"kind": "provider_annotation_outside_declared_video",
                        "utterance_id": utterance_id, "provider_start_s": start, "provider_end_s": end,
                        "provider_video_duration_s": durations[video], "overrun_s": end - durations[video]})
                utterances.append({"utterance_id": utterance_id, "start_s": start, "end_s": end, "text": text})
            captions[video] = sorted(utterances, key=lambda r: (r["start_s"], r["end_s"], r["utterance_id"]))
            metadata_by_video[video] = compact_bindings
            annotation_issues[video] = issues
    by_id = {p.stem: p for p in paths}
    if set(durations) != set(captions) or set(by_id) != set(durations):
        raise ValueError("Extracted YODAS2 audio, duration and text IDs must match selected parts exactly")
    rows = []
    for video, path in sorted(by_id.items()):
        row = _base_row(path, directory, resolution, metadata_by_video[video], dataset_id, source_url, license, language)
        row.update(_audio_facts(path, durations[video]))
        row.update(canonical_source_id="yodas:provider-video:" + video,
            canonical_recording_id="yodas:provider-video:" + video, provider_video_id=video,
            provider_language_subset=subset, split="train", unknown_speaker_group="yodas:unverified-speakers",
            text="\n".join(r["text"] for r in captions[video]), provider_utterances=captions[video],
            provider_annotation_status="timing_out_of_bounds" if annotation_issues[video] else "within_declared_video",
            provider_annotation_issues=annotation_issues[video],
            category_basis="YODAS2 supplied speech transcripts and language subset; not a per-frame speech or genre annotation",
            provider_identity_scope="Corpus video ID, not a YouTube video ID or speaker identity")
        if not captions[video]:
            row.update(excluded=True, exclusion_reason="YODAS2 original provider transcript is empty; speech content is unverified",
                speech=None, speech_presence="unknown", provider_annotation_status="missing_provider_transcript")
        rows.append(row)
    return rows


def build_waveform_index(directory, *, dataset, metadata, dataset_id=None, source_url=None, license=None):
    """Build exact per-file provider rows; technical/content admission stays separate."""
    if dataset not in ("mls", "yodas2"):
        raise ValueError("Waveform source adapter must be mls or yodas2")
    directory = Path(directory).resolve()
    resolution, files, bindings = _load_resolution(metadata, dataset)
    paths = _audio_files(directory)
    rows = (_mls if dataset == "mls" else _yodas2)(
        directory, paths, resolution, files, bindings, dataset_id, source_url, license)
    if {r["path"] for r in rows} != {p.relative_to(directory).as_posix() for p in paths} or len(rows) != len(paths):
        raise ValueError("Provider waveform index must cover every discovered audio file exactly")
    limitations = [
        "Provider transcripts and audio headers are checked; no independent STT, full technical QC, human listening or admission is performed.",
        "Archive bytes and extraction membership are not rehashed here; original download receipts and subsequent preparation retain their own verification scope.",
    ]
    if dataset == "yodas2":
        limitations.extend([
            "Video IDs are opaque provider identities, not YouTube IDs, channel IDs or speaker IDs.",
            "Unknown speakers and unverified cross-version aliases cannot certify held-out independence; indexed YODAS2 is conservative Train-only.",
            "The declared language subset and supplied speech text do not certify every frame's language, speech presence or content genre.",
            "Provider annotation times are preserved unchanged; out-of-duration annotations are reported and are not waveform crop boundaries or corrected alignment.",
            "Files without supplied transcripts remain explicitly excluded index rows; their speech content is not invented or transcribed automatically.",
        ])
    annotations = {"videos_with_out_of_bounds_annotations": sum(bool(r.get("provider_annotation_issues")) for r in rows),
        "out_of_bounds_utterances": sum(len(r.get("provider_annotation_issues", [])) for r in rows),
        "sources_without_provider_transcripts": sum(r.get("provider_annotation_status") == "missing_provider_transcript" for r in rows),
        "timestamps_modified": False, "scope": "Provider reference annotations; not content approval or verified alignment"}
    return {"rows": rows, "provider": resolution["provider"], "provider_metadata_bindings": bindings,
        "source_resolution_sha256": resolution["resolution_sha256"], "provider_metadata_used": True,
        "independent_STT_runs": 0, "content_approved": False, "annotation_quality": annotations,
        "limitations": limitations}
