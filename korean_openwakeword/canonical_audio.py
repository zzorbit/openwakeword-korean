"""Lossless storage of the exact, established mono16k PCM16 training input.

Container checksums protect files. Logical WAV checksums preserve historical
selection/deduplication/augmentation semantics across WAV and FLAC containers.
No input file is removed by this codec; acquisition ownership is a separate job.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from .common import bound_file, digest, read_json, write_json

POLICY = "canonical_flac_only_v1"


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def pcm_identity(path):
    """Hash decoded little-endian PCM16 in bounded blocks, independently of container."""
    hasher, frames = hashlib.sha256(), 0
    with sf.SoundFile(path) as stream:
        if (stream.samplerate, stream.channels, stream.subtype) != (16000, 1, "PCM_16"):
            raise ValueError("Canonical audio must be mono16k PCM16")
        while True:
            block = stream.read(65536, dtype="int16")
            if not len(block):
                break
            hasher.update(block.astype("<i2", copy=False).tobytes())
            frames += len(block)
    return {"sha256": hasher.hexdigest(), "frames": frames, "bytes": frames * 2}


def canonicalize_audio(source, output):
    """Normalize once with the existing algorithm; prove FLAC decodes identically."""
    from .preparation import _convert
    source, output = Path(source).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix(output.suffix + ".json")
    source_sha = digest(source)
    if receipt.exists():
        saved = read_json(receipt)
        if saved["original"]["sha256"] != source_sha:
            raise ValueError("Canonical source changed on resume")
        bound_file(output.parent, saved["canonical"])
        return saved
    wav = output.with_suffix(output.suffix + ".reference.wav")
    temporary = output.with_suffix(output.suffix + ".partial.flac")
    header = sf.info(source)
    required = int(np.ceil(header.frames * 16000 / header.samplerate)) * 4 + 1024 * 1024
    if shutil.disk_usage(output.parent).free < required:
        raise ValueError("Insufficient space for one PCM reference and lossless FLAC")
    try:
        original_info = _convert(source, wav)
        logical_sha = digest(wav)
        reference_pcm = pcm_identity(wav)
        with sf.SoundFile(wav) as reader, sf.SoundFile(temporary, "w", samplerate=16000,
                channels=1, subtype="PCM_16", format="FLAC") as writer:
            while True:
                block = reader.read(65536, dtype="int16")
                if not len(block):
                    break
                writer.write(block)
        if pcm_identity(temporary) != reference_pcm:
            raise ValueError("Lossless FLAC PCM verification failed; original preserved")
        if digest(source) != source_sha:
            raise ValueError("Source changed during canonical conversion")
        if output.exists():
            # Resume a crash between publishing FLAC and its sidecar without
            # replacing any existing bytes. Rebuild the reference and verify.
            if pcm_identity(output) != reference_pcm:
                raise ValueError("Unverified canonical output differs; existing file preserved")
        else:
            os.replace(temporary, output)
        saved = {"kind": POLICY, "original": {"path": str(source), "sha256": source_sha,
                 "bytes": source.stat().st_size}, "original_info": original_info,
                 "logical_audio_sha256": logical_sha, "pcm": reference_pcm,
                 "canonical": _binding(output), "pcm_equal": True,
                 "conversion": "bounded_polyphase_mono16k_pcm16_v1"}
        write_json(receipt, saved)
        return saved
    finally:
        wav.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def marker_for(saved, receipt):
    return {"policy": POLICY, "receipt": _binding(receipt),
            "original_info": saved["original_info"]}


def verify_canonical_audio(row, *, verify_file=True):
    marker = row.get("canonical_audio")
    if not isinstance(marker, dict) or marker.get("policy") != POLICY:
        raise ValueError("Unrecognized canonical audio provenance")
    receipt = bound_file(Path(row["path"]).parent, marker["receipt"])
    saved = read_json(receipt)
    if (saved.get("kind") != POLICY or saved.get("pcm_equal") is not True
            or row.get("original_source_sha256") != saved["original"]["sha256"]
            or row.get("logical_audio_sha256") != saved["logical_audio_sha256"]
            or marker.get("original_info") != saved["original_info"]):
        raise ValueError("Canonical audio provenance changed")
    # Selected same-volume aliases have a different path but identical bytes.
    if verify_file and digest(row["path"]) != saved["canonical"]["sha256"]:
        raise ValueError("Canonical FLAC changed after PCM verification")
    if row.get("provider_sha256", row.get("sha256")) not in (None, saved["canonical"]["sha256"]):
        raise ValueError("Canonical physical file binding changed")
    return saved


def canonicalize_prepared_source(prepared, output):
    """Adapter for the acquisition transaction; preserve row order and raw identity."""
    from .preparation import _publish_rows
    output = Path(output).resolve()
    audio = output / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    directory = Path(prepared["source"]["directory"]).resolve()
    rows, converted, bindings, pcm_bytes = [], [], [], 0
    for supplied in prepared["rows"]:
        row = dict(supplied)
        relative = Path(row["path"])
        source = (directory / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(directory):
            raise ValueError("Canonical input escapes its provider directory")
        if row.get("sha256") and row["sha256"] != digest(source):
            raise ValueError("Provider waveform changed before canonical conversion")
        target = audio / (relative.as_posix() + ".flac")
        try:
            saved = canonicalize_audio(source, target)
        except (ValueError, RuntimeError) as error:
            # Only known malformed source audio is a QC rejection. Disk,
            # permissions, verification and unknown codec failures must stop.
            malformed = (isinstance(error, sf.LibsndfileError) and error.code in (1, 3, 4)) or str(error) in (
                "Empty source audio", "Nonfinite/out-of-range source samples",
                "PCM conversion refuses nonfinite or clipping input")
            if not malformed or getattr(error, "errno", None) is not None:
                raise
            # Invalid source remains available to the established technical QC.
            # No retirement claim is made for this input.
            target = audio / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            if digest(source) != digest(target):
                raise ValueError("Failed-source preservation copy differs") from error
            row.update(path=target.relative_to(audio).as_posix(), sha256=digest(target),
                       canonical_conversion_rejected=str(error))
            rows.append(row)
            bindings.append(_binding(target))
            continue
        receipt = target.with_suffix(target.suffix + ".json")
        row.update(path=target.relative_to(audio).as_posix(), sha256=saved["canonical"]["sha256"],
                   original_source_sha256=saved["original"]["sha256"],
                   logical_audio_sha256=saved["logical_audio_sha256"],
                   canonical_audio=marker_for(saved, receipt))
        rows.append(row)
        converted.append({"original": saved["original"], "canonical": saved["canonical"]})
        pcm_bytes += saved["pcm"]["bytes"]
        bindings.extend((saved["canonical"], _binding(receipt)))
    index = output / "canonical_index.jsonl"
    _publish_rows(index, rows)
    bindings.append(_binding(index))
    source = {**prepared["source"], "directory": str(audio), "adapter": "manifest",
              "metadata": str(index), "canonical_storage_policy": POLICY}
    result = {**prepared, "source": source, "rows": rows,
              "canonical_index": _binding(index),
              "canonical_audio_bytes": sum(item["canonical"]["bytes"] for item in converted),
              "canonical_pcm16_bytes": pcm_bytes}
    return {"prepared": result, "converted": converted, "bindings": bindings}
