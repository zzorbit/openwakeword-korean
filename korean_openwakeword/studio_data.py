"""Bounded, checksum-bound public-data recipes for the training studio.

Downloads and technical preparation never imply that a person listened or that
the material was admitted.  All user decisions pass through the existing
representative-review admission code, including explicit whole-pool consent.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import threading
import zipfile

import requests

from .common import (bound_file, digest, identity, iter_rows, now, read_json, read_rows,
                     resolved, write_json, writer_lock)
from .downloads import dataset_catalog, download_checked
from .project import load_project

AS_REV = "5a2fa42a1506470d275a47ff8e1fdac5b364e6ef"
MIT_REV = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"
MIT_RIR_COUNT = 270
FMA_REV = "607364e7d8263f890d23c838802f29637600611f"
DNS_REV = "70f19285c36cca4df2338f9248775ddc50980c6b"
_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".aif", ".aiff", ".au"}


def _official_assets():
    catalog = dataset_catalog()["assets"]
    assets = {key: {**catalog[key], "id": key, "checksum_algorithm": "sha256"}
              for key in ("generic", "zeroth")}
    assets.update({
        "audioset": {"id": "audioset", "filename": "bal_train09.tar", "size_bytes": 968540160,
            "source_url": f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{AS_REV}/data/bal_train09.tar",
            "sha256": "9ab88550e77c8289acf71d19ae3f254fadad16f3cea305ef08a16949a928acf9",
            "checksum_algorithm": "sha256", "upstream_revision": AS_REV,
            "license": "Provider card CC-BY-4.0; preserve individual original YouTube rights limitations",
            "role": "noise_train"},
        "audioset_segments": {"id": "audioset_segments", "filename": "balanced_train_segments.csv", "size_bytes": 1211931,
            "source_url": f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{AS_REV}/data/balanced_train_segments.csv",
            "sha256": "3b83e5ff9ed72cfda37d4e36ce08250e107c1ea70f109f6ab6668ac200db5e94",
            "checksum_algorithm": "sha256", "upstream_revision": AS_REV, "license": "CC-BY-4.0", "role": "provider_metadata"},
        "audioset_ontology": {"id": "audioset_ontology", "filename": "ontology.json", "size_bytes": 342780,
            "source_url": f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{AS_REV}/data/ontology.json",
            "sha256": "9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d",
            "checksum_algorithm": "sha256", "upstream_revision": AS_REV, "license": "CC-BY-4.0", "role": "provider_metadata"},
        "fma": {"id": "fma", "filename": "fma_small.zip", "size_bytes": 7679594875,
            "source_url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            "sha1": "ade154f733639d52e35e32f5593efe5be76c6d70", "checksum_algorithm": "sha1",
            "upstream_revision": FMA_REV,
            "checksum_source": f"https://github.com/mdeff/fma/blob/{FMA_REV}/README.md",
            "license": "Per-track Creative Commons conditions from original tracks.csv; not the HF loader code license",
            "role": "noise_music_train", "selection": "First 120 licensed tracks by provider numeric ID; 30-second clips, approximately one hour total"},
        "fma_metadata": {"id": "fma_metadata", "filename": "fma_metadata.zip", "size_bytes": 358412441,
            "source_url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
            "sha1": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733", "checksum_algorithm": "sha1",
            "checksum_source": f"https://github.com/mdeff/fma/blob/{FMA_REV}/README.md",
            "upstream_revision": FMA_REV, "license": "FMA metadata CC-BY-4.0; audio license is per track", "role": "provider_metadata"},
        "dns_fan": _dns("fan_Freesound_validated_100355_0.wav", "a3d2ede479606cae392b5332973ed1a1a6e9f0d4176e030bb22920379e3cd5ba", "train", "freesound:100355"),
        "dns_door": _dns("door_Freesound_validated_102956_0.wav", "ae7bc15f5460a4809922e5d4594a487b8cfe7f96e0ead159a3627f75d46add85", "validation", "freesound:102956"),
        "electronic_validation": {"id": "electronic_validation", "filename": "1-13613-A-37.wav", "size_bytes": 441044,
            "source_url": "https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/audio/1-13613-A-37.wav",
            "git_sha1": "bb19394d9ff398d31aab405c4ce12c25ff8cdff0", "checksum_algorithm": "git_sha1",
            "upstream_revision": "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6", "license": "Original Freesound 13613 by hookhead CC0; ESC-50 corpus CC-BY-NC-3.0 attribution preserved",
            "canonical_source_id": "freesound:13613", "split": "validation", "role": "held_out_electronic_smoke"},
    })
    return list(assets.values())


def _dns(filename, checksum, split, source):
    return {"id": "dns_" + ("fan" if split == "train" else "door"), "filename": filename,
            "size_bytes": 320044, "source_url": f"https://media.githubusercontent.com/media/microsoft/DNS-Challenge/{DNS_REV}/datasets/noise/{filename}",
            "sha256": checksum, "checksum_algorithm": "sha256", "upstream_revision": DNS_REV,
            "license": "CC0-1.0 Freesound selection per pinned DNS README", "canonical_source_id": source,
            "split": split, "role": "environment_smoke"}


def _preset(preset):
    aliases = {"operation_check": "check", "smoke": "check", "trial": "test", "commercial": "full", "production": "full"}
    preset = aliases.get(preset, preset)
    if preset not in ("check", "test", "full"):
        raise ValueError("Material preset must be check, test or full")
    return preset


def _automatic_source_plan(preset, generic_hours=7000, korean_hours=None):
    """Offline source targets; neither downloaded nor admitted duration."""
    preset = _preset(preset)
    if preset == "check":
        components = []
    elif preset == "test":
        components = [{"id": "yodas2", "provider": "yodas2", "pool": "korean_negative",
                       "name": "한국어 웹 음성", "target_hours": 2000 / 3 if korean_hours is None else korean_hours}]
    else:
        from .commercial_sources import commercial_negative_plan
        plan = commercial_negative_plan(generic_hours, 3000 if korean_hours is None else korean_hours)
        components = [{"id": row["sources"][0]["id"], "provider": row["sources"][0]["id"],
                       "pool": row["pool"], "name": row["name"], "target_hours": row["target_hours"]}
                      for row in plan["components"]]
    return {"version": 1, "enabled": True, "components": components,
            "state": "requires_download_preparation_and_review",
            "scope": "Unique technically usable Train targets; actual review/admission remains required."}


def _automatic_space_plan(preset, config=None, material_acquisition_policy="automatic_v1"):
    from .automatic_sources import automatic_storage_plan
    preset = _preset(preset)
    default_generic, default_korean, originals, rounds = {
        "check": (2000, 1, 1000, 1), "test": (2000, 2000 / 3, 20000, 3),
        "full": (7000, 3000, 100000, 5)}[preset]
    selected = config or {"targets": {"positive_originals": originals, "generic_hours": default_generic,
                                     "korean_hours": default_korean}, "augmentation_rounds": rounds}
    return automatic_storage_plan(selected, generic_hours=selected["targets"]["generic_hours"],
                                  korean_hours=selected["targets"]["korean_hours"], preset=preset,
                                  material_acquisition_policy=material_acquisition_policy)


def material_recipe(preset="check", workspace=None, evaluation_policy=None, noise_material_policy=None,
                    material_acquisition_policy=None):
    """Preview known downloads and limits without accessing network or audio."""
    preset = _preset(preset)
    if material_acquisition_policy not in (None, "automatic_v1", "automatic_v2", "automatic_v3"):
        raise ValueError("Unsupported material acquisition policy")
    automatic = material_acquisition_policy in ("automatic_v1", "automatic_v2", "automatic_v3")
    policy = noise_material_policy or ("reuse_negative_waveforms" if preset == "full" else "official_example")
    if policy not in ("official_example", "reuse_negative_waveforms"):
        raise ValueError("Unsupported training-noise material policy")
    # Keep smoke pins available for historical receipts, but never request the
    # standalone fan in a new training recipe. Official-example AudioSet stays
    # unfiltered; commercial reuse omits the separate augmentation download.
    assets = [a for a in _official_assets() if a["id"] not in ("dns_fan", "dns_door", "electronic_validation")]
    if policy == "reuse_negative_waveforms":
        assets = [a for a in assets if a["id"] not in ("audioset", "audioset_segments", "audioset_ontology", "fma", "fma_metadata")]
    if automatic and preset in ("test", "full"):
        assets = [a for a in assets if a["id"] != "zeroth" and not (preset == "full" and a["id"] == "generic")]
    for asset in assets:
        path = _cached_asset(asset, Path(workspace).resolve() / "shared" / "downloads") if workspace else None
        asset["local_path"] = str(path) if path else None
        asset["availability"] = "cached_requires_integrity_check" if path and path.is_file() else "download_required"
    from .studio_evaluation_data import evaluation_catalog, evaluation_preview
    strict_common = bool(evaluation_policy and evaluation_policy.get("evaluation_materials_recipe") == 6)
    # Recipe 6 starts from the common corpus itself. The historical 193 clips
    # and Zeroth reserve preview remain available only to their old recipes.
    evaluation = {"recipe_version": 6, "assets": [], "size_bytes": 0} if strict_common else evaluation_catalog()
    result = {"schema_version": 1, "recipe_version": evaluation["recipe_version"], "preset": preset,
            "noise_material_policy": policy, "assets": assets,
            "evaluation_assets": evaluation["assets"], "evaluation_sources": [] if strict_common else evaluation_preview(),
            "evaluation_download_bytes": evaluation["size_bytes"],
            "known_download_bytes": sum(a["size_bytes"] for a in assets) + evaluation["size_bytes"],
            "additional_download": {"id": "mit_rir", "source_url": f"https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/tree/{MIT_REV}/16khz",
                "upstream_revision": MIT_REV, "files": MIT_RIR_COUNT, "size_bytes": None,
                "license": "Unknown in provider card; preserve MIT original source terms and review before use",
                "verification": "Each file's LFS SHA256 and size from the pinned provider tree; no guessed hash or room identity"},
            "estimated_workspace_bytes": 90 * 1024**3,
            "automatic_target_hours": {"generic_nominal": 2000, "korean_train": 1},
            "shortages": [] if preset == "check" else [
                "This bounded recipe supplies only publisher-nominal 2,000h Generic and 1h Korean Train; connect additional reviewed materials to reach the selected test/full targets.",
                "More diverse independent continuous Negative evaluation is required for a production claim."],
            "limitations": [
                "The public Generic file contains features only, not waveform audio, and its 2,000h is nominal publisher duration.",
                "FMA's original checksum-verified small archive is 7.68GB even though only 120 tracks are extracted; cached archives are reused.",
                "Unknown AudioSet/FMA speaker identities and MIT rooms stay conservative Train-only groups; no invented independence.",
                "Held-out Development adds environmental sounds, music/electronic sounds, Korean field speech/announcements and reserved Korean read speech; the small curated set does not prove broad population performance.",
                "Downloads and technical QC are not human listening, admission, or Final evaluation approval."],
            "human_review_required": True, "Final_accessed": False}
    if policy == "reuse_negative_waveforms":
        result["limitations"] = [text for text in result["limitations"] if not text.startswith(("FMA's", "Unknown AudioSet/FMA"))]
        result["limitations"].extend([
            "Training noise reuses reviewed Train waveform sources admitted as Generic/Korean Negative; the feature-only Generic file cannot be mixed as audio.",
            "This download still supplies only nominal 2,000h Generic features and 1h Korean Train. Additional reviewed raw Negative sources must meet the selected hours and existing noise-category coverage; they are not automatically downloaded.",
            "MIT rooms retain conservative Train-only groups; no invented room independence."])
        if preset == "full":
            from .commercial_sources import commercial_negative_plan
            result["training_negative_plan"] = commercial_negative_plan()
    if automatic:
        result["material_acquisition_policy"] = material_acquisition_policy
        result["automatic_source_plan"] = _automatic_source_plan(preset)
        result["known_download_bytes_scope"] = "Pinned base and common-evaluation archives only; automatic provider shards/tracks are resolved and preflighted separately."
        result["automatic_target_hours"] = ({"generic_raw": 7000, "korean_train": 3000} if preset == "full"
            else {"generic_nominal": 2000, "korean_train": 2000 / 3 if preset == "test" else 1})
        result["shortages"] = []
        if preset != "check":
            result["limitations"] = [text for text in result["limitations"] if not text.startswith((
                "The public Generic", "This download still supplies", "Held-out Development"))]
            if preset == "test":
                result["limitations"].append("Generic 2,000h are publisher-nominal feature-only hours; Korean audio is automatically selected from pinned YODAS2 sources.")
            result["limitations"].extend([
                "Additional source download bytes depend on the pinned shards and selected original tracks; preflight each transfer before downloading.",
                "Provider hours, technical usable hours, representative review and admitted hours are distinct; failed QC may require another shard."])
    if evaluation_policy and evaluation_policy.get("evaluation_materials_recipe") in (5, 6):
        from .evaluation_materials import evaluation_material_catalog, evaluation_material_preview
        from .evaluation_rir import common_rir_preview, common_rir_catalog
        common = evaluation_material_catalog()
        shared_hashes = {a["sha256"] for a in assets if a.get("sha256")}
        extra_download = sum(a["size_bytes"] for a in common["archives"] if a.get("sha256") not in shared_hashes)
        extra_download += common_rir_catalog()["archive"]["size_bytes"]
        result.update(recipe_version=evaluation_policy["evaluation_materials_recipe"], evaluation_materials_policy=evaluation_policy,
                      evaluation_sources=evaluation_material_preview() + ([] if strict_common else evaluation_preview()) + [common_rir_preview()],
                      common_evaluation_catalog=common,
                      evaluation_download_bytes=evaluation["size_bytes"] + extra_download,
                      known_download_bytes=result["known_download_bytes"] + extra_download,
                      estimated_workspace_bytes=110 * 1024**3)
        result["limitations"].append("30h means unique prepared original audio, not repeated mixtures or approved listening; missing buckets block the common evaluation.")
        if strict_common:
            result["evaluation_source_scope"] = "common_30h_only"
            result["limitations"] = [text for text in result["limitations"] if "reserved Korean read speech" not in text]
            result["limitations"].append("Only the common corpus and its held-out RIRs enter this benchmark; legacy supplemental clips and training-source reserves are excluded.")
    if automatic:
        space = _automatic_space_plan(preset, material_acquisition_policy=material_acquisition_policy)
        # Include the pinned base/evaluation archives separately from provider
        # audio forecasts; this is a planning estimate, not a disk reservation.
        base = result["known_download_bytes"]
        working = 15 * 1024**3
        result["automatic_space_plan"] = {**space, "base_archive_bytes": base,
            "base_working_bytes": working,
            "minimum_total_bytes": base + working + space["minimum_known_bytes"] + space["future_buffer_bytes"] + space.get("transient_archive_bytes", 0),
            "estimated_total_bytes": base + working + space["estimated_total_bytes"]}
        result["estimated_workspace_bytes"] = result["automatic_space_plan"]["estimated_total_bytes"]
        if material_acquisition_policy == "automatic_v3":
            result["audio_storage_policy"] = "canonical_flac_only_v1"
            result["limitations"].append(
                "New public training inputs retain verified canonical FLAC. Only newly acquired backend-owned raw audio is retired after exact PCM verification; existing sources, archives, TTS and evaluation audio are preserved. PCM storage is a reference budget until real FLAC sizes are measured.")
    return result


def _cached_asset(asset, directory):
    """Bounded paths used by this studio and the documented manual commands."""
    primary = directory / asset["filename"]
    candidates = (primary, directory.parent / asset["filename"],
                  directory.parent.parent / "downloads" / asset["filename"])
    return next((p for p in candidates if p.is_file()), primary)


def _emit(progress, label, done=None, total=None, unit=None):
    if progress:
        progress({"label": label, "done": done, "total": total, "unit": unit,
                  "task_id": os.environ.get("KOWW_STUDIO_TASK_ID")})


def _checksum(path, algorithm):
    if algorithm == "sha256":
        return digest(path)
    h = hashlib.sha1()
    if algorithm == "git_sha1":
        h.update(f"blob {Path(path).stat().st_size}\0".encode("ascii"))
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _download(asset, directory, progress=None):
    """Use existing SHA256 downloader; retain original upstream SHA1 pins too."""
    target = _cached_asset(asset, directory)
    if not asset["source_url"].startswith("https://"):
        raise ValueError("Public-data sources must use HTTPS")
    _emit(progress, "자료 다운로드·해시 확인: " + asset["id"], 0, asset["size_bytes"], "bytes")
    algorithm = asset["checksum_algorithm"]
    if algorithm == "sha256" and target.parent != directory:
        if target.stat().st_size != asset["size_bytes"] or digest(target) != asset["sha256"]:
            raise ValueError("Existing manual download differs from the pinned asset")
        result = {"path": str(target), "sha256": asset["sha256"], "bytes": target.stat().st_size,
                  "reused": True, "completed_at": now(), "content_approved": False}
    elif algorithm == "sha256":
        # Keep the already-tested download implementation and expose its
        # actual partial-file length while it is working, including resume.
        stopped = threading.Event()
        def monitor():
            while not stopped.wait(1):
                partial = target.with_name(target.name + ".part")
                try:
                    size = partial.stat().st_size if partial.is_file() else target.stat().st_size if target.is_file() else 0
                    _emit(progress, "자료 다운로드·해시 확인: " + asset["id"], size, asset["size_bytes"], "bytes")
                except OSError:
                    pass  # Atomic rename may occur between stat and exists.
        observer = threading.Thread(target=monitor, daemon=True) if progress else None
        if observer:
            observer.start()
        try:
            result = download_checked(asset["source_url"], target, asset["sha256"], asset["size_bytes"])
        finally:
            stopped.set()
            if observer:
                observer.join()
    else:
        directory.mkdir(parents=True, exist_ok=True)
        expected = asset[algorithm]
        if algorithm not in ("sha1", "git_sha1") or not re.fullmatch("[0-9a-f]{40}", expected):
            raise ValueError("Expected the original upstream SHA1 or Git blob checksum")
        with writer_lock(directory / ("." + target.name + ".download")):
            reused = target.is_file()
            partial = target.with_name(target.name + ".part")
            if not reused:
                offset = partial.stat().st_size if partial.is_file() else 0
                if offset > asset["size_bytes"]:
                    raise ValueError("Partial file exceeds pinned upstream size")
                if offset < asset["size_bytes"]:
                    if shutil.disk_usage(directory).free < asset["size_bytes"] - offset:
                        raise ValueError("Insufficient workspace space for public-data download")
                    headers = {"Accept-Encoding": "identity", **({"Range": f"bytes={offset}-"} if offset else {})}
                    with requests.get(asset["source_url"], headers=headers, stream=True, timeout=(30, 180)) as response:
                        response.raise_for_status()
                        if response.status_code == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if not match or int(match[1]) != offset or int(match[3]) != asset["size_bytes"]:
                                raise ValueError("Mismatched public-data resume range")
                        elif response.status_code == 200:
                            offset = 0
                        else:
                            raise ValueError("Unexpected public-data response")
                        with partial.open("ab" if offset else "wb") as stream:
                            for chunk in response.iter_content(8 * 1024**2):
                                if not chunk:
                                    continue
                                if offset + len(chunk) > asset["size_bytes"]:
                                    raise ValueError("Public-data response exceeds pinned size")
                                stream.write(chunk)
                                offset += len(chunk)
                                _emit(progress, "자료 다운로드: " + asset["id"], offset, asset["size_bytes"], "bytes")
                if partial.stat().st_size != asset["size_bytes"] or _checksum(partial, algorithm) != expected:
                    raise ValueError("Public-data download checksum mismatch; partial is preserved")
                partial.replace(target)
            if target.stat().st_size != asset["size_bytes"] or _checksum(target, algorithm) != expected:
                raise ValueError("Existing public-data archive differs from upstream checksum")
            result = {"path": str(target), "sha256": digest(target), "bytes": target.stat().st_size,
                      "verified_upstream_algorithm": algorithm, "verified_upstream_checksum": expected,
                      "reused": reused, "completed_at": now(), "content_approved": False}
    result.update(source_url=asset["source_url"], license=asset["license"], upstream_revision=asset.get("upstream_revision"))
    # A manual download's original receipt can already be bound by another
    # project. Leave it untouched when reusing the original in place.
    receipt = (target.with_name(target.name + ".receipt.json") if target.parent == directory
               else directory / (target.name + ".reused.receipt.json"))
    write_json(receipt, result)
    _emit(progress, "자료 확인 완료: " + asset["id"], asset["size_bytes"], asset["size_bytes"], "bytes")
    return target


def _archive_target(root, name):
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or "\\" in name or ":" in name
            or any(p in (".", "..") or p.rstrip(" .") != p for p in parts)
            or any(re.fullmatch(r"(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p, re.I) for p in parts)):
        raise ValueError("Unsafe public archive member path")
    target = root.joinpath(*parts).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Public archive member escapes extraction directory")
    return target


def extract_archive(archive, output, *, members=None, max_bytes=40 * 1024**3):
    """Safely resume exact selected files; never follow links or call extractall."""
    archive, output = Path(archive).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = set(members) if members is not None else None
    request = {"archive": {"path": str(archive), "sha256": digest(archive)},
               "members": sorted(requested) if requested is not None else None, "max_bytes": max_bytes}
    receipt_path = output / "extraction_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("request") != request:
            raise ValueError("Archive extraction request changed; use another directory")
        if all((output / item["path"]).is_file() and digest(output / item["path"]) == item["sha256"] for item in receipt["files"]):
            return {**receipt, "resumed": True}
        raise ValueError("Published extracted files changed; original archive preserved")
    with writer_lock(output):
        is_zip = zipfile.is_zipfile(archive)
        handle = zipfile.ZipFile(archive) if is_zip else tarfile.open(archive, "r:*")
        with handle:
            inventory = handle.infolist() if is_zip else handle.getmembers()
            selected, names, targets, required = [], set(), set(), 0
            for member in inventory:
                name = member.filename if is_zip else member.name
                directory_member = member.is_dir() if is_zip else member.isdir()
                if directory_member and name.rstrip("/") in ("", "."):
                    continue  # A conventional tar root entry creates nothing.
                # Validate every member, even one outside the selected subset.
                target = _archive_target(output, name.rstrip("/"))
                mode = member.external_attr >> 16 if is_zip else member.mode
                if (is_zip and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)) or (not is_zip and not (member.isfile() or member.isdir())):
                    raise ValueError("Archive links and special files are not allowed")
                if directory_member:
                    continue
                if name in names or target in targets:
                    raise ValueError("Duplicate public archive member path")
                names.add(name)
                targets.add(target)
                if requested is not None and name not in requested:
                    continue
                size = member.file_size if is_zip else member.size
                if not 0 <= size <= 512 * 1024**2:
                    raise ValueError("Public archive member exceeds extraction safety limit")
                required += size
                selected.append((member, target, size))
            if requested is not None and requested - names:
                raise ValueError("Requested public archive members are missing")
            if required > max_bytes or shutil.disk_usage(output).free < required:
                raise ValueError("Insufficient disk space or extraction exceeds recipe budget")
            records = []
            for member, target, size in selected:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                source = handle.open(member) if is_zip else handle.extractfile(member)
                with source, partial.open("wb") as stream:
                    remaining = size
                    while remaining:
                        block = source.read(min(1024**2, remaining))
                        if not block:
                            raise EOFError("Truncated public archive member")
                        stream.write(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise ValueError("Archive member exceeds its declared size")
                actual = digest(partial)
                if target.exists() and digest(target) != actual:
                    raise ValueError("Interrupted extraction contains a conflicting file")
                os.replace(partial, target)
                records.append({"path": target.relative_to(output).as_posix(), "sha256": actual, "bytes": size})
        receipt = {"request": request, "files": records, "bytes": required, "completed_at": now(), "content_approved": False}
        write_json(receipt_path, receipt)
        return {**receipt, "resumed": False}


def _fetch_mit(shared, progress=None):
    directory = shared / "mit_rir"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "provider_tree.json"
    url = f"https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/{MIT_REV}/16khz?limit=1000"
    if manifest.is_file():
        saved = read_json(manifest)
        if saved.get("source_url") != url:
            raise ValueError("MIT cached provider revision changed")
        entries = saved["entries"]
    else:
        response = requests.get(url, timeout=(30, 120))
        response.raise_for_status()
        if response.links.get("next"):
            raise ValueError("MIT provider listing unexpectedly paginated")
        entries = response.json()
        write_json(manifest, {"source_url": url, "entries": entries, "retrieved_at": now()})
    entries = [r for r in entries if r.get("type") == "file" and r.get("path", "").endswith(".wav")]
    if len(entries) != MIT_RIR_COUNT or len({r["path"] for r in entries}) != MIT_RIR_COUNT:
        raise ValueError(f"Pinned MIT provider inventory must contain exactly {MIT_RIR_COUNT} WAVs")
    for order, row in enumerate(entries):
        path = _archive_target(directory, row["path"])
        checksum = row.get("lfs", {}).get("oid")
        if not isinstance(checksum, str) or not re.fullmatch("[0-9a-f]{64}", checksum) or row["size"] > 8 * 1024**2:
            raise ValueError("MIT provider file lacks bounded LFS SHA256 metadata")
        download_checked(f"https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/{MIT_REV}/{row['path']}", path, checksum, row["size"])
        _emit(progress, "잔향 자료 확인", order + 1, len(entries), "files")
    return directory / "16khz"


def _fma_selection(metadata, archive, output):
    """Select 120 provider-licensed original tracks, retaining original metadata."""
    with zipfile.ZipFile(archive) as stream:
        available = {Path(n).stem: n for n in stream.namelist() if n.endswith(".mp3") and Path(n).stem.isdigit()}
    with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = list(zip(next(reader), next(reader)))
        license_i, subset_i = columns.index(("track", "license")), columns.index(("set", "subset"))
        selected = []
        for row in reader:
            if not row or not row[0].isdigit():
                continue
            track = f"{int(row[0]):06d}"
            terms = row[license_i].strip()
            if (track in available and row[subset_i] == "small"
                    and terms.lower() not in ("", "nan", "unknown", "none")):
                selected.append((int(track), available[track]))
    chosen = [name for _, name in sorted(selected)[:120]]
    if len(chosen) != 120:
        raise ValueError("FMA archive and original metadata need 120 licensed small-subset tracks")
    return extract_archive(archive, output, members=chosen, max_bytes=200 * 1024**2)


def _write_rows(path, rows):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    if path.exists() and path.stat().st_size and path.read_text(encoding="utf-8") != value:
        raise ValueError("Existing studio manifest differs from immutable prepared sources")
    if not path.exists() or not path.stat().st_size:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    return {"path": str(path.resolve()), "sha256": digest(path)}


def _candidate_pool_hours(root, queues):
    """Measure technical raw candidates without treating them as approved."""
    from .data import coverage_interval, union_hours
    intervals = {"generic_negative": {}, "korean_negative": {}}
    seen = {pool: set() for pool in intervals}
    for queue in queues:
        selected = set(queue.get("pools", ())).intersection(intervals)
        if not selected or queue.get("evaluation_only"):
            continue
        for row in iter_rows(resolved(root, queue["pool_manifest"])):
            if row.get("category") == "rir" or row.get("split") != "train":
                continue
            start, end = row.get("start_s", 0), row.get("end_s", row.get("duration_s", 0))
            for pool in selected:
                if pool == "korean_negative" and not (row.get("speech") and row.get("language") in ("ko", "kor", "Korean")):
                    continue
                exact = (row["sha256"], start, end)
                if exact in seen[pool]:
                    continue
                seen[pool].add(exact)
                source, interval = coverage_interval(row, start, end)
                intervals[pool].setdefault(source, []).append(interval)
    return {pool: union_hours(values) for pool, values in intervals.items()}


def prepare_materials(project_path, workspace, preset="check", progress=None, evaluation_policy=None,
                      noise_material_policy=None, material_acquisition_policy=None):
    """Download, normalize and reserve sources, then return actual review gates."""
    from .preparation import create_source_index, prepare_public_audio
    from .assets import import_generic
    from .frontend import PINNED_ASSETS

    project_path = Path(project_path).resolve()
    root, config = load_project(project_path)
    workspace = Path(workspace).resolve()
    if any(part.lower() == ".codex" for part in workspace.parts):
        raise ValueError("Public learning material must not be stored inside .codex")
    shared = workspace / "shared"
    downloads = shared / "downloads"
    recipe = material_recipe(preset, workspace, evaluation_policy, noise_material_policy,
                             material_acquisition_policy)
    automatic = recipe.get("material_acquisition_policy") in ("automatic_v1", "automatic_v2", "automatic_v3")
    workspace.mkdir(parents=True, exist_ok=True)
    # A minimum preflight before any download; actual per-file checks also run.
    missing = sum(a["size_bytes"] for a in recipe["assets"] if a["availability"] == "download_required")
    if recipe.get("recipe_version") in (5, 6):
        from .evaluation_rir import common_rir_catalog
        shared_pins = {a["sha256"] for a in recipe["assets"] if a.get("sha256")}
        for asset in recipe["common_evaluation_catalog"]["archives"]:
            if asset.get("sha256") in shared_pins:
                continue
            cached = _cached_asset(asset, downloads)
            if not cached or not cached.is_file():
                missing += asset["size_bytes"]
        rir_archive = common_rir_catalog()["archive"]
        if not (downloads / "common_rir" / rir_archive["filename"]).is_file():
            missing += rir_archive["size_bytes"]
    reserve = 4 * 1024**3
    if automatic and recipe["preset"] in ("test", "full"):
        space = _automatic_space_plan(recipe["preset"], config, material_acquisition_policy)
        namespace = "automatic_sources_" + material_acquisition_policy.removeprefix("automatic_")
        ongoing = (root / ".studio/materials" / namespace / "request.json").is_file()
        if not ongoing:
            reserve = max(reserve, space["minimum_known_bytes"] + space["future_buffer_bytes"] + 15 * 1024**3 + space.get("transient_archive_bytes", 0))
        check = {**space, "missing_base_archive_bytes": missing, "required_free_bytes": missing + reserve,
                 "available_bytes": shutil.disk_usage(workspace).free, "provider_request_exists": ongoing,
                 "audio_download_started": False, "created_at": now()}
        write_json(root / ".studio/materials/automatic_base_preflight.json", check)
    free = shutil.disk_usage(workspace).free
    if free < missing + reserve:
        if automatic:
            required = missing + reserve
            raise ValueError(f"자료 준비 공간이 부족합니다. 최소 필요 {required / 1024**3:,.1f} GiB / "
                f"선택 드라이브 여유 {free / 1024**3:,.1f} GiB / 부족 {(required - free) / 1024**3:,.1f} GiB. "
                "이는 정규화 음원·feature·미다운로드 기본 자료의 하한이며, 추가 원본 용량은 별도입니다. "
                "목표량을 줄이지 않고 다운로드 전에 중단했습니다.")
        raise ValueError("Not enough selected-workspace space for the displayed data recipe and preparation")
    assets = {a["id"]: a for a in recipe["assets"]}
    legacy_extra = root / ".studio/materials/extra_source_index.jsonl"
    if recipe.get("recipe_version") != 6 and legacy_extra.exists() and any(r.get("canonical_source_id") in ("freesound:102956", "freesound:13613") for r in read_rows(legacy_extra)):
        assets.update({a["id"]: a for a in _official_assets() if a["id"] in ("dns_door", "electronic_validation")})
    paths = {key: _download(asset, downloads, progress) for key, asset in assets.items()}
    if "zeroth" in paths:
        extract_archive(paths["zeroth"], shared / "zeroth")
    if recipe["noise_material_policy"] == "official_example":
        extract_archive(paths["audioset"], shared / "audioset")
        # The publisher's tracks.csv expands to 260,414,445 bytes. Keep a
        # bounded allowance for that one CSV, excluding the much larger
        # feature tables that also live in the metadata archive.
        extract_archive(paths["fma_metadata"], shared / "fma_metadata", members=["fma_metadata/tracks.csv"], max_bytes=512 * 1024**2)
        tracks = shared / "fma_metadata/fma_metadata/tracks.csv"
        _fma_selection(tracks, paths["fma"], shared / "fma_1h")
    mit = _fetch_mit(shared, progress)
    stage = root / ".studio" / "materials"
    stage.mkdir(parents=True, exist_ok=True)

    # The old shared/check_noise directory can still contain the retired fan.
    # Never scan or rewrite it. Only historical evaluation-only sources need
    # this separate directory; fresh recipes have no extra smoke source at all.
    extra = shared / "legacy_evaluation_smoke"
    extra_rows = []
    for key in (key for key in ("dns_door", "electronic_validation") if key in assets):
        extra.mkdir(parents=True, exist_ok=True)
        asset, original = assets[key], paths[key]
        destination = extra / asset["filename"]
        if destination.exists() and digest(destination) != digest(original):
            raise ValueError("Existing held-out smoke audio differs from original")
        if not destination.exists():
            shutil.copyfile(original, destination)
        extra_rows.append({"path": destination.name, "dataset_id": "dns_pinned_smoke" if key.startswith("dns") else "esc50_electronic_smoke",
            "canonical_source_id": asset["canonical_source_id"], "canonical_recording_id": asset["canonical_source_id"],
            "source_url": asset["source_url"], "license": asset["license"], "sha256": digest(destination),
            "split": asset["split"], "category": "music" if key == "electronic_validation" else "environment",
            "language": "unknown", "speech": None if key == "electronic_validation" else False,
            "speech_presence": "unknown" if key == "electronic_validation" else "declared",
            "category_basis": "ESC-50 electronic alarm (music/electronic evaluation category)" if key == "electronic_validation" else "Pinned DNS Freesound environmental source metadata"})
    sources = []
    if "zeroth" in paths:
        sources.append({"id": "zeroth", "directory": shared / "zeroth", "adapter": "zeroth", "license": assets["zeroth"]["license"],
            "source_url": "https://www.openslr.org/40/", "target_hours": 1, "validation_fraction": .1, "pools": ["korean_negative", "noise"]})
    sources.append({"id": "mit_rir", "directory": mit, "adapter": "mit_rir", "validation_fraction": 0, "pools": ["rir"]})
    if recipe["noise_material_policy"] == "official_example":
        sources[-1:-1] = [
            {"id": "audioset", "directory": shared / "audioset", "adapter": "audioset", "metadata": paths["audioset_segments"],
             "ontology": paths["audioset_ontology"], "validation_fraction": 0, "pools": ["noise"]},
            {"id": "fma", "directory": shared / "fma_1h", "adapter": "fma", "metadata": tracks, "validation_fraction": 0, "pools": ["noise"]},
        ]
    if extra_rows:
        extra_index = stage / "legacy_evaluation_source_index.jsonl"
        _write_rows(extra_index, extra_rows)
        sources.append({"id": "legacy_evaluation_smoke", "directory": extra,
                        "index": extra_index, "validation_fraction": 0, "pools": ["noise"]})
    def prepare_source(source):
        name = source["id"]
        provided_index = source.get("source_index") or source.get("index")
        if isinstance(provided_index, dict):
            provided_index = bound_file(root, provided_index)
        index = Path(provided_index) if provided_index else stage / (name + "_source_index.jsonl")
        if not provided_index:
            create_source_index(source["directory"], output=index, dataset=source["adapter"],
                                dataset_id=name, metadata=source.get("metadata"), ontology=source.get("ontology"),
                                license=source.get("license"), source_url=source.get("source_url"))
        # Automatic refill adds a new immutable preparation when its candidate
        # index grows; it must never rewrite a partial/finished old preparation.
        suffix = "_" + digest(index)[:16] if provided_index else ""
        return prepare_public_audio(project_path, input_directory=source["directory"], output=stage / (name + suffix),
                    source_index=index, target_hours=source.get("target_hours"),
                    validation_fraction=source.get("validation_fraction", 0), representatives_per_stratum=2)

    automatic_preparation = None
    if automatic and recipe["preset"] in ("test", "full"):
        from .automatic_sources import prepare_automatic_sources
        automatic_preparation = prepare_automatic_sources(project_path, workspace,
            generic_hours=config["targets"]["generic_hours"], korean_hours=config["targets"]["korean_hours"],
            preset=recipe["preset"], progress=progress, qualify_source=prepare_source,
            **({"material_acquisition_policy": material_acquisition_policy} if material_acquisition_policy in ("automatic_v2", "automatic_v3") else {}))
        for supplied in automatic_preparation["sources"]:
            # Provider selection records describe their Negative purpose. The
            # Studio review explicitly admits the same originals for mixing,
            # without editing those immutable provider records or old consent.
            pools = list(supplied["pools"])
            if set(pools).intersection(("generic_negative", "korean_negative")) and "noise" not in pools:
                pools.append("noise")
            sources.append({**supplied, "pools": pools})
    queues, reserves, preparations = [], [], []
    for order, source in enumerate(sources):
        name = source["id"]
        _emit(progress, "자료 변환·분리: " + name, order, len(sources), "datasets")
        record = source.get("prepared") or prepare_source(source)
        binding = record["preparation"]
        preparation = binding["path"]
        queues.append({"id": "materials_" + name, "kind": "content", "queue": record["artifacts"]["representative_queue"]["path"],
                       "pool_manifest": record["artifacts"]["source_manifest"]["path"], "preparation": preparation,
                        "pools": source["pools"], "label": name + " 대표 음원 확인",
                        **({"automatic_component": source["component"]} if source.get("component") else {})})
        if source.get("component"):
            title = {"audioset_unbalanced": "AudioSet 일반 오디오 확인", "mls": "MLS 다국어 낭독 확인",
                     "fma_full": "FMA 음악 확인", "yodas2": "YODAS2 한국어 음성 확인"}[source["component"]]
            queues[-1].update(title=title, label=title,
                description="자료 종류가 맞고 소리가 깨지지 않으면 보존하고, 이상한 항목만 폐기·재검수로 표시하세요.")
        if recipe.get("recipe_version") != 6:
            reserves.extend(read_rows(record["artifacts"]["validation_reserve"]["path"]))
        preparations.append({"id": name, "preparation": binding, "pools": list(source["pools"]), "train_hours": record["train_hours"],
                             "validation_hours": record["validation_hours"], "target_met": record["target_met"]})
    validation = _write_rows(stage / "noise_validation.jsonl", [r for r in reserves if r["category"] != "rir"])
    negative = _write_rows(stage / "negative_validation.jsonl", [{**r, "kind": "negative", "calls": []}
                          for r in reserves if r["category"] != "rir"])
    generic_output = resolved(root, config["pool_features"]["generic_negative"])
    if "generic" in paths:
        metadata_path = generic_output / "metadata.json"
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            if (metadata.get("feature_sha256") != assets["generic"]["sha256"]
                    or Path(metadata.get("feature_path", "")).resolve() != paths["generic"].resolve()
                    or metadata.get("canonical_frontend_assets") != {"melspectrogram": PINNED_ASSETS["melspectrogram.onnx"], "embedding": PINNED_ASSETS["embedding_model.onnx"]}):
                raise ValueError("Existing Generic import differs from the studio recipe")
        else:
            _emit(progress, "일반 비호출 특징 전체 검사·연결")
            metadata = import_generic(paths["generic"], generic_output, {"model_sha256": {
                "melspectrogram": PINNED_ASSETS["melspectrogram.onnx"], "embedding": PINNED_ASSETS["embedding_model.onnx"]}})
        generic_import = {"features": str(paths["generic"]), "output": str(generic_output), "imported": True,
                          "publisher_declared_hours": metadata["publisher_declared_hours"], "unique_source_hours": None}
    else:
        generic_import = {"mode": "raw_generic", "output": str(generic_output), "imported": False,
                          "publisher_declared_hours": None, "unique_source_hours": None,
                          "scope": "Raw sources are technical candidates; hours become admitted only after review."}
    shortages = list(recipe["shortages"])
    if automatic_preparation:
        shortages.extend(automatic_preparation.get("shortages", []))
    if "generic" in paths and config["targets"]["generic_hours"] > generic_import["publisher_declared_hours"]:
        shortages.append(f"Generic target {config['targets']['generic_hours']}h exceeds the available publisher nominal {generic_import['publisher_declared_hours']:g}h; choose raw Generic preparation for this budget")
    if (not automatic or recipe["preset"] == "check") and config["targets"]["korean_hours"] > 1:
        shortages.append(f"Korean target {config['targets']['korean_hours']}h exceeds this bounded 1h preparation; add reviewed admissions")
    if any(not r["target_met"] for r in preparations):
        shortages.append("One or more source preparations did not reach the actual unique Train duration quota")
    result = {"kind": "studio_material_preparation_v1", "project": str(project_path), "preset": _preset(preset),
              "noise_material_policy": recipe["noise_material_policy"],
              "state": "awaiting_human_review", "review_queues": queues, "preparations": preparations,
              "generic_feature_import": generic_import,
              "validation_noise_manifest": validation, "negative_validation_manifest": negative,
              "validation_source_count": len(reserves), "validation_unique_source_hours": sum(r["duration_s"] for r in reserves) / 3600,
              "shortages": shortages, "limitations": recipe["limitations"], "human_listened_count": 0,
              "training_admitted": False, "Final_accessed": False, "created_at": now()}
    if automatic:
        result.update(material_acquisition_policy=material_acquisition_policy,
                      automatic_source_plan=_automatic_source_plan(recipe["preset"], config["targets"]["generic_hours"], config["targets"]["korean_hours"]),
                      candidate_unique_hours=_candidate_pool_hours(root, queues))
        if automatic_preparation:
            result["automatic_source_preparation"] = {key: value for key, value in automatic_preparation.items() if key != "sources"}
    legacy_result = stage / (f"preparation_result_{material_acquisition_policy}.json" if automatic else "preparation_result.json")
    if not legacy_result.exists():
        write_json(legacy_result, result)
    if evaluation_policy and evaluation_policy.get("evaluation_materials_recipe") in (5, 6):
        result["evaluation_materials_policy"] = evaluation_policy
    if recipe.get("recipe_version") == 6:
        result["evaluation_source_scope"] = "common_30h_only"
    return ensure_evaluation_materials(project_path, workspace, result, progress)


def ensure_evaluation_materials(project, workspace, prepared, progress=None):
    """Add only the new held-out recipe; keep legacy Train approvals untouched."""
    from copy import deepcopy
    from .studio_evaluation_data import prepare_evaluation_materials, prepare_common_evaluation_materials
    root, _ = load_project(project)
    result = deepcopy(prepared)
    if result.get("recipe_version") == 6 and (
            result.get("evaluation_materials_policy", {}).get("evaluation_materials_recipe") != 6
            or result.get("evaluation_source_scope") != "common_30h_only"):
        raise ValueError("Unified common evaluation preparation lost its source-pool binding")
    expanded = result.get("evaluation_materials_policy", {}).get("evaluation_materials_recipe") in (5, 6)
    evaluation = (prepare_common_evaluation_materials if expanded else prepare_evaluation_materials)(project, workspace, result, progress)
    binding = evaluation["preparation"]
    version = evaluation["recipe_version"]
    descriptor = {"id": f"materials_evaluation_v{version}_" + evaluation["catalog_id"], "kind": "content",
        "evaluation_only": True, "intended_use": "held_out_development_noise", "pools": [],
        "queue": evaluation["artifacts"]["representative_queue"]["path"],
        "pool_manifest": evaluation["artifacts"]["source_manifest"]["path"], "preparation": binding["path"],
        "label": "학습과 분리한 환경음·음악·전자음·한국어 음성 시험용 배경 음원"}
    # Replace the old evaluation queue, not the training review queues. The new
    # preparation inherits only hash-verified admissions, including discards.
    result["review_queues"] = [q for q in result.get("review_queues", [])
                               if not q.get("evaluation_only") or q["id"] == descriptor["id"]]
    ids = {q["id"] for q in result.get("review_queues", [])}
    if descriptor["id"] not in ids:
        result.setdefault("review_queues", []).append(descriptor)
    result.update(recipe_version=version, evaluation_preparation=binding,
                  evaluation_noise_source_count=evaluation["source_count"],
                  evaluation_noise_type_counts=evaluation["eval_type_counts"],
                  evaluation_only_review_required=True)
    if expanded:
        result["evaluation_material_coverage"] = evaluation["material_coverage"]
        result["evaluation_rir_coverage"] = evaluation["rir_coverage"]
    if version == 6:
        result["evaluation_source_scope"] = evaluation["evaluation_source_scope"]
        result["negative_duration_budget"] = evaluation.get("negative_duration_budget")
        descriptor["evaluation_source_scope"] = evaluation["evaluation_source_scope"]
    path = Path(binding["path"]).parent / f"preparation_result_v{version}.json"
    if not path.exists():
        write_json(path, result)
    return result


def prepare_additional_materials(project_path, workspace, source, *, noise_material_policy=None):
    """Prepare an explicitly supplied public corpus folder through the same UI.

    The source is user-supplied; no licensing, category, transcript, speaker or
    room fact is guessed. A manifest adapter requires actual per-file metadata.
    """
    from .preparation import create_source_index, prepare_public_audio
    root, config = load_project(project_path)
    if not isinstance(source, dict):
        raise ValueError("Additional material must describe one public corpus folder")
    source = dict(source)
    dataset, pool = source.get("dataset", "manifest"), source.get("pool")
    if dataset not in ("zeroth", "manifest", "audioset", "fma", "mit_rir", "mls", "yodas2"):
        raise ValueError("Choose an existing public-data adapter")
    if pool not in ("generic_negative", "korean_negative", "noise", "rir"):
        raise ValueError("Additional pool must be generic_negative, korean_negative, noise or rir")
    if noise_material_policy not in (None, "official_example", "reuse_negative_waveforms"):
        raise ValueError("Unsupported training-noise material policy")
    reuse = noise_material_policy == "reuse_negative_waveforms"
    if reuse and pool == "noise":
        raise ValueError("Reused training noise must be supplied as raw generic_negative or korean_negative audio, not a separate noise source")
    if dataset == "mit_rir" and pool != "rir":
        raise ValueError("The MIT RIR adapter belongs to the rir pool; use a provider manifest for another RIR corpus")
    workspace = Path(workspace).expanduser().resolve()
    directory_value = Path(source.get("directory", "")).expanduser()
    directory = resolved(workspace, directory_value)
    if not source.get("directory") or not directory.is_dir():
        raise ValueError("Select an existing directory of public source audio")
    if any(p.lower() == ".codex" for p in directory.parts):
        raise ValueError("Additional learning material must not be stored inside .codex")
    for key in ("metadata", "ontology", "room_metadata"):
        if source.get(key):
            source[key] = str(resolved(workspace, Path(source[key]).expanduser()))
    metadata = source.get("metadata")
    if dataset in ("manifest", "audioset", "fma", "mls", "yodas2") and not metadata:
        raise ValueError("This adapter requires the original provider metadata file")
    if dataset == "audioset" and not source.get("ontology"):
        raise ValueError("AudioSet also requires its original ontology JSON")
    hours = source.get("hours")
    if pool != "rir" and (type(hours) not in (int, float) or not __import__("math").isfinite(hours) or hours <= 0):
        raise ValueError("Additional raw audio needs a positive finite preparation-hour quota")
    # These provider-known defaults remain the corpus's actual public terms;
    # arbitrary folder contents cannot be labelled as a verified archive.
    if dataset == "zeroth":
        source.setdefault("source_url", "https://www.openslr.org/40/")
        source.setdefault("license", dataset_catalog()["assets"]["zeroth"]["license"])
    request = {**source, "directory": str(directory), "wakeword": config["wakeword"]}
    if reuse:
        request["noise_material_policy"] = noise_material_policy
    key = identity(request)[:20]
    stage = root / ".studio" / "materials" / ("additional_" + key)
    index = stage.with_name(stage.name + "_source_index.jsonl")
    create_source_index(directory, output=index, dataset=dataset, metadata=metadata,
                        ontology=source.get("ontology"), room_metadata=source.get("room_metadata"),
                        dataset_id=source.get("dataset_id"), source_url=source.get("source_url"),
                        license=source.get("license"))
    record = prepare_public_audio(project_path, input_directory=directory, output=stage,
                    source_index=index, target_hours=None if pool == "rir" else hours,
                    validation_fraction=source.get("validation_fraction", .1),
                    dataset_id=source.get("dataset_id"), source_url=source.get("source_url"),
                    license=source.get("license"), category=source.get("category"),
                    category_basis=source.get("category_basis"), language=source.get("language", "unknown"),
                    speech=source.get("speech"), representatives_per_stratum=2)
    descriptor = {"id": "materials_additional_" + key, "kind": "content",
        "queue": record["artifacts"]["representative_queue"]["path"],
        "pool_manifest": record["artifacts"]["source_manifest"]["path"],
        "preparation": record["preparation"]["path"],
        "pools": [pool, "noise"] if reuse and pool in ("generic_negative", "korean_negative") else [pool],
        "label": "추가 공개 자료 · " + pool}
    return {"kind": "studio_additional_material_preparation_v1", "source": request,
            "review_queue": descriptor, "preparation": record["preparation"],
            "target_met": record["target_met"], "train_hours": record["train_hours"],
            "validation_reserve": record["artifacts"]["validation_reserve"],
            "human_listened_count": 0, "training_admitted": False, "Final_accessed": False}


def _reused_noise_shortages(root, config, receipts):
    """Check approved raw reuse without rewriting old single-pool admissions."""
    shortages, categories = [], set()
    for receipt in receipts:
        path = bound_file(root, receipt) if isinstance(receipt, dict) else resolved(root, receipt)
        record = read_json(path)
        negatives = []
        for pool in ("generic_negative", "korean_negative"):
            binding = record.get("source_manifests", {}).get(pool, record.get("manifests", {}).get(pool))
            if binding:
                negatives.extend(read_rows(bound_file(root, binding)))
        binding = record.get("source_manifests", {}).get("noise", record.get("manifests", {}).get("noise"))
        noise = read_rows(bound_file(root, binding)) if binding else []
        key = lambda row: (row["sha256"], row.get("start_s", 0), row.get("end_s", row.get("duration_s", 0)))
        negative_keys, noise_keys = {key(row) for row in negatives}, {key(row) for row in noise}
        if negative_keys != noise_keys:
            shortages.append("비호출 원본 재사용에는 같은 Train 음원을 비호출·noise 양쪽에 승인한 새 검수 receipt가 필요합니다. 기존 단일 용도 승인은 변경하지 않습니다.")
        if any(Path(row["path"]).suffix.lower() not in _AUDIO for row in noise):
            raise ValueError("Training noise reuse requires waveform audio, not precomputed features")
        categories.update(row["category"] for row in noise)
    missing = sorted(set(config["augmentation"]["category_weights"]) - categories)
    if missing:
        shortages.append("재사용할 학습용 비호출 원본에 소음 종류가 부족합니다: " + ", ".join(missing))
    return list(dict.fromkeys(shortages))


def _automatic_admission_coverage(root, prepared, receipts):
    """Do not let one oversized shard replace another source's target."""
    from .data import coverage_interval, union_hours
    components = prepared.get("automatic_source_plan", {}).get("components", [])
    targets = {row["provider"]: row["target_hours"] for row in components}
    pools = {row["provider"]: row["pool"] for row in components}
    known = {digest(resolved(root, queue["preparation"])): queue["automatic_component"]
             for queue in prepared.get("review_queues", []) if queue.get("automatic_component")}
    intervals, seen = {key: {} for key in targets}, {key: set() for key in targets}
    for binding in receipts:
        path = bound_file(root, binding) if isinstance(binding, dict) else resolved(root, binding)
        record = read_json(path)
        component = known.get(record.get("request", {}).get("preparation", {}).get("sha256"))
        if component not in targets:
            continue
        manifest = record.get("source_manifests", {}).get(pools[component])
        if not manifest:
            continue
        for row in iter_rows(bound_file(root, manifest)):
            start, end = row.get("start_s", 0), row.get("end_s", row.get("duration_s", 0))
            exact = (row["sha256"], start, end)
            if exact in seen[component]:
                continue
            seen[component].add(exact)
            source, interval = coverage_interval(row, start, end)
            intervals[component].setdefault(source, []).append(interval)
    hours = {key: union_hours(values) for key, values in intervals.items()}
    shortages = [f"{key}: reviewed unique Train {hours[key]:g}h is below selected source target {target:g}h"
                 for key, target in targets.items() if hours[key] + 1e-9 < target]
    return {"targets": targets, "admitted_unique_hours": hours, "shortages": shortages,
            "target_met": not shortages}


def admit_materials(project_path, prepared, answers_by_preparation=None, *, admissions=()):
    """Consume actual saved review answers and connect their exact admitted pools."""
    from .preparation import apply_prepared_review, connect_materials
    root, config = load_project(project_path)
    strict_common = prepared.get("recipe_version") == 6
    if strict_common:
        if (prepared.get("evaluation_source_scope") != "common_30h_only"
                or prepared.get("evaluation_materials_policy", {}).get("evaluation_materials_recipe") != 6):
            raise ValueError("Common evaluation preparation pool scope changed")
        if ((root / "final_lock.json").exists() or any((root / "features").rglob("contract.json"))
                or any((root / "runs").rglob("contract.json"))):
            raise ValueError("Existing evaluation/training inputs are frozen; use a new experiment for the unified common corpus")
    answers = answers_by_preparation or {}
    receipts, all_pools, evaluation_rows, evaluation_admissions = [], set(), [], []
    for queue in prepared.get("review_queues", []):
        answer = answers.get(queue["preparation"], answers.get(queue["id"]))
        if answer is None:
            raise ValueError("Saved representative answers are required for " + queue["id"])
        if isinstance(answer, dict):
            answer = bound_file(root, answer)
        if queue.get("evaluation_only"):
            from .studio_evaluation_data import admit_evaluation_review
            if strict_common:
                evaluation_preparation = read_json(resolved(root, queue["preparation"]))
                if (evaluation_preparation.get("recipe_version") != 6
                        or evaluation_preparation.get("evaluation_source_scope") != "common_30h_only"
                        or queue.get("evaluation_source_scope") != "common_30h_only"):
                    raise ValueError("Legacy evaluation admissions cannot fill the unified common corpus")
            evaluation = admit_evaluation_review(project_path, queue, answer, prepared)
            evaluation_rows.extend(evaluation["rows"])
            evaluation_admissions.append(evaluation["admission"])
            continue
        record = apply_prepared_review(project_path, preparation=queue["preparation"], answers=answer,
                                       pools=queue["pools"], connect=False)
        receipts.append(record["receipt"])
        all_pools.update(queue["pools"])
    receipts.extend(admissions)
    for receipt in admissions:
        path = bound_file(root, receipt) if isinstance(receipt, dict) else resolved(root, receipt)
        all_pools.update(read_json(path).get("manifests", {}))
    if not receipts:
        raise ValueError("No reviewed material admissions are available")
    # Recompute real source duration after merging additional admissions instead
    # of preserving a preset's now-obsolete planning shortage.  Generic's
    # publisher-nominal hours stay distinct from waveform unique hours.
    from .data import coverage_interval, union_hours
    durations, raw_generic_bindings = {}, []
    for pool in ("generic_negative", "korean_negative"):
        intervals, seen_intervals = {}, set()
        for receipt in receipts:
            record_path = bound_file(root, receipt) if isinstance(receipt, dict) else resolved(root, receipt)
            record = read_json(record_path)
            if record.get("kind") != "prepared_public_audio_admission_v1" or record.get("state") != "accepted":
                raise ValueError("Additional materials require an authentic accepted preparation receipt")
            binding = record.get("source_manifests", {}).get(pool, record.get("manifests", {}).get(pool))
            if binding:
                if pool == "generic_negative":
                    raw_generic_bindings.append(binding)
                for row in read_rows(bound_file(root, binding)):
                    start, end = row.get("start_s", 0), row.get("end_s", row.get("duration_s", 0))
                    exact = (row["sha256"], start, end)
                    if exact in seen_intervals:
                        continue
                    seen_intervals.add(exact)
                    source, interval = coverage_interval(row, start, end)
                    intervals.setdefault(source, []).append(interval)
        durations[pool] = union_hours(intervals)
    generic = prepared.get("generic_feature_import") or {}
    nominal = generic.get("publisher_declared_hours", 0) if generic.get("imported") else 0
    shortages = []
    generic_target = config["targets"]["generic_hours"]
    korean_target = config["targets"]["korean_hours"]
    # Original overlap with the precomputed file cannot be established, so
    # adding raw sources to the legacy array cannot certify a larger pool.
    raw_generic = (durations["generic_negative"] + 1e-9 >= generic_target
                   and (not generic.get("imported") or generic_target > nominal))
    generic_available = durations["generic_negative"] if raw_generic or not generic.get("imported") else nominal
    if (generic or "generic_negative" in all_pools) and generic_available + 1e-9 < generic_target:
        shortages.append(f"Active Generic feature cache covers {generic_available:g}h, below selected {generic_target:g}h; additional raw Generic admissions require a separately rebuilt feature cache, not addition to the legacy 2,000h file")
    if "korean_negative" in all_pools and durations["korean_negative"] + 1e-9 < korean_target:
        shortages.append(f"Korean unique admitted Train duration {durations['korean_negative']:g}h is below selected {korean_target:g}h")
    if prepared.get("noise_material_policy") == "reuse_negative_waveforms":
        shortages.extend(_reused_noise_shortages(root, config, receipts))
    automatic_coverage = None
    if prepared.get("material_acquisition_policy") in ("automatic_v1", "automatic_v2", "automatic_v3"):
        automatic_coverage = _automatic_admission_coverage(root, prepared, receipts)
        shortages.extend(automatic_coverage["shortages"])
    material_coverage = duration_budget = None
    if prepared.get("recipe_version") in (5, 6):
        from .evaluation_materials import evaluation_material_coverage
        if strict_common:
            from .studio_evaluation_data import assert_common_source_scope
            assert_common_source_scope(evaluation_rows)
        material_coverage = evaluation_material_coverage((r for r in evaluation_rows if r["category"] != "rir"),
            **({"strict_common_pool": True} if strict_common else {}))
        shortages.extend(material_coverage["shortages"])
        if strict_common:
            from .negative_streams import negative_scoring_budget
            duration_budget = negative_scoring_budget([r for r in evaluation_rows if r["category"] != "rir"],
                warmup_s=config["runtime"]["warmup_s"], frame_samples=config["runtime"]["frame_samples"])
            if duration_budget["eligible_seconds"] + .001 < 30 * 3600:
                shortages.append("공통 평가의 실제 채점 예정 시간이 30시간보다 짧습니다. 초기화 여유분의 검수 결과를 확인하세요.")
        admitted_rirs = [r for r in evaluation_rows if r["category"] == "rir"]
        if len(admitted_rirs) < 80 or len({r["canonical_room_id"] for r in admitted_rirs}) < 20:
            shortages.append("공통 평가 잔향 자료 부족: 대표 검수에서 폐기된 묶음을 확인하세요.")
    # Do not freeze partial pool manifests: the next resume may supply another
    # authentic admission and merge once. Existing experiments stay immutable.
    if shortages:
        return {"state": "awaiting_additional_materials", "admissions": receipts, "pools_ready": [],
                "shortages": shortages, "admitted_unique_hours": durations,
                "evaluation_material_coverage": material_coverage,
                **({"automatic_source_admission": automatic_coverage} if automatic_coverage is not None else {}),
                **({"negative_duration_budget": duration_budget} if strict_common else {}),
                "generic_publisher_nominal_hours": nominal, "Final_accessed": False}
    connection = connect_materials(project_path, admissions=receipts, pools=sorted(all_pools))
    # All source reserves remain separate, including added datasets.  They
    # are never promoted to Train merely to meet a selected time target.
    reserves, seen = [], set()
    reserve_bindings = []
    evaluation_ids = {r.get(key) for r in evaluation_rows if r["category"] != "rir"
                      for key in ("canonical_source_id", "canonical_recording_id") if r.get(key)}
    for receipt in (() if strict_common else receipts):
        record_path = bound_file(root, receipt) if isinstance(receipt, dict) else resolved(root, receipt)
        record = read_json(record_path)
        binding = record["evaluation_reserve"]
        reserve_bindings.append(binding)
        for row in read_rows(bound_file(root, binding)):
            if any(row.get(key) in evaluation_ids for key in ("canonical_source_id", "canonical_recording_id") if row.get(key)):
                continue  # The new fixed source interval replaces a duplicate legacy reserve, never adds hours.
            if evaluation_rows and row.get("canonical_source_id") in ("freesound:102956", "freesound:13613"):
                continue  # Retain v1 files/receipts; v2 uses its independent varied set.
            key = (row["sha256"], row.get("start_s", 0), row.get("end_s", row["duration_s"]))
            if row["category"] != "rir" and key not in seen:
                if row["category"] == "competing_speech" and row.get("language") in ("Korean", "ko", "kor"):
                    row = {**row, "eval_type": "korean_read_speech" if str(row.get("canonical_source_id", "")).startswith("zeroth:") else "korean_speech"}
                reserves.append(row)
                seen.add(key)
    rir_rows = [r for r in evaluation_rows if r["category"] == "rir"]
    for row in evaluation_rows:
        if row["category"] == "rir":
            continue
        key = (row["sha256"], row.get("start_s", 0), row.get("end_s", row["duration_s"]))
        if key not in seen:
            reserves.append(row)
            seen.add(key)
    stage = root / ".studio" / "materials"
    reserve_key = identity(sorted(reserve_bindings + evaluation_admissions, key=lambda x: x["path"]))[:16]
    mixture_rows = ([r for r in reserves if r.get("positive_mix_eligible") is not False]
                    if strict_common else reserves)
    validation = _write_rows(stage / ("noise_validation_" + reserve_key + ".jsonl"), mixture_rows)
    negative_rows = [{**r, "kind": "negative", "calls": []} for r in reserves]
    negative_streams = None
    if prepared.get("recipe_version") in (5, 6):
        from .negative_streams import pack_negative_streams
        negative_streams = pack_negative_streams(project_path, negative_rows,
                                                 stage / ("negative_streams_" + reserve_key),
                                                 **({"strict_common_pool": True} if strict_common else {}))
        negative_rows = negative_streams["rows"]
    negative = _write_rows(stage / ("negative_validation_" + reserve_key + ".jsonl"),
                           negative_rows)
    destination = resolved(root, config["manifests"]["noise_validation"])
    config_changed = False
    if destination.exists() and destination.stat().st_size and digest(destination) != validation["sha256"]:
        if not evaluation_rows or any((root / "features").rglob("contract.json")) or any((root / "runs").rglob("contract.json")):
            raise ValueError("Existing evaluation/training inputs are frozen; use a new experiment for the revised noise recipe")
        relative = f"data/materials/validation_noise_v{prepared.get('recipe_version', 2)}_" + reserve_key + ".jsonl"
        config["manifests"]["noise_validation"] = relative
        destination = resolved(root, relative)
        config_changed = True
    _write_rows(destination, mixture_rows)
    rir_manifest = None
    if rir_rows:
        relative = f"data/materials/validation_rir_v{prepared.get('recipe_version', 5)}_" + reserve_key + ".jsonl"
        rir_manifest = _write_rows(resolved(root, relative), rir_rows)
        config["manifests"]["rir_validation"] = relative
        config_changed = True
    if raw_generic:
        relative_cache = "features/generic_negative_raw_" + identity(sorted(raw_generic_bindings, key=lambda x: x["path"]))[:16]
        config["pool_features"]["generic_negative"] = relative_cache
        config_changed = True
        generic = {"mode": "raw_generic", "imported": False, "output": str(resolved(root, relative_cache)),
                   "unique_source_hours": durations["generic_negative"], "publisher_declared_hours": None,
                   "previous_legacy_cache_preserved": True}
    if config_changed:
        write_json(project_path, config)
    return {"state": "connected", "connection": connection, "admissions": receipts,
            "pools_ready": sorted(all_pools), "generic_feature_import": generic,
            "negative_validation_manifest": negative,
            "negative_streams": {k: v for k, v in negative_streams.items() if k != "rows"} if negative_streams else None,
            "validation_noise_manifest": validation,
            "validation_rir_manifest": rir_manifest,
            "evaluation_material_coverage": material_coverage,
            **({"automatic_source_admission": automatic_coverage} if automatic_coverage is not None else {}),
            **({"negative_duration_budget": duration_budget} if strict_common else {}),
            "evaluation_noise_admissions": evaluation_admissions,
            "evaluation_noise_recipe_version": prepared.get("recipe_version", 2) if evaluation_rows else 1,
            **({"evaluation_source_scope": "common_30h_only"} if strict_common else {}),
            "admitted_unique_hours": durations, "generic_publisher_nominal_hours": nominal,
            "shortages": shortages, "Final_accessed": False}
