"""Pinned common-only room responses; measured and simulated rooms stay explicit."""
from collections import Counter
from importlib.resources import files
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import zipfile

import numpy as np
import soundfile as sf

from .common import bound_file, digest, identity, iter_rows, read_json, resolved, write_json, writer_lock
from .downloads import download_checked
from .project import load_project, require_standard_common_download
from .storage import project_storage, storage_paths
from .source_identity import partition_keys


def estimate_rt60_t20(values, sample_rate):
    """Broadband Schroeder -5..-25dB T20 extrapolation, not provider RT60.

    No filter-bank or room measurement is implied. R2, sample count and fitting
    bounds are returned so a group is traceable rather than inferred from a name.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("RT60 estimation needs finite nonempty mono impulse data")
    energy = np.cumsum((values * values)[::-1])[::-1]
    if energy[0] <= 1e-16:
        raise ValueError("RT60 estimation refuses a silent RIR")
    db = 10 * np.log10(np.maximum(energy / energy[0], 1e-30))
    mask = (db <= -5) & (db >= -25)
    if np.count_nonzero(mask) < 160:
        raise ValueError("RIR lacks at least ten milliseconds of a usable -5..-25dB decay")
    times, levels = np.flatnonzero(mask) / sample_rate, db[mask]
    slope, intercept = np.polyfit(times, levels, 1)
    predicted = slope * times + intercept
    residual = float(np.sum((levels - predicted) ** 2))
    variance = float(np.sum((levels - levels.mean()) ** 2))
    r2 = 1 - residual / variance if variance else 0
    rt60 = -60 / float(slope) if slope < 0 else math.inf
    if not math.isfinite(rt60) or not .1 <= rt60 <= 3 or r2 < .95:
        raise ValueError("RIR broadband T20 estimate lacks a stable descending fit")
    return {"rt60_s": rt60, "method": "broadband_schroeder_T20_minus5_to_minus25dB_v1",
            "fit_r_squared": r2, "fit_start_s": float(times[0]), "fit_end_s": float(times[-1]),
            "fit_samples": len(times), "slope_db_per_s": float(slope),
            "scope": "Computed broadband extrapolation of the provided finite RIR, not a measured room RT60 or octave-band estimate"}


def common_rir_catalog():
    return read_json(files("korean_openwakeword") / "resources/evaluation_rirs_v1.json")


def common_rir_preview():
    catalog = common_rir_catalog()
    return {"id": catalog["id"], "name": "공통 평가용 공간 울림", "file_count": len(catalog["assets"]),
        "size_bytes": catalog["archive"]["size_bytes"], "download_size_scope": "First download is the whole pinned archive; only selected RIRs are extracted",
        "room_count": len({row["canonical_room_id"] for row in catalog["assets"]}),
        "simulated_room_count": len({row["canonical_room_id"] for row in catalog["assets"] if row["acoustic_source_type"] == "simulated"}),
        "type_labels": ["일반 실내 울림", "강한 실내 울림"], "source_url": catalog["source_url"],
        "license": catalog["license"], "scope": catalog["scope"]}


def _assert_source_separation(root, config, catalog):
    known = {fact for row in catalog["assets"] for fact in partition_keys(row)}
    for pool, manifest in config["manifests"].items():
        for row in iter_rows(resolved(root, manifest)):
            if row.get("split") not in ("train", "final"):
                continue
            overlap = known.intersection(partition_keys(row))
            if overlap:
                raise ValueError("Common evaluation RIR overlaps " + row["split"] + " sources/rooms in " + pool + ": " + str(sorted(overlap)[:3]))


def _resumable_output(output, catalog):
    binding = {"catalog_sha256": identity(catalog), "kind": "common_rir_preparation_in_progress_v1"}
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"audio", "preparation.json", "rir_validation.jsonl", "receipt.json", ".writer.lock"}
    if any(path.name not in allowed for path in output.iterdir()):
        raise ValueError("Common RIR output has unrelated files; use a dedicated preparation directory")
    journal = output / "preparation.json"
    if journal.exists() and read_json(journal) != binding:
        raise ValueError("Interrupted common RIR preparation uses a different immutable catalog")
    audio = output / "audio"
    expected = {row["id"] + ".wav": row["sha256"] for row in catalog["assets"]}
    if audio.exists():
        for path in audio.iterdir():
            if path.is_file() and path.name.endswith(".wav.part") and path.name[:-5] in expected and path.stat().st_size <= 2 * 1024 * 1024:
                continue  # Not an admitted WAV; replace from a verified member on resume.
            if not path.is_file() or path.name not in expected or digest(path) != expected[path.name]:
                raise ValueError("Interrupted common RIR output has an unexpected or altered audio file")
    if not journal.exists():
        write_json(journal, binding)


@project_storage
def prepare_common_rirs(project, output, archive=None, log=None, cache_directory=None, acquisition_policy=None):
    """Prepare common RIRs only; never alter Train or reserve/unlock Final."""
    root, config = load_project(project)
    require_standard_common_download(acquisition_policy)
    require_standard_common_download(config.get("common_download_policy"))
    catalog = common_rir_catalog()
    _assert_source_separation(root, config, catalog)
    output = Path(output).resolve()
    with writer_lock(output):
        return _prepare_common_rirs_locked(root, catalog, output, archive, log, cache_directory)


def _prepare_common_rirs_locked(root, catalog, output, archive, log, cache_directory):
    if (output / "receipt.json").is_file():
        receipt = read_json(output / "receipt.json")
        require_standard_common_download(receipt.get("acquisition_policy"))
        if receipt.get("catalog_sha256") != identity(catalog):
            raise ValueError("Existing common RIR preparation has another immutable catalog")
        for row in receipt["source_rows"]:
            bound_file(root, row)
        return receipt
    _resumable_output(output, catalog)
    pin = catalog["archive"]
    downloaded_and_verified = archive is None
    if downloaded_and_verified:
        directory = Path(cache_directory).resolve() if cache_directory is not None else storage_paths(root)["root"] / "downloads" / "common_rir"
        if any(part.lower() == ".codex" for part in directory.parts):
            raise ValueError("Bulk RIR cache must not be stored inside .codex")
        archive = directory / pin["filename"]
        if log:
            log("공통 잔향 원본 archive 준비·무결성 확인")
        from .material_downloads import borrowed_download
        if borrowed_download(archive, "sha256", pin["sha256"], pin["size_bytes"]) is None:
            download_checked(pin["source_url"], archive, pin["sha256"], pin["size_bytes"])
    archive = Path(archive).resolve()
    if archive.stat().st_size != pin["size_bytes"] or (not downloaded_and_verified and digest(archive) != pin["sha256"]):
        raise ValueError("Common RIR archive differs from its pinned size/SHA256")
    rows = []
    with zipfile.ZipFile(archive) as source:
        for asset in catalog["assets"]:
            name = PurePosixPath(asset["archive_member"])
            if name.is_absolute() or ".." in name.parts or name.suffix.lower() != ".wav":
                raise ValueError("Unsafe RIR archive member")
            info = source.getinfo(str(name))
            if info.file_size != asset["size_bytes"] or info.file_size > 2 * 1024 * 1024:
                raise ValueError("RIR archive member exceeds its exact bounded size")
            payload = source.read(info)
            if hashlib.sha256(payload).hexdigest() != asset["sha256"]:
                raise ValueError("Common RIR member hash changed")
            destination = output / "audio" / (asset["id"] + ".wav")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and digest(destination) != asset["sha256"]:
                raise ValueError("Prepared common RIR file changed")
            if not destination.exists():
                partial = destination.with_name(destination.name + ".part")
                partial.write_bytes(payload)
                partial.replace(destination)
            values, rate = sf.read(destination, dtype="float64", always_2d=True)
            if values.shape[1] != 1 or rate != 16000:
                raise ValueError("Pinned common RIRs must already be mono 16kHz")
            estimated = estimate_rt60_t20(values[:, 0], rate)
            if abs(estimated["rt60_s"] - asset["rt60_s"]) > 1e-6:
                raise ValueError("Pinned common RIR RT60 estimate differs from decoded source")
            rows.append({**asset, "sample_id": asset["id"], "path": str(destination), "split": "validation",
                "category": "rir", "speech": False, "provenance": "public_dataset",
                "start_s": 0, "end_s": len(values) / rate, "technical_pass": True,
                "source_id": asset["canonical_source_id"], "source_url": catalog["source_url"],
                "license": catalog["license"], "dataset_id": catalog["id"],
                "individual_human_listening": False, "training_admitted": False, "content_approved": False})
    manifest = output / "rir_validation.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    receipt = {"kind": "common_rir_preparation_v1", "catalog_sha256": identity(catalog),
        "archive": {"path": str(archive), "sha256": pin["sha256"], "bytes": pin["size_bytes"]},
        "manifest": {"path": str(manifest), "sha256": digest(manifest)}, "source_rows": rows,
        "coverage": {"rir_sources": len(rows), "room_count": len({r["canonical_room_id"] for r in rows}),
                     "groups": dict(Counter(row["reverb_group"] for row in rows)),
                     "acoustic_source_types": dict(Counter(row["acoustic_source_type"] for row in rows))},
        "human_listening": 0, "content_approved": False, "Final_unlocked": False,
        "scope": catalog["scope"]}
    write_json(output / "receipt.json", receipt)
    return receipt


def render_rir_review_previews(project, rir_rows, output, call_row=None, count=6):
    """Render a few actual filters on an existing approved synthetic call/chirp."""
    from scipy.signal import fftconvolve
    from .audio import pcm16, rms
    from .data import seek_audio
    root, _ = load_project(project)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding = {"rir_rows": identity(rir_rows), "call_row": identity(call_row), "count": count}
    journal = output / "preview_recipe.json"
    if journal.is_file() and read_json(journal) != binding:
        raise ValueError("RIR review previews use another immutable source/selection recipe")
    if not journal.exists():
        if any(output.iterdir()):
            raise ValueError("Use an empty dedicated RIR preview directory")
        write_json(journal, binding)
    receipt = output / "previews.json"
    if receipt.is_file():
        rows = read_json(receipt)
        for row in rows:
            bound_file(root, row)
        return rows
    selected = []
    active_groups = {row["reverb_group"] for row in rir_rows}
    for group in ("moderate", "strong"):
        candidates = sorted((row for row in rir_rows if row["reverb_group"] == group),
                            key=lambda row: identity([row["canonical_room_id"], row["sha256"]]))
        rooms = set()
        for row in candidates:
            if row["canonical_room_id"] in rooms:
                continue
            rooms.add(row["canonical_room_id"]); selected.append(row)
            if len(rooms) >= math.ceil(count / max(1, len(active_groups))):
                break
    if call_row is not None:
        if call_row.get("provenance") != "qwen_tts" or call_row.get("human_reference_used") is not False:
            raise ValueError("RIR review call must be an existing Qwen-only call, never a human recording")
        bound_file(root, call_row)
        a, b = call_row["call_start_s"], call_row["call_end_s"]
        dry, _ = seek_audio(root, {**call_row, "start_s": a, "end_s": b}, b - a)
        description = "검수한 기존 합성 호출어에 공간 울림만 적용"
    else:
        from scipy.signal import chirp
        time = np.arange(8000) / 16000
        dry = .15 * chirp(time, 250, .5, 3000) * np.sin(np.pi * time / .5) ** 2
        description = "고정 합성 시험음에 공간 울림만 적용 · 호출어 발음 검수가 아닙니다"
    dry = np.r_[np.zeros(3200), dry, np.zeros(3200)]
    rows = []
    for index, source in enumerate(selected[:count]):
        bound_file(root, source)
        info = sf.info(source["path"])
        impulse, _ = seek_audio(root, source, info.duration)
        wet = fftconvolve(dry, impulse / np.sqrt(np.sum(impulse ** 2)))
        gain = min(.95 / max(np.max(np.abs(wet)), 1e-9), .1 / max(rms(wet), 1e-9))
        path = output / f"rir_review_{index+1:02d}_{source['reverb_group']}.wav"
        import io
        payload = io.BytesIO()
        sf.write(payload, pcm16(wet * gain), 16000, subtype="PCM_16", format="WAV")
        expected = payload.getvalue()
        if path.exists() and digest(path) != hashlib.sha256(expected).hexdigest():
            raise ValueError("An immutable RIR review preview changed")
        if not path.exists():
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(expected)
            partial.replace(path)
        rows.append({"sample_id": source["sample_id"], "path": str(path), "sha256": digest(path),
            "reverb_group": source["reverb_group"], "canonical_room_id": source["canonical_room_id"],
            "source_sha256": source["sha256"], "description": description,
            "source_type": source["acoustic_source_type"], "call_generation_performed": False,
            "individual_human_listening": False})
    write_json(receipt, rows)
    return rows
