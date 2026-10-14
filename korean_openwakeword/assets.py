"""Pinned upstream downloads; sample inference needs no model download."""
from pathlib import Path
import shutil
import requests

from .common import digest, now, write_json
from .frontend import PINNED_ASSETS
from .runtime import sample_bundle, verify_bundle


def prepare_frontend(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, sha in PINNED_ASSETS.items():
        path = output / name
        if path.is_file() and digest(path) == sha:
            continue
        # Bundled bytes are the exact upstream pinned assets; avoid a redundant download.
        bundled = sample_bundle() / "frontend" / name
        if bundled.is_file() and digest(bundled) == sha:
            shutil.copyfile(bundled, path)
        else:
            url = f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{name}"
            temporary = path.with_suffix(".part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
            if digest(temporary) != sha:
                raise ValueError("Upstream model checksum mismatch")
            temporary.replace(path)
    receipt = {"completed_at": now(), "sha256": PINNED_ASSETS, "release": "v0.5.1"}
    write_json(output / "receipt.json", receipt)
    return receipt


def export_sample(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Sample destination must be empty")
    shutil.copytree(sample_bundle(), output, dirs_exist_ok=True)
    return verify_bundle(output)


GENERIC_SHA = "721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f"
GENERIC_REVISION = "985bf1b47e7f19c07741af82bfe32d5a9dc56096"
GENERIC_FILE = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
GENERIC_URL = f"https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/{GENERIC_REVISION}/{GENERIC_FILE}"


def download_generic(output):
    """Resumable ~17.28GB file. Nominal publisher hours, not verified unique hours."""
    from .downloads import download_checked
    output = Path(output).resolve()
    record = download_checked(GENERIC_URL, output, GENERIC_SHA, 17280000128)
    record.update({
        "publisher_declared_hours": 2000, "unique_source_hours": None, "upstream_revision": GENERIC_REVISION,
        "source_url": GENERIC_URL})
    write_json(output.with_suffix(".receipt.json"), record)
    return record


def import_generic(path, output, frontend_identity):
    """Import only the publisher's pinned precomputed Generic feature file.

    All rows are scanned; only the pinned full file is supported. No fabricated
    source IDs, waveform augmentation or train/Final disjointness are asserted.
    """
    import numpy as np
    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new Generic metadata/cache namespace")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (5625000, 16, 96) or array.dtype != np.float16 or digest(path) != GENERIC_SHA:
        raise ValueError("Legacy bridge only supports the complete checksum-pinned publisher array")
    for i in range(0, len(array), 4096):
        block = array[i:i+4096]
        if not np.isfinite(block).all() or np.any(np.ptp(block.astype(np.float32), axis=(1, 2)) == 0):
            raise ValueError("Nonfinite/degenerate Generic feature row")
    output.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": 1, "role": "generic_negative", "split": "train", "feature_path": str(path),
        "feature_rows": len(array), "original_count": None, "rounds": 1, "imported_precomputed": True,
        "feature_sha256": GENERIC_SHA, "frontend_id": "publisher_legacy_speech_embedding_1",
        "legacy_generic_bridge": True, "canonical_frontend_assets": frontend_identity["model_sha256"],
        "publisher_declared_hours": 2000, "unique_source_hours": None, "waveform_augmentation_applied": False,
        "split_audit_passed": True, "split_audit_scope": "publisher train partition; row source IDs unavailable",
        "quality_review": "accepted", "smoke_test_only": False,
        "provenance": "publisher_supported_legacy_embedding_space",
        "source_url": GENERIC_URL, "scanned_rows": len(array), "created_at": now(),
        "limitations": ["Historical extractor and coordinate identity not proven.",
                         "Complete train/evaluation overlap cannot be audited without row sources.",
                         "2000h is publisher nominal, not verified unique source duration."]}
    write_json(output / "metadata.json", meta)
    return meta
