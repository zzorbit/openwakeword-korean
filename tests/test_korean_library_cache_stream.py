"""Small CPU fixtures prove stream verification parity without scanning Generic.

The reference verify_cache below is frozen from canonical features.py SHA
2633af416d421e41bff36874b87a1220b6785008693bcab30664ae01db6c1d8a.
"""
from pathlib import Path
import struct

import numpy as np
import pytest

from korean_openwakeword import features
from korean_openwakeword.common import digest, read_json, read_rows, resolved, write_json


def legacy_verify_cache(directory):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    feature_path = resolved(directory, meta.get("feature_path", "features.npy"))
    if digest(feature_path) != meta["feature_sha256"]:
        raise ValueError("Feature payload hash changed")
    values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (meta["feature_rows"], 16, 96) or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Invalid feature shape/dtype")
    for i in range(0, len(values), 4096):
        if not np.isfinite(values[i:i+4096]).all():
            raise ValueError("Nonfinite feature values")
    if meta.get("imported_precomputed"):
        from korean_openwakeword.assets import GENERIC_SHA
        if meta.get("role") != "generic_negative" or meta["feature_sha256"] != GENERIC_SHA or meta["feature_rows"] != 5625000 or values.dtype != np.float16:
            raise ValueError("Unsupported imported feature cache")
        return meta
    if digest(directory / "augmentation.jsonl") != meta["journal_sha256"] or digest(directory / "contract.json") != meta["contract_sha256"]:
        raise ValueError("Feature journal/contract changed")
    rows = read_rows(directory / "augmentation.jsonl")
    if len(rows) != len(values) or meta["feature_rows"] != meta["original_count"] * meta["rounds"]:
        raise ValueError("Actual output is not N×rounds")
    per_original = {}
    for i, row in enumerate(rows):
        if row["feature_row"] != i:
            raise ValueError("Feature order changed")
        per_original.setdefault(row["sample_id"], []).append((row["round_id"], row["seed"], row["pcm_sha256"]))
    for entries in per_original.values():
        if [r[0] for r in entries] != list(range(meta["rounds"])) or len({r[1] for r in entries}) != meta["rounds"]:
            raise ValueError("Missing rounds or repeated augmentation seeds")
        if meta["rounds"] > 1 and len({r[2] for r in entries}) != meta["rounds"]:
            raise ValueError("Repeated PCM across rounds; inspect augmentation diversity")
    return meta


def cache(tmp_path, values, version=(1, 0), trailer=b""):
    directory = tmp_path / "cache"
    directory.mkdir()
    path = directory / "features.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, values, version=version, allow_pickle=True)
        stream.write(trailer)
    count = values.shape[0]
    rows = [{"feature_row": i, "sample_id": f"source{i}", "round_id": 0,
             "seed": 30 + i, "pcm_sha256": f"{i + 1:064x}"} for i in range(count)]
    (directory / "augmentation.jsonl").write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(directory / "contract.json", {"fixture": "CPU-only small stream parity"})
    meta = {"feature_rows": count, "original_count": count, "rounds": 1,
            "feature_sha256": digest(path), "journal_sha256": digest(directory / "augmentation.jsonl"),
            "contract_sha256": digest(directory / "contract.json")}
    write_json(directory / "metadata.json", meta)
    return directory, path, meta


def update_payload_sha(directory, path):
    meta = read_json(directory / "metadata.json")
    meta["feature_sha256"] = digest(path)
    write_json(directory / "metadata.json", meta)


def assert_rejected_by_both(directory, message=None):
    with pytest.raises((ValueError, OSError, EOFError)) as original:
        legacy_verify_cache(directory)
    with pytest.raises((ValueError, OSError, EOFError)) as fixed:
        features.verify_cache(directory)
    assert type(original.value) is type(fixed.value)
    if message is not None:
        assert message in str(original.value) and message in str(fixed.value)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("dtype", ["<f2", "<f4", "<f8", ">f2", ">f4"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("block_bytes", [7, 8 * 1024 * 1024])
def test_all_npy_headers_dtypes_orders_and_partial_blocks_return_identical_metadata(tmp_path, monkeypatch, version, dtype, order, block_bytes):
    values = np.array(np.random.default_rng(8).uniform(-.5, .5, (3, 16, 96)), dtype=dtype, order=order)
    values[0, 0, :4] = [0., -0., np.finfo(dtype).tiny, np.finfo(dtype).max]
    directory, _, meta = cache(tmp_path, values, version)
    monkeypatch.setattr(features, "FEATURE_SCAN_BLOCK_BYTES", block_bytes)
    assert legacy_verify_cache(directory) == features.verify_cache(directory) == meta


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_unaligned_header_and_chunk_boundaries_keep_stored_float_values(tmp_path, monkeypatch, version):
    values = np.arange(3 * 16 * 96, dtype=np.float32).reshape(3, 16, 96) / 1000
    directory, path, _ = cache(tmp_path, values, version)
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 16, 96), }"
    prefix_bytes = 10 if version == (1, 0) else 12
    while (prefix_bytes + len(header) + 1) % 4 == 0:
        header += " "
    raw_header = (header + "\n").encode("utf-8" if version == (3, 0) else "latin1")
    with path.open("wb") as stream:
        stream.write(np.lib.format.magic(*version))
        stream.write(struct.pack("<H" if version == (1, 0) else "<I", len(raw_header)))
        stream.write(raw_header)
        stream.write(values.tobytes())
    update_payload_sha(directory, path)
    monkeypatch.setattr(features, "FEATURE_SCAN_BLOCK_BYTES", 13)
    assert legacy_verify_cache(directory) == features.verify_cache(directory)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, 7, 2304, 4607])
def test_every_nonfinite_position_is_rejected_including_cross_chunk_values(tmp_path, monkeypatch, value, index):
    values = np.ones((3, 16, 96), np.float32)
    values.reshape(-1)[index] = value
    directory, _, _ = cache(tmp_path, values)
    monkeypatch.setattr(features, "FEATURE_SCAN_BLOCK_BYTES", 7)
    assert_rejected_by_both(directory, "Nonfinite feature values")


@pytest.mark.parametrize("corruption", ["header", "payload", "truncation", "trailer"])
def test_any_corrupt_file_byte_preserves_hash_first_rejection(tmp_path, corruption):
    directory, path, _ = cache(tmp_path, np.ones((3, 16, 96), np.float16))
    payload = bytearray(path.read_bytes())
    if corruption == "header":
        payload[0] ^= 1
    elif corruption == "payload":
        payload[-1] ^= 1
    elif corruption == "truncation":
        payload = payload[:-1]
    else:
        payload.extend(b"changed trailer")
    path.write_bytes(payload)
    assert_rejected_by_both(directory, "Feature payload hash changed")


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_current_hash_does_not_make_truncated_array_acceptable(tmp_path, version):
    directory, path, _ = cache(tmp_path, np.ones((3, 16, 96), np.float32), version)
    path.write_bytes(path.read_bytes()[:-1])
    update_payload_sha(directory, path)
    # Windows reports an oversized read-only mapping as WinError 8; Linux
    # reports ValueError. Both implementations must preserve that rejection.
    assert_rejected_by_both(directory)


@pytest.mark.parametrize("dtype", [np.int16, np.complex64, object, [("value", "f4")]])
def test_object_and_nonfloating_dtype_are_still_rejected_without_deserialization(tmp_path, dtype):
    values = np.ones((3, 16, 96), dtype=dtype)
    directory, _, _ = cache(tmp_path, values)
    assert_rejected_by_both(directory, "Python objects" if dtype is object else "Invalid feature shape/dtype")


@pytest.mark.parametrize("shape", [(3, 15, 96), (3, 16, 95), (3, 16, 96, 1)])
def test_shape_validation_is_unchanged(tmp_path, shape):
    directory, _, _ = cache(tmp_path, np.ones(shape, np.float16))
    assert_rejected_by_both(directory, "Invalid feature shape/dtype")


def test_tail_bytes_are_hashed_without_being_treated_as_extra_array_elements(tmp_path, monkeypatch):
    trailer = np.array([np.nan, np.inf], np.float32).tobytes() + b"permitted trailer"
    directory, _, _ = cache(tmp_path, np.ones((3, 16, 96), np.float32), trailer=trailer)
    monkeypatch.setattr(features, "FEATURE_SCAN_BLOCK_BYTES", 7)
    assert legacy_verify_cache(directory) == features.verify_cache(directory)


@pytest.mark.parametrize("case", ["journal_hash", "contract_hash", "row_count", "row_order", "round_count", "repeated_seed", "repeated_pcm"])
def test_journal_contract_and_n_times_rounds_controls_are_unchanged(tmp_path, case):
    directory, _, _ = cache(tmp_path, np.ones((3, 16, 96), np.float32))
    meta = read_json(directory / "metadata.json")
    rows = read_rows(directory / "augmentation.jsonl")
    if case == "journal_hash":
        (directory / "augmentation.jsonl").write_text("[]\n", encoding="utf-8")
    elif case == "contract_hash":
        write_json(directory / "contract.json", {"changed": True})
    elif case == "round_count":
        meta["rounds"] = 3
    else:
        if case == "row_count":
            rows.pop()
        elif case == "row_order":
            rows[0]["feature_row"] = 2
        else:
            meta.update(original_count=1, rounds=3)
            for index, row in enumerate(rows):
                row.update(sample_id="one_source", round_id=index)
            rows[1]["seed" if case == "repeated_seed" else "pcm_sha256"] = rows[0]["seed" if case == "repeated_seed" else "pcm_sha256"]
        (directory / "augmentation.jsonl").write_text(
            "".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
        meta["journal_sha256"] = digest(directory / "augmentation.jsonl")
    write_json(directory / "metadata.json", meta)
    assert_rejected_by_both(directory)


def test_imported_precomputed_gate_remains_strict_on_small_unsupported_pool(tmp_path):
    directory, _, _ = cache(tmp_path, np.ones((3, 16, 96), np.float16))
    meta = read_json(directory / "metadata.json")
    meta.update(imported_precomputed=True, role="generic_negative")
    write_json(directory / "metadata.json", meta)
    assert_rejected_by_both(directory, "Unsupported imported feature cache")


def test_explicit_stream_reads_every_byte_once_and_checks_every_payload_value(tmp_path, monkeypatch):
    values = np.ones((3, 16, 96), np.float32)
    values.reshape(-1)[1] = np.nan
    directory, path, _ = cache(tmp_path, values, version=(3, 0), trailer=b"extra tail")
    original_open, original_finite = Path.open, np.isfinite
    reads, scanned = [], []

    class ReadSpy:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, count=-1):
            start = self.stream.tell()
            block = self.stream.read(count)
            reads.append((start, len(block)))
            return block

    def opened(candidate, mode="r", *args, **kwargs):
        stream = original_open(candidate, mode, *args, **kwargs)
        return ReadSpy(stream) if candidate == path and mode == "rb" else stream

    def finite(block, *args, **kwargs):
        scanned.append(block.size)
        return original_finite(block, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(np, "isfinite", finite)
    monkeypatch.setattr(features, "FEATURE_SCAN_BLOCK_BYTES", 101)
    with pytest.raises(ValueError, match="Nonfinite feature values"):
        features.verify_cache(directory)
    assert sum(length for _, length in reads) == path.stat().st_size
    positive_reads = [(start, length) for start, length in reads if length]
    assert positive_reads[0][0] == 0
    assert all(b_start == a_start + a_length for (a_start, a_length), (b_start, _) in zip(positive_reads, positive_reads[1:]))
    assert max(length for _, length in reads) <= 101
    assert sum(scanned) == values.size, "Later values must still be scanned after the first NaN"
