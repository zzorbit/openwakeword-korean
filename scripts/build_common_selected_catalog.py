"""Maintainer-only: derive exact member pins from verified local public ZIPs.

No network access or source rewriting. Full upstream checksums are verified
before a selected member is read. Runtime selective transfer must not claim
that it verified these whole-archive checksums.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from korean_openwakeword.common import identity, read_json
from korean_openwakeword.evaluation_materials import _split_zip


POLICY = "selected_common_zip_v1"
GAP, MAX_RANGE = 256 * 1024, 16 * 1024**2


def verify_volume(path, pin):
    if path.stat().st_size != pin["size_bytes"]:
        raise ValueError("Local archive size differs: " + pin["id"])
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(block); md5.update(block)
    algorithm = pin.get("checksum_algorithm", "sha256")
    if {"sha256": sha.hexdigest(), "md5": md5.hexdigest()}[algorithm] != pin[algorithm]:
        raise ValueError("Local archive upstream checksum differs: " + pin["id"])
    print("Verified original archive: " + pin["id"], flush=True)
    return {key: pin[key] for key in ("id", "filename", "source_url", "size_bytes", "checksum_algorithm", "md5") if key in pin} | {"sha256": sha.hexdigest()}


def spans(start, end, volumes):
    result, base = [], 0
    for volume in volumes:
        left, right = max(start, base), min(end, base + volume["size_bytes"])
        if left < right:
            result.append({"volume_id": volume["id"], "offset": left - base, "length": right - left})
        base += volume["size_bytes"]
    if sum(value["length"] for value in result) != end - start:
        raise ValueError("Selected span escapes verified source volumes")
    return result


def pin_member(archive, name, volumes, expected=None):
    info = archive.getinfo(name)
    archive.fp.seek(info.header_offset)
    initial = archive.fp.read(30)
    if len(initial) != 30 or initial[:4] != b"PK\x03\x04":
        raise ValueError("Selected local ZIP header changed")
    n, e = struct.unpack_from("<HH", initial, 26)
    header = initial + archive.fp.read(n + e)
    if len(header) != 30 + n + e or info.compress_type not in (0, 8) or info.flag_bits & 1:
        raise ValueError("Unsupported selected ZIP member")
    if len(header) + info.compress_size > MAX_RANGE:
        raise ValueError("Selected member exceeds bounded transport size: " + name)
    compressed, original, crc, count = hashlib.sha256(), hashlib.sha256(), 0, 0
    decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
    remaining = info.compress_size
    while remaining:
        block = archive.fp.read(min(1024**2, remaining))
        if not block:
            raise EOFError("Truncated selected compressed source")
        remaining -= len(block); compressed.update(block)
        decoded = decoder.decompress(block) if decoder else block
        original.update(decoded); crc = zlib.crc32(decoded, crc); count += len(decoded)
    if decoder:
        tail = decoder.flush()
        original.update(tail); crc = zlib.crc32(tail, crc); count += len(tail)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Selected compressed source is not exactly one deflate stream")
    if count != info.file_size or crc != info.CRC or expected and original.hexdigest() != expected:
        raise ValueError("Selected original member differs: " + name)
    start, end = info.header_offset, info.header_offset + len(header) + info.compress_size
    return {"name": name, "size_bytes": info.file_size, "compressed_bytes": info.compress_size,
        "crc32": f"{crc:08x}", "sha256": original.hexdigest(), "compression_method": info.compress_type,
        "flags": info.flag_bits, "header_offset": start, "data_offset": start + len(header),
        "header_bytes": len(header), "header_sha256": hashlib.sha256(header).hexdigest(),
        "compressed_sha256": compressed.hexdigest(), "spans": spans(start, end, volumes)}


def plan(members, volumes):
    ranges = sorted((m["header_offset"], m["data_offset"] + m["compressed_bytes"]) for m in members)
    merged = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= GAP and end - merged[-1][0] <= MAX_RANGE:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    by_volume = {volume["id"]: 0 for volume in volumes}
    requests = 0
    for start, end in merged:
        for item in spans(start, end, volumes):
            by_volume[item["volume_id"]] += item["length"]; requests += 1
    return {"merge_gap_bytes": GAP, "maximum_range_bytes": MAX_RANGE,
        "selected_download_bytes": sum(by_volume.values()), "download_bytes_by_volume": by_volume,
        "range_requests": requests, "selected_member_bytes": sum(m["size_bytes"] for m in members),
        "compressed_member_bytes": sum(m["compressed_bytes"] for m in members)}


def build(downloads, output):
    root = Path(__file__).resolve().parents[1]
    common = read_json(root / "korean_openwakeword/resources/evaluation_materials_v1.json")
    rir = read_json(root / "korean_openwakeword/resources/evaluation_rirs_v1.json")
    pins = {a["id"]: a for a in common["archives"]}
    downloads = Path(downloads).resolve()
    fsd_paths = [downloads / pins[key]["filename"] for key in ("fsd_eval", "fsd_eval_zip")]
    fsd_volumes = [verify_volume(path, pins[key]) for path, key in zip(fsd_paths, ("fsd_eval", "fsd_eval_zip"))]
    rir_pin = {**rir["archive"], "id": "common_rir", "checksum_algorithm": "sha256"}
    rir_path = downloads / "common_rir" / rir_pin["filename"]
    rir_volumes = [verify_volume(rir_path, rir_pin)]
    archive, view = _split_zip(fsd_paths)
    try:
        fsd_members = []
        for index, row in enumerate(common["selections"]["fsd_eval"], 1):
            fsd_members.append(pin_member(archive, row["member"], fsd_volumes))
            if index % 100 == 0:
                print(f"Pinned FSD members: {index}/1861", flush=True)
    finally:
        archive.close(); view.close()
    import zipfile
    with zipfile.ZipFile(rir_path) as archive:
        rir_members = [pin_member(archive, row["archive_member"], rir_volumes, row["sha256"]) for row in rir["assets"]]
    groups = {}
    for key, source, volumes, members in (("fsd_eval", common, fsd_volumes, fsd_members), ("common_rir", rir, rir_volumes, rir_members)):
        groups[key] = {"id": key, "policy": POLICY, "source_catalog_sha256": identity(source),
            "selection_order_sha256": identity([row["name"] for row in members]),
            "volumes": volumes, "members": members, "plan": plan(members, volumes)}
    result = {"schema_version": 1, "policy": POLICY,
        "verification_scope": "Pins derived from checksum-verified original archives. Selective transfer verifies exact selected local headers, compressed bytes, CRC32 and original member SHA256; it does not verify a whole remote archive checksum.",
        "groups": groups}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output = Path(output)
    if output.exists() and output.read_text("utf-8") != text:
        raise ValueError("Existing transport catalog differs; use a new version instead of replacing it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({key: value["plan"] for key, value in groups.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.downloads, args.output)
