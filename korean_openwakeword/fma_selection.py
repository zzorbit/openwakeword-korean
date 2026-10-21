"""Freeze a diverse, whole-track FMA full selection before any audio download.

Durations and licenses here are publisher metadata, not decoded/admitted hours
or an independent determination of usage rights. Original MP3 bytes are never
decoded or changed by selection. Known held-out aliases can be excluded.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
import ast
import csv
from itertools import chain
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata
from urllib.parse import unquote, urlsplit

from .common import digest, identity
from .budgets import DEFAULT_FMA_HOURS


_MEMBER = re.compile(r"fma_full/(\d{3})/(\d{6})\.mp3")
_UNKNOWN = {"", "none", "nan", "null", "unknown", "n/a"}
_LICENSE_POLICIES = {"allow-derivatives", "commercial-only"}


def _get(entry, names, default=None):
    for name in names:
        if isinstance(entry, Mapping) and name in entry:
            return entry[name]
        if not isinstance(entry, Mapping) and hasattr(entry, name):
            return getattr(entry, name)
    return default


def _integer(value, name, *, maximum=1024**4):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"FMA ZIP entry needs a bounded nonnegative {name}")
    return value


def _entries(index_entries):
    if isinstance(index_entries, Mapping):
        if any(name in index_entries for name in ("filename", "name", "archive_member", "path")):
            index_entries = [index_entries]
        else:
            index_entries = index_entries.values()
    rows = {}
    for number, entry in enumerate(index_entries):
        if number >= 1_000_000:
            raise ValueError("FMA ZIP entry inventory exceeds its bounded limit")
        member = _get(entry, ("filename", "name", "archive_member", "member", "path"))
        if not isinstance(member, str):
            raise ValueError("FMA ZIP entry needs its actual archive filename")
        matched = _MEMBER.fullmatch(member)
        if matched is None:
            continue
        if matched[1] != matched[2][:3]:
            raise ValueError("FMA archive directory and track ID disagree")
        track = int(matched[2])
        if track < 1 or track in rows:
            raise ValueError("FMA full ZIP repeats or invalidates a track ID")
        compressed = _integer(_get(entry, ("compress_size", "compressed_bytes", "compressed_size")), "compressed size")
        original = _integer(_get(entry, ("file_size", "original_bytes", "uncompressed_bytes", "uncompressed_size")), "original size")
        if compressed == 0 or original == 0:
            raise ValueError("FMA original audio member cannot be empty")
        offset = _integer(_get(entry, ("header_offset", "local_header_offset", "offset")), "header offset")
        checksum = _get(entry, ("CRC", "crc32", "crc"))
        if isinstance(checksum, str) and re.fullmatch(r"[0-9a-fA-F]{8}", checksum):
            checksum = int(checksum, 16)
        checksum = _integer(checksum, "CRC32", maximum=0xFFFFFFFF)
        flags = _integer(_get(entry, ("flag_bits", "flags"), 0), "flags", maximum=0xFFFF)
        compression = _integer(_get(entry, ("compress_type", "compression", "compression_type"), 8), "compression", maximum=0xFFFF)
        if flags & 1:
            raise ValueError("Encrypted FMA audio members are unsupported")
        extra = _get(entry, ("extra",), b"")
        extra_size = len(extra) if isinstance(extra, (bytes, bytearray)) else _get(entry, ("extra_bytes",), 0)
        extra_size = _integer(extra_size, "extra length", maximum=65535)
        rows[track] = {"track_id": track, "archive_member": member, "filename": member,
            "compressed_bytes": compressed, "original_bytes": original, "header_offset": offset,
            "crc32": f"{checksum:08x}", "compress_type": compression, "flag_bits": flags,
            # A local ZIP header can differ in extra-field length. Payload is
            # exact; this header-inclusive transfer figure remains an estimate.
            "estimated_header_bytes": 30 + len(member.encode("utf-8")) + extra_size}
    if not rows:
        raise ValueError("FMA full ZIP has no original MP3 members")
    return rows


def _id_set(values, kind):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Excluded {kind} IDs must be a collection")
    result = set()
    for value in values:
        if isinstance(value, str):
            prefix = f"fma:{kind}:"
            value = value[len(prefix):] if value.startswith(prefix) else value
            if not value.isdigit():
                raise ValueError(f"Excluded {kind} ID must be numeric or canonical")
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Excluded {kind} ID must be positive")
        result.add(value)
    return result


def _artist_name(value):
    return " ".join(unicodedata.normalize("NFKC", value).replace("_", " ").casefold().split())


def _url(value):
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        return None
    host = parsed.hostname.casefold().removeprefix("www.")
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is not None and port not in (80, 443):
        host += f":{port}"
    path = unquote(parsed.path).rstrip("/") or "/"
    return host, path


def _license(terms, policy):
    """Conservative provider-label filter; never a new grant of permissions."""
    normalized = terms.strip().lower()
    if normalized in _UNKNOWN:
        return "missing_license", None
    normalized = normalized.replace("–", "-").replace("—", "-")
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    cc_url = re.search(r"creativecommons\.org/licenses/(by(?:-nc)?(?:-sa|-nd)?)/", normalized)
    cc_name = re.match(r"^(?:creative commons\s+)?attribution(?:-(?:noncommercial|sharealike|noderivatives))*\s+[1-4]\.\d\b", normalized)
    cc_short = re.match(r"^cc[ -]+by(?:[ -]+(?:nc|sa|nd))*[ -]+[1-4]\.\d\b", normalized)
    public_domain = (normalized in ("public domain", "public domain dedication")
                     or re.match(r"^cc0(?:\s|-)1\.0\b", normalized)
                     or "creativecommons.org/publicdomain/zero/" in normalized
                     or "creativecommons.org/publicdomain/mark/" in normalized)
    if not (cc_url or cc_name or cc_short or public_domain):
        return "unrecognized_license", None
    no_derivatives = "noderivative" in compact or bool(re.search(r"(?:\bnd\b|/by(?:-nc)?-nd/)", normalized))
    noncommercial = "noncommercial" in compact or bool(re.search(r"(?:\bnc\b|/by-nc(?:-sa|-nd)?/)", normalized))
    sharealike = "sharealike" in compact or bool(re.search(r"(?:\bsa\b|/by(?:-nc)?-sa/)", normalized))
    if no_derivatives:
        return "no_derivatives_license", None
    if policy == "commercial-only" and noncommercial:
        return "noncommercial_license", None
    return None, {"noncommercial": noncommercial, "sharealike": sharealike,
                  "provider_label_only": True, "usage_permission_independently_verified": False}


def _genres(raw):
    if not raw.strip() or raw.strip().lower() in _UNKNOWN:
        return []
    try:
        values = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        raise ValueError("FMA track genres need the original provider integer list") from None
    if (not isinstance(values, (list, tuple)) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values)):
        raise ValueError("FMA track genres need positive integer provider IDs")
    return sorted(set(values))


def _metadata(path, archive, policy, excluded_tracks, excluded_artists, excluded_names, excluded_urls):
    candidates, rejected, seen = [], Counter(), set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            upper, lower, first = next(reader), next(reader), next(reader)
        except StopIteration:
            raise ValueError("FMA selection needs the original multiheader tracks.csv") from None
        if len(upper) != len(lower) or not first:
            raise ValueError("FMA tracks.csv header widths disagree")
        columns = list(zip(upper, lower))
        def column(key, required=False):
            if columns.count(key) > 1 or required and key not in columns:
                raise ValueError(f"FMA tracks.csv needs one {key} column")
            return columns.index(key) if key in columns else None
        duration_i, license_i, artist_i = (column(key, True) for key in (("track", "duration"), ("track", "license"), ("artist", "id")))
        name_i, url_i = column(("artist", "name")), column(("track", "url"))
        top_i, genres_i, title_i = column(("track", "genre_top")), column(("track", "genres")), column(("track", "title"))
        for row in chain([] if first[0] == "track_id" else [first], reader):
            if not row or len(row) != len(columns) or not row[0].isdigit() or int(row[0]) < 1:
                raise ValueError("FMA tracks.csv rows need unique positive IDs and original column widths")
            track = int(row[0])
            if track in seen:
                raise ValueError("FMA tracks.csv repeats a track ID")
            seen.add(track)
            if track not in archive:
                rejected["missing_archive_member"] += 1
                continue
            if track in excluded_tracks:
                rejected["excluded_track_id"] += 1
                continue
            artist_text = row[artist_i].strip()
            if not artist_text.isdigit() or int(artist_text) < 1:
                rejected["missing_artist_id"] += 1
                continue
            artist = int(artist_text)
            artist_name = row[name_i].strip() if name_i is not None else ""
            original_url = row[url_i].strip() if url_i is not None else ""
            if artist in excluded_artists:
                rejected["excluded_artist_id"] += 1
                continue
            if artist_name and _artist_name(artist_name) in excluded_names:
                rejected["excluded_artist_name"] += 1
                continue
            url = _url(original_url)
            if url and any(url[0] == host and (url[1] == prefix or url[1].startswith(prefix.rstrip("/") + "/")) for host, prefix in excluded_urls):
                rejected["excluded_original_url_prefix"] += 1
                continue
            reason, license_flags = _license(row[license_i], policy)
            if reason is not None:
                rejected[reason] += 1
                continue
            try:
                seconds = float(row[duration_i])
            except ValueError:
                seconds = math.nan
            if not math.isfinite(seconds) or seconds <= 0:
                rejected["invalid_provider_duration"] += 1
                continue
            try:
                genres = _genres(row[genres_i]) if genres_i is not None else []
            except ValueError:
                rejected["invalid_provider_genres"] += 1
                continue
            top = row[top_i].strip() if top_i is not None else ""
            if top.lower() in _UNKNOWN:
                top = ""
            stratum = "genre_top:" + top if top else "genre_id:" + str(genres[0]) if genres else "provider_genre_unknown"
            candidates.append({**archive[track], "provider_duration_s": seconds,
                "license": row[license_i], "license_flags": license_flags,
                "artist_id": artist, "artist_name": artist_name,
                "canonical_source_id": f"fma:track:{track}", "canonical_artist_id": f"fma:artist:{artist}",
                "original_source_url": original_url, "provider_genre_top": top,
                "provider_genres": genres, "provider_title": row[title_i] if title_i is not None else "",
                "selection_stratum": stratum})
    if not seen:
        raise ValueError("FMA tracks.csv contains no original track records")
    rejected["missing_provider_track_metadata"] += len(set(archive) - seen)
    return candidates, {key: count for key, count in sorted(rejected.items()) if count}, len(seen)


def _ordered(candidates, seed, max_artist_hours, skipped):
    groups = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        groups[row["selection_stratum"]][row["artist_id"]].append(row)
    artists, tracks, seconds, artist_seconds = {}, {}, defaultdict(float), defaultdict(float)
    genre_rank = {genre: identity([seed, "genre", genre]) for genre in groups}
    for genre, members in groups.items():
        artists[genre] = deque(sorted(members, key=lambda artist: identity([seed, "artist", genre, artist])))
        tracks[genre] = {artist: deque(sorted(rows, key=lambda row: identity([seed, "track", genre, row["track_id"]]))) for artist, rows in members.items()}
    while artists:
        genre = min(artists, key=lambda value: (seconds[value], genre_rank[value]))
        artist = artists[genre].popleft()
        row = tracks[genre][artist].popleft()
        if tracks[genre][artist]:
            artists[genre].append(artist)
        if not artists[genre]:
            del artists[genre]
        if max_artist_hours is not None and artist_seconds[artist] + row["provider_duration_s"] > max_artist_hours * 3600 + 1e-8:
            skipped["artist_duration_cap"] += 1
            continue
        seconds[genre] += row["provider_duration_s"]
        artist_seconds[artist] += row["provider_duration_s"]
        yield row


def _quantities(rows):
    return {"track_count": len(rows), "provider_hours": math.fsum(row["provider_duration_s"] for row in rows) / 3600,
        "member_payload_bytes": sum(row["compressed_bytes"] for row in rows),
        "original_mp3_bytes": sum(row["original_bytes"] for row in rows),
        "estimated_member_transfer_bytes": sum(row["compressed_bytes"] + row["estimated_header_bytes"] for row in rows)}


def select_fma_tracks(tracks_csv, index_entries, *, target_hours=DEFAULT_FMA_HOURS, seed=20261021,
                      license_policy="allow-derivatives", excluded_track_ids=(),
                      excluded_artist_ids=(), excluded_artist_names=(), original_url_prefixes=(),
                      max_artist_hours=None, reserve_hours=None):
    """Return a frozen, JSON-serializable selection with deterministic reserves.

    Genre strata receive whole tracks by lowest selected metadata duration;
    seeded artist round-robin within each stratum avoids first-ID selection.
    The default permits declared NC/SA terms but rejects unknown/ND labels.
    It does not certify legal permission, source independence or actual hours.
    """
    for name, value in (("target_hours", target_hours), ("max_artist_hours", max_artist_hours), ("reserve_hours", reserve_hours)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or name != "reserve_hours" and value == 0):
            raise ValueError(f"{name} must be a finite positive number (reserve may be zero)")
    if target_hours is None or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("A positive target and nonnegative integer seed are required")
    if license_policy not in _LICENSE_POLICIES:
        raise ValueError("FMA license policy must be allow-derivatives or commercial-only")
    normalized_inputs = []
    for name, values in (("artist names", excluded_artist_names), ("URL prefixes", original_url_prefixes)):
        if isinstance(values, (str, bytes)):
            raise ValueError(f"Excluded {name} must be a collection of nonempty strings")
        values = tuple(values)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Excluded {name} must be a collection of nonempty strings")
        normalized_inputs.append(values)
    excluded_artist_names, original_url_prefixes = normalized_inputs
    names = sorted({_artist_name(value) for value in excluded_artist_names})
    urls = sorted({_url(value) for value in original_url_prefixes}, key=str)
    if any(value is None for value in urls):
        raise ValueError("Excluded original URL prefixes must be valid HTTP(S) URLs")
    track_ids, artist_ids = _id_set(excluded_track_ids, "track"), _id_set(excluded_artist_ids, "artist")
    metadata = Path(tracks_csv).resolve()
    if not metadata.is_file():
        raise ValueError("FMA selection needs the original tracks.csv file")
    metadata_sha256 = digest(metadata)
    archive = _entries(index_entries)
    candidates, rejected, metadata_count = _metadata(metadata, archive, license_policy, track_ids, artist_ids, set(names), urls)
    if digest(metadata) != metadata_sha256:
        raise ValueError("FMA metadata changed during track selection")
    eligible_hours = math.fsum(row["provider_duration_s"] for row in candidates) / 3600
    selected, reserve, seconds = [], [], 0.0
    skipped = Counter()
    ordered = _ordered(candidates, seed, max_artist_hours, skipped)
    for row in ordered:
        selected.append(row)
        seconds += row["provider_duration_s"]
        if seconds + 1e-8 >= target_hours * 3600:
            break
    target_met = seconds + 1e-8 >= target_hours * 3600
    reserve_hours = max(1, float(target_hours) * .05) if reserve_hours is None else reserve_hours
    reserve_seconds = 0.0
    if target_met and reserve_hours:
        for row in ordered:
            reserve.append(row)
            reserve_seconds += row["provider_duration_s"]
            if reserve_seconds + 1e-8 >= reserve_hours * 3600:
                break
    quantities, reserve_quantities = _quantities(selected), _quantities(reserve)
    genres, artist_counts, artist_times = defaultdict(list), Counter(), defaultdict(float)
    for row in selected:
        genres[row["selection_stratum"]].append(row)
        artist_counts[row["artist_id"]] += 1
        artist_times[row["artist_id"]] += row["provider_duration_s"]
    result = {"schema_version": 1, "kind": "fma_full_track_selection_v1", "selection_policy": "genre_duration_artist_round_robin_v1",
        "target_hours": float(target_hours), "seed": seed, "license_policy": license_policy,
        "max_artist_hours": max_artist_hours, "reserve_target_hours": float(reserve_hours),
        "metadata": {"path": str(metadata), "sha256": metadata_sha256},
        "archive_index_sha256": identity([archive[key] for key in sorted(archive)]),
        "excluded_track_ids": sorted(track_ids), "excluded_artist_ids": sorted(artist_ids),
        "excluded_artist_names_normalized": names,
        "excluded_original_url_prefixes_normalized": [list(value) for value in urls],
        "metadata_track_count": metadata_count, "archive_track_count": len(archive),
        "eligible_track_count": len(candidates), "eligible_provider_hours": eligible_hours,
        "rejected_counts": rejected, "selection_skipped_counts": dict(skipped), "selected": selected, "reserve": reserve,
        "selected_summary": quantities, "reserve_summary": reserve_quantities,
        "selected_provider_hours": quantities["provider_hours"], "selected_track_count": len(selected),
        "estimated_member_payload_bytes": quantities["member_payload_bytes"],
        "estimated_member_transfer_bytes": quantities["estimated_member_transfer_bytes"],
        "target_met": target_met, "shortfall_provider_hours": max(0, float(target_hours) - quantities["provider_hours"]),
        "reserve_target_met": reserve_quantities["provider_hours"] + 1e-12 >= reserve_hours,
        "genre_distribution": {genre: _quantities(rows) for genre, rows in sorted(genres.items())},
        "selected_artist_count": len(artist_counts), "max_selected_artist_track_count": max(artist_counts.values(), default=0),
        "max_selected_artist_provider_hours": max(artist_times.values(), default=0) / 3600,
        "duration_scope": "Provider metadata only; decoded whole-track time must reach the target, using frozen reserve tracks only when needed.",
        "transfer_scope": "Compressed member payload sizes are exact index values; local-header and retry overhead is separately bounded by the downloader.",
        "exclusion_scope": "Explicit known IDs, normalized artist-name equality and source-URL path prefixes only; unknown aliases and acoustic duplicates remain unverified.",
        "content_approved": False, "admitted_hours": 0, "independent_STT_runs": 0,
        "audio_reencoded": False, "whole_archive_sha1_verified": False}
    result["selection_sha256"] = identity(result)
    return result
