"""Read original public corpus metadata without STT or invented identities."""
from collections import Counter, defaultdict
import csv
from itertools import chain
import math
from pathlib import Path
import re

from .common import digest, read_json

_AUDIO = frozenset({".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".au"})
_UNKNOWN = frozenset({"", "none", "nan", "null", "unknown", "n/a"})


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def _files(directory):
    if not directory.is_dir():
        raise ValueError("Provider input must be an existing audio directory")
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO)
    if not paths or any(not path.resolve().is_relative_to(directory) for path in paths):
        raise ValueError("Provider input needs audio files without escaping symlinks")
    return paths


def _descendants(ontology, root):
    visited, todo = set(), [root]
    while todo:
        current = todo.pop()
        if current in visited:
            continue
        if current not in ontology:
            raise ValueError(f"AudioSet ontology is missing class {current}")
        visited.add(current)
        children = ontology[current].get("child_ids")
        if not isinstance(children, list) or any(not isinstance(child, str) for child in children):
            raise ValueError("AudioSet ontology needs declared child_ids")
        todo.extend(children)
    return visited


def _audioset(directory, paths, metadata, ontology, dataset_id, source_url, license):
    if metadata is None or ontology is None:
        raise ValueError("AudioSet needs the original segments CSV and ontology JSON")
    metadata, ontology_path = Path(metadata).resolve(), Path(ontology).resolve()
    nodes = read_json(ontology_path)
    if (not isinstance(nodes, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
            or not isinstance(row.get("name"), str) for row in nodes)
            or len({row["id"] for row in nodes}) != len(nodes)):
        raise ValueError("AudioSet ontology class IDs must be unique")
    classes = {row["id"]: row for row in nodes}
    speech, music = _descendants(classes, "/m/09x0r"), _descendants(classes, "/m/04rlf")
    by_video = {}
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader((line for line in stream if line.strip() and not line.lstrip().startswith("#")), skipinitialspace=True)
        for line in reader:
            if len(line) != 4 or not re.fullmatch(r"[A-Za-z0-9_-]{11}", line[0]):
                raise ValueError("AudioSet segments CSV needs video ID, start, end and quoted labels")
            start, end = float(line[1]), float(line[2])
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError("AudioSet parent time interval must be finite and positive")
            ids = [value.strip() for value in line[3].split(",")]
            if not ids or any(value not in classes for value in ids):
                raise ValueError("AudioSet segment contains unknown ontology labels")
            if line[0] in by_video:
                raise ValueError("Repeated AudioSet video ID is ambiguous for one-file-per-video audio; use an unambiguous subset")
            by_video[line[0]] = (start, end, ids)
    bindings = [_binding(metadata), _binding(ontology_path)]
    rows = []
    for path in paths:
        video = path.stem
        if video not in by_video:
            raise ValueError(f"AudioSet audio filename has no exact provider segment: {path.name}")
        start, end, ids = by_video[video]
        has_speech = bool(set(ids) & speech)
        category = "competing_speech" if has_speech else "music" if set(ids) & music else "environment"
        rows.append({"path": path.relative_to(directory).as_posix(), "sha256": digest(path),
                     "dataset_id": dataset_id or "audioset", "source_url": source_url or "https://research.google.com/audioset/download.html",
                     "original_audio_url": f"https://www.youtube.com/watch?v={video}",
                     "license": license or "AudioSet metadata CC-BY-4.0; individual original YouTube audio rights not certified",
                     "canonical_source_id": "audioset:video:" + video,
                     "canonical_recording_id": "audioset:video:" + video, "parent_start_s": start, "parent_end_s": end,
                     "category": category, "speech": True if has_speech else None,
                     "speech_presence": "provider_label_present" if has_speech else "unknown",
                     "language": "unknown", "class_ids": ids, "class_names": [classes[value]["name"] for value in ids],
                     "category_basis": "AudioSet original provider ontology presence labels; absence does not certify no speech/music",
                     "provider_metadata_bindings": bindings, "original_dataset_partition": metadata.stem})
    return rows, bindings, ["AudioSet presence labels do not certify unlabeled sounds absent, language, speaker identity or individual original audio permission."]


def _fma(directory, paths, metadata, dataset_id, source_url):
    if metadata is None:
        raise ValueError("FMA needs its original two-header tracks.csv")
    metadata = Path(metadata).resolve()
    grouped = defaultdict(list)
    for path in paths:
        if not path.stem.isdigit():
            raise ValueError(f"FMA audio filename needs its numeric provider track ID: {path.name}")
        grouped[int(path.stem)].append(path)
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            upper, lower, first = next(reader), next(reader), next(reader)
        except StopIteration:
            raise ValueError("FMA tracks.csv is missing its original headers") from None
        if len(upper) != len(lower):
            raise ValueError("FMA tracks.csv header widths differ")
        if not first:
            raise ValueError("FMA tracks.csv has an empty index/data header")
        columns = list(zip(upper, lower))
        def position(key, required=False):
            if columns.count(key) > 1 or required and key not in columns:
                raise ValueError(f"FMA metadata needs one {key} column")
            return columns.index(key) if key in columns else None
        license_column = position(("track", "license"), True)
        artist_column = position(("artist", "id"), True)
        partition_column, url_column = position(("set", "split")), position(("track", "url"))
        records, seen = {}, set()
        for row in chain([] if first[0] == "track_id" else [first], reader):
            if not row or len(row) != len(columns) or not row[0].isdigit():
                raise ValueError("FMA track rows need a numeric ID and the original column count")
            track = int(row[0])
            if track in seen:
                raise ValueError("FMA track IDs must be unique")
            seen.add(track)
            if track in grouped:
                records[track] = (row[license_column], row[artist_column].strip(),
                                  row[partition_column] if partition_column is not None else "unknown",
                                  row[url_column].strip() if url_column is not None else "")
    bindings = [_binding(metadata)]
    if set(grouped) != set(records):
        raise ValueError("FMA audio filename has no exact provider track metadata")
    rows = []
    for track, versions in sorted(grouped.items()):
        terms, artist, partition, url = records[track]
        # Keep the original MP3 when a previous conversion sits in the same folder.
        chosen = min(versions, key=lambda p: (p.suffix.lower() != ".mp3", p.as_posix()))
        for path in versions:
            common = {"path": path.relative_to(directory).as_posix(), "sha256": digest(path), "track_id": track,
                      "canonical_source_id": f"fma:track:{track}", "canonical_recording_id": f"fma:track:{track}",
                      "dataset_id": dataset_id or "fma", "provider_metadata_bindings": bindings}
            if terms.strip().lower() in _UNKNOWN or path != chosen:
                rows.append({**common, "excluded": True, "exclusion_reason": "Missing per-track provider license" if terms.strip().lower() in _UNKNOWN else "Duplicate file format of the same provider track; original MP3 preferred"})
                continue
            rows.append({**common, "license": terms, "source_url": source_url or url or "https://github.com/mdeff/fma",
                         "category": "music", "speech": None, "speech_presence": "unknown", "language": "unknown",
                         "canonical_artist_id": "fma:artist:" + artist if artist else None,
                         "original_dataset_partition": partition,
                         "category_basis": "FMA original music corpus and per-track metadata; vocal presence/language not inferred"})
    return rows, bindings, ["FMA artist IDs are provider artist identities, not certified speaker identities. Per-track licenses are preserved; none are replaced by a corpus-wide license."]


def _mit(directory, paths, metadata, room_metadata, dataset_id, source_url, license):
    bindings, by_path = [], {}
    if metadata is not None:
        metadata = Path(metadata).resolve()
        content = read_json(metadata)
        records = content if isinstance(content, list) else content.get("files")
        if not isinstance(records, list):
            raise ValueError("MIT optional file metadata needs a files list")
        for row in records:
            path = Path(row["path"])
            path = path.resolve() if path.is_absolute() else (directory / path).resolve()
            if not path.is_relative_to(directory) or path in by_path:
                raise ValueError("MIT provider metadata paths must be unique and inside input")
            by_path[path] = row
        bindings.append(_binding(metadata))
    rooms = {}
    if room_metadata is not None:
        room_metadata = Path(room_metadata).resolve()
        if room_metadata.suffix.lower() == ".csv":
            with room_metadata.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
        else:
            content = read_json(room_metadata)
            records = content if isinstance(content, list) else content.get("files")
        if not isinstance(records, list):
            raise ValueError("MIT actual room metadata needs path/canonical_room_id records")
        for row in records:
            room, path = row.get("canonical_room_id"), Path(row["path"])
            path = path.resolve() if path.is_absolute() else (directory / path).resolve()
            if not isinstance(room, str) or not room.strip() or path in rooms or not path.is_relative_to(directory):
                raise ValueError("MIT actual room mapping needs unique paths and actual nonempty room IDs")
            rooms[path] = room
        if set(rooms) != {path.resolve() for path in paths}:
            raise ValueError("MIT room metadata must cover every audio file exactly")
        bindings.append(_binding(room_metadata))
    if by_path and set(by_path) != {path.resolve() for path in paths}:
        raise ValueError("MIT file metadata must cover every audio file exactly")
    collection = dataset_id or "mit_environmental_impulse_responses"
    declared_rooms = {path: rooms.get(path.resolve(), by_path.get(path.resolve(), {}).get("canonical_room_id")) for path in paths}
    if any(room is not None and (not isinstance(room, str) or not room.strip()) for room in declared_rooms.values()):
        raise ValueError("MIT provider room identity must be nonempty")
    complete_rooms = all(room is not None for room in declared_rooms.values())
    rows = []
    for path in paths:
        supplied = by_path.get(path.resolve(), {})
        expected = supplied.get("sha256")
        actual = digest(path)
        if expected is not None and expected != actual:
            raise ValueError("MIT source differs from its provider file hash")
        room = declared_rooms[path] if complete_rooms else None
        rows.append({"path": path.relative_to(directory).as_posix(), "sha256": actual,
                     "dataset_id": collection, "canonical_source_id": "mit_ir:file:" + path.relative_to(directory).as_posix(),
                     "canonical_recording_id": "mit_ir:file:" + path.relative_to(directory).as_posix(),
                     "canonical_room_id": room or "mit_ir:unknown_collection_room_group",
                     "room_identity_scope": "provider_room_identity" if room else "conservative_collection_group_not_independent_room",
                     "category": "rir", "speech": False, "speech_presence": "impulse_response_filter", "language": "not_applicable",
                     "source_url": source_url or "https://mcdermottlab.mit.edu/Reverb/IR_Survey.html",
                     "license": supplied.get("license") or license or "Unknown; verify original MIT source usage conditions",
                     "category_basis": "MIT published measured environmental impulse response corpus; not filename-inferred room independence",
                     "provider_metadata_bindings": bindings,
                     "room_metadata_available": room is not None,
                     "provider_declared_room_id": declared_rooms[path],
                     "independent_room_partition_available": complete_rooms})
    limitation = [] if complete_rooms else ["MIT files without a complete actual room mapping all share one conservative collection room group. This is not proof of independent rooms; prepare the whole collection for Train with validation_fraction=0 and use a separate RIR corpus for held-out evaluation."]
    return rows, bindings, limitation


def build_provider_index(input_directory, *, dataset, metadata=None, ontology=None,
                         room_metadata=None, dataset_id=None, source_url=None, license=None):
    """Create per-file rows from official AudioSet/FMA/MIT metadata, no decoding."""
    directory = Path(input_directory).resolve()
    paths = _files(directory)
    if dataset == "audioset":
        rows, bindings, limitations = _audioset(directory, paths, metadata, ontology, dataset_id, source_url, license)
    elif dataset == "fma":
        rows, bindings, limitations = _fma(directory, paths, metadata, dataset_id, source_url)
    elif dataset == "mit_rir":
        rows, bindings, limitations = _mit(directory, paths, metadata, room_metadata, dataset_id, source_url, license)
    else:
        raise ValueError("Provider dataset must be audioset, fma or mit_rir")
    return {"rows": sorted(rows, key=lambda row: row["path"]),
            "excluded": [{"path": row["path"], "canonical_source_id": row["canonical_source_id"], "reason": row["exclusion_reason"]} for row in rows if row.get("excluded")],
            "provider_metadata": bindings, "limitations": limitations,
            "category_counts": dict(Counter(row["category"] for row in rows if not row.get("excluded"))),
            "independent_STT_runs": 0, "speaker_identities_inferred": False,
            "room_identities_inferred": False, "audio_changed": False}
