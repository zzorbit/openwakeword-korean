"""Stream frozen clip archives; retain only fields needed by score summaries.

The original archive remains the provenance authority. Candidate exports copy
every original field, one record at a time, rather than replacing that evidence
with a reduced display/analysis format.
"""
from collections.abc import Sequence
import json
import os
from pathlib import Path
import uuid

from .paged_site import iter_clip_array


def _identity(path):
    value = Path(path).stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


class FrozenClipRows(Sequence):
    """Repeatable bounded-record view of an already hash-verified JSON array."""
    def __init__(self, path, count):
        self.path, self.count = Path(path), count
        self.identity = _identity(path)

    def __len__(self):
        return self.count

    def __iter__(self):
        if _identity(self.path) != self.identity:
            raise ValueError("Frozen evaluation clips changed during analysis")
        count = 0
        for row in iter_clip_array(self.path):
            count += 1
            yield row
        if count != self.count or _identity(self.path) != self.identity:
            raise ValueError("Frozen evaluation clips changed during analysis")

    def __getitem__(self, index):
        # Compatibility for callers inspecting individual records. Bulk code
        # must iterate once instead of repeatedly indexing this disk view.
        if isinstance(index, slice):
            wanted = range(*index.indices(len(self)))
            return [row for i, row in enumerate(self) if i in wanted]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return next(row for i, row in enumerate(self) if i == index)


class RestrictedClipRows:
    def __init__(self, rows, ids):
        self.rows, self.ids = rows, tuple(ids)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for row in self.rows:
            # Nested source metadata is read-only and can be shared safely.
            yield dict(row, results={key: row["results"][key] for key in self.ids})


def write_clip_array(path, rows):
    """Atomically write an iterable without constructing a second full JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("[\n")
            for index, row in enumerate(rows):
                if index:
                    stream.write(",\n")
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2))
            stream.write("\n]\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_ROW_FIELDS = ("sample_id", "original_id", "kind", "category", "snr_db",
               "reverb_condition", "eligible_seconds", "provenance")
_CALL_FIELDS = ("original_utterance_id", "speaker_id", "provenance", "success",
                "latency_s", "category", "distance_m", "direction", "microphone")


def statistic_projection(row):
    """Discard bulky source/component/event diagnostics from RAM, not exports."""
    result = {key: row[key] for key in _ROW_FIELDS if key in row}
    result["results"] = {
        model: {"calls": [{key: call[key] for key in _CALL_FIELDS if key in call}
                           for call in value["calls"]],
                "false_activations": value["false_activations"],
                "max_score": value["max_score"]}
        for model, value in row["results"].items()}
    return result
