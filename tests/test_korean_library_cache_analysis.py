"""Large archive operations must be bounded by a record, not its metadata sum."""
import json
import tracemalloc

import pytest

from korean_openwakeword.cache_analysis import FrozenClipRows, write_clip_array


def test_frozen_rows_stream_many_large_provenance_records(tmp_path):
    path = tmp_path / "clips.json"
    record = {"source_notes": "x" * 128000, "nested": {"license": "preserve"}}
    with path.open("w", encoding="utf-8") as stream:
        stream.write("[")
        for index in range(128):
            if index:
                stream.write(",")
            stream.write(json.dumps(dict(record, index=index)))
        stream.write("]")
    rows = FrozenClipRows(path, 128)
    tracemalloc.start()
    try:
        assert sum(row["index"] for row in rows) == sum(range(128))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert path.stat().st_size > 15_000_000
    assert peak < 3_000_000  # A whole-archive read would exceed 15 MB alone.


def test_frozen_rows_reject_concurrent_change_and_wrong_count(tmp_path):
    path = tmp_path / "clips.json"
    write_clip_array(path, [{"n": 1}, {"n": 2}])
    rows = FrozenClipRows(path, 2)
    iterator = iter(rows)
    assert next(iterator) == {"n": 1}
    path.write_text("[{\"n\":9}]", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        list(iterator)
    with pytest.raises(ValueError, match="changed"):
        list(FrozenClipRows(path, 2))


def test_failed_stream_does_not_replace_published_json_or_leave_scratch(tmp_path):
    path = tmp_path / "clips.json"
    path.write_text("original", encoding="utf-8")
    def failing_rows():
        yield {"preserved": True}
        raise ValueError("interrupted")
    with pytest.raises(ValueError, match="interrupted"):
        write_clip_array(path, failing_rows())
    assert path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [path]
