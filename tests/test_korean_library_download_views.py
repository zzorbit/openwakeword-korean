"""Public download rows describe real transfers, cache checks and queue state."""
import threading

import pytest

from korean_openwakeword.downloads import download_many


def test_complete_queue_is_visible_and_another_provider_fills_a_finished_slot():
    release = threading.Event()
    records, started = [], []
    items = [{"id": key, "label": key.upper(), "group": key, "group_label": key.upper(),
              "size_bytes": 100} for key in ("slow", "fast", "next")]

    def worker(item, report):
        started.append(item["id"])
        if item["id"] == "slow":
            assert release.wait(5)
        elif item["id"] == "next":
            release.set()
        report({"phase": "download", "done": 100, "total": 100, "unit": "bytes",
                "network_bytes": 100, "network_active": False})
        return item["id"]

    assert download_many(items, worker, progress=records.append, max_workers=2) == ["slow", "fast", "next"]
    assert started.index("next") > started.index("fast")
    assert all(row["state"] == "waiting" for row in records[0]["files"])
    assert [row["label"] for row in records[0]["files"]] == ["SLOW", "FAST", "NEXT"]
    assert all(row["state"] == "complete" for row in records[-1]["files"])
    assert records[-1]["network_bytes"] == 300


def test_transfer_offsets_cache_hash_rates_and_snapshots_stay_distinct():
    records = []

    def worker(item, report):
        if item["id"] == "resume":
            report({"phase": "download", "done": 80, "total": 100, "unit": "bytes",
                    "network_bytes": 10, "network_active": True, "bytes_per_second": 5,
                    "eta_seconds": 4, "attempt": 1})
            report({"phase": "hash", "done": 0, "total": 100, "unit": "bytes",
                    "network_bytes": 30, "network_active": False, "bytes_per_second": 1000000})
            return {"reused": False}
        report({"phase": "hash", "done": 0, "total": 100, "unit": "bytes",
                "network_bytes": 0, "network_active": False, "bytes_per_second": 1000000})
        return {"reused": True}

    download_many([{"id": "resume", "size_bytes": 100}, {"id": "cache", "size_bytes": 100}],
                  worker, progress=records.append, max_workers=1)
    transfer = next(record["files"][0] for record in records
                    if record["files"][0]["state"] == "downloading" and record["files"][0]["network_bytes"] == 10)
    assert transfer["done"] == 80 and transfer["bytes_per_second"] == 5
    assert records[0]["files"][0]["state"] == "waiting"
    assert records[-1]["files"][0]["state"] == "complete"
    assert records[-1]["files"][1]["state"] == "reused"
    assert records[-1]["network_bytes"] == 30
    assert all(row["bytes_per_second"] is None for row in records[-1]["files"])
    assert all(record["files"][1]["bytes_per_second"] is None for record in records)


def test_composite_member_hash_does_not_mark_the_whole_job_as_transferred():
    records = []

    def worker(item, report):
        report({"phase": "hash", "done": 12, "total": 100, "unit": "bytes",
                "aggregate_done": True, "network_bytes": 0})
        report({"phase": "retry", "done": 12, "total": 100, "unit": "bytes",
                "network_bytes": 0, "attempt": 2})
        raise ValueError("fixture selected archive is unavailable")

    with pytest.raises(ValueError, match="selected archive"):
        download_many([{"id": "music", "size_bytes": 100}, {"id": "later", "size_bytes": 10}],
                      worker, progress=records.append, max_workers=1)
    assert any(record["files"][0]["state"] == "verifying" and record["files"][0]["done"] == 12
               for record in records)
    assert any(record["files"][0]["state"] == "retrying" and record["files"][0]["attempt"] == 2
               for record in records)
    assert records[-1]["files"][0]["state"] == "failed"
    assert records[-1]["files"][1]["state"] == "waiting"
    assert not any(record["completed"] for record in records)


def test_unknown_size_does_not_invent_totals_or_nonfinite_rates():
    records = []

    def worker(item, report):
        report({"phase": "download", "done": 12, "total": None, "unit": "bytes",
                "network_bytes": 12, "bytes_per_second": float("inf"), "eta_seconds": True})
        return {"reused": False}

    download_many([{"id": "unknown"}], worker, progress=records.append, max_workers=1)
    assert records[-1]["files"][0]["state"] == "complete"
    assert all(record["files"][0]["total"] is None for record in records)
    assert all(record["files"][0]["bytes_per_second"] is None for record in records)
    assert all(record["files"][0]["eta_seconds"] is None for record in records)
