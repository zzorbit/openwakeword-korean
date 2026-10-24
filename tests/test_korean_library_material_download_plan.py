"""All independent fixed originals share one queue; raw refill stays bounded."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from korean_openwakeword import material_downloads as module, official_music
from test_korean_library_material_download_queue import asset, complete_fixture, isolated_plan


@pytest.mark.parametrize("preset,automatic,components", [("check", False, 0), ("full", True, 4)])
def test_complete_fixed_inventory_is_queued_once_and_large_refill_is_only_announced(tmp_path, monkeypatch, preset, automatic, components):
    names = ("generic", "zeroth", "audioset", "audioset_segments", "audioset_ontology")
    payloads = {name + ".bin": (name + " exact fixture original").encode() for name in names}
    base = [asset(name, "training.example" if name != "zeroth" else "korean.example", payloads[name + ".bin"]) for name in names]
    mit = [asset(f"rir_{index:03d}", "mit.example", f"unchanged impulse {index}".encode()) for index in range(270)]
    common = [deepcopy(base[1])] + [asset(name, "common.example", (name + " original").encode())
        for name in ("musan", "seoul", "fsd_eval", "fsd_eval_zip", "fsd_metadata", "fsd_ground_truth")]
    payloads.update({row["filename"]: (row["id"] + " original").encode() for row in common[1:]})
    payloads.update({row["filename"]: f"unchanged impulse {index}".encode() for index, row in enumerate(mit)})
    arguments = isolated_plan(monkeypatch, tmp_path, base=base, mit=mit, common=common, payloads=payloads)
    basic, recipe, config = arguments
    recipe.update(preset=preset)
    if automatic:
        recipe["material_acquisition_policy"] = "automatic_v3"
        config["targets"].update(generic_hours=3500, korean_hours=1500)
    else:
        recipe["official_noise_acquisition_policy"] = official_music.POLICY
    original_arguments = deepcopy(arguments)
    snapshots, calls, batches = [], [], []
    def report(event):
        if event.get("phase") == "download_plan":
            snapshots.append(event)
    real_many = module.download_many
    def queue(items, worker, **kwargs):
        values = list(items)
        batches.append(values)
        return real_many(values, worker, **kwargs)
    monkeypatch.setattr(module, "download_many", queue)
    def worker(owner, item, progress):
        calls.append(item)
        if item["kind"] == "official_music":
            # The existing selected-range helper is separately checked against
            # every original byte and decoded PCM in official_music tests.
            return {"receipt": "unchanged-selected-music-receipt", "files": []}
        return complete_fixture(owner, item, progress, payloads)
    monkeypatch.setattr(module.MaterialDownloads, "_acquire", worker)
    with module.MaterialDownloads(tmp_path, report) as owner:
        owner.prepare(*arguments)
        assert len(batches) == 1
        items = batches[0]
        assert len({item["id"] for item in items}) == len(items)
        assert sum(item["kind"] == "mit" for item in items) == 270
        assert sum(item["kind"] == "common" for item in items) == 6  # Zeroth is shared.
        assert sum(item["kind"] == "common_rir" for item in items) == 1
        assert sum(item["kind"] == "official_music" for item in items) == int(not automatic)
        assert not any(item["id"].startswith("automatic:") for item in items)
        before_start = snapshots[-1]["files"]
        assert len([row for row in before_start if row.get("deferred")]) == components
        assert all(row["state"] == "waiting" for row in before_start)
        assert all(row["total"] is None for row in before_start if row.get("deferred"))
        assert len(calls) == len(items)
        assert list(owner.base_paths) == list(names)
        for row in basic:
            path = owner.base_paths[row["id"]]
            assert path.read_bytes() == payloads[row["filename"]]
            assert module.borrowed_download(path, "sha256", row["sha256"], row["size_bytes"]) == path.resolve()
        fsd = [item for item in items if item.get("asset", {}).get("id", "").startswith("fsd_")]
        assert {item["group"] for item in fsd} == {"fsd50k"}
        assert not any(item["id"] == "common:zeroth" for item in items)
        if automatic:
            owner.finish_automatic({"component_actual_hours": {"audioset_unbalanced": 1500, "mls": 1500, "fma_full": 500, "yodas2": 1500}})
            assert all(row["state"] == "complete" for row in owner.files.values() if row.get("deferred"))
    assert arguments == original_arguments


def test_bounded_automatic_status_never_claims_completion_before_actual_preparation():
    owner = module.MaterialDownloads(Path.cwd())
    owner.files["automatic:yodas2"] = {"id": "automatic:yodas2", "deferred": True,
        "state": "waiting", "done": 0, "total": None, "target_hours": 2}
    owner.report({"label": "학습용 비호출 자료 자동 준비: yodas2", "done": .5, "total": 2, "unit": "hours"})
    assert owner.files["automatic:yodas2"]["state"] == "downloading"
    owner.finish_automatic({"component_actual_hours": {"yodas2": 1}})
    assert owner.files["automatic:yodas2"]["state"] == "failed"
    assert owner.files["automatic:yodas2"]["done"] == 1


@pytest.mark.parametrize("existing", [False, True])
def test_real_md5_archive_verification_can_be_borrowed_only_unchanged_in_this_call(tmp_path, monkeypatch, existing):
    from korean_openwakeword import evaluation_materials, downloads
    payload = b"exact pinned common archive"
    asset = {"id": "fsd_fixture", "filename": "fixture.zip", "size_bytes": len(payload),
             "checksum_algorithm": "md5", "md5": hashlib.md5(payload).hexdigest(),
             "source_url": "https://fixture.example/archive.zip"}
    target = tmp_path / asset["filename"]
    (target if existing else target.with_name(target.name + ".part")).write_bytes(payload)
    monkeypatch.setattr(downloads.requests, "get", lambda *a, **k: pytest.fail("A complete original/partial needs verification, not another transfer"))
    with module.MaterialDownloads(tmp_path) as owner:
        assert evaluation_materials._download_archive(asset, tmp_path, verification_cache=owner.verification_cache) == target
        assert owner.verification_cache.matches(target, "md5", asset["md5"])
        assert owner._remember(target, {"algorithm": "md5", "expected": asset["md5"], "size_bytes": len(payload)})
        receipt = target.with_name(target.name + ".receipt.json")
        original = (target.read_bytes(), receipt.read_bytes(), target.stat().st_mtime_ns)
        real_verify, verified = downloads.verify_checksum, []
        def verify(*args, **kwargs):
            verified.append(args[0])
            return real_verify(*args, **kwargs)
        monkeypatch.setattr(downloads, "verify_checksum", verify)
        assert evaluation_materials._download_archive(asset, tmp_path) == target
        assert verified == []
        assert original == (target.read_bytes(), receipt.read_bytes(), target.stat().st_mtime_ns)
    assert evaluation_materials._download_archive(asset, tmp_path) == target
    assert verified == [target]  # Scope ended: normal verification is mandatory again.
