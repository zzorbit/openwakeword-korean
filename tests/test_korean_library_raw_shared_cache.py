"""Whole-shard sharing keeps exact input and historical evidence across budgets."""
import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword import automatic_sources, canonical_audio, canonical_sources
from korean_openwakeword import commercial_sources
from korean_openwakeword.common import digest, identity, read_json, write_json


PROVIDERS = ("audioset_unbalanced", "mls", "yodas2")


def snapshot(directory):
    return {str(path.relative_to(directory)): digest(path)
            for path in directory.rglob("*") if path.is_file()
            and ".canonical_lock" not in path.parts}


class RawShard:
    """A pinned local acquisition with the real canonical conversion/transaction."""
    def __init__(self, directory, component, monkeypatch):
        self.directory = directory
        selections = {"audioset_unbalanced": {"parts": [0]},
                      "mls": {"language": "french", "parts": ["1_1_1"]},
                      "yodas2": {"subset": "ko000", "parts": [0]}}
        self.candidate = {"component": component, "selection": selections[component],
                          "key": "fixed_shard", "archive_bytes": 32}
        self.backend = automatic_sources.PublicSourceBackend(directory.parent,
            storage_policy="canonical_flac_only_v1")
        self.acquisitions = 0
        self.raw = directory / "audio/recording.wav"
        self.resolution_path = directory / "originals/source_resolution.json"
        monkeypatch.setattr(self.backend, "_prepare", self.acquire)
        monkeypatch.setattr(automatic_sources.requests, "get",
                            lambda *a, **k: pytest.fail("No network is needed for shared-cache reuse"))

    def acquire(self, candidate, directory, **kwargs):
        self.acquisitions += 1
        self.raw.parent.mkdir(parents=True, exist_ok=True)
        self.resolution_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.raw, np.random.default_rng(73).normal(0, .1, (4417, 2)),
                 44100, subtype="PCM_24")
        provider = copy.deepcopy(commercial_sources._PROVIDERS[candidate["component"]])
        asset_path = {"audioset_unbalanced": "data/unbal_train000.tar",
                      "mls": "data/mls_french/train/audio/1_1_1.tar.gz",
                      "yodas2": "data/ko000/audio/00000000.tar.gz"}[candidate["component"]]
        asset = {"path": asset_path, "filename": Path(asset_path).name,
                 "source_url": commercial_sources._resolve_url(provider, asset_path),
                 "size_bytes": 32, "checksum_algorithm": "sha256", "sha256": "a" * 64,
                 "upstream_revision": provider["revision"]}
        resolution = {"kind": "commercial_source_resolution_v1", "provider": provider,
                      "source_id": candidate["component"], "selection": candidate["selection"],
                      "assets": [asset], "metadata_assets": [], "total_bytes": 32}
        resolution["resolution_sha256"] = identity(resolution)
        write_json(self.resolution_path, resolution)
        return {"source": {"directory": str(self.raw.parent), "adapter": "fixture"},
                "rows": [{"path": self.raw.name, "sha256": digest(self.raw),
                          "canonical_source_id": "provider:recording", "split": "train"}],
                "receipts": [{"path": str(self.resolution_path), "sha256": digest(self.resolution_path)}],
                "reserved_download_bytes": 32}

    def run(self, hours=2000 / 3, *, excluded=(), candidate=None, max_bytes=1024**2):
        return self.backend.prepare(candidate or self.candidate, self.directory,
            target_hours=hours, excluded_track_ids=excluded,
            max_download_bytes=1024**2, max_extract_bytes=max_bytes)


@pytest.mark.parametrize("component", PROVIDERS)
def test_whole_raw_shard_reuses_other_budget_without_changing_any_file(tmp_path, monkeypatch, component):
    source = RawShard(tmp_path / "part", component, monkeypatch)
    first = source.run()
    row = first["rows"][0]
    flac = Path(first["source"]["directory"]) / row["path"]
    samples = sf.read(flac, dtype="int16")[0]
    before = snapshot(source.directory)
    monkeypatch.setattr(canonical_audio, "canonicalize_prepared_source",
                        lambda *a: pytest.fail("Identical prepared audio must not be converted twice"))
    again = source.run(1500, excluded=[2, 1, 2])
    assert source.acquisitions == 1 and not source.raw.exists()
    assert first["rows"] == again["rows"]
    assert again["canonical_receipt"] == first["canonical_receipt"] and again["resumed"]
    assert snapshot(source.directory) == before
    assert np.array_equal(samples, sf.read(flac, dtype="int16")[0])
    # A later return to the original scale also uses the original exact request.
    assert source.run(excluded=[3])["canonical_receipt"] == first["canonical_receipt"]
    assert source.run()["canonical_receipt"] == first["canonical_receipt"]


@pytest.mark.parametrize("after_unlink", [False, True])
def test_cross_budget_resumes_verified_pending_retirement_without_reacquisition(
        tmp_path, monkeypatch, after_unlink):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    retire = canonical_sources._retire_original
    def interrupt(directory, mapping, claim):
        if after_unlink:
            retire(directory, mapping, claim)
        raise RuntimeError("interrupted after durable PCM commit")
    monkeypatch.setattr(canonical_sources, "_retire_original", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        source.run()
    stable = {name: digest(source.directory / name) for name in (
        "canonical_request.json", "canonical_acquisition.json", "canonical_pending.json")}
    monkeypatch.setattr(canonical_sources, "_retire_original", retire)
    result = source.run(1500, excluded=[3])
    assert source.acquisitions == 1 and not source.raw.exists()
    assert stable == {name: digest(source.directory / name) for name in stable}
    assert canonical_sources.verify_canonical_source(source.directory)["canonical_receipt"] == result["canonical_receipt"]


@pytest.mark.parametrize("field", ["component", "selection", "key", "archive_bytes"])
def test_changed_candidate_still_refuses_reuse(tmp_path, monkeypatch, field):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    source.run()
    changed = copy.deepcopy(source.candidate)
    changed[field] = {"component": "mls", "selection": {"parts": [1]},
                      "key": "another_shard", "archive_bytes": 33}[field]
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="another request"):
        source.run(1500, candidate=changed)
    assert source.acquisitions == 1 and snapshot(source.directory) == before


@pytest.mark.parametrize("change", ["seed", "storage", "unknown_new", "unknown_removed", "conversion"])
def test_only_the_two_known_noncausal_fields_are_ignored(tmp_path, monkeypatch, change):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    request = {"candidate": source.candidate, "target_hours": 2,
               "excluded_track_ids": [], "seed": automatic_sources.SEED,
               "storage_policy": "canonical_flac_only_v1"}
    if change == "unknown_removed":
        request["future_input_policy"] = "v1"
    def run(value):
        return canonical_sources.prepare_canonical_source(source.directory, request=value,
            acquire=lambda: source.acquire(source.candidate, source.directory),
            canonicalize=canonical_audio.canonicalize_prepared_source, reuse_raw_shard=True)
    run(request)
    changed = copy.deepcopy(request)
    changed["target_hours"] = 1500
    if change == "seed": changed["seed"] += 1
    if change == "storage": changed["storage_policy"] = "another_policy"
    if change == "unknown_new": changed["future_input_policy"] = "v1"
    if change == "unknown_removed": del changed["future_input_policy"]
    if change == "conversion": changed["resampler"] = "another_algorithm"
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="another request"):
        run(changed)
    assert source.acquisitions == 1 and snapshot(source.directory) == before


def test_current_publisher_revision_cannot_be_bypassed_by_a_budget_change(tmp_path, monkeypatch):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    source.run()
    provider = {**commercial_sources._PROVIDERS["audioset_unbalanced"], "revision": "b" * 40}
    monkeypatch.setitem(commercial_sources._PROVIDERS, "audioset_unbalanced", provider)
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="provider or revision"):
        source.run(1500)
    assert source.acquisitions == 1 and snapshot(source.directory) == before


def test_different_retained_publisher_selection_is_not_accepted(tmp_path, monkeypatch):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    source.run()
    resolution = read_json(source.resolution_path)
    resolution["selection"] = {"parts": [9]}
    resolution["resolution_sha256"] = identity({k: v for k, v in resolution.items()
                                               if k != "resolution_sha256"})
    write_json(source.resolution_path, resolution)
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="publisher selection changed"):
        source.run(1500)
    assert source.acquisitions == 1 and snapshot(source.directory) == before


def test_cross_budget_can_finish_a_valid_conversion_that_stopped_before_commit(tmp_path, monkeypatch):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    convert = canonical_audio.canonicalize_prepared_source
    def interrupt(prepared, output):
        convert(prepared, output)
        raise RuntimeError("interrupted before pending commit")
    monkeypatch.setattr(canonical_audio, "canonicalize_prepared_source", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        source.run()
    assert source.raw.exists() and not (source.directory / "canonical_pending.json").exists()
    request_sha = digest(source.directory / "canonical_request.json")
    monkeypatch.setattr(canonical_audio, "canonicalize_prepared_source", convert)
    result = source.run(1500)
    assert source.acquisitions == 1 and not source.raw.exists()
    assert digest(source.directory / "canonical_request.json") == request_sha
    assert canonical_sources.verify_canonical_source(source.directory)["canonical_receipt"] == result["canonical_receipt"]


def test_missing_early_resolution_fails_closed_but_original_request_can_resume(tmp_path, monkeypatch):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    monkeypatch.setattr(source.backend, "_prepare", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("interrupted before publisher resolution")))
    with pytest.raises(RuntimeError, match="interrupted"):
        source.run()
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="needs pinned source resolution"):
        source.run(1500)
    assert snapshot(source.directory) == before
    monkeypatch.setattr(source.backend, "_prepare", source.acquire)
    source.run()
    assert source.acquisitions == 1


@pytest.mark.parametrize("corruption", ["audio", "conversion", "metadata", "missing_sidecar"])
def test_cross_budget_never_accepts_changed_pcm_provenance_or_payload(tmp_path, monkeypatch, corruption):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    result = source.run()
    audio = Path(result["source"]["directory"]) / result["rows"][0]["path"]
    sidecar = audio.with_suffix(audio.suffix + ".json")
    if corruption == "audio": audio.write_bytes(b"corrupt audio")
    if corruption == "conversion":
        saved = read_json(sidecar); saved["conversion"] = "another_algorithm"
        write_json(sidecar, saved)
    if corruption == "metadata": source.resolution_path.unlink()
    if corruption == "missing_sidecar": sidecar.unlink()
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="changed|missing|resolution"):
        source.run(1500)
    assert source.acquisitions == 1 and snapshot(source.directory) == before


def test_current_storage_cap_still_applies_to_reused_shard(tmp_path, monkeypatch):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    source.run()
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="storage bound"):
        source.run(1500, max_bytes=1)
    assert source.acquisitions == 1 and snapshot(source.directory) == before


@pytest.mark.parametrize("field", ["target_hours", "excluded_track_ids"])
def test_fma_selection_parameters_remain_strict_even_with_raw_flag(tmp_path, monkeypatch, field):
    source = RawShard(tmp_path / "part", "audioset_unbalanced", monkeypatch)
    request = {"candidate": {"component": "fma_full", "selection": {"attempt": 0}},
               "target_hours": 2, "excluded_track_ids": [], "seed": automatic_sources.SEED,
               "storage_policy": "canonical_flac_only_v1"}
    def run(value):
        return canonical_sources.prepare_canonical_source(source.directory, request=value,
            acquire=lambda: source.acquire(source.candidate, source.directory),
            canonicalize=canonical_audio.canonicalize_prepared_source, reuse_raw_shard=True)
    run(request)
    changed = {**request, field: 3 if field == "target_hours" else [9]}
    before = snapshot(source.directory)
    with pytest.raises(ValueError, match="another request"):
        run(changed)
    assert source.acquisitions == 1 and snapshot(source.directory) == before
