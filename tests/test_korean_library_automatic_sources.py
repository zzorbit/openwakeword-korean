"""Small fake-provider contracts; no real corpora, network or model training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from korean_openwakeword import automatic_sources as module
from korean_openwakeword.common import digest, read_json


class Backend:
    def __init__(self, counts=5):
        self.counts = counts
        self.inventories, self.prepared = [], []
        self.same_first_two = False

    def inventory(self, component):
        self.inventories.append(component)
        return [{"component": component, "key": f"{component}_{n}",
                 "selection": {"part": n}, "archive_bytes": 50}
                for n in range(self.counts)]

    def prepare(self, candidate, directory, **kwargs):
        self.prepared.append((deepcopy(candidate), kwargs))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        n = candidate["selection"]["part"]
        name = candidate["component"]
        content = f"{name}_{0 if self.same_first_two and n < 2 else n}".encode()
        original = directory / "original.wav"
        original.write_bytes(content)
        row = {"path": original.name, "sha256": digest(original),
               "canonical_source_id": f"fixture:{name}:{n}", "split": "train"}
        if name == "fma_full":
            row["track_id"] = n + 1
        return {"source": {"directory": str(directory), "adapter": "manifest"},
                "rows": [row], "reserved_download_bytes": 100,
                "original_storage_bytes": 100, "receipts": [], "artifacts": []}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    config = {"wakeword": "보라야", "project_id": "bora",
              "targets": {"positive_originals": 0}, "augmentation_rounds": 1}
    monkeypatch.setattr(module, "load_project", lambda path: (root, config))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    monkeypatch.setattr(module.sf, "info", lambda path: SimpleNamespace(frames=64000, samplerate=16000, channels=1))
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("No fixture may access network"))
    return root / "project.json", tmp_path / "workspace", config


def supply(setup, backend, *, preset="test", generic=2000, korean=2/3600, qualify=None):
    project, workspace, _ = setup
    return module.prepare_automatic_sources(project, workspace, preset=preset,
        generic_hours=generic, korean_hours=korean, backend=backend, qualify_source=qualify)


def test_offline_policy_and_exact_downstream_payload():
    policy = module.automatic_source_policy(7000, 3000, "production")
    assert policy["component_targets"] == {"audioset_unbalanced": 3000, "mls": 3000,
        "fma_full": 1000, "yodas2": 3000}
    space = module.automatic_storage_plan({}, generic_hours=7000, korean_hours=3000, preset="full")
    assert space["normalized_pcm16_bytes"] == 1_152_000_000_000
    assert space["negative_feature_payload_bytes"] == 110_592_000_000
    assert space["estimated_total_bytes"] > space["minimum_known_bytes"]
    assert module.automatic_source_policy(2000, 2000/3, "trial")["component_targets"] == {"yodas2": 2000/3}
    assert not module.automatic_source_policy(2000, 1, "check")["component_targets"]


def test_check_keeps_existing_bounded_recipe_without_provider_access(setup):
    backend = Backend()
    result = supply(setup, backend, preset="check", korean=1)
    assert not result["sources"] and not result["shortages"]
    assert backend.inventories == backend.prepared == []


def test_compact_storage_forecast_keeps_identical_data_targets_and_known_payload():
    old_policy = module.automatic_source_policy(7000, 3000, "production")
    current_policy = module.automatic_source_policy(7000, 3000, "production", material_acquisition_policy="automatic_v2")
    assert old_policy["component_targets"] == current_policy["component_targets"]
    assert old_policy["version"] == 1 and "storage_policy" not in old_policy
    assert current_policy["version"] == 2
    old = module.automatic_storage_plan({}, generic_hours=7000, korean_hours=3000)
    current = module.automatic_storage_plan({}, generic_hours=7000, korean_hours=3000,
                                            material_acquisition_policy="automatic_v2")
    for field in ("normalized_pcm16_bytes", "negative_feature_payload_bytes", "positive_feature_payload_bytes", "minimum_known_bytes"):
        assert current[field] == old[field]
    assert current["estimated_retained_source_bytes"] == 2_585_000_000_000
    assert current["estimated_download_bytes"] == 2_465_000_000_000
    assert current["transient_archive_bytes"] == 8 * 1024**3
    assert current["estimated_total_bytes"] < old["estimated_total_bytes"]
    assert current["estimated_original_archive_bytes"] == current["estimated_retained_source_bytes"]


def test_canonical_forecast_separates_feature_minimum_pcm_reference_and_real_peak():
    config = {"targets": {"positive_originals": 100000}, "augmentation_rounds": 5}
    previous = module.automatic_storage_plan(config, material_acquisition_policy="automatic_v2")
    current = module.automatic_storage_plan(config, material_acquisition_policy="automatic_v3")
    assert current["storage_policy"] == "canonical_flac_only_v1"
    assert current["normalized_pcm16_bytes"] == previous["normalized_pcm16_bytes"] == 1_152_000_000_000
    assert current["minimum_known_bytes"] == current["negative_feature_payload_bytes"] + current["positive_feature_payload_bytes"]
    assert current["canonical_flac_estimated_bytes"] is None
    assert current["estimated_retained_source_bytes"] == 0
    assert current["eliminated_retained_original_forecast_bytes"] == previous["estimated_retained_source_bytes"]
    assert current["estimated_download_bytes"] == previous["estimated_download_bytes"]
    assert current["transient_source_bytes"] == 125_000_000_000
    assert current["transient_source_bytes"] > previous["transient_archive_bytes"]
    assert current["estimated_total_bytes"] == (current["minimum_known_bytes"]
        + current["normalized_pcm16_bytes"] + current["future_buffer_bytes"] + current["transient_source_bytes"])


def test_canonical_dedup_uses_raw_identity_and_raw_header_duration(tmp_path):
    import numpy as np
    import soundfile as sf
    directory = tmp_path / "canonical"
    directory.mkdir()
    rows = []
    for number in range(2):
        file = directory / f"{number}.flac"
        sf.write(file, np.zeros(16000, dtype=np.int16), 16000, format="FLAC", subtype="PCM_16")
        rows.append({"path": file.name, "sha256": digest(file),
            "original_source_sha256": str(number) * 64,
            "canonical_source_id": f"fixture:{number}",
            "canonical_audio": {"original_info": {"input_frames": 44099,
                "input_sample_rate": 44100, "input_channels": 1}}})
    assert rows[0]["sha256"] == rows[1]["sha256"]
    selected = module._validate_rows({"directory": str(directory)}, rows,
        tmp_path / "selected", set(), set())
    assert len(selected) == 2
    assert all(row["duration_s"] == 44099 / 44100 for row in selected)
    assert not module._validate_rows({"directory": str(directory)}, rows,
        tmp_path / "selected_again", {row["original_source_sha256"] for row in rows}, set())


@pytest.mark.parametrize("qualify", [False, True])
def test_v3_backend_acquisition_canonical_selection_and_resume(tmp_path, monkeypatch, qualify):
    import numpy as np
    import soundfile as sf
    root, workspace = tmp_path / "project", tmp_path / "workspace"
    root.mkdir()
    config = {"wakeword": "보라야", "project_id": "bora",
              "targets": {"positive_originals": 0}, "augmentation_rounds": 1}
    if qualify:
        from korean_openwakeword.project import create_project
        from korean_openwakeword.common import write_json
        project = create_project(root, "보라야", "bora", positive_originals=1000,
            augmentation_rounds=1, updates=10000, generic_hours=2000, korean_hours=2 / 3600)
        config = read_json(project)
        config["audio_storage_policy"] = "canonical_flac_only_v1"
        write_json(project, config)
    monkeypatch.setattr(module, "load_project", lambda path: (root, config))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    class CanonicalBackend(module.PublicSourceBackend):
        acquisitions = 0
        def inventory(self, component):
            return [{"component": component, "key": "fixture", "archive_bytes": 1000}]
        def _prepare(self, candidate, directory, **kwargs):
            self.acquisitions += 1
            directory = Path(directory)
            raw = directory / "raw"
            raw.mkdir(parents=True)
            original = raw / "speech.wav"
            sf.write(original, np.random.default_rng(8).normal(0, .03, 44100 * 2), 44100, subtype="PCM_16")
            self.original = original
            self.original_sha = digest(original)
            return {"source": {"directory": str(raw), "adapter": "manifest"},
                "rows": [{"path": original.name, "sha256": self.original_sha,
                          "canonical_source_id": "fixture:speech", "split": "train",
                          "dataset_id": "procedural_fixture", "source_url": "https://example.test/speech",
                          "license": "test-fixture", "category": "competing_speech",
                          "category_basis": "explicit test fixture", "language": "ko", "speech": True}],
                "reserved_download_bytes": original.stat().st_size,
                "original_storage_bytes": original.stat().st_size, "receipts": [], "artifacts": []}
    backend = CanonicalBackend(workspace / "shared/automatic_sources_v3", storage_policy="canonical_flac_only_v1")
    kwargs = dict(preset="test", generic_hours=2000, korean_hours=2 / 3600,
                  backend=backend, material_acquisition_policy="automatic_v3")
    if qualify:
        from korean_openwakeword.preparation import prepare_public_audio
        def qualify_source(source):
            return prepare_public_audio(root / "project.json", input_directory=source["directory"],
                output=root / "prepared", source_index=source["source_index"],
                target_hours=source["target_hours"], validation_fraction=source["validation_fraction"])
        kwargs["qualify_source"] = qualify_source
    first = module.prepare_automatic_sources(root / "project.json", workspace, **kwargs)
    request = read_json(first["plan"]["request"]["path"])
    assert request["limits"]["max_download_bytes"] == 4 * 1024**4
    assert request["limits"]["max_download_bytes"] > request["limits"]["max_workspace_bytes"]
    source = first["sources"][0]
    row = module.read_rows(source["source_index"])[0]
    assert row["path"].endswith(".wav.flac")
    assert row["original_source_sha256"] == backend.original_sha
    assert not backend.original.exists()
    assert first["canonical_storage_bytes"] > 0 and first["retired_original_bytes"] > 0
    assert first["component_actual_hours"] == {"yodas2": 2 / 3600}
    if qualify:
        prepared = source["prepared"]
        admitted_candidates = module.read_rows(prepared["artifacts"]["source_manifest"]["path"])
        assert len(admitted_candidates) == 1
        candidate = admitted_candidates[0]
        assert candidate["path"].endswith(".flac")
        assert candidate["original_sha256"] == backend.original_sha
        assert candidate["sample_id"] == "public_" + module.identity(["fixture:speech", backend.original_sha])[:24]
        assert candidate["logical_audio_sha256"] == row["logical_audio_sha256"]
        assert not list((root / "prepared/audio").glob("*"))
    repeated = module.prepare_automatic_sources(root / "project.json", workspace, **kwargs)
    assert backend.acquisitions == 1
    assert repeated["component_actual_hours"] == first["component_actual_hours"]
    assert repeated["source_storage_bytes"] == first["source_storage_bytes"]
    (Path(source["directory"]) / row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.prepare_automatic_sources(root / "project.json", workspace, **kwargs)


def test_compact_policy_uses_separate_namespace_and_same_source_selection(setup):
    old_backend = Backend()
    old = supply(setup, old_backend)
    old_request = setup[0].parent / ".studio/materials/automatic_sources_v1/request.json"
    prior = old_request.read_bytes()
    compact_backend = Backend()
    compact = module.prepare_automatic_sources(setup[0], setup[1], preset="test",
        generic_hours=2000, korean_hours=2/3600, backend=compact_backend,
        material_acquisition_policy="automatic_v2")
    assert old_request.read_bytes() == prior
    assert "/automatic_sources_v2/" in compact["plan"]["request"]["path"].replace("\\", "/")
    assert old_backend.prepared[0][0] == compact_backend.prepared[0][0]
    assert compact["sources"][0]["id"] == old["sources"][0]["id"]
    assert compact["storage_policy"] == "verified_extracted_only_v1"
    assert compact["component_actual_hours"] == old["component_actual_hours"]


def test_compact_resume_rechecks_full_preservation_receipt_before_selected_hours(setup, monkeypatch):
    from korean_openwakeword import compact_sources
    class ProofBackend(Backend):
        def prepare(self, candidate, directory, **kwargs):
            result = super().prepare(candidate, directory, **kwargs)
            proof = Path(directory) / "compact_receipt.json"
            proof.write_text("{}\n", encoding="utf-8")
            return {**result, "compact_receipt": {"path": str(proof.resolve()), "sha256": digest(proof)},
                    "retired_archive_bytes": 50, "temporary_archive_peak_bytes": 50}
    backend = ProofBackend()
    verified = []
    monkeypatch.setattr(compact_sources, "verify_compact_source", lambda directory: verified.append(directory))
    kwargs = {"preset": "test", "generic_hours": 2000, "korean_hours": 2/3600,
              "backend": backend, "material_acquisition_policy": "automatic_v2"}
    first = module.prepare_automatic_sources(setup[0], setup[1], **kwargs)
    second = module.prepare_automatic_sources(setup[0], setup[1], **kwargs)
    assert len(backend.prepared) == len(verified) == 1
    assert second["component_actual_hours"] == first["component_actual_hours"]
    assert second["retired_archive_bytes"] == second["temporary_archive_peak_bytes"] == 50
    (verified[0] / "compact_receipt.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt changed"):
        module.prepare_automatic_sources(setup[0], setup[1], **kwargs)
    assert len(verified) == 1


def test_public_backend_compact_preserves_original_bytes_and_provider_rows(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    import tarfile
    from korean_openwakeword import commercial_sources
    from korean_openwakeword.common import identity
    provider = commercial_sources._PROVIDERS["audioset_unbalanced"]
    original = b"same original audio bytes; neither reencoding nor gain changes" * 8
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        info = tarfile.TarInfo("corpus/fixture0001.wav"); info.size = len(original)
        stream.addfile(info, io.BytesIO(original))
    payloads = {}
    def asset(path, data):
        url = commercial_sources._resolve_url(provider, path)
        payloads[url] = data
        return {"path": path, "filename": Path(path).name, "source_url": url,
                "size_bytes": len(data), "checksum_algorithm": "sha256",
                "sha256": hashlib.sha256(data).hexdigest(),
                "upstream_revision": provider["revision"], "license": provider["license"]}
    assets = [asset("data/unbal_train000.tar", archive.getvalue())]
    metadata = [asset("data/unbalanced_train_segments.csv", b'fixture0001,0,4,"/m/environment"\n'),
                asset("data/ontology.json", json.dumps([
                    {"id": key, "name": key, "child_ids": []}
                    for key in ("/m/09x0r", "/m/04rlf", "/m/environment")]).encode())]
    resolution = {"kind": "commercial_source_resolution_v1", "source_id": "audioset_unbalanced",
                  "provider": provider, "selection": {"parts": [0]}, "assets": assets,
                  "metadata_assets": metadata, "total_bytes": sum(a["size_bytes"] for a in assets + metadata),
                  "resolved_at": "fixture"}
    resolution["resolution_sha256"] = identity({k: v for k, v in resolution.items() if k != "resolved_at"})
    monkeypatch.setattr(commercial_sources, "resolve_commercial_source", lambda *a, **k: resolution)
    class Response:
        status_code = 200
        headers = {}
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield self.data
    monkeypatch.setattr(module.requests, "get", lambda url, **kwargs: Response(payloads[url]))
    candidate = {"component": "audioset_unbalanced", "selection": {"parts": [0]}}
    legacy_root, compact_root = tmp_path / "legacy", tmp_path / "compact"
    args = {"target_hours": 1, "max_download_bytes": resolution["total_bytes"], "max_extract_bytes": 1024**2}
    legacy = module.PublicSourceBackend(legacy_root).prepare(candidate, legacy_root / "parts/source", **args)
    compact = module.PublicSourceBackend(compact_root, storage_policy="verified_extracted_only_v1").prepare(
        candidate, compact_root / "parts/source", **args)
    for result in (legacy, compact):
        assert (Path(result["source"]["directory"]) / "part_0000/corpus/fixture0001.wav").read_bytes() == original
    def comparable(rows):
        return [{**row, "provider_metadata_bindings": [binding["sha256"] for binding in row["provider_metadata_bindings"]]} for row in rows]
    assert comparable(legacy["rows"]) == comparable(compact["rows"])
    assert legacy["original_storage_bytes"] - compact["original_storage_bytes"] == assets[0]["size_bytes"]
    assert (legacy_root / "parts/source/originals" / assets[0]["path"]).is_file()
    assert not (compact_root / "parts/source/originals" / assets[0]["path"]).exists()
    proof = Path(compact["compact_receipt"]["path"])
    proof_sha = digest(proof)
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: pytest.fail("Completed compact cache must not download"))
    resumed = module.PublicSourceBackend(compact_root, storage_policy="verified_extracted_only_v1").prepare(
        candidate, compact_root / "parts/source", **args)
    assert resumed["rows"] == compact["rows"] and digest(proof) == proof_sha


def test_each_full_component_refills_after_real_callback_shortfall(setup):
    backend = Backend()
    qualified = []
    def qualify(source):
        qualified.append(source["id"])
        assert source["target_hours"] is None and source["validation_fraction"] == 0
        return {"train_hours": 2/3600, "target_met": True, "fixture_post_qc": True}
    progress = []
    result = module.prepare_automatic_sources(setup[0], setup[1], preset="full",
        generic_hours=7/3600, korean_hours=3/3600, backend=backend,
        qualify_source=qualify, progress=progress.append)
    assert not result["shortages"]
    assert len(result["sources"]) == len(qualified) == 7
    assert all(source["prepared"]["fixture_post_qc"] for source in result["sources"])
    for source in result["sources"]:
        assert source["provider"] == source["component_id"] == source["component"]
        assert source["pools"] == (["korean_negative"] if source["provider"] == "yodas2" else ["generic_negative"])
    assert result["hours_scope"] == "post_technical_QC_Train"
    assert not result["content_approved"] and not result["training_admitted"] and not result["Final_accessed"]
    post_qc = [p for p in progress if "기술 검사 완료" in p["label"]]
    assert len(post_qc) == len(qualified) and all(p["done"] <= p["total"] + 4/3600 for p in post_qc)


def test_resume_reuses_frozen_selection_and_changed_audio_fails_closed(setup):
    backend = Backend()
    first = supply(setup, backend)
    request = Path(first["plan"]["request"]["path"])
    before = request.read_bytes()
    second = supply(setup, backend)
    assert len(backend.prepared) == 1
    assert second["component_actual_hours"] == first["component_actual_hours"]
    assert request.read_bytes() == before
    source = first["sources"][0]
    (Path(source["directory"]) / "original.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="waveform changed"):
        supply(setup, backend)


@pytest.mark.parametrize("policy", ["automatic_v1", "automatic_v2"])
def test_interrupted_selection_recounts_fallback_copies_before_source_commit(setup, monkeypatch, policy):
    backend = Backend()
    def prepare():
        return module.prepare_automatic_sources(setup[0], setup[1], preset="test", generic_hours=2000,
            korean_hours=2/3600, backend=backend, material_acquisition_policy=policy)
    frozen = module._frozen
    def interrupt_before_commit(path, record):
        if path.name == "source.json":
            raise InterruptedError("fixture: copied selection before source record commit")
        return frozen(path, record)
    def cannot_link(*args):
        raise OSError("fixture filesystem does not support hardlinks")
    monkeypatch.setattr(module.os, "link", cannot_link)
    monkeypatch.setattr(module, "_frozen", interrupt_before_commit)
    with pytest.raises(InterruptedError, match="before source record commit"):
        prepare()
    monkeypatch.setattr(module, "_frozen", frozen)
    result = prepare()
    source = result["sources"][0]
    selected = Path(source["directory"]) / "original.wav"
    assert result["source_storage_bytes"] == 100 + selected.stat().st_size
    assert selected.read_bytes() == b"yodas2_0"


@pytest.mark.parametrize("independent_copy", [False, True])
def test_existing_selection_distinguishes_hardlink_payload_from_copy_cost(setup, independent_copy):
    source_dir = setup[1] / "original"
    source_dir.mkdir(parents=True)
    original = source_dir / "example.wav"
    original.write_bytes(b"fixture original payload")
    selected = setup[1] / "selected"
    selected.mkdir()
    destination = selected / original.name
    if independent_copy:
        destination.write_bytes(original.read_bytes())
    else:
        module.os.link(original, destination)
    row = {"path": original.name, "sha256": digest(original), "canonical_source_id": "fixture:one"}
    counter = {"bytes": 0}
    module._validate_rows({"directory": str(source_dir)}, [row], selected, set(), set(),
                          storage_counter=counter, max_copy_bytes=original.stat().st_size)
    assert counter["bytes"] == (original.stat().st_size if independent_copy else 0)
    if independent_copy:
        with pytest.raises(ValueError, match="복사본"):
            module._validate_rows({"directory": str(source_dir)}, [row], selected, set(), set(),
                                  storage_counter={"bytes": 0}, max_copy_bytes=original.stat().st_size - 1)
        assert destination.read_bytes() == original.read_bytes()


def test_duplicates_do_not_replace_missing_unique_hours(setup):
    backend = Backend()
    backend.same_first_two = True
    result = supply(setup, backend, korean=6/3600)
    assert not result["shortages"]
    assert len(backend.prepared) == 3 and len(result["sources"]) == 2
    assert result["component_actual_hours"]["yodas2"] == pytest.approx(8/3600)


def test_bad_header_is_reported_and_next_part_refills(setup, monkeypatch):
    def info(path):
        if "yodas2_0" in str(path):
            raise RuntimeError("fixture bad audio header")
        return SimpleNamespace(frames=64000, samplerate=16000, channels=1)
    monkeypatch.setattr(module.sf, "info", info)
    backend = Backend()
    result = supply(setup, backend)
    assert len(backend.prepared) == 2 and not result["shortages"]
    rejected = setup[0].parent / ".studio/materials/automatic_sources_v1/parts/yodas2_0/source.json"
    assert read_json(rejected)["header_rejections"][0]["reason"] == "invalid_audio_header"


def test_all_technical_rejects_refill_but_arbitrary_callback_errors_stop(setup):
    backend = Backend()
    def qualify(source):
        if source["id"].endswith("_0"):
            raise ValueError("All sources failed technical checks; see checks records")
        return {"train_hours": 4/3600, "target_met": True}
    result = supply(setup, backend, qualify=qualify)
    assert len(backend.prepared) == 2 and len(result["sources"]) == 1
    assert not result["shortages"]
    assert any(path.endswith("qualification_failed.json") for path in result["artifacts"])
    def broken(source):
        raise ValueError("provider identity changed")
    with pytest.raises(ValueError, match="identity changed"):
        supply(setup, backend, qualify=broken)


def test_workspace_cap_is_cumulative_before_next_part_transfer(setup):
    space = module.automatic_storage_plan(setup[2], generic_hours=2000, korean_hours=6/3600, preset="test")
    setup[2]["automatic_data_limits"] = {"max_workspace_bytes": space["minimum_known_bytes"] + 150,
                                       "max_download_bytes": 10000}
    backend = Backend()
    result = supply(setup, backend, korean=6/3600)
    assert len(backend.prepared) == 1
    assert result["source_storage_bytes"] == 100
    assert result["shortages"] and any("누적 저장 한도" in text for text in result["shortages"])


def test_download_budget_does_not_reset_for_each_part(setup):
    setup[2]["automatic_data_limits"] = {"max_download_bytes": 100,
                                       "max_workspace_bytes": 10**12}
    backend = Backend()
    result = supply(setup, backend, korean=6/3600)
    assert len(backend.prepared) == 1 and result["actual_download_bytes"] == 100
    assert result["shortages"]


def test_total_known_payload_is_checked_before_any_inventory_or_download(setup, monkeypatch):
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1_248_000_000_000))
    backend = Backend()
    with pytest.raises(ValueError, match="정규화 음원.*최소"):
        supply(setup, backend, preset="full", generic=7000, korean=3000)
    assert backend.inventories == backend.prepared == []
    preflight = setup[0].parent / ".studio/materials/automatic_sources_v1/space_preflight.json"
    assert read_json(preflight)["audio_download_started"] is False


def test_changed_target_and_changed_index_preserve_previous_data(setup):
    backend = Backend()
    result = supply(setup, backend)
    with pytest.raises(ValueError, match="request changed"):
        supply(setup, backend, korean=3/3600)
    index = Path(result["sources"][0]["source_index"])
    index.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index changed"):
        supply(setup, backend)


def test_exhausted_provider_remains_short_instead_of_lowering_target(setup):
    backend = Backend(counts=1)
    result = supply(setup, backend, korean=10/3600)
    assert result["shortages"] and len(backend.prepared) == 1
    assert result["component_actual_hours"]["yodas2"] == pytest.approx(4/3600)


def test_default_inventory_has_pinned_routes_and_bounded_mls_batches(tmp_path, monkeypatch):
    import korean_openwakeword.commercial_sources as providers
    calls = []
    def tree(provider, path, session):
        calls.append((provider["revision"], path))
        if path == "data":
            return [{"type": "file", "path": "data/unbal_train001.tar", "size": 100}]
        if "mls_" in path:
            return [{"type": "file", "path": f"{path}/{n}_2_000.tar.gz", "size": 100} for n in range(17)]
        return [{"type": "file", "path": path + "/00000000.tar.gz", "size": 100}]
    monkeypatch.setattr(providers, "_tree", tree)
    backend = module.PublicSourceBackend(tmp_path)
    assert backend.inventory("audioset_unbalanced")[0]["selection"] == {"parts": [1]}
    korean = backend.inventory("yodas2")
    assert {p["selection"]["subset"] for p in korean} == set(module._KOREAN)
    mls = backend.inventory("mls")
    assert len(mls) == 14 and all(1 <= len(p["selection"]["parts"]) <= 16 for p in mls)
    assert {p["selection"]["language"] for p in mls} == set(module._LANGUAGES)
    assert len({revision for revision, _ in calls}) == 3


@pytest.mark.parametrize("row", [
    {"path": "../escape.wav", "canonical_source_id": "fixture:escape"},
    {"path": "original.wav", "canonical_source_id": "fixture:final", "split": "final"},
])
def test_unsafe_or_final_rows_are_not_selected(tmp_path, row):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "original.wav").write_bytes(b"not decoded")
    with pytest.raises(ValueError, match="escapes|held-out"):
        module._validate_rows({"directory": str(directory)}, [row], tmp_path / "selected", set(), set())
