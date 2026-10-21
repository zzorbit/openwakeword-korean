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
