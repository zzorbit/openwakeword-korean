"""Preparation telemetry uses real copy bytes and preserves training results."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import korean_openwakeword.train as module
from korean_openwakeword.project import training_batch
from test_korean_library_shared_delivery import _equal, tiny_inputs


def test_training_progress_context_is_optional_nested_and_restored():
    outer, inner = [], []
    module._report_training_progress("unobserved", "not recorded")
    with module.training_progress(outer.append):
        module._report_training_progress("outer", "first")
        with pytest.raises(RuntimeError, match="fixture"):
            with module.training_progress(inner.append):
                module._report_training_progress("inner", "nested")
                raise RuntimeError("fixture")
        module._report_training_progress("outer", "restored")
        with module.training_progress(None):
            module._report_training_progress("unobserved", "temporarily disabled")
    module._report_training_progress("unobserved", "after scope")
    assert [row["label"] for row in outer] == ["first", "restored"]
    assert [row["label"] for row in inner] == ["nested"]
    assert set(outer[0]) == {"phase", "label", "done", "total", "unit"}
    with pytest.raises(ValueError, match="callable"):
        with module.training_progress("not a callback"):
            pytest.fail("invalid callback entered")


def _fake_torch(monkeypatch, free_bytes):
    allocations, writes = [], []

    class Tensor:
        def __init__(self, data):
            self.data = data

        def to(self, device):
            assert device == "cuda:0"
            return self

        def __setitem__(self, index, value):
            writes.append((self.data.shape, index.start, index.stop))
            self.data[index] = value.data

    def empty(shape, *, dtype, device):
        assert device == "cuda:0"
        value = Tensor(np.empty(shape, dtype=dtype))
        allocations.append(value)
        return value

    torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (free_bytes, free_bytes)),
        empty=empty, float16=np.float16, float32=np.float32,
        from_numpy=lambda data: Tensor(data))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return allocations, writes


def test_gpu_preparation_reports_only_selected_pool_bytes_and_copies_in_original_order(monkeypatch):
    # The largest and smallest pools fit; the middle pool stays on mmap.
    pools = {"large": np.arange(8200 * 2, dtype=np.float32).reshape(8200, 2),
             "middle": np.zeros((8200, 2), dtype=np.float16),
             "small": np.arange(2000 * 2, dtype=np.float16).reshape(2000, 2)}
    allocations, writes = _fake_torch(monkeypatch, 2 * 1024**3 + 75000)
    records = []
    with module.training_progress(records.append):
        delivery = module.InputDelivery(pools, "cuda:0", batch={role: 1 for role in pools})
    assert list(delivery.cached) == ["large", "small"]
    assert [array.data.shape for array in allocations] == [(8200, 2), (2000, 2)]
    assert [start for _, start, _ in writes] == [0, 4096, 8192, 0]
    assert np.array_equal(delivery.cached["large"].data, pools["large"])
    assert np.array_equal(delivery.cached["small"].data, pools["small"])
    assert [row["done"] for row in records] == [0, 32768, 65536, 65600, 73600]
    assert all(row["total"] == 73600 and row["unit"] == "bytes" for row in records)
    assert all(row["phase"] == "gpu_cache_loading" for row in records)
    assert delivery.report["middle"]["mode"] == "CPU_mmap_gather"


@pytest.mark.parametrize("device,mode", [("cpu", "auto"), ("cuda:0", "mmap")])
def test_uncached_delivery_does_not_claim_gpu_copy_progress(monkeypatch, device, mode):
    allocations, writes = _fake_torch(monkeypatch, 2 * 1024**3 + 75000)
    pools = {"positive": np.ones((3, 16, 96), dtype=np.float32)}
    records = []
    with module.training_progress(records.append):
        delivery = module.InputDelivery(pools, device, mode=mode, batch={"positive": 1})
    assert not allocations and not writes and not records and not delivery.cached


def test_preparation_callbacks_preserve_every_checkpoint_and_exported_model(tiny_inputs):
    import torch
    _, root, config, metadata, pools = tiny_inputs
    seed = config["training"]["seeds"][0]
    baseline_dir, observed_dir = root / "baseline", root / "observed"
    baseline = module._train_member(config, root, metadata, pools, seed, baseline_dir, True, False)
    records = []

    def observe(row):
        if row["phase"] == "training":
            # First update telemetry is written only after preparation ends.
            assert not (observed_dir / "progress.json").exists()
        records.append(deepcopy(row))

    with module.training_progress(observe):
        actual = module._train_member(config, root, metadata, pools, seed, observed_dir, True, False)
    assert [row["phase"] for row in records] == ["model_initialization", "training"]
    assert records[-1]["done"] == 0 and records[-1]["total"] == 4
    for field in ("model_sha256", "contract_id", "exposure", "delivery", "updates"):
        assert actual[field] == baseline[field]
    for update in config["training"]["checkpoints"]:
        left = torch.load(baseline_dir / f"checkpoint_{update:06d}.pt", weights_only=False)
        right = torch.load(observed_dir / f"checkpoint_{update:06d}.pt", weights_only=False)
        assert _equal(left, right), f"Preparation telemetry changed update {update}"


def test_train_reports_pool_verification_before_auto_device_selection(tiny_inputs, monkeypatch):
    import korean_openwakeword.benchmark as benchmark
    project, root, config, metadata, pools = tiny_inputs
    config = deepcopy(config)
    config["training"]["device"] = "auto"
    for role, array in pools.items():
        cache = root / config["pool_features"][role]
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "features.npy", array)
    monkeypatch.setattr(module, "load_project", lambda project: (root, config))
    monkeypatch.setattr(module, "verify_cache", lambda cache: metadata[next(
        role for role, relative in config["pool_features"].items() if root / relative == cache)])
    monkeypatch.setattr(benchmark, "doctor", lambda: {})
    monkeypatch.setattr(benchmark, "benchmark_training", lambda *a, **kw: {"fastest_measured": "cpu"})
    members = []
    monkeypatch.setattr(module, "_train_member", lambda *a: members.append(a) or {"seed": a[4]})
    records = []
    with module.training_progress(records.append):
        module.train(project, root / "runs/preparation", smoke=True)
    validation = [row for row in records if row["phase"] == "feature_validation"]
    assert [row["done"] for row in validation] == [0, 1, 1, 2, 2, 3, 3, 4]
    assert all(row["total"] == len(training_batch(config)) for row in validation)
    assert records[-1]["phase"] == "device_selection"
    assert [args[4] for args in members] == config["training"]["seeds"]
    assert all(args[8] == "cpu" for args in members)
