"""One immutable input cache, independent seed/update results for each member."""
import copy
import hashlib

import numpy as np
import pytest

from korean_openwakeword.project import create_project, load_project, training_batch
import korean_openwakeword.train as module


@pytest.fixture
def tiny_inputs(tmp_path, monkeypatch, request):
    import torch
    device = getattr(request, "param", "cpu")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("Actual CUDA delivery parity requires an available GPU")
    else:
        # Other toy cases stay CPU-only, even on workstations with a GPU.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    project = create_project(tmp_path / "toy", "보라야", "bora", positive_originals=7,
                             augmentation_rounds=1, updates=4)
    root, config = load_project(project)
    config = copy.deepcopy(config)
    # Internal loop fixture only: do not alter the public baseline or a real project.
    config["batch"] = {role: 2 for role in config["batch"]}
    config["training"].update(device=device, cpu_threads=1, updates=4, schedule_horizon=4,
                              checkpoints=[2, 4], log_every=2, seeds=[41, 42, 43])
    pools = {role: np.random.default_rng(i).normal(size=(7, 16, 96)).astype(
        np.float16 if role == "generic_negative" else np.float32)
        for i, role in enumerate(training_batch(config))}
    metadata = {role: {"role": role, "split": "train", "split_audit_passed": True,
                       "frontend_id": "CPU-only toy", "rounds": 1, "feature_rows": 7}
                for role in pools}
    return project, root, config, metadata, pools


def _equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


@pytest.mark.parametrize("tiny_inputs", ["cpu", "cuda:0"], indirect=True)
def test_shared_delivery_preserves_every_batch_model_optimizer_sampler_and_rng_bit(tiny_inputs, monkeypatch):
    import torch
    _, root, config, metadata, pools = tiny_inputs
    captures = {"separate": [], "shared": []}
    phase = "separate"
    original_batch = module.InputDelivery.batch
    def capture_batch(self, indices):
        x, y = original_batch(self, indices)
        captures[phase].append({
            "indices": {role: value.copy() for role, value in indices.items()},
            "float32_input_and_labels_sha256": hashlib.sha256(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes()).hexdigest(),
        })
        assert x.dtype == torch.float32
        return x, y
    monkeypatch.setattr(module.InputDelivery, "batch", capture_batch)
    separate = []
    for index, seed in enumerate(config["training"]["seeds"], 1):
        separate.append(module._train_member(config, root, metadata, pools, seed,
            root / f"separate{index}", True, False))
    phase = "shared"
    shared = module._SharedInputDelivery(pools, config["training"])
    together = []
    for index, seed in enumerate(config["training"]["seeds"], 1):
        together.append(module._train_member(config, root, metadata, pools, seed,
            root / f"shared{index}", True, False, delivery=shared))
    assert shared.creations == 1 and shared.member_uses == 3
    assert shared.value.device == config["training"]["device"]
    if shared.value.device.startswith("cuda"):
        assert set(shared.value.cached) == set(pools)
        assert all(value.device.type == "cuda" for value in shared.value.cached.values())
    assert _equal(captures["separate"], captures["shared"])
    assert len(captures["shared"]) == 12
    for index, (left, right) in enumerate(zip(separate, together), 1):
        assert left["model_sha256"] == right["model_sha256"]
        assert left["exposure"] == right["exposure"]
        assert left["contract_id"] == right["contract_id"]
        assert left["delivery"] == right["delivery"]
        assert right["input_delivery_reuse"]["created_for_this_member"] == (index == 1)
        assert right["input_delivery_reuse"]["member_uses_this_invocation"] == index
        for update in (2, 4):
            a = torch.load(root / f"separate{index}" / f"checkpoint_{update:06d}.pt", weights_only=False)
            b = torch.load(root / f"shared{index}" / f"checkpoint_{update:06d}.pt", weights_only=False)
            assert _equal(a, b), f"Seed {left['seed']}, update {update} changed"


def test_train_constructs_one_delivery_for_three_members_and_zero_for_completed_resume(tiny_inputs, monkeypatch):
    _, root, config, metadata, pools = tiny_inputs
    for role, array in pools.items():
        cache = root / config["pool_features"][role]
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "features.npy", array)
    monkeypatch.setattr(module, "load_project", lambda project: (root, config))
    monkeypatch.setattr(module, "verify_cache", lambda cache: metadata[next(
        role for role, relative in config["pool_features"].items() if root / relative == cache)])
    original_init = module.InputDelivery.__init__
    constructions = []
    def count_init(self, *args, **kwargs):
        constructions.append(args[1])
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(module.InputDelivery, "__init__", count_init)
    output = root / "runs/shared"
    result = module.train(root / "project.json", output, smoke=True)
    assert constructions == ["cpu"]
    assert result["input_delivery_reuse"]["cache_constructions"] == 1
    assert result["input_delivery_reuse"]["unfinished_members_using_delivery"] == 3
    resumed = module.train(root / "project.json", output, smoke=True, resume=True)
    assert constructions == ["cpu"]
    assert resumed["input_delivery_reuse"]["cache_constructions"] == 0
    assert resumed["input_delivery_reuse"]["unfinished_members_using_delivery"] == 0
    assert [r["model_sha256"] for r in result["members"]] == [r["model_sha256"] for r in resumed["members"]]


@pytest.mark.parametrize("changed", ["pool", "device", "counts", "order"])
def test_delivery_reuse_rejects_different_pool_device_or_batch(tiny_inputs, changed):
    _, _, config, _, pools = tiny_inputs
    batch = training_batch(config)
    shared = module._SharedInputDelivery(pools, config["training"])
    delivery = shared("cpu", batch)
    other_pools, other_device, other_batch = pools, "cpu", batch
    if changed == "pool":
        other_pools = {**pools, "positive": pools["positive"].copy()}
    elif changed == "device":
        other_device = "cuda:0"
    elif changed == "counts":
        other_batch = {**batch, "positive": batch["positive"] + 1}
    else:
        other_batch = dict(reversed(list(batch.items())))
    with pytest.raises(ValueError, match="identical verified pools"):
        delivery.validate_member(other_pools, other_device, other_batch)


def test_lazy_factory_does_not_create_or_change_rng_until_used(tiny_inputs):
    import torch
    _, _, config, _, pools = tiny_inputs
    before = torch.get_rng_state().clone()
    shared = module._SharedInputDelivery(pools, config["training"])
    assert shared.value is None and shared.creations == 0
    shared("cpu", training_batch(config))
    assert torch.equal(before, torch.get_rng_state())
