"""Explicit training loop, immutable inputs, bounded caches and resumable state.

No experimental AST rewriting or machine-specific paths are used by this API.
The selected CNN, sampler, dropout hook, optimizer and schedule math are retained.
"""
import copy
import math
import random
import time
from pathlib import Path
import numpy as np

from .common import digest, identity, now, read_json, resolved, write_json, append_row, writer_lock
from .features import verify_cache
from .project import BATCH, LEGACY_BATCH, training_batch, load_project, workload_summary
from .storage import project_storage


def validate_training_scale(config, metadata):
    """A chosen budget is a real experiment, not automatically a smoke test.

    Keep input admission and evaluation requirements independent of compute
    budget. R comparisons must hold the selected original pool fixed; 100k is
    the historical reference workload, not a hard-coded eligibility rule.
    """
    positive = metadata["positive"]
    target, rounds = config["targets"]["positive_originals"], config["augmentation_rounds"]
    if positive["original_count"] != target or positive["rounds"] != rounds or positive["feature_rows"] != target * rounds:
        raise ValueError("Positive cache must match the selected TTS original count and N×rounds; use a separate matching configuration")
    for role, target_hours in (("generic_negative", config["targets"]["generic_hours"]),
                               ("korean_negative", config["targets"]["korean_hours"])):
        m = metadata[role]
        hours = m.get("publisher_declared_hours") if m.get("legacy_generic_bridge") else m.get("unique_source_hours")
        if hours is None or not math.isfinite(hours) or hours < target_hours:
            raise ValueError(f"{role} scale is below {target_hours}h (publisher nominal must remain distinct)")


class PoolSampler:
    def __init__(self, sizes, seed, batch=None):
        self.batch = dict(batch if batch is not None else LEGACY_BATCH if set(sizes) == set(LEGACY_BATCH) else BATCH)
        if set(sizes) != set(self.batch) or not sizes or min(sizes.values()) < 1:
            raise ValueError("Every declared batch role needs a nonempty feature pool")
        self.sizes = dict(sizes)
        self.rngs = {r: np.random.default_rng(np.random.SeedSequence([seed, i])) for i, r in enumerate(self.batch)}
        self.draws = dict.fromkeys(self.batch, 0)
        self.seen = {r: np.zeros(n, bool) for r, n in sizes.items()}

    def next_indices(self):
        result = {}
        for r, count in self.batch.items():
            idx = self.rngs[r].integers(self.sizes[r], size=count)
            self.draws[r] += count
            self.seen[r][idx] = True
            result[r] = idx
        return result

    def state(self):
        return {"rng": {r: copy.deepcopy(g.bit_generator.state) for r, g in self.rngs.items()},
                "draws": dict(self.draws), "seen": {r: x.copy() for r, x in self.seen.items()}}

    def restore(self, state):
        self.draws = dict(state["draws"])
        for r in self.batch:
            self.rngs[r].bit_generator.state = state["rng"][r]
            self.seen[r][:] = state["seen"][r]


def build_classifier(name="cnn128", dropout=.1):
    import torch
    from torch import nn
    if name.startswith("cnn"):
        width = int(name[3:])
        class TimeChannels(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)
        model = nn.Sequential(TimeChannels(), nn.Conv1d(96, width, 5, padding=2), nn.GroupNorm(8, width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, 1))
    elif name.startswith("mlp"):
        width = int(name[3:])
        model = nn.Sequential(nn.Flatten(), nn.Linear(1536, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))
    else:
        raise ValueError("Unknown stable classifier")
    if dropout:
        final = next(m for m in reversed(list(model.modules())) if isinstance(m, nn.Linear))
        def drop(module, inputs):
            return (torch.nn.functional.dropout(inputs[0], p=dropout, training=module.training),)
        final.register_forward_pre_hook(drop)
    return model


def apply_backend(settings):
    import torch
    torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = settings["CUDA_matmul_TF32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_TF32"]
    torch.backends.mha.set_fastpath_enabled(settings["MHA_fastpath"])


def lr_at(step, settings):
    if settings["schedule"] == "constant":
        return settings["learning_rate"]
    total = settings["schedule_horizon"]
    warm = max(1, int(total * .02))
    factor = step / warm if step <= warm else .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    return settings["learning_rate"] * factor


class InputDelivery:
    """Preserve stored dtype on GPU, convert sampled rows to float32 exactly.

    Large pools are copied in bounded sequential chunks only if they fit the
    measured budget. Uncached mmap pools use the original CPU gather path.
    """
    def __init__(self, pools, device, budget_mb=0, mode="auto", batch=None):
        import torch
        self.batch_counts = dict(batch if batch is not None else LEGACY_BATCH if set(pools) == set(LEGACY_BATCH) else BATCH)
        self.pools, self.device, self.cached = pools, device, {}
        budget = 0
        if device.startswith("cuda") and mode != "mmap":
            free, _ = torch.cuda.mem_get_info(device)
            safe = max(0, free - 2 * 1024**3)
            budget = min(safe, budget_mb * 1024**2) if budget_mb else safe
        for role in sorted(pools, key=lambda r: pools[r].nbytes, reverse=True):
            array = pools[role]
            if array.nbytes <= budget:
                cached = torch.empty(array.shape, dtype=torch.float16 if array.dtype == np.float16 else torch.float32, device=device)
                for i in range(0, len(array), 4096):
                    cached[i:i+4096] = torch.from_numpy(np.array(array[i:i+4096], copy=True)).to(device)
                self.cached[role] = cached
                budget -= array.nbytes
        self.report = {r: {"mode": "GPU_pool_cache" if r in self.cached else "CPU_mmap_gather",
                          "rows": len(a), "storage_bytes": a.nbytes} for r, a in pools.items()}

    def batch(self, indices):
        import torch
        xs, ys = [], []
        for role, count in self.batch_counts.items():
            if role in self.cached:
                x = self.cached[role][torch.as_tensor(indices[role], device=self.device)].float()
            else:
                x = torch.from_numpy(np.asarray(self.pools[role][indices[role]], dtype=np.float32)).to(self.device)
            xs.append(x)
            ys.append(torch.full((count, 1), float(role == "positive"), device=self.device))
        return torch.cat(xs), torch.cat(ys)


    def validate_member(self, pools, device, batch):
        """Reuse only the same verified array objects, device and role order."""
        if (self.device != device or list(self.batch_counts.items()) != list(batch.items())
                or set(self.pools) != set(pools)
                or any(self.pools[role] is not pools[role] for role in pools)):
            raise ValueError("Shared input delivery needs identical verified pools, device and batch order")


class _SharedInputDelivery:
    """Invocation-local cache, constructed after the first unfinished model.

    InputDelivery's usual free-VRAM measurement and 2 GiB model/optimizer
    reserve stay unchanged. Constructing after model allocation also avoids
    consuming memory for a completed-only resume. No sampler or RNG is shared.
    """
    def __init__(self, pools, settings):
        self.pools, self.settings = pools, settings
        self.value, self.creations, self.member_uses = None, 0, 0

    def __call__(self, device, batch):
        created = self.value is None
        if created:
            self.value = InputDelivery(self.pools, device,
                self.settings.get("gpu_cache_budget_mb", 0), self.settings.get("delivery", "auto"), batch)
            self.creations += 1
        self.value.validate_member(self.pools, device, batch)
        self.member_uses += 1
        self.value._member_reuse_receipt = {
            "scope": "Same immutable verified pools within this train invocation only",
            "cache_constructions_this_invocation": self.creations,
            "member_uses_this_invocation": self.member_uses,
            "created_for_this_member": created,
            "sampler_and_random_states_shared": False,
        }
        return self.value


def _train_member(config, root, metadata, pools, seed, output, smoke, resume,
                  selected_device=None, delivery=None):
    import torch
    settings = config["training"]
    device = settings["device"]
    if device == "auto":
        device = selected_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install CUDA PyTorch or explicitly select CPU")
    torch.set_num_threads(settings["cpu_threads"])
    apply_backend(settings["backend"])
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_classifier(settings["architecture"], settings["dropout"]).to(device)
    optimizer = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[settings["optimizer"]](model.parameters(),
        lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    batch = training_batch(config)
    sampler = PoolSampler({r: len(a) for r, a in pools.items()}, seed, batch)
    versions = {"torch": torch.__version__, "numpy": np.__version__}
    contract = {"schema_version": 1, "config": config, "seed": seed, "smoke_test_only": smoke,
        "metadata": metadata, "device": device, "versions": versions, "implementation_sha256": digest(__file__)}
    contract_id = identity(contract)
    with writer_lock(output):
        prior = output / "contract.json"
        if prior.exists():
            if not resume or read_json(prior) != contract:
                raise ValueError("Existing training output or changed resume contract")
            if (output / "receipt.json").exists():
                receipt = read_json(output / "receipt.json")
                if digest(output / "model.onnx") != receipt["model_sha256"]:
                    raise ValueError("Completed model changed")
                return receipt
        else:
            write_json(prior, contract)
        # The lazy shared factory is invoked only after contract/completion
        # checks and this member's model allocation. Cache copies use no RNG;
        # every member retains its own seeded model, optimizer and sampler.
        if delivery is None:
            delivery = InputDelivery(pools, device, settings.get("gpu_cache_budget_mb", 0), settings.get("delivery", "auto"), batch)
        elif callable(delivery):
            delivery = delivery(device, batch)
        delivery.validate_member(pools, device, batch)
        delivery_reuse = dict(getattr(delivery, "_member_reuse_receipt", {
            "scope": "Single member or explicit caller-owned delivery",
            "sampler_and_random_states_shared": False,
        }))
        step = 0
        if resume and (output / "last_checkpoint.json").exists():
            record = read_json(output / "last_checkpoint.json")
            path = output / record["path"]
            if digest(path) != record["sha256"]:
                raise ValueError("Resume checkpoint changed")
            # Only load this run's own hash-bound trusted checkpoint, never arbitrary downloaded pickle.
            state = torch.load(path, map_location=device, weights_only=False)
            if state["contract_id"] != contract_id:
                raise ValueError("Checkpoint belongs to a different experiment")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            sampler.restore(state["sampler"]); step = state["step"]
            torch.set_rng_state(state["torch_rng"].cpu()); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
            if device.startswith("cuda"):
                torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        started = time.perf_counter()
        write_json(output / "progress.json", {"state": "running", "pid": __import__("os").getpid(),
            "started_at": now(), "completed_updates": step, "total_updates": settings["updates"], "device": device})
        for step in range(step + 1, settings["updates"] + 1):
            x, y = delivery.batch(sampler.next_indices())
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, settings)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            elements = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = elements.mean() if settings["positive_loss_weight"] == 1 else (elements * (1 + (settings["positive_loss_weight"] - 1) * y)).mean()
            loss.backward(); optimizer.step()
            if step == 1 or step % settings["log_every"] == 0 or step == settings["updates"]:
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(loss_value):
                    raise ValueError("Nonfinite training loss")
                row = {"update": step, "loss": loss_value, "elapsed_s": time.perf_counter() - started,
                       "learning_rate": optimizer.param_groups[0]["lr"], "draws": dict(sampler.draws)}
                append_row(output / "training.jsonl", row)
                write_json(output / "progress.json", {"state": "running", **row, "total_updates": settings["updates"]})
                print(f"seed {seed}: {step}/{settings['updates']} loss={loss_value:.6f}", flush=True)
            if step in settings["checkpoints"] or step == settings["updates"]:
                path = output / f"checkpoint_{step:06d}.pt"
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state(),
                    "step": step, "contract_id": contract_id, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
                temporary = path.with_suffix(".tmp")
                torch.save(state, temporary); temporary.replace(path)
                write_json(output / "last_checkpoint.json", {"path": path.name, "sha256": digest(path), "update": step})
        model.eval().cpu()
        scoring = torch.nn.Sequential(model, torch.nn.Sigmoid()).eval()
        torch.onnx.export(scoring, torch.zeros(1, 16, 96), str(output / "model.onnx"), input_names=["input"],
            output_names=["wake_score"], dynamic_axes={"input": {0: "batch"}, "wake_score": {0: "batch"}},
            opset_version=17, dynamo=False)
        import onnxruntime as ort
        probe = np.random.default_rng(seed).normal(size=(8, 16, 96)).astype(np.float32)
        actual = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"]).run(None, {"input": probe})[0]
        with torch.no_grad():
            expected = scoring(torch.from_numpy(probe)).numpy()
        if not np.allclose(actual, expected, atol=1e-5, rtol=1e-4):
            raise ValueError("ONNX export mismatch")
        exposure = {r: {"drawn": sampler.draws[r], "distinct_feature_rows": int(sampler.seen[r].sum()),
            "distinct_originals": len(np.unique(np.flatnonzero(sampler.seen[r]) // metadata[r].get("rounds", 1))),
            "pool_rows": len(pools[r]), "unique_source_hours": metadata[r].get("unique_source_hours")} for r in batch}
        receipt = {"state": "smoke_trained_not_performance_validated" if smoke else "trained_not_final_evaluated",
            "model_sha256": digest(output / "model.onnx"), "contract_id": contract_id,
            "seed": seed, "updates": step, "exposure": exposure, "delivery": delivery.report,
            "input_delivery_reuse": delivery_reuse,
            "device": device, "compute_precision": "float32 with recorded backend; no AMP",
            "elapsed_this_invocation_s": time.perf_counter() - started,
            "onnx_max_absolute_export_error": float(np.max(np.abs(actual - expected))), "completed_at": now()}
        receipt["selected_workload"] = workload_summary(config)
        write_json(output / "receipt.json", receipt)
        write_json(output / "progress.json", {"state": "completed", "completed_updates": step})
        return receipt


@project_storage
def train(project, output=None, smoke=False, resume=False):
    root, config = load_project(project)
    workload = workload_summary(config)
    name = f"{config['project_id']}_{config['training']['architecture']}_R{config['augmentation_rounds']}"
    if workload["workload_kind"] != "reference_100k_R5_50k" or workload["model_count"] != 3:
        name += f"_N{workload['positive_tts_originals']}_U{workload['updates_per_model']}_M{workload['model_count']}"
    directory = Path(output).resolve() if output else root / "runs" / name
    metadata, pools = {}, {}
    for role in training_batch(config):
        cache = resolved(root, config["pool_features"][role])
        meta = verify_cache(cache)
        if meta.get("role") != role or meta.get("split") != "train" or not meta.get("split_audit_passed"):
            raise ValueError("Wrong pool role/split/audit")
        if not smoke and (meta.get("smoke_test_only") or meta.get("quality_review") != "accepted"):
            raise ValueError("Formal training refuses smoke or unapproved feature inputs")
        metadata[role] = meta
        pools[role] = np.load(resolved(cache, meta.get("feature_path", "features.npy")), mmap_mode="r", allow_pickle=False)
    ids = {m["frontend_id"] for m in metadata.values() if not m.get("legacy_generic_bridge")}
    if len(ids) != 1:
        raise ValueError("Feature pools use different frontend identities")
    if metadata["generic_negative"].get("legacy_generic_bridge"):
        from .frontend import PINNED_ASSETS
        from .assets import GENERIC_SHA
        bridge = metadata["generic_negative"]
        canonical = {"melspectrogram": PINNED_ASSETS["melspectrogram.onnx"], "embedding": PINNED_ASSETS["embedding_model.onnx"]}
        if bridge.get("canonical_frontend_assets") != canonical or bridge["feature_sha256"] != GENERIC_SHA or bridge.get("scanned_rows") != 5625000:
            raise ValueError("Legacy Generic bridge does not match the qualified publisher/frontend binding")
    positive = metadata["positive"]
    if positive["rounds"] != config["augmentation_rounds"]:
        raise ValueError("Positive cache rounds differ from the declared experiment")
    if not smoke:
        validate_training_scale(config, metadata)
    receipts = []
    selected_device = None
    if config["training"]["device"] == "auto":
        from .benchmark import benchmark_training, doctor
        resource = doctor()
        if "GPU" in resource:
            resource["GPU"].pop("free_bytes", None)
        probe_key = identity([resource, config["training"]["architecture"], training_batch(config), digest(__file__)])[:16]
        probe_path = root / "reports" / f"resource_probe_{probe_key}.json"
        prior = directory / f"member1_seed{config['training']['seeds'][0]}" / "contract.json"
        if resume and prior.exists():
            selected_device = read_json(prior)["device"]
        else:
            probe = read_json(probe_path) if probe_path.exists() else benchmark_training(probe_path, steps=10, architecture=config["training"]["architecture"], batch=training_batch(config))
            selected_device = probe["fastest_measured"]
    shared_delivery = _SharedInputDelivery(pools, config["training"])
    for index, seed in enumerate(config["training"]["seeds"], 1):
        receipts.append(_train_member(config, root, metadata, pools, seed,
            directory / f"member{index}_seed{seed}", smoke, resume, selected_device, shared_delivery))
    return {"directory": str(directory), "members": receipts, "formal_training": not smoke,
            "selected_workload": workload_summary(config),
            "input_delivery_reuse": {"scope": "This invocation only; exact pools/device/batch; independent member RNGs",
                                     "cache_constructions": shared_delivery.creations,
                                     "unfinished_members_using_delivery": shared_delivery.member_uses},
            "next": "Development evaluation and threshold selection; then freeze and independent Final"}
