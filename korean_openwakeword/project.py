"""Portable projects. All default paths are relative to the project file."""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import re
import math

from .common import read_json, write_json, resolved

BATCH = {"generic_negative": 768, "korean_negative": 256, "positive": 50}
SPLITS = ("train", "validation", "final")
ADVANCED_TRAINING_KEYS = frozenset({
    "architecture", "optimizer", "learning_rate", "weight_decay", "dropout",
    "positive_loss_weight", "schedule", "schedule_horizon", "checkpoints", "seeds",
    "device", "delivery", "gpu_cache_budget_mb", "cpu_threads", "log_every", "backend",
})


def default_profile():
    return read_json(files("korean_openwakeword") / "resources" / "default_profile.json")


def configure_budget(config, *, positive_originals=None, augmentation_rounds=None,
                     updates=None, training_overrides=None):
    """Return a new configuration; never edit a running/historical experiment.

    Workload choices and stable learning settings are separate. Changing the
    update budget also adjusts the default schedule and checkpoint positions.
    An explicit advanced horizon/checkpoint list can override this for a
    controlled comparison using a shared schedule.
    """
    result = deepcopy(config)
    previous_n = result["targets"]["positive_originals"]
    previous_r = result["augmentation_rounds"]
    if positive_originals is not None:
        result["targets"]["positive_originals"] = positive_originals
    if augmentation_rounds is not None:
        result["augmentation_rounds"] = augmentation_rounds
    t = result["training"]
    if updates is not None:
        if type(updates) is not int or updates < 1:
            raise ValueError("Updates per model must be a positive integer")
        t["updates"] = t["schedule_horizon"] = updates
        t["checkpoints"] = sorted({max(1, updates // 5), max(1, updates * 2 // 5), updates})
    overrides = training_overrides or {}
    if set(overrides) - ADVANCED_TRAINING_KEYS:
        raise ValueError("Unsupported advanced training setting: " + str(sorted(set(overrides) - ADVANCED_TRAINING_KEYS)))
    t.update(deepcopy(overrides))
    n, r = result["targets"]["positive_originals"], result["augmentation_rounds"]
    if n != previous_n:
        # Avoid colliding with an immutable cache built for another N.
        result["pool_features"]["positive"] = f"features/positive_N{n}_R{r}"
    elif r != previous_r:
        result["pool_features"]["positive"] = f"features/positive_N{n}_R{r}"
    return validate_project(result)


def workload_summary(config):
    """Exact counters/storage payload, without guessing a computer's ETA."""
    n, r = config["targets"]["positive_originals"], config["augmentation_rounds"]
    t = config["training"]
    members, rows = len(t["seeds"]), n * r
    feature_bytes = rows * 16 * 96 * 4  # local positive cache is float32
    return {
        "positive_tts_originals": n, "augmentation_rounds": r,
        "positive_feature_rows": rows, "positive_feature_payload_bytes": feature_bytes,
        "positive_feature_payload_gib": round(feature_bytes / 1024**3, 3),
        "updates_per_model": t["updates"], "model_count": members,
        "total_optimizer_updates": t["updates"] * members,
        "positive_draws_per_model": t["updates"] * BATCH["positive"],
        "mean_positive_draws_per_feature_per_model": round(t["updates"] * BATCH["positive"] / rows, 3),
        "schedule_horizon": t["schedule_horizon"],
        "warmup_updates": max(1, int(t["schedule_horizon"] * .02)) if t["schedule"] == "cosine_with_2percent_warmup" else 0,
        "checkpoints": list(t["checkpoints"]), "architecture": t["architecture"],
        "device": t["device"],
        "workload_kind": "reference_100k_R5_50k" if (n, r, t["updates"]) == (100000, 5, 50000) else "user_selected_budget",
        "performance_status": "requires_this_word_development_and_independent_final",
        "notes": [
            "Storage is only the positive float32 feature payload, excluding WAVs, journals, Negative caches and checkpoints.",
            "Draws use replacement sampling; a mean draw count does not guarantee every feature was seen.",
            "Validation/Final targets and Negative quality requirements are unchanged by a smaller training budget.",
            "Architecture defaults are evidence from Chunsik, not a universal Korean optimum. Workload changes can change performance.",
            "ETA requires measured throughput on this computer; init does not run generation or training.",
        ],
    }


def validate_project(config):
    word = config.get("wakeword", "")
    if not isinstance(word, str) or not re.fullmatch(r"[가-힣]+", word):
        raise ValueError("wakeword must be a standalone Korean Hangul word/phrase without a command")
    if config.get("schema_version") != 1 or config.get("language") != "Korean":
        raise ValueError("Unsupported project schema or language")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", config.get("project_id", "")):
        raise ValueError("project_id needs a safe ASCII name, e.g. bora")
    if config.get("batch") != BATCH or config.get("hard_negative_samples") != 0:
        raise ValueError("Baseline batch is 768/256/50/0; sampling changes require a separately declared experiment")
    if type(config.get("augmentation_rounds")) is not int or config["augmentation_rounds"] not in (1, 3, 5):
        raise ValueError("Only R1/R3/R5 are supported")
    for key in ("positive_originals", "validation_originals", "final_originals"):
        if type(config["targets"][key]) is not int or config["targets"][key] < 1:
            raise ValueError("TTS original counts must be positive integers")
    if config.get("human_reference_used") is not False or config.get("positive_text_policy") != "standalone_only":
        raise ValueError("No human references or appended commands in Positive generation")
    t = config["training"]
    for key in ("learning_rate", "weight_decay", "dropout", "positive_loss_weight"):
        if not isinstance(t[key], (float, int)) or not math.isfinite(t[key]):
            raise ValueError("Training numbers must be finite")
    for key in ("updates", "schedule_horizon", "cpu_threads", "log_every"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError("Training counters must be positive integers")
    if any(type(s) is not int or s < 0 for s in t["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if config["tts"]["batch_size"] != 1:
        raise ValueError("Stable Qwen generator uses cached references and batch_size=1; qualify other batching separately")
    if t["architecture"] not in ("cnn64", "cnn128", "mlp32", "mlp128", "mlp256"):
        raise ValueError("Supported stable architectures: cnn64/cnn128/mlp32/mlp128/mlp256")
    if t["optimizer"] not in ("Adam", "AdamW") or t["schedule"] not in ("constant", "cosine_with_2percent_warmup"):
        raise ValueError("Unsupported optimizer/schedule")
    if len(t["seeds"]) not in (1, 3) or len(set(t["seeds"])) != len(t["seeds"]):
        raise ValueError("One single model or three independently seeded models are supported")
    if t["updates"] < 1 or t["schedule_horizon"] < t["updates"]:
        raise ValueError("Schedule horizon must cover the actual optimizer updates")
    checkpoints = t["checkpoints"]
    if (not isinstance(checkpoints, list) or not checkpoints or
        any(type(s) is not int or s < 1 or s > t["updates"] for s in checkpoints) or
        checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != t["updates"]):
        raise ValueError("Checkpoints must be sorted unique updates within the budget and include its final update")
    if t["device"] != "auto" and t["device"] != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", t["device"]):
        raise ValueError("Training device must be auto, cpu or cuda[:index]")
    if t.get("delivery", "auto") not in ("auto", "mmap"):
        raise ValueError("Input delivery must be auto or mmap")
    if type(t.get("gpu_cache_budget_mb", 0)) is not int or t.get("gpu_cache_budget_mb", 0) < 0:
        raise ValueError("GPU cache budget must be a nonnegative integer")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or not 0 <= t["dropout"] < 1 or t["positive_loss_weight"] <= 0:
        raise ValueError("Invalid training settings")
    r = config["runtime"]
    if not math.isfinite(r["threshold"]) or not 0 < r["threshold"] < 1 or type(r["consecutive_frames"]) is not int or r["consecutive_frames"] < 1:
        raise ValueError("Invalid detection rule")
    if r["vad_enabled"] or r["noise_suppression_enabled"]:
        raise ValueError("Stable reference runtime supports VAD/NS off; qualify other paths separately")
    if r["sample_rate"] != 16000 or r["frame_samples"] != 1280 or r["tail_policy"] != "drop_partial_frame":
        raise ValueError("Reference input is mono 16kHz PCM16 / 80ms frames")
    for key in ("warmup_s", "cooldown_s", "call_postroll_s"):
        if not math.isfinite(r[key]) or r[key] < 0:
            raise ValueError("Runtime durations cannot be negative")
    if abs(r["warmup_s"] / .08 - round(r["warmup_s"] / .08)) > 1e-8:
        raise ValueError("Warmup must align to an 80ms frame")
    a = config["augmentation"]
    for key in ("clean_probability", "moderate_probability", "rir_probability"):
        if not isinstance(a[key], (int, float)) or not math.isfinite(a[key]) or not 0 <= a[key] <= 1:
            raise ValueError("Augmentation probabilities must be finite and between 0 and 1")
    if a["clean_probability"] + a["moderate_probability"] > 1:
        raise ValueError("Clean and moderate augmentation probabilities exceed 1")
    for key in ("moderate_snr_db", "hard_snr_db", "gain_db"):
        bounds = a[key]
        if not isinstance(bounds, list) or len(bounds) != 2 or any(not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError("Augmentation ranges must have two ordered finite values")
    weights = a["category_weights"]
    if set(weights) != {"environment", "music", "competing_speech"} or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("All three noise categories need positive finite sampling weights")
    if not math.isfinite(a["korean_speech_weight"]) or a["korean_speech_weight"] <= 0 or not math.isfinite(a["peak_limit"]) or not 0 < a["peak_limit"] < 1:
        raise ValueError("Invalid Korean interference weight or peak limit")
    return config


def create_project(output, wakeword, project_id, *, positive_originals=None,
                   augmentation_rounds=None, updates=None, training_overrides=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty project directory; existing experiments are immutable")
    config = default_profile()
    config.update(wakeword=wakeword, project_id=project_id)
    config = configure_budget(config, positive_originals=positive_originals,
        augmentation_rounds=augmentation_rounds, updates=updates, training_overrides=training_overrides)
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("data/positive", "data/negative", "data/materials", "data/eval", "features", "runs", "reports", "models", "review"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for path in config["manifests"].values():
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        (output / path).write_text("", encoding="utf-8")
    write_json(output / "project.json", config)
    write_json(output / "reports" / "setup_summary.json", workload_summary(config))
    write_json(output / "voices.json", voice_candidates())
    write_json(output / "approval.json", {"schema_version": 1, "state": "pending", "wakeword": wakeword,
        "human_reference_used": False, "generation_policy_approved": False,
        "note": "Approve the pilot and synthetic reference families before bulk generation; no invented listening decisions."})
    return output / "project.json"


def save_configuration(project, output, **choices):
    """Save a separate snapshot alongside the source to preserve relative paths."""
    root, config = load_project(project)
    output = Path(output).resolve()
    if (root / "final_lock.json").exists():
        raise ValueError("Final is frozen; use a new project for another training configuration")
    if output.parent != root or output.suffix.lower() != ".json":
        raise ValueError("Save the configuration as a new .json beside the original project file")
    if output.exists():
        raise ValueError("Configuration output already exists; never overwrite an experiment")
    result = configure_budget(config, **choices)
    write_json(output, result)
    write_json(root / "reports" / (output.stem + "_setup_summary.json"), workload_summary(result))
    return output


def load_project(path):
    path = Path(path).resolve()
    return path.parent, validate_project(read_json(path))


def voice_candidates():
    """Requested personas are candidates, not 120 certified distinct speakers."""
    ages = ("young adult", "middle-aged", "older adult", "adult")
    textures = ("clear light", "warm rounded", "low resonant", "bright nasal", "slightly husky", "soft airy", "crisp focused", "deep mellow", "smooth balanced", "rich textured", "gentle thin", "firm dry", "light breathy", "dense midrange", "clear ringing")
    families = []
    for i in range(120):
        split = "train" if i < 100 else "validation" if i < 110 else "final"
        families.append({"family_id": f"{split}_voice_{i:03d}", "split": split,
            "description": f"A native Korean {ages[(i // 30) % 4]} {'woman' if i % 2 else 'man'} with a {textures[i % 15]} timbre, pitch variation {i // 15 + 1}. Natural dry speech.",
            "style": ("neutral", "friendly", "urgent", "projected")[i % 4], "speed": ("slow", "natural", "quick")[(i // 4) % 3], "intonation": ("level", "rising", "falling")[(i // 12) % 3], "vocal_effort": ("soft", "normal", "strong")[(i // 36) % 3], "call_length": ("short", "natural", "drawn_out")[(i // 3) % 3],
            "provenance": "qwen_synthetic", "status": "candidate"})
    return {"schema_version": 1, "families": families,
        "note": "Descriptions/IDs do not prove acoustic diversity. Audit a representative pilot and disjoint families."}


def status(project):
    root, config = load_project(project)
    from .common import read_rows
    pools = {name: len(read_rows(resolved(root, path))) for name, path in config["manifests"].items()}
    return {"wakeword": config["wakeword"], "project_id": config["project_id"],
        "workload": workload_summary(config),
        "manifest_rows": pools, "approval": read_json(root / "approval.json"),
        "training_receipts": [{"path": str(p.relative_to(root)), "state": read_json(p).get("state")}
                              for p in sorted((root / "runs").glob("*/member*/receipt.json"))],
        "formal_training_receipts": [str(p.relative_to(root)) for p in sorted((root / "runs").glob("*/member*/receipt.json"))
                                     if read_json(p).get("state") == "trained_not_final_evaluated"],
        "note": "Counts and receipts are not proof of a currently live process or population performance."}
