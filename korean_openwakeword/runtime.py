"""CPU ONNX reference: one shared frontend, ordered mean, frozen event rules."""
from importlib.resources import files
import importlib.metadata
from pathlib import Path
import math
import numpy as np
import soundfile as sf

from .common import read_json, bound_file
from .events import EventDetector


def sample_bundle():
    return Path(str(files("korean_openwakeword") / "resources" / "sample_chunsik"))


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("aggregation") != "ordered_float64_mean_to_float32":
        raise ValueError("Unsupported bundle schema or score aggregation")
    members = manifest["members"]
    if len(members) not in (1, 3) or len({m["path"] for m in members}) != len(members):
        raise ValueError("One model or three distinct member files required")
    if len({m["sha256"] for m in members}) != len(members):
        raise ValueError("Duplicate heads cannot be counted as independent members")
    for entry in members + manifest["frontend"]:
        p = Path(entry["path"])
        if p.is_absolute() or not (directory / p).resolve().is_relative_to(directory):
            raise ValueError("Bundle assets must stay inside the portable bundle")
        bound_file(directory, entry)
    r = manifest["runtime"]
    if r["sample_rate"] != 16000 or r["frame_samples"] != 1280 or r["vad_enabled"] or r["noise_suppression_enabled"]:
        raise ValueError("Only the qualified raw 16kHz/80ms ONNX path is supported")
    if not math.isfinite(r["threshold"]) or not 0 < r["threshold"] < 1 or type(r["consecutive_frames"]) is not int or r["consecutive_frames"] < 1 or any(not math.isfinite(r[k]) or r[k] < 0 for k in ("warmup_s", "cooldown_s", "call_postroll_s")) or r["tail_policy"] != "drop_partial_frame":
        raise ValueError("Invalid bundle event/runtime rule")
    return manifest


class WakeWordDetector:
    """Use ``process(frame)`` for one PCM16 80ms frame; reset between sessions.

    This object is stateful and is not shared between simultaneous streams.
    Model construction preserves caller NumPy RNG. Initial history is reset by
    the exact same policy used in the original scored sample model.
    """
    def __init__(self, bundle=None):
        self.directory = Path(bundle) if bundle is not None else sample_bundle()
        self.manifest = verify_bundle(self.directory)
        if importlib.metadata.version("openwakeword") != "0.6.0":
            raise ValueError("The reference path is pinned to openwakeword==0.6.0")
        from openwakeword.model import Model
        members = self.manifest["members"]
        self.keys = [Path(m["path"]).stem for m in members]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Model filenames need unique stems")
        frontend = {Path(m["path"]).name: str(bound_file(self.directory, m)) for m in self.manifest["frontend"]}
        state = np.random.get_state()
        try:
            np.random.seed(20260928)
            self.model = Model(wakeword_models=[str(bound_file(self.directory, m)) for m in members],
                inference_framework="onnx", enable_speex_noise_suppression=False, vad_threshold=0,
                melspec_model_path=frontend["melspectrogram.onnx"],
                embedding_model_path=frontend["embedding_model.onnx"])
        finally:
            np.random.set_state(state)
        if list(self.model.models) != self.keys or any(self.model.model_inputs[k] != 16 or self.model.model_outputs[k] != 1 for k in self.keys):
            raise ValueError("Classifiers must consume 16×96 features and return one probability")
        self.runtime = self.manifest["runtime"]
        self.reset()

    def reset(self):
        state = np.random.get_state()
        try:
            np.random.seed(20260928)
            self.model.reset()
        finally:
            np.random.set_state(state)
        self.frames = 0
        self.gate = EventDetector(self.runtime["threshold"], self.runtime["consecutive_frames"], self.runtime["cooldown_s"])

    def score_members(self, frame):
        frame = np.asarray(frame)
        if frame.dtype != np.int16 or frame.shape != (1280,):
            raise ValueError("Expected a mono np.int16 array of exactly 1280 samples at 16kHz")
        prediction = self.model.predict(frame)
        values = np.asarray([prediction[k] for k in self.keys], dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Invalid classifier probabilities")
        return values

    def process(self, frame):
        values = self.score_members(frame)
        # Preserve the selected ensemble's operation order exactly.
        if len(values) == 3:
            score = np.float32((np.float64(values[0]) + np.float64(values[1]) + np.float64(values[2])) / 3)
        else:
            score = values[0]
        start = self.frames * .08
        self.frames += 1
        timestamp = self.frames * .08
        eligible = start + 1e-9 >= self.runtime["warmup_s"]
        decision = self.gate.update(float(score), timestamp, eligible)
        return {"time_s": timestamp, "score": float(score), "member_scores": values.tolist(),
                "eligible": eligible, **decision}

    def predict_file(self, path):
        self.reset()
        events, maximum, frames = [], 0.0, 0
        with sf.SoundFile(path) as audio:
            if audio.samplerate != 16000 or audio.channels != 1 or audio.subtype != "PCM_16":
                raise ValueError("Evaluation input must already be mono 16kHz PCM16; no hidden resampling/gain")
            while True:
                frame = audio.read(1280, dtype="int16")
                if len(frame) != 1280:
                    break
                row = self.process(frame)
                frames += 1
                maximum = max(maximum, row["score"])
                if row["event"]:
                    events.append(row)
            tail = audio.frames - frames * 1280
        return {"wakeword": self.manifest["wakeword"], "events": events, "max_score": maximum,
                "frames": frames, "dropped_tail_samples": tail,
                "runtime": self.runtime, "evaluation_scope": "single supplied audio file, not population validation"}
