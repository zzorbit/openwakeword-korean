"""Pinned original 80ms streaming features, preserved as portable library code."""
from contextlib import contextmanager
import importlib.metadata
from pathlib import Path
import numpy as np
from .audio import sha256_file
CLIP_SAMPLES=32000
HOP_SAMPLES=1280
FEATURE_SHAPE=(16,96)
PINNED_OPENWAKEWORD_VERSION="0.6.0"
PINNED_ASSETS={"melspectrogram.onnx":"ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f","embedding_model.onnx":"70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"}
@contextmanager
def preserved_startup_rng():
    """Upstream uses NumPy's global RNG for discarded startup embeddings.

Instances are sequential CPU workers, not shared across Python threads. Seed
and restore around upstream initialization/reset to avoid polluting caller RNG.
The final16features after2s do not include the startup random features.
"""
    previous = np.random.get_state()
    try:
        np.random.seed(20260928)
        yield
    finally:
        np.random.set_state(previous)

class StreamingFrontend:
    # Existing cache writer expects this field. Its scope is deliberately
    # narrow; it does not establish compatibility of a precomputed data pool.
    verified = True
    verification_scope = "pinned_assets_and_canonical_80ms_implementation_only"
    precomputed_generic_compatibility = "unverified_offline_provenance_and_semantics"

    def __init__(self, directory: Path, ncpu: int = 2, *, frames=16):
        if not isinstance(ncpu, int) or ncpu < 1:
            raise ValueError("ncpu must be a positive integer")
        directory = Path(directory).resolve()
        paths = {name: directory / name for name in PINNED_ASSETS}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Pinned frontend asset required: {path}")
            if sha256_file(path) != PINNED_ASSETS[name]:
                raise ValueError(f"Frontend asset differs from audited pin: {path}")
        version = importlib.metadata.version("openwakeword")
        if version != PINNED_OPENWAKEWORD_VERSION:
            raise ValueError("openWakeWord version changed; audit the new implementation before updating its pin")
        import openwakeword.utils
        from openwakeword.utils import AudioFeatures
        self.ncpu = ncpu
        from .training_context import clip_samples
        self.clip_samples = clip_samples(frames)
        self.feature_shape = (frames, 96)
        adapter_digest = sha256_file(Path(__file__))
        if frames == FEATURE_SHAPE[0]:
            from .studio import _context_profile_compatible_digest
            adapter_digest = _context_profile_compatible_digest(adapter_digest)
        self.runtime_kwargs = dict(melspec_model_path=str(paths["melspectrogram.onnx"]),
                                   embedding_model_path=str(paths["embedding_model.onnx"]),
                                   device="cpu", ncpu=ncpu)
        with preserved_startup_rng():
            self.extractor = AudioFeatures(inference_framework="onnx", **self.runtime_kwargs)
        self.identity = {
            "schema_version": 1, "algorithm": "AudioFeatures.streaming_80ms_independent_2s_last16_v1",
            "model_sha256": {"melspectrogram": PINNED_ASSETS["melspectrogram.onnx"],
                             "embedding": PINNED_ASSETS["embedding_model.onnx"]},
            "openwakeword_version": version,
            "openwakeword_utils_sha256": sha256_file(Path(openwakeword.utils.__file__)),
            "adapter_sha256": adapter_digest,
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "numpy_version": np.__version__, "provider": "CPUExecutionProvider", "cpu_threads": ncpu,
            "input": "16kHz_PCM16", "clip_samples": CLIP_SAMPLES, "feature_shape": list(FEATURE_SHAPE),
            "hop_samples": HOP_SAMPLES, "chunks_per_clip": 25,
            "reset_policy": "AudioFeatures.reset before every independent clip",
            "warmup_samples_added": 0, "extraction_at_samples": CLIP_SAMPLES,
            "startup_embeddings": "upstream startup buffers; none included in final16frames after2s",
            "mel_frame_support_samples": 512, "mel_hop_samples": 160,
            "mel_window_frames": 76, "mel_window_starts_in_clip": list(range(1, 122, 8)),
            "phase_offset_from_offline_embed_clips_samples": 160,
            "melspec_transform": "x/10+2", "waveform_modifications": "none",
            "verification_scope": self.verification_scope,
            "precomputed_generic_compatibility": self.precomputed_generic_compatibility,
        }
        if frames != FEATURE_SHAPE[0]:
            self.identity.update(algorithm="AudioFeatures.streaming_80ms_independent_extended_v1",
                clip_samples=self.clip_samples, feature_shape=list(self.feature_shape),
                chunks_per_clip=self.clip_samples // HOP_SAMPLES, extraction_at_samples=self.clip_samples,
                startup_embeddings="upstream startup buffers; none included in the returned window",
                mel_window_starts_in_clip=list(range(1, 1 + frames * 8, 8)))

    def embed(self, clips: np.ndarray) -> np.ndarray:
        if not isinstance(clips, np.ndarray) or clips.dtype != np.int16 or clips.ndim != 2 or clips.shape[1] != self.clip_samples or len(clips) == 0:
            raise ValueError("StreamingFrontend expects nonempty PCM16 with the configured context length")
        features = np.empty((len(clips), *self.feature_shape), dtype=np.float32)
        for index, clip in enumerate(clips):
            with preserved_startup_rng():
                self.extractor.reset()
            for offset in range(0, self.clip_samples, HOP_SAMPLES):
                processed = self.extractor(clip[offset:offset + HOP_SAMPLES])
                if processed != HOP_SAMPLES:
                    raise ValueError("Upstream chunk consumption changed; repeat frontend alignment audit")
            result = np.asarray(self.extractor.get_features(self.feature_shape[0]), dtype=np.float32)
            if result.shape != (1, *self.feature_shape) or not np.isfinite(result).all():
                raise ValueError("Upstream streaming feature shape/finite-value validation failed")
            features[index] = result[0]
        return features


class FastStreamingFrontend(StreamingFrontend):
    """Same retained computations; omit startup/early embeddings never returned."""
    def embed(self, clips):
        if clips.dtype != np.int16 or clips.ndim != 2 or clips.shape[1] != self.clip_samples:
            raise ValueError("Independent PCM16 with the configured context length required")
        features = np.empty((len(clips), *self.feature_shape), np.float32)
        extractor = self.extractor
        chunks = self.clip_samples // HOP_SAMPLES
        first = chunks - self.feature_shape[0]
        for index, clip in enumerate(clips):
            extractor.raw_data_buffer.clear()
            extractor.melspectrogram_buffer = np.ones((76, 32))
            for chunk in range(chunks):
                offset = chunk * 1280
                extractor._buffer_raw_data(clip[offset:offset + 1280])
                extractor._streaming_melspectrogram(1280)
                if chunk >= first:
                    mel = extractor.melspectrogram_buffer[-76:].astype(np.float32)[None, :, :, None]
                    feature = np.asarray(extractor.embedding_model_predict(mel), np.float32)
                    if feature.shape != (96,) or not np.isfinite(feature).all():
                        raise ValueError("Retained embedding shape/values changed")
                    features[index, chunk - first] = feature
        return features


def qualified_fast_frontend(directory, clips, *, frames=16):
    """Qualify representative actual inputs and edge cases on this environment."""
    import time
    canonical = StreamingFrontend(directory, ncpu=2, frames=frames)
    candidate = FastStreamingFrontend(directory, ncpu=2, frames=frames)
    probe = np.asarray(clips, np.int16)
    if len(probe) == 0:
        raise ValueError("Actual source probes are required before feature acceleration")
    rng = np.random.default_rng(20260928)
    size = canonical.clip_samples
    edges = np.zeros((4, size), np.int16)
    edges[1] = rng.integers(-100, 101, size, dtype=np.int16)
    edges[2, size // 2:] = 32767
    edges[3, 10000:10001] = -32768
    probe = np.concatenate([probe, edges])
    start = time.perf_counter(); expected = canonical.embed(probe); reference_s = time.perf_counter() - start
    start = time.perf_counter(); actual = candidate.embed(probe); candidate_s = time.perf_counter() - start
    reverse = candidate.embed(probe[::-1])[::-1]
    equal = bool(np.array_equal(expected.view("u4"), actual.view("u4")) and np.array_equal(actual.view("u4"), reverse.view("u4")))
    report = {"actual_source_clips": len(clips), "edge_clips": 4, "bitwise_equal": equal,
        "maximum_absolute_error": float(np.max(np.abs(expected - actual))),
        "canonical_elapsed_s": reference_s, "candidate_elapsed_s": candidate_s,
        "adapter_sha256": canonical.identity["adapter_sha256"], "canonical_identity": canonical.identity,
        "scope": ("Independent2s CPU features only; does not change continuous inference or adopt GPU inference" if frames == 16
                  else "Independent extended-context PCM16 CPU features")}
    chosen = candidate if equal and candidate_s < reference_s else canonical
    report["adopted"] = "qualified_fast_CPU" if chosen is candidate else "canonical_CPU"
    return chosen, report
