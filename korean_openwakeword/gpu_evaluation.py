"""CUDA FP32 evaluation with one score path for every audio session.

The portable bundle and its original frontend remain unchanged. A derived mel
graph prevents independent sessions from sharing the original global dB floor.
CUDA FP32 is a separately qualified backend, not bitwise CPU arithmetic.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import soundfile as sf

from .common import bound_file, digest
from .evaluation_backend import CUDA_SCORING_PROTOCOL
from .evaluation_workers import replay_session_scores
from .frontend import PINNED_ASSETS
from .runtime import bundle_candidates, verify_bundle
from .task_progress import activity as report_activity

_RUNTIME_ENVIRONMENT_LOCK = threading.RLock()
GPU_MAX_PCM_BYTES = 512 * 1024 ** 2
_PCM_FRAME_BYTES = 1280 * 2


@contextmanager
def evaluation_runtime_environment(environment):
    """Scope managed GPU DLL/runtime settings to one evaluation consumer.

    Loaded native libraries cannot be swapped or unloaded safely; the resolver
    checks compatibility before this context is entered. Studio uses a fresh
    worker for each stage, and Python callers must do the same after importing
    a different ONNX Runtime build.
    """
    environment = dict(environment or {})
    allowed = {"KOWW_GPU_RUNTIME_PATH", "KOWW_GPU_DLL_PATHS"}
    if any(key not in allowed or not isinstance(value, str) or not value for key, value in environment.items()):
        raise ValueError("GPU runtime environment must contain only nonempty runtime and DLL paths")
    with _RUNTIME_ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in allowed | {"PATH", "PYTHONDONTWRITEBYTECODE"}}
        previous_bytecode = sys.dont_write_bytecode
        try:
            os.environ.update(environment)
            # The managed runtime inventory is immutable after installation.
            # Keep bytecode out of the verified runtime package directory.
            sys.dont_write_bytecode = True
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            yield
        finally:
            sys.dont_write_bytecode = previous_bytecode
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _cuda_runtime():
    """Load only the explicitly selected GPU package in this worker process."""
    target = os.environ.get("KOWW_GPU_RUNTIME_PATH")
    if target:
        target = Path(target).resolve()
        if not (target / "onnxruntime" / "__init__.py").is_file():
            raise ValueError("KOWW_GPU_RUNTIME_PATH must contain the GPU onnxruntime package")
        existing = sys.modules.get("onnxruntime")
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(target):
            raise RuntimeError("A different ONNX Runtime is already loaded; start a separate GPU worker with its package path first")
        # A Python -m subprocess can put its working directory before
        # PYTHONPATH. Move the selected runtime ahead of that CPU installation
        # even when the runtime path was already supplied by the launcher.
        location = str(target)
        if location in sys.path:
            sys.path.remove(location)
        sys.path.insert(0, location)
    directories = []
    explicit = os.environ.get("KOWW_GPU_DLL_PATHS")
    if explicit:
        directories.extend(Path(value).resolve() for value in explicit.split(os.pathsep) if value)
    elif target:
        dependency_root = target.parent / "cuda13" / "nvidia"
        if dependency_root.is_dir():
            directories.extend(sorted({path.parent for path in dependency_root.rglob("*.dll")}))
    handles = []
    for directory in dict.fromkeys(directories):
        if not directory.is_dir():
            raise ValueError("GPU DLL directory does not exist: " + str(directory))
        if os.name == "nt":
            handles.append(os.add_dll_directory(str(directory)))
    if directories:
        os.environ["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
    import onnxruntime as ort
    if target and not Path(ort.__file__).resolve().is_relative_to(target):
        raise RuntimeError("선택한 GPU용 ONNX Runtime 경로에서 라이브러리를 불러오지 못했습니다.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("불러온 ONNX Runtime에서 CUDA 실행 공급자를 사용할 수 없습니다.")
    # CUDA's DEFAULT convolution algorithm can emit a warning at every new
    # batch shape even while the native CUDA kernel works correctly. Keep
    # errors visible and validate actual providers instead of flooding logs.
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    return ort, handles


def _per_session_mel_model(path):
    """Preserve the pinned single-stream function while exposing safe batches."""
    import onnx
    from onnx import helper, numpy_helper

    original = Path(path).read_bytes()
    source_hash = hashlib.sha256(original).hexdigest()
    if source_hash != PINNED_ASSETS["melspectrogram.onnx"]:
        raise ValueError("CUDA batching requires the audited pinned mel frontend")
    model = onnx.load_model_from_string(original)
    if {entry.domain: entry.version for entry in model.opset_import} != {"": 13}:
        raise ValueError("Pinned mel graph opset changed")
    reductions = [node for node in model.graph.node if node.name == "ReduceMax_24" and node.op_type == "ReduceMax"]
    clips = [node for node in model.graph.node if node.name == "Clip_28" and node.op_type == "Clip"]
    if len(reductions) != 1 or len(clips) != 1:
        raise ValueError("Pinned mel graph dynamic-range nodes changed")
    reduction, clip = reductions[0], clips[0]
    attributes = {entry.name: helper.get_attribute_value(entry) for entry in reduction.attribute}
    if attributes != {"keepdims": 0} or len(clip.input) != 3:
        raise ValueError("Pinned mel graph dynamic-range semantics changed")
    constants = {entry.name: numpy_helper.to_array(entry) for entry in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for entry in node.attribute:
                if entry.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(entry.t)
    upper = constants.get(clip.input[2])
    if upper is None or np.asarray(upper).size != 1 or not np.isposinf(upper).all():
        raise ValueError("Replacing the mel floor requires the original positive-infinite upper bound")
    reduction.attribute[0].i = 1
    reduction.attribute.extend([helper.make_attribute("axes", [1, 2, 3])])
    clip.CopyFrom(helper.make_node("Max", list(clip.input[:2]), list(clip.output),
                                  name="KOWW_Probe_PerSessionFloor_Max"))
    onnx.checker.check_model(model)
    derived = model.SerializeToString()
    return derived, {"policy": "pinned_mel_independent_session_dynamic_range_v1",
        "source_sha256": source_hash, "derived_sha256": hashlib.sha256(derived).hexdigest(),
        "changed_nodes": ["ReduceMax_24 axes=[1,2,3], keepdims=1", "Clip_28 to broadcast Max; original upper bound +inf"],
        "source_model_modified": False, "derived_storage": "worker memory only"}


class _GpuBatchScorer:
    def __init__(self, bundle, model_ids, stats):
        self.ort, self.dll_handles = _cuda_runtime()
        self.bundle = Path(bundle).resolve()
        manifest = verify_bundle(self.bundle)
        self.runtime = manifest["runtime"]
        expected_ids = [row["id"] for row in bundle_candidates(manifest)]
        if list(model_ids) != expected_ids or len(manifest["members"]) != 3:
            raise ValueError("Qualified CUDA evaluation requires the original ordered three heads and their mean")
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.provider_options = {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT",
            "gpu_mem_limit": str(2 * 1024 ** 3), "do_copy_in_default_stream": "1"}

        def session(source):
            result = self.ort.InferenceSession(source, sess_options=options,
                providers=[("CUDAExecutionProvider", self.provider_options)])
            if result.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA session initialization fell back to CPU")
            if result.get_provider_options().get("CUDAExecutionProvider", {}).get("use_tf32") != "0":
                raise RuntimeError("CUDA FP32 evaluation requires TF32 to be explicitly disabled")
            result.disable_fallback()
            return result

        original_mel = next(self.bundle / entry["path"] for entry in manifest["frontend"]
                            if Path(entry["path"]).name == "melspectrogram.onnx")
        embedding = next(self.bundle / entry["path"] for entry in manifest["frontend"]
                         if Path(entry["path"]).name == "embedding_model.onnx")
        derived, lineage = _per_session_mel_model(original_mel)
        self.mel = session(derived)
        self.embedding = session(str(embedding))
        self.heads = [session(str(self.bundle / entry["path"])) for entry in manifest["members"]]
        self.initial_features = self._initial_history()
        stats.update({"backend": CUDA_SCORING_PROTOCOL,
            "onnxruntime_version": self.ort.__version__, "onnxruntime_module": str(Path(self.ort.__file__).resolve()),
            "providers": self.mel.get_providers(), "provider_options": self.mel.get_provider_options(),
            "mel_derivation": lineage, "bundle_manifest_sha256": digest(self.bundle / "manifest.json"),
            "initial_history": "seeded CUDA reset history copied independently into every session",
            "precision": "FP32; TF32 disabled; no AMP or model quantization"})

    def _initial_history(self):
        seed = np.random.RandomState(20260928).randint(-1000, 1000, 16000 * 4).astype(np.float32)
        mel = self.mel.run(None, {self.mel.get_inputs()[0].name: seed[None, :]})[0]
        mel = np.asarray(mel).reshape(-1, 32) / 10 + 2
        windows = np.stack([mel[start:start + 76] for start in range(0, len(mel) - 75, 8)])
        embedded = self.embedding.run(None, {self.embedding.get_inputs()[0].name: windows[:, :, :, None]})[0]
        return np.asarray(embedded).reshape(len(windows), 96)

    def score_batch(self, pcms):
        mels = [np.ones((76, 32)) for _ in pcms]
        features = [self.initial_features.copy() for _ in pcms]
        results = [[] for _ in pcms]
        lengths = [len(pcm) // 1280 for pcm in pcms]
        total_frames, completed_frames = sum(lengths), 0
        long_batch = max(lengths, default=0) >= 750
        if long_batch:
            report_activity("gpu_frames", "긴 음원 GPU 묶음 계산 중", 0, total_frames,
                            audio_seconds=0, batch_audio_seconds=total_frames * .08, batch_files=len(pcms))
        for t in range(max((len(pcm) // 1280 for pcm in pcms), default=0)):
            end = (t + 1) * 1280
            active = [index for index, pcm in enumerate(pcms) if end <= len(pcm)]
            raw = np.stack([pcms[index][max(0, end - 1760):end] for index in active]).astype(np.float32)
            mel = self.mel.run(None, {self.mel.get_inputs()[0].name: raw})[0]
            mel = np.asarray(mel).reshape(len(active), -1, 32) / 10 + 2
            for slot, index in enumerate(active):
                # Each independent stream only feeds its latest model window.
                mels[index] = np.vstack((mels[index], mel[slot]))[-76:]
            windows = np.stack([mels[index][-76:].astype(np.float32) for index in active])[:, :, :, None]
            embedded = self.embedding.run(None, {self.embedding.get_inputs()[0].name: windows})[0]
            embedded = np.asarray(embedded).reshape(len(active), 96)
            for slot, index in enumerate(active):
                features[index] = np.vstack((features[index], embedded[slot]))[-16:]
            inputs = np.stack([features[index][-16:].astype(np.float32) for index in active])
            values = np.column_stack([head.run(None, {head.get_inputs()[0].name: inputs})[0].reshape(-1)
                                      for head in self.heads]).astype(np.float32)
            if t < 5:
                values.fill(0)
            mean = ((values[:, 0].astype(np.float64) + values[:, 1].astype(np.float64)
                     + values[:, 2].astype(np.float64)) / 3).astype(np.float32)
            values = np.column_stack((values, mean))
            for slot, index in enumerate(active):
                results[index].append(values[slot])
            completed_frames += len(active)
            if long_batch and (t % 32 == 31 or completed_frames == total_frames):
                report_activity("gpu_frames", "긴 음원 GPU 묶음 계산 중", completed_frames, total_frames,
                    audio_seconds=completed_frames * .08, batch_audio_seconds=total_frames * .08,
                    batch_files=len(pcms))
        return [np.asarray(values, dtype=np.float32).reshape(-1, 4) for values in results]


def _fits_group(count, cached_bytes, pcm_bytes, batch_size, max_audio_bytes):
    return count < batch_size and cached_bytes + pcm_bytes <= max_audio_bytes


def gpu_scoring_group_start(offsets, start, batch_size, max_audio_bytes=GPU_MAX_PCM_BYTES):
    """Recover the uninterrupted batch boundary for a legacy partial checkpoint."""
    if start in (0, len(offsets) - 1):
        return start
    group_start = count = cached_bytes = 0
    for index in range(start + 1):
        pcm_bytes = (offsets[index + 1] - offsets[index]) * _PCM_FRAME_BYTES
        if pcm_bytes > max_audio_bytes:
            # Over-limit files always score alone, including when preceded by a cached group.
            group_start, count, cached_bytes = index, 0, 0
            continue
        if not count or not _fits_group(count, cached_bytes, pcm_bytes, batch_size, max_audio_bytes):
            group_start, count, cached_bytes = index, 0, 0
        count += 1
        cached_bytes += pcm_bytes
    return group_start


class _StreamPcm:
    """Read the current GPU frame window from a long file on demand."""
    def __init__(self, audio, frames):
        self.audio, self.samples = audio, frames * 1280

    def __len__(self):
        return self.samples

    def __getitem__(self, span):
        self.audio.seek(span.start)
        window = self.audio.read(span.stop - span.start, dtype="int16")
        if len(window) != span.stop - span.start:
            raise ValueError("GPU evaluation audio changed during streaming")
        return window


def score_gpu_sessions(bundle, model_ids, items, *, batch_size=128, qualification=None,
                       stats=None, max_audio_bytes=GPU_MAX_PCM_BYTES, emit_from=0, on_group_complete=None):
    """Yield ordered CUDA scores, calling the checkpoint hook after each group."""
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("GPU evaluation batch_size must be an integer from 1 to 128")
    if type(max_audio_bytes) is not int or max_audio_bytes < 1:
        raise ValueError("GPU max_audio_bytes must be a positive integer")
    if not isinstance(qualification, dict) or qualification.get("user_approved_numerical_differences") is not True:
        raise ValueError("GPU evaluation requires recorded qualification and explicit approval of numerical differences")
    stats = {} if stats is None else stats
    began = time.perf_counter()
    stats.update({"qualification": deepcopy(qualification), "batch_size": batch_size,
        "max_audio_bytes": max_audio_bytes, "gpu_sessions": 0, "gpu_frames": 0,
        "gpu_streamed_sessions": 0, "gpu_batches": 0, "maximum_cached_pcm_bytes": 0,
        "gpu_scoring_wall_seconds": 0.0, "pcm_read_wall_seconds": 0.0, "metric_replay_wall_seconds": 0.0})
    scorer = _GpuBatchScorer(bundle, model_ids, stats)

    def flush(batch):
        before = time.perf_counter()
        values = scorer.score_batch([entry[3] for entry in batch])
        stats["gpu_scoring_wall_seconds"] += time.perf_counter() - before
        if len(values) != len(batch):
            raise ValueError("GPU batch returned a different number of sessions")
        stats["gpu_batches"] += 1
        for (index, row, path, pcm), scores in zip(batch, values):
            if (scores.dtype != np.float32 or scores.shape != (len(pcm) // 1280, len(model_ids))
                    or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
                raise ValueError("GPU evaluation scores have invalid shape, precision or values")
            if index < emit_from:
                continue
            before = time.perf_counter()
            mean = ((scores[:, 0].astype(np.float64) + scores[:, 1].astype(np.float64)
                     + scores[:, 2].astype(np.float64)) / 3).astype(np.float32)
            if not np.array_equal(mean.view("u4"), scores[:, 3].view("u4")):
                raise ValueError("GPU ordered mean differs from reference score replay")
            # Preserve the former replay boundary: cached PCM excludes the partial last frame.
            audio_samples = pcm.audio.frames if isinstance(pcm, _StreamPcm) else len(pcm)
            clip = replay_session_scores(row, path, model_ids, scorer.runtime, audio_samples, scores)
            stats["metric_replay_wall_seconds"] += time.perf_counter() - before
            clip["evaluation_compute"] = {"backend": "cuda_batch_fp32"}
            stats["gpu_sessions"] += 1
            stats["gpu_frames"] += len(scores)
            stats["elapsed_wall_seconds"] = time.perf_counter() - began
            yield index, clip, scores

    batch, cached_bytes = [], 0
    previous_index = None
    try:
        for index, row, path in items:
            if type(index) is not int or previous_index is not None and index != previous_index + 1:
                raise ValueError("GPU evaluation requires contiguous ordered session indices")
            previous_index = index
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
                raise ValueError("GPU evaluation inputs must already be mono16kHz PCM16")
            frames = info.frames // 1280
            pcm_bytes = frames * _PCM_FRAME_BYTES
            if batch and not _fits_group(len(batch), cached_bytes, pcm_bytes, batch_size, max_audio_bytes):
                yield from flush(batch)
                if on_group_complete is not None:
                    on_group_complete()
                batch, cached_bytes = [], 0
            if pcm_bytes > max_audio_bytes:
                # Check a streamed input again before its scores can be committed.
                path = Path(path).resolve()
                before_stat = path.stat()
                file_identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                               value.st_mtime_ns, value.st_ctime_ns)
                bound_file(path.parent, {"path": str(path), "sha256": row["sha256"]})
                with sf.SoundFile(path) as audio:
                    group_scores = list(flush([(index, row, path, _StreamPcm(audio, frames))]))
                if file_identity(path.stat()) != file_identity(before_stat):
                    raise ValueError("GPU evaluation streamed input changed during scoring")
                stats["gpu_streamed_sessions"] += 1
                yield from group_scores
                if on_group_complete is not None:
                    on_group_complete()
                continue
            before = time.perf_counter()
            pcm, rate = sf.read(path, frames=frames * 1280, dtype="int16")
            stats["pcm_read_wall_seconds"] += time.perf_counter() - before
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != frames * 1280:
                raise ValueError("GPU evaluation input changed during PCM caching")
            batch.append((index, row, path, pcm))
            cached_bytes += pcm.nbytes
            stats["maximum_cached_pcm_bytes"] = max(stats["maximum_cached_pcm_bytes"], cached_bytes)
        if batch:
            yield from flush(batch)
            if on_group_complete is not None:
                on_group_complete()
    finally:
        stats["elapsed_wall_seconds"] = time.perf_counter() - began
