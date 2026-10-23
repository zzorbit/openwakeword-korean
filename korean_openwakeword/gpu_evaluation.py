"""Explicit CUDA batch evaluation with whole-session CPU boundary rechecks.

The portable bundle and its original frontend remain unchanged. A derived mel
graph prevents independent sessions from sharing the original global dB floor.
CUDA FP32 is a separately qualified backend, not bitwise CPU arithmetic.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import io
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

from .common import bound_file, digest
from .evaluation_workers import score_session
from .frontend import PINNED_ASSETS
from .runtime import WakeWordDetector, bundle_candidates

_CPU_REFERENCE = None
_CPU_MODEL_IDS = None


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
        sys.path.insert(0, str(target))
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
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable; GPU evaluation cannot silently become CPU evaluation")
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
        self.reference = WakeWordDetector(bundle)
        self.runtime = self.reference.runtime
        self.bundle = Path(bundle).resolve()
        expected_ids = [row["id"] for row in bundle_candidates(self.reference.manifest)]
        if list(model_ids) != expected_ids or len(self.reference.manifest["members"]) != 3:
            raise ValueError("Qualified CUDA evaluation requires the original ordered three heads and their mean")
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.provider_options = {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT",
            "gpu_mem_limit": str(2 * 1024 ** 3), "do_copy_in_default_stream": "1"}

        def session(source):
            result = self.ort.InferenceSession(source, sess_options=options,
                providers=[("CUDAExecutionProvider", self.provider_options)])
            if result.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA session initialization fell back to CPU")
            if result.get_provider_options().get("CUDAExecutionProvider", {}).get("use_tf32") != "0":
                raise RuntimeError("CUDA FP32 evaluation requires TF32 to be explicitly disabled")
            return result

        original_mel = next(self.bundle / entry["path"] for entry in self.reference.manifest["frontend"]
                            if Path(entry["path"]).name == "melspectrogram.onnx")
        embedding = next(self.bundle / entry["path"] for entry in self.reference.manifest["frontend"]
                         if Path(entry["path"]).name == "embedding_model.onnx")
        derived, lineage = _per_session_mel_model(original_mel)
        self.mel = session(derived)
        self.embedding = session(str(embedding))
        self.heads = [session(str(self.bundle / entry["path"])) for entry in self.reference.manifest["members"]]
        self.reference.reset()
        self.initial_features = self.reference.model.preprocessor.feature_buffer.copy()
        stats.update({"backend": "cuda_batch_fp32_whole_session_boundary_cpu_v1",
            "onnxruntime_version": self.ort.__version__, "onnxruntime_module": str(Path(self.ort.__file__).resolve()),
            "providers": self.mel.get_providers(), "provider_options": self.mel.get_provider_options(),
            "mel_derivation": lineage, "bundle_manifest_sha256": digest(self.bundle / "manifest.json"),
            "initial_history": "exact CPU reference seeded reset history copied independently into every session",
            "precision": "FP32; TF32 disabled; no AMP or model quantization"})

    def score_batch(self, pcms):
        mels = [np.ones((76, 32)) for _ in pcms]
        features = [self.initial_features.copy() for _ in pcms]
        results = [[] for _ in pcms]
        for t in range(max((len(pcm) // 1280 for pcm in pcms), default=0)):
            end = (t + 1) * 1280
            active = [index for index, pcm in enumerate(pcms) if end <= len(pcm)]
            raw = np.stack([pcms[index][max(0, end - 1760):end] for index in active]).astype(np.float32)
            mel = self.mel.run(None, {self.mel.get_inputs()[0].name: raw})[0]
            mel = np.asarray(mel).reshape(len(active), -1, 32) / 10 + 2
            for slot, index in enumerate(active):
                mels[index] = np.vstack((mels[index], mel[slot]))[-970:]
            windows = np.stack([mels[index][-76:].astype(np.float32) for index in active])[:, :, :, None]
            embedded = self.embedding.run(None, {self.embedding.get_inputs()[0].name: windows})[0]
            embedded = np.asarray(embedded).reshape(len(active), 96)
            for slot, index in enumerate(active):
                features[index] = np.vstack((features[index], embedded[slot]))[-120:]
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
        return [np.asarray(values, dtype=np.float32).reshape(-1, 4) for values in results]


class _ReplayDetector:
    def __init__(self, runtime, scores):
        self.runtime, self.scores = runtime, scores

    def reset(self):
        self.index = 0

    def score_members(self, frame):
        values = self.scores[self.index, :3]
        self.index += 1
        return values


def _wav_bytes(pcm):
    stream = io.BytesIO()
    sf.write(stream, pcm, 16000, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def _score_cached(detector, row, wav, model_ids):
    clip, scores = score_session(detector, row, io.BytesIO(wav), model_ids)
    clip["path"] = str(row["path"])
    return clip, scores


def _initialize_cpu_recheck(bundle, model_ids):
    global _CPU_REFERENCE, _CPU_MODEL_IDS
    _CPU_REFERENCE = WakeWordDetector(bundle)
    _CPU_MODEL_IDS = list(model_ids)
    frontend = _CPU_REFERENCE.model.preprocessor
    sessions = list(_CPU_REFERENCE.model.models.values()) + [frontend.melspec_model, frontend.embedding_model]
    for session in sessions:
        options = session.get_session_options()
        if session.get_providers() != ["CPUExecutionProvider"] or options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1:
            raise RuntimeError("CPU boundary rechecks require the unchanged single-thread CPU reference")


def _cpu_cached_case(case):
    position, row, wav = case
    return position, _score_cached(_CPU_REFERENCE, row, wav, _CPU_MODEL_IDS)


class _CpuRechecker:
    """Process isolation keeps NumPy's seeded reference resets independent."""
    def __init__(self, bundle, model_ids, reference, workers):
        self.bundle, self.model_ids = str(bundle), list(model_ids)
        self.reference, self.workers, self.pool = reference, workers, None

    def score(self, cases):
        if not cases:
            return {}
        if self.workers == 1:
            return {position: _score_cached(self.reference, row, wav, self.model_ids)
                    for position, row, wav in cases}
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_cpu_recheck,
                initargs=(self.bundle, self.model_ids))
        return dict(self.pool.map(_cpu_cached_case, cases, chunksize=1))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def _boundary_near(scores, runtime, margin):
    eligible = np.arange(len(scores)) * .08 + 1e-9 >= runtime["warmup_s"]
    return bool(np.any(np.abs(scores[eligible] - runtime["threshold"]) <= margin))


def _difference_stats(stats, before_clip, before, after_clip, after, model_ids, runtime):
    stats["maximum_cpu_recheck_score_difference"] = max(stats["maximum_cpu_recheck_score_difference"],
                                                        float(np.max(np.abs(before - after), initial=0)))
    eligible = np.arange(len(before)) * .08 + 1e-9 >= runtime["warmup_s"]
    threshold = runtime["threshold"]
    stats["cpu_recheck_threshold_crossings"] += int(np.count_nonzero(
        (before[eligible] >= threshold) != (after[eligible] >= threshold)))
    for name, field in (("event", "events_s"), ("success", "success"),
                        ("false_activation", "false_activations"), ("latency", "latency_s")):
        if any(before_clip["results"][key][field] != after_clip["results"][key][field] for key in model_ids):
            stats["cpu_recheck_" + name + "_changed_sessions"] += 1


def score_gpu_sessions(bundle, model_ids, items, *, batch_size=128, boundary_margin=.001,
                       qualification=None, stats=None, max_audio_bytes=512 * 1024 ** 2,
                       cpu_recheck_workers=4):
    """Yield ordered ``(index, clip, scores)`` for previously validated inputs.

    A near-threshold eligible score in ANY member or the ordered mean causes
    that entire session to be replaced by CPU reference scores and decisions.
    The margin is an explicit conservative policy, not a proved global error
    bound. Qualification and numerical-difference approval must be recorded by
    the caller. Oversized individual files use the streaming CPU path.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("GPU evaluation batch_size must be an integer from 1 to 128")
    if isinstance(boundary_margin, bool) or not isinstance(boundary_margin, (int, float)) or not math.isfinite(boundary_margin) or not 0 < boundary_margin < 1:
        raise ValueError("GPU boundary_margin must be a positive finite value below one")
    if type(max_audio_bytes) is not int or max_audio_bytes < 1:
        raise ValueError("GPU max_audio_bytes must be a positive integer")
    if type(cpu_recheck_workers) is not int or not 1 <= cpu_recheck_workers <= 8:
        raise ValueError("CPU recheck workers must be an integer from 1 to 8")
    if not isinstance(qualification, dict) or qualification.get("user_approved_numerical_differences") is not True:
        raise ValueError("GPU evaluation requires recorded qualification and explicit approval of numerical differences")
    stats = {} if stats is None else stats
    began = time.perf_counter()
    stats.update({"qualification": deepcopy(qualification), "batch_size": batch_size,
        "boundary_margin": boundary_margin, "max_audio_bytes": max_audio_bytes,
        "cpu_recheck_workers": cpu_recheck_workers, "gpu_sessions": 0, "gpu_frames": 0,
        "cpu_rechecked_sessions": 0, "cpu_long_sessions": 0, "cpu_long_frames": 0,
        "cpu_rechecked_rate": 0.0, "gpu_batches": 0, "maximum_cached_pcm_bytes": 0,
        "gpu_scoring_wall_seconds": 0.0, "cpu_recheck_wall_seconds": 0.0,
        "pcm_read_wall_seconds": 0.0, "metric_replay_wall_seconds": 0.0,
        "maximum_cpu_recheck_score_difference": 0.0, "cpu_recheck_threshold_crossings": 0,
        "cpu_recheck_event_changed_sessions": 0, "cpu_recheck_success_changed_sessions": 0,
        "cpu_recheck_false_activation_changed_sessions": 0, "cpu_recheck_latency_changed_sessions": 0,
        "numerical_comparison_scope": "Only whole sessions rechecked on CPU; other CUDA scores are not asserted bitwise identical",
        "boundary_policy": "Any eligible member/mean within margin triggers whole-session CPU replacement; no guaranteed global error bound"})
    scorer = _GpuBatchScorer(bundle, model_ids, stats)
    rechecker = _CpuRechecker(bundle, model_ids, scorer.reference, cpu_recheck_workers)

    def flush(batch):
        before = time.perf_counter()
        values = scorer.score_batch([entry[3] for entry in batch])
        stats["gpu_scoring_wall_seconds"] += time.perf_counter() - before
        if len(values) != len(batch):
            raise ValueError("GPU batch returned a different number of sessions")
        gpu_results, cases = [], []
        before = time.perf_counter()
        for position, ((index, row, path, pcm), scores) in enumerate(zip(batch, values)):
            if scores.dtype != np.float32 or scores.shape != (len(pcm) // 1280, len(model_ids)) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError("GPU evaluation scores have invalid shape, precision or values")
            wav = _wav_bytes(pcm)
            clip, replay = _score_cached(_ReplayDetector(scorer.runtime, scores), row, wav, model_ids)
            if not np.array_equal(replay.view("u4"), scores.view("u4")):
                raise ValueError("GPU ordered mean differs from reference score replay")
            gpu_results.append((clip, scores))
            if _boundary_near(scores, scorer.runtime, boundary_margin):
                cases.append((position, row, wav))
        stats["metric_replay_wall_seconds"] += time.perf_counter() - before
        before = time.perf_counter()
        checked = rechecker.score(cases)
        stats["cpu_recheck_wall_seconds"] += time.perf_counter() - before
        stats["gpu_batches"] += 1
        for position, (index, row, path, pcm) in enumerate(batch):
            clip, scores = gpu_results[position]
            corrected = position in checked
            if corrected:
                cpu_clip, cpu_scores = checked[position]
                _difference_stats(stats, clip, scores, cpu_clip, cpu_scores, model_ids, scorer.runtime)
                clip, scores = cpu_clip, cpu_scores
                stats["cpu_rechecked_sessions"] += 1
            stats["gpu_sessions"] += 1
            stats["gpu_frames"] += len(scores)
            stats["cpu_rechecked_rate"] = stats["cpu_rechecked_sessions"] / stats["gpu_sessions"]
            clip["evaluation_compute"] = {"backend": "cpu_reference_boundary_recheck" if corrected else "cuda_batch_fp32",
                "boundary_margin": boundary_margin, "whole_session_cpu_rechecked": corrected}
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
            pcm_bytes = frames * 1280 * 2
            if batch and (len(batch) >= batch_size or cached_bytes + pcm_bytes > max_audio_bytes):
                yield from flush(batch)
                batch, cached_bytes = [], 0
            if pcm_bytes > max_audio_bytes:
                # The caller's post-consumption check runs only on the next
                # iterator advance. A streamed file must be checked before
                # yielding, so a changed session can never be committed first.
                path = Path(path).resolve()
                before_stat = path.stat()
                file_identity = lambda value: (value.st_dev, value.st_ino,
                    value.st_size, value.st_mtime_ns, value.st_ctime_ns)
                bound_file(path.parent, {"path": str(path), "sha256": row["sha256"]})
                row = {**row, "path": str(path)}
                clip, scores = score_session(scorer.reference, row, path, model_ids)
                if file_identity(path.stat()) != file_identity(before_stat):
                    raise ValueError("GPU evaluation streamed input changed during CPU scoring")
                clip["evaluation_compute"] = {"backend": "cpu_reference_oversized_stream", "whole_session_cpu_rechecked": False}
                stats["cpu_long_sessions"] += 1
                stats["cpu_long_frames"] += len(scores)
                stats["elapsed_wall_seconds"] = time.perf_counter() - began
                yield index, clip, scores
                continue
            before = time.perf_counter()
            pcm, rate = sf.read(path, frames=frames * 1280, dtype="int16")
            stats["pcm_read_wall_seconds"] += time.perf_counter() - before
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != frames * 1280:
                raise ValueError("GPU evaluation input changed during PCM caching")
            row = {**row, "path": str(path)}
            batch.append((index, row, path, pcm))
            cached_bytes += pcm.nbytes
            stats["maximum_cached_pcm_bytes"] = max(stats["maximum_cached_pcm_bytes"], cached_bytes)
        if batch:
            yield from flush(batch)
    finally:
        rechecker.close()
        stats["elapsed_wall_seconds"] = time.perf_counter() - began
