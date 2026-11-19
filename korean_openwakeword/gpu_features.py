"""CUDA FP32 embeddings for independent two-second training examples."""
from contextlib import contextmanager
import os
from pathlib import Path
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .common import digest, read_json
from .frontend import CLIP_SAMPLES, FEATURE_SHAPE, HOP_SAMPLES, PINNED_ASSETS, StreamingFrontend
from .gpu_evaluation import _cuda_runtime, _per_session_mel_model, evaluation_runtime_environment
from .gpu_runtime_install import gpu_runtime_environment
from .task_progress import emit as report_task_progress


_CUDA = "CUDAExecutionProvider"
_CLIP_BATCH = 16
_CLIP_HOPS = CLIP_SAMPLES // HOP_SAMPLES
_MEL_WINDOW = 76
_MEL_BANDS = 32
_MEL_LOOKBACK = 480
_QUALIFICATION_ATOL = 1e-3
_QUALIFICATION_RTOL = 1e-4


def feature_backend(directories, selected):
    """Respect the numerical contract shared by every extracted feature pool."""
    incomplete = False
    recorded = set()
    for directory in directories:
        directory = Path(directory)
        complete = (directory / "metadata.json").is_file()
        contract = directory / "contract.json"
        if contract.is_file():
            frontend = read_json(contract).get("frontend", {})
        elif complete:
            metadata = read_json(directory / "metadata.json")
            if metadata.get("legacy_generic_bridge") or "frontend" not in metadata:
                continue
            frontend = metadata["frontend"]
        else:
            frontend = None
        incomplete |= not complete
        if frontend is not None:
            recorded.add("cuda" if frontend.get("provider") == _CUDA else "cpu")
    if len(recorded) > 1:
        raise ValueError("저장된 특징의 CPU/GPU 계산 방식이 서로 다릅니다.")
    requested = {"cu128": "cuda", "cpu": "cpu"}.get(selected)
    existing = next(iter(recorded), None)
    if not incomplete:
        return existing or "cpu"
    if existing and requested and requested != existing:
        raise ValueError("이미 작성한 특징의 계산 장치가 현재 선택과 다릅니다. "
                         "기존 특징은 기록된 장치에서 이어가거나 새 프로젝트에서 다시 추출하세요.")
    return existing or requested or "cpu"


@contextmanager
def cuda_feature_frontend(root, frontend_directory, device_id=0):
    """Use the installed project-owned ONNX CUDA runtime for the whole stage."""
    environment = {} if os.environ.get("KOWW_GPU_RUNTIME_PATH") else gpu_runtime_environment(root)
    with evaluation_runtime_environment(environment):
        yield CudaFeatureFrontend(frontend_directory, device_id)


class CudaFeatureFrontend:
    def __init__(self, directory, device_id):
        self.directory = Path(directory).resolve()
        mel_path = self.directory / "melspectrogram.onnx"
        embedding_path = self.directory / "embedding_model.onnx"
        if digest(embedding_path) != PINNED_ASSETS["embedding_model.onnx"]:
            raise ValueError("CUDA 특징 추출은 검증된 임베딩 모델이 필요합니다.")
        self.ort, self.dll_handles = _cuda_runtime()
        from onnxruntime.capi import onnxruntime_pybind11_state

        mel_model, lineage = _per_session_mel_model(mel_path)
        provider_options = {"device_id": str(device_id), "use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT",
                            "gpu_mem_limit": str(2 * 1024 ** 3), "do_copy_in_default_stream": "1"}
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        def session(source):
            model = self.ort.InferenceSession(source, sess_options=options, providers=[(_CUDA, provider_options)])
            actual = model.get_provider_options().get(_CUDA, {})
            if (model.get_providers()[:1] != [_CUDA] or actual.get("use_tf32") != "0"
                    or actual.get("device_id") != str(device_id)):
                raise RuntimeError("CUDA 특징 추출 세션이 선택한 GPU와 FP32 설정으로 실행되지 않았습니다.")
            model.disable_fallback()
            return model, model.get_inputs()[0].name

        self.mel, self.mel_input = session(mel_model)
        self.embedding, self.embedding_input = session(str(embedding_path))
        self.identity = {
            "schema_version": 1, "algorithm": "independent_80ms_last16_cuda_fp32_v1",
            "model_sha256": {"melspectrogram": PINNED_ASSETS[mel_path.name],
                             "embedding": PINNED_ASSETS[embedding_path.name]},
            "mel_derivation": lineage, "adapter_sha256": digest(__file__),
            "provider": _CUDA, "device_id": device_id, "onnxruntime_version": self.ort.__version__,
            "onnxruntime_native_sha256": digest(onnxruntime_pybind11_state.__file__),
            "numpy_version": np.__version__, "clip_batch": _CLIP_BATCH,
            "input": "16kHz_PCM16", "clip_samples": CLIP_SAMPLES, "feature_shape": list(FEATURE_SHAPE),
            "hop_samples": HOP_SAMPLES, "precision": "FP32; TF32 disabled",
        }
        self.qualification = None

    def _mel(self, windows):
        outputs = self.mel.run(None, {self.mel_input: np.ascontiguousarray(windows, dtype=np.float32)})[0]
        return outputs.reshape(len(windows), -1, _MEL_BANDS) / 10 + 2

    def _block(self, clips):
        count = len(clips)
        # Keep the same CUDA tensor geometry after a pause and for a short tail.
        padded = np.zeros((_CLIP_BATCH, CLIP_SAMPLES), dtype=np.int16)
        padded[:count] = clips
        first = self._mel(padded[:, :HOP_SAMPLES])
        window_size = HOP_SAMPLES + _MEL_LOOKBACK
        windows = sliding_window_view(padded, window_size, axis=1)[:, HOP_SAMPLES-_MEL_LOOKBACK::HOP_SAMPLES, :]
        following = self._mel(windows.reshape(-1, window_size))
        after_first = following.reshape(_CLIP_BATCH, _CLIP_HOPS-1, -1, _MEL_BANDS)
        mel_steps = after_first.shape[2]
        if windows.shape[1] != _CLIP_HOPS-1:
            raise ValueError("CUDA 특징 추출 입력 프레임 수가 모델과 맞지 않습니다.")
        history = np.concatenate([
            np.ones((_CLIP_BATCH, _MEL_WINDOW, _MEL_BANDS), dtype=np.float32), first,
            after_first.reshape(_CLIP_BATCH, -1, _MEL_BANDS),
        ], axis=1)
        starts = history.shape[1] - _MEL_WINDOW - np.arange(FEATURE_SHAPE[0]-1, -1, -1) * mel_steps
        retained = sliding_window_view(history, _MEL_WINDOW, axis=1)[:, starts, :, :].swapaxes(-1, -2)
        inputs = np.ascontiguousarray(retained.reshape(-1, _MEL_WINDOW, _MEL_BANDS, 1))
        features = self.embedding.run(None, {self.embedding_input: inputs})[0]
        values = np.asarray(features, dtype=np.float32).reshape(_CLIP_BATCH, *FEATURE_SHAPE)[:count].copy()
        if not np.isfinite(values).all():
            raise ValueError("CUDA 특징 추출 결과에 유효하지 않은 값이 있습니다.")
        return values

    def _qualify(self, clips):
        report_task_progress("feature_frontend", "CUDA 특징 추출 모델 실행·동일성 확인", eta_eligible=False)
        actual = clips[:8]
        rng = np.random.default_rng(20260928)
        edges = np.zeros((4, CLIP_SAMPLES), dtype=np.int16)
        edges[1] = rng.integers(-100, 101, CLIP_SAMPLES, dtype=np.int16)
        edges[2, CLIP_SAMPLES//2:] = 32767
        edges[3, 10000] = -32768
        probes = np.concatenate((actual, edges))
        start = time.perf_counter()
        expected = StreamingFrontend(self.directory).embed(probes)
        cpu_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = self._block(probes)
        cuda_seconds = time.perf_counter() - start
        reversed_result = self._block(probes[::-1])[::-1]
        equal = np.array_equal(result.view("u4"), reversed_result.view("u4"))
        maximum = float(np.max(np.abs(expected - result)))
        if not equal or not np.allclose(expected, result, atol=_QUALIFICATION_ATOL, rtol=_QUALIFICATION_RTOL):
            raise RuntimeError(f"CUDA 특징 추출 검증을 통과하지 못했습니다 (최대 절대 오차 {maximum:.6g}).")
        self.qualification = {"adopted": "qualified_CUDA_FP32", "actual_source_clips": len(actual),
                              "edge_clips": len(edges), "maximum_absolute_error": maximum,
                              "batch_order_bitwise_equal": True, "canonical_elapsed_s": cpu_seconds,
                              "candidate_elapsed_s": cuda_seconds}

    def embed(self, clips):
        if (not isinstance(clips, np.ndarray) or clips.ndim != 2 or clips.dtype != np.int16
                or not len(clips) or clips.shape[1] != CLIP_SAMPLES):
            raise ValueError("CUDA 특징 추출은 비어 있지 않은 2초 16 kHz PCM16 음원이 필요합니다.")
        if self.qualification is None:
            self._qualify(clips)
        return np.concatenate([self._block(clips[start:start+_CLIP_BATCH])
                               for start in range(0, len(clips), _CLIP_BATCH)])
