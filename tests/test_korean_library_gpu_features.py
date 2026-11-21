"""The CUDA feature adapter keeps independent audio windows and cache identity."""
from contextlib import contextmanager

import numpy as np
import pytest

from korean_openwakeword import features, gpu_features, gpu_runtime_install, studio_steps
from korean_openwakeword.common import write_json
from korean_openwakeword.project import BATCH, create_project, load_project


def test_batched_cuda_layout_matches_the_original_streaming_geometry():
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    pytest.importorskip("openwakeword")
    from importlib.resources import files
    from korean_openwakeword.frontend import StreamingFrontend
    from korean_openwakeword.gpu_evaluation import _per_session_mel_model

    assets = files("korean_openwakeword") / "resources/sample_chunsik/frontend"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    # The same ONNX models/packing run on CPU here so geometry is tested on CI.
    adapter = gpu_features.CudaFeatureFrontend.__new__(gpu_features.CudaFeatureFrontend)
    adapter.mel = ort.InferenceSession(_per_session_mel_model(assets / "melspectrogram.onnx")[0],
                                      sess_options=options, providers=["CPUExecutionProvider"])
    adapter.embedding = ort.InferenceSession(str(assets / "embedding_model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
    adapter.mel_input = adapter.mel.get_inputs()[0].name
    adapter.embedding_input = adapter.embedding.get_inputs()[0].name
    rng = np.random.default_rng(20260928)
    clips = rng.integers(-3000, 3001, (4, 32000), dtype=np.int16)
    clips[0] = 0
    clips[1, 16000:] = 32767
    expected = StreamingFrontend(assets).embed(clips)
    actual = adapter._block(clips)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(actual.view("u4"), adapter._block(clips[::-1])[::-1].view("u4"))
    np.testing.assert_array_equal(actual[-1:].view("u4"), adapter._block(clips[-1:]).view("u4"))


def test_feature_backends_preserve_bound_pools_and_forbid_mixed_features(tmp_path):
    generic, korean, positive = (tmp_path / role for role in ("generic", "korean", "positive"))
    directories = [generic, korean, positive]
    write_json(generic / "metadata.json", {"legacy_generic_bridge": True})
    assert gpu_features.feature_backend(directories, "cu128") == "cuda"
    assert gpu_features.feature_backend(directories, "cpu") == "cpu"
    assert gpu_features.feature_backend(directories, None) == "cpu"
    write_json(korean / "contract.json", {"frontend": {"provider": "CUDAExecutionProvider"}})
    assert gpu_features.feature_backend(directories, None) == "cuda"
    with pytest.raises(ValueError, match="계산 장치"):
        gpu_features.feature_backend(directories, "cpu")
    write_json(positive / "contract.json", {"frontend": {"provider": "CPUExecutionProvider"}})
    with pytest.raises(ValueError, match="서로 다릅니다"):
        gpu_features.feature_backend(directories, None)
    (positive / "contract.json").unlink()
    write_json(korean / "contract.json", {"frontend": {"provider": "CPUExecutionProvider"}})
    with pytest.raises(ValueError, match="계산 장치"):
        gpu_features.feature_backend(directories, "cu128")
    write_json(korean / "metadata.json", {})
    write_json(positive / "metadata.json", {"frontend": {"provider": "CPUExecutionProvider"}})
    assert gpu_features.feature_backend(directories, "cu128") == "cpu"


@pytest.mark.parametrize("runtime", ("managed", "missing", "explicit"))
def test_studio_selected_cuda_is_used_in_every_new_extracted_pool(tmp_path, monkeypatch, runtime):
    settings = studio_steps.preset_settings("check")
    project = create_project(tmp_path / "wake", "보라야", "bora", positive_originals=settings["originals"],
                             augmentation_rounds=settings["rounds"], updates=settings["updates"],
                             generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    root, config = load_project(project)
    directories = {role: root / config["pool_features"][role] for role in BATCH}
    write_json(directories["generic_negative"] / "metadata.json", {"legacy_generic_bridge": True})
    calls, installations = [], []
    frontend = type("TestCudaFrontend", (), {"qualification": {"adopted": "qualified_CUDA_FP32"}})()

    @contextmanager
    def resource(expected_root, directory, device):
        assert expected_root == root and directory == root / config["frontend_dir"] and device == 0
        assert installations == ([root] if runtime == "missing" else [])
        yield frontend

    def extract(path, role, *, resume, frontend, pause_requested):
        assert pause_requested is None
        assert path == project and frontend is not None
        calls.append((role, resume, frontend))
        return {"feature_rows": settings["originals"] if role == "positive" else 3}

    monkeypatch.setattr(gpu_features, "cuda_feature_frontend", resource)
    monkeypatch.setattr(features, "build_features", extract)
    monkeypatch.setattr(features, "verify_cache", lambda directory, **kwargs: {"feature_rows": 5625000})
    monkeypatch.delenv("KOWW_GPU_RUNTIME_PATH", raising=False)
    if runtime == "explicit":
        monkeypatch.setenv("KOWW_GPU_RUNTIME_PATH", str(tmp_path / "runtime"))
    monkeypatch.setattr(gpu_runtime_install, "_managed_reason", lambda: None)
    monkeypatch.setattr(gpu_runtime_install, "gpu_runtime_environment",
                        lambda directory: {"KOWW_GPU_RUNTIME_PATH": str(directory / "runtime")} if runtime == "managed" else {})
    monkeypatch.setattr(studio_steps, "_prepare_evaluation_runtime", installations.append)
    result = studio_steps.run_stage(project, tmp_path, settings, "features",
                                    {"inputs": {"dependency_backend": "cu128"}})
    assert calls == [(role, False, frontend) for role in BATCH if role != "generic_negative"]
    assert result["result"]["cuda_feature_qualification"] == frontend.qualification
