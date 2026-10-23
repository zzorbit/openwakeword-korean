"""GPU batching is opt-in; boundary sessions retain whole-session CPU decisions."""
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, write_json
from korean_openwakeword.evaluation_workers import score_session


MODEL_IDS = ["head_a", "head_b", "head_c", "mean"]
RUNTIME = {"threshold": .5, "consecutive_frames": 2, "cooldown_s": .16,
           "warmup_s": .16, "call_postroll_s": .08}
QUALIFICATION = {"kind": "isolated-test-qualification", "user_approved_numerical_differences": True}


def _scores(members):
    values = np.asarray(members, dtype=np.float32)
    mean = ((values[:, 0].astype(np.float64) + values[:, 1].astype(np.float64)
             + values[:, 2].astype(np.float64)) / 3).astype(np.float32)
    return np.column_stack((values, mean))


def _wave(tmp_path, marker, *, frames=8, tail=0, index=100, kind="negative"):
    path = tmp_path / f"session_{marker}.wav"
    sf.write(path, np.full(frames * 1280 + tail, marker, np.int16), 16000, subtype="PCM_16")
    row = {"sample_id": f"source_{marker}", "source_id": f"fixture:{marker}",
           "kind": kind, "category": "environment", "path": str(path), "sha256": digest(path),
           "split": "validation", "provenance": "isolated_test_fixture", "calls": []}
    if kind == "positive":
        row["calls"] = [{"start_s": .24, "end_s": .48}]
    return index, row, path


def _fake_scorer(monkeypatch, gpu_values, cpu_values):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    batches, cpu_reads = [], []
    class Reference:
        runtime = RUNTIME
        def reset(self):
            self.position = 0
        def score_members(self, pcm):
            marker = int(pcm[0])
            cpu_reads.append((marker, self.position))
            result = cpu_values[marker][self.position, :3]
            self.position += 1
            return result
    class Scorer:
        def __init__(self, bundle, model_ids, stats):
            assert model_ids == MODEL_IDS
            self.runtime = RUNTIME
            self.reference = Reference()
        def score_batch(self, pcms):
            markers = [int(pcm[0]) for pcm in pcms]
            batches.append(markers)
            assert all(len(pcm) % 1280 == 0 for pcm in pcms)
            return [gpu_values[marker][:len(pcm) // 1280].copy() for marker, pcm in zip(markers, pcms)]
    monkeypatch.setattr(module, "_GpuBatchScorer", Scorer)
    return module, batches, cpu_reads


@pytest.mark.parametrize("near_column", ["member", "mean"])
def test_any_eligible_member_or_mean_boundary_rechecks_the_whole_session(monkeypatch, tmp_path, near_column):
    actual = _scores([[.9, .8, .7]] * 8)
    actual[3] = _scores([[.50025, .1, .2] if near_column == "member" else [.7, .6, .2]])[0]
    reference = _scores([[.05, .1, .15]] * 8)
    module, _, cpu_reads = _fake_scorer(monkeypatch, {10: actual}, {10: reference})
    item = _wave(tmp_path, 10, index=19450, kind="positive")
    stats = {}
    result = list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [item], qualification=QUALIFICATION,
                  boundary_margin=.001, cpu_recheck_workers=1, stats=stats))
    index, clip, scores = result[0]
    expected, expected_scores = score_session(
        module._GpuBatchScorer(tmp_path, MODEL_IDS, {}).reference, item[1], item[2], MODEL_IDS)
    assert index == 19450 and np.array_equal(scores.view("u4"), expected_scores.view("u4"))
    assert np.array_equal(scores.view("u4"), reference.view("u4"))
    assert clip["results"] == expected["results"]
    assert stats["cpu_rechecked_sessions"] == 1
    assert {frame for marker, frame in cpu_reads if marker == 10} == set(range(8))


def test_warmup_only_boundary_is_ignored_and_away_scores_remain_gpu(monkeypatch, tmp_path):
    actual = _scores([[.1, .2, .3]] * 8)
    actual[0] = _scores([[.5001, .1, .2]])[0]
    reference = _scores([[.05, .1, .15]] * 8)
    module, _, cpu_reads = _fake_scorer(monkeypatch, {10: actual}, {10: reference})
    stats = {}
    output = list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [_wave(tmp_path, 10)],
                  qualification=QUALIFICATION, cpu_recheck_workers=1, stats=stats))
    assert np.array_equal(output[0][2].view("u4"), actual.view("u4"))
    assert not cpu_reads and stats.get("cpu_rechecked_sessions", 0) == 0


@pytest.mark.parametrize("near_frame,rechecked", [(2, False), (3, True)])
def test_boundary_eligibility_uses_the_same_non_aligned_warmup_as_cpu(monkeypatch, tmp_path, near_frame, rechecked):
    monkeypatch.setitem(RUNTIME, "warmup_s", .17)
    actual = _scores([[.1, .2, .3]] * 8)
    actual[near_frame] = _scores([[.5001, .1, .2]])[0]
    reference = _scores([[.05, .1, .15]] * 8)
    module, _, _ = _fake_scorer(monkeypatch, {10: actual}, {10: reference})
    stats = {}
    output = list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [_wave(tmp_path, 10)],
                  qualification=QUALIFICATION, cpu_recheck_workers=1, stats=stats))
    assert stats.get("cpu_rechecked_sessions", 0) == int(rechecked)
    assert np.array_equal(output[0][2], reference if rechecked else actual)


def test_batch_order_tail_drop_and_negative_reset_history_are_preserved(monkeypatch, tmp_path):
    values = {10: _scores([[.9, .8, .7]] * 8),
              20: _scores([[.1, .2, .3]] * 5),
              30: _scores([[.8, .7, .6]] * 8)}
    module, batches, cpu_reads = _fake_scorer(monkeypatch, values, values)
    items = [_wave(tmp_path, 10, tail=777, index=512),
             _wave(tmp_path, 20, frames=5, tail=1, index=513),
             _wave(tmp_path, 30, index=514)]
    output = list(module.score_gpu_sessions(tmp_path, MODEL_IDS, iter(items), batch_size=2,
                  qualification=QUALIFICATION, cpu_recheck_workers=1))
    assert [index for index, _, _ in output] == [512, 513, 514]
    assert batches == [[10, 20], [30]] and not cpu_reads
    for (_, row, path), (_, clip, scores) in zip(items, output):
        assert clip["sample_id"] == row["sample_id"]
        assert clip["duration_s"] == len(scores) * .08
        assert len(scores) == sf.info(path).frames // 1280
        assert np.array_equal(scores.view("u4"), values[int(sf.read(path, dtype="int16")[0][0])].view("u4"))
    assert output[1][1]["results"]["head_a"]["false_activations"] == 0


def test_audio_memory_budget_reduces_batch_size_without_changing_inputs(monkeypatch, tmp_path):
    values = {marker: _scores([[.1, .2, .3]] * 8) for marker in (10, 20, 30)}
    module, batches, _ = _fake_scorer(monkeypatch, values, values)
    items = [_wave(tmp_path, marker, index=index) for index, marker in enumerate(values)]
    budget = 8 * 1280 * 2 + 16
    result = list(module.score_gpu_sessions(tmp_path, MODEL_IDS, items, batch_size=128,
                  qualification=QUALIFICATION, max_audio_bytes=budget, cpu_recheck_workers=1))
    assert batches == [[10], [20], [30]]
    assert [index for index, _, _ in result] == [0, 1, 2]
    assert all(np.array_equal(scores, values[int(index * 10 + 10)]) for index, _, scores in result)


def test_long_single_session_streams_through_cpu_instead_of_exceeding_gpu_audio_budget(monkeypatch, tmp_path):
    values = {10: _scores([[.1, .2, .3]] * 8)}
    module, batches, cpu_reads = _fake_scorer(monkeypatch, values, values)
    stats = {}
    result = list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [_wave(tmp_path, 10)],
                  qualification=QUALIFICATION, max_audio_bytes=1024, cpu_recheck_workers=1, stats=stats))
    assert not batches and len(cpu_reads) == 8
    assert stats["cpu_long_sessions"] == 1 and stats.get("cpu_rechecked_sessions", 0) == 0
    assert np.array_equal(result[0][2], values[10])


def test_long_cpu_stream_changed_during_scoring_never_yields_a_committable_session(monkeypatch, tmp_path):
    values = {10: _scores([[.1, .2, .3]] * 8)}
    module, _, _ = _fake_scorer(monkeypatch, values, values)
    item = _wave(tmp_path, 10)
    original = module.score_session
    def changed(detector, row, path, model_ids):
        result = original(detector, row, path, model_ids)
        before = Path(path).stat()
        with Path(path).open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        return result
    monkeypatch.setattr(module, "score_session", changed)
    published = []
    with pytest.raises(ValueError, match="(?i)(changed|verification)"):
        for result in module.score_gpu_sessions(tmp_path, MODEL_IDS, [item], qualification=QUALIFICATION,
                      max_audio_bytes=1024, cpu_recheck_workers=1):
            published.append(result)
    assert not published


def test_gpu_requires_explicit_numerical_difference_approval_before_loading(monkeypatch, tmp_path):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    monkeypatch.setattr(module, "_GpuBatchScorer", lambda *args, **kwargs: pytest.fail("Unapproved GPU cannot initialize"))
    for qualification in (None, {}, {"user_approved_numerical_differences": False}):
        with pytest.raises(ValueError, match="(?i)(approv|qualif|numerical)"):
            list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [], qualification=qualification, cpu_recheck_workers=1))


@pytest.mark.parametrize("parameter,value", [("batch_size", 0), ("batch_size", True),
    ("boundary_margin", -1), ("boundary_margin", float("nan")), ("max_audio_bytes", 0)])
def test_invalid_gpu_workload_options_fail_before_any_model_load(monkeypatch, tmp_path, parameter, value):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    monkeypatch.setattr(module, "_GpuBatchScorer", lambda *args, **kwargs: pytest.fail("Invalid GPU options cannot initialize"))
    options = {"qualification": QUALIFICATION, "cpu_recheck_workers": 1, parameter: value}
    with pytest.raises(ValueError):
        list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [], **options))


@pytest.mark.parametrize("defect", ["shape", "precision", "nonfinite", "probability", "mean"])
def test_unexpected_score_shape_or_arithmetic_fails_closed(monkeypatch, tmp_path, defect):
    values = _scores([[.1, .2, .3]] * 8)
    if defect == "shape":
        values = values[:, :2]
    elif defect == "precision":
        values = values.astype(np.float64)
    elif defect == "nonfinite":
        values[0, 0] = np.nan
    elif defect == "probability":
        values[0, 0] = 1.1
    else:
        values[0, 3] += .001
    malformed = {10: values}
    module, _, _ = _fake_scorer(monkeypatch, malformed, malformed)
    with pytest.raises(ValueError, match="(?i)(shape|column|score|mean)"):
        list(module.score_gpu_sessions(tmp_path, MODEL_IDS, [_wave(tmp_path, 10)],
             qualification=QUALIFICATION, cpu_recheck_workers=1))


def test_cpu_only_runtime_is_rejected_instead_of_silent_gpu_fallback(monkeypatch):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    import onnxruntime as ort
    monkeypatch.delenv("KOWW_GPU_RUNTIME_PATH", raising=False)
    monkeypatch.delenv("KOWW_GPU_DLL_PATHS", raising=False)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="cannot silently become CPU"):
        module._cuda_runtime()


@pytest.mark.parametrize("providers,tf32", [(["CPUExecutionProvider"], "0"),
                                             (["CUDAExecutionProvider", "CPUExecutionProvider"], "1")])
def test_actual_session_provider_and_tf32_are_verified(monkeypatch, tmp_path, providers, tf32):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    calls = []
    class Session:
        def get_providers(self):
            return providers
        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"use_tf32": tf32}}
    def create(source, *, sess_options, providers):
        calls.append((sess_options, providers))
        return Session()
    ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(), InferenceSession=create,
                          __version__="test-only", __file__=str(tmp_path / "fake_ort.py"))
    class Reference:
        manifest = {"members": [{"path": name + ".onnx", "sha256": str(index) * 64}
                                 for index, name in enumerate(MODEL_IDS[:3])],
                    "frontend": [{"path": "melspectrogram.onnx"}, {"path": "embedding_model.onnx"}]}
        runtime = RUNTIME
        def __init__(self, bundle):
            pass
    monkeypatch.setattr(module, "_cuda_runtime", lambda: (ort, []))
    monkeypatch.setattr(module, "WakeWordDetector", Reference)
    monkeypatch.setattr(module, "_per_session_mel_model", lambda path: (b"isolated-graph", {}))
    with pytest.raises(RuntimeError, match="(?i)(fell back|TF32)"):
        module._GpuBatchScorer(tmp_path, MODEL_IDS, {})
    assert calls[0][1] == [("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT",
        "gpu_mem_limit": str(2 * 1024 ** 3), "do_copy_in_default_stream": "1"})]
    assert calls[0][0].intra_op_num_threads == calls[0][0].inter_op_num_threads == 1


@pytest.mark.parametrize("samples", [1280, 1760])
def test_derived_mel_batch_matches_independent_cpu_sessions_without_cross_session_floor(samples):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    from korean_openwakeword.runtime import sample_bundle
    import onnxruntime as ort
    original = sample_bundle() / "frontend/melspectrogram.onnx"
    before = digest(original)
    derived, lineage = module._per_session_mel_model(original)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(original), sess_options=options, providers=["CPUExecutionProvider"])
    batched = ort.InferenceSession(derived, sess_options=options, providers=["CPUExecutionProvider"])
    base = np.random.default_rng(101).normal(0, 3000, samples).astype(np.float32)
    audio = np.stack([base * gain for gain in (1, .3, .1, .03, .01, .001, .0001, 0)]).astype(np.float32)
    independent = np.concatenate([reference.run(None, {reference.get_inputs()[0].name: value[None]})[0]
                                   for value in audio])
    old_batch = reference.run(None, {reference.get_inputs()[0].name: audio})[0]
    new_batch = batched.run(None, {batched.get_inputs()[0].name: audio})[0]
    assert not np.array_equal(old_batch.view("u4"), independent.view("u4")), "Fixture must expose the original cross-session coupling"
    assert np.array_equal(new_batch.view("u4"), independent.view("u4"))
    assert lineage["source_sha256"] == before and lineage["source_model_modified"] is False
    assert digest(original) == before


def test_unrecognized_mel_or_changed_opset_fails_closed(monkeypatch, tmp_path):
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    from korean_openwakeword.runtime import sample_bundle
    import onnx
    wrong = tmp_path / "wrong.onnx"
    wrong.write_bytes(b"not the pinned model")
    with pytest.raises(ValueError, match="audited pinned"):
        module._per_session_mel_model(wrong)
    original = sample_bundle() / "frontend/melspectrogram.onnx"
    model = onnx.load(original)
    model.opset_import[0].version += 1
    onnx.save(model, wrong)
    # Even a deliberately overridden test pin must not bypass semantic checks.
    monkeypatch.setitem(module.PINNED_ASSETS, "melspectrogram.onnx", digest(wrong))
    with pytest.raises(ValueError, match="opset changed"):
        module._per_session_mel_model(wrong)


def test_cpu_resume_identity_is_unchanged_and_gpu_policy_and_prefix_are_bound(tmp_path):
    from korean_openwakeword.evaluate import evaluation_binding
    from korean_openwakeword.runtime import bundle_candidates, deployment_candidate
    project, manifest, bundle = tmp_path / "project.json", tmp_path / "manifest.jsonl", tmp_path / "bundle"
    write_json(project, {"test": "unchanged-config"})
    manifest.write_text("fixed-inputs\n", encoding="utf-8")
    package = {"members": [{"path": name + ".onnx", "sha256": str(index) * 64}
                           for index, name in enumerate(MODEL_IDS[:3])]}
    write_json(bundle / "manifest.json", package)
    legacy = identity({"project_sha256": digest(project), "bundle_sha256": digest(bundle / "manifest.json"),
        "manifest_sha256": digest(manifest), "stage": "development", "candidate": deployment_candidate(package)["id"],
        "smoke": False, "model_ids": [row["id"] for row in bundle_candidates(package)],
        "scoring_protocol": "reference_cpu_session_v1"})
    assert evaluation_binding(project, bundle, manifest) == legacy
    prefix = {"committed_sessions": 64, "cpu_binding": "exact-previous-cpu-binding",
              "frame_prefix_sha256": "1" * 64, "journal_prefix_sha256": "2" * 64}
    original = evaluation_binding(project, bundle, manifest, backend="cuda", cpu_prefix=prefix)
    assert original != legacy
    assert evaluation_binding(project, bundle, manifest, backend="cuda", cpu_prefix=prefix,
                              boundary_margin=.002) != original
    assert evaluation_binding(project, bundle, manifest, backend="cuda", cpu_prefix=prefix,
                              gpu_batch_size=64) != original
    assert evaluation_binding(project, bundle, manifest, backend="cuda",
                              cpu_prefix={**prefix, "frame_prefix_sha256": "3" * 64}) != original
    with pytest.raises(ValueError, match="backend"):
        evaluation_binding(project, bundle, manifest, backend="silently-auto")


@pytest.mark.parametrize("stage,approved,message", [("development", False, "explicit approval"),
                                                       ("final", True, "Development only")])
def test_evaluate_rejects_unapproved_gpu_and_independent_final_before_any_scoring(tmp_path, stage, approved, message):
    from korean_openwakeword.evaluate import evaluate
    from korean_openwakeword.project import create_project
    from korean_openwakeword.runtime import sample_bundle
    project = create_project(tmp_path / "project", "춘식아", "chunsik")
    with pytest.raises(ValueError, match=message):
        evaluate(project, sample_bundle(), tmp_path / "forbidden", stage=stage,
                 backend="cuda", numerical_differences_approved=approved)
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("defect,message", [("hash", "Missing or changed"),
    ("header", "mono16kHz PCM16"), ("call", "call boundaries"),
    ("interval", "source interval"), ("frames", "frame count")])
def test_exact_use_gpu_input_validation_rejects_each_immutable_input_defect(tmp_path, defect, message):
    from korean_openwakeword.evaluate import _verified_gpu_items
    _, row, path = _wave(tmp_path, 10)
    offsets = [0, 8]
    if defect == "hash":
        row["sha256"] = "0" * 64
    elif defect == "header":
        values, rate = sf.read(path, dtype="int16")
        sf.write(path, np.column_stack([values, values]), rate, subtype="PCM_16")
        row["sha256"] = digest(path)
    elif defect == "call":
        row["calls"] = [{"start_s": .24, "end_s": 99}]
    elif defect == "interval":
        row.update(start_s=1, end_s=0)
    else:
        offsets = [0, 9]
    with pytest.raises(ValueError, match=message):
        list(_verified_gpu_items(tmp_path, [row], [path], offsets, 0))


def test_exact_use_input_validation_detects_mutation_and_does_not_reopen_cpu_prefix(tmp_path):
    from korean_openwakeword.evaluate import _verified_gpu_items
    _, row, path = _wave(tmp_path, 10)
    absent = tmp_path / "already-committed-prefix-need-not-be-reopened.wav"
    iterator = _verified_gpu_items(tmp_path, [{"path": str(absent)}, row], [absent, path], [0, 8, 16], 1)
    assert next(iterator)[0] == 1
    before = path.stat()
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match="changed during GPU consumption"):
        next(iterator)


def test_default_evaluation_still_uses_reference_cpu_without_entering_gpu(monkeypatch, tmp_path):
    from korean_openwakeword import evaluate
    from test_korean_library_evaluation_preflight import _evaluation_inputs
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    monkeypatch.setattr(module, "score_gpu_sessions", lambda *args, **kwargs: pytest.fail("CPU default must not enter CUDA"))
    project, root, bundle, manifest, _ = _evaluation_inputs(tmp_path)
    result = evaluate.evaluate(project, bundle, root / "reports/cpu-default", manifest=manifest, workers=1, smoke=True)
    assert result["execution"]["backend"] == "cpu"
    assert result["evaluation_binding"] == evaluate.evaluation_binding(project, bundle, manifest, smoke=True)


@pytest.mark.skipif(not os.environ.get("KOWW_GPU_QUALIFICATION_BUNDLE"), reason="Explicit isolated CUDA qualification only")
def test_real_threshold_tie_rechecks_cpu_and_leaves_model_and_pcm_unchanged():
    module = importlib.import_module("korean_openwakeword.gpu_evaluation")
    from korean_openwakeword.runtime import WakeWordDetector, bundle_candidates
    bundle = Path(os.environ["KOWW_GPU_QUALIFICATION_BUNDLE"])
    manifest = Path(os.environ["KOWW_GPU_QUALIFICATION_MANIFEST"])
    identifier = os.environ.get("KOWW_GPU_QUALIFICATION_SAMPLE", "original_0879_reverb_moderate_music_section1_snr0")
    with manifest.open(encoding="utf-8") as stream:
        row = next(json.loads(line) for line in stream if json.loads(line)["sample_id"] == identifier)
    detector = WakeWordDetector(bundle)
    ids = [item["id"] for item in bundle_candidates(detector.manifest)]
    path = Path(row["path"])
    bindings = {str(bundle / item["path"]): item["sha256"] for item in detector.manifest["members"] + detector.manifest["frontend"]}
    bindings[str(path)] = row["sha256"]
    expected, cpu = score_session(detector, row, path, ids)
    if identifier == "original_0879_reverb_moderate_music_section1_snr0":
        assert cpu[46, ids.index("member2_seed20260929")] == np.float32(detector.runtime["threshold"])
    stats = {}
    actual = list(module.score_gpu_sessions(bundle, ids, [(19450, row, path)], batch_size=128,
                  qualification=QUALIFICATION, boundary_margin=.001, cpu_recheck_workers=1, stats=stats))
    assert actual[0][0] == 19450
    assert stats["cpu_rechecked_sessions"] == 1
    assert np.array_equal(actual[0][2].view("u4"), cpu.view("u4"))
    assert actual[0][1]["results"] == expected["results"]
    assert all(digest(path) == sha for path, sha in bindings.items())
