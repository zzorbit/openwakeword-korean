"""Runtime optimization must preserve sampling and fail safely before loading."""
from contextlib import contextmanager
from itertools import cycle
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from korean_openwakeword.project import create_project
from korean_openwakeword.tts_runtime import CloneRuntime, generate_with_runtime
from korean_openwakeword import tts_runtime


def test_unapproved_and_final_locked_generation_never_loads_weights(tmp_path, monkeypatch):
    project = create_project(tmp_path / "word", "보라야", "bora")
    def forbidden(*args, **kwargs):
        pytest.fail("Weights loaded before approval/Final checks")
    monkeypatch.setattr(CloneRuntime, "_load", forbidden)
    with pytest.raises(ValueError, match="Approve"):
        generate_with_runtime(project)
    with pytest.raises(ValueError, match="Final"):
        generate_with_runtime(project, split="final")


def fake_cuda_runtime(tmp_path, monkeypatch, difference="none", warm_seconds=.5, cold_seconds=2, backend="auto"):
    # An independent CPU generator stands in for CUDA; the two RNG states
    # advance independently so a CUDA-only regression cannot escape the test.
    cuda_rng = torch.Generator(device="cpu")
    cuda_rng.manual_seed(812)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_rng.get_state().clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: cuda_rng.set_state(state.clone()))
    @contextmanager
    def fork_rng(devices):
        state = torch.get_rng_state().clone(); gpu_state = cuda_rng.get_state().clone()
        try:
            yield
        finally:
            torch.set_rng_state(state); cuda_rng.set_state(gpu_state)
    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    eager = object(); fast = object()
    predictor = SimpleNamespace(generate=eager)
    inner = SimpleNamespace(forward=eager)
    t = np.arange(9600)/16000
    class Model:
        model = SimpleNamespace(talker=SimpleNamespace(code_predictor=predictor, model=inner))
        graph_calls = 0
        stages = []
        def generate_voice_clone(self, **kwargs):
            value = torch.rand(1).item() + torch.rand(1, generator=cuda_rng).item()
            wave = np.rint((.1*np.sin(2*np.pi*250*t) + value*.002)*32768)/32768
            stage = "eager"
            if predictor.generate is fast:
                self.graph_calls += 1
                stage = "cold" if self.graph_calls == 1 else "warm" if self.graph_calls == 2 else "fast"
                if difference == stage + "_pcm": wave = wave + .03
                if difference == stage + "_qc": wave = wave + 1e-9
                if difference == stage + "_cpu_rng": torch.rand(1)
                if difference == stage + "_cuda_rng": torch.rand(1, generator=cuda_rng)
            self.stages.append(stage)
            return [wave], 16000
    runtime = CloneRuntime("unused", {}, tmp_path, backend=backend)
    runtime.model = Model()
    def enable():
        runtime.predictor = {"original_generate": eager}
        runtime.talker = {"original_forward": eager}
        predictor.generate = inner.forward = fast
    monkeypatch.setattr(runtime, "_enable", enable)
    clocks = cycle((0, 1, 0, cold_seconds, 0, warm_seconds))
    monkeypatch.setattr(tts_runtime, "time", SimpleNamespace(perf_counter=lambda: next(clocks)))
    return runtime, cuda_rng, eager, fast


def one_eager_call(runtime, cuda_rng):
    torch.manual_seed(73)
    before_cuda = cuda_rng.get_state().clone()
    expected, _ = runtime.model.generate_voice_clone()
    expected_rng = torch.get_rng_state().clone()
    expected_cuda = cuda_rng.get_state().clone()
    torch.manual_seed(73)
    cuda_rng.set_state(before_cuda)
    return expected, expected_rng, expected_cuda


@pytest.mark.parametrize("difference", ["none", "cold_pcm", "cold_qc", "cold_cpu_rng", "cold_cuda_rng",
                                       "warm_pcm", "warm_qc", "warm_cpu_rng", "warm_cuda_rng"])
def test_cuda_qualification_preserves_one_sample_rng_and_uses_eager_on_difference(tmp_path, monkeypatch, difference):
    runtime, cuda_rng, eager, fast = fake_cuda_runtime(tmp_path, monkeypatch, difference)
    expected, expected_rng, expected_cuda = one_eager_call(runtime, cuda_rng)
    prompt = [object()]
    result, _ = runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=prompt)
    assert np.array_equal(result[0], expected[0])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(cuda_rng.get_state(), expected_cuda)
    assert runtime.disabled is (difference != "none")
    assert runtime.report["qualification_clips_added_to_training"] == 0
    qualification = runtime.report["qualifications"][-1]
    if difference == "none":
        assert qualification["accepted"]
        assert qualification["graph_warm_seconds"] == .5
        assert qualification["minimum_warm_speedup_met"]
        assert qualification["warm_cpu_rng_equal"] and qualification["warm_cuda_rng_equal"]
    if difference != "none":
        assert not qualification["accepted"]
        if difference.endswith("_qc"):
            assert qualification[difference.split("_")[0] + "_pcm_samples_equal"]
            assert not qualification[difference.split("_")[0] + "_quality_checks_equal"]
        # Following requests must continue to use original eager methods.
        expected, expected_rng, expected_cuda = one_eager_call(runtime, cuda_rng)
        result, _ = runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=prompt)
        assert np.array_equal(result[0], expected[0])
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(cuda_rng.get_state(), expected_cuda)
        assert runtime.model.model.talker.code_predictor.generate is eager
        assert runtime.model.model.talker.model.forward is eager


@pytest.mark.parametrize("warm_seconds,accepted", [(.5, True), (.95, True), (.951, False), (1, False), (1.2, False)])
def test_only_equivalent_warm_backend_with_five_percent_speedup_is_selected(tmp_path, monkeypatch, warm_seconds, accepted):
    runtime, cuda_rng, eager, fast = fake_cuda_runtime(tmp_path, monkeypatch, warm_seconds=warm_seconds, cold_seconds=20)
    expected, expected_rng, expected_cuda = one_eager_call(runtime, cuda_rng)
    prompt = [object()]
    result, _ = runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=prompt)
    assert np.array_equal(result[0], expected[0])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(cuda_rng.get_state(), expected_cuda)
    qualification = runtime.report["qualifications"][-1]
    assert qualification["accepted"] is accepted
    assert qualification["minimum_warm_speedup_met"] is accepted
    assert qualification["graph_cold_seconds"] == 20  # Cold cost never selects the backend.
    assert qualification["graph_warm_seconds"] == warm_seconds
    assert runtime.disabled is (not accepted)
    if accepted:
        # The same reference is not qualified or repeated on later requests.
        runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=prompt)
        assert len(runtime.report["qualifications"]) == 1
        assert runtime.model.stages[-1] == "fast"
    else:
        assert "5% faster" in qualification["reason"]
        assert runtime.report["actual_backend"] == "eager"


def test_explicit_eager_never_qualifies_or_repeats(tmp_path, monkeypatch):
    runtime, cuda_rng, eager, fast = fake_cuda_runtime(tmp_path, monkeypatch, backend="eager")
    expected, expected_rng, expected_cuda = one_eager_call(runtime, cuda_rng)
    def forbidden(): pytest.fail("Explicit eager must never create graphs")
    monkeypatch.setattr(runtime, "_enable", forbidden)
    result, _ = runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=[object()])
    assert np.array_equal(result[0], expected[0])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(cuda_rng.get_state(), expected_cuda)
    assert runtime.report["qualifications"] == []


@pytest.mark.parametrize("setup_error", [ValueError, RuntimeError, TypeError, AttributeError])
def test_partial_graph_setup_failure_restores_both_methods_and_one_call_rng(tmp_path, monkeypatch, setup_error):
    from korean_openwakeword import tts_cuda_graph
    runtime, cuda_rng, eager, fast = fake_cuda_runtime(tmp_path, monkeypatch)
    expected, expected_rng, expected_cuda = one_eager_call(runtime, cuda_rng)
    # Exercise the actual _enable restoration rather than the fake helper.
    monkeypatch.setattr(runtime, "_enable", CloneRuntime._enable.__get__(runtime))
    def predictor_setup(model, exact_shapes):
        model.model.talker.code_predictor.generate = fast
        torch.rand(1); torch.rand(1, generator=cuda_rng)
        return {"original_generate": eager}
    def talker_setup(model):
        model.model.talker.model.forward = fast
        torch.rand(1); torch.rand(1, generator=cuda_rng)
        raise setup_error("partial talker setup failure")
    monkeypatch.setattr(tts_cuda_graph, "enable_predictor_graph", predictor_setup)
    monkeypatch.setattr(tts_cuda_graph, "enable_exact_talker_graph", talker_setup)
    result, _ = runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=[object()])
    assert np.array_equal(result[0], expected[0])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(cuda_rng.get_state(), expected_cuda)
    assert runtime.disabled
    assert runtime.model.model.talker.code_predictor.generate is eager
    assert runtime.model.model.talker.model.forward is eager
    assert "partial talker" in runtime.report["qualifications"][-1]["reason"]


def test_another_reference_gets_its_own_same_seed_warm_qualification(tmp_path, monkeypatch):
    runtime, cuda_rng, eager, fast = fake_cuda_runtime(tmp_path, monkeypatch)
    first_prompt, second_prompt = [object()], [object()]
    for prompt in (first_prompt, second_prompt):
        expected, expected_rng, expected_cuda = one_eager_call(runtime, cuda_rng)
        result, _ = runtime.generate_voice_clone(text="보라야", language="Korean", voice_clone_prompt=prompt)
        assert np.array_equal(result[0], expected[0])
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(cuda_rng.get_state(), expected_cuda)
    assert len(runtime.report["qualifications"]) == 2
    assert len(runtime.checked) == 2
    assert all(q["accepted"] and q["minimum_warm_speedup_met"] for q in runtime.report["qualifications"])
    assert runtime.report["qualification_clips_added_to_training"] == 0


def test_cpu_model_retains_original_path_without_cuda_rng_access(tmp_path, monkeypatch):
    from korean_openwakeword import tts
    eager = object()
    model = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
            talker=SimpleNamespace(code_predictor=SimpleNamespace(generate=eager), model=SimpleNamespace(forward=eager))),
        generate_voice_clone=lambda **kwargs: ([np.ones(1000)*.1], 16000))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbidden(*args, **kwargs): pytest.fail("A CPU model must not touch CUDA qualification")
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(tts, "_load_model", lambda *args: model)
    runtime = CloneRuntime("unused", {"device": "cpu", "attention": "sdpa"}, tmp_path)
    monkeypatch.setattr(runtime, "_enable", forbidden)
    result, rate = runtime.generate_voice_clone(text="보라야", voice_clone_prompt=[object()])
    assert rate == 16000 and np.all(result[0] == .1)
    assert runtime.disabled and runtime.report["qualifications"] == []
