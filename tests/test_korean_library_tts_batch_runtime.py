"""Four-row CUDA inference protects individual endings and the serial runtime."""
from contextlib import contextmanager, nullcontext
from itertools import cycle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from korean_openwakeword import tts, tts_cuda_graph, tts_runtime


class FakeTalker:
    def __init__(self):
        self.code_predictor = SimpleNamespace(generate=object())
        self.model = SimpleNamespace(forward=object())

    def generate(self, *, max_new_tokens, eos_token_id):
        return SimpleNamespace(sequences=torch.tensor([
            [3, eos_token_id, eos_token_id, eos_token_id],
            [3, 4, 5, eos_token_id],
            [3, 4, 5, 6],
            [eos_token_id, eos_token_id, eos_token_id, eos_token_id],
        ]))


@pytest.mark.parametrize("limit,unfinished", [(4, "token_limit"), (6, "without_eos")])
def test_batch_termination_uses_first_eos_and_counts_each_row(limit, unfinished):
    talker = FakeTalker()
    model = SimpleNamespace(model=SimpleNamespace(talker=talker))
    with tts_runtime._capture_batch_termination(model, limit) as endings:
        talker.generate(max_new_tokens=limit, eos_token_id=91)
    assert [(row["termination"], row["generated_tokens"], row["token_limit"]) for row in endings] == [
        ("eos", 2, limit), ("eos", 4, limit), (unfinished, 4, limit), ("eos", 1, limit),
    ]
    assert "generate" not in vars(talker)


def mock_batch_runtime(tmp_path, monkeypatch, scenario=None):
    gpu_rng = torch.Generator()
    gpu_rng.manual_seed(99)
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index: None)
    monkeypatch.setattr(torch.cuda, "manual_seed", gpu_rng.manual_seed)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda index: gpu_rng.get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, index: gpu_rng.set_state(state))
    monkeypatch.setattr(tts, "_short_clone_token_limit", lambda *args, **kwargs: 6)

    @contextmanager
    def fork_rng(devices):
        cpu, gpu = torch.get_rng_state(), gpu_rng.get_state()
        try:
            yield
        finally:
            torch.set_rng_state(cpu)
            gpu_rng.set_state(gpu)

    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    talker = FakeTalker()
    stock_methods = (talker.code_predictor.generate, talker.model.forward)
    single_methods = (object(), object())
    batch_methods = (object(), object())
    timeline = []
    wave = .15 * np.sin(2 * np.pi * 230 * np.arange(9600) / 16000)

    class FakeQwen:
        model = SimpleNamespace(talker=talker)
        fail_replay = False

        def generate_voice_clone(self, *, text, language, voice_clone_prompt, max_new_tokens):
            assert text == ["보라야"] * 4 and language == "Korean" and len(voice_clone_prompt) == 1
            methods = talker.code_predictor.generate, talker.model.forward
            assert methods in (stock_methods, batch_methods)
            accelerated = methods == batch_methods
            timeline.append(methods)
            if self.fail_replay and accelerated:
                raise torch.cuda.OutOfMemoryError("test batch OOM")
            offset = (torch.rand(4) + torch.rand(4, generator=gpu_rng)).numpy()
            if accelerated and scenario == "rng":
                torch.rand(1, generator=gpu_rng)
            samples = [wave + value * .002 for value in offset]
            if accelerated and scenario == "last_pcm":
                samples[-1] = samples[-1] + .03
            talker.generate(max_new_tokens=max_new_tokens, eos_token_id=91)
            return samples, 16000

    runtime = tts_runtime.CloneRuntime("unused", {}, tmp_path)
    runtime.model = FakeQwen()
    runtime._cuda_graph_capable = True
    runtime.predictor, runtime.talker = {"original_generate": stock_methods[0]}, {"original_forward": stock_methods[1]}
    talker.code_predictor.generate, talker.model.forward = single_methods

    def capture_predictor(model, *, exact_shapes, batch_size):
        assert (talker.code_predictor.generate, talker.model.forward) == stock_methods
        assert exact_shapes and batch_size == 4
        talker.code_predictor.generate = batch_methods[0]

    def capture_talker(model, *, batch_size):
        assert batch_size == 4
        talker.model.forward = batch_methods[1]
        return {"graphs": {}}

    monkeypatch.setattr(tts_cuda_graph, "enable_predictor_graph", capture_predictor)
    monkeypatch.setattr(tts_cuda_graph, "enable_exact_talker_graph", capture_talker)
    clock = cycle((0, 2, 0, 3, 0, .6))
    monkeypatch.setattr(tts_runtime, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    return runtime, gpu_rng, timeline, talker, stock_methods, batch_methods, single_methods


def generate(runtime, prompt, seed=814):
    return runtime.generate_short_clone_batch(batch_seed=seed, expected_reuses=30, text="보라야",
                                              language="Korean", voice_clone_prompt=prompt)


def test_four_row_qualification_replays_seed_and_restores_serial_graph(tmp_path, monkeypatch):
    runtime, gpu_rng, timeline, talker, stock, batch, single = mock_batch_runtime(tmp_path, monkeypatch)
    prompt = [object()]
    initial_cpu, initial_gpu = torch.get_rng_state(), gpu_rng.get_state()
    first = generate(runtime, prompt)
    second = generate(runtime, prompt)
    assert timeline == [stock, batch, batch, batch]
    assert len(first[0]) == 4 and first[1:] == second[1:]
    assert all(np.array_equal(left, right) for left, right in zip(first[0], second[0], strict=True))
    assert (talker.code_predictor.generate, talker.model.forward) == single
    assert torch.equal(initial_cpu, torch.get_rng_state()) and torch.equal(initial_gpu, gpu_rng.get_state())
    assert [row["accepted"] for row in runtime.report["batch_qualifications"]] == [True]


@pytest.mark.parametrize("scenario", ["last_pcm", "rng"])
def test_parity_failure_in_any_row_returns_no_batch(tmp_path, monkeypatch, scenario):
    runtime, gpu_rng, timeline, talker, stock, batch, single = mock_batch_runtime(tmp_path, monkeypatch, scenario)
    prompt = [object()]
    assert generate(runtime, prompt) is None and generate(runtime, prompt) is None
    assert timeline == [stock, batch]
    assert not runtime.report["batch_qualifications"][-1]["accepted"]
    assert (talker.code_predictor.generate, talker.model.forward) == single


def test_batch_oom_keeps_single_graph_and_disables_only_four_row_path(tmp_path, monkeypatch):
    runtime, gpu_rng, timeline, talker, stock, batch, single = mock_batch_runtime(tmp_path, monkeypatch)
    prompt = [object()]
    assert generate(runtime, prompt)
    runtime.model.fail_replay = True
    assert generate(runtime, prompt, 815) is None
    assert generate(runtime, [object()], 816) is None
    assert 4 not in runtime._batch_graphs and not runtime.disabled
    assert (talker.code_predictor.generate, talker.model.forward) == single


def test_eight_row_oom_allows_the_already_qualified_four_row_fallback(tmp_path, monkeypatch):
    runtime, _, _, _, _, _, _ = mock_batch_runtime(tmp_path, monkeypatch)
    prompt = [object()]
    assert generate(runtime, prompt)
    run = runtime._run_batch

    def fail_wide(kwargs, limit, *, accelerated):
        if len(kwargs["text"]) == 8:
            raise torch.cuda.OutOfMemoryError("test wide batch OOM")
        return run(kwargs, limit, accelerated=accelerated)

    monkeypatch.setattr(runtime, "_run_batch", fail_wide)
    assert runtime.generate_short_clone_wide_batch(batch_seed=814, expected_reuses=30, text="보라야",
                                                  language="Korean", voice_clone_prompt=prompt) is None
    assert runtime._disabled_batch_sizes == {8} and 4 in runtime._batch_graphs
    assert generate(runtime, prompt)


def test_partial_batch_capture_failure_restores_both_single_methods(tmp_path, monkeypatch):
    runtime, _, timeline, talker, stock, _, single = mock_batch_runtime(tmp_path, monkeypatch)

    def fail_talker(model, *, batch_size):
        talker.model.forward = object()
        raise RuntimeError("test capture failure after method replacement")

    monkeypatch.setattr(tts_cuda_graph, "enable_exact_talker_graph", fail_talker)
    assert generate(runtime, [object()]) is None
    assert timeline == [stock] and 4 not in runtime._batch_graphs and not runtime.disabled
    assert (talker.code_predictor.generate, talker.model.forward) == single


def test_incompatible_device_or_prompt_does_not_invoke_cuda(tmp_path, monkeypatch):
    runtime, _, timeline, *_ = mock_batch_runtime(tmp_path, monkeypatch)
    assert generate(runtime, [object(), object()]) is None
    runtime._cuda_graph_capable = False
    assert generate(runtime, [object()]) is None
    assert timeline == []


def test_mixed_voice_batch_keeps_prompt_order_and_restores_scalar_graph_and_rng(tmp_path, monkeypatch):
    runtime, gpu_rng, _, talker, stock, _, single = mock_batch_runtime(tmp_path, monkeypatch)
    runtime._mixed_voice_batch_capable = True
    prompts = [object() for _ in range(4)]
    calls = []

    def synthesize(*, text, language, voice_clone_prompt, max_new_tokens):
        assert text == ["보리야"] * 4 and language == "Korean"
        assert voice_clone_prompt == prompts and max_new_tokens == 6
        assert (talker.code_predictor.generate, talker.model.forward) == stock
        calls.append(voice_clone_prompt)
        values = (torch.rand(4) + torch.rand(4, generator=gpu_rng)).numpy()
        talker.generate(max_new_tokens=max_new_tokens, eos_token_id=91)
        return [np.array([index, value]) for index, value in enumerate(values)], 16000

    monkeypatch.setattr(runtime.model, "generate_voice_clone", synthesize)
    original_cpu, original_gpu = torch.get_rng_state(), gpu_rng.get_state()
    options = {"batch_seed": 814, "voice_clone_prompts": [[p] for p in prompts],
               "text": "보리야", "language": "Korean"}
    first = runtime.generate_short_clone_mixed_voice_batch(**options)
    second = runtime.generate_short_clone_mixed_voice_batch(**options)

    assert len(calls) == 2 and first[1:] == second[1:]
    assert [wave[0] for wave in first[0]] == list(range(4))
    assert all(np.array_equal(a, b) for a, b in zip(first[0], second[0], strict=True))
    assert [row["termination"] for row in first[3]] == ["eos", "eos", "without_eos", "eos"]
    assert torch.equal(original_cpu, torch.get_rng_state()) and torch.equal(original_gpu, gpu_rng.get_state())
    assert (talker.code_predictor.generate, talker.model.forward) == single


def test_mixed_voice_batch_stays_on_model_device_after_oom(tmp_path, monkeypatch):
    runtime, _, _, talker, stock, _, single = mock_batch_runtime(tmp_path, monkeypatch)
    options = {"batch_seed": 814, "voice_clone_prompts": [[object()] for _ in range(4)],
               "text": "보리야", "language": "Korean"}
    called = []

    def oom(**kwargs):
        assert (talker.code_predictor.generate, talker.model.forward) == stock
        called.append("synthesize")
        raise torch.cuda.OutOfMemoryError("fixture OOM")

    monkeypatch.setattr(runtime.model, "generate_voice_clone", oom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.append("release"))
    assert runtime.generate_short_clone_mixed_voice_batch(**options) is None
    assert called == []

    runtime._mixed_voice_batch_capable = True
    assert runtime.generate_short_clone_mixed_voice_batch(**options) is None
    assert runtime.generate_short_clone_mixed_voice_batch(**options) is None
    assert called == ["synthesize", "release"]
    assert runtime.report["mixed_voice_batch_fallback"] == "fixture OOM"
    assert not runtime.disabled and not runtime._batch_disabled
    assert (talker.code_predictor.generate, talker.model.forward) == single
