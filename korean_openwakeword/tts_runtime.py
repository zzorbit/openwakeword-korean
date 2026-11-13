"""Choose a synthesis path for the device and amount of remaining work.

Keep the original TTS contract, seeds, sampling, weights and technical QC.
Runtime qualification clips are never added to training or counted as listening.
"""
from collections import OrderedDict
from functools import lru_cache
import os
from pathlib import Path
import time
from types import MethodType
from weakref import WeakMethod
import numpy as np
from .common import digest, now, write_json, identity
from .project import load_project
from .task_progress import emit


_GRAPH_BACKEND = "exact_shape_predictor_and_talker_cuda_graph"
_PROMPT_CACHE_LIMIT = 544  # Covers a complete Studio Train and Validation voice bank.
_MIN_WARM_GAIN = .05


def _graph_capacity_exceeded(error):
    return isinstance(error, ValueError) and str(error) == "Graph capacity exceeded"


def _cpu_thread_budget(requested):
    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    available = process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        available = min(available, len(os.sched_getaffinity(0)))
    # A process can have a smaller thread allowance than its visible cores.
    caps = (os.environ.get(name, "").partition(",")[0].strip()
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    return min(requested, available, *(int(cap) for cap in caps if cap.isascii() and cap.isdecimal() and int(cap)))


def _cache_tokenization(model):
    tokenize = getattr(model, "_tokenize_texts", None)
    if not callable(tokenize):
        return
    # Avoid keeping the loaded model alive via its own cached bound method.
    original = WeakMethod(tokenize) if isinstance(tokenize, MethodType) else lambda: tokenize

    @lru_cache(maxsize=2 * _PROMPT_CACHE_LIMIT)
    def one(text):
        # Qwen's tokenizer produces input IDs that generation only reads.
        return original()([text])[0]

    def tokenize_texts(texts):
        if isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str):
            return [one(texts[0])]
        return original()(texts)

    model._tokenize_texts = tokenize_texts


class CloneRuntime:
    def __init__(self, model_path, settings, root, backend="auto", cpu_threads=4, *, expected_reuses=None):
        self.model_path, self.settings, self.root = model_path, settings, root
        self.backend, self.cpu_threads = backend, cpu_threads
        self.expected_reuses = expected_reuses
        self.cuda_device = 0
        self.model = None
        self.predictor = self.talker = None
        self.checked = {}
        self.eager_prompts = {}
        self._prompt_cache = OrderedDict()
        self._short_clone_limits = {}
        self.disabled = backend == "eager" or expected_reuses == 1
        self.report = {"requested_backend": backend, "created_at": now(), "qualifications": [],
                       "actual_backend": "eager", "expected_utterances_per_prompt": expected_reuses,
                       "minimum_warm_speedup_fraction": _MIN_WARM_GAIN,
                       "qualification_clips_added_to_training": 0, "Final_audio_accessed": False}

    def _load(self):
        if self.model is not None:
            return
        emit("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
        import torch
        from .tts import _load_model
        threads = _cpu_thread_budget(self.cpu_threads)
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        self.model = _load_model(self.model_path, self.settings, self.root)
        _cache_tokenization(self.model)
        devices, dtypes = set(), set()
        for parameter in self.model.model.parameters():
            devices.add(parameter.device)
            dtypes.add(parameter.dtype)
        device = next(iter(devices)) if len(devices) == 1 else None
        cuda = device is not None and device.type == "cuda"
        if cuda:
            self.cuda_device = device.index
        self.report.update(parameter_devices=sorted(map(str, devices)), parameter_dtypes=sorted(map(str, dtypes)),
                           cpu_threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(device) if cuda else None,
                           torch=torch.__version__, transformers=__import__("transformers").__version__)
        if not cuda or torch.version.hip or dtypes != {torch.bfloat16} or self.settings["attention"] != "sdpa":
            self.disabled = True
        self._save()
        emit("tts_generation", "호출어 음성 생성", eta_eligible=True)

    def _save(self):
        write_json(self.root / "runtime" / "tts_execution.json", self.report)

    def create_voice_clone_prompt(self, **kwargs):
        self._load()
        audio, text = kwargs.get("ref_audio"), kwargs.get("ref_text")
        x_vector = kwargs.get("x_vector_only_mode", False)
        cache_key = None
        devices = self.report.get("parameter_devices", ())
        cpu_prompt = devices == ["cpu"]
        cuda_prompt = len(devices) == 1 and devices[0].startswith("cuda:")
        if ((cpu_prompt or cuda_prompt)
                and set(kwargs) <= {"ref_audio", "ref_text", "x_vector_only_mode"}
                and isinstance(audio, str) and isinstance(text, str) and type(x_vector) is bool):
            # Callers verify the prompt WAV before opening it. Hashing also
            # keeps an external replacement at the same path from reusing it.
            path = Path(audio)
            if path.is_file():
                cache_key = (digest(path), text, x_vector)
                if cache_key in self._prompt_cache:
                    self._prompt_cache.move_to_end(cache_key)
                    return self._prompt_cache[cache_key]
        emit("tts_prompt_preparation", "기준 목소리 복제 정보 준비", eta_eligible=False)
        if cache_key is not None:
            import torch
            cpu_before = torch.get_rng_state()
            cuda_before = torch.cuda.get_rng_state(self.cuda_device) if cuda_prompt else None
        prompt = self.model.create_voice_clone_prompt(**kwargs)
        if (cache_key is not None and torch.equal(cpu_before, torch.get_rng_state())
                and (cuda_before is None or torch.equal(cuda_before, torch.cuda.get_rng_state(self.cuda_device)))):
            self._prompt_cache[cache_key] = prompt
            if len(self._prompt_cache) > _PROMPT_CACHE_LIMIT:
                _, evicted = self._prompt_cache.popitem(last=False)
                for decisions in (self.checked, self.eager_prompts):
                    for key in [key for key in decisions if key[0] == id(evicted)]:
                        del decisions[key]
        emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
        return prompt

    def _enable(self):
        import torch
        from .tts_cuda_graph import enable_predictor_graph, enable_exact_talker_graph
        # Capture warmup must never advance the caller's sample RNG.
        talker = self.model.model.talker
        original_predictor = talker.code_predictor.generate
        original_talker = talker.model.forward
        try:
            with torch.random.fork_rng(devices=[self.cuda_device]), torch.inference_mode():
                predictor = enable_predictor_graph(self.model, exact_shapes=True)
                talker_graph = enable_exact_talker_graph(self.model)
        except Exception:
            # Restore both entry points even when either setup function fails
            # after partially changing the model. Never leave a mixed backend.
            talker.code_predictor.generate = original_predictor
            talker.model.forward = original_talker
            raise
        self.predictor, self.talker = predictor, talker_graph

    def _eager(self, kwargs):
        import torch
        if self.predictor is None:
            with torch.inference_mode():
                return self.model.generate_voice_clone(**kwargs)
        talker = self.model.model.talker
        predictor = talker.code_predictor
        inner = talker.model
        fast_predictor, fast_talker = predictor.generate, inner.forward
        predictor.generate = self.predictor["original_generate"]
        inner.forward = self.talker["original_forward"]
        try:
            with torch.inference_mode():
                return self.model.generate_voice_clone(**kwargs)
        finally:
            predictor.generate, inner.forward = fast_predictor, fast_talker

    def _disable_graph(self):
        if self.predictor is not None:
            self.model.model.talker.code_predictor.generate = self.predictor["original_generate"]
            self.model.model.talker.model.forward = self.talker["original_forward"]
            self.predictor = self.talker = None
        self.checked.clear()
        self.eager_prompts.clear()
        self.disabled = True
        self.report["actual_backend"] = "eager"

    def generate_short_clone(self, *, expected_reuses=None, retry=False, **kwargs):
        """Apply the shared short-utterance budget without changing device selection."""
        from .tts import _short_clone_token_limit
        self._load()
        key = kwargs.get("max_new_tokens"), kwargs.get("text"), retry
        if key not in self._short_clone_limits:
            self._short_clone_limits[key] = _short_clone_token_limit(self.model, key[0], text=key[1], retry=retry)
        limit = self._short_clone_limits[key]
        if limit is not None:
            kwargs["max_new_tokens"] = limit
        wavs, rate = self.generate_voice_clone(expected_reuses=expected_reuses, **kwargs)
        return wavs, rate, limit

    def generate_voice_clone(self, *, expected_reuses=None, **kwargs):
        self._load()
        import torch
        if expected_reuses is None:
            expected_reuses = self.expected_reuses
        if self.disabled:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return self._eager(kwargs)
        prompt = kwargs["voice_clone_prompt"]
        key = (id(prompt), identity({k: v for k, v in kwargs.items() if k != "voice_clone_prompt"}))
        if key in self.eager_prompts:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return self._eager(kwargs)
        if key in self.checked:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            cpu_before = torch.get_rng_state()
            cuda_before = torch.cuda.get_rng_state(self.cuda_device)
            try:
                with torch.inference_mode():
                    return self.model.generate_voice_clone(**kwargs)
            except (ValueError, torch.cuda.OutOfMemoryError) as error:
                local_capacity = _graph_capacity_exceeded(error)
                if not local_capacity and not isinstance(error, torch.cuda.OutOfMemoryError):
                    raise
                # A later seed can be longer than the first qualified seed.
                # Retry the same sample from its original RNG state. The next
                # graph call prefills the cache again before any replay.
                torch.set_rng_state(cpu_before)
                torch.cuda.set_rng_state(cuda_before, self.cuda_device)
                if local_capacity:
                    del self.checked[key]
                    self.eager_prompts[key] = prompt
                else:
                    self._disable_graph()
                self.report.setdefault("replay_fallbacks", []).append(
                    {"at": now(), "reason": str(error), "scope": "prompt" if local_capacity else "runtime"})
                self._save()
            return self._eager(kwargs)
        if expected_reuses == 1:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return self._eager(kwargs)
        # Compare each actual reference's first requested utterance at exactly
        # the caller's seed. All qualification repeats are ephemeral; only one
        # result and the RNG progression of one eager call are retained.
        emit("tts_qualification", "기준 목소리의 GPU 가속 확인", eta_eligible=False)
        cpu_before = torch.get_rng_state().clone()
        device = self.cuda_device
        cuda_before = torch.cuda.get_rng_state(device).clone()
        with torch.random.fork_rng(devices=[device]):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            expected, rate = self._eager(kwargs)
            torch.cuda.synchronize(device)
            eager_seconds = time.perf_counter() - started
            cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        qualification = {"accepted": False, "eager_seconds": eager_seconds,
                         "expected_utterances_per_prompt": expected_reuses,
                         "minimum_warm_speedup_fraction": _MIN_WARM_GAIN, "at": now()}
        from .tts import technical_qc
        from .audio import pcm16
        try:
            eq_qc, eq_signal = technical_qc(expected[0] if expected is not None and len(expected) else [], rate)
            if eq_qc["passed"] is not True:
                # The caller will reject this exact eager output. Replaying it
                # twice on the graph would waste time and could hit capacity.
                torch.set_rng_state(cpu_after)
                torch.cuda.set_rng_state(cuda_after, device)
                qualification.update(skipped="technical_qc_failed", quality_reasons=eq_qc["reasons"])
                self.report["qualifications"].append(qualification)
                self._save()
                emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
                return expected, rate
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            graphs = self.talker.get("graphs") if self.talker is not None else None
            bucket_count = len(graphs) if graphs is not None else None
            if self.predictor is None:
                self._enable()
            with torch.inference_mode():
                actual, actual_rate = self.model.generate_voice_clone(**kwargs)
            torch.cuda.synchronize(device)
            graph_seconds = time.perf_counter() - started

            def compare_waveform(values, sample_rate):
                qc, signal = technical_qc(values[0], sample_rate)
                same_pcm = sample_rate == rate and signal is not None and np.array_equal(pcm16(eq_signal[0]), pcm16(signal[0]))
                return same_pcm, qc == eq_qc

            cold_pcm, cold_qc = compare_waveform(actual, actual_rate)
            cold_cpu_rng = torch.equal(cpu_after, torch.get_rng_state())
            cold_cuda_rng = torch.equal(cuda_after, torch.cuda.get_rng_state(device))
            qualification.update(graph_cold_seconds=graph_seconds,
                cold_pcm_samples_equal=cold_pcm, cold_quality_checks_equal=cold_qc,
                cold_cpu_rng_equal=cold_cpu_rng, cold_cuda_rng_equal=cold_cuda_rng)
            if not all((cold_pcm, cold_qc, cold_cpu_rng, cold_cuda_rng)):
                raise ValueError("Eager/cold graph PCM, QC or sampling RNG differs")

            # With no new graph buckets the first comparison already measures
            # steady-state replay. A new capture requires a warmed repeat.
            reused_graphs = graphs is not None and len(graphs) == bucket_count
            if reused_graphs:
                warm_seconds, warm_pcm, warm_qc = graph_seconds, cold_pcm, cold_qc
                warm_cpu_rng, warm_cuda_rng = cold_cpu_rng, cold_cuda_rng
            else:
                with torch.random.fork_rng(devices=[device]):
                    torch.set_rng_state(cpu_before)
                    torch.cuda.set_rng_state(cuda_before, device)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        warm, warm_rate = self.model.generate_voice_clone(**kwargs)
                    torch.cuda.synchronize(device)
                    warm_seconds = time.perf_counter() - started
                    warm_cpu_rng = torch.equal(cpu_after, torch.get_rng_state())
                    warm_cuda_rng = torch.equal(cuda_after, torch.cuda.get_rng_state(device))
                warm_pcm, warm_qc = compare_waveform(warm, warm_rate)
            faster = eager_seconds > 0 and 0 <= warm_seconds <= eager_seconds * (1 - _MIN_WARM_GAIN)
            savings = eager_seconds - warm_seconds
            extra_time = graph_seconds + (0 if reused_graphs else warm_seconds)
            enough_reuse = expected_reuses is None or savings * (expected_reuses - 1) > extra_time
            qualification.update(graph_warm_seconds=warm_seconds,
                reused_graph_buckets=reused_graphs, graph_qualification_extra_seconds=extra_time,
                warm_pcm_samples_equal=warm_pcm, warm_quality_checks_equal=warm_qc,
                warm_cpu_rng_equal=warm_cpu_rng, warm_cuda_rng_equal=warm_cuda_rng,
                pcm_samples_equal=cold_pcm and warm_pcm,
                quality_checks_equal=cold_qc and warm_qc,
                sampling_rng_equal=cold_cpu_rng and cold_cuda_rng and warm_cpu_rng and warm_cuda_rng,
                warm_speedup_fraction=(1 - warm_seconds / eager_seconds) if eager_seconds > 0 else None,
                minimum_warm_speedup_met=faster)
            if not all((warm_pcm, warm_qc, warm_cpu_rng, warm_cuda_rng)):
                raise ValueError("Eager/warm graph PCM, QC or sampling RNG differs")
        except (ValueError, RuntimeError, TypeError, AttributeError) as error:
            local_capacity = self.predictor is not None and _graph_capacity_exceeded(error)
            if local_capacity:
                self.eager_prompts[key] = prompt
            else:
                self._disable_graph()
            torch.set_rng_state(cpu_after)
            torch.cuda.set_rng_state(cuda_after, device)
            qualification.update(fallback="eager", reason=str(error), fallback_scope="prompt" if local_capacity else "runtime")
            self.report["qualifications"].append(qualification)
            self._save()
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return expected, rate
        # A slower prompt can still share the graphs with faster prompts.
        # Retain each prompt so a recycled object ID cannot inherit a verdict.
        if faster and enough_reuse:
            self.checked[key] = prompt
            qualification.update(accepted=True, backend=_GRAPH_BACKEND)
            self.report["actual_backend"] = _GRAPH_BACKEND
        else:
            self.eager_prompts[key] = prompt
            qualification.update(fallback="eager", fallback_scope="prompt",
                reason="Warm CUDA graph is not at least 5% faster than eager" if not faster
                else "Expected remaining utterances do not cover graph qualification cost")
        self.report["qualifications"].append(qualification)
        self._save()
        emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
        return (actual, actual_rate) if qualification["accepted"] else (expected, rate)


def generate_with_runtime(project, split="train", limit=None, pilot=False, method="clone", backend="auto", cpu_threads=4):
    if backend not in ("auto", "eager") or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Use backend auto/eager and a positive CPU thread count")
    from .tts import generate
    root, config = load_project(project)
    # Lazy loading retains generate's approval, split and Final checks before
    # opening weights, prompts or audio. Design retains the original path.
    model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, backend, cpu_threads) if method == "clone" else None
    return generate(project, split, limit, pilot, method, model=model)
