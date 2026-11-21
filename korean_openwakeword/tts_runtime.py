"""Choose a synthesis path for the device and amount of remaining work.

Keep the original TTS contract, seeds, sampling, weights and technical QC.
Runtime qualification clips are never added to training or counted as listening.
"""
from collections import OrderedDict
from contextlib import contextmanager
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
_CLONE_BATCH_SIZE = 4
_BATCH_GRAPH_BACKEND = "exact_shape_predictor_and_talker_cuda_graph_batch4"
MIXED_VOICE_BATCH_SIZE = _CLONE_BATCH_SIZE
_MIXED_VOICE_BATCH_BACKEND = "eager_cuda_mixed_voice_batch4"
_PROMPT_CACHE_LIMIT = 544  # Covers a complete Studio Train and Validation voice bank.
_MIN_WARM_GAIN = .05


def _graph_capacity_exceeded(error):
    return isinstance(error, ValueError) and str(error) == "Graph capacity exceeded"


@contextmanager
def _capture_batch_termination(model, limit):
    """Distinguish the first EOS in each row from padding added afterward."""
    talker = model.model.talker
    generate = talker.generate
    previous = vars(talker).get("generate")
    evidence = []
    observed = []

    def observe(*args, **kwargs):
        result = generate(*args, **kwargs)
        observed.append((getattr(result, "sequences", None), kwargs.get("max_new_tokens"), kwargs.get("eos_token_id")))
        return result

    talker.generate = observe
    try:
        yield evidence
        if len(observed) != 1:
            raise ValueError("Batch generation must expose one sampled sequence per audio")
        sequence, observed_limit, eos = observed[0]
        if (sequence is None or sequence.ndim != 2 or sequence.shape[0] != _CLONE_BATCH_SIZE
                or not 0 < sequence.shape[1] <= limit or observed_limit != limit or type(eos) is not int):
            raise ValueError("Batch generation did not expose valid termination evidence")
        for row in sequence.tolist():
            if eos in row:
                count, ending = row.index(eos) + 1, "eos"
            else:
                count = len(row)
                ending = "token_limit" if count == limit else "without_eos"
            evidence.append({"termination": ending, "generated_tokens": count, "token_limit": limit})
    finally:
        if previous is None:
            del talker.generate
        else:
            talker.generate = previous


def _same_batch_audio(expected, actual, limit):
    from .audio import pcm16
    from .tts import check_short_clone_audio

    expected_audio, expected_rate, expected_ends = expected
    actual_audio, actual_rate, actual_ends = actual
    if actual_rate != expected_rate or actual_ends != expected_ends:
        return False
    for left, right, ending in zip(expected_audio, actual_audio, expected_ends, strict=True):
        _, expected_qc, expected_signal = check_short_clone_audio(left, expected_rate, limit, ending)
        _, actual_qc, actual_signal = check_short_clone_audio(right, actual_rate, limit, ending)
        if expected_qc != actual_qc:
            return False
        if expected_signal is None or actual_signal is None:
            if expected_signal is not actual_signal or not np.array_equal(left, right, equal_nan=True):
                return False
        elif not np.array_equal(pcm16(expected_signal[0]), pcm16(actual_signal[0])):
            return False
    return True


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
        self._cuda_graph_capable = False
        self._batch_disabled = backend == "eager"
        self._batch_graph = None
        self._batch_checked = {}
        self._batch_rejected = {}
        self._mixed_voice_batch_capable = False
        self._mixed_voice_batch_disabled = False
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
        self._cuda_graph_capable = bool(cuda and not torch.version.hip and dtypes == {torch.bfloat16}
                                        and self.settings["attention"] == "sdpa")
        self._mixed_voice_batch_capable = bool(cuda and not torch.version.hip)
        if not self._cuda_graph_capable:
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
                for decisions in (self.checked, self.eager_prompts, self._batch_checked, self._batch_rejected):
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

    def _short_clone_limit(self, configured_limit, text, retry):
        from .tts import _short_clone_token_limit
        key = configured_limit, text, retry
        if key not in self._short_clone_limits:
            self._short_clone_limits[key] = _short_clone_token_limit(self.model, configured_limit, text=text, retry=retry)
        return self._short_clone_limits[key]

    def generate_short_clone(self, *, expected_reuses=None, retry=False, termination=None, **kwargs):
        """Apply the shared short-utterance budget without changing device selection."""
        self._load()
        limit = self._short_clone_limit(kwargs.get("max_new_tokens"), kwargs.get("text"), retry)
        if limit is not None:
            kwargs["max_new_tokens"] = limit
        wavs, rate = self.generate_voice_clone(expected_reuses=expected_reuses, termination=termination, **kwargs)
        return wavs, rate, limit

    @contextmanager
    def _use_batch_graph(self):
        """Temporarily install the four-row graph around Qwen's batch call."""
        import torch
        from .tts_cuda_graph import enable_exact_talker_graph, enable_predictor_graph

        talker = self.model.model.talker
        predictor, inner = talker.code_predictor, talker.model
        previous = predictor.generate, inner.forward
        try:
            if self._batch_graph is None:
                if self.predictor is not None:
                    predictor.generate = self.predictor["original_generate"]
                    inner.forward = self.talker["original_forward"]
                with torch.random.fork_rng(devices=[self.cuda_device]):
                    enable_predictor_graph(self.model, exact_shapes=True, batch_size=_CLONE_BATCH_SIZE)
                    graph = enable_exact_talker_graph(self.model, batch_size=_CLONE_BATCH_SIZE)
                self._batch_graph = predictor.generate, inner.forward, graph["graphs"]
            else:
                predictor.generate, inner.forward = self._batch_graph[:2]
            yield
        finally:
            predictor.generate, inner.forward = previous

    def _run_batch(self, kwargs, limit, *, accelerated):
        import torch

        with _capture_batch_termination(self.model, limit) as endings:
            if accelerated:
                with torch.inference_mode(), self._use_batch_graph():
                    wavs, rate = self.model.generate_voice_clone(**kwargs)
            else:
                wavs, rate = self._eager(kwargs)
        if len(wavs) != _CLONE_BATCH_SIZE:
            raise ValueError("Batch generation returned an unexpected number of audio clips")
        return wavs, rate, endings

    def _qualify_batch(self, kwargs, limit, expected_reuses, report):
        import torch

        device = self.cuda_device
        original_cpu, original_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)

        def measure(accelerated):
            torch.set_rng_state(original_cpu)
            torch.cuda.set_rng_state(original_cuda, device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = self._run_batch(kwargs, limit, accelerated=accelerated)
            torch.cuda.synchronize(device)
            return result, time.perf_counter() - start

        expected, eager_seconds = measure(False)
        expected_cpu, expected_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        report["eager_seconds"] = eager_seconds
        buckets = len(self._batch_graph[2]) if self._batch_graph is not None else None
        actual, cold_seconds = measure(True)

        def parity(result):
            return (_same_batch_audio(expected, result, limit)
                    and torch.equal(expected_cpu, torch.get_rng_state())
                    and torch.equal(expected_cuda, torch.cuda.get_rng_state(device)))

        report.update(graph_cold_seconds=cold_seconds, cold_parity=parity(actual))
        if not report["cold_parity"]:
            raise ValueError("Eager/cold batch graph PCM, QC, termination or RNG differs")
        reused_buckets = buckets == len(self._batch_graph[2])
        warm_seconds = cold_seconds
        if not reused_buckets:
            warm, warm_seconds = measure(True)
            if not parity(warm):
                report["warm_parity"] = False
                raise ValueError("Eager/warm batch graph PCM, QC, termination or RNG differs")
        report.update(graph_warm_seconds=warm_seconds, warm_parity=True, reused_graph_buckets=reused_buckets)
        savings = eager_seconds - warm_seconds
        if eager_seconds <= 0 or savings < eager_seconds * _MIN_WARM_GAIN:
            report["reason"] = "Warm batch graph did not improve on the same CUDA eager batch"
            return None
        overhead = max(0, cold_seconds - warm_seconds) + (0 if reused_buckets else warm_seconds)
        if expected_reuses is not None and savings * (expected_reuses - 1) <= overhead:
            report["reason"] = "Expected batches do not amortize CUDA graph capture"
            return None
        return actual

    def generate_short_clone_batch(self, *, batch_seed, expected_reuses=None, retry=False, **kwargs):
        """Synthesize four clips for one prompt, returning each clip's EOS evidence."""
        if type(batch_seed) is not int or not 0 <= batch_seed < 2**32:
            raise ValueError("A batch seed must be a 32-bit unsigned integer")
        self._load()
        prompt = kwargs.get("voice_clone_prompt")
        text, language = kwargs.get("text"), kwargs.get("language")
        if (self._batch_disabled or not self._cuda_graph_capable or not isinstance(prompt, list) or len(prompt) != 1
                or not isinstance(text, str) or not isinstance(language, str)):
            return None
        from .tts import _short_clone_token_limit
        import torch

        limit = _short_clone_token_limit(self.model, kwargs.get("max_new_tokens"), text=text, retry=retry)
        if not limit:
            return None
        kwargs.update(max_new_tokens=limit, text=[text] * _CLONE_BATCH_SIZE)
        key = (id(prompt), identity({name: value for name, value in kwargs.items() if name != "voice_clone_prompt"}))
        if key in self._batch_rejected:
            return None
        qualified = key in self._batch_checked
        report = {"at": now(), "batch_size": _CLONE_BATCH_SIZE, "accepted": False,
                  "expected_batches_per_prompt": expected_reuses, "requalified_replay": qualified}
        try:
            if not qualified:
                emit("tts_qualification", "4개 동시 합성의 CUDA 가속 확인", eta_eligible=False)
            device = self.cuda_device
            with torch.cuda.device(device), torch.random.fork_rng(devices=[device]):
                torch.default_generator.manual_seed(batch_seed)
                torch.cuda.manual_seed(batch_seed)
                result = (self._run_batch(kwargs, limit, accelerated=True) if qualified
                          else self._qualify_batch(kwargs, limit, expected_reuses, report))
        except (ValueError, RuntimeError, TypeError, AttributeError) as error:
            report["reason"] = str(error)
            result = None
            if isinstance(error, torch.cuda.OutOfMemoryError) or self._batch_graph is None:
                self._batch_graph = None
                self._batch_disabled = True
        if result is None:
            self._batch_checked.pop(key, None)
            self._batch_rejected[key] = prompt
        elif not qualified:
            self._batch_checked[key] = prompt
            report.update(accepted=True, backend=_BATCH_GRAPH_BACKEND)
            self.report["batch_backend"] = _BATCH_GRAPH_BACKEND
        if not qualified or result is None:
            self.report.setdefault("batch_qualifications", []).append(report)
            self._save()
        emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
        if result is None:
            return None
        wavs, rate, endings = result
        return wavs, rate, limit, endings

    def mixed_voice_batch_available(self):
        """Use mixed-voice inference only on the CUDA device holding the model."""
        self._load()
        return self._mixed_voice_batch_capable and not self._mixed_voice_batch_disabled

    def generate_short_clone_mixed_voice_batch(self, *, batch_seed, voice_clone_prompts, **kwargs):
        """Generate one copy of the same phrase for each prompt, in slot order."""
        if type(batch_seed) is not int or not 0 <= batch_seed < 2**32:
            raise ValueError("A batch seed must be a 32-bit unsigned integer")
        text, language = kwargs.get("text"), kwargs.get("language")
        if (not isinstance(text, str) or not isinstance(language, str)
                or not isinstance(voice_clone_prompts, (list, tuple))
                or len(voice_clone_prompts) != MIXED_VOICE_BATCH_SIZE
                or any(not isinstance(prompt, list) or len(prompt) != 1 for prompt in voice_clone_prompts)):
            return None
        prompts = [prompt[0] for prompt in voice_clone_prompts]
        if len({id(prompt) for prompt in prompts}) != MIXED_VOICE_BATCH_SIZE or not self.mixed_voice_batch_available():
            return None
        limit = self._short_clone_limit(kwargs.get("max_new_tokens"), text, False)
        if limit is None:
            return None

        import torch

        kwargs.update(text=[text] * MIXED_VOICE_BATCH_SIZE, voice_clone_prompt=prompts, max_new_tokens=limit)
        device = self.cuda_device
        try:
            with torch.cuda.device(device), torch.random.fork_rng(devices=[device]):
                torch.default_generator.manual_seed(batch_seed)
                torch.cuda.manual_seed(batch_seed)
                wavs, rate, endings = self._run_batch(kwargs, limit, accelerated=False)
        except (ValueError, RuntimeError, TypeError, AttributeError) as error:
            # The scheduler retries these items through the scalar path on this model's device.
            self._mixed_voice_batch_disabled = True
            self.report["mixed_voice_batch_fallback"] = str(error)
            oom = isinstance(error, torch.cuda.OutOfMemoryError)
        else:
            if self.report.get("mixed_voice_batch_backend") != _MIXED_VOICE_BATCH_BACKEND:
                self.report["mixed_voice_batch_backend"] = _MIXED_VOICE_BATCH_BACKEND
                self._save()
            emit("tts_generation", "유사어 음성 생성", eta_eligible=True)
            return wavs, rate, limit, endings

        if oom:
            # The failed inference may retain buffers until its exception goes out of scope.
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        self._save()
        return None

    def generate_ending_repair_clone(self, *, attempt, **kwargs):
        """Capture EOS for exactly the waveform returned to the repair journal."""
        from .tts import _capture_qwen_termination, _short_clone_token_limit
        self._load()
        limit = _short_clone_token_limit(self.model, kwargs.get("max_new_tokens"), text=kwargs.get("text"),
                                         ending_attempt=attempt)
        if limit is not None:
            kwargs["max_new_tokens"] = limit
        emit("tts_generation", "끝 음절 재생성 중", eta_eligible=True)
        # This small, exceptional workload skips graph replay qualification so
        # every observed termination belongs to the returned synthesis.
        with _capture_qwen_termination(self.model, limit) as evidence:
            wavs, rate = self._eager(kwargs)
        return wavs, rate, evidence

    def generate_voice_clone(self, *, expected_reuses=None, termination=None, **kwargs):
        self._load()
        import torch
        from .tts import _capture_qwen_termination

        def observed(call, *args, **options):
            if termination is None:
                return call(*args, **options)
            with _capture_qwen_termination(self.model, kwargs.get("max_new_tokens")) as evidence:
                result = call(*args, **options)
            termination.update(evidence)
            return result

        if expected_reuses is None:
            expected_reuses = self.expected_reuses
        if self.disabled:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return observed(self._eager, kwargs)
        prompt = kwargs["voice_clone_prompt"]
        key = (id(prompt), identity({k: v for k, v in kwargs.items() if k != "voice_clone_prompt"}))
        if key in self.eager_prompts:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return observed(self._eager, kwargs)
        if key in self.checked:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            cpu_before = torch.get_rng_state()
            cuda_before = torch.cuda.get_rng_state(self.cuda_device)
            try:
                with torch.inference_mode():
                    return observed(self.model.generate_voice_clone, **kwargs)
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
            return observed(self._eager, kwargs)
        if expected_reuses == 1:
            emit("tts_generation", "호출어 음성 생성", eta_eligible=True)
            return observed(self._eager, kwargs)
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
            expected, rate = observed(self._eager, kwargs)
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
