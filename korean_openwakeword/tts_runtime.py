"""Lazy CUDA acceleration with equivalent output and measured warm speedup.

Keep the original TTS contract, seeds, sampling, weights and technical QC.
Runtime qualification clips are never added to training or counted as listening.
"""
import time
import numpy as np
from .common import now, write_json, identity
from .project import load_project


class CloneRuntime:
    def __init__(self, model_path, settings, root, backend="auto", cpu_threads=4):
        self.model_path, self.settings, self.root = model_path, settings, root
        self.backend, self.cpu_threads = backend, cpu_threads
        self.model = None
        self.predictor = self.talker = None
        self.checked = {}
        self.disabled = backend == "eager"
        self.report = {"requested_backend": backend, "created_at": now(), "qualifications": [],
                       "minimum_warm_speedup_fraction": .05,
                       "qualification_clips_added_to_training": 0, "Final_audio_accessed": False}

    def _load(self):
        if self.model is not None:
            return
        import torch
        from .tts import _load_model
        if torch.cuda.is_available() and self.settings["device"] != "cpu":
            torch.set_num_threads(self.cpu_threads)
        self.model = _load_model(self.model_path, self.settings, self.root)
        devices = {str(p.device) for p in self.model.model.parameters()}
        dtypes = {str(p.dtype) for p in self.model.model.parameters()}
        self.report.update(parameter_devices=sorted(devices), parameter_dtypes=sorted(dtypes),
                           cpu_threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           torch=torch.__version__, transformers=__import__("transformers").__version__)
        if devices != {"cuda:0"} or dtypes != {"torch.bfloat16"} or self.settings["attention"] != "sdpa":
            self.disabled = True
        self._save()

    def _save(self):
        write_json(self.root / "runtime" / "tts_execution.json", self.report)

    def create_voice_clone_prompt(self, **kwargs):
        self._load()
        return self.model.create_voice_clone_prompt(**kwargs)

    def _enable(self):
        import torch
        from .tts_cuda_graph import enable_predictor_graph, enable_exact_talker_graph
        # Capture warmup must never advance the caller's sample RNG.
        talker = self.model.model.talker
        original_predictor = talker.code_predictor.generate
        original_talker = talker.model.forward
        try:
            with torch.random.fork_rng(devices=[0]), torch.inference_mode():
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
        talker = self.model.model.talker
        predictor = talker.code_predictor
        inner = talker.model
        fast_predictor, fast_talker = predictor.generate, inner.forward
        if self.predictor:
            predictor.generate = self.predictor["original_generate"]
            inner.forward = self.talker["original_forward"]
        try:
            with torch.inference_mode():
                return self.model.generate_voice_clone(**kwargs)
        finally:
            predictor.generate, inner.forward = fast_predictor, fast_talker

    def generate_voice_clone(self, **kwargs):
        self._load()
        import torch
        if self.disabled:
            return self._eager(kwargs)
        key = (id(kwargs["voice_clone_prompt"]), identity({k:v for k,v in kwargs.items() if k != "voice_clone_prompt"}))
        if key in self.checked:
            with torch.inference_mode():
                return self.model.generate_voice_clone(**kwargs)
        # Compare each actual reference's first requested utterance at exactly
        # the caller's seed. All qualification repeats are ephemeral; only one
        # result and the RNG progression of one eager call are retained.
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(0).clone()
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.synchronize(); started = time.perf_counter()
            expected, rate = self._eager(kwargs)
            torch.cuda.synchronize(); eager_seconds = time.perf_counter()-started
            cpu_after, cuda_after = torch.get_rng_state(), torch.cuda.get_rng_state(0)
        qualification = {"accepted": False, "eager_seconds": eager_seconds,
                         "minimum_warm_speedup_fraction": .05, "at": now()}
        try:
            if not self.predictor:
                self._enable()
            torch.cuda.synchronize(); started = time.perf_counter()
            with torch.inference_mode():
                actual, actual_rate = self.model.generate_voice_clone(**kwargs)
            torch.cuda.synchronize(); graph_seconds = time.perf_counter()-started
            from .tts import technical_qc
            from .audio import pcm16
            eq_qc, eq_signal = technical_qc(expected[0], rate)
            if eq_signal is None:
                raise ValueError("No comparable PCM for CUDA qualification")

            def compare_waveform(values, sample_rate):
                qc, signal = technical_qc(values[0], sample_rate)
                same_pcm = sample_rate == rate and signal is not None and np.array_equal(pcm16(eq_signal[0]), pcm16(signal[0]))
                return same_pcm, qc == eq_qc

            cold_pcm, cold_qc = compare_waveform(actual, actual_rate)
            cold_cpu_rng = torch.equal(cpu_after, torch.get_rng_state())
            cold_cuda_rng = torch.equal(cuda_after, torch.cuda.get_rng_state(0))
            qualification.update(graph_cold_seconds=graph_seconds,
                cold_pcm_samples_equal=cold_pcm, cold_quality_checks_equal=cold_qc,
                cold_cpu_rng_equal=cold_cpu_rng, cold_cuda_rng_equal=cold_cuda_rng)
            if not all((cold_pcm, cold_qc, cold_cpu_rng, cold_cuda_rng)):
                raise ValueError("Eager/cold graph PCM, QC or sampling RNG differs")

            # Graph creation and per-length capture belong to the first run.
            # Repeat once from the same seed to time only the warmed backend.
            with torch.random.fork_rng(devices=[0]):
                torch.set_rng_state(cpu_before); torch.cuda.set_rng_state(cuda_before, 0)
                torch.cuda.synchronize(); started = time.perf_counter()
                with torch.inference_mode():
                    warm, warm_rate = self.model.generate_voice_clone(**kwargs)
                torch.cuda.synchronize(); warm_seconds = time.perf_counter()-started
                warm_cpu_rng = torch.equal(cpu_after, torch.get_rng_state())
                warm_cuda_rng = torch.equal(cuda_after, torch.cuda.get_rng_state(0))
            warm_pcm, warm_qc = compare_waveform(warm, warm_rate)
            faster = eager_seconds > 0 and warm_seconds >= 0 and warm_seconds <= eager_seconds * .95
            qualification.update(graph_warm_seconds=warm_seconds,
                warm_pcm_samples_equal=warm_pcm, warm_quality_checks_equal=warm_qc,
                warm_cpu_rng_equal=warm_cpu_rng, warm_cuda_rng_equal=warm_cuda_rng,
                pcm_samples_equal=cold_pcm and warm_pcm,
                quality_checks_equal=cold_qc and warm_qc,
                sampling_rng_equal=cold_cpu_rng and cold_cuda_rng and warm_cpu_rng and warm_cuda_rng,
                warm_speedup_fraction=(1 - warm_seconds / eager_seconds) if eager_seconds > 0 else None,
                minimum_warm_speedup_met=faster)
            if not all((warm_pcm, warm_qc, warm_cpu_rng, warm_cuda_rng)):
                raise ValueError("Eager/warm graph PCM, QC or sampling RNG differs")
            if not faster:
                raise ValueError("Warm CUDA graph is not at least 5% faster than eager")
        except (ValueError, RuntimeError, TypeError, AttributeError) as error:
            self.disabled = True
            if self.predictor:
                self.model.model.talker.code_predictor.generate = self.predictor["original_generate"]
                self.model.model.talker.model.forward = self.talker["original_forward"]
            torch.set_rng_state(cpu_after); torch.cuda.set_rng_state(cuda_after, 0)
            qualification.update(fallback="eager", reason=str(error))
            self.report["qualifications"].append(qualification)
            self.report["actual_backend"] = "eager"
            self._save()
            return expected, rate
        # Retain the prompt: Python must not reuse an ID for another reference.
        self.checked[key] = kwargs["voice_clone_prompt"]
        qualification.update(accepted=True, backend="exact_shape_predictor_and_talker_cuda_graph")
        self.report["qualifications"].append(qualification)
        self.report["actual_backend"] = "exact_shape_predictor_and_talker_cuda_graph"
        self._save()
        return actual, actual_rate


def generate_with_runtime(project, split="train", limit=None, pilot=False, method="clone", backend="auto", cpu_threads=4):
    if backend not in ("auto", "eager") or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Use backend auto/eager and a positive CPU thread count")
    from .tts import generate
    root, config = load_project(project)
    # Lazy loading retains generate's approval, split and Final checks before
    # opening weights, prompts or audio. Design retains the original path.
    model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, backend, cpu_threads) if method == "clone" else None
    return generate(project, split, limit, pilot, method, model=model)
