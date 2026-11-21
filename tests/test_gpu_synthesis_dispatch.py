from contextlib import nullcontext
import random
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from korean_openwakeword import hard_tts, pilot, reference_candidates as candidates, tts
from korean_openwakeword.common import digest, read_json, read_rows
from test_call_voice_fill import write_torn_batch
from test_hard_voice_batching import MixedVoiceModel, hard_project
from test_korean_library_hard_tts import PHRASES
from test_korean_library_reference_candidates import pool, waveform
from test_korean_library_tts_batch_runtime import mock_batch_runtime
from test_korean_library_tts_journal import approved_project, cpu_seed, SeededQwen


def test_reference_batch_checks_each_stop_and_reuses_bound_raw_audio_after_interruption(pool, monkeypatch):
    root, project, _, _, saved, _ = pool
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved), rng=random.Random(19))
    output = candidates.candidate_jobs(root) / plan["batch_id"] / "audio"
    monkeypatch.setattr(tts, "_qwen_audio_token_limit", lambda *args: 12)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: None)
    calls = []

    class Talker:
        def generate(self, *, max_new_tokens, eos_token_id, sequences):
            return SimpleNamespace(sequences=torch.tensor(sequences))

    class Model:
        device = torch.device("cuda:0")
        model = SimpleNamespace(talker=Talker())

        def generate_voice_design(self, *, text, language, instruct, max_new_tokens, **kwargs):
            calls.append(instruct)
            if len(calls) == 2:
                raise RuntimeError("interrupted scalar retry")
            if isinstance(instruct, list):
                self.model.talker.generate(max_new_tokens=max_new_tokens, eos_token_id=99,
                    sequences=[[3] * 11 + [99], [4] * 12, [5] * 8 + [99] * 4, [6] * 10 + [99] * 2])
                return [np.tile(waveform(700 + index), 4) for index in range(4)], 16000
            self.model.talker.generate(max_new_tokens=max_new_tokens, eos_token_id=99, sequences=[[3] * 11 + [99]])
            return [np.tile(waveform(800 + len(calls)), 4)], 16000

    model = Model()
    with pytest.raises(RuntimeError, match="interrupted scalar"):
        candidates.generate_candidate_batch(project, output, "job", plan, [], load_model=lambda *args: model)
    original_receipt = digest(output / "design_batch/batch.json")
    saved_row = output / f"{plan['candidates'][0]['candidate_id']}.json"
    original_candidate = digest(saved_row)
    rows = candidates.generate_candidate_batch(project, output, "job", plan, [], load_model=lambda *args: model)
    assert [row["attempt"] for row in rows] == [0, 1, 0, 0, 0]
    assert [index for index, row in enumerate(rows) if "reference_design_batch" in row] == [0, 2, 3]
    assert [len(call) for call in calls if isinstance(call, list)] == [4]
    assert digest(output / "design_batch/batch.json") == original_receipt and digest(saved_row) == original_candidate
    rejected = read_json(output / "design_batch/1.wav.stop.json")
    assert rejected["termination"] == "token_limit"
    assert all(candidates._verify_candidate(root, row).exists() for row in rows)


def test_preflight_torn_batch_resumes_bound_waves_and_retries_only_failed_voice(tmp_path, monkeypatch):
    project = approved_project(tmp_path / "voice", "clone", count=4)
    config = read_json(project)
    root = project.parent
    directory = root / "data/references/train"
    refs = read_rows(directory / "references.jsonl")
    for index in (2, 3):
        path = directory / f"voice_{index}.wav"
        sf.write(path, SeededQwen.waveform(index), 16000, subtype="FLOAT")
        refs.append(dict(refs[0], path=str(path.relative_to(root)), sha256=digest(path), voice_family_id=f"voice_{index}"))
    scope = {"kind": "preflight", "revision": "pilot-four", "references": refs, "excluded_sha256": [],
             "batch_policy": tts.PREFLIGHT_BATCH_POLICY}
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    model = MixedVoiceModel(reject_one=True)
    finish = tts._finish_call_batch

    def interrupt(output, intent):
        write_torn_batch(output, intent)
        raise KeyboardInterrupt

    monkeypatch.setattr(tts, "_finish_call_batch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pilot._preflight_scope_samples(project, root, config, "train", scope, model, ceiling=1)
    out, _ = tts.call_generation_location(root, "train", scope)
    saved = read_json(out / "pending_cuda_batch.json")["rows"]
    monkeypatch.setattr(tts, "_finish_call_batch", finish)

    def stop():
        return True

    with pytest.raises(tts.CallGenerationPaused):
        pilot._preflight_scope_samples(project, root, config, "train", scope, model, ceiling=2,
                                       pause_requested=stop)
    assert read_rows(out / "candidates.jsonl") == saved
    rows, passing, _ = pilot._preflight_scope_samples(project, root, config, "train", scope, model, ceiling=2)
    assert rows[:4] == saved and list(sorted(passing)) == [0, 1, 2, 3]
    assert rows[4]["attempt"] == rows[4]["slot"] == 1
    assert len(model.batches) == len(model.singles) == 1
    assert [row["voice_family_id"] for row in rows[:4]] == [f"voice_{index}" for index in range(4)]


def test_graph_rejection_reuses_same_device_eager_batch(tmp_path, monkeypatch):
    runtime, _, timeline, _, stock, graph, _ = mock_batch_runtime(tmp_path, monkeypatch, scenario="last_pcm")
    runtime._mixed_voice_batch_capable = True
    prompt = [object()]
    options = dict(batch_seed=91, expected_reuses=30, allow_eager=True, text="보라야", language="Korean",
                   voice_clone_prompt=prompt, max_new_tokens=51)
    result = runtime.generate_short_clone_batch(**options)
    replay = runtime.generate_short_clone_batch(**options)
    assert len(result[0]) == 4 and result[1:] == replay[1:]
    assert all(np.array_equal(a, b) for a, b in zip(result[0], replay[0], strict=True))
    assert timeline == [stock, graph, stock]
    assert runtime.report["same_voice_eager_batch_backend"] == "eager_cuda_same_voice_batch4"


def test_hard_failed_four_uses_two_and_reopens_frozen_receipts(hard_project):
    class ShrinkingModel(MixedVoiceModel):
        def mixed_voice_pair_available(self):
            return True

        def generate_short_clone_mixed_voice_batch(self, *, batch_seed, voice_clone_prompts, text, language, **kwargs):
            size = len(voice_clone_prompts)
            self.batches.append(size)
            if size == 4:
                return None
            waves = [SeededQwen.waveform((batch_seed + index) % 2**32) for index in range(size)]
            ends = [dict(termination="eos", generated_tokens=10, token_limit=51) for _ in waves]
            return waves, 16000, 51, ends

    model = ShrinkingModel()
    result = hard_tts.generate_hard(hard_project, PHRASES[:1], model=model, samples_per_phrase=5)
    rows = read_rows(result["candidates"])
    assert result["full_target_reached"] and [row["slot"] for row in rows] == [0, 1, 2, 3, 4]
    assert model.batches == [4, 2, 2] and len(model.singles) == 1
    assert [row["cuda_batch"]["slots"] for row in rows[:4]] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    frozen = digest(result["candidates"])
    assert hard_tts.generate_hard(hard_project, PHRASES[:1], samples_per_phrase=5)["full_target_reached"]
    assert digest(result["candidates"]) == frozen
