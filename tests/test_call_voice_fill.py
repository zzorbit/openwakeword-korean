"""Same-voice production batches resume from complete, checked journal evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from korean_openwakeword import pilot, tts
from korean_openwakeword.common import digest, read_json, read_rows, write_json
from test_korean_library_tts_journal import SeededQwen, approved_project, cpu_seed


class BatchModel:
    def __init__(self, reject_first=False):
        self.reject_first = reject_first
        self.calls = []

    def create_voice_clone_prompt(self, *, ref_text, **kwargs):
        return ref_text

    def generate_short_clone_batch(self, *, batch_seed, expected_reuses, voice_clone_prompt, **kwargs):
        self.calls.append(("batch", voice_clone_prompt, batch_seed, expected_reuses))
        wavs = [SeededQwen.waveform((batch_seed + index) % 2**32) for index in range(tts.CALL_BATCH_SIZE)]
        endings = [{"termination": "eos", "generated_tokens": 12, "token_limit": 51} for _ in wavs]
        if self.reject_first:
            endings[2] = {"termination": "token_limit", "generated_tokens": 51, "token_limit": 51}
            self.reject_first = False
        return wavs, 16000, 51, endings

    def generate_short_clone(self, *, retry, termination, voice_clone_prompt, **kwargs):
        import numpy as np

        seed = int(np.random.get_state()[1][0])
        self.calls.append(("single", voice_clone_prompt, seed, retry))
        termination.update(termination="eos", generated_tokens=12, token_limit=51)
        return [SeededQwen.waveform(seed)], 16000, 51


def fixture(tmp_path, monkeypatch, count, *, revision="voice-fill"):
    project = approved_project(tmp_path / "word", "clone", count=count)
    config = read_json(project)
    references = read_rows(project.parent / "data/references/train/references.jsonl")
    references = [dict(reference, text=reference["voice_family_id"]) for reference in references]
    scope = {"kind": "production", "revision": revision, "references": references,
             "excluded_sha256": [], "schedule": tts.VOICE_FILL_SCHEDULE}
    monkeypatch.setattr(tts, "_seed", cpu_seed)
    return project, config, scope


def test_each_voice_fills_its_target_before_next_and_retries_only_one_failed_batch_slot(tmp_path, monkeypatch):
    project, config, scope = fixture(tmp_path, monkeypatch, 18)
    model = BatchModel(reject_first=True)
    result = tts.generate(project, limit=18, pilot=True, model=model, staged_scope=scope)
    assert result["full_target_reached"] and result["technical_pass"] == 18 and result["attempts"] == 19
    rows = read_rows(result["review_queue"])
    assert [row["slot"] for row in rows] == [0, 2, 4, 6, 4, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert [row["slot"] for row in rows if not row["qc"]["passed"]] == [4]
    assert [row["slot"] for row in rows if row["attempt"] > 0] == [4]
    assert rows[2]["ending_boundary_qc"]["termination"] == "token_limit"
    assert rows[4]["ending_boundary_qc"]["termination"] == "eos"
    assert [call[:2] for call in model.calls] == [
        ("batch", "voice_0"), ("single", "voice_0"), ("batch", "voice_0"), ("single", "voice_0"),
        ("batch", "voice_1"), ("batch", "voice_1"), ("single", "voice_1"),
    ]
    assert [call[3] for call in model.calls if call[0] == "batch"] == [2, 1, 2, 1]
    assert [call[3] for call in model.calls if call[0] == "single"] == [True, False, False]
    assert all(row["cuda_batch"]["slots"] == [0, 2, 4, 6] for row in rows[:4])
    clone = BatchModel()
    assert tts.generate(project, limit=18, pilot=True, model=clone, staged_scope=scope)["full_target_reached"]
    assert clone.calls == []


def test_resume_commits_pending_batch_after_one_row_and_a_torn_row_without_resynthesis(tmp_path, monkeypatch):
    project, _, scope = fixture(tmp_path, monkeypatch, 16)
    model = BatchModel(reject_first=True)
    original_finish = tts._finish_call_batch

    def interrupt(output, intent):
        write_torn_batch(output, intent)
        raise KeyboardInterrupt

    monkeypatch.setattr(tts, "_finish_call_batch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tts.generate(project, limit=16, pilot=True, model=model, staged_scope=scope)
    assert [call[:2] for call in model.calls] == [("batch", "voice_0")]
    directory, _ = tts.call_generation_location(project.parent, "train", scope)
    pending = directory / "pending_cuda_batch.json"
    assert pending.is_file()
    staged = read_json(pending)["rows"]
    monkeypatch.setattr(tts, "_finish_call_batch", original_finish)
    restarted = BatchModel()
    result = tts.generate(project, limit=16, pilot=True, model=restarted, staged_scope=scope)
    assert result["full_target_reached"] and result["attempts"] == 17
    assert not pending.exists()
    rows = read_rows(result["review_queue"])
    assert rows[:4] == staged
    assert [call[:2] for call in restarted.calls] == [
        ("single", "voice_0"), ("batch", "voice_0"), ("batch", "voice_1"), ("batch", "voice_1"),
    ]
    assert [row["slot"] for row in rows if row["attempt"] > 0] == [4]


def write_torn_batch(output, intent):
    rows = [json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") for row in intent["rows"]]
    with (output / "candidates.jsonl").open("ab") as journal:
        journal.write(rows[0] + os.linesep.encode() + rows[1][:20])
        journal.flush()
        os.fsync(journal.fileno())


def test_abrupt_worker_exit_archives_its_lock_and_recovers_accepted_batch(tmp_path, monkeypatch):
    from korean_openwakeword.studio import StudioManager
    from korean_openwakeword.studio_jobs import process_identity

    project, _, scope = fixture(tmp_path, monkeypatch, 8)
    scope_file = tmp_path / "scope.json"
    write_json(scope_file, scope)
    crash = """
import os
from pathlib import Path
import sys
from test_call_voice_fill import BatchModel, cpu_seed, read_json, tts, write_torn_batch
from korean_openwakeword.common import write_json
from korean_openwakeword.studio_jobs import process_identity

def interrupt(output, intent):
    write_torn_batch(output, intent)
    write_json(Path(sys.argv[2]).with_name('worker.json'), {
        'pid': os.getpid(), 'creation_identity': process_identity(os.getpid()), 'task_id': 'test-positives'})
    os._exit(86)

tts._seed = cpu_seed
tts._finish_call_batch = interrupt
tts.generate(Path(sys.argv[1]), limit=8, pilot=True, model=BatchModel(reject_first=True),
             staged_scope=read_json(sys.argv[2]))
"""
    test_dir = Path(__file__).resolve().parent
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(test_dir), str(test_dir.parent), os.environ.get("PYTHONPATH", "")]))
    child = subprocess.run([sys.executable, "-B", "-c", crash, str(project), str(scope_file)],
                           cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45)
    assert child.returncode == 86, child.stderr
    worker = read_json(tmp_path / "worker.json")
    assert worker["creation_identity"] and process_identity(worker["pid"]) is None
    directory, _ = tts.call_generation_location(project.parent, "train", scope)
    intent = directory / "pending_cuda_batch.json"
    saved = read_json(intent)["rows"]
    manager = StudioManager(tmp_path)
    manager._recover_dead_locks({"project": str(project), "job": worker})
    archives = list((project.parent / ".studio/recovered_locks").glob("*.json"))
    assert len(archives) == 1 and read_json(archives[0])["record"]["pid"] == worker["pid"]

    resumed = BatchModel()
    result = tts.generate(project, limit=8, pilot=True, model=resumed, staged_scope=scope)
    rows = read_rows(result["review_queue"])
    assert result["full_target_reached"] and rows[:4] == saved and not intent.exists()
    assert [call[:2] for call in resumed.calls] == [("single", "voice_0"), ("batch", "voice_1")]
    assert [row["slot"] for row in rows if row["attempt"] > 0] == [4]


def test_studio_pauses_only_after_publishing_current_batch_and_resumes_at_next(tmp_path, monkeypatch):
    from korean_openwakeword.studio_jobs import request_stage_pause
    from korean_openwakeword.studio_worker import _speech_generation_stage

    project, _, scope = fixture(tmp_path, monkeypatch, 8)
    request = {"project": project, "workspace": tmp_path, "settings": {}, "stage": "positives", "task_id": "first"}
    request_path = tmp_path / "first/request.json"
    first_job = {**request, "directory": request_path.parent}

    class PausingBatch(BatchModel):
        def __init__(self, job):
            super().__init__()
            self.job = job

        def generate_short_clone_batch(self, **kwargs):
            result = super().generate_short_clone_batch(**kwargs)
            request_stage_pause({}, self.job, candidate_preparation=False)
            return result

    def run_stage(model):
        def run(project_path, _workspace, _settings, stage, _context, *, pause_requested):
            assert stage == "positives"
            return tts.generate(project_path, limit=8, pilot=True, model=model,
                                staged_scope=scope, pause_requested=pause_requested)
        return run

    first_model = PausingBatch(first_job)
    result = _speech_generation_stage(request, request_path, run_stage(first_model))
    directory, _ = tts.call_generation_location(project.parent, "train", scope)
    saved = read_rows(directory / "candidates.jsonl")
    assert result == {"state": "paused"}
    assert len(saved) == 4 and all(row["qc"]["passed"] for row in saved)
    assert [call[:2] for call in first_model.calls] == [("batch", "voice_0")]
    assert not (directory / "pending_cuda_batch.json").exists()

    next_request = {**request, "task_id": "next"}
    next_path = tmp_path / "next/request.json"
    next_model = PausingBatch({**next_request, "directory": next_path.parent})
    result = _speech_generation_stage(next_request, next_path, run_stage(next_model))
    assert result["full_target_reached"] and read_rows(result["review_queue"])[:4] == saved
    assert [call[:2] for call in next_model.calls] == [("batch", "voice_1")]


def test_exhausted_first_voice_blocks_next_voice_and_a_new_bank_selects_voice_fill(tmp_path, monkeypatch):
    project, _, scope = fixture(tmp_path, monkeypatch, 8)
    banks = {"train": scope["references"], "validation": []}

    class FailedVoice(BatchModel):
        def generate_short_clone_batch(self, **kwargs):
            return None

        def generate_short_clone(self, **kwargs):
            wavs, rate, cap = super().generate_short_clone(**kwargs)
            return [wavs[0] * 0], rate, cap

    model = FailedVoice()
    chosen, _, results = pilot._generate_production_banks(
        project, project.parent, banks, {"train": 8, "validation": 1}, scope["revision"], set(), model)
    assert all(part["schedule"] == tts.VOICE_FILL_SCHEDULE for part in chosen.values())
    assert not results["train"]["full_target_reached"] and not results["validation"]["full_target_reached"]
    assert {row["voice_family_id"] for row in results["train"]["exhausted_slots"]} == {"voice_0"}
    assert len(model.calls) == 4 * tts.CLONE_ATTEMPTS and all(call[1] == "voice_0" for call in model.calls)


def test_new_studio_bank_completes_train_then_validation_and_reopens_without_synthesis(tmp_path, monkeypatch):
    import soundfile as sf

    project, _, scope = fixture(tmp_path, monkeypatch, 8)
    root = project.parent
    validation = []
    for index, source in enumerate(scope["references"]):
        path = root / "data/references/validation" / f"voice_{index}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, SeededQwen.waveform(100 + index), 16000, subtype="FLOAT")
        family = f"evaluation_voice_{index}"
        validation.append({**source, "path": path.relative_to(root).as_posix(), "sha256": digest(path),
                           "split": "validation", "voice_family_id": family, "text": family})
    banks = {"train": scope["references"], "validation": validation}
    targets = {"train": 8, "validation": 8}
    model = BatchModel()
    scopes, directories, results = pilot._generate_production_banks(
        project, root, banks, targets, scope["revision"], set(), model)
    assert [call[:2] for call in model.calls] == [
        ("batch", "voice_0"), ("batch", "voice_1"),
        ("batch", "evaluation_voice_0"), ("batch", "evaluation_voice_1"),
    ]
    assert all(summary["full_target_reached"] and summary["technical_pass"] == 8 for summary in results.values())
    journals = [directory / "candidates.jsonl" for directory in directories.values()]
    saved = [digest(journal) for journal in journals]
    resumed = BatchModel()
    next_scopes, next_directories, next_results = pilot._generate_production_banks(
        project, root, banks, targets, scope["revision"], set(), resumed)
    assert (next_scopes, next_directories, next_results) == (scopes, directories, results)
    assert not resumed.calls and [digest(journal) for journal in journals] == saved
