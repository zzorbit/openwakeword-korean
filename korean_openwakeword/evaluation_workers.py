"""Bounded whole-session CPU workers; never split a stateful audio stream.

Every worker owns a reference detector. Frame order, reset history, gates and
ordered float64 ensemble arithmetic are identical to the sequential path.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from .data import source_key
from .events import EventDetector, PositiveTracker
from .runtime import WakeWordDetector
from .common import read_json, write_json

_DETECTOR = None
_MODEL_IDS = None


def available_memory_bytes():
    """Best-effort available physical RAM, without an optional dependency."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                            ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                            ("extended_available", ctypes.c_ulonglong)]
            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.available)
        elif hasattr(os, "sysconf"):
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return None


def worker_plan(requested, sessions, *, logical_cpus=None, available_bytes=None):
    """Use measured-safe CPU concurrency, capped by cores and free RAM.

    Auto avoids spawn overhead for fewer than 64 sessions and uses at most
    eight workers, bounded by half the logical cores and available RAM.
    One GiB per worker
    and two GiB left for the caller are conservative allocation limits, not a
    promise about arbitrary third-party model peak memory.
    """
    if requested != "auto" and (isinstance(requested, bool) or not str(requested).isdigit()
                                 or not 1 <= int(requested) <= 16):
        raise ValueError("Evaluation workers must be auto or an integer from 1 to 16")
    cpus = max(1, logical_cpus if logical_cpus is not None else (os.cpu_count() or 1))
    available = available_memory_bytes() if available_bytes is None else available_bytes
    memory_limit = max(1, (available - 2 * 1024 ** 3) // 1024 ** 3) if available is not None else 1
    wanted = min(8, max(1, cpus // 2)) if requested == "auto" else int(requested)
    if requested == "auto" and sessions < 64:
        wanted = 1
    selected = min(wanted, cpus, memory_limit, max(1, sessions))
    return {"requested_workers": requested, "workers": int(selected), "logical_cpus": cpus,
            "available_memory_bytes": available, "memory_budget_per_worker_bytes": 1024 ** 3,
            "backend": "reference_cpu_onnx", "parallel_unit": "whole_audio_session",
            "pending_sessions_limit": int(selected * 2), "onnx_threads_per_session": 1,
            "scope": "CPU arithmetic unchanged; GPU equivalence is not assumed"}


def score_session(detector, row, path, model_ids):
    """Score exactly one reset session with the original 80 ms reference path."""
    row = {**row, "calls": deepcopy(row.get("calls", []))}
    runtime = detector.runtime
    detector.reset()
    gates = {key: EventDetector(runtime["threshold"], runtime["consecutive_frames"], runtime["cooldown_s"])
             for key in model_ids}
    session = {"session_id": row["sample_id"], "source_id": source_key(row), "calls": row["calls"],
               "provenance": row.get("provenance", "qwen_tts" if row["kind"] == "positive" else "public_dataset"),
               "voice_family_id": row.get("voice_family_id"), "category": row["category"],
               "speaker_id": row.get("speaker_id")}
    session.update({key: row[key] for key in ("distance_m", "direction", "microphone") if key in row})
    for index, call in enumerate(session["calls"]):
        call.setdefault("call_id", row["sample_id"] + f":call{index}")
        call.setdefault("original_utterance_id", row.get("original_id", row["sample_id"]))
    trackers = {key: PositiveTracker(session, runtime["call_postroll_s"]) for key in model_ids}
    events = {key: [] for key in model_ids}
    max_scores = dict.fromkeys(model_ids, 0.0)
    frames = 0
    frame_scores = []
    with sf.SoundFile(path) as audio:
        if audio.samplerate != 16000 or audio.channels != 1 or audio.subtype != "PCM_16":
            raise ValueError("Scored WAV must already be mono16kHz PCM16; no playback-only gain")
        for call in session["calls"]:
            if not 0 <= call["start_s"] < call["end_s"] <= audio.frames / 16000:
                raise ValueError("Call boundary exceeds evaluation file")
            if call["start_s"] < runtime["warmup_s"]:
                raise ValueError("Positive call starts during excluded warmup; fix the test audio")
        while True:
            samples = audio.read(1280, dtype="int16")
            if len(samples) < 1280:
                break
            scores = detector.score_members(samples)
            if len(scores) == 3:
                scores = np.r_[scores, np.float32((np.float64(scores[0]) + np.float64(scores[1])
                                                   + np.float64(scores[2])) / 3)]
            frame_scores.append(scores)
            timestamp = (frames + 1) * .08
            eligible = frames * .08 + 1e-9 >= runtime["warmup_s"]
            for key, score in zip(model_ids, scores):
                decision = gates[key].update(float(score), timestamp, eligible)
                max_scores[key] = max(max_scores[key], float(score))
                if decision["event"]:
                    events[key].append(timestamp)
                if eligible and row["kind"] == "positive":
                    trackers[key].update(timestamp, float(score), decision)
            frames += 1
    results = {}
    for key in model_ids:
        calls = trackers[key].rows
        results[key] = {"calls": calls, "success": all(call["success"] for call in calls) if calls else None,
                        "events_s": events[key],
                        "false_activations": len(events[key]) if row["kind"] == "negative" else trackers[key].outside_call_events,
                        "max_score": max((call["max_wake_score"] or 0 for call in calls), default=max_scores[key]),
                        "latency_s": calls[0]["latency_s"] if len(calls) == 1 else None,
                        "activity": gates[key].activity_statistics()}
    clip = {**row, "path": str(path), "duration_s": frames * .08,
            "eligible_seconds": max(0, frames * .08 - runtime["warmup_s"]), "results": results}
    return clip, np.asarray(frame_scores, dtype=np.float32).reshape(-1, len(model_ids))


def _initialize_worker(bundle, model_ids):
    global _DETECTOR, _MODEL_IDS
    _DETECTOR = WakeWordDetector(bundle)
    _MODEL_IDS = model_ids
    frontend = _DETECTOR.model.preprocessor
    sessions = list(_DETECTOR.model.models.values()) + [frontend.melspec_model, frontend.embedding_model]
    for session in sessions:
        options = session.get_session_options()
        if session.get_providers() != ["CPUExecutionProvider"] or options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1:
            raise ValueError("Parallel evaluation requires the reference CPU ONNX path with one thread per session")


def _score_worker(item):
    index, row, path = item
    return index, *score_session(_DETECTOR, row, path, _MODEL_IDS)


def score_sessions(bundle, model_ids, items, workers, detector_factory=WakeWordDetector):
    """Yield in input order with at most 2×workers submitted sessions."""
    if workers == 1:
        detector = detector_factory(bundle)
        for index, row, path in items:
            yield index, *score_session(detector, row, path, model_ids)
        return
    context = multiprocessing.get_context("spawn")
    pending = deque()
    iterator = iter(items)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize_worker, initargs=(str(bundle), model_ids)) as pool:
        for _ in range(workers * 2):
            item = next(iterator, None)
            if item is None:
                break
            pending.append(pool.submit(_score_worker, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(_score_worker, item))


class EvaluationJournal:
    """Disk-backed scores plus a hash-checked ordered checkpoint every 32 clips.

    Only incomplete, explicitly resumed runs may reuse this directory. A crash
    can lose at most 31 already calculated sessions; frozen input hashes must
    still match. Successful scores.npz keeps the existing portable format.
    """
    def __init__(self, output, binding, offsets, model_ids, resume=False):
        self.output = Path(output)
        self.work = self.output / "_evaluation_work"
        self.state_path = self.work / "checkpoint.json"
        self.journal_path = self.work / "clips.jsonl"
        self.scores_path = self.work / "frames.npy"
        self.binding = binding
        self.offsets = offsets
        self.model_ids = model_ids
        self.shape = (int(offsets[-1]), len(model_ids))
        self.clips = []
        self.frame_hash = hashlib.sha256()
        self.clip_hash = hashlib.sha256()
        self.journal_bytes = 0
        self.pending_lines = []
        payload_bytes = self.shape[0] * self.shape[1] * 4
        if shutil.disk_usage(self.output).free < payload_bytes * 2 + 64 * 1024 ** 2:
            raise ValueError("Insufficient project disk space for evaluation score cache and publication")
        if resume:
            if not self.state_path.is_file():
                raise ValueError("No resumable evaluation checkpoint; choose a new output directory")
            try:
                state = read_json(self.state_path)
                if state.get("schema_version") != 1 or state.get("binding") != binding or state.get("shape") != list(self.shape):
                    raise ValueError("Evaluation resume inputs/model/runtime changed; choose a new output directory")
                self.scores = np.load(self.scores_path, mmap_mode="r+")
                if self.scores.dtype != np.float32 or self.scores.shape != self.shape:
                    raise ValueError("Evaluation resume score cache shape or precision changed")
                count = state["committed_sessions"]
                if not isinstance(count, int) or not 0 <= count < len(offsets):
                    raise ValueError("Invalid evaluation checkpoint session count")
                for start in range(0, int(offsets[count]), 65536):
                    self.frame_hash.update(self.scores[start:min(start + 65536, int(offsets[count]))].tobytes())
                if self.frame_hash.hexdigest() != state["frame_prefix_sha256"]:
                    raise ValueError("Evaluation resume score prefix changed")
                self.journal_bytes = state["journal_bytes"]
                with self.journal_path.open("r+b") as stream:
                    raw = stream.read(self.journal_bytes)
                    self.clip_hash.update(raw)
                    if self.clip_hash.hexdigest() != state["journal_prefix_sha256"]:
                        raise ValueError("Evaluation resume clip journal changed")
                    self.clips = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
                    if len(self.clips) != count:
                        raise ValueError("Evaluation resume clip journal count changed")
                    stream.truncate(self.journal_bytes)
            except BaseException:
                self.close()
                raise
        else:
            self.work.mkdir()
            self.scores = np.lib.format.open_memmap(self.scores_path, mode="w+", dtype=np.float32, shape=self.shape)
            self.journal_path.touch()
            self.commit()
        self.resumed_sessions = len(self.clips)

    def add(self, index, clip, scores):
        if index != len(self.clips) or scores.shape != (self.offsets[index + 1] - self.offsets[index], len(self.model_ids)):
            raise ValueError("Evaluation worker session order or frame count changed")
        if scores.dtype != np.float32 or not np.isfinite(scores).all():
            raise ValueError("Invalid evaluation worker score precision or values")
        self.scores[self.offsets[index]:self.offsets[index + 1]] = scores
        self.frame_hash.update(scores.tobytes())
        self.clips.append(clip)
        encoded = (json.dumps(clip, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self.pending_lines.append(encoded)
        self.clip_hash.update(encoded)
        self.journal_bytes += len(encoded)
        if len(self.pending_lines) >= 32:
            self.commit()

    def commit(self):
        self.scores.flush()
        with self.journal_path.open("ab") as stream:
            stream.writelines(self.pending_lines)
            stream.flush()
            os.fsync(stream.fileno())
        self.pending_lines.clear()
        write_json(self.state_path, {"schema_version": 1, "binding": self.binding, "shape": list(self.shape),
                                   "committed_sessions": len(self.clips), "journal_bytes": self.journal_bytes,
                                   "frame_prefix_sha256": self.frame_hash.hexdigest(),
                                   "journal_prefix_sha256": self.clip_hash.hexdigest()})

    def publish_scores(self):
        self.commit()
        temporary = self.work / "publishing_scores.npz"
        np.savez_compressed(temporary, scores=self.scores, offsets=np.asarray(self.offsets, dtype=np.int64),
                            model_ids=np.asarray(self.model_ids))
        os.replace(temporary, self.output / "scores.npz")

    def publish_clips(self):
        """Keep portable JSON without constructing a second huge JSON string."""
        temporary = self.work / "publishing_clips.json"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("[\n")
            for index, row in enumerate(self.clips):
                if index:
                    stream.write(",\n")
                encoded = json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("  " + encoded.replace("\n", "\n  "))
            stream.write("\n]\n")
        os.replace(temporary, self.output / "clips.json")

    def close(self):
        if getattr(self, "scores", None) is not None:
            self.scores._mmap.close()
            self.scores = None

    def discard_completed_work(self):
        """Delete only this run's known scratch files after a completion receipt."""
        self.close()
        if not (self.output / "receipt.json").is_file():
            raise ValueError("Preserve evaluation checkpoint until a completion receipt exists")
        for path in (self.scores_path, self.journal_path, self.state_path,
                     self.work / "publishing_scores.npz", self.work / "publishing_clips.json"):
            path.unlink(missing_ok=True)
        self.work.rmdir()
