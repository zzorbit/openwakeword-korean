"""Optional, invocation-scoped task telemetry; never a completion receipt.

Library callers do not need a Studio worker. A worker installs one callback
for its own request, and the callback owns persistence and throttling.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import time


_CALLBACK = ContextVar("studio_task_progress_callback", default=None)
_GENERATION_CLOCK = ContextVar("studio_generation_clock", default=None)


@contextmanager
def task_progress(callback):
    if callback is not None and not callable(callback):
        raise ValueError("Task progress callback must be callable or None")
    token = _CALLBACK.set(callback)
    try:
        yield
    finally:
        _CALLBACK.reset(token)


def emit(phase, label, done=None, total=None, unit=None, **details):
    callback = _CALLBACK.get()
    if callback is not None:
        clock = _GENERATION_CLOCK.get()
        if clock is not None and phase == "tts_qualification":
            # The qualification output is kept, but its cost and count cannot
            # be compared to a normal generation attempt.
            clock.qualified = True
        callback({"phase": phase, "label": label, "done": done,
                  "total": total, "unit": unit, **details})


class _GenerationAttempt:
    def __init__(self):
        self.started = time.perf_counter()
        self.qualified = False

    def committed(self, passed):
        """Count an attempt only after its technical verdict is durable."""
        if not self.qualified:
            emit("tts_generation", "음성 생성", unit="개", eta_eligible=True,
                 rate_sample=(int(passed), time.perf_counter() - self.started))


@contextmanager
def generation_attempt():
    """Time a new synthesis attempt, after model and reference prompt setup."""
    if _CALLBACK.get() is None:
        yield None
        return
    emit("tts_generation", "음성 생성", eta_eligible=True)
    attempt = _GenerationAttempt()
    token = _GENERATION_CLOCK.set(attempt)
    try:
        yield attempt
    finally:
        _GENERATION_CLOCK.reset(token)


def activity(phase, label, done=None, total=None, unit="frames", **details):
    """Report current computation separately from durable completed sessions."""
    emit("scoring_activity", label, activity={"phase": phase, "label": label,
         "done": done, "total": total, "unit": unit, **details}, eta_eligible=False)
