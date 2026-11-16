import numpy as np
import pytest

from korean_openwakeword.calibration import PreparedScores
from korean_openwakeword.tune import iter_replay


def compare_with_cached_search(calls, peaks):
    clip = {"sample_id": "positive", "kind": "positive", "category": "clean", "calls": calls}
    runtime = {"threshold": .5, "consecutive_frames": 1, "warmup_s": 0., "cooldown_s": 0.,
               "call_postroll_s": 0.}
    values = np.zeros((16, 1), dtype=np.float32)
    values[peaks, 0] = .9
    offsets = [0, len(values)]
    replayed = next(iter_replay([clip], values, offsets, ["model"], runtime))["results"]["model"]["calls"]
    cached = PreparedScores([clip], offsets, runtime).summary(
        values[:, 0], .5, 1, {"id": "model", "kind": "single"})["all_positive"]
    assert [call["call_id"] for call in replayed] == [call["call_id"] for call in calls]
    assert cached["successes"] == sum(call["success"] for call in replayed)
    return replayed, cached


@pytest.mark.parametrize("reverse", [False, True])
def test_separate_calls_match_by_time_with_input_order_preserved(reverse):
    calls = [{"call_id": "early", "start_s": .16, "end_s": .32},
             {"call_id": "late", "start_s": .8, "end_s": .96}]
    if reverse:
        calls.reverse()
    matched, cached = compare_with_cached_search(calls, [2, 10])
    assert cached["successes"] == 2
    timestamps = {call["call_id"]: call["event_time_s"] for call in matched}
    assert timestamps == pytest.approx({"early": .24, "late": .88})


def test_equal_start_uses_first_input_call_when_windows_overlap():
    calls = [{"call_id": "first", "start_s": .16, "end_s": .40},
             {"call_id": "second", "start_s": .16, "end_s": .32}]
    matched, cached = compare_with_cached_search(calls, [2])
    first, second = matched
    assert first["success"] and first["ambiguous_event_assignment"]
    assert not second["success"]
    assert cached["latency_p95_s"] == pytest.approx(.24 - .40)
