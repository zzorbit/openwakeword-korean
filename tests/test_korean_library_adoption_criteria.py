"""Displayed goals must describe the exact gates used for a saved judgement."""
from copy import deepcopy

import pytest

from korean_openwakeword.evaluate import adoption, adoption_criteria


def _inputs():
    config = {"evaluation_targets": {"clean_recall": .973, "snr10_recall": .941,
              "snr5_recall": .883, "fa_per_hour": .125, "latency_p95_s": .48},
              "targets": {"development_negative_hours": [32, 40]}}
    summary = {"id": "one", "kind": "single", "clean": {"recall": .973},
               "conditions": {f"{category}:{snr}": {"recall": value}
                              for category in ("environment", "music", "competing_speech")
                              for snr, value in ((10, .941), (5, .883))},
               "negative": {"hours": 32, "fa_per_hour": .125},
               "primary": {"latency_p95_s": .48}}
    return config, summary


def test_snapshot_uses_explicit_custom_targets_without_mutating_or_aliasing_config():
    config, _ = _inputs()
    before = deepcopy(config)
    snapshot = adoption_criteria(config)
    assert snapshot == {"schema_version": 1, **config["evaluation_targets"], "minimum_negative_hours": 32}
    snapshot["clean_recall"] = .1
    assert config == before
    config["evaluation_targets"]["fa_per_hour"] = .07
    config["targets"]["development_negative_hours"][0] = 50
    assert snapshot["fa_per_hour"] == .125 and snapshot["minimum_negative_hours"] == 32


def test_all_inclusive_boundaries_pass_and_stricter_targets_fail_without_rescoring():
    config, summary = _inputs()
    original = deepcopy(summary)
    assert adoption([summary], config)["passed"]
    config["evaluation_targets"]["clean_recall"] += .001
    config["evaluation_targets"]["snr10_recall"] += .001
    config["evaluation_targets"]["snr5_recall"] += .001
    config["evaluation_targets"]["fa_per_hour"] -= .001
    config["evaluation_targets"]["latency_p95_s"] -= .001
    config["targets"]["development_negative_hours"][0] += 1
    result = adoption([summary], config)
    assert not result["passed"]
    assert len(result["reasons"]) == 10  # Clean, six category/SNR gates, hours, FA, latency.
    assert summary == original


@pytest.mark.parametrize("key", ["clean_recall", "snr10_recall", "snr5_recall", "fa_per_hour", "latency_p95_s"])
def test_missing_targets_are_not_replaced_by_unrelated_library_defaults(key):
    config, _ = _inputs()
    del config["evaluation_targets"][key]
    with pytest.raises(KeyError):
        adoption_criteria(config)


def test_missing_duration_gate_is_not_invented():
    config, _ = _inputs()
    del config["targets"]["development_negative_hours"]
    with pytest.raises(KeyError):
        adoption_criteria(config)
