"""Deterministic controller tests model capacity without transferring public data."""
import math

import pytest

from korean_openwakeword.download_control import AdaptiveConcurrency


class Transfer:
    def __init__(self, controller, active=None):
        self.controller, self.now, self.bytes = controller, 0.0, 0
        self.read = self.write = 0.0
        controller.update(0, 0, network_active=controller.limit if active is None else active)

    def step(self, seconds=3, rate=100_000, *, active=None, read=1, write=.01, **kwargs):
        self.now += seconds
        self.bytes += int(seconds * rate)
        self.read += read
        self.write += write
        self.controller.update(self.now, self.bytes,
            network_active=self.controller.limit if active is None else active,
            read_seconds=self.read, write_seconds=self.write, **kwargs)


def test_capacity_growth_is_probed_and_capped():
    control = AdaptiveConcurrency(ceiling=6)
    transfer = Transfer(control)
    for _ in range(8):
        transfer.step(rate=100_000 * control.limit)
        assert 1 <= control.limit <= 6
    assert control.limit == 6


def test_fixed_bandwidth_rolls_back_and_waits_before_reprobing():
    control = AdaptiveConcurrency(cooldown_seconds=15)
    transfer = Transfer(control)
    transfer.step()
    assert control.limit == 4 and control.reason == "probe"
    transfer.step()
    assert control.limit == 3 and control.reason == "no_gain"
    for _ in range(5):
        transfer.step()
        assert control.limit == 3
    transfer.step()
    assert control.limit == 4  # Changed conditions can be tested after cooldown.


def test_a_probe_that_loses_throughput_returns_to_prior_limit():
    control = AdaptiveConcurrency()
    transfer = Transfer(control)
    transfer.step(rate=1_000_000)
    transfer.step(rate=400_000)
    assert control.limit == 3 and control.reason == "no_gain"


def test_retry_immediately_halves_admission_and_does_not_require_a_sample():
    control = AdaptiveConcurrency()
    transfer = Transfer(control)
    transfer.step()
    assert control.limit == 4
    transfer.step(seconds=.1, retry=True)
    assert control.limit == 2 and control.reason == "retry"
    transfer.step(seconds=.1, retry=True)
    assert control.limit == 1
    transfer.step(seconds=.1, retry=True)
    assert control.limit == 1


def test_slow_disk_reduces_admission_without_changing_any_transfer():
    control = AdaptiveConcurrency()
    transfer = Transfer(control)
    transfer.step(read=1, write=1)
    assert control.limit == 1 and control.reason == "disk_pressure"


def test_transfer_capacity_can_drop_after_reaching_ceiling():
    control = AdaptiveConcurrency(ceiling=3)
    transfer = Transfer(control)
    transfer.step(rate=1_000_000)
    transfer.step(rate=500_000)
    assert control.limit == 1 and control.reason == "throughput_drop"


def test_hash_and_cache_only_gaps_are_not_network_time_or_bytes():
    control = AdaptiveConcurrency()
    control.update(0, 0, network_active=3)
    control.update(1, 100_000, network_active=0, read_seconds=1)
    control.update(10_001, 100_000, network_active=3, read_seconds=1)
    control.update(10_003, 300_000, network_active=3, read_seconds=3)
    assert control.limit == 4
    assert control._probe[1] == pytest.approx(100_000)


def test_no_fresh_network_bytes_never_probes_even_if_workers_are_busy():
    control = AdaptiveConcurrency()
    control.update(0, 0, network_active=3)
    control.update(100, 0, network_active=0)
    control.update(1000, 0, network_active=0)
    assert control.limit == 3 and control.reason == "initial"


def test_unsaturated_hash_occupied_pool_does_not_probe():
    control = AdaptiveConcurrency()
    transfer = Transfer(control, active=1)
    transfer.step(active=1)
    assert control.limit == 3


def test_tiny_files_can_accumulate_enough_evidence_across_waves():
    control = AdaptiveConcurrency()
    transfer = Transfer(control)
    transfer.step(seconds=1, rate=8_000)
    transfer.step(seconds=1, rate=8_000)
    assert control.limit == 3
    transfer.step(seconds=1, rate=8_000)
    assert control.limit == 4


def test_empty_queue_does_not_start_a_useless_probe():
    control = AdaptiveConcurrency()
    transfer = Transfer(control)
    transfer.step(backlog=False)
    assert control.limit == 3


def test_out_of_order_producer_time_is_clamped_without_negative_time():
    control = AdaptiveConcurrency()
    control.update(10, 0, network_active=3)
    control.update(9, 1000, network_active=3)
    control.update(13, 301_000, network_active=3)
    assert control.limit == 4 and control._probe[1] == pytest.approx(301_000 / 3)


@pytest.mark.parametrize("kwargs", [dict(ceiling=0), dict(ceiling=True), dict(initial=0),
    dict(window_seconds=0), dict(window_seconds=math.inf), dict(cooldown_seconds=-1)])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AdaptiveConcurrency(**kwargs)


def test_a_small_ceiling_clamps_default_initial_limit():
    assert AdaptiveConcurrency(ceiling=1).limit == 1


@pytest.mark.parametrize("kwargs", [dict(now=math.nan), dict(network_bytes=-1),
    dict(network_bytes=True), dict(network_active=-1), dict(read_seconds=math.inf)])
def test_invalid_observations_are_rejected(kwargs):
    observation = dict(now=0, network_bytes=0, network_active=3)
    observation.update(kwargs)
    with pytest.raises(ValueError):
        AdaptiveConcurrency().update(**observation)


def test_counter_reset_is_rejected_instead_of_becoming_fake_network_rate():
    control = AdaptiveConcurrency()
    control.update(0, 1000, network_active=3)
    with pytest.raises(ValueError, match="cumulative"):
        control.update(1, 0, network_active=3)


def test_pre_backoff_transfers_cannot_be_mistaken_for_the_lower_limit():
    control = AdaptiveConcurrency(initial=8, cooldown_seconds=3)
    transfer = Transfer(control)
    transfer.step(seconds=.1, retry=True, active=8)
    assert control.limit == 4
    for _ in range(5):
        transfer.step(active=8, rate=8_000_000)
        assert control.limit == 4
    transfer.step(active=4)
    transfer.step(active=4)
    assert control.limit == 5


def test_pre_backoff_disk_pressure_cannot_repeatedly_halve_an_inactive_target():
    control = AdaptiveConcurrency(initial=8, cooldown_seconds=0)
    transfer = Transfer(control)
    transfer.step(retry=True, active=8)
    assert control.limit == 4
    for _ in range(5):
        transfer.step(active=8, read=1, write=1)
    assert control.limit == 4


def test_a_preexisting_counter_baseline_does_not_become_fresh_network_bytes():
    control = AdaptiveConcurrency()
    control.update(0, 10**12, network_active=3)
    control.update(3, 10**12 + 300_000, network_active=3, read_seconds=3)
    assert control._probe[1] == pytest.approx(100_000)
