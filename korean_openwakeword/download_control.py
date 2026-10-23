"""Small, bounded whole-file admission controller; it never interrupts a file."""
import math


class AdaptiveConcurrency:
    """Probe aggregate fresh-transfer goodput, ignoring cache/hash-only intervals.

    Counters passed to ``update`` are cumulative for this batch. ``network_active``
    is the current number of transfers after the event; the preceding value owns
    the elapsed interval. Limits govern future admission, not already active work.
    This is a conservative online heuristic, not a promise of a global optimum.
    """

    def __init__(self, ceiling=16, initial=3, window_seconds=3.0, cooldown_seconds=15.0):
        if type(ceiling) is not int or ceiling < 1 or type(initial) is not int or initial < 1:
            raise ValueError("Download concurrency limits must be positive integers")
        if (not math.isfinite(window_seconds) or window_seconds <= 0
                or not math.isfinite(cooldown_seconds) or cooldown_seconds < 0):
            raise ValueError("Download observation window and cooldown must be finite and nonnegative")
        self.ceiling, self.limit = ceiling, min(initial, ceiling)
        self.window_seconds, self.cooldown_seconds = window_seconds, cooldown_seconds
        self.reason = "initial"
        self._previous = None
        self._cooldown_until = -math.inf
        self._probe = self._reference_rate = None
        self._reset_window()

    def _reset_window(self):
        self._seconds = self._load_seconds = self._bytes = 0.0
        self._read = self._write = 0.0
        self._over_limit = False

    def _backoff(self, now, reason, limit=None):
        self.limit = max(1, self.limit // 2) if limit is None else limit
        self.reason = reason
        self._cooldown_until = now + self.cooldown_seconds
        self._probe = self._reference_rate = None
        self._reset_window()

    def update(self, now, network_bytes, *, network_active, write_seconds=0.0,
               read_seconds=0.0, retry=False, backlog=True):
        """Observe producer timestamps and newly received bytes, never file offsets."""
        counters = (network_bytes, read_seconds, write_seconds)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in counters)
                or isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now)
                or type(network_active) is not int or network_active < 0):
            raise ValueError("Download observations must contain finite nonnegative counters")
        previous = self._previous
        # Events from different workers can reach the queue in a different order.
        # Do not invent negative elapsed time for their producer timestamps.
        if previous is not None:
            now = max(now, previous[0])
            if any(value < old for value, old in zip(counters, previous[2:])):
                raise ValueError("Download observation counters must be cumulative")
        self._previous = (now, network_active, *counters)
        if retry:
            self._backoff(now, "retry")
            return
        if previous is None:
            return
        if now < self._cooldown_until or previous[0] < self._cooldown_until:
            self._reset_window()
            return
        elapsed = now - previous[0]
        if previous[1] > 0:
            self._seconds += elapsed
            self._load_seconds += elapsed * previous[1]
            self._over_limit |= previous[1] > self.limit
        self._bytes += network_bytes - previous[2]
        self._read += read_seconds - previous[3]
        self._write += write_seconds - previous[4]
        if self._seconds < self.window_seconds or self._bytes < 16 * 1024:
            return
        rate = self._bytes / self._seconds
        io_seconds = self._read + self._write
        disk_pressure = io_seconds and self._write >= .05 and self._write / io_seconds > .25
        loaded = self._load_seconds / self._seconds >= .75 * self.limit
        over_limit = self._over_limit
        self._reset_window()
        if over_limit:
            return  # Previously admitted files must drain before judging a lower limit.
        if disk_pressure:
            self._backoff(now, "disk_pressure")
            return
        if not backlog or not loaded:
            return
        if self._probe is not None:
            old_limit, old_rate = self._probe
            if rate < old_rate * 1.10:
                self._backoff(now, "no_gain", old_limit)
                return
            self._probe = None
            self._reference_rate = rate
            self.reason = "gain"
        elif self._reference_rate is not None and rate < self._reference_rate * .80:
            self._backoff(now, "throughput_drop")
            return
        self._reference_rate = rate
        if self.limit < self.ceiling:
            self._probe = (self.limit, rate)
            self.limit += 1
            self.reason = "probe"
