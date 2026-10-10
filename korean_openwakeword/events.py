"""Preserved event/matching/statistics policies from the verified experiment."""
import math
from collections import defaultdict
import numpy as np
from scipy.stats import chi2
FRAME_SAMPLES=1280
SAMPLE_RATE=16000

def poisson_rate_interval(events: int, hours: float, confidence: float = 0.95) -> dict:
    """Exact two-sided Garwood interval, conditional on a homogeneous Poisson model."""
    if not isinstance(events, int) or events < 0 or not math.isfinite(hours) or hours < 0:
        raise ValueError("Poisson counts must be nonnegative integers and hours finite/nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0,1)")
    if hours == 0:
        if events:
            raise ValueError("Events cannot have zero observation time")
        return dict(events=0, hours=0, fa_per_hour=None, ci95_lower=None, ci95_upper=None)
    alpha = 1 - confidence
    lower = 0.0 if events == 0 else float(chi2.ppf(alpha / 2, 2 * events) / (2 * hours))
    upper = float(chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / (2 * hours))
    return dict(events=events, hours=hours, fa_per_hour=events / hours,
                ci95_lower=lower, ci95_upper=upper)

class EventDetector:
    """A candidate is a rising edge of a qualified consecutive-frame streak.

    Cooldown suppresses candidates; a sustained high score does not generate a
    new event when cooldown expires. Warmup consumes score/streak state but
    suppresses events and does not start cooldown. The first eligible frame is
    a new eligible edge: if already qualified, it emits without resetting any
    model or score history. Eligible high/qualified occupancy is also measured,
    so a stuck-high output cannot look good solely through sparse event counts.
    """
    def __init__(self, threshold: float, consecutive_frames: int, cooldown_s: float):
        self.threshold, self.required, self.cooldown_s = threshold, consecutive_frames, cooldown_s
        self.streak, self.was_qualified, self.last_event_s = 0, False, -float("inf")
        self.was_eligible = False
        self.eligible_frames = self.high_frames = self.qualified_frames = 0
        self.high_streak = self.qualified_streak = 0
        self.longest_high_streak = self.longest_qualified_streak = 0

    def update(self, score: float, timestamp_s: float, eligible: bool) -> dict:
        above = score >= self.threshold
        self.streak = self.streak + 1 if above else 0
        qualified = self.streak >= self.required
        eligibility_edge = eligible and not self.was_eligible
        candidate = qualified and (not self.was_qualified or eligibility_edge)
        if eligible:
            self.eligible_frames += 1
            self.high_frames += int(above)
            self.qualified_frames += int(qualified)
        self.high_streak = self.high_streak + 1 if eligible and above else 0
        self.qualified_streak = self.qualified_streak + 1 if eligible and qualified else 0
        self.longest_high_streak = max(self.longest_high_streak, self.high_streak)
        self.longest_qualified_streak = max(self.longest_qualified_streak, self.qualified_streak)
        suppression = None
        if candidate:
            if not eligible:
                suppression = "warmup"
            elif timestamp_s - self.last_event_s + 1e-9 < self.cooldown_s:
                suppression = "cooldown"
        emitted = candidate and suppression is None
        if emitted:
            self.last_event_s = timestamp_s
        self.was_qualified = qualified
        self.was_eligible = eligible
        return dict(event=emitted, candidate=candidate, suppression=suppression,
                    above_threshold=above, qualified=qualified,
                    eligibility_edge=eligibility_edge, consecutive_count=self.streak)

    def activity_statistics(self) -> dict:
        seconds_per_frame = FRAME_SAMPLES / SAMPLE_RATE
        return dict(eligible_frames=self.eligible_frames,
                    eligible_above_threshold_frames=self.high_frames,
                    eligible_above_threshold_s=self.high_frames * seconds_per_frame,
                    eligible_above_threshold_fraction=self.high_frames / self.eligible_frames if self.eligible_frames else None,
                    eligible_qualified_s=self.qualified_frames * seconds_per_frame,
                    eligible_qualified_fraction=self.qualified_frames / self.eligible_frames if self.eligible_frames else None,
                    longest_eligible_high_streak_s=self.longest_high_streak * seconds_per_frame,
                    longest_eligible_qualified_streak_s=self.longest_qualified_streak * seconds_per_frame)

class PositiveTracker:
    """One emitted event matches at most one call, earliest unmatched eligible call."""
    def __init__(self, session: dict, postroll_s: float):
        self.rows = []
        for call in session.get("calls", []):
            self.rows.append(dict(call, session_id=session["session_id"], source_id=session["source_id"],
                                 provenance=session["provenance"],
                                 speaker_id=call.get("speaker_id", session.get("speaker_id")),
                                 voice_family_id=session.get("voice_family_id"),
                                 category=call.get("category", session["category"]),
                                 microphone=call.get("microphone", session.get("microphone", "unspecified")),
                                 distance_m=call.get("distance_m", session.get("distance_m")),
                                 direction=call.get("direction", session.get("direction", "unspecified")),
                                 window_end_s=call["end_s"] + postroll_s,
                                 max_wake_score=None, max_wake_score_time_s=None,
                                 success=False, event_time_s=None, latency_s=None,
                                 suppressed_cooldown_candidates=0, ambiguous_event_assignment=False))
        self.active, self.next_index = [], 0
        self.duplicate_events, self.outside_call_events = 0, 0

    def update(self, timestamp_s: float, score: float, decision: dict) -> None:
        while self.next_index < len(self.rows) and self.rows[self.next_index]["start_s"] <= timestamp_s + 1e-9:
            self.active.append(self.rows[self.next_index])
            self.next_index += 1
        self.active = [row for row in self.active if timestamp_s <= row["window_end_s"] + 1e-9]
        for row in self.active:
            if row["max_wake_score"] is None or score > row["max_wake_score"]:
                row["max_wake_score"], row["max_wake_score_time_s"] = score, timestamp_s
            if decision["suppression"] == "cooldown":
                row["suppressed_cooldown_candidates"] += 1
        if not decision["event"]:
            return
        available = [row for row in self.active if not row["success"]]
        if available:
            selected = available[0]
            selected.update(success=True, event_time_s=timestamp_s,
                            latency_s=timestamp_s - selected["end_s"],
                            ambiguous_event_assignment=len(available) > 1)
        elif self.active:
            self.duplicate_events += 1
        else:
            self.outside_call_events += 1

def positive_statistics(rows: list[dict], seed: int = 20260928) -> dict:
    if not rows:
        return dict(calls=0, successes=0, recall=None, independent_units=0, ci95=None)
    units = defaultdict(list)
    for row in rows:
        unit = row["speaker_id"] if row["provenance"] == "human" else row["original_utterance_id"]
        units[unit].append(int(row["success"]))
    counts = np.array([[sum(values), len(values)] for values in units.values()], dtype=float)
    ci = None
    if len(units) >= 2:
        rng = np.random.default_rng(seed)
        estimates = []
        # Cluster bootstrap keeps the same speaker/original's variants together.
        for _ in range(2000):
            sampled = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
            estimates.append(sampled[0] / sampled[1])
        ci = [float(value) for value in np.quantile(estimates, [0.025, 0.975])]
    latencies = [row["latency_s"] for row in rows if row["success"]]
    return dict(calls=len(rows), successes=sum(row["success"] for row in rows),
                recall=sum(row["success"] for row in rows) / len(rows),
                independent_units=len(units),
                uncertainty_unit="speaker" if rows[0]["provenance"] == "human" else "original_utterance",
                ci95=ci, ci_method="2000-replicate cluster percentile bootstrap, fixed seed",
                small_sample_warning=len(units) < 20,
                homogeneous_outcomes_warning=len({row["success"] for row in rows}) < 2,
                latency_p50_s=float(np.quantile(latencies, 0.5)) if latencies else None,
                latency_p95_s=float(np.quantile(latencies, 0.95)) if latencies else None,
                missed_calls=sum(not row["success"] for row in rows))

def positive_report(rows: list[dict]) -> dict:
    report = {}
    for provenance in ("qwen_tts", "human"):
        selected = [row for row in rows if row["provenance"] == provenance]
        groups = {}
        for dimension in ("category", "speaker_id", "distance_m", "direction", "microphone"):
            values = defaultdict(list)
            for row in selected:
                values[str(row.get(dimension, "unspecified"))].append(row)
            groups[dimension] = {key: positive_statistics(group) for key, group in values.items()}
        report[provenance] = dict(overall=positive_statistics(selected), by=groups)
    return report
