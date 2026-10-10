"""Preserved deterministic audio utilities from this project (2026-10-09)."""
from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read, downmix and resample without normalizing each recording."""
    samples, original_rate = sf.read(path, dtype="float64", always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    mono = samples.mean(axis=1)
    if original_rate != sample_rate:
        divisor = gcd(original_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, original_rate // divisor)
    return mono.astype(np.float64, copy=False)


def pcm16(samples: np.ndarray) -> np.ndarray:
    """Round explicitly so persisted PCM and inference input have identical scale."""
    samples = np.asarray(samples)
    if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0) > 1:
        raise ValueError("PCM conversion refuses nonfinite or clipping input")
    return np.clip(np.rint(samples * 32768), -32768, 32767).astype(np.int16)


def write_pcm16(path: str | Path, samples: np.ndarray) -> None:
    sf.write(path, pcm16(samples), SAMPLE_RATE, subtype="PCM_16")


def rms(samples: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("RMS mask shape differs from samples")
        values = values[mask]
    if values.size == 0:
        raise ValueError("Cannot compute RMS of an empty active region")
    return float(np.sqrt(np.mean(np.square(values))))


def intervals_mask(length: int, intervals_s: list[list[float]],
                   sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start_s, end_s in intervals_s:
        start, end = round(start_s * sample_rate), round(end_s * sample_rate)
        if not 0 <= start < end <= length:
            raise ValueError(f"Active interval [{start_s}, {end_s}] exceeds audio")
        mask[start:end] = True
    if not mask.any():
        raise ValueError("At least one nonempty active interval is required")
    return mask


def active_speech_mask(samples: np.ndarray, call_start_s: float,
                       call_end_s: float, relative_db: float = -40.0) -> np.ndarray:
    """20-ms RMS mask inside explicit call bounds; computed on clean speech only.

    This is a documented energy heuristic, not a semantic VAD. Explicit manual
    intervals are preferable when silence/quiet syllables need closer review.
    """
    bounds = intervals_mask(len(samples), [[call_start_s, call_end_s]])
    start, end = np.flatnonzero(bounds)[[0, -1]]
    end += 1
    hop = round(0.02 * SAMPLE_RATE)
    windows = [(i, min(i + hop, end)) for i in range(start, end, hop)]
    levels = [rms(samples[i:j]) for i, j in windows]
    maximum = max(levels)
    if maximum <= 1e-8:
        raise ValueError("Call bounds contain silence")
    minimum = maximum * 10 ** (relative_db / 20)
    mask = np.zeros(len(samples), dtype=bool)
    for (i, j), level in zip(windows, levels):
        if level >= minimum:
            mask[i:j] = True
    return mask


def mask_intervals(mask: np.ndarray) -> list[list[float]]:
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return [[int(a) / SAMPLE_RATE, int(b) / SAMPLE_RATE]
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def noise_scale(speech: np.ndarray, noise: np.ndarray, mask: np.ndarray,
                snr_db: float) -> float:
    speech_rms, noise_rms = rms(speech, mask), rms(noise, mask)
    if speech_rms <= 1e-8 or noise_rms <= 1e-8:
        raise ValueError("Speech and noise must be non-silent in the clean active region")
    return speech_rms / (noise_rms * 10 ** (snr_db / 20))


def measured_snr(speech: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> float:
    return float(20 * np.log10(rms(speech, mask) / rms(noise, mask)))


def common_headroom_gain(speech: np.ndarray, scaled_noises: list[np.ndarray],
                         peak_limit: float = 0.98) -> float:
    """One gain for an utterance's clean, EVERY mix and noise-only control."""
    if not 0 < peak_limit < 1:
        raise ValueError("peak_limit must be strictly between zero and one")
    maximum = float(np.max(np.abs(speech), initial=0))
    for noise in scaled_noises:
        if noise.shape != speech.shape:
            raise ValueError("All conditions must use the same clip length")
        maximum = max(maximum, float(np.max(np.abs(speech + noise), initial=0)),
                      float(np.max(np.abs(noise), initial=0)))
    return min(1.0, peak_limit / maximum) if maximum else 1.0
