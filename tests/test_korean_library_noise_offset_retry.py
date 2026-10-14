"""Non-silent offset retry preserves valid-window PCM and sampling draws.

The reference below freezes augment before this fix (features.py SHA-256
91dd9e69f7a2b635d23add330ab3ee269fc540355a6b2bd9bd79fd542dfbb40b).
It is intentionally independent of the new offset selection helper.
"""
import hashlib

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

from korean_openwakeword import features
from korean_openwakeword.audio import active_speech_mask, measured_snr, noise_scale, pcm16, rms
from korean_openwakeword.common import digest
from korean_openwakeword.data import source_key
from korean_openwakeword.project import default_profile


def legacy_augment(root, row, role, round_id, settings, noises, rirs):
    seed = features.seed_for(settings["seed"], role, row["sample_id"], round_id)
    rng = np.random.default_rng(seed)
    clip = np.zeros(32000, dtype=np.float64)
    if role == "positive":
        a, b = row["call_start_s"], row["call_end_s"]
        audio, _ = features.seek_audio(root, {**row, "start_s": a, "end_s": b}, b - a)
        if len(audio) > 29440:
            raise ValueError("Complete word must fit a 2s clip with 80ms head/tail; quarantine, do not cut syllables")
        start = int(rng.integers(1280, 32000 - len(audio) - 1280 + 1))
        clip[start:start + len(audio)] = audio
        bounds = [start / 16000, (start + len(audio)) / 16000]
        mask = active_speech_mask(clip, *bounds)
    else:
        audio, _ = features.seek_audio(root, row, 2)
        clip[:min(len(audio), 32000)] = audio[:32000]
        bounds = None
        mask = active_speech_mask(clip, 0, min(len(audio), 32000) / 16000) if row.get("speech") and rms(clip) > 1e-8 else None
    details = {"sample_id": row["sample_id"], "source_id": source_key(row), "round_id": round_id,
               "seed": seed, "call_bounds_in_clip": bounds, "noise_id": None, "rir_id": None,
               "snr_db": None, "measured_snr_db": None}
    if rirs and rng.random() < settings["rir_probability"]:
        rir = rirs[int(rng.integers(len(rirs)))]
        impulse, _ = features.seek_audio(root, rir, 2)
        norm = np.sqrt(np.sum(impulse ** 2))
        if norm <= 1e-8:
            raise ValueError("Silent RIR")
        clip = fftconvolve(clip, impulse / norm)[:32000]
        details["rir_id"] = source_key(rir)
    draw = rng.random()
    if noises and draw >= settings["clean_probability"]:
        categories = list(settings["category_weights"])
        available = features.prepare_noise_index(settings, noises)
        if any(not group for group, _ in available.values()):
            raise ValueError("All declared noise categories need admitted material; do not silently renormalize")
        weights = np.asarray([settings["category_weights"][k] for k in categories], float)
        category = categories[int(rng.choice(len(categories), p=weights / weights.sum()))]
        group, probabilities = available[category]
        noise = group[int(rng.choice(len(group), p=probabilities))]
        samples, offset = features.seek_audio(root, noise, 2, rng)
        if len(samples) < 32000:
            samples = np.tile(samples, int(np.ceil(32000 / len(samples))))
        samples = samples[:32000]
        details.update(noise_id=source_key(noise), noise_category=category, noise_offset_s=offset)
        if mask is not None:
            key = "moderate_snr_db" if draw < settings["clean_probability"] + settings["moderate_probability"] else "hard_snr_db"
            snr = float(rng.uniform(*settings[key]))
            scaled = samples * noise_scale(clip, samples, mask, snr)
            details.update(snr_db=snr, measured_snr_db=measured_snr(clip, scaled, mask))
        else:
            scaled = samples * 10 ** (float(rng.uniform(-12, 0)) / 20)
            details["noise_gain_only"] = True
        clip += scaled
    gain = float(rng.uniform(*settings["gain_db"]))
    clip *= 10 ** (gain / 20)
    peak = float(np.max(np.abs(clip)))
    headroom = min(1.0, settings["peak_limit"] / peak) if peak else 1.0
    clip *= headroom
    samples = pcm16(clip)
    details.update(gain_db=gain, headroom_gain=headroom,
                   pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest())
    return samples, details


@pytest.fixture
def inputs(tmp_path):
    rate = 16000

    def source(name, audio, **values):
        path = tmp_path / (name + ".wav")
        sf.write(path, audio, rate, subtype="PCM_16")
        return {"sample_id": name, "source_id": "fixture:" + name, "path": name + ".wav",
                "sha256": digest(path), **values}

    positive = source("call", .12 * np.sin(np.arange(8000) * .051),
                      speech=True, call_start_s=0., call_end_s=.5)
    negative = source("negative", .10 * np.sin(np.arange(32000) * .047), speech=True)
    noises = [source("noise_" + category, .02 * np.sin(np.arange(80000) * (.023 + number * .007)),
                     category=category, language="ko" if category == "competing_speech" else None)
              for number, category in enumerate(("environment", "music", "competing_speech"))]
    short = source("noise_short", .02 * np.sin(np.arange(1600) * .081), category="environment")
    impulse = np.zeros(32000); impulse[0] = .8; impulse[720] = .15; impulse[2000] = .07
    rir = source("rir", impulse, category="rir")
    settings = default_profile()["augmentation"]
    return tmp_path, positive, negative, noises, short, [rir], settings


@pytest.mark.parametrize("role", ["positive", "korean_negative", "hard_negative", "generic_negative"])
@pytest.mark.parametrize("short_file", [False, True])
@pytest.mark.parametrize("with_rir", [False, True])
def test_first_valid_window_is_bitwise_identical_to_frozen_baseline(inputs, role, short_file, with_rir):
    root, positive, negative, noises, short, rirs, settings = inputs
    row = positive if role == "positive" else {**negative, "speech": role != "generic_negative"}
    if short_file:
        noises = [short]
        settings = {**settings, "category_weights": {"environment": 1}}
    settings = {**settings, "rir_probability": 1. if with_rir else 0.}
    for round_id in range(5):
        before, original = legacy_augment(root, row, role, round_id, settings, noises, rirs)
        after, fixed = features.augment(root, row, role, round_id, settings, noises, rirs)
        assert np.array_equal(before, after)
        assert all(fixed[key] == value for key, value in original.items())
        if fixed["noise_id"] is not None:
            assert fixed["noise_offset_attempts"] == 1
            assert fixed["noise_first_offset_exclusion_reason"] is None


@pytest.mark.parametrize("role,first_window", [
    ("positive", "all_silent"), ("positive", "silent_during_call"),
    ("generic_negative", "all_silent"),
])
def test_retry_same_source_preserves_snr_gain_rir_and_deterministic_pcm(inputs, monkeypatch, role, first_window):
    root, positive, negative, noises, _, rirs, settings = inputs
    row = positive if role == "positive" else {**negative, "speech": False}
    settings = {**settings, "clean_probability": 0., "moderate_probability": 1., "rir_probability": 1.}
    _, baseline = legacy_augment(root, row, role, 0, settings, noises, rirs)
    original_seek = features.seek_audio
    reads = []

    def quiet_first(root, source, seconds=2, rng=None):
        samples, offset = original_seek(root, source, seconds, rng)
        if source["sample_id"].startswith("noise_"):
            reads.append((source_key(source), offset))
            if len(reads) == 1:
                samples = np.zeros_like(samples)
                if first_window == "silent_during_call":
                    samples[:200] = .1  # audible outside the call, zero in its active region
        return samples, offset

    monkeypatch.setattr(features, "seek_audio", quiet_first)
    first, details = features.augment(root, row, role, 0, settings, noises, rirs)
    assert details["noise_offset_attempts"] == len(reads) == 2
    assert len({source for source, _ in reads}) == 1
    assert details["noise_first_offset_s"] == baseline["noise_offset_s"]
    assert details["noise_first_offset_exclusion_reason"] == (
        "silent_in_clean_active_region" if role == "positive" else "silent_in_whole_2s_clip")
    for key in ("seed", "snr_db", "gain_db", "rir_id", "noise_id", "noise_category"):
        assert details[key] == baseline[key]
    if role == "positive":
        assert abs(details["measured_snr_db"] - details["snr_db"]) < 1e-10
    else:
        assert details["snr_db"] is None and details["noise_gain_only"] is True
    initial_reads = reads[:]
    reads.clear()
    repeated, repeated_details = features.augment(root, row, role, 0, settings, noises, rirs)
    assert np.array_equal(first, repeated)
    assert details == repeated_details and reads == initial_reads


def test_whole_clip_energy_does_not_replace_active_region_check(inputs, monkeypatch):
    root, positive, _, noises, _, _, settings = inputs
    settings = {**settings, "clean_probability": 0., "moderate_probability": 1.}
    original_seek = features.seek_audio
    counter = 0

    def speech_region_silent(root, source, seconds=2, rng=None):
        nonlocal counter
        samples, offset = original_seek(root, source, seconds, rng)
        if source["sample_id"].startswith("noise_"):
            counter += 1
            samples = np.zeros_like(samples); samples[:200] = .1
            assert rms(samples) > features.MIN_NOISE_RMS
        return samples, offset

    monkeypatch.setattr(features, "seek_audio", speech_region_silent)
    with pytest.raises(ValueError, match=r"source fixture:noise_.*scope=clean_active_region.*after 64 offsets"):
        features.augment(root, positive, "positive", 0, settings, noises, [])
    assert counter == features.MAX_NOISE_OFFSET_ATTEMPTS


@pytest.mark.parametrize("role", ["positive", "generic_negative"])
def test_exhausted_silent_source_raises_with_context_without_clean_fallback(inputs, monkeypatch, role):
    root, positive, negative, noises, _, _, settings = inputs
    row = positive if role == "positive" else {**negative, "speech": False}
    settings = {**settings, "clean_probability": 0., "moderate_probability": 1.}
    original_seek = features.seek_audio
    reads = []

    def silent_noise(root, source, seconds=2, rng=None):
        samples, offset = original_seek(root, source, seconds, rng)
        if source["sample_id"].startswith("noise_"):
            reads.append(source_key(source)); samples = np.zeros_like(samples)
        return samples, offset

    monkeypatch.setattr(features, "seek_audio", silent_noise)
    with pytest.raises(ValueError, match=r"source fixture:noise_.*category=.*after 64 offsets.*clean fallback is disabled"):
        features.augment(root, row, role, 0, settings, noises, [])
    assert len(reads) == 64 and len(set(reads)) == 1


def test_unrelated_io_error_is_not_swallowed_or_retried(inputs, monkeypatch):
    root, positive, _, noises, _, _, settings = inputs
    settings = {**settings, "clean_probability": 0., "moderate_probability": 1.}
    original_seek = features.seek_audio
    counter = 0

    def broken_second_read(root, source, seconds=2, rng=None):
        nonlocal counter
        if source["sample_id"].startswith("noise_"):
            counter += 1
            if counter == 2:
                raise OSError("fixture read failure must propagate")
        samples, offset = original_seek(root, source, seconds, rng)
        if source["sample_id"].startswith("noise_"):
            samples = np.zeros_like(samples)
        return samples, offset

    monkeypatch.setattr(features, "seek_audio", broken_second_read)
    with pytest.raises(OSError, match="read failure must propagate"):
        features.augment(root, positive, "positive", 0, settings, noises, [])
    assert counter == 2
