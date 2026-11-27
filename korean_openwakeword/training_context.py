"""The waveform and feature window shared by training and detection."""
import math

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280
DEFAULT_FEATURE_FRAMES = 16
CONTEXT_LEAD_FRAMES = 9
MAX_FEATURE_FRAMES = 93


def feature_frames(runtime):
    frames = runtime.get("feature_frames", DEFAULT_FEATURE_FRAMES)
    if type(frames) is not int or not DEFAULT_FEATURE_FRAMES <= frames <= MAX_FEATURE_FRAMES:
        raise ValueError("Unsupported wake-word feature context")
    return frames


def clip_samples(frames):
    return (frames + CONTEXT_LEAD_FRAMES) * FRAME_SAMPLES


def clip_seconds(frames):
    return clip_samples(frames) / SAMPLE_RATE


def max_call_seconds(frames):
    return (clip_samples(frames) - 2 * FRAME_SAMPLES) / SAMPLE_RATE


def frames_for_call(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 8:
        raise ValueError("호출어 발음 길이는 0초 초과, 8초 이하로 입력하세요.")
    hops = math.ceil(seconds * SAMPLE_RATE / FRAME_SAMPLES - 1e-10) + 2
    return max(DEFAULT_FEATURE_FRAMES, hops - CONTEXT_LEAD_FRAMES)


def pronunciation_limits(config):
    frames = feature_frames(config.get("runtime", {}))
    return {} if frames == DEFAULT_FEATURE_FRAMES else {"max_length": max_call_seconds(frames)}
