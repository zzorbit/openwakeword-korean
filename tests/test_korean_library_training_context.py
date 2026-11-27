import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest
from korean_openwakeword.features import augment
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.training_context import clip_samples, feature_frames, frames_for_call, pronunciation_limits
from korean_openwakeword.tts import encode_candidate_pcm16, technical_qc


def test_complete_three_second_call_and_continuous_background(tmp_path):
    project = create_project(tmp_path / "model", "보라야보라야", "long", call_duration_s=3)
    root, config = load_project(project)
    frames = feature_frames(config["runtime"])
    assert frames == 31
    assert config["runtime"]["warmup_s"] == 3.2
    size = clip_samples(frames)
    speech = .1 * np.sin(np.arange(48000) * .05)
    qc, converted = technical_qc(speech, 16000, **pronunciation_limits(config))
    assert qc["passed"]
    assert not technical_qc(speech, 16000)[0]["passed"]
    pcm, bounds = encode_candidate_pcm16(converted, qc, **pronunciation_limits(config))
    assert bounds == [0, 3]
    call_path = root / "call.wav"
    noise_path = root / "noise.wav"
    sf.write(call_path, pcm, 16000)
    sf.write(noise_path, .02 * np.sin(np.arange(160000) * .04), 16000)
    call = {"sample_id": "call", "sha256": digest(call_path), "path": call_path.name,
            "call_start_s": 0., "call_end_s": 3.}
    noise = {"sample_id": "noise", "sha256": digest(noise_path), "path": noise_path.name,
             "category": "environment"}
    settings = {**config["augmentation"], "clean_probability": 0., "category_weights": {"environment": 1}}
    with pytest.raises(ValueError, match="Complete word must fit a 2s clip"):
        augment(root, call, "positive", 0, settings, [noise], [])
    mixed, details = augment(root, call, "positive", 0, settings, [noise], [], size=size)
    start, end = [round(time * 16000) for time in details["call_bounds_in_clip"]]
    assert len(mixed) == 51200 and end - start == 48000
    assert start >= 1280 and size - end >= 1280
    assert np.any(mixed[:start]) and np.any(mixed[end:])
    assert 0 <= details["noise_offset_s"] <= 6.8
    repeat, _ = augment(root, call, "positive", 0, settings, [noise], [], size=size)
    np.testing.assert_array_equal(repeat, mixed)


@pytest.mark.parametrize(("requested", "frames"), [(1.84, 16), (1.85, 17), (3, 31), (8, 93)])
def test_window_length_uses_complete_eighty_millisecond_frames(requested, frames):
    assert frames_for_call(requested) == frames


def test_mixed_context_trains_shared_classifier_without_padding_publisher_rows():
    torch = pytest.importorskip("torch")
    from korean_openwakeword.train import InputDelivery, build_classifier

    pools = {"generic_negative": np.zeros((2, 16, 96), np.float16),
             "positive": np.ones((2, 31, 96), np.float32),
             "hard_negative": np.zeros((2, 31, 96), np.float32)}
    delivery = InputDelivery(pools, "cpu", batch={role: 2 for role in pools})
    batch, labels = delivery.batch({role: np.arange(2) for role in pools})
    assert [x.shape for x in batch] == [(2, 16, 96), (4, 31, 96)]
    assert labels.flatten().tolist() == [0, 0, 1, 1, 0, 0]
    model = build_classifier("cnn64", 0.)
    logits = torch.cat([model(x) for x in batch])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
