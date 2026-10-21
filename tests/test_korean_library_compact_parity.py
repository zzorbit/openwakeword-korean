"""Archive retirement cannot change retained audio or the resulting training."""
import copy
import io
from importlib.resources import files
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_rows, write_json
from korean_openwakeword.compact_sources import prepare_compact_source
from korean_openwakeword.features import augment
from korean_openwakeword.frontend import StreamingFrontend
from korean_openwakeword.preparation import prepare_public_audio
from korean_openwakeword.project import create_project, load_project, training_batch
from korean_openwakeword.studio_data import extract_archive
import korean_openwakeword.train as training
from test_korean_library_compact_sources import archive_bytes, make_resolution, Transport
from test_korean_library_shared_delivery import _equal


def test_compact_retention_preserves_raw_normalized_pcm_features_and_training(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Fixture audio is procedural, not a claimed human/TTS performance sample.
    members = {}
    formats = ((48000, "FLAC", "PCM_24", "environment"),
               (44100, "FLAC", "PCM_24", "music"),
               (24000, "WAV", "PCM_16", "competing_speech"))
    for i, (rate, container, subtype, category) in enumerate(formats):
        noise = np.random.default_rng(100 + i).normal(0, .035, (rate * 3, 2))
        buffer = io.BytesIO()
        sf.write(buffer, noise, rate, format=container, subtype=subtype)
        members[f"corpus/{category}.{container.lower()}"] = buffer.getvalue()
    packed = archive_bytes(members, compressed=True)
    resolution, payloads = make_resolution([packed])
    monkeypatch.setattr(requests, "get", Transport(payloads).get)
    compact = prepare_compact_source(resolution, tmp_path / "compact",
        max_download_bytes=resolution["total_bytes"], max_extract_bytes=10 * 1024**2)
    legacy_archive = tmp_path / "legacy.tar.gz"
    legacy_archive.write_bytes(packed)
    extract_archive(legacy_archive, tmp_path / "legacy", max_bytes=10 * 1024**2)
    assert legacy_archive.is_file()
    assert not list((tmp_path / "compact/originals/data").glob("*.tar"))

    roots, configurations, rows = [], [], []
    for label, directory in (("legacy", tmp_path / "legacy"),
                             ("compact", Path(compact["audio_directory"]) / "part_0000")):
        project = create_project(tmp_path / (label + "_word"), "보라야", "bora",
                                 positive_originals=1, augmentation_rounds=5, updates=4)
        root, config = load_project(project)
        roots.append(root); configurations.append(config)
        index = []
        for name, raw in members.items():
            source = directory / name
            assert source.read_bytes() == raw
            category = source.stem
            index.append({"path": name, "dataset_id": "procedural_parity_fixture",
                "source_url": "https://example.org/procedural-fixture", "license": "fixture only",
                "category": category, "category_basis": "procedural fixture category",
                "speech": category == "competing_speech", "language": "Korean" if category == "competing_speech" else "unknown",
                "canonical_source_id": "fixture:" + category})
        index_path = tmp_path / (label + "_index.jsonl")
        index_path.write_text("".join(__import__("json").dumps(r) + "\n" for r in index), encoding="utf-8")
        prepared = prepare_public_audio(project, input_directory=directory, output="prepared",
            source_index=index_path, validation_fraction=0, preview_seconds=1)
        assert prepared["train_source_count"] == 3
        rows.append(sorted(read_rows(prepared["artifacts"]["source_manifest"]["path"]),
                           key=lambda r: r["canonical_source_id"]))
    for old, new in zip(*rows):
        for key in ("sample_id", "source_id", "canonical_source_id", "original_sha256", "sha256", "duration_s", "category"):
            assert old[key] == new[key], key
        assert Path(old["path"]).read_bytes() == Path(new["path"]).read_bytes()

    waveforms, augmentation_details = [], []
    for root, config, backgrounds in zip(roots, configurations, rows):
        call = root / "call.wav"
        sf.write(call, .06 * np.sin(np.arange(16000) * .11), 16000, subtype="PCM_16")
        positive = {"sample_id": "fixture_call", "canonical_source_id": "fixture_call",
            "path": str(call), "sha256": digest(call), "call_start_s": 0, "call_end_s": 1}
        impulse = root / "rir.wav"
        response = np.zeros(1600); response[[0, 160, 800]] = [1, .3, .1]
        sf.write(impulse, response, 16000, subtype="PCM_16")
        rir = {"sample_id": "fixture_room", "canonical_room_id": "fixture_room",
            "path": str(impulse), "sha256": digest(impulse)}
        settings = {**config["augmentation"], "clean_probability": 0, "rir_probability": 1}
        samples, details = [], []
        for role in training_batch(config):
            row = positive if role == "positive" else backgrounds[0]
            for round_id in range(5 if role == "positive" else 1):
                pcm, detail = augment(root, row, role, round_id, settings, backgrounds, [rir])
                samples.append(pcm); details.append(detail)
        waveforms.append(np.stack(samples)); augmentation_details.append(details)
    assert np.array_equal(*waveforms)
    assert augmentation_details[0] == augmentation_details[1]
    frontend = StreamingFrontend(files("korean_openwakeword") / "resources/sample_chunsik/frontend", ncpu=1)
    embeddings = [frontend.embed(pcm) for pcm in waveforms]
    assert np.array_equal(*embeddings)
    positive_start = list(training_batch(configurations[0])).index("positive")
    # Generic/Korean each supply one row; Positive contributes five successive rounds.
    for rounds in (1, 3, 5):
        assert np.array_equal(embeddings[0][positive_start:positive_start + rounds],
                              embeddings[1][positive_start:positive_start + rounds])

    config = copy.deepcopy(configurations[0])
    config["batch"] = {role: 2 for role in training_batch(config)}
    config["training"].update(device="cpu", cpu_threads=1, updates=4, schedule_horizon=4,
                              checkpoints=[2, 4], log_every=2)
    metadata = {role: {"role": role, "split": "train", "split_audit_passed": True,
        "frontend_id": identity(frontend.identity), "rounds": 5 if role == "positive" else 1,
        "feature_rows": 5 if role == "positive" else 1} for role in training_batch(config)}
    receipts = []
    for label, root, embedding in zip(("legacy", "compact"), roots, embeddings):
        pools, offset = {}, 0
        for role in training_batch(config):
            count = metadata[role]["feature_rows"]
            pools[role] = embedding[offset:offset + count]; offset += count
        receipts.append(training._train_member(config, root, metadata, pools, 41,
                                               root / "parity_model", True, False))
    assert receipts[0]["model_sha256"] == receipts[1]["model_sha256"]
    assert receipts[0]["exposure"] == receipts[1]["exposure"]
    for step in (2, 4):
        states = [torch.load(root / "parity_model" / f"checkpoint_{step:06d}.pt", weights_only=False)
                  for root in roots]
        assert _equal(*states), "Model/optimizer/sampler/RNG must be identical"
