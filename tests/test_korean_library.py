"""Library contracts, portable real model replay, and new-word integration."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, identity, read_json, write_json
from korean_openwakeword.project import create_project, load_project, BATCH, LEGACY_BATCH
from korean_openwakeword.runtime import WakeWordDetector, sample_bundle, verify_bundle
from korean_openwakeword.features import build_features, compare_rounds, verify_cache
from korean_openwakeword.train import PoolSampler, train, build_classifier


def wave(path, seed=1, seconds=1):
    rng = np.random.default_rng(seed)
    t = np.arange(round(16000 * seconds)) / 16000
    values = .15 * np.sin((220 + seed) * 2 * np.pi * t) + rng.normal(0, .005, len(t))
    sf.write(path, values, 16000, subtype="PCM_16")


class StubFrontend:
    identity = {"test": "functional feature stub, never production"}
    def embed(self, clips):
        values = clips.astype(np.float32).reshape(len(clips), 16, 2000)
        return np.repeat(values.mean(axis=2)[:, :, None], 96, axis=2)


@pytest.fixture
def project(tmp_path):
    path = create_project(tmp_path / "newword", "보라야", "bora",
        positive_originals=3, augmentation_rounds=1, updates=4)
    root, config = load_project(path)
    config["augmentation_rounds"] = 1
    config["augmentation"]["clean_probability"] = 1
    config["augmentation"]["moderate_probability"] = 0
    config["augmentation"]["rir_probability"] = 0
    config["pool_features"]["positive"] = "features/positive_R1"
    config["training"].update(device="cpu", updates=4, schedule_horizon=4, checkpoints=[2, 4], log_every=2)
    write_json(path, config)
    for role, manifest in (("positive", "positive_train"), ("generic_negative", "generic_negative"), ("korean_negative", "korean_negative"), ("hard_negative", "hard_negative")):
        rows = []
        for i in range(3):
            audio = root / "data" / (role + str(i) + ".wav")
            wave(audio, (1 if role == "positive" else 20 if role == "generic_negative" else 40) + i)
            row = {"sample_id": f"{role}_{i}", "source_id": f"{role}:{i}", "split": "train",
                "path": str(audio.relative_to(root)), "sha256": digest(audio), "speech": role != "generic_negative"}
            if role == "positive":
                row.update(provenance="qwen_tts", human_reference_used=False, text="보라야", voice_family_id=f"train_voice_{i}", call_start_s=0, call_end_s=1)
            elif role == "hard_negative":
                row.update(provenance="qwen_tts", human_reference_used=False, text="보람아", voice_family_id=f"hard_voice_{i}", category="competing_speech")
            else:
                row.update(provenance="public_dataset", source_url="https://example.org/fixture", license="test-only", category="competing_speech" if role == "korean_negative" else "environment")
            rows.append(row)
        (root / config["manifests"][manifest]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        build_features(path, role, frontend=StubFrontend(), smoke=True)
    return path


def test_new_word_isolation(tmp_path):
    a = create_project(tmp_path / "a", "보라야", "bora")
    b = create_project(tmp_path / "b", "다솜아", "dasom")
    assert read_json(a)["wakeword"] != read_json(b)["wakeword"]
    assert read_json(a)["batch"] == BATCH
    assert not any("F:" in str(v) or "C:" in str(v) for v in read_json(a).values())
    with pytest.raises(ValueError):
        create_project(a.parent, "다솜아", "dasom")
    with pytest.raises(ValueError):
        create_project(tmp_path / "bad", "춘식아 불켜줘", "bad")


@pytest.mark.legacy_parity
def test_sampler_matches_original_and_replays():
    Original = pytest.importorskip("wakeword_lab.training").PoolSampler
    sizes = dict(generic_negative=11, korean_negative=13, positive=17)
    new, old = PoolSampler(sizes, 42), Original(sizes, 42)
    for _ in range(5):
        a, b = new.next_indices(), old.next_indices()
        assert all(np.array_equal(a[r], b[r]) for r in LEGACY_BATCH)
    state = new.state(); expected = new.next_indices(); new.restore(state)
    new.restore(state); actual = new.next_indices()
    assert all(np.array_equal(expected[r], actual[r]) for r in LEGACY_BATCH)


@pytest.mark.legacy_parity
def test_classifier_initialization_matches_selected():
    import torch
    classifier = pytest.importorskip("wakeword_lab.temporal_classifier_v1").classifier
    torch.manual_seed(19); old = classifier()
    torch.manual_seed(19); new = build_classifier(dropout=0)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


def test_round_counts_nested_and_immutable(project):
    root, _ = load_project(project)
    build_features(project, "positive", output=root / "features" / "positive_R3", rounds=3, frontend=StubFrontend(), smoke=True)
    report = compare_rounds(root / "features" / "positive_R1", root / "features" / "positive_R3")
    assert report["shared_original_rounds"] == 3
    assert verify_cache(root / "features" / "positive_R3")["feature_rows"] == 9
    with pytest.raises(ValueError):
        build_features(project, "positive", rounds=1, frontend=StubFrontend(), smoke=True)
    meta = build_features(project, "positive", rounds=1, frontend=StubFrontend(), smoke=True, resume=True)
    assert meta["feature_rows"] == 3


def test_formal_training_refuses_unreviewed(project):
    with pytest.raises(ValueError, match="unapproved"):
        train(project)


def test_actual_small_three_model_training_and_resume(project, monkeypatch):
    root, config = load_project(project)
    complete = train(project, root / "runs" / "complete", smoke=True)
    assert len(complete["members"]) == 3
    assert complete["selected_workload"]["positive_tts_originals"] == 3
    assert complete["selected_workload"]["total_optimizer_updates"] == 12
    assert all(r["exposure"]["positive"]["drawn"] == 400 for r in complete["members"])
    assert all(r["exposure"]["hard_negative"]["drawn"] == 400 for r in complete["members"])
    again = train(project, root / "runs" / "complete", smoke=True, resume=True)
    assert [r["model_sha256"] for r in again["members"]] == [r["model_sha256"] for r in complete["members"]]
    from korean_openwakeword.bundle import package_run
    from korean_openwakeword.assets import prepare_frontend
    prepare_frontend(root / "models" / "frontend")
    package = package_run(project, root / "runs" / "complete", root / "bundle")
    assert package["wakeword"] == "보라야"
    assert package["members"][0]["smoke_test_only"] is True


def test_interrupted_resume_is_exact(project, monkeypatch):
    import korean_openwakeword.train as module
    import torch
    root, config = load_project(project)
    config["training"]["seeds"] = [41]
    write_json(project, config)
    full = train(project, root / "runs" / "full", smoke=True)
    original = module.write_json
    fired = False
    def interrupt_after_checkpoint(path, value):
        nonlocal fired
        original(path, value)
        if Path(path).name == "last_checkpoint.json" and not fired:
            fired = True
            raise RuntimeError("simulated interruption after a committed checkpoint")
    monkeypatch.setattr(module, "write_json", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated"):
        train(project, root / "runs" / "interrupted", smoke=True)
    monkeypatch.setattr(module, "write_json", original)
    resumed = train(project, root / "runs" / "interrupted", smoke=True, resume=True)
    a = torch.load(root / "runs/full/member1_seed41/checkpoint_000004.pt", weights_only=False)
    b = torch.load(root / "runs/interrupted/member1_seed41/checkpoint_000004.pt", weights_only=False)
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    assert full["members"][0]["model_sha256"] == resumed["members"][0]["model_sha256"]
    assert full["members"][0]["exposure"] == resumed["members"][0]["exposure"]


def test_fast_frontend_qualifies_real_sources(tmp_path):
    from korean_openwakeword.assets import prepare_frontend
    from korean_openwakeword.frontend import qualified_fast_frontend, StreamingFrontend
    assets = tmp_path / "frontend"; prepare_frontend(assets)
    probe = np.random.default_rng(42).integers(-15000, 15000, (2, 32000), dtype=np.int16)
    selected, report = qualified_fast_frontend(assets, probe)
    expected = StreamingFrontend(assets).embed(probe)
    assert report["bitwise_equal"] is True
    assert np.array_equal(selected.embed(probe), expected)


def test_gpu_gather_preserves_feature_bits():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from korean_openwakeword.train import InputDelivery
    pools = {r: np.random.default_rng(i).normal(size=(7, 16, 96)).astype(np.float16) for i, r in enumerate(BATCH)}
    indices = PoolSampler({r: 7 for r in BATCH}, 32).next_indices()
    cpu_x, cpu_y = InputDelivery(pools, "cpu").batch(indices)
    gpu_x, gpu_y = InputDelivery(pools, "cuda:0", budget_mb=100).batch(indices)
    assert torch.equal(cpu_x, gpu_x.cpu()) and torch.equal(cpu_y, gpu_y.cpu())


def test_dev_cache_replay_and_tuning_are_exact(tmp_path):
    from korean_openwakeword.demo import run_demo
    from korean_openwakeword.tune import replay, tune
    from korean_openwakeword.site import export_site
    result = run_demo(tmp_path / "demo")
    root = tmp_path / "demo"
    report, clips = read_json(root / "evaluation/report.json"), read_json(root / "evaluation/clips.json")
    with np.load(root / "evaluation/scores.npz") as c:
        again = replay(clips, c["scores"], c["offsets"], c["model_ids"].tolist(), report["runtime"])
    assert [r["results"] for r in clips] == [r["results"] for r in again]
    tuned = tune(root / "project/project.json", root / "sample_model", root / "evaluation", root / "tuned", [report["runtime"]["threshold"]])
    assert tuned["adoption"]["passed"] is False  # short subset cannot approve Final
    export_site(root / "tuned/evaluation", root / "tuned/site")
    assert (root / "tuned/site/index.html").is_file()
    from korean_openwakeword.bundle import freeze
    tampered = read_json(root / "tuned/evaluation/report.json")
    tampered["adoption"]["passed"] = True
    write_json(root / "tuned/evaluation/report.json", tampered)
    with pytest.raises(ValueError, match="receipt"):
        freeze(root / "project/project.json", root / "tuned/bundle", root / "tuned/evaluation/report.json")


@pytest.mark.legacy_parity
def test_sample_is_relocatable_and_same_as_original(tmp_path):
    from korean_openwakeword.assets import export_sample
    SharedFrontendEnsemblePredictor = pytest.importorskip("wakeword_lab.shared_frontend_ensemble_runtime_v1").SharedFrontendEnsemblePredictor
    EventDetector = pytest.importorskip("wakeword_lab.stream_eval").EventDetector
    relocated = tmp_path / "relocated"
    export_sample(relocated)
    new = WakeWordDetector(relocated)
    m = read_json(relocated / "manifest.json")
    old_runtime = {"vad_enabled": False}
    members = [{"path": str(relocated / r["path"]), "sha256": r["sha256"]} for r in m["members"]]
    original = SharedFrontendEnsemblePredictor(members, relocated / "frontend", old_runtime, "mean")
    audio_rows = read_json(relocated / "audio_manifest.json")
    compared = 0
    for row in audio_rows:
        new.reset(); original.reset()
        gate = EventDetector(m["runtime"]["threshold"], 2, 1)
        with sf.SoundFile(relocated / row["path"]) as f:
            index = 0
            while True:
                samples = f.read(1280, dtype="int16")
                if len(samples) < 1280:
                    break
                a, b = new.process(samples), original.score(samples)
                assert np.float32(a["score"]).view("u4") == np.float32(b).view("u4")
                old_event = gate.update(b, (index + 1) * .08, index * .08 + 1e-9 >= 2)
                assert a["event"] == old_event["event"]
                assert np.array_equal(np.asarray(a["member_scores"], np.float32), original.last["member_scores"])
                index += 1; compared += 1
    assert compared > 500
    (relocated / m["members"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_bundle(relocated)


def test_sample_requires_canonical_input():
    detector = WakeWordDetector()
    with pytest.raises(ValueError):
        detector.process(np.zeros(1280, dtype=np.float32))
    with pytest.raises(ValueError):
        detector.process(np.zeros((1280, 2), dtype=np.int16))


def test_final_generation_and_evaluation_locked(tmp_path):
    path = create_project(tmp_path / "project", "보라야", "bora")
    from korean_openwakeword.tts import generate, generate_references
    with pytest.raises(ValueError, match="locked"):
        generate(path, split="final")
    with pytest.raises(ValueError, match="reserved"):
        generate_references(path, split="final")


def test_review_never_certifies_unheard(tmp_path):
    from korean_openwakeword.review import apply_review
    path = create_project(tmp_path / "project", "보라야", "bora")
    root = path.parent
    audio = root / "data" / "pilot.wav"; wave(audio)
    queue = root / "review" / "queue.jsonl"
    queue.write_text(json.dumps({"sample_id": "pilot_1", "path": "data/pilot.wav", "sha256": digest(audio)}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unheard"):
        apply_review(path, queue, {"pilot_1": {"decision": "keep", "listened": False}}, "pilot")
    assert read_json(root / "approval.json")["generation_policy_approved"] is False


def test_tts_pilot_binds_method_and_bulk_has_no_appended_commands(tmp_path):
    from korean_openwakeword.tts import generate
    from korean_openwakeword.review import apply_review
    from korean_openwakeword.common import read_rows
    path = create_project(tmp_path / "word", "보라야", "bora", positive_originals=2)
    class FakeQwen:
        def generate_voice_design(self, text, language, **kwargs):
            assert text == "보라야" and language == "Korean"
            x = .15 * np.sin(np.arange(16000) / 20) + np.random.normal(0,.002,16000)
            return [x], 16000
    result = generate(path, pilot=True, limit=2, method="design", model=FakeQwen())
    assert result["technical_pass"] == 2 and result["independent_STT_runs"] == 0
    queue = path.parent / "data/pilot/train/candidates.jsonl"
    rows = read_rows(queue)
    apply_review(path, queue, {r["sample_id"]:{"decision":"keep", "listened":True} for r in rows}, "pilot")
    assert read_json(path.parent / "approval.json")["generation_method"] == "design"
    with pytest.raises(ValueError, match="Approve"):
        generate(path, method="clone", limit=2, model=FakeQwen())
    bulk = generate(path, method="design", model=FakeQwen())
    assert bulk["target"] == 2 and bulk["full_target_reached"]
    assert bulk["policy_admitted"] == 2 and bulk["human_listened"] == 0


def test_reference_reuse_preserves_listening_binding(tmp_path):
    from korean_openwakeword.reference_bank import import_references
    from korean_openwakeword.review import apply_review
    src = create_project(tmp_path / "old", "춘식아", "chunsik")
    dst = create_project(tmp_path / "new", "보라야", "bora")
    folder = src.parent / "data/references/train"; folder.mkdir(parents=True)
    audio = folder / "reference.wav"; wave(audio)
    row = {"sample_id":"ref1", "source_id":"synthetic:ref1", "voice_family_id":"family1", "split":"train", "path":str(audio.relative_to(src.parent)), "sha256":digest(audio), "provenance":"qwen_synthetic_reference", "human_reference_used":False, "text":"합성 참조입니다", "description":"test-only synthetic reference"}
    queue = folder / "references.jsonl"; queue.write_text(json.dumps(row)+"\n",encoding="utf-8")
    apply_review(src,queue,{"ref1":{"decision":"keep","listened":True}},"references")
    result = import_references(dst,src)
    assert result["reused_families"] == 1 and result["new_word_pilot_still_required"]
    assert read_json(dst.parent / "approval.json")["generation_policy_approved"] is False


def test_source_family_leaks_and_human_dev_are_rejected(tmp_path):
    from korean_openwakeword.data import audit_project
    from korean_openwakeword.evaluate import evaluate
    path = create_project(tmp_path / "project", "춘식아", "chunsik")
    root, cfg = load_project(path)
    row = {"sample_id":"one", "source_id":"same-original", "voice_family_id":"same-family", "sha256":"placeholder", "path":"unused.wav", "provenance":"qwen_tts", "human_reference_used":False,"text":"춘식아"}
    for split in ("train", "validation"):
        (root / cfg["manifests"]["positive_"+split]).write_text(json.dumps(dict(row,split=split))+"\n",encoding="utf-8")
    assert audit_project(path,check_files=False)["passed"] is False
