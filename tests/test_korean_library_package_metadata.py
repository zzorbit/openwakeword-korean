"""Portable metadata must describe the classifier heads actually deployed."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from threading import Barrier

import pytest

from korean_openwakeword import bundle as bundle_module
from korean_openwakeword.bundle import package_run
from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.project import create_project
from korean_openwakeword.runtime import WakeWordDetector, sample_bundle, verify_bundle


@pytest.fixture
def package_inputs(tmp_path):
    project = create_project(tmp_path / "word", "보라야", "bora")
    config = read_json(project)
    config["training"]["seeds"] = [41]
    write_json(project, config)
    run = project.parent / "runs/one_member"
    member = run / "member1_seed41"
    member.mkdir(parents=True)
    sample = sample_bundle()
    source = read_json(sample / "manifest.json")
    shutil.copyfile(sample / source["members"][0]["path"], member / "model.onnx")
    write_json(member / "contract.json", {"config": config, "smoke_test_only": True})
    write_json(member / "receipt.json", {"model_sha256": digest(member / "model.onnx"),
                                         "state": "smoke_trained_not_performance_validated"})
    frontend = project.parent / config["frontend_dir"]
    frontend.mkdir(parents=True)
    for asset in source["frontend"]:
        shutil.copyfile(sample / asset["path"], frontend / Path(asset["path"]).name)
    return project, run, project.parent / "bundle"


def test_one_member_package_declares_single_without_breaking_legacy_reading(package_inputs):
    project, run, output = package_inputs
    packaged = package_run(project, run, output)
    assert len(packaged["members"]) == 1
    assert packaged["aggregation"] == "single"
    assert packaged["selected_candidate"]["kind"] == "single"
    assert packaged["selected_candidate"]["aggregation"] == "single"
    # Earlier one-member exports used the mean label. Preserve read-only
    # compatibility; only newly packaged metadata gets the precise label.
    packaged["aggregation"] = "ordered_float64_mean_to_float32"
    write_json(output / "manifest.json", packaged)
    assert verify_bundle(output)["selected_candidate"]["kind"] == "single"


def test_model_copied_during_source_change_must_match_training_receipt(package_inputs, monkeypatch):
    project, run, output = package_inputs
    member = run / "member1_seed41"
    model = member / "model.onnx"
    receipt = member / "receipt.json"
    original_model, original_receipt = model.read_bytes(), receipt.read_bytes()
    copyfile = shutil.copyfile

    def copy_changed_model(source, destination):
        if source == model:
            model.write_bytes(b"Different model after checking the training receipt")
        return copyfile(source, destination)

    monkeypatch.setattr(bundle_module.shutil, "copyfile", copy_changed_model)
    with pytest.raises(ValueError):
        package_run(project, run, output)
    assert not output.exists()
    assert receipt.read_bytes() == original_receipt

    model.write_bytes(original_model)
    monkeypatch.setattr(bundle_module.shutil, "copyfile", copyfile)
    published = package_run(project, run, output)
    assert published["members"][0]["sha256"] == read_json(receipt)["model_sha256"]


def test_package_rejects_a_project_update_before_publication(package_inputs, monkeypatch):
    project, run, output = package_inputs
    verify = bundle_module.verify_bundle
    original_project = project.read_bytes()

    def verify_then_change_project(path):
        verified = verify(path)
        project.write_bytes(original_project + b" ")
        return verified

    with monkeypatch.context() as patches:
        patches.setattr(bundle_module, "verify_bundle", verify_then_change_project)
        with pytest.raises(ValueError, match="Project configuration changed"):
            package_run(project, run, output)
    assert not output.exists()

    project.write_bytes(original_project)
    packaged = package_run(project, run, output)
    assert packaged["project_config_sha256"] == digest(project)


@pytest.mark.parametrize("precreated", [False, True])
def test_failed_package_leaves_destination_reusable(package_inputs, precreated):
    project, run, output = package_inputs
    asset = project.parent / read_json(project)["frontend_dir"] / "embedding_model.onnx"
    original = asset.read_bytes()
    asset.unlink()
    if precreated:
        output.mkdir()

    with pytest.raises(FileNotFoundError):
        package_run(project, run, output)
    assert output.is_dir() is precreated
    if precreated:
        assert not any(output.iterdir())
    assert not any(output.parent.glob(".bundle-*"))

    asset.write_bytes(original)
    published = package_run(project, run, output)
    assert verify_bundle(output) == published


def test_competing_packages_publish_only_one_verified_bundle(package_inputs, monkeypatch):
    project, run, output = package_inputs
    output.mkdir()
    copyfile = shutil.copyfile
    workers_ready = Barrier(2)

    def concurrent_copy(source, destination):
        if source.name == "model.onnx":
            assert not any(output.iterdir())
            workers_ready.wait(timeout=10)
        return copyfile(source, destination)

    monkeypatch.setattr(bundle_module.shutil, "copyfile", concurrent_copy)
    successes = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(package_run, project, run, output) for _ in range(2)]
        for job in futures:
            try:
                successes.append(job.result())
            except ValueError as exc:
                assert str(exc) == "Bundle destination must be empty"
    assert successes == [verify_bundle(output)]
    assert not any(output.parent.glob(".bundle-*"))


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_bundle_requires_one_of_each_frontend_asset(package_inputs, problem):
    project, run, output = package_inputs
    packaged = package_run(project, run, output)
    if problem == "missing":
        packaged["frontend"] = []
    else:
        packaged["frontend"].append(packaged["frontend"][0])
    write_json(output / "manifest.json", packaged)
    with pytest.raises(ValueError, match="frontend"):
        verify_bundle(output)


def test_sample_prediction_returns_a_snapshot_of_runtime(tmp_path):
    import numpy as np
    import soundfile as sf

    audio = tmp_path / "one_frame.wav"
    sf.write(audio, np.zeros(1280, dtype=np.int16), 16000, subtype="PCM_16")
    detector = WakeWordDetector()
    threshold = detector.runtime["threshold"]
    detector.runtime["extension"] = {"tags": []}
    report = detector.predict_file(audio)
    report["runtime"]["threshold"] = 0.25
    report["runtime"]["extension"]["tags"].append("private report tag")
    assert detector.runtime["threshold"] == detector.manifest["runtime"]["threshold"] == threshold
    assert detector.runtime["extension"]["tags"] == []
    detector.reset()
    assert detector.gate.threshold == threshold
