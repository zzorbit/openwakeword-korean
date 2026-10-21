"""Portable metadata must describe the classifier heads actually deployed."""
from pathlib import Path
import shutil

from korean_openwakeword.bundle import package_run
from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.project import create_project
from korean_openwakeword.runtime import sample_bundle, verify_bundle


def test_one_member_package_declares_single_without_breaking_legacy_reading(tmp_path):
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
    output = project.parent / "bundle"
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
