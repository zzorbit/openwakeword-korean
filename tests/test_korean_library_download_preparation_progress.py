from korean_openwakeword import studio_progress
from korean_openwakeword.common import read_json, write_json


def _file(group, *, identifier=None, state="complete", **values):
    return {"id": identifier or group, "group": group, "state": state, **values}


def _binding(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return {"path": str(path), "sha256": "a" * 64}


def _base_receipt(materials, source):
    directory = materials / source
    artifacts = {key: _binding(directory / filename) for key, filename in {
        "source_manifest": "train_sources.jsonl", "validation_reserve": "validation_reserve.jsonl",
        "unused_manifest": "unused_sources.jsonl", "representative_queue": "representatives.jsonl",
        "rejections": "rejected.json",
    }.items()}
    path = directory / "preparation_receipt.json"
    write_json(path, {"kind": "public_audio_preparation_v1", "state": "awaiting_human_review", "wakeword": "봉순아",
                      "output_directory": str(directory), "target_met": True,
                      "contract": _binding(directory / "contract.json"), "artifacts": artifacts})
    return path


def _prepared(root, files, *, version=None, progress=None):
    return studio_progress.prepared_download_groups(root, "봉순아", files, {"evaluation_materials_recipe": version},
        progress or {}, read_metadata=read_json)


def test_independent_base_receipt_completes_only_its_row(tmp_path):
    materials = tmp_path / ".studio" / "materials"
    rows = [_file("zeroth", identifier="base:zeroth"), _file("audioset"), _file("official_music"), _file("mit_rir")]
    music = _base_receipt(materials, "fma")
    assert _prepared(tmp_path, rows) == ["official_music"]
    _base_receipt(materials, "mit_rir")
    _base_receipt(materials, "zeroth")
    assert _prepared(tmp_path, rows) == ["mit_rir", "official_music", "zeroth"]

    complete = read_json(music)
    write_json(music, {**complete, "wakeword": "다른 호출어"})
    assert _prepared(tmp_path, rows) == ["mit_rir", "zeroth"]
    write_json(music, complete)
    (music.parent / "train_sources.jsonl").unlink()
    rows.append(_file("mit_rir", identifier="incomplete-transfer", state="verifying"))
    assert _prepared(tmp_path, rows) == ["zeroth"]


def test_shared_zeroth_waits_for_common_audio_but_rir_can_complete_independently(tmp_path, monkeypatch):
    catalog, rir_sha, pin = "0" * 16, "b" * 64, {"sha256": "c" * 64, "size_bytes": 32768}
    monkeypatch.setattr(studio_progress, "_download_evaluation_catalog", lambda version: (catalog, rir_sha, pin, 3))
    materials = tmp_path / ".studio" / "materials"
    directory = materials / f"evaluation_v6_{catalog}"
    rows = [_file("zeroth", identifier="base:zeroth"), _file("musan"), _file("seoul"), _file("fsd50k"), _file("common_rir")]
    _base_receipt(materials, "zeroth")
    assert _prepared(tmp_path, rows, version=6) == []
    rir_check = {"stage": "data_download", "label": "공통 잔향 원본 archive 준비·무결성 확인"}
    assert _prepared(tmp_path, rows, version=6, progress=rir_check) == ["fsd50k", "musan", "seoul", "zeroth"]
    assert _prepared(tmp_path, rows, version=6, progress={**rir_check, "stage": "evaluation"}) == []

    write_json(directory / "rir" / "receipt.json", {"kind": "common_rir_preparation_v1", "catalog_sha256": rir_sha,
        "archive": {"sha256": pin["sha256"], "bytes": pin["size_bytes"]}, "coverage": {"rir_sources": 3},
        "manifest": _binding(directory / "rir" / "rir_validation.jsonl")})
    assert _prepared(tmp_path, rows, version=6) == ["common_rir"]

    receipt = {"kind": "studio_evaluation_noise_preparation_v6", "catalog_id": catalog, "recipe_version": 6,
               "state": "awaiting_human_review", "wakeword": "봉순아", "evaluation_source_scope": "common_30h_only",
               "artifacts": {"source_manifest": _binding(directory / "evaluation_sources.jsonl"),
                             "representative_queue": _binding(directory / "representatives.jsonl")}}
    write_json(directory / "preparation.json", receipt)
    assert _prepared(tmp_path, rows, version=6) == ["common_rir", "fsd50k", "musan", "seoul", "zeroth"]

    (materials / "zeroth" / "preparation_receipt.json").unlink()
    assert _prepared(tmp_path, rows, version=6) == ["common_rir", "fsd50k", "musan", "seoul"]
    rows[0] = _file("zeroth", identifier="common:zeroth")
    assert _prepared(tmp_path, rows, version=6) == ["common_rir", "fsd50k", "musan", "seoul", "zeroth"]
    write_json(directory / "preparation.json", {**receipt, "evaluation_source_scope": "another_scope"})
    assert _prepared(tmp_path, rows, version=6) == ["common_rir"]


def test_automatic_component_has_measured_hours_and_a_finished_transfer(tmp_path):
    rows = [_file("automatic:mls", unit="hours", total=20, prepared_hours=21),
            _file("automatic:yodas2", unit="hours", total=20, prepared_hours=19.9),
            _file("automatic:audioset_unbalanced", unit="hours", total=20, done=20),
            _file("automatic:fma_full", state="downloading", unit="hours", total=20, prepared_hours=20)]
    assert _prepared(tmp_path, rows) == ["automatic:mls"]
