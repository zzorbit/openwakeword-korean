"""Original provider metadata is reused without speech/room inventions."""
import csv
import json

import pytest

from korean_openwakeword.common import digest, write_json
from korean_openwakeword.provider_index import build_provider_index


def _audio(folder, name):
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_bytes(b"Indexer must hash this fixture without decoding audio: " + name.encode())
    return path


def _audioset(tmp_path):
    folder = tmp_path / "audio"
    for name in ("aaaaaaaaaaa.flac", "bbbbbbbbbbb.flac", "ccccccccccc.flac"):
        _audio(folder, name)
    ontology = tmp_path / "ontology.json"
    write_json(ontology, [{"id": "/m/09x0r", "name": "Speech", "child_ids": ["speech_child"]},
                         {"id": "speech_child", "name": "Conversation", "child_ids": []},
                         {"id": "/m/04rlf", "name": "Music", "child_ids": ["music_child"]},
                         {"id": "music_child", "name": "Instrument", "child_ids": []},
                         {"id": "environment", "name": "Fan", "child_ids": []}])
    metadata = tmp_path / "balanced_train_segments.csv"
    metadata.write_text('# original AudioSet comment\n# YTID, start, end, positive_labels\n'
                        'aaaaaaaaaaa, 12.0, 22.0, "speech_child,music_child"\n'
                        'bbbbbbbbbbb, 0.0, 10.0, "music_child"\n'
                        'ccccccccccc, 5.0, 15.0, "environment"\n', encoding="utf-8")
    return folder, metadata, ontology


def test_audioset_reads_descendants_parent_offsets_and_unknown_language(tmp_path):
    folder, metadata, ontology = _audioset(tmp_path)
    protected = {path: digest(path) for path in folder.iterdir()}
    result = build_provider_index(folder, dataset="audioset", metadata=metadata, ontology=ontology)
    speech, music, environment = result["rows"]
    assert result["category_counts"] == {"competing_speech": 1, "music": 1, "environment": 1}
    assert speech["speech"] is True and speech["parent_start_s"] == 12 and speech["parent_end_s"] == 22
    assert speech["canonical_recording_id"] == "audioset:video:aaaaaaaaaaa"
    assert speech["canonical_source_id"] == "audioset:video:aaaaaaaaaaa"
    assert "start_s" not in speech  # Parent coordinates are not local audio intervals.
    assert music["speech"] is None and environment["speech"] is None
    assert all(row["language"] == "unknown" and "canonical_speaker_id" not in row for row in result["rows"])
    assert result["provider_metadata"] == [{"path": str(metadata), "sha256": digest(metadata)}, {"path": str(ontology), "sha256": digest(ontology)}]
    assert protected == {path: digest(path) for path in protected}
    assert result["independent_STT_runs"] == 0 and result["audio_changed"] is False


@pytest.mark.parametrize("change", ["missing_labels", "duplicate_video", "invalid_time", "unknown_class", "missing_ontology"])
def test_audioset_refuses_ambiguous_or_missing_provider_metadata(tmp_path, change):
    folder, metadata, ontology = _audioset(tmp_path)
    if change == "missing_labels":
        _audio(folder, "ddddddddddd.flac")
    elif change == "duplicate_video":
        with metadata.open("a", encoding="utf-8") as stream:
            stream.write('aaaaaaaaaaa, 22, 32, "speech_child"\n')
    elif change == "invalid_time":
        metadata.write_text('aaaaaaaaaaa,nan,22,"speech_child"\n', encoding="utf-8")
    elif change == "unknown_class":
        metadata.write_text('aaaaaaaaaaa,12,22,"invented"\n', encoding="utf-8")
    else:
        ontology = None
    with pytest.raises(ValueError):
        build_provider_index(folder, dataset="audioset", metadata=metadata, ontology=ontology)


def _fma(tmp_path):
    folder = tmp_path / "music"
    _audio(folder, "000002.mp3"); _audio(folder, "000002.wav"); _audio(folder, "000005.mp3")
    metadata = tmp_path / "tracks.csv"
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "artist", "track", "set", "track"])
        writer.writerow(["", "id", "license", "split", "url"])
        writer.writerow(["track_id", "", "", "", ""])
        writer.writerow([2, 17, "Attribution-NonCommercial 3.0", "training", "https://example.org/track2"])
        writer.writerow([5, 18, "", "test", "https://example.org/track5"])
    return folder, metadata


def test_fma_preserves_track_license_and_artist_and_excludes_missing_or_duplicate(tmp_path):
    folder, metadata = _fma(tmp_path)
    result = build_provider_index(folder, dataset="fma", metadata=metadata, license="Must not overwrite per-track license")
    assert len(result["rows"]) == 3 and len(result["excluded"]) == 2
    kept = next(row for row in result["rows"] if not row.get("excluded"))
    assert kept["path"] == "000002.mp3"
    assert kept["canonical_source_id"] == "fma:track:2" and kept["canonical_artist_id"] == "fma:artist:17"
    assert kept["license"] == "Attribution-NonCommercial 3.0"
    assert kept["speech"] is None and kept["language"] == "unknown"
    assert "canonical_speaker_id" not in kept
    assert "split" not in kept and kept["original_dataset_partition"] == "training"
    assert {row["reason"] for row in result["excluded"]} == {"Missing per-track provider license", "Duplicate file format of the same provider track; original MP3 preferred"}


def test_fma_does_not_invent_metadata_for_unlisted_track(tmp_path):
    folder, metadata = _fma(tmp_path)
    _audio(folder, "000999.mp3")
    with pytest.raises(ValueError, match="exact provider track"):
        build_provider_index(folder, dataset="fma", metadata=metadata)


def test_mit_without_room_metadata_keeps_one_conservative_group(tmp_path):
    folder = tmp_path / "rir"
    _audio(folder, "h001_Bedroom.wav"); _audio(folder, "h002_Office.wav")
    result = build_provider_index(folder, dataset="mit_rir")
    assert len({row["canonical_room_id"] for row in result["rows"]}) == 1
    assert all(row["room_identity_scope"] == "conservative_collection_group_not_independent_room" for row in result["rows"])
    assert all(row["speech"] is False and row["independent_room_partition_available"] is False for row in result["rows"])
    assert result["room_identities_inferred"] is False and result["limitations"]


def test_mit_exact_file_hashes_and_actual_complete_room_mapping(tmp_path):
    folder = tmp_path / "rir"
    a, b = _audio(folder, "a.wav"), _audio(folder, "b.wav")
    metadata = tmp_path / "provider.json"
    write_json(metadata, {"files": [{"path": str(a), "sha256": digest(a)}, {"path": str(b), "sha256": digest(b)}]})
    rooms = tmp_path / "rooms.csv"
    rooms.write_text("path,canonical_room_id\na.wav,provider:roomA\nb.wav,provider:roomB\n", encoding="utf-8")
    result = build_provider_index(folder, dataset="mit_rir", metadata=metadata, room_metadata=rooms)
    assert {row["canonical_room_id"] for row in result["rows"]} == {"provider:roomA", "provider:roomB"}
    assert not result["limitations"]
    assert all(row["independent_room_partition_available"] for row in result["rows"])
    metadata.write_text(json.dumps({"files": [{"path": str(a), "sha256": "0" * 64}, {"path": str(b), "sha256": digest(b)}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="provider file hash"):
        build_provider_index(folder, dataset="mit_rir", metadata=metadata, room_metadata=rooms)


def test_mit_partial_room_metadata_never_claims_independent_split(tmp_path):
    folder = tmp_path / "rir"
    a, b = _audio(folder, "a.wav"), _audio(folder, "b.wav")
    metadata = tmp_path / "provider.json"
    write_json(metadata, {"files": [{"path": str(a), "canonical_room_id": "roomA"}, {"path": str(b)}]})
    result = build_provider_index(folder, dataset="mit_rir", metadata=metadata)
    assert len({row["canonical_room_id"] for row in result["rows"]}) == 1
    assert result["rows"][0]["provider_declared_room_id"] == "roomA"
    assert all(not row["independent_room_partition_available"] for row in result["rows"])


def test_provider_directory_symlink_escape_is_refused(tmp_path):
    folder = tmp_path / "rir"
    folder.mkdir()
    outside = _audio(tmp_path, "outside.wav")
    try:
        (folder / "alias.wav").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink permission unavailable")
    with pytest.raises(ValueError, match="escaping symlinks"):
        build_provider_index(folder, dataset="mit_rir")


def test_public_cli_writes_immutable_complete_provider_index(tmp_path):
    folder, metadata, ontology = _audioset(tmp_path)
    from korean_openwakeword.cli import dispatch, parser
    from korean_openwakeword.common import read_rows
    output = tmp_path / "public_index.jsonl"
    arguments = parser().parse_args(["index-data", "--dataset", "audioset", "--input", str(folder),
                    "--metadata", str(metadata), "--ontology", str(ontology), "--output", str(output)])
    result = dispatch(arguments)
    assert result["source_count"] == 3
    assert len(read_rows(output)) == 3
    initial = digest(output)
    assert dispatch(arguments)["source_index"]["sha256"] == initial
    assert digest(output) == initial


def test_new_audioset_train_source_cannot_leak_into_old_mixed_evaluation(tmp_path):
    folder, metadata, ontology = _audioset(tmp_path)
    from korean_openwakeword.project import create_project
    from korean_openwakeword.common import read_json
    from korean_openwakeword.data import audit_project
    project = create_project(tmp_path / "project", "보라야", "source_alias_fixture")
    config = read_json(project)
    indexed = build_provider_index(folder, dataset="audioset", metadata=metadata, ontology=ontology)
    source = {**indexed["rows"][0], "sample_id": "train_audio", "split": "train", "provenance": "public_dataset"}
    (project.parent / config["manifests"]["noise"]).write_text(json.dumps(source) + "\n", encoding="utf-8")
    # Existing completed evaluations used video IDs for component sources.
    old = {"sample_id": "old_mixed_evaluation", "source_id": "fixture:mixed", "split": "validation",
           "category": "competing_speech", "kind": "positive", "sha256": "0" * 64,
           "component_source_ids": ["heldout:synthetic_call", "audioset:video:aaaaaaaaaaa"]}
    result = audit_project(project, check_files=False, extra_rows=[old])
    assert result["passed"] is False
    assert any(row["key"] == "source" and row["id"] == "audioset:video:aaaaaaaaaaa" for row in result["leaks"])
