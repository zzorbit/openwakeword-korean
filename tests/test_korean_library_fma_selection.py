"""Selection freezes real provider identities and metadata, not listening approval."""
import csv
import json
from pathlib import Path
import zipfile

import pytest

from korean_openwakeword.common import digest, identity
from korean_openwakeword.fma_selection import select_fma_tracks


def _spec(track, **changes):
    return {"track": track, "artist": track, "artist_name": f"Artist {track}",
        "duration": 60, "license": "Attribution 4.0 International",
        "url": f"https://freemusicarchive.org/music/artist{track}/album/song/",
        "genre": "Rock", "genres": "[12]", "title": f"Song {track}", **changes}


def _csv(tmp_path, specs, name="tracks.csv"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "artist", "artist", "track", "track", "track", "track", "track", "track"])
        writer.writerow(["", "id", "name", "duration", "license", "url", "genre_top", "genres", "title"])
        writer.writerow(["track_id", "", "", "", "", "", "", "", ""])
        for row in specs:
            writer.writerow([row[key] for key in ("track", "artist", "artist_name", "duration", "license", "url", "genre", "genres", "title")])
    return path


def _entry(track, **changes):
    name = f"{track:06d}"
    value = {"filename": f"fma_full/{name[:3]}/{name}.mp3", "compressed_bytes": 1000+track,
        "original_bytes": 1200+track, "header_offset": track*10000,
        "crc32": "42e047d1", "compress_type": 8, "flag_bits": 8}
    value.update(changes)
    return value


def _select(tmp_path, specs, **options):
    return select_fma_tracks(_csv(tmp_path, specs), [_entry(row["track"]) for row in specs],
                             reserve_hours=0, **options)


def test_whole_tracks_overshoot_without_trim_and_preserve_provider_and_zip_facts(tmp_path):
    path = _csv(tmp_path, [_spec(2, duration=75)])
    before = digest(path)
    result = select_fma_tracks(path, [_entry(2)], target_hours=70/3600, reserve_hours=0)
    assert result["target_met"] and result["selected_provider_hours"] == pytest.approx(75/3600)
    row = result["selected"][0]
    assert row["archive_member"] == "fma_full/000/000002.mp3"
    assert row["canonical_source_id"] == "fma:track:2"
    assert row["provider_duration_s"] == 75 and row["original_bytes"] == 1202
    assert row["crc32"] == "42e047d1" and row["header_offset"] == 20000
    assert row["license"] == "Attribution 4.0 International"
    assert row["provider_title"] == "Song 2" and row["provider_genres"] == [12]
    assert result["estimated_member_payload_bytes"] == 1002
    assert result["metadata"]["sha256"] == before == digest(path)
    assert result["content_approved"] is False and result["admitted_hours"] == 0
    assert result["audio_reencoded"] is False and result["independent_STT_runs"] == 0
    assert result["whole_archive_sha1_verified"] is False
    assert result["selection_sha256"] == identity({k: v for k, v in result.items() if k != "selection_sha256"})
    json.dumps(result, allow_nan=False)


def test_genre_duration_balancing_includes_both_provider_genres_before_repeating(tmp_path):
    specs = [_spec(n, artist=n, genre="Rock" if n < 5 else "Jazz") for n in range(1, 9)]
    result = _select(tmp_path, specs, target_hours=120/3600)
    assert len(result["selected"]) == 2
    assert {row["provider_genre_top"] for row in result["selected"]} == {"Rock", "Jazz"}
    assert set(result["genre_distribution"]) == {"genre_top:Jazz", "genre_top:Rock"}


def test_artists_receive_a_first_turn_before_a_second_track_in_the_same_genre(tmp_path):
    specs = [_spec(n, artist=(n-1)//3+1) for n in range(1, 10)]
    result = _select(tmp_path, specs, target_hours=180/3600)
    assert len({row["artist_id"] for row in result["selected"]}) == 3
    assert result["selected_artist_count"] == 3
    assert result["max_selected_artist_track_count"] == 1


def test_input_order_does_not_change_the_seeded_selection_and_seed_can_change_it(tmp_path):
    specs = [_spec(n, artist=n//3+1, genre="Rock" if n % 2 else "Jazz") for n in range(1, 31)]
    a = select_fma_tracks(_csv(tmp_path, specs, "a.csv"), [_entry(row["track"]) for row in specs], target_hours=420/3600, reserve_hours=180/3600, seed=8)
    b = select_fma_tracks(_csv(tmp_path, list(reversed(specs)), "b.csv"), [_entry(row["track"]) for row in reversed(specs)], target_hours=420/3600, reserve_hours=180/3600, seed=8)
    c = select_fma_tracks(tmp_path / "a.csv", [_entry(row["track"]) for row in specs], target_hours=420/3600, reserve_hours=180/3600, seed=9)
    ids = lambda result: [row["track_id"] for row in result["selected"] + result["reserve"]]
    assert ids(a) == ids(b) and ids(a) != ids(c)
    assert a["archive_index_sha256"] == b["archive_index_sha256"]


def test_reserve_is_frozen_disjoint_and_continues_the_same_selection_order(tmp_path):
    specs = [_spec(n) for n in range(1, 9)]
    path, entries = _csv(tmp_path, specs), [_entry(row["track"]) for row in specs]
    result = select_fma_tracks(path, entries, target_hours=120/3600, reserve_hours=120/3600)
    combined = select_fma_tracks(path, entries, target_hours=240/3600, reserve_hours=0)
    ids = lambda rows: [row["track_id"] for row in rows]
    assert ids(result["selected"] + result["reserve"]) == ids(combined["selected"])
    assert set(ids(result["selected"])).isdisjoint(ids(result["reserve"]))
    assert result["reserve_target_met"] and result["reserve_summary"]["track_count"] == 2
    assert result["reserve_summary"]["member_payload_bytes"] > 0


def test_license_filter_excludes_unknown_and_no_derivatives_but_preserves_nc_sa_labels(tmp_path):
    terms = ["Attribution 4.0 International", "Attribution-NonCommercial-ShareAlike 3.0 International",
        "Attribution-NoDerivatives 4.0 International", "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        "", "unknown", "custom permission needed", "CC0 1.0 Universal", "Public Domain",
        "https://creativecommons.org/licenses/by-sa/4.0/"]
    specs = [_spec(n+1, license=license) for n, license in enumerate(terms)]
    result = _select(tmp_path, specs, target_hours=100)
    kept = {row["track_id"] for row in result["selected"]}
    assert kept == {1, 2, 8, 9, 10}
    assert result["rejected_counts"] == {"missing_license": 2, "no_derivatives_license": 2, "unrecognized_license": 1}
    nc = next(row for row in result["selected"] if row["track_id"] == 2)
    assert nc["license"] == terms[1]
    assert nc["license_flags"] == {"noncommercial": True, "sharealike": True, "provider_label_only": True, "usage_permission_independently_verified": False}


def test_commercial_only_policy_additionally_excludes_noncommercial_labels(tmp_path):
    specs = [_spec(1), _spec(2, license="Attribution-NonCommercial 3.0"), _spec(3, license="CC BY-SA 4.0"), _spec(4, license="CC0 1.0 Universal")]
    result = _select(tmp_path, specs, target_hours=100, license_policy="commercial-only")
    assert {row["track_id"] for row in result["selected"]} == {1, 3, 4}
    assert result["rejected_counts"]["noncommercial_license"] == 1


def test_known_ids_artist_names_and_album_url_prefixes_are_excluded_without_prefix_overmatch(tmp_path):
    specs = [_spec(1), _spec(2, artist=202), _spec(3, artist_name="Advent_Chamber Orchestra"),
        _spec(4, url="http://www.freemusicarchive.org/music/Artist/Album/song/"),
        _spec(5, url="https://freemusicarchive.org/music/Artist/AlbumOther/song/"), _spec(6)]
    result = _select(tmp_path, specs, target_hours=100,
        excluded_track_ids=["fma:track:1"], excluded_artist_ids=["fma:artist:202"],
        excluded_artist_names=["  ADVENT chamber   orchestra "],
        original_url_prefixes=["https://freemusicarchive.org/music/Artist/Album/"])
    assert {row["track_id"] for row in result["selected"]} == {5, 6}
    assert result["rejected_counts"] == {"excluded_artist_id": 1, "excluded_artist_name": 1,
        "excluded_original_url_prefix": 1, "excluded_track_id": 1}
    assert "unknown aliases" in result["exclusion_scope"]
    # Frozen plans are compared after loading JSON on resume. They must contain
    # native JSON arrays, not tuples that change type during serialization.
    assert result["excluded_original_url_prefixes_normalized"] == [["freemusicarchive.org", "/music/Artist/Album"]]
    assert json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False)) == result


def test_shortfall_is_reported_and_never_filled_by_repeating_tracks(tmp_path):
    result = _select(tmp_path, [_spec(2, duration=90)], target_hours=1)
    assert not result["target_met"] and result["shortfall_provider_hours"] == pytest.approx(.975)
    assert result["selected_track_count"] == 1 and result["reserve"] == []
    assert result["eligible_provider_hours"] == pytest.approx(.025)


def test_optional_artist_cap_can_cause_an_explicit_shortfall(tmp_path):
    result = _select(tmp_path, [_spec(n, artist=7) for n in range(1, 5)], target_hours=180/3600, max_artist_hours=120/3600)
    assert len(result["selected"]) == 2 and not result["target_met"]
    assert result["selection_skipped_counts"] == {"artist_duration_cap": 2}


def test_zipinfo_objects_and_index_mapping_preserve_the_same_archive_facts(tmp_path):
    path = _csv(tmp_path, [_spec(2)])
    entry = _entry(2)
    info = zipfile.ZipInfo(entry["filename"])
    info.compress_size, info.file_size, info.header_offset = entry["compressed_bytes"], entry["original_bytes"], entry["header_offset"]
    info.CRC, info.compress_type, info.flag_bits = int(entry["crc32"], 16), 8, 8
    a = select_fma_tracks(path, [info], target_hours=.001, reserve_hours=0)
    b = select_fma_tracks(path, {entry["filename"]: entry}, target_hours=.001, reserve_hours=0)
    assert a["selected"] == b["selected"] and a["archive_index_sha256"] == b["archive_index_sha256"]


@pytest.mark.parametrize("changes,reason", [( {"duration": "nan"}, "invalid_provider_duration"),
    ({"duration": 0}, "invalid_provider_duration"), ({"artist": ""}, "missing_artist_id"),
    ({"genres": "[invented]"}, "invalid_provider_genres")])
def test_bad_provider_fields_are_counted_as_ineligible(tmp_path, changes, reason):
    result = _select(tmp_path, [_spec(2, **changes)], target_hours=1)
    assert result["selected"] == [] and result["rejected_counts"][reason] == 1


def test_unknown_genre_is_reported_without_inventing_a_music_label(tmp_path):
    result = _select(tmp_path, [_spec(2, genre="", genres="[]")], target_hours=.001)
    assert result["selected"][0]["selection_stratum"] == "provider_genre_unknown"
    assert result["selected"][0]["provider_genres"] == []


@pytest.mark.parametrize("change", ["duplicate_metadata", "duplicate_archive", "bad_folder", "encrypted", "bad_crc"])
def test_ambiguous_or_unsafe_source_index_is_refused(tmp_path, change):
    specs, entries = [_spec(2)], [_entry(2)]
    if change == "duplicate_metadata": specs.append(_spec(2))
    elif change == "duplicate_archive": entries.append(_entry(2))
    elif change == "bad_folder": entries[0]["filename"] = "fma_full/001/000002.mp3"
    elif change == "encrypted": entries[0]["flag_bits"] = 1
    else: entries[0]["crc32"] = "invalid"
    with pytest.raises(ValueError):
        select_fma_tracks(_csv(tmp_path, specs), entries, target_hours=.001)


def test_non_audio_archive_members_are_not_download_candidates(tmp_path):
    entries = [_entry(2), {"filename": "fma_full/README.txt"}, {"filename": "../evil.mp3"}]
    result = select_fma_tracks(_csv(tmp_path, [_spec(2)]), entries, target_hours=.001, reserve_hours=0)
    assert result["archive_track_count"] == 1


@pytest.mark.parametrize("options", [{"target_hours": 0}, {"target_hours": float("nan")}, {"target_hours": None},
    {"target_hours": True}, {"seed": -1}, {"seed": True}, {"reserve_hours": -1},
    {"max_artist_hours": 0}, {"license_policy": "all"}, {"excluded_track_ids": "12"},
    {"excluded_artist_ids": [True]}, {"excluded_artist_names": "one"},
    {"original_url_prefixes": ["file:///oops"]}])
def test_invalid_settings_fail_before_selection(tmp_path, options):
    with pytest.raises(ValueError):
        select_fma_tracks(_csv(tmp_path, [_spec(2)]), [_entry(2)], **options)


def test_generator_exclusions_are_not_lost_during_validation(tmp_path):
    result = _select(tmp_path, [_spec(2, artist_name="Protected")], target_hours=1,
                     excluded_artist_names=(name for name in ["protected"]))
    assert result["selected"] == []
    assert result["rejected_counts"]["excluded_artist_name"] == 1
