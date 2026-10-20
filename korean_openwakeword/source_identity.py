"""Declared lineage shared by Train, common evaluation and independent Final.

Dataset names are provenance, not proof of independence. A recording keeps the
same identity when it is renamed, segmented, resampled or used in a mixture.
"""


def partition_keys(row):
    """Yield stable partition keys without retaining a whole manifest in RAM."""
    source = row.get("canonical_source_id", row.get("source_id"))
    recording = row.get("canonical_recording_id", row.get("parent_source_id"))
    for kind, value in (("source", source), ("sha256", row.get("sha256")),
                        ("recording", recording), ("family", row.get("voice_family_id")),
                        ("speaker", row.get("canonical_speaker_id")),
                        ("room", row.get("canonical_room_id")),
                        ("book", row.get("canonical_book_id")),
                        ("artist", row.get("canonical_artist_id"))):
        if value:
            yield kind, value
    # Recording IDs and source IDs belong to the same source namespace. Without
    # this alias, a segment could call a known source its parent and evade audit.
    if recording:
        yield "source", recording
    for field in ("source_id", "canonical_source_id", "canonical_recording_id", "parent_source_id"):
        if row.get(field) and row[field] not in (source, recording):
            yield "source", row[field]
    for key in ("original_sha256", "normalized_pcm_sha256"):
        if row.get(key):
            yield ("sha256" if key == "original_sha256" else "pcm"), row[key]
    for field, kind in (("component_source_ids", "source"),
                        ("component_recording_ids", "source"),
                        ("component_sha256", "sha256"),
                        ("component_original_sha256", "sha256"),
                        ("component_pcm_sha256", "pcm"),
                        ("component_voice_family_ids", "family"),
                        ("canonical_speaker_ids", "speaker"),
                        ("canonical_artist_ids", "artist"),
                        ("component_book_ids", "book"),
                        ("component_room_ids", "room")):
        for value in row.get(field, []):
            if value:
                yield kind, value
    for component in row.get("components", []):
        if isinstance(component, dict):
            yield from partition_keys(component)


def component_lineage(row, role):
    """Copy declared source facts; do not invent speakers, rooms or artists."""
    keys = ("canonical_source_id", "source_id", "canonical_recording_id", "parent_source_id",
            "sha256", "original_sha256", "normalized_pcm_sha256", "normalized_frames",
            "voice_family_id", "canonical_speaker_id", "canonical_speaker_ids",
            "canonical_artist_id", "canonical_artist_ids", "canonical_room_id", "canonical_book_id",
            "provenance", "human_reference_used", "source_url", "original_source_url", "license",
            "path", "split", "language", "dataset_id", "eval_type", "category", "speech", "speaker_id")
    return {"role": role, **{key: row[key] for key in keys if key in row}}
