"""Call display metadata follows frozen references, never pool order or consent."""
import copy
import hashlib
import http.client
import json
from pathlib import Path
import threading

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, write_json
from test_korean_library_studio import create, manager  # noqa: F401


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def frozen_bank(manager):
    project = create(manager)
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    families, rows, originals = [], [], {}
    for split, per_group in (("train", 2), ("validation", 3)):
        number = 0
        for gender in ("man", "woman"):
            for age in ("child", "teenager", "young adult", "middle-aged", "older adult"):
                for member in range(per_group):
                    number += 1
                    family = f"{split}-stable-family-{number * 17}"
                    families.append({"family_id": family, "split": split, "gender": gender, "age_group": age})
                    originals[family] = sha(family)
                    rows.append({"sample_id": f"call-{family}", "text": state["wakeword"], "split": split,
                        "review_group": "학습용" if split == "train" else "공통 평가용",
                        "voice_family_id": family, "voice_reference_sha256": sha(family), "voice_number": number,
                        "sha256": sha("audio-" + family), "path": f"review/{family}.wav", "provenance": "qwen_tts", "qc": {"passed": True}})
    write_json(root / "voices.json", {"families": families})
    manifest = {"kind": "reference_voice_review_v3", "voice_bank": {"path": "voices.json", "sha256": digest(root / "voices.json")},
                "originals": originals, "replacements": {}}
    path = root / "review" / "frozen_reference_display.json"
    write_json(path, manifest)
    state["inputs"]["reference_voice_review"] = {"path": str(path.relative_to(root)), "sha256": digest(path)}
    state["settings"].update(train_references=20, validation_references=30)
    manager._save(state)
    return state, root, families, rows, manifest


def update_manifest(state, root, manifest):
    path = root / state["inputs"]["reference_voice_review"]["path"]
    write_json(path, manifest)
    state["inputs"]["reference_voice_review"]["sha256"] = digest(path)


def test_all_20_training_and_30_evaluation_labels_follow_identity_with_one_index(manager, monkeypatch):
    state, root, families, rows, _ = frozen_bank(manager)
    shuffled = list(reversed(rows[20:])) + list(reversed(rows[:20]))
    original = copy.deepcopy(shuffled)
    count = 0
    real_index = manager._call_voice_label_index

    def counted(value):
        nonlocal count
        count += 1
        return real_index(value)

    monkeypatch.setattr(manager, "_call_voice_label_index", counted)
    labels = manager._call_voice_labels(state, shuffled)
    by_family = {(item["split"], item["family_id"]): item for item in families}
    for audio, label in zip(shuffled, labels):
        family = by_family[audio["split"], audio["voice_family_id"]]
        gender = studio._REFERENCE_PREVIEW_TRAIT_LABELS["gender"][family["gender"]]
        age = studio._REFERENCE_PREVIEW_TRAIT_LABELS["age_group"][family["age_group"]]
        assert label == {"gender": gender, "age_group": age, "voice_number": audio["voice_number"],
                         "label": f"{audio['review_group']} · {gender} · {age} · {audio['voice_number']}번"}
    assert count == 1
    assert shuffled == original
    assert manager.workers.calls == []


def test_revision_and_conditioning_links_never_borrow_another_reference(manager):
    state, _, _, rows, _ = frozen_bank(manager)
    row = rows[0]
    conditioned = {**row, "voice_reference_sha256": sha("conditioned"), "clone_reference_preparation": {
        "kind": "clone_reference_preparation", "policy": "clone_reference_edge_silence_v1",
        "source": {"sha256": row["voice_reference_sha256"]}, "prepared": {"sha256": sha("conditioned")}}}
    wrong_revision = {**row, "voice_reference_sha256": sha("unrelated-replacement")}
    unbound_conditioning = {**conditioned, "clone_reference_preparation": {
        **conditioned["clone_reference_preparation"], "source": {}}}
    labels = manager._call_voice_labels(state, [conditioned, wrong_revision, unbound_conditioning])
    assert labels[0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    assert [label["label"] for label in labels[1:]] == ["학습용 · 1번", "학습용 · 1번"]


def test_unknown_legacy_duplicate_bank_and_mismatched_nested_metadata_fall_back(manager):
    state, root, families, rows, manifest = frozen_bank(manager)
    unknown = {**rows[0], "voice_family_id": "unknown", "voice_number": 17}
    wrong_split = {**rows[0], "split": "validation", "review_group": "공통 평가용"}
    nested_wrong = {**rows[0], "preflight_reference": {"voice_family_id": "another-family", "split": "validation", "gender": "woman", "age_group": "older adult"}}
    legacy = {key: value for key, value in rows[0].items() if key != "voice_reference_sha256"}
    labels = manager._call_voice_labels(state, [unknown, wrong_split, nested_wrong, legacy])
    assert labels[0]["label"] == "학습용 · 17번"
    assert labels[1]["label"] == "공통 평가용 · 1번"
    assert labels[2]["label"] == labels[3]["label"] == "학습용 · 남성 · 어린이 · 1번"
    # Contradictory duplicate family ids are ambiguous, not an ordinal relabel.
    families.insert(1, {**families[0], "gender": "woman"})
    write_json(root / "voices.json", {"families": families})
    manifest["voice_bank"]["sha256"] = digest(root / "voices.json")
    update_manifest(state, root, manifest)
    assert manager._call_voice_labels(state, [rows[0]])[0]["label"] == "학습용 · 1번"
    (root / "voices.json").unlink()
    assert manager._call_voice_labels(state, [rows[0]])[0]["label"] == "학습용 · 1번"


def test_same_family_name_in_two_splits_and_new_hash_bound_preflight_revision(manager):
    state, root, families, rows, manifest = frozen_bank(manager)
    families[20]["family_id"] = families[0]["family_id"]
    families[20].update(gender="woman", age_group="older adult")
    write_json(root / "voices.json", {"families": families})
    manifest["voice_bank"]["sha256"] = digest(root / "voices.json")
    update_manifest(state, root, manifest)
    evaluation = {**rows[20], "voice_family_id": rows[0]["voice_family_id"], "voice_reference_sha256": rows[0]["voice_reference_sha256"]}
    labels = manager._call_voice_labels(state, [rows[0], evaluation])
    assert labels[0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    assert labels[1]["label"] == "공통 평가용 · 여성 · 노년 · 1번"
    replacement = {**rows[0], "voice_reference_sha256": sha("new-approved-reference")}
    pilot = {**replacement, "preflight_reference": {"voice_family_id": replacement["voice_family_id"], "split": "train",
              "sha256": replacement["voice_reference_sha256"], "gender": "man", "age_group": "child"}}
    path = root / "review" / "frozen_preflight.jsonl"
    path.write_text(json.dumps(pilot) + "\n", encoding="utf-8")
    state["inputs"]["call_pilot_reviews"] = [{"queue": {"path": str(path.relative_to(root)), "sha256": digest(path)}}]
    assert manager._call_voice_labels(state, [replacement])[0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    # Preflight remains able to display its own saved direct metadata as before.
    assert manager._call_voice_labels(state, [pilot])[0]["label"] == "학습용 · 남성 · 어린이 · 1번"


@pytest.mark.parametrize("source", ["reference_manifest", "preflight_queue"])
def test_valid_commercial_scale_frozen_metadata_can_exceed_two_megabytes(manager, source):
    state, root, _, rows, manifest = frozen_bank(manager)
    padding = "x" * (2 * 1024**2 + 1)
    if source == "reference_manifest":
        manifest["fixture_padding"] = padding
        update_manifest(state, root, manifest)
    else:
        (root / "voices.json").unlink()
        row = {**rows[0], "preflight_reference": {"voice_family_id": rows[0]["voice_family_id"], "split": "train",
                  "sha256": rows[0]["voice_reference_sha256"], "gender": "man", "age_group": "child"}, "fixture_padding": padding}
        path = root / "review" / "commercial_size_preflight.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        state["inputs"]["call_pilot_reviews"] = [{"queue": {"path": str(path.relative_to(root)), "sha256": digest(path)}}]
    assert manager._call_voice_labels(state, [rows[0]])[0]["label"] == "학습용 · 남성 · 어린이 · 1번"


@pytest.mark.parametrize("optional", [
    {"gender": ["man"], "age_group": {"value": "child"}},
    {"preflight_reference": ["not-a-reference"]},
    {"clone_reference_preparation": "legacy-unknown"},
    {"clone_reference_preparation": ["legacy-unknown"]},
    {"clone_reference_preparation": {"kind": "clone_reference_preparation", "policy": "clone_reference_edge_silence_v1", "source": ["unknown"], "prepared": {"sha256": sha("unknown")}}},
    {"clone_reference_preparation": {"kind": "clone_reference_preparation", "policy": "clone_reference_edge_silence_v1", "source": {"sha256": sha("source")}, "prepared": "unknown"}},
])
def test_malformed_optional_legacy_metadata_is_only_an_unknown_display_fallback(manager, optional):
    state, _, _, rows, _ = frozen_bank(manager)
    row = {**rows[0], **optional}
    if "clone_reference_preparation" in optional:
        row["voice_reference_sha256"] = sha("unknown")
    label = manager._call_voice_labels(state, [row])[0]
    assert label["label"] == ("학습용 · 1번" if "clone_reference_preparation" in optional else "학습용 · 남성 · 어린이 · 1번")


@pytest.fixture
def display_http(manager):
    server = studio.make_studio_server(manager, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def get_json(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", path)
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    assert response.status == 200, payload
    return payload


def test_active_post_http_labels_and_positive_page_audio_ordinals_remain_exact(manager, display_http, monkeypatch):
    state, root, _, rows, _ = frozen_bank(manager)
    queue = root / "review" / "production.jsonl"
    queue.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    descriptor = {"id": "positive", "kind": "positive", "queue": str(queue), "queue_sha256": digest(queue)}
    state.update(plan=[{"id": "positive_review", "kind": "review"}], index=0, status="waiting_review",
                 results={"positive_review": {"state": "waiting_review", "reviews": [descriptor]}})
    state["settings"]["pilot_review_policy"] = studio.PREPOST_PILOT_POLICY
    manager._save(state)
    before = {path: digest(path) for path in (queue, root / "voices.json", root / "approval.json", manager._path(state["id"]))}
    active = get_json(display_http, f"/review/queue?project={state['id']}&queue=positive&rev={descriptor['queue_sha256']}")
    assert len(active["items"]) == 50
    assert active["items"][0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    assert active["items"][49]["label"] == "공통 평가용 · 여성 · 노년 · 30번"
    assert [item["id"] for item in active["items"]] == [row["sample_id"] for row in rows]
    assert [item["audio"].split("?")[0] for item in active["items"]] == [f"/review/audio/{index}" for index in range(50)]
    # The displayed audio ordinal remains independent from its reference family.
    row = {**rows[16], "sample_id": "production-audio-801"}
    page = [(800, 801, row), (801, 802, {**row, "sample_id": "production-audio-802"})]
    monkeypatch.setattr(manager, "positive_preview_rows", lambda project, start, limit: (state, page, 3000))
    whole = get_json(display_http, f"/positive-preview/queue?project={state['id']}&start=800")
    assert [item["number"] for item in whole["items"]] == [801, 802]
    assert [item["voice_number"] for item in whole["items"]] == [17, 17]
    assert whole["items"][0]["label"] == "학습용 · 여성 · 중년 · 801번"
    assert [item["id"] for item in whole["items"]] == ["production-audio-801", "production-audio-802"]
    assert [item["audio"] for item in whole["items"]] == [f"/positive-preview/audio/{index}?project={state['id']}" for index in (800, 801)]
    assert before == {path: digest(path) for path in before}
    assert manager.workers.calls == []


def test_completed_post_view_uses_the_same_metadata_and_original_audio_mapping(manager, monkeypatch):
    state, _, _, rows, _ = frozen_bank(manager)
    state["settings"]["pilot_review_policy"] = studio.PREPOST_PILOT_POLICY
    monkeypatch.setattr(manager, "_history_result", lambda value, stage: {} if stage == "positive_review" else None)
    answer = {"decisions": {row["sample_id"]: {"decision": "keep", "listened": True} for row in rows}}
    monkeypatch.setattr(manager, "history_playback_rows", lambda project, step: (state, answer, rows))
    view = manager._completed_stage_views(state)["pronunciation"]["review"]
    assert len(view["items"]) == 50
    assert view["items"][0]["label"] == "학습용 · 남성 · 어린이 · 1번"
    assert view["items"][49]["label"] == "공통 평가용 · 여성 · 노년 · 30번"
    assert [item["id"] for item in view["items"]] == [row["sample_id"] for row in rows]
    assert [item["audio"] for item in view["items"]] == [f"/history-stage/audio/{index}?project={state['id']}&step=pronunciation" for index in range(50)]
