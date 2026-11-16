"""Studio review queue validation and the hash-bound pronunciation draft."""

from pathlib import Path

from .common import bound_file, digest, identity, now, read_json, read_rows, write_json
from .hard_pilot import HARD_SAMPLE_REVIEW_POLICY
from .pilot import PREPOST_PILOT_POLICY


def review_descriptors(state):
    if state["status"] != "waiting_review":
        return []
    stage = state["plan"][state["index"]]["id"]
    reviews = state["results"][stage].get("reviews", [])
    return [r for r in reviews if r["id"] not in state["inputs"].get("review_results", {})]


def call_review_phase(state, review):
    if state["settings"].get("pilot_review_policy") == PREPOST_PILOT_POLICY:
        return {"pilot": "pre", "positive": "post"}.get(review["kind"])
    return None


def verify_review_queue(state, review):
    path = Path(review["queue"])
    if digest(path) != review.get("queue_sha256") or (review.get("pool_manifest") and
            digest(review["pool_manifest"]) != review.get("pool_sha256")):
        raise ValueError("검수 목록 또는 승인할 원본 목록이 변경되었습니다.")
    if review.get("synthetic_review_policy"):
        selection = read_json(bound_file(Path(state["project"]).parent, review["representative_selection"]))
        if (review["synthetic_review_policy"] not in ("one_per_phrase_v1", HARD_SAMPLE_REVIEW_POLICY)
                or selection.get("policy") != review["synthetic_review_policy"]
                or selection.get("source_manifest", {}).get("sha256") != review["pool_sha256"]
                or selection.get("representative_queue", {}).get("sha256") != review["queue_sha256"]):
            raise ValueError("유사어 대표 선정 기록이 현재 검수 목록과 다릅니다.")
    rows = read_rows(path)
    if not rows:
        raise ValueError("검수할 음원이 없습니다.")
    return rows


def preflight_draft_path(state, review):
    return Path(state["project"]).parent / ".studio/review-drafts" / (
        identity([review["id"], review["queue_sha256"]]) + ".json")


def read_preflight_draft(state, review, rows):
    from .review import PLAY_STARTED_POLICY, validate_started_decision
    if (call_review_phase(state, review) != "pre" or any(
            row.get("pronunciation_policy") != PREPOST_PILOT_POLICY
            or "preflight_round" not in row or row.get("qc", {}).get("passed") is not True for row in rows)):
        raise ValueError("자동 저장할 호출어 사전 검수 목록이 아닙니다.")
    expected = {"policy": PLAY_STARTED_POLICY, "project_id": state["id"],
                "queue_id": review["id"], "queue_sha256": review["queue_sha256"],
                "queued_audio_sha256": {row["sample_id"]: row["sha256"] for row in rows}}
    path = preflight_draft_path(state, review)
    if not path.exists():
        return {**expected, "revision": 0, "decisions": {}}
    draft = read_json(path)
    if (any(draft.get(key) != value for key, value in expected.items())
            or type(draft.get("revision")) is not int or draft["revision"] < 0
            or not isinstance(draft.get("decisions"), dict)
            or not set(draft["decisions"]).issubset(expected["queued_audio_sha256"])):
        raise ValueError("자동 저장한 사전 검수 초안의 음원 연결이 바뀌었습니다.")
    if (draft["revision"] == 0 and draft["decisions"]
            or draft["revision"] > 0 and (
                not isinstance(draft.get("saved_at"), str) or not draft["saved_at"]
                or draft.get("source_scope") != "Playback events and explicit decisions entered through the local review screen")):
        raise ValueError("자동 저장한 사전 검수 초안의 입력 출처가 올바르지 않습니다.")
    for answer in draft["decisions"].values():
        validate_started_decision(answer)
    return draft


def draft_response(draft, rows):
    return {"state": "draft_saved", "autosave_policy": draft["policy"],
            "queue_sha256": draft["queue_sha256"],
            "draft_decisions": draft["decisions"], "draft_revision": draft["revision"],
            "complete": set(draft["decisions"]) == {row["sample_id"] for row in rows},
            "kept": sum(answer["decision"] == "keep" for answer in draft["decisions"].values()), "total": len(rows)}


def save_preflight_draft(state, review, rows, payload):
    from .review import validate_started_decision
    draft = read_preflight_draft(state, review, rows)
    if type(payload.get("revision")) is not int or payload["revision"] != draft["revision"]:
        raise ValueError("다른 화면에서 판정이 바뀌었습니다. 저장된 초안을 다시 불러오세요.")
    sample = next((row for row in rows if row["sample_id"] == payload.get("sample_id")), None)
    if not sample or payload.get("audio_sha256") != sample["sha256"]:
        raise ValueError("자동 저장할 음원이 바뀌었습니다. 화면을 새로고침하세요.")
    bound_file(Path(state["project"]).parent, sample)
    decision = payload.get("decision")
    started, completed = payload.get("playback_started"), payload.get("playback_completed")
    if type(started) is not bool or type(completed) is not bool or completed and not started:
        raise ValueError("재생 시작과 재생 완료 기록을 확인하세요.")
    previous = draft["decisions"].get(sample["sample_id"], {})
    if decision == "clear":
        if started or completed or previous.get("playback_started") or previous.get("playback_completed"):
            raise ValueError("실제로 재생한 기록은 지울 수 없습니다. 정상 또는 다시 생성을 선택하세요.")
        changed = draft["decisions"].pop(sample["sample_id"], None) is not None
    else:
        answer = {"decision": decision, "playback_started": started or previous.get("playback_started", False),
                  "playback_completed": completed or previous.get("playback_completed", False)}
        answer["listened"] = answer["playback_completed"]
        validate_started_decision(answer)
        changed = answer != previous
        draft["decisions"][sample["sample_id"]] = answer
    if changed:
        draft.update(revision=draft["revision"] + 1, saved_at=now(),
                     source_scope="Playback events and explicit decisions entered through the local review screen")
        write_json(preflight_draft_path(state, review), draft)
    return draft_response(draft, rows)
