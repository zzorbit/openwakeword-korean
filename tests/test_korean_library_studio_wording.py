"""Display generation budgets and validate user phrases without changing runs."""
from pathlib import Path

import pytest

from korean_openwakeword.common import digest, read_json, write_immutable_rows
from test_korean_library_studio import create, manager  # noqa: F401


@pytest.mark.parametrize("preset,total,train", [("check",3000,1000),("trial",22000,20000),("production",102000,100000)])
def test_positive_progress_separates_training_and_common_evaluation(manager,preset,total,train):
    view = create(manager,preset=preset)
    manager.start(view["id"])
    state = manager._load(view["id"])
    state["index"] = next(i for i,s in enumerate(state["plan"]) if s["id"]=="positives")
    state["job"]["stage"] = "positives"
    manager._save(state)
    root = Path(state["project"]).parent
    for split,passes in (("train",1),("validation",2)):
        rows = [{"qc":{"passed":True},"voice_family_id":split} for _ in range(passes)]
        if split == "validation":
            rows.append({"qc":{"passed":False},"voice_family_id":split})
        path = root / f"data/generated/{split}/candidates.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        write_immutable_rows(path,rows)
    before = {path:digest(path) for path in (manager._path(view["id"]),root/"approval.json")}
    described = manager.describe(state)
    assert described["progress"] == {"done":3,"total":total,"unit":"개","label":"호출어 생성 완료"}
    assert described["counts"] == {"tts_generated":4,"technical_passed":3}
    assert described["next_action"] == f"학습용 1 / {train:,}개 · 공통 평가용 2 / 2,000개"
    assert all(digest(path)==sha for path,sha in before.items())


def waiting_phrases(manager,**kwargs):
    view = create(manager,**kwargs)
    state = manager._load(view["id"])
    state.update(status="waiting_input",auto_continue=False,
                 index=next(i for i,s in enumerate(state["plan"]) if s["id"]=="hard_phrases"))
    state["results"]["hard_phrases"] = {"state":"waiting_input","input_kind":"hard_phrases","suggestions":["보리야"]}
    manager._save(state)
    return state


@pytest.mark.parametrize("preset,per_phrase",[("check",20),("trial",100),("production",500)])
def test_phrase_guidance_distinguishes_text_count_from_audio_count(manager,preset,per_phrase):
    state = waiting_phrases(manager,preset=preset)
    path = manager._path(state["id"])
    before = path.read_bytes()
    view = manager.describe(state)
    described = view["input"]
    assert described["title"] == "유사어 문구 선택"
    assert f"문구마다 {per_phrase:,}개의 유사어 음성" in described["description"]
    assert described["samples_per_phrase"] == described["voice_count"] == per_phrase
    assert described["description"] == f"문구마다 {per_phrase:,}개의 유사어 음성을 만듭니다."
    assert described["label"] == "유사어 문구" and "한 줄" not in described["hint"]
    assert view["next_action"] == "유사어 문구를 입력하고 저장하세요."
    assert "직접 정하세요" in described["hint"] and "1~16자" in described["hint"]
    assert described["suggestions"] == ["보리야"]
    assert path.read_bytes()==before
    assert "title" not in state["results"]["hard_phrases"]


@pytest.mark.parametrize("phrases",[["보리야"]*2,["다른 말"],["hello"],["보리야"]*33,["보라야"]])
def test_invalid_phrases_fail_at_input_before_any_generation_or_saved_approval(manager,phrases):
    state = waiting_phrases(manager)
    path = manager._path(state["id"])
    before = path.read_bytes()
    approval = digest(Path(state["project"]).parent/"approval.json")
    with pytest.raises(ValueError):
        manager.submit_input(state["id"],{"phrases":phrases})
    assert path.read_bytes()==before and not manager.workers.items
    assert digest(Path(state["project"]).parent/"approval.json")==approval


@pytest.mark.parametrize("count",[1,32,40,200])
def test_phrase_input_supported_bounds_remain_available(manager,count):
    state = waiting_phrases(manager)
    phrases = ["나"+chr(0xac00+i) for i in range(count)]
    manager.submit_input(state["id"],{"phrases":phrases})
    saved = manager._load(state["id"])
    assert saved["inputs"]["hard_phrases"] == phrases
    assert saved["inputs"]["hard_phrases_approved"] is True
    assert len(manager.workers.items) == 1
    assert read_json(manager.workers.items[0].request_path)["stage"] == "hard_phrases"
