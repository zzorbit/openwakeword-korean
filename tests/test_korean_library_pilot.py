"""Pilot coverage follows admitted Train voices, not a fixed clip count."""
from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import append_row, digest, identity, read_json, read_rows, write_json, write_immutable_rows
from korean_openwakeword.pilot import prepare_pilot
from korean_openwakeword.project import create_project
from korean_openwakeword.review import apply_review


def reference_fixture(tmp_path, count=20):
    project = create_project(tmp_path / "word", "보라야", "bora", train_voice_families=count,
                             validation_voice_families=2, positive_originals=max(1000,count))
    root, config = project.parent, read_json(project)
    references = []
    for index, family in enumerate(f for f in read_json(root / "voices.json")["families"] if f["split"] == "train"):
        path = root / "data/references/train" / (family["family_id"] + ".wav")
        path.parent.mkdir(parents=True,exist_ok=True)
        sf.write(path,.1*np.sin(np.arange(4800)*(index+200)*2*np.pi/16000),16000,subtype="PCM_16")
        references.append({"sample_id":family["family_id"],"voice_family_id":family["family_id"],
            "split":"train","path":str(path.relative_to(root)),"sha256":digest(path),
            "text":config["tts"]["reference_text"],"description":family["description"],
            "provenance":"qwen_synthetic_reference","human_reference_used":False,"status":"pending_review"})
    write_immutable_rows(root/"data/references/train/references.jsonl",references)
    approval=read_json(root/"approval.json")
    approval["accepted_reference_families"]=[row["voice_family_id"] for row in references]
    write_json(root/"approval.json",approval)
    return project,references


def contract_for(project,references):
    config=read_json(project)
    return {"word":config["wakeword"],"split":"train","tts":config["tts"],"method":"clone",
        "references":references,"pilot":True,"implementation_sha256":digest(
            Path(__file__).parents[1]/"korean_openwakeword/tts.py")}


def add_calls(project,references,slots,*,passed=True,attempt=0):
    root,config=project.parent,read_json(project)
    folder=root/"data/pilot/train"
    folder.mkdir(parents=True,exist_ok=True)
    contract=contract_for(project,references)
    if not (folder/"contract.json").exists(): write_json(folder/"contract.json",contract)
    rows=[]
    for slot in slots:
        family=references[slot%len(references)]["voice_family_id"]
        sample_id=f"train_{slot:06d}_attempt{attempt}"
        path=folder/(sample_id+".wav")
        sf.write(path,.1*np.sin(np.arange(6400)*(slot+201+attempt)*2*np.pi/16000),16000,subtype="PCM_16")
        row={"sample_id":sample_id,"slot":slot,"source_id":"qwen:bora:"+sample_id,
            "path":str(path.relative_to(root)),"sha256":digest(path),"split":"train",
            "voice_family_id":family,"text":config["wakeword"],"provenance":"qwen_tts","human_reference_used":False,
            "qc":{"passed":passed,"reasons":[] if passed else ["fixture_failure"],"pronunciation_listened":False},
            "axes":{"voice_family":family,"style":"fixture","speed":"fixture","intonation":"fixture",
                    "vocal_effort":"fixture","call_length":"fixture","text_profile":"standalone"},
            "generation_contract_id":identity(contract),"generation_method":"clone","tts_settings_id":identity(config["tts"]),
            "individual_human_listening":False}
        append_row(folder/"candidates.jsonl",row)
        rows.append(row)
    return rows


def source_hashes(root):
    return {str(path):digest(path) for path in root.rglob("*") if path.is_file()}


def fail_if_generated(*args,**kwargs):
    raise AssertionError("Complete saved pilot must not invoke TTS")


def test_legacy_twenty_four_calls_reuse_exactly_twenty_distinct_voices_without_source_changes(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path)
    rows=add_calls(project,references,range(24))
    write_json(project.parent/"data/pilot/train/summary.json",{"target":24,"technical_pass":24,"full_target_reached":True})
    before=source_hashes(project.parent)
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",fail_if_generated)
    summary=prepare_pilot(project,20)
    queue=read_rows(summary["review_queue"])
    assert summary["full_target_reached"] and summary["target"]==summary["technical_pass"]==20
    assert summary["attempts"]==24 and summary["reused_existing_count"]==20
    assert queue==rows[:20]
    assert summary["selected_reference_families"]==[row["voice_family_id"] for row in references]
    assert summary["missing_reference_families"]==[]
    assert all(digest(path)==sha for path,sha in before.items())
    assert summary["human_listened"]==summary["policy_admitted"]==0
    assert prepare_pilot(project)["review_queue"]==summary["review_queue"]
    assert read_json(project.parent/"approval.json")["generation_policy_approved"] is False


@pytest.mark.parametrize("count",[20,100,500])
def test_new_pilot_covers_every_train_voice_once_and_forwards_runtime_options(tmp_path,monkeypatch,count):
    project,references=reference_fixture(tmp_path,count)
    calls=[]
    def generate(project_arg,split,limit,pilot,method,**kwargs):
        calls.append((project_arg,split,limit,pilot,method,kwargs))
        add_calls(project_arg,references,range(limit))
        return {"target":limit,"technical_pass":limit}
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",generate)
    before=digest(project.parent/"approval.json")
    summary=prepare_pilot(project,count,tts_backend="eager",tts_cpu_threads=7)
    assert calls==[(project,"train",count,True,"clone",{"backend":"eager","cpu_threads":7})]
    assert summary["target"]==summary["selected_reference_count"]==summary["technical_pass"]==count
    assert summary["reused_existing_count"]==0 and summary["full_target_reached"]
    assert [row["voice_family_id"] for row in read_rows(summary["review_queue"])]==[row["voice_family_id"] for row in references]
    assert len(set(row["voice_family_id"] for row in read_rows(summary["review_queue"])))==count
    assert digest(project.parent/"approval.json")==before


def test_incomplete_pilot_reuses_passes_and_keeps_failed_voice_visible(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path,4)
    initial=add_calls(project,references,[0,1])
    add_calls(project,references,[2],passed=False)
    def resume(project_arg,split,limit,pilot,method,**kwargs):
        assert limit==4
        add_calls(project_arg,references,[2],attempt=1)
        add_calls(project_arg,references,[3],passed=False)
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",resume)
    summary=prepare_pilot(project)
    assert summary["reused_existing_count"]==2 and summary["technical_pass"]==3
    assert summary["full_target_reached"] is False
    assert summary["missing_reference_families"]==[references[3]["voice_family_id"]]
    assert read_rows(summary["review_queue"])[:2]==initial
    assert len(read_rows(project.parent/"data/pilot/train/candidates.jsonl"))==5


def test_helper_accepts_actual_generator_journal_without_model_download(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path,3)
    prompts=[]
    class FixtureRuntime:
        def __init__(self,*args,**kwargs): pass
        def create_voice_clone_prompt(self,**kwargs):
            prompts.append(kwargs)
            return kwargs["ref_audio"]
        def generate_voice_clone(self,**kwargs):
            t=np.arange(9600)/16000
            return [.12*np.sin(2*np.pi*220*t)+np.random.normal(0,.002,len(t))],16000
    monkeypatch.setattr("korean_openwakeword.tts_runtime.CloneRuntime",FixtureRuntime)
    before=digest(project.parent/"approval.json")
    summary=prepare_pilot(project,3)
    assert summary["full_target_reached"] and summary["technical_pass"]==3
    assert len(prompts)==3
    assert [Path(prompt["ref_audio"]).name for prompt in prompts]==[Path(row["path"]).name for row in references]
    assert [row["voice_family_id"] for row in read_rows(summary["review_queue"])]==[row["voice_family_id"] for row in references]
    assert digest(project.parent/"approval.json")==before


@pytest.mark.parametrize("problem",["wrong_family","wrong_word","wrong_method","human","bad_contract_id","bad_tts_id","changed_wave","duplicate_passing_slot","missing_contract","changed_contract"])
def test_saved_pilot_provenance_errors_block_before_generation(tmp_path,monkeypatch,problem):
    project,references=reference_fixture(tmp_path,2)
    rows=add_calls(project,references,range(2))
    folder=project.parent/"data/pilot/train"
    if problem=="wrong_family": rows[0]["voice_family_id"]=references[1]["voice_family_id"]
    elif problem=="wrong_word": rows[0]["text"]="다른 호출어"
    elif problem=="wrong_method": rows[0]["generation_method"]="design"
    elif problem=="human": rows[0]["human_reference_used"]=True
    elif problem=="bad_contract_id": rows[0]["generation_contract_id"]="changed"
    elif problem=="bad_tts_id": rows[0]["tts_settings_id"]="changed"
    elif problem=="changed_wave": (project.parent/rows[0]["path"]).write_bytes(b"changed")
    elif problem=="duplicate_passing_slot": rows[1]=dict(rows[0],sample_id="duplicate")
    elif problem=="missing_contract": (folder/"contract.json").unlink()
    elif problem=="changed_contract": write_json(folder/"contract.json",{"changed":True})
    (folder/"candidates.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    before=source_hashes(project.parent)
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",fail_if_generated)
    with pytest.raises(ValueError): prepare_pilot(project)
    assert all(digest(path)==sha for path,sha in before.items())


def test_missing_admitted_voice_or_explicit_partial_count_cannot_silently_reduce_review(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path,4)
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",fail_if_generated)
    for count in (3,5,0,True):
        with pytest.raises(ValueError,match="must match every selected"): prepare_pilot(project,count)
    approval=read_json(project.parent/"approval.json")
    approval["accepted_reference_families"].pop()
    write_json(project.parent/"approval.json",approval)
    with pytest.raises(ValueError,match="must match every selected"): prepare_pilot(project,4)
    assert not (project.parent/"data/pilot/train").exists()


def test_duplicate_reference_id_and_reference_hash_changes_fail_closed(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path,2)
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",fail_if_generated)
    path=project.parent/"data/references/train/references.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in references+[references[0]]),encoding="utf-8")
    with pytest.raises(ValueError,match="unique admitted"): prepare_pilot(project)
    path.write_text("".join(json.dumps(row)+"\n" for row in references),encoding="utf-8")
    (project.parent/references[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed bound file"): prepare_pilot(project)


def test_human_veto_is_not_bypassed_by_a_legacy_accepted_reference_status(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path,2)
    references[0]["status"]="accepted"
    path=project.parent/"data/references/train/references.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in references),encoding="utf-8")
    queue=project.parent/"review/one_reference.jsonl"
    write_immutable_rows(queue,references[:1])
    apply_review(project,queue,{references[0]["sample_id"]:{"decision":"discard","listened":False}},"references")
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",fail_if_generated)
    with pytest.raises(ValueError,match="human discard"): prepare_pilot(project)


def test_missing_generation_capacity_is_reported_before_tts(tmp_path,monkeypatch):
    project,references=reference_fixture(tmp_path,4)
    config=read_json(project)
    config["targets"]["positive_originals"]=1
    write_json(project,config)
    monkeypatch.setattr("korean_openwakeword.tts_runtime.generate_with_runtime",fail_if_generated)
    with pytest.raises(ValueError,match="positive generation target must cover"): prepare_pilot(project)
