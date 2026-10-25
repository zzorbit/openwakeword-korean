"""Empty manifest creation is distinct from a later verified population."""
from pathlib import Path
import pytest

from korean_openwakeword.common import digest, read_json, resolved, write_json
from korean_openwakeword.project import create_project
from korean_openwakeword.studio import StudioManager


def example(tmp_path,role="positive_train",writer="positives",old=b""):
    project=create_project(tmp_path/"word","보라야","bora")
    root,config=project.parent,read_json(project)
    path=resolved(root,config["manifests"][role])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(old)
    initial=root/"initial.json"
    write_json(initial,{"result":{"state":"completed"},"artifact_bindings":[{"path":str(path),"sha256":digest(path)}]})
    path.write_bytes(b'{"sample_id":"generated"}\n')
    populated=root/"populated.json"
    write_json(populated,{"result":{"state":"completed"},"artifact_bindings":[{"path":str(path),"sha256":digest(path)}]})
    state={"project":str(project),"history":[{"stage":"data_connect","result":str(initial),"sha256":digest(initial)},
        {"stage":writer,"result":str(populated),"sha256":digest(populated)}]}
    return state,path,initial,populated


@pytest.mark.parametrize("role,writer",[("positive_train","positives"),("positive_validation","positives"),("hard_negative","hard_connect")])
def test_empty_manifests_can_be_filled_only_with_verified_later_writer(tmp_path,role,writer):
    state,path,initial,populated=example(tmp_path,role,writer)
    before={p:digest(p) for p in (path,initial,populated)}
    StudioManager._verify_history(None,state)
    assert all(digest(p)==sha for p,sha in before.items())


@pytest.mark.parametrize("tamper",["wrong_writer","wrong_role","nonempty_initial","waiting_writer","failed_writer",
    "missing_binding","changed_receipt","changed_current_file","earlier_writer","changed_audio"])
def test_manifest_update_never_hides_missing_evidence_or_modified_artifacts(tmp_path,tamper):
    role="generic_negative" if tamper=="wrong_role" else "positive_train"
    state,path,initial,populated=example(tmp_path,role=role,old=b"old-data" if tamper=="nonempty_initial" else b"")
    if tamper=="wrong_writer": state["history"][1]["stage"]="evaluation"
    if tamper in ("waiting_writer","failed_writer","missing_binding"):
        value=read_json(populated)
        if tamper=="waiting_writer": value["result"]["state"]="waiting_input"
        if tamper=="failed_writer": value["error"]={"message":"failed"}
        if tamper=="missing_binding": value["artifact_bindings"]=[]
        write_json(populated,value)
        state["history"][1]["sha256"]=digest(populated)
    if tamper=="changed_receipt": populated.write_text("{}",encoding="utf-8")
    if tamper=="changed_current_file": path.write_bytes(b"unrecorded audio rows")
    if tamper=="earlier_writer": state["history"].reverse()
    if tamper=="changed_audio":
        audio=path.parent/"immutable.wav"
        audio.write_bytes(b"original")
        value=read_json(initial)
        value["artifact_bindings"].append({"path":str(audio),"sha256":digest(audio)})
        write_json(initial,value)
        state["history"][0]["sha256"]=digest(initial)
        audio.write_bytes(b"changed")
    with pytest.raises(ValueError): StudioManager._verify_history(None,state)
