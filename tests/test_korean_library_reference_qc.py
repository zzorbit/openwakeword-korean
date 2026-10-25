"""Automatic reference admission is technical evidence, not human listening."""
import json
import struct

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, read_rows, write_json, write_immutable_rows
from korean_openwakeword.project import create_project
from korean_openwakeword.reference_qc import admit_references
from korean_openwakeword.reference_bank import import_references
from korean_openwakeword.review import apply_review


def fixture_bank(tmp_path, train=4, validation=3):
    project = create_project(tmp_path / "bank", "보라야", "bora", train_voice_families=train,
                             validation_voice_families=validation)
    root = project.parent
    config = read_json(project)
    bank = read_json(root / "voices.json")
    rows = []
    for index, family in enumerate(f for f in bank["families"] if f["split"] in ("train", "validation")):
        path = root / "data/references" / family["split"] / (family["family_id"] + ".wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = .1 * np.sin(np.arange(8000) * (200 + index * 7) * 2 * np.pi / 16000)
        sf.write(path, samples, 16000, subtype="FLOAT")
        rows.append({"sample_id":family["family_id"],"voice_family_id":family["family_id"],
            "path":str(path.relative_to(root)),"sha256":digest(path),"split":family["split"],
            "text":config["tts"]["reference_text"],"description":family["description"],
            "provenance":"qwen_synthetic_reference","human_reference_used":False,
            "model":config["tts"]["design_model"],"status":"pending_review"})
    for split in ("train", "validation"):
        write_immutable_rows(root / f"data/references/{split}/references.jsonl", [r for r in rows if r["split"] == split])
    queue = root / "review/fixture_queue.jsonl"
    write_immutable_rows(queue, rows)
    return project, rows, queue, {"train":train,"validation":validation}


def publish_changed_rows(project, rows, queue):
    for split in ("train", "validation"):
        path = project.parent / f"data/references/{split}/references.jsonl"
        path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows if row["split"] == split),encoding="utf-8")
    queue.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")


def test_full_fifty_reference_voices_pass_without_inventing_human_approval(tmp_path):
    project, rows, queue, counts = fixture_bank(tmp_path,20,30)
    root = project.parent
    before = {str(path):digest(path) for path in [root/"voices.json",queue] + list((root/"data/references").rglob("*")) if path.is_file()}
    result = admit_references(project,counts,[queue])
    assert result["state"] == "accepted" and result["passed"] == result["checked"] == 50
    approval = read_json(root / "approval.json")
    assert approval["state"] == "pending" and approval["generation_policy_approved"] is False
    assert set(approval["accepted_reference_families"]) == {row["voice_family_id"] for row in rows}
    assert "reference_review_evidence" not in approval and "reference_review_sha256" not in approval
    assert not (root / "review/references_results.json").exists()
    receipt = read_json(root / result["receipt"]["path"])
    assert receipt["human_listened_count"] == receipt["independent_STT_runs"] == 0
    assert receipt["acoustic_diversity_measured"] is False
    assert all(digest(path) == sha for path,sha in before.items())
    approval_hash = digest(root / "approval.json")
    assert admit_references(project,counts,[queue])["receipt"] == result["receipt"]
    assert digest(root / "approval.json") == approval_hash


@pytest.mark.parametrize("problem,reason", [
    ("silence","near_silence"),("clipping","severe_clipping"),("nonfinite","empty_or_nonfinite_audio"),
    ("stereo","nonmono_audio"),("short","duration_outlier"),("long","duration_outlier"),
    ("corrupt","unreadable_audio"),("text","reference_text_or_design_changed"),
    ("model","invalid_synthetic_reference_provenance"),("human","invalid_synthetic_reference_provenance"),
    ("changed_hash","audio_missing_or_changed")])
def test_bad_reference_blocks_without_modifying_approval_or_any_other_voice(tmp_path,problem,reason):
    project,rows,queue,counts = fixture_bank(tmp_path)
    root = project.parent
    row = rows[0]
    path = root / row["path"]
    if problem == "silence": samples=np.zeros(8000)
    elif problem == "clipping": samples=np.ones(8000)
    elif problem == "nonfinite": samples=np.full(8000,np.nan)
    elif problem == "stereo": samples=np.ones((8000,2))*.1
    elif problem == "short": samples=np.ones(100)*.1
    elif problem == "long": samples=np.ones(16000*31)*.1
    else: samples=None
    if samples is not None:
        sf.write(path,samples,16000,subtype="FLOAT")
    elif problem == "corrupt": path.write_bytes(b"not a wave")
    elif problem == "text": row["text"]="다른 문장"
    elif problem == "model": row["model"]="unknown-tts-model"
    elif problem == "human": row["human_reference_used"]=True
    elif problem == "changed_hash": path.write_bytes(b"changed")
    if problem != "changed_hash": row["sha256"]=digest(path)
    publish_changed_rows(project,rows,queue)
    before = (root / "approval.json").read_bytes()
    result=admit_references(project,counts,[queue])
    assert result["state"] == "blocked" and reason in result["failed"][0]["reasons"]
    assert (root / "approval.json").read_bytes() == before
    assert read_json(root / result["receipt"]["path"])["accepted_reference_families"] == []


def test_exact_pcm_duplicates_are_detected_even_when_wav_bytes_differ(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    root=project.parent
    first=root/rows[0]["path"]
    other=root/rows[-1]["path"]
    body=bytearray(first.read_bytes())
    payload=b"different header metadata"
    body.extend(b"JUNK"+struct.pack("<I",len(payload))+payload+(b"\0" if len(payload)%2 else b""))
    body[4:8]=struct.pack("<I",len(body)-8)
    other.write_bytes(body)
    rows[-1]["sha256"]=digest(other)
    assert rows[0]["sha256"] != rows[-1]["sha256"]
    assert np.array_equal(sf.read(first)[0],sf.read(other)[0])
    publish_changed_rows(project,rows,queue)
    result=admit_references(project,counts,[queue])
    assert result["state"] == "blocked"
    assert sum("exact_decoded_audio_duplicate" in row["reasons"] for row in result["failed"]) == 2


@pytest.mark.parametrize("decision", ["discard","review"])
def test_human_veto_is_not_silently_approved_by_technical_checks(tmp_path,decision):
    project,rows,queue,counts=fixture_bank(tmp_path)
    answers={row["sample_id"]:{"decision":"keep","listened":True} for row in rows}
    answers[rows[0]["sample_id"]]={"decision":decision,"listened":False}
    apply_review(project,queue,answers,"references")
    approval_bytes=(project.parent/"approval.json").read_bytes()
    blocked=admit_references(project,counts,[queue])
    assert blocked["state"] == "blocked" and "human_discard_or_review_pending" in blocked["failed"][0]["reasons"]
    assert (project.parent/"approval.json").read_bytes() == approval_bytes
    answers[rows[0]["sample_id"]]={"decision":"keep","listened":True}
    apply_review(project,queue,answers,"references")
    assert admit_references(project,counts,[queue])["state"] == "accepted"


def test_optional_human_rejection_removes_acceptance_and_future_qc_cannot_override_it(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    assert admit_references(project,counts,[queue])["state"] == "accepted"
    answers={row["sample_id"]:{"decision":"keep","listened":True} for row in rows}
    answers[rows[0]["sample_id"]]={"decision":"discard","listened":True}
    apply_review(project,queue,answers,"references")
    assert rows[0]["voice_family_id"] not in read_json(project.parent/"approval.json")["accepted_reference_families"]
    assert admit_references(project,counts,[queue])["state"] == "blocked"


def test_final_reference_admission_is_not_opened_by_auto_check(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    before=(project.parent/"approval.json").read_bytes()
    with pytest.raises(ValueError,match="Final remains locked"):
        admit_references(project,{"final":1})
    assert (project.parent/"approval.json").read_bytes() == before


def test_changed_human_review_evidence_and_mismatched_playback_queue_fail_closed(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    with pytest.raises(ValueError,match="playback queues differ"):
        admit_references(project,counts,[])
    apply_review(project,queue,{row["sample_id"]:{"decision":"keep","listened":True} for row in rows},"references")
    approval=read_json(project.parent/"approval.json")
    (project.parent/approval["reference_review_evidence"][0]["path"]).write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="changed bound file"):
        admit_references(project,counts,[queue])


def test_reference_technical_receipt_is_immutable_and_tampering_is_rejected(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    result=admit_references(project,counts,[queue])
    approval_bytes=(project.parent/"approval.json").read_bytes()
    path=project.parent/result["receipt"]["path"]
    receipt=read_json(path)
    receipt["human_listened_count"]=7
    write_json(path,receipt)
    with pytest.raises(ValueError,match="Immutable reference technical receipt changed"):
        admit_references(project,counts,[queue])
    assert (project.parent/"approval.json").read_bytes()==approval_bytes


def test_automatic_reference_bank_can_be_reused_across_projects_without_faking_listening(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    admit_references(project,counts,[queue])
    original={row["voice_family_id"]:row["sha256"] for row in rows}
    for number in range(2):
        destination=create_project(tmp_path/f"destination{number}","달빛아",f"moon{number}",
            train_voice_families=counts["train"],validation_voice_families=counts["validation"])
        for split in ("train","validation"):
            assert import_references(destination,project,split)["reused_families"]==counts[split]
        local=read_json(destination.parent/"approval.json")
        assert local["state"]=="pending" and local["generation_policy_approved"] is False
        assert local["reused_reference_review_evidence"]==[]
        assert "reference_review_sha256" not in local
        assert not (destination.parent/"review/references_results.json").exists()
        report=read_json(destination.parent/local["reference_technical_admission"]["path"])
        assert report["human_listened_count"]==0 and report["checked_audio_sha256"]==original
        for split in ("train","validation"):
            imported=read_rows(destination.parent/f"data/references/{split}/references.jsonl")
            assert all(digest(destination.parent/row["path"])==original[row["voice_family_id"]] for row in imported)
        project=destination


def test_automatic_import_checks_proof_and_respects_optional_manual_veto(tmp_path):
    project,rows,queue,counts=fixture_bank(tmp_path)
    result=admit_references(project,counts,[queue])
    answers={rows[0]["sample_id"]:{"decision":"discard","listened":False}}
    single=project.parent/"review/one.jsonl"
    write_immutable_rows(single,rows[:1])
    apply_review(project,single,answers,"references")
    destination=create_project(tmp_path/"destination","달빛아","moon",train_voice_families=3)
    assert import_references(destination,project,"train")["reused_families"]==counts["train"]-1
    assert rows[0]["voice_family_id"] not in read_json(destination.parent/"approval.json")["accepted_reference_families"]
    bound_receipt=read_json(project.parent/result["receipt"]["path"])
    journal=project.parent/bound_receipt["quality_journal"]["path"]
    journal.write_text("",encoding="utf-8")
    other=create_project(tmp_path/"other","달빛아","other")
    with pytest.raises(ValueError,match="changed bound file"):
        import_references(other,project,"train")
