"""Reuse an admitted synthetic bank without importing personal recordings."""
import shutil
from .common import bound_file, digest, identity, now, read_json, read_rows, write_json, append_row, snapshot_json, require_final_lock
from .project import load_project
from .reference_qc import admitted_reference_rows, admit_references, _human_answers


def import_references(project, source_project, split="train"):
    root, config = load_project(project)
    source, _ = load_project(source_project)
    if split == "final" and not (root / "final_lock.json").exists():
        raise ValueError("Final references remain reserved until freeze")
    if split == "final":
        require_final_lock(project)
    approval = read_json(source / "approval.json")
    evidence = source / "review" / "references_results.json"
    bindings = approval.get("reference_review_evidence", []) + approval.get("reused_reference_review_evidence", [])
    reviews = [read_json(bound_file(source, e)) for e in bindings]
    if approval.get("reference_review_sha256"):
        if not evidence.exists() or digest(evidence) != approval["reference_review_sha256"]:
            raise ValueError("Source bank needs bound human listening evidence")
        if not reviews:
            reviews = [read_json(evidence)]
    automatic = admitted_reference_rows(source, approval) if split != "final" else {}
    answers = _human_answers(source, approval)
    if not reviews and not automatic:
        raise ValueError("Source bank needs bound listening or automatic reference evidence")
    rows = read_rows(source / "data" / "references" / split / "references.jsonl")
    accepted = []
    technical = []
    destination = root / "data" / "references" / split
    if (destination / "references.jsonl").exists():
        raise ValueError("Reference destination is immutable; import into a fresh project")
    destination.mkdir(parents=True, exist_ok=True)
    families = read_json(root / "voices.json")
    families["families"] = [f for f in families["families"] if f["split"] != split]
    for row in rows:
        if row.get("voice_family_id") not in approval.get("accepted_reference_families", []):
            continue
        matching = [r for r in reviews if r.get("queued_audio_sha256", {}).get(row["sample_id"]) == row["sha256"] and r["decisions"].get(row["sample_id"], {}).get("decision") == "keep" and r["decisions"][row["sample_id"]].get("listened") is True]
        is_automatic = automatic.get(row["voice_family_id"]) == row["sha256"]
        if (not matching and not is_automatic) or answers.get((row["sample_id"], row["sha256"])) in ("discard", "review"):
            continue
        if row.get("provenance") != "qwen_synthetic_reference" or row.get("human_reference_used") is not False or row.get("split") != split:
            raise ValueError("Only this split's Qwen synthetic references can be reused")
        audio = bound_file(source, row)
        target = destination / (row["voice_family_id"] + ".wav")
        shutil.copyfile(audio, target)
        if digest(target) != row["sha256"]:
            raise ValueError("Reference copy differs from its admitted source")
        cloned = dict(row, path=str(target.relative_to(root)), status="accepted" if matching else "pending_review")
        if matching:
            cloned["reused_from_review_id"] = identity(matching[0])
        else:
            cloned["reused_from_technical_receipt_sha256"] = approval["reference_technical_admission"]["sha256"]
            technical.append(row["voice_family_id"])
        append_row(destination / "references.jsonl", cloned)
        accepted.append(row["voice_family_id"])
        families["families"].append({"family_id": row["voice_family_id"], "split": split, "description": row["description"], "status": "accepted", "provenance": "qwen_synthetic"})
    if not accepted:
        raise ValueError("No admitted references available in this split")
    write_json(root / "voices.json", families)
    counts = {name:len(read_rows(root / f"data/references/{name}/references.jsonl")) for name in ("train", "validation")}
    technical_result = admit_references(project, {name:count for name,count in counts.items() if count}) if technical else None
    if technical_result and technical_result["state"] != "accepted":
        raise ValueError("Imported reference voices did not pass local automatic checks")
    local = read_json(root / "approval.json")
    local["accepted_reference_families"] = sorted(set(local.get("accepted_reference_families", [])) | set(accepted))
    local.setdefault("reused_reference_review_evidence", []).extend(snapshot_json(root, review, "reference_import") for review in reviews)
    local["reused_reference_evidence"] = {"source_review_ids": [identity(review) for review in reviews],
        "source_technical_receipt":approval.get("reference_technical_admission") if technical else None,
        "families": accepted, "imported_at": now()}
    write_json(root / "approval.json", local)
    return {"reused_families": len(accepted), "new_word_pilot_still_required": True, "human_reference_used": False}
