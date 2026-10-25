"""One pronunciation sample per admitted Train voice, with frozen-call reuse."""
from collections import Counter
from pathlib import Path

from .common import bound_file, digest, identity, read_json, read_rows, write_immutable_rows
from .project import load_project
from .reference_qc import _human_answers


def _passing_slots(root, config, references, directory):
    """Validate the saved generation contract and every recorded attempt."""
    journal = directory / "candidates.jsonl"
    rows = read_rows(journal)
    contract_path = directory / "contract.json"
    expected = {"word":config["wakeword"], "split":"train", "tts":config["tts"], "method":"clone",
        "references":references, "pilot":True, "implementation_sha256":digest(Path(__file__).with_name("tts.py"))}
    if contract_path.exists():
        if read_json(contract_path) != expected:
            raise ValueError("Pilot model/reference/settings changed; preserve the saved run and use a new project")
    elif rows:
        raise ValueError("Pilot candidates exist without their frozen generation contract")
    contract_id, tts_id = identity(expected), identity(config["tts"])
    passing, samples = {}, set()
    for row in rows:
        slot = row.get("slot")
        if (type(slot) is not int or slot < 0 or row.get("sample_id") in samples
                or row.get("voice_family_id") != references[slot % len(references)]["voice_family_id"]
                or row.get("split") != "train" or row.get("text") != config["wakeword"]
                or row.get("provenance") != "qwen_tts" or row.get("human_reference_used") is not False
                or row.get("generation_method") != "clone" or row.get("generation_contract_id") != contract_id
                or row.get("tts_settings_id") != tts_id):
            raise ValueError("Pilot call identity, reference assignment or generation provenance changed")
        bound_file(root, row)
        samples.add(row["sample_id"])
        if row.get("qc", {}).get("passed") is True:
            if slot in passing:
                raise ValueError("Pilot journal has multiple passing calls for the same slot")
            passing[slot] = row
    return rows, passing


def prepare_pilot(project, train_count=None, *, tts_backend="auto", tts_cpu_threads=4):
    """Publish one successful call for every admitted Train reference, in order.

    A complete legacy 24-call pilot with 20 references needs no new generation:
    its first 20 slots are reused without rewriting any old journal or summary.
    This prepares a review queue; it never invents listening or approves a word.
    """
    if tts_backend not in ("auto", "eager") or type(tts_cpu_threads) is not int or tts_cpu_threads < 1:
        raise ValueError("Use TTS backend auto/eager and a positive CPU thread count")
    root, config = load_project(project)
    approval = read_json(root / "approval.json")
    journal = root / "data/references/train/references.jsonl"
    references = [row for row in read_rows(journal) if row.get("status") == "accepted"
                  or row.get("voice_family_id") in approval.get("accepted_reference_families", [])]
    ids = [row.get("voice_family_id") for row in references]
    if not references or None in ids or len(set(ids)) != len(ids):
        raise ValueError("A nonempty, unique admitted Train reference bank is required")
    target = len(references) if train_count is None else train_count
    if type(target) is not int or target != len(references):
        raise ValueError("Pilot count must match every selected admitted Train reference")
    answers = _human_answers(root, approval)
    for row in references:
        if (row.get("provenance") != "qwen_synthetic_reference" or row.get("human_reference_used") is not False
                or row.get("split") != "train"):
            raise ValueError("Pilot references must be this split's Qwen synthetic voices")
        if answers.get((row["sample_id"], row["sha256"])) in ("discard", "review"):
            raise ValueError("A selected pilot reference has a human discard or pending review")
        bound_file(root, row)
    directory = root / "data/pilot/train"
    rows, passing = _passing_slots(root, config, references, directory)
    reused = sum(slot in passing for slot in range(target))
    if reused != target:
        if config["targets"]["positive_originals"] < target:
            raise ValueError("The positive generation target must cover every Train reference before pilot generation")
        from .tts_runtime import generate_with_runtime
        generate_with_runtime(project, "train", target, True, "clone", backend=tts_backend, cpu_threads=tts_cpu_threads)
        rows, passing = _passing_slots(root, config, references, directory)
    selected = [passing[slot] for slot in range(target) if slot in passing]
    directory.mkdir(parents=True, exist_ok=True)
    queue = directory / ("review_one_per_voice_" + identity(selected) + ".jsonl")
    write_immutable_rows(queue, selected)
    bindings = {"references_journal":{"path":str(journal.relative_to(root)), "sha256":digest(journal)},
        "candidate_journal":{"path":str((directory / 'candidates.jsonl').relative_to(root)),
                             "sha256":digest(directory / "candidates.jsonl")},
        "generation_contract":{"path":str((directory / 'contract.json').relative_to(root)),
                               "sha256":digest(directory / "contract.json")},
        "review_queue":{"path":str(queue.relative_to(root)), "sha256":digest(queue)}}
    return {"attempts":len(rows), "technical_pass":len(selected), "target":target, "pilot":True,
        "human_listened":0, "policy_admitted":0, "independent_STT_runs":0,
        "review_queue":str(queue), "full_target_reached":len(selected) == target,
        "selected_reference_count":target, "selected_reference_families":ids,
        "reused_existing_count":reused, "missing_reference_families":[ids[slot] for slot in range(target) if slot not in passing],
        "distribution":{"voice_family":dict(Counter(row["voice_family_id"] for row in selected))}, "bindings":bindings}
