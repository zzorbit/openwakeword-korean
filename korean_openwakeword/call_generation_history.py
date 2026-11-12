"""Read the exact previous clone release without rewriting its frozen history.

Only reference-generation and preflight review behavior changed in this release.
The preceding clone generator and edge-conditioning rules remain valid evidence.
This bridge pins their source fingerprints and preserves every old receipt byte.
"""
from copy import deepcopy

from .common import digest, identity, read_json, read_rows, write_json


PREVIOUS_TTS = "9f39f255e364f0186f8e866bfb317dbef90d6c01f3481147c04637b3d0585aee"
PREVIOUS_REFERENCE = "710efa83b8a9d5203c3eb2006f821b0e1153ce0c33c283ac64b5ba698383a51f"
LEGACY_TTS = "a8a62455fee84260915d489aa2bccb0249ad874c9bfb2496678a5c725292473e"
POLICY = "prepared_clone_bounded_recovery_v1"
BRIDGE = "preserved_clone_history_after_reference_generation_v1"

# These exact TTS revisions only change when already-defined attempts are run
# and keep previously accepted audio excluded across the preflight sweeps.
# Keep the generation/QC evidence fingerprint on both checkout formats.
SCHEDULING_ONLY_TTS_DIGESTS = {
    "5d2b347f2b00ea11a9a5ecfd912d75fa5fb9843126cd64e46ab051c079e18334":
        "64cf14186bd4fab8ddbd0444830539ba453e199f01d76b6f2b4bbc9d5ed7dc55",
    "d9a0bd6407f3e19cc1bb9aa2c780c0d2f5dfa5e805ff5489d1b8a781e08987f7":
        "e5167c6a96979f424685544d1c726252a89052d17936bb1f3500309ece930e37",
}


def previous_contract(expected):
    policy = expected.get("clone_input_policy", {})
    if (expected.get("method") != "clone" or policy.get("policy") != POLICY
            or policy.get("reference_policy") != "clone_reference_edge_silence_v1"
            or policy.get("reference_rules") != {"frame_ms": 20, "context_ms": 200,
                "minimum_removed_edge_ms": 750, "activity_rms": .0001,
                "format": "WAV", "subtype": "FLOAT"}
            or policy.get("original_attempts") != 3 or policy.get("maximum_attempts") != 6):
        return None
    old = deepcopy(expected)
    old["implementation_sha256"] = PREVIOUS_TTS
    old["clone_input_policy"]["reference_implementation_sha256"] = PREVIOUS_REFERENCE
    return old


def preserved_context(root, output, expected, actual, *, create=False):
    """Return epochs for known old data, or None for the existing validator.

    A new receipt is published only when a caller really needs to append work.
    Original contract.json, generation_update.json and old JSONL stay untouched.
    """
    old = previous_contract(expected)
    if old is None or old == expected:
        return None
    legacy = {key: value for key, value in old.items() if key != "clone_input_policy"}
    legacy["implementation_sha256"] = LEGACY_TTS
    if actual not in (old, legacy):
        return None
    journal, contract = output / "candidates.jsonl", output / "contract.json"
    update_path = output / "generation_update.json"
    rows = read_rows(journal)
    old_update, legacy_count = None, 0
    if actual == legacy:
        if not update_path.exists():
            # The existing exact a8 bridge handles never-updated legacy queues.
            return None
        update = read_json(update_path)
        if update.get("target_contract") == expected:
            return None
        count = update.get("legacy_row_count")
        if type(count) is not int or not 0 <= count <= len(rows):
            raise ValueError("Invalid preserved clone journal boundary")
        from .tts import _journal_prefix
        wanted = {"kind": POLICY, "source_contract": {
            "path": str(contract.relative_to(root)), "sha256": digest(contract)},
            "source_contract_id": identity(legacy), "target_contract": old,
            "target_contract_id": identity(old), "legacy_row_count": count,
            "legacy_journal_prefix_sha256": _journal_prefix(journal, count)}
        if update != wanted:
            raise ValueError("Previous clone recovery receipt or journal prefix changed")
        old_update = {"path": str(update_path.relative_to(root)), "sha256": digest(update_path)}
        legacy_count = count
    elif update_path.exists():
        raise ValueError("Unexpected recovery receipt attached to a previous current contract")
    receipt_path = output / "generation_updates" / (identity(expected) + ".json")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        boundary = receipt.get("previous_row_count")
        if type(boundary) is not int or not legacy_count <= boundary <= len(rows):
            raise ValueError("Invalid preserved clone update boundary")
    else:
        receipt, boundary = None, len(rows)
    from .tts import _journal_prefix
    wanted = {"kind": BRIDGE,
        "source_contract": {"path": str(contract.relative_to(root)), "sha256": digest(contract)},
        "source_contract_id": identity(actual), "previous_contract": old,
        "previous_contract_id": identity(old), "previous_update": old_update,
        "previous_row_count": boundary,
        "previous_journal_prefix_sha256": _journal_prefix(journal, boundary),
        "target_contract": expected, "target_contract_id": identity(expected)}
    if receipt is not None and receipt != wanted:
        raise ValueError("Preserved clone update or original journal prefix changed")
    if receipt is None and create:
        write_json(receipt_path, wanted)
        receipt = wanted
    binding = ({"path": str(receipt_path.relative_to(root)), "sha256": digest(receipt_path)}
               if receipt is not None else None)
    epochs = []
    if legacy_count:
        epochs.append({"stop": legacy_count, "historical": True,
            "current_id": identity(legacy), "implementation_sha256": LEGACY_TTS, "update": None})
    epochs.append({"stop": boundary, "historical": False, "current_id": identity(old),
        "implementation_sha256": PREVIOUS_TTS, "update": old_update})
    return rows, {"current_id": identity(expected), "legacy_count": 0,
                  "update": binding, "epochs": epochs}
