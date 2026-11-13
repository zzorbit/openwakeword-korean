"""Read audited generation releases without rewriting their frozen history."""
from copy import deepcopy

from .common import digest, identity, read_json, read_rows, write_json


PREVIOUS_TTS = "9f39f255e364f0186f8e866bfb317dbef90d6c01f3481147c04637b3d0585aee"
PREVIOUS_REFERENCE = "710efa83b8a9d5203c3eb2006f821b0e1153ce0c33c283ac64b5ba698383a51f"
LEGACY_TTS = "a8a62455fee84260915d489aa2bccb0249ad874c9bfb2496678a5c725292473e"
POLICY = "prepared_clone_bounded_recovery_v1"
BRIDGE = "preserved_clone_history_after_reference_generation_v1"

# These exact revisions keep original synthesis settings, per-attempt seeds and
# QC. New production retries record their conditioned prompt and retry phase.
# Keep the existing contract fingerprint on both checkout formats.
SCHEDULING_ONLY_TTS_DIGESTS = {
    # Short-utterance budgets are recorded per new attempt. Existing banks
    # keep their scope, seeds, accepted files and technical QC contract.
    "35e797140ce4bc13235a0d9e259b129d63b7bbd66aac23fe2b1f8f624d9808bf":
        "64cf14186bd4fab8ddbd0444830539ba453e199f01d76b6f2b4bbc9d5ed7dc55",
    "4b80f8a5561ba98454b8e6fd8b0449301b6bf3852d6eb78b8f3af043b52bb488":
        "e5167c6a96979f424685544d1c726252a89052d17936bb1f3500309ece930e37",
    # The positive-only runtime limit can shorten rejected suffixes. The
    # original seeds, supported passing audio and technical QC are preserved.
    "25d0243343bda89d78c29bc4df61b79d6b4855fde88a76ec63358b78cabd4e8b":
        "64cf14186bd4fab8ddbd0444830539ba453e199f01d76b6f2b4bbc9d5ed7dc55",
    "f66d7545a171067d6ae5100a1ca039929e972293233ccf53ce9f0e93b9a304c5":
        "e5167c6a96979f424685544d1c726252a89052d17936bb1f3500309ece930e37",
    "af3d47e4a6a37257811ae7800d33a0971ccf03f6f932ce76f306add44745aaec":
        "64cf14186bd4fab8ddbd0444830539ba453e199f01d76b6f2b4bbc9d5ed7dc55",
    "7cb7481a601ea33ac7256623438ad314564a5ec1f2e946db9ecfe1a64f2f6f07":
        "e5167c6a96979f424685544d1c726252a89052d17936bb1f3500309ece930e37",
    "5d2b347f2b00ea11a9a5ecfd912d75fa5fb9843126cd64e46ab051c079e18334":
        "64cf14186bd4fab8ddbd0444830539ba453e199f01d76b6f2b4bbc9d5ed7dc55",
    "d9a0bd6407f3e19cc1bb9aa2c780c0d2f5dfa5e805ff5489d1b8a781e08987f7":
        "e5167c6a96979f424685544d1c726252a89052d17936bb1f3500309ece930e37",
}

# The new opt-in preparation policy records its own recipe and still verifies
# older edge-trim receipts exactly; previous production scopes remain usable.
CONDITIONING_ONLY_REFERENCE_DIGESTS = {
    "c26c086ecc6bf0fc72b0d721fe984d867a21ee730a21ee19543c00c97b8c822e":
        "4323c23f792ad308af81a305f8fdfb07e0b176d38b69d2f9dc63cb403fd262ae",
    "6766a71cf7464f113dcf1d2716af4f7ee99e386f62340666d641be34da7d862a":
        "c4a6a2a2e9385217a9821feba19cb3d7efea03431a34a657caa4f73186cc3332",
}

_PREVIOUS_HARD_IMPLEMENTATIONS = {
    "hard_tts.py": {"c7790a5d04f87c807f1a4f9dd29ba7caec55f3b48ce3dc660ca54b3f2b5f40bb",
                    "6b6dd2de36feb9d801355eb2c1cbedb8a3edd3bf15376d72f2cd341696d1c283",
                    "409dd788bbfdc68a2564d6b64940ea52099d49ad842f996dc0908e5dbe342e25",
                    "91fc1bcfb0ef14e2f563416a7ebe562061cfe217c83b979e9a9a011c767c85e9",
                    "53d3bb6a2d9c346cfab015caa962499ac7f0a5564c200cf55609c12f4285836c",
                    "43d748a34bcc746b69852acfe3dd08269200e6174afcc94fa5a50ac08c9f0713"},
    "hard_pilot.py": {"7c155ed6634c480f95c0723b281afb38c23fc70545d14034e2559a0c1b8bdc8a",
                      "e7fc3d6d1315ad1cf023ed623cb7ef916f10931cbbc172dcad8a59c8ff47a93a",
                      "154ec141c93335af08740c571a391d6f82b2b44e8c04b9ac641125f0315e7a09",
                      "dcc43508e9c22f101f34c50eb411c94c78fdfec9b11043fc1c12c08d1bbecc18",
                      "79dbf0a72191c7bc2372d149e002490f17cec935dcd49dfa6b83476e323c39fb",
                      "cef26cefa5d291ca003ec2e98711b929ce3ad0686512c1aee9858924091c7630"},
}


def compatible_hard_contract(actual, expected, *, pilot=False):
    """Runtime, pause and output-scope updates keep Hard's generation/QC rules intact."""
    if actual == expected:
        return True
    adjusted = deepcopy(actual)
    if pilot:
        keys = "implementation_sha256", "technical_qc_sha256"
        implementation, qc = (actual.get(key) for key in keys)
        for key in keys:
            adjusted[key] = expected[key]
    else:
        keys = "implementation", "technical_qc_implementation"
        if not all(isinstance(actual.get(key), dict) and "sha256" in actual[key] for key in keys):
            return False
        implementation, qc = (actual[key]["sha256"] for key in keys)
        for key in keys:
            adjusted[key]["sha256"] = expected[key]["sha256"]
    name = "hard_pilot.py" if pilot else "hard_tts.py"
    return (adjusted == expected and implementation in _PREVIOUS_HARD_IMPLEMENTATIONS[name]
            and qc in SCHEDULING_ONLY_TTS_DIGESTS)


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
