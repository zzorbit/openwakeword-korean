"""Review-only compatibility updates preserve existing human decisions."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json, snapshot_json
from test_korean_library_studio import manager  # noqa: F401
from test_reference_candidate_manager import review, send  # noqa: F401


def previous_contract():
    return read_json(Path(__file__).parent / "fixtures/reference_optional_listening_previous_contract.json")


def target_contract():
    # Frozen from the independently preserved optional-listening release's
    # execution-contract-after.json (SHA256 3b0824b1ca44af46970f2eeca378c5ed3e2959133eb28155ffbef16aa5d8bab6).
    contract = read_json(Path(__file__).parent / "fixtures/optional_listening_target_contract.json")
    assert identity(contract["files"]) == contract["computation_sha256"]
    return contract


def pin_previous(service, project_id):
    service._execution_contract = target_contract()
    state = service._load(project_id)
    state["implementation_id"] = "0" * 64
    state["execution_contract"] = {**previous_contract(),
        "settings_sha256": identity(state["settings"]), "plan_sha256": identity(state["plan"])}
    service._save(state)
    return state


def test_only_exact_review_source_change_is_audited():
    previous, current = previous_contract(), target_contract()
    update = studio._audited_reference_candidate_update(previous, current)
    assert update["kind"] == "audited_reference_optional_listening_update"
    assert update["changed_files"] == ["reference_candidates.py"]
    assert identity(previous["files"]) == previous["computation_sha256"]
    assert identity(current["files"]) == current["computation_sha256"]
    for side in ("source", "target"):
        old, new = deepcopy(previous), deepcopy(current)
        changed = old if side == "source" else new
        changed["files"]["tts.py"] = "0" * 64
        changed["computation_sha256"] = identity(changed["files"])
        assert studio._audited_reference_candidate_update(old, new) is None
        old, new = deepcopy(previous), deepcopy(current)
        (old if side == "source" else new)["stages"]["references"] = "f" * 64
        assert studio._audited_reference_candidate_update(old, new) is None


def test_migration_preserves_unheard_selection_and_exact_review_position(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    assert service.finish_candidates() == 0
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "toggle", view["items"][0])
    before = deepcopy(pin_previous(service, project_id))
    ledger = root / ".studio/reference-candidates/state.json"
    files_before = {path: digest(path) for path in root.rglob("*") if path.is_file()}
    compatibility = service._migration_view(before)
    assert compatibility["state"] == "available"
    assert compatibility["audited_reference_candidate_update"]["kind"] == "audited_reference_optional_listening_update"
    service.migrate(project_id)
    after = service._load(project_id)
    for key in before.keys() - {"implementation_id", "execution_contract", "migrations", "updated_at"}:
        assert after.get(key) == before[key], key
    assert all(digest(path) == sha for path, sha in files_before.items()
               if path != service._path(project_id))
    saved = read_json(ledger)
    assert saved["selected"] and not saved["heard"]
    assert after["execution_contract"]["computation_sha256"] == service._execution_contract["computation_sha256"]


def test_completed_and_downstream_work_are_never_rebound(review):
    service, project_id, root = review
    state = pin_previous(service, project_id)
    completed = deepcopy(state)
    completed["status"] = "completed"
    service._save(completed)
    before = digest(service._path(project_id))
    service.migrate(project_id)
    assert digest(service._path(project_id)) == before
    for changes in ({"status": "completed"}, {"index": next(i for i, row in enumerate(state["plan"]) if row["id"] == "positives")},
                    {"results": {"pilot": {"state": "completed"}}}):
        altered = deepcopy(state)
        altered.update(changes)
        assert not service._can_update_reference_optional_listening(altered)
    target = root / "data/pilot/already-generated.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"existing downstream artifact")
    assert not service._can_update_reference_optional_listening(state)


def test_frozen_legacy_binding_must_pass_original_strict_overlay(review, monkeypatch):
    service, project_id, root = review
    state = pin_previous(service, project_id)
    import korean_openwakeword.reference_voice_review as voice_review
    checks = []
    def overlay(checked_root, banks, binding):
        checks.append((checked_root, banks, binding))
        if binding.get("invalid_fixture"):
            raise ValueError("invalid old listening evidence")
        return banks
    monkeypatch.setattr(voice_review, "overlay_reference_banks", overlay)
    for kind in ("reference_voice_review_v2", "reference_voice_review_v3"):
        binding = snapshot_json(root, {"kind": kind}, "old-review")
        altered = deepcopy(state)
        altered["inputs"]["reference_voice_review"] = binding
        assert not service._can_update_reference_candidates(altered)
        assert service._can_update_reference_optional_listening(altered)
        assert checks[-1][2] == binding
        marked = snapshot_json(root, {"kind": kind, "listening_policy": "optional_candidate_listening_v1"}, "marked-review")
        altered["inputs"]["reference_voice_review"] = marked
        assert not service._can_update_reference_optional_listening(altered)
        altered["inputs"]["reference_voice_review"] = {**binding, "invalid_fixture": True}
        assert not service._can_update_reference_optional_listening(altered)
    altered["inputs"]["reference_voice_review"] = {"path": "missing.json", "sha256": "0" * 64}
    assert not service._can_update_reference_optional_listening(altered)
