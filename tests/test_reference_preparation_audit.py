"""Historical preparation release boundaries, independent of later releases."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, read_json
from test_korean_library_studio import manager  # noqa: F401 - pytest fixture
from test_reference_candidate_manager import review  # noqa: F401 - pytest fixture


def previous_contract():
    return read_json(Path(__file__).parent / "fixtures/reference_preparation_previous_contract.json")


def target_contract():
    # Exact bytes from storage/support/reference-preparation-stage-20261108/
    # target-execution-contract.json. Confirmed by the preserved GaYoung
    # b94c7b590f6840dcb737f1fbecac6d33/receipt.json (SHA256
    # ea687238d35119d3b345833af487056c1c2898fc2bb93a3f307baeadcbc71b86),
    # recording the actual 151d... -> 9bad... preparation migration.
    path = Path(__file__).parent / "fixtures/reference_preparation_target_contract.json"
    assert digest(path) == "3813a881f9b97145a3aa689c518d0da4770b4f10218db5fb00109d206692ff56"
    target = read_json(path)
    assert target["computation_sha256"] == "9badcb7d019ea1d04e7784ea177c82fbd5d613c6283a28ea9cb7985ef2bb9b6b"
    assert identity(target["stages"]) == "26f7efbaaae71b689d10c53a1d4dd9c56919d8b44a43dd0bd17b1802e1301fb9"
    return target


@pytest.fixture
def preparation_release(monkeypatch):
    # Exercise the already registered historical preparation release, before
    # the later optional-listening composition superseded the same source.
    # Production registries and current-source admission remain unchanged.
    monkeypatch.setattr(studio, "_AUDITED_REFERENCE_OPTIONAL_LISTENING_UPDATES", {})
    return target_contract()


def test_preparation_update_accepts_only_the_audited_contract(preparation_release):
    previous = previous_contract()
    current = preparation_release
    update = studio._audited_reference_candidate_update(previous, current)
    assert update is not None
    assert update["kind"] == "audited_reference_candidate_preparation_update"
    assert update["changed_files"] == ["reference_candidates.py", "studio_steps.py"]
    assert update["source_computation_sha256"] == previous["computation_sha256"]
    assert update["target_computation_sha256"] == current["computation_sha256"]
    for item in (previous, current):
        assert identity(item["files"]) == item["computation_sha256"]
    for side in ("source", "target"):
        old, new = deepcopy(previous), deepcopy(current)
        changed = old if side == "source" else new
        changed["files"]["tts.py"] = "0" * 64
        changed["computation_sha256"] = identity(changed["files"])
        assert studio._audited_reference_candidate_update(old, new) is None
    altered = deepcopy(current)
    altered["stages"]["references"] = "f" * 64
    assert studio._audited_reference_candidate_update(previous, altered) is None


def test_preparation_migration_preserves_existing_review_and_completed_stages(review, preparation_release):
    mgr, project_id, root = review
    mgr._execution_contract = deepcopy(preparation_release)
    state = mgr._load(project_id)
    state["settings"].pop("reference_preparation_policy", None)
    state.pop("reference_preparation_policy", None)
    state["implementation_id"] = "0" * 64
    state["execution_contract"] = {**previous_contract(),
        "settings_sha256": identity(state["settings"]), "plan_sha256": identity(state["plan"])}
    mgr._save(state)
    mgr.reference_preview_queue(project_id)
    before = deepcopy(mgr._load(project_id))
    ledger = root / ".studio/reference-candidates/state.json"
    before_ledger = digest(ledger) if ledger.exists() else None
    config_hash, approval_hash = digest(state["project"]), digest(root / "approval.json")
    view = mgr._migration_view(before)
    assert view["state"] == "available"
    mgr.migrate(project_id)
    after = mgr._load(project_id)
    assert after["reference_preparation_policy"] == "initial_candidate_batches_v1"
    for key in ("settings", "plan", "index", "inputs", "history", "results", "config_sha256"):
        assert after.get(key) == before.get(key), key
    assert digest(state["project"]) == config_hash
    assert digest(root / "approval.json") == approval_hash
    assert (digest(ledger) if ledger.exists() else None) == before_ledger
    assert after["execution_contract"]["computation_sha256"] == preparation_release["computation_sha256"]


def test_historical_preparation_target_does_not_authorize_current_registry_or_source():
    previous, historical = previous_contract(), target_contract()
    later = studio._AUDITED_REFERENCE_OPTIONAL_LISTENING_UPDATES[previous["computation_sha256"]]
    assert later["target_computation_sha256"] != historical["computation_sha256"]
    assert studio._audited_reference_candidate_update(previous, historical) is None
    assert studio._audited_reference_candidate_update(previous, studio._execution_contract()) is None


def test_preparation_migration_does_not_accept_downstream_or_frozen_work(review):
    mgr, project_id, _ = review
    original = mgr._load(project_id)
    for changes in ({"status": "completed"}, {"inputs": {"reference_voice_review": {"path": "frozen", "sha256": "0" * 64}}}):
        state = deepcopy(original)
        state.update(changes)
        assert not mgr._can_update_reference_candidates(state)
    state = deepcopy(original)
    state["index"] = next(i for i, row in enumerate(state["plan"]) if row["id"] == "positives")
    assert not mgr._can_update_reference_candidates(state)
