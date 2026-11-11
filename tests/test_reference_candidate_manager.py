"""Candidate Studio integration: synthetic fixtures, resident fake worker, no GPU."""
import io
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from korean_openwakeword import clone_reference, reference_candidates as candidates, studio, studio_worker
from korean_openwakeword.common import bound_file, digest, identity, read_json, write_json
from test_korean_library_studio import create, http_server, manager


class Resident:
    def __init__(self, pid):
        self.pid, self.stdin, self.returncode = pid, io.BytesIO(), None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


@pytest.fixture
def review(manager, monkeypatch):
    model = create(manager, reference_voice_review=True, navigation_checkpoints=True)
    state = manager._load(model["id"])
    state["settings"].pop("reference_selection_policy", None)  # Preserve the existing-source v2 fixture.
    state["settings"].pop("reference_preparation_policy", None)
    root = Path(state["project"]).parent
    families = read_json(root / "voices.json")["families"]
    config = read_json(state["project"])
    from korean_openwakeword.tts import REFERENCE_TEXT
    for split in ("train", "validation"):
        rows = []
        chosen = [family for family in families if family["split"] == split][:state["settings"][f"{split}_references"]]
        for number, family in enumerate(chosen):
            target = root / "data/references" / split / f"{number}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"synthetic original {family['family_id']}".encode())
            rows.append({"sample_id": family["family_id"], "voice_family_id": family["family_id"],
                         "path": target.relative_to(root).as_posix(), "sha256": digest(target),
                         "split": split, "provenance": "qwen_synthetic_reference", "human_reference_used": False,
                         "text": config["tts"].get("reference_text", REFERENCE_TEXT), "model": config["tts"]["design_model"]})
        (root / "data/references" / split / "references.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    state.update(index=next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot"),
                 status="paused", auto_continue=False,
                 navigation_checkpoint={"policy": "explicit_story_v1", "step": "voices", "state": "awaiting_next"})
    state["settings"]["pilot_review_policy"] = studio.PREPOST_PILOT_POLICY
    state["execution_contract"].update(settings_sha256=identity(state["settings"]), plan_sha256=identity(state["plan"]))
    manager._save(state)
    install_resident(manager, monkeypatch)
    return manager, model["id"], root


def install_resident(manager, monkeypatch):
    residents = []
    launch_stage = manager._popen
    def popen(args, **kwargs):
        if args[-1] != "--reference-review-worker":
            return launch_stage(args, **kwargs)
        worker = Resident(51000 + len(residents))
        residents.append(worker)
        manager.workers.identities[worker.pid] = f"resident-{worker.pid}"
        return worker
    monkeypatch.setattr(manager, "_popen", popen)
    monkeypatch.setattr(studio, "_implementation", lambda: manager._implementation_id)
    import korean_openwakeword.tts as tts
    monkeypatch.setattr(tts, "_seed", lambda seed: None)
    class Model:
        counter = 0
        def generate_voice_design(self, **kwargs):
            self.counter += 1
            rate = 16000
            wave = .15 * np.sin(2 * np.pi * (200 + self.counter * 37) * np.arange(rate) / rate)
            return [wave], rate
    fake_model = Model()
    manager.fake_model = fake_model
    manager.residents = residents
    manager.finish_candidates = lambda: studio_worker.execute(
        json.loads(residents[-1].stdin.getvalue().splitlines()[-1]),
        reference_review=True, load_model=lambda *args: fake_model)


@pytest.fixture
def planned_review(manager, monkeypatch):
    model = create(manager, reference_voice_review=True, navigation_checkpoints=True)
    state = manager._load(model["id"])
    state["settings"].pop("reference_preparation_policy", None)  # Existing metadata-only v3 compatibility fixture.
    state["execution_contract"]["settings_sha256"] = identity(state["settings"])
    manager._save(state)
    root = Path(state["project"]).parent
    assert state["settings"]["reference_selection_policy"] == "candidate_pool_v3"
    assert not list((root / "data/references").rglob("*.wav"))
    install_resident(manager, monkeypatch)
    return manager, model["id"], root


def send(service, project_id, action, row=None, **values):
    if row is not None:
        values.update(id=row["id"], sha256=row["sha256"])
    return service.reference_preview_queue(project_id, {"action": action, **values})


def test_get_is_idle_then_more_allows_zero_selection_and_reuses_worker(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    assert view["mode"] == "candidate_pool_v2" and view["editable"]
    assert len(view["groups"]) == 20 and view["groups"][0]["target"] == 2
    assert view["groups"][10]["target"] == 3 and not view["items"] and not service.residents
    group = view["active_group"]
    send(service, project_id, "more", group=group)
    assert read_json(root / ".studio/reference-candidates/state.json")["job"]["status"] == "running"
    assert service.finish_candidates() == 0
    first = service.reference_preview_queue(project_id)
    assert len(first["items"]) == 5 and not any(row["heard"] for row in first["items"])
    send(service, project_id, "more", group=group)
    assert service.finish_candidates() == 0
    second = service.reference_preview_queue(project_id)
    assert len(service.residents) == 1
    assert {row["id"] for row in first["items"]}.isdisjoint(row["id"] for row in second["items"])


def test_selection_persists_across_batches_caps_and_preserves_listening_facts(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    service.finish_candidates()
    first = service.reference_preview_queue(project_id)
    row = first["items"][0]
    selected_unheard = send(service, project_id, "toggle", row)
    assert selected_unheard["accepted"][0]["selected"] and not selected_unheard["accepted"][0]["heard"]
    send(service, project_id, "toggle", row)
    send(service, project_id, "hear", row)
    send(service, project_id, "toggle", row)
    send(service, project_id, "more", group=first["active_group"])
    service.finish_candidates()
    second = service.reference_preview_queue(project_id)
    assert [entry["id"] for entry in second["accepted"]] == [row["id"]]
    assert second["accepted"][0]["audio"]
    for other in second["items"][:2]:
        send(service, project_id, "hear", other)
    send(service, project_id, "toggle", second["items"][0])
    with pytest.raises(ValueError, match="최대"):
        send(service, project_id, "toggle", second["items"][1])
    with pytest.raises(ValueError, match="수량"):
        send(service, project_id, "more", group=first["active_group"])
    next_view = send(service, project_id, "approve", group=first["active_group"])
    assert next_view["active_group"] != first["active_group"] and not next_view["review_complete"]
    assert "reference_voice_review" not in service._load(project_id)["inputs"]
    assert service.reference_candidate_audio(project_id, row["id"], row["sha256"]).is_file()
    with pytest.raises(ValueError, match="바뀌"):
        service.reference_candidate_audio(project_id, row["id"], "f" * 64)


def test_candidate_job_blocks_other_project_and_final_next_gate(review, monkeypatch):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    state = service._load(project_id)
    assert service._reference_review_job_alive(state)
    other = create(service, wakeword="다음아")
    with pytest.raises(ValueError, match="기준 목소리"):
        service.start(other["id"])
    service.finish_candidates()
    service.reference_preview_queue(project_id)
    monkeypatch.setattr(service, "_navigation_checkpoint_transition", lambda state: None)
    with pytest.raises(ValueError, match="기준 목소리"):
        service.start(project_id)


def test_interrupted_request_preserves_accepted_and_can_generate_again(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    worker = service.residents[-1]
    worker.returncode = 1
    service.workers.identities.pop(worker.pid)
    failed = service.reference_preview_queue(project_id)
    assert failed["job"]["status"] == "failed" and failed["items"] == []
    send(service, project_id, "more", group=view["active_group"])
    assert len(service.residents) == 2
    service.finish_candidates()
    assert len(service.reference_preview_queue(project_id)["items"]) == 5


def test_all_twenty_groups_freeze_only_selected_and_preserve_originals(review, monkeypatch):
    service, project_id, root = review
    _, originals = service.reference_preview_rows(project_id, include_design=True)
    original_bytes = {row["path"]: (root / row["path"]).read_bytes() for row in originals}
    import korean_openwakeword.reference_qc as reference_qc
    monkeypatch.setattr(reference_qc, "admitted_reference_rows", lambda *args: {row["voice_family_id"]: row["sha256"] for row in originals})
    view = service.reference_preview_queue(project_id)
    for group in view["groups"]:
        assert view["active_group"] == group["key"]
        send(service, project_id, "more", group=group["key"])
        assert service.finish_candidates() == 0
        view = service.reference_preview_queue(project_id)
        for row in view["items"][:group["target"]]:
            send(service, project_id, "hear", row)
            send(service, project_id, "toggle", row)
        view = send(service, project_id, "approve", group=group["key"])
    assert view["review_complete"] and not view["editable"]
    frozen_state = service._load(project_id)
    binding = frozen_state["inputs"]["reference_voice_review"]
    manifest = read_json(bound_file(root, binding))
    assert manifest["kind"] == "reference_voice_review_v2" and len(manifest["replacements"]) == 50
    from korean_openwakeword.reference_voice_review import overlay_reference_banks
    banks = {split: [row for row in originals if row["split"] == split] for split in ("train", "validation")}
    effective = overlay_reference_banks(root, banks, binding)
    assert all(row["path"].startswith(".studio/reference-candidates/jobs/") for bank in effective.values() for row in bank)
    assert all((root / path).read_bytes() == payload for path, payload in original_bytes.items())
    assert len(service.residents) == 1
    with pytest.raises(ValueError, match="변경"):
        send(service, project_id, "more", group=view["active_group"])


def test_all_twenty_groups_can_be_selected_unheard_without_inventing_listening_evidence(review, monkeypatch):
    service, project_id, root = review
    _, originals = service.reference_preview_rows(project_id, include_design=True)
    originals_before = {row["path"]: digest(root / row["path"]) for row in originals}
    import korean_openwakeword.reference_qc as reference_qc
    monkeypatch.setattr(reference_qc, "admitted_reference_rows", lambda *args: {row["voice_family_id"]: row["sha256"] for row in originals})
    view = service.reference_preview_queue(project_id)
    for group in view["groups"]:
        send(service, project_id, "more", group=group["key"])
        assert service.finish_candidates() == 0
        view = service.reference_preview_queue(project_id)
        for row in view["items"][:group["target"]]:
            view = send(service, project_id, "toggle", row)
        assert not any(row["heard"] for row in view["accepted"])
        view = send(service, project_id, "approve", group=group["key"])
    assert view["review_complete"] and not view["editable"]
    state = service._load(project_id)
    manifest = read_json(bound_file(root, state["inputs"]["reference_voice_review"]))
    assert manifest["heard"] == {} and manifest["listening_policy"] == candidates.OPTIONAL_LISTENING_POLICY
    ledger = read_json(root / ".studio/reference-candidates/state.json")
    assert ledger["heard"] == {} and len(ledger["approved"]) == 20
    assert originals_before == {path: digest(root / path) for path in originals_before}
    assert len(service.residents) == 1


def test_completed_receipt_recovers_without_original_resident_handle(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    service.finish_candidates()
    worker = service._reference_worker
    service._reference_worker = None  # HTTP process restart, old worker still alive.
    try:
        view = service.reference_preview_queue(project_id)
        assert len(view["items"]) == 5 and view["job"] is None
    finally:
        service._reference_worker = worker


def test_failed_partial_batch_resumes_same_plan_without_recomputing_valid_audio(review, monkeypatch):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    state_path = root / ".studio/reference-candidates/state.json"
    first_job = read_json(state_path)["job"]
    generate = service.fake_model.generate_voice_design
    calls = []
    def interrupted(**kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            raise RuntimeError("fixture temporary interruption")
        return generate(**kwargs)
    monkeypatch.setattr(service.fake_model, "generate_voice_design", interrupted)
    assert service.finish_candidates() == 1
    # Simulate a hard interruption before the worker publishes its receipt.
    result = Path(first_job["directory"]) / "result.json"
    result.replace(result.with_name("fixture-unpublished-result.json"))
    service.residents[-1].returncode = 1
    service.workers.identities.pop(service.residents[-1].pid)
    failed = service.reference_preview_queue(project_id)
    assert failed["job"]["status"] == "failed"
    audio = Path(first_job["directory"]) / "audio"
    preserved = {path: path.read_bytes() for path in audio.glob("*.wav")}
    assert len(preserved) == 2
    send(service, project_id, "more", group=view["active_group"])
    assert read_json(state_path)["job"]["directory"] == first_job["directory"]
    assert service.finish_candidates() == 0
    ready = service.reference_preview_queue(project_id)
    assert len(ready["items"]) == 5 and len(calls) == 6
    assert all(path.read_bytes() == payload for path, payload in preserved.items())
    assert (Path(first_job["directory"]) / "fixture-unpublished-result.json").is_file()


def test_exhausted_quality_seeds_draw_fresh_plan_instead_of_retry_loop(review):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    path = root / ".studio/reference-candidates/state.json"
    first = read_json(path)["job"]
    request = read_json(Path(first["directory"]) / "request.json")
    write_json(Path(first["directory"]) / "result.json", {"task_id": first["task_id"],
        "request_sha256": first["request_sha256"], "error": "technical attempts exhausted", "retry_same_plan": False})
    assert service.reference_preview_queue(project_id)["job"]["status"] == "failed"
    send(service, project_id, "more", group=view["active_group"])
    second = read_json(path)["job"]
    next_request = read_json(Path(second["directory"]) / "request.json")
    assert first["directory"] != second["directory"]
    assert request["plan"]["batch_id"] != next_request["plan"]["batch_id"]


def test_http_candidate_settings_match_saved_options_actual_qc_seed_without_rewriting(review, http_server, monkeypatch):
    service, project_id, root = review
    # Force the actual generation QC to reject the first raw reference, so the
    # HTTP view must expose the successful retry seed rather than the plan seed.
    check_wave = clone_reference._check_wave
    attempts = []
    def reject_first_attempt(path):
        reasons, metrics = check_wave(path)
        attempts.append(path)
        return (["fixture_qc_reject"] if len(attempts) == 1 else reasons), metrics
    monkeypatch.setattr(clone_reference, "_check_wave", reject_first_attempt)
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    ledger_path = root / ".studio/reference-candidates/state.json"
    request = read_json(Path(read_json(ledger_path)["job"]["directory"]) / "request.json")
    assert service.finish_candidates() == 0
    view = service.reference_preview_queue(project_id)
    first = view["items"][0]
    assert first["seed"] == request["plan"]["candidates"][0]["attempt_seeds"][1]
    assert first["seed"] != request["plan"]["candidates"][0]["seed"]
    send(service, project_id, "hear", first)
    send(service, project_id, "toggle", first)
    before = ledger_path.read_bytes()
    status, body, _ = http_server(f"/reference-preview/queue?project={project_id}")
    assert status == 200
    public = json.loads(body)
    for row in [*public["items"], *public["accepted"]]:
        controls = candidates.OPTIONS["ages"][row["age_group"]]["controls"]
        expected = [{"id": field["id"], "label": field["label"],
                     "value": next(option["label"] for option in controls[field["id"]]
                                   if option["id"] == row["voice_controls"][field["id"]])}
                    for field in candidates.OPTIONS["fields"]]
        assert row["voice_settings"] == expected and len(expected) == 5
    assert public["accepted"][0]["seed"] == first["seed"]
    assert public["items"][0]["seed"] == first["seed"]
    assert ledger_path.read_bytes() == before
    assert "voice_settings" not in read_json(ledger_path)["candidates"][first["id"]]


def test_candidate_settings_missing_unknown_metadata_is_display_only():
    describe = studio.StudioManager._reference_candidate_settings
    assert describe({}) == []
    assert describe({"age_group": "unknown", "voice_controls": {}}) == []
    row = {"age_group": "child", "voice_controls": {"pitch": "unsupported"}}
    before = deepcopy(row)
    settings = describe(row)
    assert len(settings) == 5 and all(entry["value"] == "" for entry in settings)
    assert settings[0]["label"] == "음높이" and row == before


def planned_checkpoint(service, project_id, monkeypatch):
    """Execute metadata stages with every baseline generator forbidden."""
    import korean_openwakeword.tts as tts
    from korean_openwakeword.studio_steps import run_stage
    def forbidden(*args, **kwargs):
        pytest.fail("A merged reference stage must not load/generate a baseline voice bank")
    monkeypatch.setattr(tts, "_load_model", forbidden)
    monkeypatch.setattr(tts, "generate_references", forbidden)
    state = service._load(project_id)
    root = Path(state["project"]).parent
    generated = run_stage(state["project"], service.workspace, state["settings"], "references")
    assert generated["state"] == "completed"
    context = {"results": {"references": generated}, "inputs": {}}
    checked = run_stage(state["project"], service.workspace, state["settings"], "references_review", context)
    assert checked["state"] == "completed"
    assert not list((root / "data/references").rglob("*.wav"))
    assert not list((root / "data/references").rglob("references.jsonl"))
    state["results"].update(references=generated, references_review=checked)
    state.update(status="paused", auto_continue=False,
                 index=next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "pilot"),
                 navigation_checkpoint={"policy": "explicit_story_v1", "step": "voices", "state": "awaiting_next"})
    service._save(state)
    return state


def test_merged_new_project_has_slot_preview_and_metadata_preparing_without_gpu(planned_review, monkeypatch):
    service, project_id, root = planned_review
    state = service._load(project_id)
    view = service.describe(state)
    assert view["settings"]["reference_selection_policy"] == "candidate_pool_v3"
    preview = view["reference_preview"]
    assert preview["candidate_selection"] and preview["planned_count"] == 50 and preview["count"] == 0
    initial = service.reference_preview_queue(project_id)
    assert initial["mode"] == "candidate_pool_v2" and not initial["editable"]
    assert initial["job"]["phase"] == "preparing" and initial["job"]["status"] == "running"
    assert not service.residents and not (root / ".studio/reference-candidates/state.json").exists()
    with pytest.raises(ValueError, match="변경"):
        send(service, project_id, "more", group=initial["active_group"])
    planned_checkpoint(service, project_id, monkeypatch)
    ready = service.reference_preview_queue(project_id)
    assert ready["editable"] and ready["job"] is None and ready["items"] == []
    send(service, project_id, "more", group=ready["active_group"])
    request = read_json(json.loads(service.residents[-1].stdin.getvalue().splitlines()[-1]))
    assert request["plan"]["originals"] == {} and len(request["plan"]["candidates"]) == 5
    assert service.finish_candidates() == 0
    assert len(service.reference_preview_queue(project_id)["items"]) == 5
    assert service.fake_model.counter == 5
    assert not list((root / "data/references").rglob("*.wav"))


@pytest.mark.parametrize("status", ["error", "interrupted", "waiting_input", "waiting_review"])
def test_merged_metadata_failure_stops_preparing_poll(planned_review, status):
    service, project_id, root = planned_review
    state = service._load(project_id)
    state["status"] = status
    service._save(state)
    view = service.reference_preview_queue(project_id)
    assert not view["editable"] and view["job"]["status"] == "failed"
    assert view["job"]["phase"] == "preparing" and not service.residents


def test_merged_all_groups_publish_only_selected_banks_admission_and_actual_pilot_inputs(planned_review, monkeypatch):
    service, project_id, root = planned_review
    planned_checkpoint(service, project_id, monkeypatch)
    view = service.reference_preview_queue(project_id)
    selected = {}
    groups = view["groups"]
    for position, group in enumerate(groups):
        send(service, project_id, "more", group=group["key"])
        assert service.finish_candidates() == 0
        view = service.reference_preview_queue(project_id)
        for row in view["items"][:group["target"]]:
            send(service, project_id, "hear", row)
            send(service, project_id, "toggle", row)
            selected[row["id"]] = row["sha256"]
        if position == len(groups) - 1:
            save = service._save
            def interrupted_binding(state):
                if state.get("inputs", {}).get("reference_voice_review"):
                    raise RuntimeError("fixture interruption before workflow binding")
                return save(state)
            monkeypatch.setattr(service, "_save", interrupted_binding)
            with pytest.raises(RuntimeError, match="fixture interruption"):
                send(service, project_id, "approve", group=group["key"])
            ledger = read_json(root / ".studio/reference-candidates/state.json")
            assert len(ledger["approved"]) == 20
            assert "reference_voice_review" not in service._load(project_id)["inputs"]
            before = {path: path.read_bytes() for path in (root / "data/references").rglob("references.jsonl")}
            monkeypatch.setattr(service, "_save", save)
            view = service.reference_preview_queue(project_id)
            assert all(path.read_bytes() == data for path, data in before.items())
        else:
            view = send(service, project_id, "approve", group=group["key"])
    assert view["review_complete"] and not view["editable"] and len(selected) == 50
    state = service._load(project_id)
    manifest = read_json(bound_file(root, state["inputs"]["reference_voice_review"]))
    assert manifest["kind"] == "reference_voice_review_v3"
    from korean_openwakeword.reference_qc import admitted_reference_rows
    admitted = admitted_reference_rows(root, read_json(root / "approval.json"))
    assert len(admitted) == 50 and set(admitted.values()) == set(selected.values())
    monkeypatch.setenv("KOWW_STUDIO_REFERENCE_REVIEW", json.dumps(state["inputs"]["reference_voice_review"]))
    from korean_openwakeword.pilot import _preflight_banks
    banks = _preflight_banks(root, state["settings"])
    assert len(banks["train"]) == 20 and len(banks["validation"]) == 30
    assert {row["sha256"] for bank in banks.values() for row in bank} == set(selected.values())
    assert {row["sha256"] for row in banks["train"]}.isdisjoint(row["sha256"] for row in banks["validation"])
    assert service.fake_model.counter == 100  # Twenty batches of five, no hidden baseline generation.
    assert len(service.residents) == 1
    # Inverse old-server window: the frozen workflow input is durable but the
    # ledger still shows a previous selection/approval state. Display uses the
    # bound manifest and leaves the saved ledger untouched.
    ledger_path = root / ".studio/reference-candidates/state.json"
    ledger = read_json(ledger_path)
    last = groups[-1]["key"]
    ledger["approved"].remove(last)
    last_selected = ledger["selected"].pop(last)
    for candidate_id in last_selected:
        ledger["heard"].pop(candidate_id)
    write_json(ledger_path, ledger)
    old_bytes = ledger_path.read_bytes()
    hydrated = service.reference_preview_queue(project_id)
    assert hydrated["review_complete"] and not hydrated["editable"]
    assert all(group["approved"] for group in hydrated["groups"])
    assert hydrated["groups"][-1]["selected_count"] == 3
    assert len(hydrated["accepted"]) == 3 and all(row["heard"] and row["selected"] for row in hydrated["accepted"])
    assert ledger_path.read_bytes() == old_bytes and service.fake_model.counter == 100


def test_exact_merge_migration_preserves_existing_v2_candidates_and_selected_two(review, monkeypatch):
    service, project_id, root = review
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    assert service.finish_candidates() == 0
    view = service.reference_preview_queue(project_id)
    for row in view["items"][:2]:
        send(service, project_id, "hear", row)
        send(service, project_id, "toggle", row)
    ledger_path = root / ".studio/reference-candidates/state.json"
    before = ledger_path.read_bytes()
    state = service._load(project_id)
    original_settings = deepcopy(state["settings"])
    old = deepcopy(state["execution_contract"])
    changed = ["reference_candidates.py", "reference_voice_review.py", "pilot.py", "studio_steps.py"]
    for name in changed:
        old["files"][name] = "b" * 64
    old["computation_sha256"] = identity(old["files"])
    old["stages"] = {stage: "c" * 64 for stage in old["stages"]}
    state.update(implementation_id="pre-merge-v2", execution_contract=old)
    service._save(state)
    current = service._execution_contract
    monkeypatch.setitem(studio._AUDITED_REFERENCE_CANDIDATE_MERGE_UPDATES, old["computation_sha256"], {
        "source_stages_sha256": identity(old["stages"]), "target_computation_sha256": current["computation_sha256"],
        "target_stages_sha256": identity(current["stages"]), "changed_files": changed, "evidence_sha256": "d" * 64})
    assert service._migration_view(state)["audited_reference_candidate_update"]["kind"] == "audited_reference_candidate_merge_update"
    service.migrate(project_id)
    migrated = service._load(project_id)
    assert migrated["settings"] == original_settings and "reference_selection_policy" not in migrated["settings"]
    assert ledger_path.read_bytes() == before
    after = service.reference_preview_queue(project_id)
    assert after["groups"][0]["selected_count"] == 2 and len(after["accepted"]) == 2
    assert {row["id"] for row in after["items"]} == {row["id"] for row in view["items"]}
    assert ledger_path.read_bytes() == before and service.fake_model.counter == 5


def two_candidate_batches(service, project_id):
    view = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    assert service.finish_candidates() == 0
    first = service.reference_preview_queue(project_id)
    send(service, project_id, "more", group=view["active_group"])
    assert service.finish_candidates() == 0
    return first, service.reference_preview_queue(project_id)


def test_candidate_batch_history_get_is_read_only_and_defaults_to_latest_five(review, http_server):
    service, project_id, root = review
    first, second = two_candidate_batches(service, project_id)
    first_id, second_id = first["current_batch"], second["current_batch"]
    assert second["batches"] == [{"id": first_id, "number": 1, "count": 5},
                                 {"id": second_id, "number": 2, "count": 5}]
    ledger_path = root / ".studio/reference-candidates/state.json"
    before = ledger_path.read_bytes()
    workflow = service._path(project_id).read_bytes()
    for expected, batch_id in ((first, first_id), (second, second_id), (first, first_id)):
        status, body, _ = http_server(f"/reference-preview/queue?project={project_id}&batch={batch_id}")
        assert status == 200
        viewed = json.loads(body)
        assert viewed["view_batch"] == batch_id and viewed["current_batch"] == second_id
        assert [row["id"] for row in viewed["items"]] == [row["id"] for row in expected["items"]]
        assert len(viewed["items"]) == 5 and all(row["audio"] and len(row["voice_settings"]) == 5 for row in viewed["items"])
        assert ledger_path.read_bytes() == before and service._path(project_id).read_bytes() == workflow
    latest = service.reference_preview_queue(project_id)
    assert latest["view_batch"] == second_id and [row["id"] for row in latest["items"]] == [row["id"] for row in second["items"]]
    assert service.fake_model.counter == 10 and len(service.residents) == 1
    assert read_json(ledger_path)["current_batch"][latest["active_group"]] == second_id


def test_history_selection_response_retains_view_and_accepted_tray_can_target_other_batch(review, http_server):
    service, project_id, root = review
    first, second = two_candidate_batches(service, project_id)
    first_id, second_id = first["current_batch"], second["current_batch"]
    chosen = first["items"][0]
    headers = {"X-Koww-Token": service.token}
    def post(action, batch_id):
        return http_server(f"/reference-preview/save?project={project_id}&batch={batch_id}", "POST",
            {"action": action, "id": chosen["id"], "sha256": chosen["sha256"]}, headers=headers)
    for action in ("hear", "toggle"):
        status, body, _ = post(action, first_id)
        assert status == 200
        viewed = json.loads(body)
        assert viewed["view_batch"] == first_id and viewed["current_batch"] == second_id
        assert [row["id"] for row in viewed["items"]] == [row["id"] for row in first["items"]]
    assert viewed["items"][0]["heard"] and viewed["items"][0]["selected"]
    assert [row["id"] for row in viewed["accepted"]] == [chosen["id"]]
    # Accepted rows from an older batch remain playable/removable while a newer
    # batch is displayed. The query controls only the returned presentation.
    status, body, _ = post("toggle", second_id)
    assert status == 200
    viewed = json.loads(body)
    assert viewed["view_batch"] == second_id and viewed["accepted"] == []
    assert [row["id"] for row in viewed["items"]] == [row["id"] for row in second["items"]]
    replay = service.reference_preview_queue(project_id, batch_id=first_id)
    assert replay["items"][0]["heard"] and not replay["items"][0]["selected"]
    assert service.fake_model.counter == 10


def test_batch_history_rejects_unknown_other_group_and_generation_action_context_without_writes(review, http_server):
    service, project_id, root = review
    first, second = two_candidate_batches(service, project_id)
    first_id = first["current_batch"]
    ledger_path = root / ".studio/reference-candidates/state.json"
    before = ledger_path.read_bytes()
    assert http_server(f"/reference-preview/queue?project={project_id}&batch=unknown-batch")[0] == 400
    for action in ("more", "approve"):
        assert http_server(f"/reference-preview/save?project={project_id}&batch={first_id}", "POST",
            {"action": action, "group": first["active_group"]}, headers={"X-Koww-Token": service.token})[0] == 400
    assert ledger_path.read_bytes() == before and service.fake_model.counter == 10
    for row in first["items"][:2]:
        send(service, project_id, "hear", row)
        send(service, project_id, "toggle", row)
    next_group = send(service, project_id, "approve", group=first["active_group"])
    assert next_group["batches"] == [] and next_group["view_batch"] is None
    before = ledger_path.read_bytes()
    assert http_server(f"/reference-preview/queue?project={project_id}&batch={first_id}")[0] == 400
    assert ledger_path.read_bytes() == before and service.fake_model.counter == 10


def test_pending_and_failed_partial_batches_are_not_browsable_history(review):
    service, project_id, root = review
    first, second = two_candidate_batches(service, project_id)
    send(service, project_id, "more", group=second["active_group"])
    pending = service.reference_preview_queue(project_id)
    assert len(pending["batches"]) == 2 and pending["items"] == []
    assert pending["view_batch"] == pending["current_batch"]
    assert pending["view_batch"] not in {batch["id"] for batch in pending["batches"]}
    with pytest.raises(ValueError, match="생성이 끝난"):
        service.reference_preview_queue(project_id, batch_id=pending["view_batch"])
    old = service.reference_preview_queue(project_id, batch_id=first["view_batch"])
    assert len(old["items"]) == 5 and old["job"]["status"] == "running"
    service.residents[-1].returncode = 1
    service.workers.identities.pop(service.residents[-1].pid)
    failed = service.reference_preview_queue(project_id)
    assert len(failed["batches"]) == 2 and failed["job"]["status"] == "failed"
    for row in first["items"][:2]:
        send(service, project_id, "hear", row)
        send(service, project_id, "toggle", row)
    advanced = send(service, project_id, "approve", group=first["active_group"])
    assert advanced["active_group"] != first["active_group"] and advanced["job"] is None
    assert advanced["batches"] == []


def test_candidate_audit_requires_exact_contract_and_excludes_frozen_downstream(review, monkeypatch):
    service, project_id, root = review
    state = service._load(project_id)
    assert service._can_update_reference_candidates(state)
    previous = deepcopy(state["execution_contract"])
    previous["files"].pop("reference_candidates.py")
    previous["computation_sha256"] = identity(previous["files"])
    previous["stages"] = {key: "b" * 64 for key in previous["stages"]}
    state.update(implementation_id="old-candidate-source", execution_contract=previous,
                 reference_voice_review_policy="grouped_v1")
    current = service._execution_contract
    changed = sorted(key for key in previous["files"].keys() | current["files"].keys()
                     if previous["files"].get(key) != current["files"].get(key))
    monkeypatch.setitem(studio._AUDITED_REFERENCE_CANDIDATE_UPDATES, previous["computation_sha256"],
        {"source_stages_sha256": identity(previous["stages"]), "target_computation_sha256": current["computation_sha256"],
         "target_stages_sha256": identity(current["stages"]), "changed_files": changed, "evidence_sha256": "a" * 64})
    service._save(state)
    compatibility = service._migration_view(state)
    assert compatibility["audited_reference_candidate_update"]
    service.migrate(project_id)
    migrated = service._load(project_id)
    assert migrated["reference_voice_review_policy"] == "candidate_pool_v2"
    assert migrated.get("history", []) == state.get("history", []) and migrated["inputs"] == state["inputs"]
    receipt = read_json(bound_file(root, migrated["migrations"][-1]))
    assert receipt["audited_reference_candidate_update"]["evidence_sha256"] == "a" * 64
    broken = deepcopy(previous)
    broken["stages"]["pilot"] = "c" * 64
    assert studio._audited_reference_candidate_update(broken, current) is None
    migrated["inputs"]["reference_voice_review"] = {"path": "already.json", "sha256": "f" * 64}
    assert not service._can_update_reference_candidates(migrated)
    migrated["inputs"].pop("reference_voice_review")
    (root / "data/pilot/train").mkdir(parents=True)
    (root / "data/pilot/train/existing.wav").write_bytes(b"preserved")
    assert not service._can_update_reference_candidates(migrated)


def test_completed_legacy_model_is_read_only_and_not_migrated(review):
    service, project_id, root = review
    state = service._load(project_id)
    state.update(status="completed", implementation_id="legacy-completed", reference_voice_review_policy="grouped_v1")
    service._save(state)
    before = service._path(project_id).read_bytes()
    service.migrate(project_id)
    assert service._path(project_id).read_bytes() == before
    assert not service.reference_preview_queue(project_id)["editable"]
    assert not (root / ".studio/reference-candidates/state.json").exists()
