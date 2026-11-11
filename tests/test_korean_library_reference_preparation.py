"""Initial candidate preparation preserves review decisions and skips valid stored audio."""
from copy import deepcopy

import pytest

from korean_openwakeword import clone_reference, reference_candidates as candidates, studio_steps as steps, tts
from korean_openwakeword.common import digest, read_json, write_json
from test_korean_library_reference_planned_candidates import planned  # noqa: F401
from test_korean_library_reference_candidates import pool, batch, choose  # noqa: F401


def test_prepare_all_groups_once_reuses_model_and_never_admits_or_marks_heard(planned):
    root, project, settings, slots, saved, calls, model = planned
    loaded = []
    result = candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: loaded.append(args) or model)
    state = read_json(root / ".studio/reference-candidates/state.json")
    assert result["status"] == "completed" and result["completed"] == 100
    assert result["completed_groups"] == 20 and [row["completed"] for row in result["rows"]] == [50, 50]
    assert len(calls) == 100 and len(loaded) == 1 and len(state["candidates"]) == 100
    assert result["reused_candidates"] == 0 and result["generated_candidates"] == 100
    assert state["selected"] == state["heard"] == {} and state["approved"] == []
    assert candidates.active_candidate_group(state) == "train:man:child"
    assert len(candidates.candidate_review_view(root, state)["items"]) == 5
    assert not (root / "data/references").exists()
    assert "reference_technical_admission" not in read_json(root / "approval.json")
    assert candidates.verify_initial_candidate_preparation(project, slots, result["receipt"])["completed"] == 100
    before = deepcopy(state)
    again = candidates.prepare_initial_candidate_batches(project, slots,
        load_model=lambda *args: pytest.fail("All ready groups must not load weights"))
    assert again["receipt"] == result["receipt"]
    assert again["reused_candidates"] == 100 and again["generated_candidates"] == 0
    assert read_json(root / ".studio/reference-candidates/state.json") == before
    damaged = root / next(iter(state["candidates"].values()))["path"]
    damaged.write_bytes(damaged.read_bytes() + b"changed")
    changed = digest(damaged)
    with pytest.raises(ValueError, match="changed bound file"):
        candidates.prepare_initial_candidate_batches(project, slots,
            load_model=lambda *args: pytest.fail("Corrupt stored evidence must not be regenerated"))
    assert digest(damaged) == changed


def test_legacy_selected_two_and_old_batch_pointer_survive_backfill(pool):
    root, project, config, originals, saved, model = pool
    plan, rows, _ = batch(pool)
    for row in rows[:2]:
        choose(pool, row)
    state_path = root / ".studio/reference-candidates/state.json"
    write_json(state_path, saved)
    protected = {key: deepcopy(saved[key]) for key in ("selected", "heard", "approved", "current_batch")}
    hashes = {row["path"]: digest(root / row["path"]) for row in originals + rows}
    before = len(model.calls)
    result = candidates.prepare_initial_candidate_batches(project, originals, load_model=model.load)
    after = read_json(state_path)
    assert result["completed"] == 100 and len(model.calls) - before == 95
    assert result["reused_candidates"] == 5 and result["generated_candidates"] == 95
    assert all(after[key] == protected[key] for key in ("selected", "heard", "approved"))
    assert all(after["current_batch"][key] == value for key, value in protected["current_batch"].items())
    assert hashes == {path: digest(root / path) for path in hashes}
    assert candidates.candidate_review_view(root, after)["accepted"][0]["candidate_id"] == rows[0]["candidate_id"]


def test_pause_only_between_batches_and_resume_without_new_random_plans(planned):
    root, project, settings, slots, saved, calls, model = planned
    paused = candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model,
        should_pause=lambda: len(calls) >= 5)
    assert paused["status"] == "paused" and paused["completed"] == 5 and len(calls) == 5
    assert candidates.preparation_status(root, verify=False)["status"] == "paused"
    state_path = root / ".studio/reference-candidates/state.json"
    first = read_json(state_path)
    first_plan = first["batches"][first["groups"][0]["key"]][0]
    # Interrupt the next group's inference after two durable candidate records.
    original = model.generate_voice_design
    def interrupt(**kwargs):
        if len(calls) == 7:
            raise RuntimeError("interrupted")
        return original(**kwargs)
    model.generate_voice_design = interrupt
    with pytest.raises(RuntimeError, match="interrupted"):
        candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model)
    partial = read_json(state_path)
    second_plan = partial["batches"][partial["groups"][1]["key"]][0]
    output = candidates._preparation_output(root, partial, second_plan)
    old = {path.name: digest(path) for path in output.glob("*.wav")}
    assert len(old) == 2
    model.generate_voice_design = original
    done = candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model)
    assert done["completed"] == len(calls) == 100
    final = read_json(state_path)
    assert final["batches"][final["groups"][0]["key"]][0] == first_plan
    assert final["batches"][final["groups"][1]["key"]][0] == second_plan
    assert old == {name: digest(output / name) for name in old}


def test_resumes_old_manual_partial_output_and_preserves_its_request(planned):
    root, project, settings, slots, saved, calls, model = planned
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / "old-manual-task" / "audio"
    request = output.parent / "request.json"
    write_json(request, {"kind": candidates.BATCH_KIND, "plan": plan, "implementation_id": "historical"})
    request_bytes = request.read_bytes()
    original = model.generate_voice_design
    def interrupt(**kwargs):
        if len(calls) == 2:
            raise RuntimeError("old worker stopped")
        return original(**kwargs)
    model.generate_voice_design = interrupt
    with pytest.raises(RuntimeError):
        candidates.generate_candidate_batch(project, output, "old-manual-task", plan, [], load_model=lambda *args: model)
    saved["job"] = {"status": "failed", "group": plan["group_key"]}
    write_json(root / ".studio/reference-candidates/state.json", saved)
    existing = {path.name: digest(path) for path in output.glob("*.wav")}
    model.generate_voice_design = original
    result = candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model,
        should_pause=lambda: len(calls) >= 5)
    assert result["status"] == "paused" and result["completed"] == 5 and len(calls) == 5
    assert request.read_bytes() == request_bytes
    assert existing == {name: digest(output / name) for name in existing}
    state = read_json(root / ".studio/reference-candidates/state.json")
    assert state["job"] is None and state["batches"][plan["group_key"]] == [plan]


def test_external_review_mutation_is_detected_and_not_overwritten(planned):
    root, project, settings, slots, saved, calls, model = planned
    path = root / ".studio/reference-candidates/state.json"
    original = model.generate_voice_design
    def mutate(**kwargs):
        if not calls:
            value = read_json(path)
            value["external_marker"] = "keep me"
            write_json(path, value)
        return original(**kwargs)
    model.generate_voice_design = mutate
    with pytest.raises(ValueError, match="외부에서 변경"):
        candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model)
    assert read_json(path)["external_marker"] == "keep me"
    assert not read_json(path)["candidates"]
    assert not (root / "data/references").exists()


def test_terminal_qc_failure_preserved_and_only_explicit_retry_draws_new_seeds(planned, monkeypatch):
    root, project, settings, slots, saved, calls, model = planned
    # Generation QC runs on the preserved raw/prepared reference; candidate QC
    # is a separate final-admission check.
    actual = clone_reference._check_wave
    monkeypatch.setattr(clone_reference, "_check_wave", lambda path: (["near_silence"], actual(path)[1]))
    with pytest.raises(ValueError, match="서로 다른 목소리 후보"):
        candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model)
    assert len(calls) == 4  # Bounded QC attempts; no automatic fresh-plan loop.
    path = root / ".studio/reference-candidates/state.json"
    first = read_json(path)
    key = first["groups"][0]["key"]
    old_plan = deepcopy(first["batches"][key][0])
    phase = read_json(root / ".studio/reference-candidates/preparation/state.json")
    assert phase["status"] == "failed" and old_plan["batch_id"] in phase["terminal_failures"]
    monkeypatch.setattr(clone_reference, "_check_wave", actual)
    result = candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model,
        should_pause=lambda: len(calls) >= 9)
    assert result["status"] == "paused" and len(calls) == 9
    after = read_json(path)
    assert after["batches"][key][0] == old_plan and len(after["batches"][key]) == 2
    new_plan = after["batches"][key][1]
    assert new_plan["batch_id"] != old_plan["batch_id"]
    assert new_plan["candidates"][0]["number"] == 6
    assert after["current_batch"][key] == old_plan["batch_id"]
    assert candidates.preparation_status(root, after)["completed_groups"] == 1


def test_bound_old_manual_terminal_qc_receipt_allows_fresh_preparation_plan(planned, monkeypatch):
    root, project, settings, slots, saved, calls, model = planned
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / "old-terminal-task" / "audio"
    request = output.parent / "request.json"
    write_json(request, {"kind": candidates.BATCH_KIND, "plan": plan})
    actual = clone_reference._check_wave
    monkeypatch.setattr(clone_reference, "_check_wave", lambda path: (["near_silence"], actual(path)[1]))
    with pytest.raises(ValueError) as failed:
        candidates.generate_candidate_batch(project, output, "old-terminal-task", plan, [], load_model=lambda *args: model)
    result_path = output.parent / "result.json"
    write_json(result_path, {"request_sha256": digest(request), "error": str(failed.value)})
    saved["job"] = {"status": "failed", "resumable": False, "group": plan["group_key"],
        "directory": str(output.parent), "task_id": output.parent.name, "request_sha256": digest(request)}
    write_json(root / ".studio/reference-candidates/state.json", saved)
    evidence = {path: path.read_bytes() for path in (request, result_path, output / "plan.json")}
    monkeypatch.setattr(clone_reference, "_check_wave", actual)
    result = candidates.prepare_initial_candidate_batches(project, slots, load_model=lambda *args: model,
        should_pause=lambda: len(calls) >= 9)
    assert result["status"] == "paused" and len(calls) == 9
    assert evidence == {path: path.read_bytes() for path in evidence}
    state = read_json(root / ".studio/reference-candidates/state.json")
    assert len(state["batches"][plan["group_key"]]) == 2 and state["job"] is None


def test_preparation_does_not_relax_manual_future_group_guard(planned):
    root, project, settings, slots, saved, calls, model = planned
    with pytest.raises(ValueError, match="현재 검수"):
        candidates.plan_candidate_batch(project, saved, saved["groups"][1]["key"])


@pytest.mark.parametrize("changed", ["plan", "request_sha"])
def test_tampered_partial_plan_or_bound_request_never_falls_back_to_new_folder(planned, changed):
    root, project, settings, slots, saved, calls, model = planned
    plan = candidates.plan_candidate_batch(project, saved, candidates.active_candidate_group(saved))
    candidates.register_candidate_batch(saved, plan)
    output = candidates.candidate_jobs(root) / "old-partial-task" / "audio"
    write_json(output / "plan.json", plan)
    request = output.parent / "request.json"
    write_json(request, {"kind": candidates.BATCH_KIND, "plan": plan})
    saved["job"] = {"status": "failed", "group": plan["group_key"], "directory": str(output.parent),
                    "task_id": output.parent.name, "request_sha256": digest(request)}
    state_path = root / ".studio/reference-candidates/state.json"
    write_json(state_path, saved)
    if changed == "plan":
        altered = deepcopy(plan)
        altered["candidates"][0]["seed"] += 1
        write_json(output / "plan.json", altered)
    else:
        write_json(request, {"kind": candidates.BATCH_KIND, "plan": plan, "changed": True})
    before = {path: path.read_bytes() for path in (output / "plan.json", request, state_path)}
    with pytest.raises(ValueError):
        candidates.prepare_initial_candidate_batches(project, slots,
            load_model=lambda *args: pytest.fail("Evidence changes must fail before loading weights"))
    assert not calls and before == {path: path.read_bytes() for path in before}
    assert not (candidates.candidate_jobs(root) / ("initial_" + plan["batch_id"])).exists()


def test_new_references_stage_prepares_and_verifies_while_legacy_metadata_policy_stays(planned, monkeypatch):
    root, project, _, slots, saved, calls, model = planned
    config = read_json(project)
    monkeypatch.setattr(steps, "load_project", lambda project: (root, config))
    monkeypatch.setattr(tts, "_load_model", lambda *args: model)
    settings = {**steps.preset_settings(), "reference_selection_policy": "candidate_pool_v3",
                "reference_preparation_policy": "initial_candidate_batches_v1",
                "pilot_review_policy": steps.PREPOST_PILOT_POLICY}
    config.update(batch=steps.BATCH, targets={"positive_originals": settings["originals"],
        "generic_hours": settings["generic_hours"], "korean_hours": settings["korean_hours"]},
        augmentation_rounds=settings["rounds"], training={"updates": settings["updates"]})
    ready = steps.run_stage(project, root / ".studio", settings, "references")
    assert ready["state"] == "completed" and ready["result"]["preparation"]["completed"] == 100
    checked = steps.run_stage(project, root / ".studio", settings, "references_review",
        {"results": {"references": ready}})
    assert checked["state"] == "completed" and len(calls) == 100
    assert [step["title"] for step in steps.workflow_plan(settings) if step["id"] == "references"] == ["기준 목소리 준비"]
    settings.pop("reference_preparation_policy")
    metadata = steps.run_stage(project, root / ".studio", settings, "references")
    assert "preparation" not in metadata["result"] and len(calls) == 100
