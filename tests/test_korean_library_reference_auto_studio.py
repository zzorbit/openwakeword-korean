"""Automatic reference checks leave pronunciation review and stored audio intact."""
from copy import deepcopy
import json
from pathlib import Path
import wave

import pytest

from korean_openwakeword import cli, reference_qc, studio, studio_steps as steps
from korean_openwakeword.common import digest, identity, read_json, write_immutable_rows, write_json
from korean_openwakeword.project import create_project
from test_korean_library_studio import create, http_server, manager  # noqa: F401


@pytest.fixture
def project(tmp_path):
    settings = steps.preset_settings()
    return create_project(tmp_path / "bora", "보라야", "bora", positive_originals=settings["originals"],
                          augmentation_rounds=settings["rounds"], updates=settings["updates"],
                          generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])


def run(project, stage, settings=None, context=None):
    return steps.run_stage(project, project.parent.parent, settings or steps.preset_settings(), stage, context)


def publish_queue(project, name="reference_queue.jsonl"):
    root = project.parent
    audio = root / "data" / "test-reference.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\xe8\x03\x18\xfc" * 1600)
    queue = root / ".studio" / name
    queue.parent.mkdir(exist_ok=True)
    write_immutable_rows(queue, [{"sample_id": "fixture", "voice_family_id": "fixture",
        "path": str(audio.relative_to(root)), "sha256": digest(audio), "split": "train"}])
    return queue


@pytest.mark.parametrize("preset", ["check", "trial", "production"])
def test_reference_policy_automates_only_reference_gate_for_all_budgets(preset):
    settings = steps.preset_settings(preset)
    assert settings["reference_admission_policy"] == "automatic_technical_v1"
    plan = {stage["id"]: stage for stage in steps.workflow_plan(settings)}
    assert plan["references_review"]["kind"] == "task"
    assert plan["references_review"]["label"] == "기준 목소리 자동 검사"
    assert plan["pilot_review"]["kind"] == "review"
    assert settings["final_locked"] is True
    settings.pop("reference_admission_policy")
    legacy = {stage["id"]: stage for stage in steps.workflow_plan(settings)}
    assert legacy["references_review"]["kind"] == "review"
    assert legacy["pilot_review"]["kind"] == "review"


def test_reference_stage_uses_technical_receipt_without_fabricating_answers(project, monkeypatch):
    root = project.parent
    before = (root / "approval.json").read_bytes()
    receipt, journal = root / "review/qc.json", root / "review/qc.jsonl"
    write_json(receipt, {"state": "accepted"})
    write_immutable_rows(journal, [{"passed": True}])
    checked = {"state": "accepted", "checked": 50, "passed": 50,
        "receipt": {"path": str(receipt.relative_to(root)), "sha256": digest(receipt)},
        "quality_journal": {"path": str(journal.relative_to(root)), "sha256": digest(journal)}}
    calls = []

    def check(path, counts, *, queues):
        calls.append((path, counts, queues))
        return checked

    monkeypatch.setattr(reference_qc, "admit_references", check)
    queue = root / ".studio" / "already-published.jsonl"
    result = run(project, "references_review", context={"results": {
        "references": {"result": {"queues": [str(queue)]}}}})
    assert calls == [(project, {"train": 20, "validation": 30}, [str(queue)])]
    assert result["state"] == "completed" and result["result"] == checked
    assert set(result["artifacts"]) == {str(receipt), str(journal)}
    assert "reviews" not in result
    assert (root / "approval.json").read_bytes() == before
    assert not (root / "review/references_results.json").exists()


def test_failed_reference_quality_stops_before_pilot_without_listening_approval(project, monkeypatch):
    quality = {"state": "blocked", "checked": 50, "failed": 1, "problems": ["near_silence"]}
    monkeypatch.setattr(reference_qc, "admit_references", lambda *a, **kw: quality)
    before = (project.parent / "approval.json").read_bytes()
    result = run(project, "references_review")
    assert result["state"] == "waiting_input" and result["input_kind"] == "reference_quality"
    assert result["quality"] == quality and "reviews" not in result
    assert (project.parent / "approval.json").read_bytes() == before
    assert not (project.parent / "data/pilot").exists()


def test_legacy_reference_policy_still_requires_saved_human_review(project, monkeypatch):
    settings = steps.preset_settings()
    settings.pop("reference_admission_policy")
    publish_queue(project)
    monkeypatch.setattr(reference_qc, "admit_references", lambda *a, **kw: pytest.fail("Legacy gate must remain human"))
    result = run(project, "references_review", settings)
    assert result["state"] == "waiting_review"
    assert result["reviews"][0]["kind"] == "references"
    assert read_json(project.parent / "approval.json")["generation_policy_approved"] is False


def test_technical_reference_success_does_not_skip_pilot_pronunciation_gate(project):
    queue = publish_queue(project, "pilot_queue.jsonl")
    result = run(project, "pilot_review", context={"results": {
        "pilot": {"result": {"queue": str(queue)}},
        "references_review": {"state": "completed", "result": {"state": "accepted", "passed": 50}}}})
    assert result["state"] == "waiting_review" and result["reviews"][0]["kind"] == "pilot"
    assert read_json(project.parent / "approval.json")["generation_policy_approved"] is False


def populate_preview(manager, project_id):
    state = manager._load(project_id)
    root = Path(state["project"]).parent
    config, bank = read_json(state["project"]), read_json(root / "voices.json")
    rows = []
    for split in ("train", "validation"):
        family = next(row for row in bank["families"] if row["split"] == split)
        audio = root / "data/references" / split / (family["family_id"] + ".wav")
        audio.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\xe8\x03\x18\xfc" * 1600)
        row = {"sample_id": family["family_id"], "voice_family_id": family["family_id"], "split": split,
               "path": str(audio.relative_to(root)), "sha256": digest(audio), "text": config["tts"]["reference_text"],
               "provenance": "qwen_synthetic_reference", "human_reference_used": False}
        write_immutable_rows(audio.parent / "references.jsonl", [row])
        rows.append(row)
    return state, root, rows


def test_optional_preview_http_reads_audio_without_decisions_or_workflow_writes(manager, http_server):
    view = create(manager)
    state, root, rows = populate_preview(manager, view["id"])
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    preview = manager.describe(state)["reference_preview"]
    assert {key: preview[key] for key in ("count", "train", "validation")} == {"count": 2, "train": 1, "validation": 1}
    status, body, _ = http_server(preview["url"])
    assert status == 200
    html = body.decode()
    assert "기준 목소리 듣기" in html and 'preload="none"' in html
    assert "/review/save" not in html and "POST" not in html
    query = f"?project={view['id']}"
    status, body, _ = http_server("/reference-preview/queue" + query)
    assert status == 200
    items = json.loads(body)["items"]
    assert [item["group"] for item in items] == ["학습용", "공통 평가용"]
    assert not any("decision" in item or "listened" in item for item in items)
    for item, row in zip(items, rows):
        status, audio, _ = http_server(item["audio"])
        assert status == 200 and audio == (root / row["path"]).read_bytes()
    assert {str(path): digest(path) for path in root.rglob("*") if path.is_file()} == before
    assert not manager.workers.items
    assert http_server("/reference-preview/save" + query, "POST", {"decisions": {}},
                       headers={"X-Koww-Token": manager.token})[0] == 400
    assert {str(path): digest(path) for path in root.rglob("*") if path.is_file()} == before


def test_optional_preview_rejects_changed_audio_and_invalid_index(manager, http_server):
    view = create(manager)
    _, root, rows = populate_preview(manager, view["id"])
    query = f"?project={view['id']}"
    assert http_server("/reference-preview/audio/-1" + query)[0] == 400
    assert http_server("/reference-preview/audio/2" + query)[0] == 400
    (root / rows[0]["path"]).write_bytes(b"changed")
    assert http_server("/reference-preview/audio/0" + query)[0] == 400


@pytest.mark.parametrize("flag", ["--train-count", "--validation-count"])
def test_reference_cli_rejects_explicit_zero_instead_of_omitting_selected_split(project, monkeypatch, flag):
    monkeypatch.setattr(reference_qc, "admit_references", lambda *a, **kw: pytest.fail("Zero cannot silently reduce scope"))
    args = cli.parser().parse_args(["check-references", "--project", str(project), flag, "0"])
    with pytest.raises(ValueError, match="1 이상"):
        cli.dispatch(args)


def test_reference_admission_audit_requires_exact_source_file_stage_and_evidence_pins(monkeypatch):
    previous = {"schema_version": 1, "files": {"studio_steps.py": "old", "tts.py": "unchanged"},
                "stages": {"references_review": "old-stage", "pilot_review": "same-stage"}}
    current = {"schema_version": 1, "files": {"studio_steps.py": "new", "reference_qc.py": "new", "tts.py": "unchanged"},
               "stages": {"references_review": "new-stage", "pilot_review": "same-stage"}}
    for contract in (previous, current):
        contract["computation_sha256"] = identity(contract["files"])
    audit = {"changed_files": ["reference_qc.py", "studio_steps.py"],
             "source_stages_sha256": identity(previous["stages"]),
             "target_stages_sha256": identity(current["stages"]),
             "target_computation_sha256": current["computation_sha256"], "evidence_sha256": "a" * 64}
    monkeypatch.setattr(studio, "_AUDITED_REFERENCE_ADMISSION_UPDATES", {previous["computation_sha256"]: audit})
    assert studio._audited_reference_admission_update(previous, current)["changed_files"] == audit["changed_files"]
    changed_source = deepcopy(previous)
    changed_source["files"]["studio_steps.py"] = "wrong source file"
    assert studio._audited_reference_admission_update(changed_source, current) is None
    changed_source = deepcopy(previous)
    changed_source["stages"]["references_review"] = "wrong source stage"
    assert studio._audited_reference_admission_update(changed_source, current) is None
    changed = deepcopy(current)
    changed["files"]["tts.py"] = "unreviewed generation change"
    changed["computation_sha256"] = identity(changed["files"])
    audit["target_computation_sha256"] = changed["computation_sha256"]
    audit["changed_files"].append("tts.py")
    assert studio._audited_reference_admission_update(previous, changed) is None
    audit["changed_files"].remove("tts.py")
    audit["target_computation_sha256"] = current["computation_sha256"]
    changed = deepcopy(current)
    changed["stages"]["pilot_review"] = "unreviewed gate change"
    assert studio._audited_reference_admission_update(previous, changed) is None
    audit["evidence_sha256"] = "unfinished"
    assert studio._audited_reference_admission_update(previous, current) is None


def old_reference_workflow(manager):
    view = create(manager)
    state = manager._load(view["id"])
    state["settings"].pop("reference_admission_policy")
    state["plan"] = steps.workflow_plan(state["settings"])
    state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "references_review")
    state.update(status="waiting_review", auto_continue=True,
                 inputs={"review_answers": {"references": {"saved": "preserved"}}},
                 results={"references_review": {"state": "waiting_review", "reviews": []}})
    state["execution_contract"].update(settings_sha256=identity(state["settings"]), plan_sha256=identity(state["plan"]))
    manager._save(state)
    return view, state


def test_audited_reference_migration_preserves_bank_answers_and_pauses_before_automatic_check(manager, monkeypatch):
    view, state = old_reference_workflow(manager)
    root = Path(state["project"]).parent
    before = {name: digest(root / name) for name in ("project.json", "voices.json", "approval.json")}
    manager._implementation_id = "automatic-reference-admission-release"
    manager._execution_contract = deepcopy(manager._execution_contract)
    manager._execution_contract["stages"]["references_review"] = "new-technical-check-stage"
    monkeypatch.setattr(studio, "_audited_reference_admission_update", lambda *args: {"kind": "fixture_audit"})
    assert manager.describe(state)["migration"]["state"] == "available"
    manager.migrate(view["id"])
    after = manager._load(view["id"])
    assert after["status"] == "paused" and after["auto_continue"] is False
    assert after["settings"]["reference_admission_policy"] == "automatic_technical_v1"
    assert after["plan"][after["index"]]["id"] == "references_review"
    assert after["plan"][after["index"]]["kind"] == "task"
    assert next(stage for stage in after["plan"] if stage["id"] == "pilot_review")["kind"] == "review"
    assert after["inputs"] == state["inputs"] and after["results"] == state["results"]
    assert all(digest(root / name) == sha for name, sha in before.items())
    assert not manager.workers.items
    assert read_json(after["migrations"][-1]["path"])["audited_reference_admission_update"]["kind"] == "fixture_audit"
    manager.migrate(view["id"])
    assert len(manager._load(view["id"])["migrations"]) == 1


@pytest.mark.parametrize("freeze", ["pilot", "generated", "trained", "final", "approved", "past_pilot", "live_worker"])
def test_reference_migration_never_reinterprets_generated_calls_final_or_live_work(manager, monkeypatch, freeze):
    _, state = old_reference_workflow(manager)
    root = Path(state["project"]).parent
    if freeze in ("pilot", "generated", "trained"):
        folder = {"pilot": "data/pilot", "generated": "data/generated", "trained": "runs"}[freeze]
        path = root / folder / "already-created.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve")
    elif freeze == "final":
        write_json(root / "final_lock.json", {"locked": True})
    elif freeze == "approved":
        approval = read_json(root / "approval.json")
        approval["generation_policy_approved"] = True
        write_json(root / "approval.json", approval)
    elif freeze == "past_pilot":
        state["index"] = next(i for i, stage in enumerate(state["plan"]) if stage["id"] == "positives")
    else:
        state["job"] = {"pid": 123, "creation_identity": "live"}
        monkeypatch.setattr(manager, "_job_alive", lambda job: bool(job))
    assert manager._can_update_reference_admission(state) is False
    manager._implementation_id = "automatic-reference-admission-release"
    manager._execution_contract = deepcopy(manager._execution_contract)
    manager._execution_contract["stages"]["references_review"] = "new-technical-check-stage"
    # Even if only later stages remain, require the audited admission change
    # to be checked rather than treating it as an unrelated UI-only update.
    manager._execution_contract["stages"]["results"] = "release-stage-fingerprint"
    monkeypatch.setattr(studio, "_audited_reference_admission_update", lambda *args: {"kind": "fixture_audit"})
    manager._save(state)
    before = manager._path(state["id"]).read_bytes()
    assert manager._migration_view(state)["state"] == "blocked"
    with pytest.raises(ValueError):
        manager.migrate(state["id"])
    assert manager._path(state["id"]).read_bytes() == before
    assert not manager.workers.items
