"""Local Studio orchestration/HTTP tests: synthetic fixtures, no downloads or GPU."""
import http.client
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import wave
from urllib.parse import urljoin

import pytest

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity, now, read_json, write_json
from korean_openwakeword.project import create_project, load_project


def page_config(page):
    return json.loads(re.search(r'<script id="koww-page-config" type="application/json">(.*?)</script>', page).group(1))


class Worker:
    def __init__(self, pid, request, identities):
        self.pid, self.request_path, self.identities = pid, Path(request), identities
        self.returncode = None

    def poll(self):
        return self.returncode

    def finish(self, output=None, *, error=None, **overrides):
        request = read_json(self.request_path)
        result = {"task_id": request["task_id"], "request_sha256": digest(self.request_path),
                  "config_sha256": request["config_sha256"], "result_config_sha256": digest(request["project"]),
                  "result": output or {"state": "completed"}, "completed_at": now()}
        if error:
            result["error"] = error
        result.update(overrides)
        write_json(request["result_path"], result)
        self.returncode = 1 if error else 0
        self.identities.pop(self.pid, None)
        return Path(request["result_path"])


class Workers:
    def __init__(self):
        self.items, self.identities, self.calls = [], {}, []

    def __call__(self, args, **kwargs):
        assert args[1:3] == ["-m", "korean_openwakeword.studio_worker"]
        self.calls.append((args, kwargs))
        worker = Worker(41000 + len(self.items), args[-1], self.identities)
        self.items.append(worker)
        self.identities[worker.pid] = "birth-" + str(worker.pid)
        return worker


@pytest.fixture
def manager(tmp_path, monkeypatch):
    workers = Workers()
    monkeypatch.setattr(studio, "process_identity", lambda pid: workers.identities.get(pid))
    service = studio.StudioManager(tmp_path / "workspace", popen=workers)
    service.workers = workers
    yield service
    service.close()


def create(manager, **kwargs):
    return manager.create({"wakeword": "보라야", "preset": "check", **kwargs})


def put_state(manager, project_id, **changes):
    state = manager._load(project_id)
    state.update(changes)
    manager._save(state)
    return state


def reference_queue(manager, project_id, *, kind="references"):
    state = manager._load(project_id)
    root, config = load_project(state["project"])
    audio = root / "data" / "fixture.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 1600)
    queue = root / "review" / "fixture.jsonl"
    row = {"sample_id": "fixture-1", "path": str(audio.relative_to(root)), "sha256": digest(audio),
           "text": "검수 프로토콜용 합성 시험 자료", "voice_family_id": "fixture-family",
           "generation_method": "clone", "tts_settings_id": identity(config["tts"])}
    queue.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    descriptor = {"id": "fixture", "kind": kind, "queue": str(queue), "queue_sha256": digest(queue)}
    put_state(manager, project_id, status="waiting_review", index=0,
              plan=[{"id": "fixture_review", "label": "시험 검수", "kind": "review"}],
              results={"fixture_review": {"state": "waiting_review", "reviews": [descriptor]}}, auto_continue=False)
    return audio, queue


@pytest.mark.parametrize("preset,originals,rounds,updates,korean", [
    ("check", 1000, 1, 10000, 1), ("trial", 20000, 3, 50000, 2000 / 3),
    ("production", 100000, 5, 100000, 1500)])
def test_create_presets_only_configure_without_work(manager, preset, originals, rounds, updates, korean):
    view = create(manager, preset=preset)
    state = manager._load(view["id"])
    _, config = load_project(state["project"])
    assert config["targets"]["positive_originals"] == originals
    assert config["augmentation_rounds"] == rounds
    assert config["training"]["updates"] == updates
    assert config["targets"]["korean_hours"] == pytest.approx(korean)
    assert config["audio_storage_policy"] == "canonical_flac_only_v1"
    assert "common_download_policy" not in config
    assert "common_download_policy" not in state["settings"]
    assert view["status"] == "ready" and not manager.workers.items
    assert not (Path(state["project"]).parent / "final_lock.json").exists()



def test_new_official_example_selection_policy_is_frozen_but_not_added_to_legacy_resume(manager):
    view = create(manager)
    state = manager._load(view["id"])
    assert state["settings"]["official_noise_acquisition_policy"] == "selective_fma_small_v1"
    assert read_json(state["project"])["official_noise_acquisition_policy"] == "selective_fma_small_v1"
    registered = manager.create({"wakeword": "다음아", "preset": "check"})
    legacy_state = manager._load(registered["id"])
    legacy = Path(legacy_state["project"])
    config = read_json(legacy)
    config.pop("official_noise_acquisition_policy")
    write_json(legacy, config)
    legacy_state["settings"].pop("official_noise_acquisition_policy")
    manager._save(legacy_state)
    before = digest(legacy)
    attached = manager.create({"wakeword": "다음아", "preset": "check"})
    attached_state = manager._load(attached["id"])
    assert "official_noise_acquisition_policy" not in attached_state["settings"]
    assert digest(legacy) == before


def test_retired_common_transport_is_preserved_on_resume_without_rewriting_config(manager):
    registered = manager.create({"wakeword": "다음아", "preset": "check"})
    saved_state = manager._load(registered["id"])
    project = Path(saved_state["project"])
    config = read_json(project)
    config["common_download_policy"] = "selected_common_zip_v1"
    write_json(project, config)
    saved_state["settings"]["common_download_policy"] = "selected_common_zip_v1"
    manager._save(saved_state)
    before = project.read_bytes()
    view = manager.create({"wakeword": "다음아", "preset": "check"})
    saved = manager._load(view["id"])
    assert saved["settings"]["common_download_policy"] == "selected_common_zip_v1"
    assert view["status"] == "ready"
    assert project.read_bytes() == before


def test_automatic_training_view_separates_targets_candidates_admissions_and_nominal_features(manager):
    view = create(manager)
    state = manager._load(view["id"])
    assert state["settings"]["material_acquisition_policy"] == "automatic_v3"
    state["results"].update({
        "data_download": {"result": {"candidate_unique_hours": {"generic_negative": 0, "korean_negative": 1.1},
                                    "generic_feature_import": {"publisher_declared_hours": 2000},
                                    "automatic_source_preparation": {"storage_policy": "canonical_flac_only_v1",
                                        "canonical_storage_bytes": 4096, "source_storage_bytes": 8192,
                                        "retired_original_bytes": 16384, "acquired_original_peak_bytes": 32768}}},
        "data_connect": {"result": {"admitted_unique_hours": {"generic_negative": 0, "korean_negative": 1}}}})
    actual = manager.describe(state)["training_materials"]
    assert actual["targets"] == {"generic_negative": 2000, "korean_negative": 1}
    assert actual["candidate"] == {"generic_negative": 0, "korean_negative": 1.1}
    assert actual["admitted"] == {"generic_negative": 0, "korean_negative": 1}
    assert actual["generic_publisher_nominal_hours"] == 2000
    assert actual["storage"]["canonical_storage_bytes"] == 4096
    assert actual["storage"]["source_storage_bytes"] == 8192
    state["settings"].pop("material_acquisition_policy")
    assert "training_materials" not in manager.describe(state)


def test_recipes_preview_has_three_budgets_known_sources_and_no_network(manager):
    recipes = manager.recipes()
    assert [row["id"] for row in recipes["presets"]] == ["check", "trial", "production"]
    assert all(row["known_download_bytes"] > 0 for row in recipes["presets"])
    assert all(row["assets"] or row.get("automatic_source_plan", {}).get("components")
               for row in recipes["presets"])
    assert all(row["material_acquisition_policy"] == "automatic_v3" for row in recipes["presets"])
    assert all(row["human_review_required"] is False for row in recipes["presets"])
    assert all(row["custom_material_human_review_required"] is True for row in recipes["presets"])
    assert all(row["material_review_policy"] == "library_fixed_sources_v1" for row in recipes["presets"])
    policies = [row["evaluation_policy"] for row in recipes["presets"]]
    assert all(policy == policies[0] for policy in policies)
    assert policies[0]["validation_originals"] == 5000
    assert manager.workers.calls == []


def test_registered_v2_workflow_keeps_its_frozen_protocol_when_opened_after_upgrade(manager):
    from korean_openwakeword.studio_steps import common_evaluation_policy
    view = create(manager)
    state = manager._load(view["id"])
    state["settings"]["evaluation_policy"] = common_evaluation_policy(2)
    manager._save(state)
    path = manager._path(view["id"])
    original = path.read_bytes()
    reopened = create(manager)
    assert reopened["id"] == view["id"]
    assert manager._load(view["id"])["settings"]["evaluation_policy"] == common_evaluation_policy(2)
    assert path.read_bytes() == original
    assert manager.workers.calls == []


def test_shared_evaluation_scope_reports_real_negative_hours_and_final_pending(manager):
    project = create(manager)
    state = manager._load(project["id"])
    state.update(status="completed", index=len(state["plan"]), results={
        "evaluation": {"state": "completed", "result": {"report": {"models": [
            {"id": "mean", "negative": {"hours": .25}}]}}}})
    view = manager.describe(state)
    assert view["evaluation_policy_sha256"] == identity(state["settings"]["evaluation_policy"])
    assert view["results"]["negative_hours"] == .25
    assert "0.25시간" in view["results"]["scope"] and "미완료" in view["results"]["scope"]
    assert view["results"]["independent_Final_completed"] is False
    assert "공통 개발 평가" in view["results"]["scope"]


@pytest.mark.parametrize("payload", [
    {"wakeword": None}, {"wakeword": 123}, {"preset": ["check"]},
    {"advanced": 123}, {"directory": 123}])
def test_invalid_create_types_are_actionable_validation_errors(manager, payload):
    with pytest.raises(ValueError):
        create(manager, **payload)
    assert manager.registry["projects"] == {}


def test_invalid_rounds_never_leave_an_unregistered_partial_project(manager, tmp_path):
    target = manager.workspace / "models" / "보라야"
    with pytest.raises(ValueError):
        create(manager, advanced={"augmentation_rounds": 2})
    assert not (target / "project.json").exists()
    assert manager.registry["projects"] == {}


@pytest.mark.parametrize("path", [".codex/models", ".git/models", "custom-models"])
def test_creation_rejects_caller_storage_paths(manager, tmp_path, path):
    target = tmp_path / path
    with pytest.raises(ValueError, match="자동"):
        create(manager, directory=str(target))
    assert not target.exists()


def test_registered_project_never_silently_accepts_conflicting_word(manager):
    first = create(manager)
    project = Path(manager._load(first["id"])["project"])
    before = digest(project)
    second = create(manager, wakeword="다른호출어")
    assert first["id"] != second["id"]
    assert Path(second["directory"]).name == "다른호출어"
    assert digest(project) == before


def test_conflicting_unregistered_or_nonempty_directory_is_not_overwritten(manager):
    target = manager.workspace / "models" / "보라야"
    project = create_project(target, "춘식아", "other")
    before = digest(project)
    created = create(manager)
    assert Path(created["directory"]) != target
    assert digest(project) == before
    unrelated = manager.workspace / "models" / "다음아"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("keep", encoding="utf-8")
    created = create(manager, wakeword="다음아")
    assert Path(created["directory"]) != unrelated
    assert (unrelated / "important.txt").read_text() == "keep"


def test_final_locked_unregistered_project_is_preserved_without_adoption(manager):
    target = manager.workspace / "models" / "보라야"
    project = create_project(target, "보라야", "standalone", positive_originals=1000,
                             augmentation_rounds=1, updates=10000)
    write_json(target / "final_lock.json", {"state": "frozen"})
    before = digest(project)
    view = create(manager)
    assert Path(view["directory"]) != target
    assert digest(project) == before


def test_duplicate_start_and_single_active_project_guard(manager):
    first = create(manager)
    manager.start(first["id"])
    manager.start(first["id"])
    assert len(manager.workers.items) == 1
    second = create(manager, wakeword="다음아")
    with pytest.raises(ValueError, match="다른 모델 학습"):
        manager.start(second["id"])
    assert len(manager.workers.items) == 1


def test_automatic_order_cooperative_pause_and_resume(manager):
    project = create(manager)
    plan = [{"id": sid, "label": sid, "kind": "task"} for sid in ("frontend", "features", "results")]
    put_state(manager, project["id"], plan=plan)
    manager.start(project["id"])
    manager.workers.items[0].finish()
    manager.tick()
    assert read_json(manager.workers.items[1].request_path)["stage"] == "features"
    assert manager.pause(project["id"])["status"] == "pausing"
    manager.workers.items[1].finish()
    manager.tick()
    assert manager._load(project["id"])["status"] == "paused"
    assert len(manager.workers.items) == 2
    manager.start(project["id"])
    manager.workers.items[2].finish()
    manager.tick()
    assert manager._load(project["id"])["status"] == "completed"
    assert [entry["stage"] for entry in manager._load(project["id"])["history"]] == ["frontend", "features", "results"]


def test_completed_legacy_project_never_blocks_next_project_or_rewrites_history(manager):
    first = create(manager)
    state = put_state(manager, first["id"], status="completed", index=19,
                      auto_continue=True, implementation_id="historical-version")
    path = manager._path(first["id"])
    preserved = path.read_bytes()
    second = create(manager, wakeword="다음아")
    assert manager.start(second["id"])["status"] == "running"
    assert len(manager.workers.items) == 1
    assert path.read_bytes() == preserved
    assert manager.start(first["id"])["status"] == "completed"
    assert path.read_bytes() == preserved


def test_last_stage_clears_automatic_schedule_and_fixes_completion_time(manager):
    view = create(manager)
    put_state(manager, view["id"], plan=[{"id": "frontend", "label": "one", "kind": "task"}])
    manager.start(view["id"])
    finished = manager.workers.items[-1].finish(completed_at="2026-10-23T03:00:00+00:00")
    manager.tick()
    state = manager._load(view["id"])
    assert state["status"] == "completed" and state["auto_continue"] is False
    assert state["completed_at"] == read_json(finished)["completed_at"]


def _paused_after_first_stage(manager):
    view = create(manager)
    manager.start(view["id"])
    manager.pause(view["id"])
    manager.workers.items[-1].finish()
    manager.tick()
    assert manager._load(view["id"])["status"] == "paused"
    return view


def test_compatible_update_is_explicit_hash_verified_and_preserves_existing_outputs(manager):
    view = _paused_after_first_stage(manager)
    state = manager._load(view["id"])
    before = manager._path(view["id"]).read_bytes()
    project_before = Path(state["project"]).read_bytes()
    history_before, inputs_before = state["history"], state["inputs"]
    manager._implementation_id = "compatible-ui-patch"
    assert manager.describe(state)["migration"]["state"] == "available"
    with pytest.raises(ValueError, match="업데이트 적용 후 이어가기"):
        manager.start(view["id"])
    assert manager._path(view["id"]).read_bytes() == before
    migrated = manager.migrate(view["id"])
    after = manager._load(view["id"])
    assert migrated["migration"] is None and len(manager.workers.items) == 1
    assert Path(after["project"]).read_bytes() == project_before
    assert after["history"] == history_before and after["inputs"] == inputs_before
    evidence = read_json(after["migrations"][-1]["path"])
    assert digest(after["migrations"][-1]["path"]) == after["migrations"][-1]["sha256"]
    assert Path(evidence["previous_workflow"]["path"]).read_bytes() == before
    assert evidence["history_sha256"] == identity(history_before)
    manager.start(view["id"])
    assert len(manager.workers.items) == 2


def test_compatibility_blocks_changed_future_computation_but_ignores_completed_stage(manager):
    view = _paused_after_first_stage(manager)
    state = manager._load(view["id"])
    manager._execution_contract = json.loads(json.dumps(manager._execution_contract))
    manager._implementation_id = "changed-code"
    manager._execution_contract["stages"]["frontend"] = "changed-only-finished-stage"
    assert manager.describe(state)["migration"]["state"] == "available"
    manager._execution_contract["stages"]["training"] = "changed-training"
    assert manager.describe(state)["migration"]["state"] == "blocked"
    original = manager._path(view["id"]).read_bytes()
    with pytest.raises(ValueError, match="남은 단계"):
        manager.migrate(view["id"])
    assert manager._path(view["id"]).read_bytes() == original


@pytest.mark.parametrize("tamper", ["settings", "plan", "history", "project"])
def test_compatibility_migration_rejects_changed_inputs_or_history(manager, tamper):
    view = _paused_after_first_stage(manager)
    state = manager._load(view["id"])
    manager._implementation_id = "ui-patch"
    if tamper == "settings":
        state["settings"]["updates"] += 1
    elif tamper == "plan":
        state["plan"][1]["id"] = "another-stage"
    elif tamper == "history":
        Path(state["history"][0]["result"]).write_text("{}", encoding="utf-8")
    else:
        config = read_json(state["project"])
        config["training"]["updates"] += 1
        write_json(state["project"], config)
    write_json(manager._path(view["id"]), state)
    before = manager._path(view["id"]).read_bytes()
    with pytest.raises(ValueError):
        manager.migrate(view["id"])
    assert manager._path(view["id"]).read_bytes() == before


def test_compatibility_migration_never_runs_while_worker_is_alive(manager):
    view = create(manager)
    manager.start(view["id"])
    before = manager._path(view["id"]).read_bytes()
    manager._implementation_id = "ui-patch"
    assert manager.describe(manager._load(view["id"]))["migration"]["state"] == "blocked"
    with pytest.raises(ValueError, match="현재 작업"):
        manager.migrate(view["id"])
    assert manager._path(view["id"]).read_bytes() == before


def test_snapshotless_legacy_update_requires_exact_audited_fingerprint(manager, monkeypatch):
    view = _paused_after_first_stage(manager)
    state = manager._load(view["id"])
    state.pop("execution_contract")
    state["implementation_id"] = "known-audited-release"
    write_json(manager._path(view["id"]), state)
    manager._implementation_id = "current-release"
    assert manager.describe(state)["migration"]["state"] == "blocked"
    monkeypatch.setattr(studio, "_LEGACY_COMPATIBLE_CONTRACTS", {
        "known-audited-release": manager._execution_contract["computation_sha256"]})
    assert manager.describe(state)["migration"]["state"] == "available"
    assert manager.migrate(view["id"])["migration"] is None


def test_completed_time_is_stable_and_legacy_offline_time_is_unclassified(manager, monkeypatch):
    view = create(manager)
    state = manager._load(view["id"])
    result_path = Path(state["project"]).parent / "historical_result.json"
    write_json(result_path, {"started_at": "2026-10-23T01:00:00+00:00", "completed_at": "2026-10-23T01:10:00+00:00"})
    state.update(started_at="2026-10-23T00:00:00+00:00", status="completed", index=19,
                 history=[{"result": str(result_path), "sha256": digest(result_path),
                           "completed_at": "2026-10-23T02:00:00+00:00"}])
    state.pop("timing_observations", None)
    write_json(manager._path(view["id"]), state)
    preserved = manager._path(view["id"]).read_bytes()
    monkeypatch.setattr(studio.time, "time", lambda: studio._timestamp("2026-10-24T00:00:00+00:00"))
    first = manager.describe(state)["timing"]
    monkeypatch.setattr(studio.time, "time", lambda: studio._timestamp("2026-10-25T00:00:00+00:00"))
    assert manager.describe(state)["timing"] == first
    assert first["total_seconds"] == 7200 and first["active_seconds"] == 600
    assert first["unclassified_seconds"] == 6600 and first["legacy_estimate"] is True
    assert manager._path(view["id"]).read_bytes() == preserved


def test_dead_worker_offline_interval_is_not_counted_as_active(manager, monkeypatch):
    view = create(manager)
    state = manager._load(view["id"])
    state.update(started_at="2026-10-23T00:00:00+00:00", status="running",
                 job={"started_at": "2026-10-23T00:00:00+00:00", "pid": 99999, "creation_identity": "gone"})
    state.pop("timing_observations", None)
    monkeypatch.setattr(studio.time, "time", lambda: studio._timestamp("2026-10-23T02:00:00+00:00"))
    timing = manager._timing(state)
    assert timing["active_seconds"] == 0 and timing["unclassified_seconds"] == 7200


def test_time_breakdown_distinguishes_review_pause_and_verified_work(manager, monkeypatch):
    view = create(manager)
    state = manager._load(view["id"])
    result_path = Path(state["project"]).parent / "recorded_work.json"
    write_json(result_path, {"started_at": "2026-10-23T00:00:00+00:00", "completed_at": "2026-10-23T00:10:00+00:00"})
    state.update(started_at="2026-10-23T00:00:00+00:00", status="paused", history=[{
        "result": str(result_path), "sha256": digest(result_path)}], timing_observations={"segments": [
            {"status": "running", "start": "2026-10-23T00:00:00+00:00", "end": "2026-10-23T00:10:00+00:00"},
            {"status": "waiting_review", "start": "2026-10-23T00:10:00+00:00", "end": "2026-10-23T00:30:00+00:00"},
            {"status": "paused", "start": "2026-10-23T00:30:00+00:00"}]})
    monkeypatch.setattr(studio.time, "time", lambda: studio._timestamp("2026-10-23T01:00:00+00:00"))
    timing = manager._timing(state)
    assert timing["active_seconds"] == 600 and timing["review_wait_seconds"] == 1200
    assert timing["paused_seconds"] == 1800 and timing["unclassified_seconds"] == 0


@pytest.mark.parametrize("field", ["task_id", "request_sha256", "config_sha256", "result_config_sha256"])
def test_mismatched_completion_is_an_error_and_does_not_advance(manager, field):
    project = create(manager)
    manager.start(project["id"])
    result = manager.workers.items[0].finish(**{field: "mismatched"})
    before = digest(result)
    manager.tick()
    state = manager._load(project["id"])
    assert state["status"] == "error" and state["index"] == 0
    assert len(manager.workers.items) == 1 and digest(result) == before


def test_saved_running_label_is_not_process_liveness_or_completion(manager):
    project = create(manager)
    manager.start(project["id"])
    child = manager.workers.items[0]
    child.returncode = 9
    manager.workers.identities.pop(child.pid)
    snapshot = manager.snapshot()["projects"][0]
    assert snapshot["process_alive"] is False
    assert snapshot["progress"]["done"] == 0
    manager.tick()
    state = manager._load(project["id"])
    assert state["status"] == "interrupted" and state["index"] == 0


def test_restart_tracks_creation_identity_and_does_not_duplicate_live_worker(manager):
    project = create(manager)
    manager.start(project["id"])
    restarted = studio.StudioManager(manager.workspace, popen=manager.workers)
    assert restarted._load(project["id"])["status"] == "running"
    restarted.start(project["id"])
    assert len(manager.workers.items) == 1
    manager.workers.identities[manager.workers.items[0].pid] = "a-recycled-pid"
    recycled = studio.StudioManager(manager.workspace, popen=manager.workers)
    assert recycled._load(project["id"])["status"] == "interrupted"
    restarted.close()
    recycled.close()


def test_dead_worker_lock_is_archived_but_live_or_recycled_pid_lock_is_preserved(manager):
    project = create(manager)
    manager.start(project["id"])
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    lock = root / "features" / ".writer.lock"
    write_json(lock, {"pid": state["job"]["pid"], "marker": "actual-worker"})
    manager.workers.items[0].returncode = 9
    manager.workers.identities[state["job"]["pid"]] = "recycled"
    with pytest.raises(ValueError, match="살아|재사용"):
        manager._recover_dead_locks(state)
    assert lock.exists()
    manager.workers.identities.clear()
    manager._recover_dead_locks(state)
    assert not lock.exists()
    records = list((root / ".studio" / "recovered_locks").glob("*.json"))
    assert len(records) == 1 and read_json(records[0])["record"]["marker"] == "actual-worker"


def test_reference_review_requires_real_bound_bytes_and_listened_answer(manager):
    project = create(manager)
    audio, _ = reference_queue(manager, project["id"])
    with pytest.raises(ValueError, match="모든 항목"):
        manager.save_review(project["id"], "fixture", {"fixture-1": {"decision": "keep", "listened": False}})
    assert manager._load(project["id"])["status"] == "waiting_review"
    audio.write_bytes(b"changed audio")
    with pytest.raises(ValueError, match="changed"):
        manager.save_review(project["id"], "fixture", {"fixture-1": {"decision": "keep", "listened": True}})
    assert manager._load(project["id"])["status"] == "waiting_review"


def test_saved_review_is_explicit_hash_bound_human_disposition_not_blanket_listening(manager):
    project = create(manager)
    audio, queue = reference_queue(manager, project["id"])
    decisions = {"fixture-1": {"decision": "keep", "listened": True}}
    result = manager.save_review(project["id"], "fixture", decisions)
    assert result["kept"] == 1
    state = manager._load(project["id"])
    evidence = state["inputs"]["review_answers"]["fixture"]
    answer = read_json(evidence["path"])
    assert digest(evidence["path"]) == evidence["sha256"]
    assert answer["source_scope"] == "Human dispositions entered through the local review screen"
    assert answer["queued_audio_sha256"] == {"fixture-1": digest(audio)}
    assert answer["queue_sha256"] == digest(queue)
    assert answer["whole_pool_policy_approved"] is False
    assert state["status"] == "paused" and not manager.workers.items
    with pytest.raises(ValueError, match="현재 검수 목록"):
        manager.save_review(project["id"], "fixture", decisions)


def test_review_hold_and_pilot_rejection_never_start_bulk_generation(manager):
    project = create(manager)
    reference_queue(manager, project["id"])
    result = manager.save_review(project["id"], "fixture", {"fixture-1": {"decision": "review", "listened": True}})
    assert result["kept"] == 0 and manager._load(project["id"])["status"] == "waiting_review"
    reference_queue(manager, project["id"], kind="pilot")
    manager.save_review(project["id"], "fixture", {"fixture-1": {"decision": "discard", "listened": True}})
    state = manager._load(project["id"])
    assert state["status"] == "waiting_review" and not manager.workers.items
    assert read_json(Path(state["project"]).parent / "approval.json")["generation_policy_approved"] is False


@pytest.fixture
def http_server(manager):
    server = studio.make_studio_server(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
    thread.start()

    def request(path, method="GET", payload=None, *, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        actual_headers = {"Host": f"127.0.0.1:{server.server_port}"}
        if payload is not None:
            raw = json.dumps(payload, ensure_ascii=False).encode()
            actual_headers["Content-Type"] = "application/json"
        actual_headers.update(headers or {})
        connection.request(method, path, body=raw, headers=actual_headers)
        response = connection.getresponse()
        content, response_headers, status = response.read(), dict(response.getheaders()), response.status
        connection.close()
        return status, content, response_headers

    request.server = server
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_http_static_ui_state_and_bound_random_port_are_usable(manager, http_server):
    assert http_server.server.server_address[0] == "127.0.0.1"
    for route, marker in (("/", "한국어 호출어 학습"), ("/studio.js", "KowwStudio"), ("/studio.css", ".project-switch")):
        status, content, headers = http_server(route)
        assert status == 200 and marker in content.decode()
        assert headers["X-Frame-Options"] == "SAMEORIGIN" and headers["Cache-Control"] == "no-store"
    status, content, _ = http_server("/api/state")
    assert status == 200 and json.loads(content)["token"] == manager.token


def test_http_csrf_origin_host_guards_and_valid_create(manager, http_server):
    payload = {"wakeword": "보라야", "preset": "check"}
    assert http_server("/api/projects", "POST", payload)[0] == 403
    headers = {"X-Koww-Token": manager.token, "Origin": "https://evil.invalid"}
    assert http_server("/api/projects", "POST", payload, headers=headers)[0] == 403
    assert http_server("/api/state", headers={"Host": "evil.invalid"})[0] == 403
    headers = {"X-Koww-Token": manager.token, "Origin": f"http://127.0.0.1:{http_server.server.server_port}"}
    status, content, _ = http_server("/api/projects", "POST", payload, headers=headers)
    assert status == 200 and json.loads(content)["wakeword"] == "보라야"
    assert len(manager.registry["projects"]) == 1


def test_http_compatible_update_requires_token_and_does_not_start_worker(manager, http_server):
    view = _paused_after_first_stage(manager)
    manager._implementation_id = "compatible-ui-update"
    route = f"/api/projects/{view['id']}/migrate"
    assert http_server(route, "POST", {})[0] == 403
    status, content, _ = http_server(route, "POST", {}, headers={"X-Koww-Token": manager.token})
    assert status == 200 and json.loads(content)["migration"] is None
    assert len(manager.workers.items) == 1


@pytest.mark.parametrize("headers,body", [
    ({"Content-Type": "application/json", "Content-Length": "1000001"}, b""),
    ({"Content-Type": "application/json", "Content-Length": "-1"}, b""),
    ({"Content-Type": "text/plain"}, b"{}"),
    ({"Content-Type": "application/json"}, b"[]"),
    ({"Content-Type": "application/json"}, b"broken json"),
    ({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, b"")])
def test_http_request_size_type_and_json_shape_are_bounded(manager, http_server, headers, body):
    status, _, _ = http_server("/api/projects", "POST", raw=body, headers={"X-Koww-Token": manager.token, **headers})
    assert status == 400 and manager.registry["projects"] == {}


def test_http_review_audio_guard_and_integrated_fetch_save_endpoints(manager, http_server):
    project = create(manager)
    audio, _ = reference_queue(manager, project["id"])
    query = f"?project={project['id']}&queue=fixture"
    status, content, _ = http_server("/review/" + query)
    assert status == 200
    page = content.decode()
    assert page_config(page) == {"queue_url": "/review/queue" + query,
        "save_url": "/review/save" + query, "token": manager.token,
        "saved_message": "저장 완료 · 다음 단계를 준비합니다."}
    status, content, _ = http_server("/review/queue" + query)
    queue = json.loads(content)
    status, original, _ = http_server(queue["items"][0]["audio"])
    assert status == 200 and original == audio.read_bytes()
    assert http_server("/review/audio/-1" + query)[0] == 400
    audio.write_bytes(b"changed")
    assert http_server(queue["items"][0]["audio"])[0] == 400
    audio.write_bytes(original)
    answers = {"fixture-1": {"decision": "keep", "listened": True}}
    assert http_server("/review/save" + query, "POST", answers)[0] == 403
    assert http_server("/review/save" + query, "POST", answers,
        headers={"X-Koww-Token": manager.token, "Origin": "https://foreign.invalid"})[0] == 403
    status, body, _ = http_server("/review/save" + query, "POST", answers,
        headers={"X-Koww-Token": manager.token})
    assert status == 200 and json.loads(body)["kept"] == 1
    assert read_json(audio.parent.parent / "review/references_results.json")["decisions"] == answers


def test_http_result_traversal_is_rejected_and_regular_report_is_served(manager, http_server):
    project = create(manager)
    root = Path(manager._load(project["id"])["project"]).parent
    site = root / "reports" / "studio_site"
    site.mkdir()
    (site / "index.html").write_text("fixture report", encoding="utf-8")
    (root / "reports" / "secret.txt").write_text("do not serve", encoding="utf-8")
    put_state(manager, project["id"], results={"results": {"result": {"site": str(site / "index.html")}}})
    assert http_server(f"/results/{project['id']}/")[0] == 200
    status, content, _ = http_server(f"/results/{project['id']}/%2e%2e/secret.txt")
    assert status == 403 and b"do not serve" not in content


def test_http_mic_assets_use_integrated_prefix_and_their_own_token(manager, http_server, monkeypatch):
    project = create(manager)
    prefix = f"/mic/{project['id']}"

    class Microphone:
        token = "fixture-mic-token"
        def info(self): return {"token": self.token, "wakeword": "보라야"}
        def start(self): return {"session_id": "fixture-session"}
        def stop(self, session): return {"stopped": session}
        def process_frame(self, session, sequence, payload): return {"session": session, "sequence": sequence, "bytes": len(payload)}

    microphone = Microphone()
    monkeypatch.setattr(manager, "microphone", lambda project_id: microphone)
    status, content, headers = http_server(prefix)
    assert status == 308 and content == b"" and headers["Location"] == prefix + "/"
    status, content, _ = http_server(prefix + "/")
    page = content.decode()
    assert status == 200 and 'src="microphone.js"' in page and 'href="microphone.css"' in page
    assert page_config(page) == {"api_base": prefix, "worklet_url": prefix + "/microphone_worklet.js"}
    for asset in ("microphone.js", "microphone.css"):
        assert http_server(urljoin(headers["Location"], asset))[0] == 200
    status, content, _ = http_server(prefix + "/microphone.js")
    from importlib.resources import files
    assert status == 200 and content == (files("korean_openwakeword") / "resources/microphone.js").read_bytes()
    assert http_server(prefix + "/microphone_worklet.js")[0] == 200
    assert http_server(prefix + "/api/info")[0] == 200
    assert http_server(prefix + "/api/start", "POST", {}, headers={"X-Koww-Token": manager.token})[0] == 403
    assert http_server(prefix + "/api/start", "POST", {},
        headers={"X-Koww-Token": microphone.token, "Origin": "https://foreign.invalid"})[0] == 403
    assert http_server(prefix + "/api/start", "POST", {}, headers={"X-Koww-Token": microphone.token})[0] == 200
    status, content, _ = http_server(prefix + "/api/frame", "POST", raw=b"\x00" * 2560,
        headers={"Content-Type": "application/octet-stream", "X-Koww-Token": microphone.token,
                 "X-Koww-Session": "fixture-session", "X-Koww-Sequence": "0"})
    assert status == 200 and json.loads(content)["bytes"] == 2560
    status, content, _ = http_server(prefix + "/api/stop", "POST", {"session_id": "fixture-session"},
        headers={"X-Koww-Token": microphone.token})
    assert status == 200 and json.loads(content)["stopped"] == "fixture-session"


def test_http_mic_initialization_failure_returns_recoverable_json(manager, http_server, monkeypatch):
    project = create(manager)
    def unavailable(project_id):
        raise RuntimeError("Model initialization failed")
    monkeypatch.setattr(manager, "microphone", unavailable)
    status, content, _ = http_server(f"/mic/{project['id']}/api/info")
    assert status == 400
    assert json.loads(content) == {"error": "Model initialization failed"}


def worker_request(tmp_path, stage="frontend"):
    project = create_project(tmp_path / "project", "보라야", "worker")
    request = {"task_id": "fixture-task", "project": str(project), "workspace": str(tmp_path),
               "settings": {}, "stage": stage, "config_sha256": digest(project),
               "result_path": str(tmp_path / "result.json")}
    path = tmp_path / "request.json"
    write_json(path, request)
    return path, request


def test_worker_rejects_changed_configuration_before_running(tmp_path, monkeypatch):
    from korean_openwakeword import studio_steps, studio_worker
    path, request = worker_request(tmp_path)
    config = read_json(request["project"])
    config["wakeword"] = "새호출어"
    write_json(request["project"], config)
    called = []
    monkeypatch.setattr(studio_steps, "run_stage", lambda *args: called.append(args))
    assert studio_worker.execute(path) == 1 and not called
    result = read_json(request["result_path"])
    assert result["task_id"] == request["task_id"] and result["request_sha256"] == digest(path)
    assert "changed before" in result["error"]["message"]


def test_worker_records_errors_and_refuses_training_setting_mutation(tmp_path, monkeypatch):
    from korean_openwakeword import studio_steps, studio_worker
    path, request = worker_request(tmp_path)
    def mutate(project, *args):
        config = read_json(project)
        config["wakeword"] = "다른호출어"
        write_json(project, config)
        return {"state": "completed"}
    monkeypatch.setattr(studio_steps, "run_stage", mutate)
    assert studio_worker.execute(path) == 1
    result = read_json(request["result_path"])
    assert "frozen project configuration" in result["error"]["message"]


def test_worker_conflicting_existing_result_is_never_overwritten(tmp_path, monkeypatch):
    from korean_openwakeword import studio_steps, studio_worker
    path, request = worker_request(tmp_path)
    write_json(request["result_path"], {"task_id": "unrelated-prior-task", "valuable": "preserve"})
    before = Path(request["result_path"]).read_bytes()
    called = []
    monkeypatch.setattr(studio_steps, "run_stage", lambda *args: called.append(args) or {"state": "completed"})
    try:
        status = studio_worker.execute(path)
    except ValueError:
        status = 1
    assert status == 1 and not called
    assert Path(request["result_path"]).read_bytes() == before


def test_worker_matching_existing_result_is_reused_without_work_or_overwrite(tmp_path, monkeypatch):
    from korean_openwakeword import studio_steps, studio_worker
    path, request = worker_request(tmp_path)
    write_json(request["result_path"], {"task_id": request["task_id"], "request_sha256": digest(path),
                                      "result": {"state": "completed"}})
    before = Path(request["result_path"]).read_bytes()
    called = []
    monkeypatch.setattr(studio_steps, "run_stage", lambda *args: called.append(args))
    assert studio_worker.execute(path) == 0 and not called
    assert Path(request["result_path"]).read_bytes() == before


def test_worker_result_cannot_escape_its_request_directory(tmp_path, monkeypatch):
    from korean_openwakeword import studio_steps, studio_worker
    path, request = worker_request(tmp_path)
    request["result_path"] = str(tmp_path / "elsewhere" / "result.json")
    write_json(path, request)
    called = []
    monkeypatch.setattr(studio_steps, "run_stage", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="beside"):
        studio_worker.execute(path)
    assert not called and not Path(request["result_path"]).exists()


def test_changed_worker_artifact_prevents_advance(manager):
    project = create(manager)
    manager.start(project["id"])
    artifact = Path(manager._load(project["id"])["project"]).parent / "reports" / "fixture.json"
    write_json(artifact, {"state": "original"})
    manager.workers.items[0].finish(artifact_bindings=[{"path": str(artifact), "sha256": digest(artifact)}])
    write_json(artifact, {"state": "tampered"})
    manager.tick()
    state = manager._load(project["id"])
    assert state["status"] == "error" and state["index"] == 0


def test_restart_accepts_completed_bound_receipt_then_launches_only_next_stage(manager):
    project = create(manager)
    manager.start(project["id"])
    manager.workers.items[0].finish()
    restarted = studio.StudioManager(manager.workspace, popen=manager.workers)
    assert restarted._load(project["id"])["index"] == 1
    restarted.start(project["id"])
    assert len(manager.workers.items) == 2
    assert read_json(manager.workers.items[1].request_path)["stage"] == "data_download"
    restarted.close()


def test_queue_changed_after_stage_result_is_not_accepted(manager):
    project = create(manager)
    _, queue = reference_queue(manager, project["id"])
    queue.write_text(queue.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="변경"):
        manager.review_queue(project["id"], "fixture")


def test_workspace_lease_rejects_second_live_owner_and_cleans_its_own_record(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "process_identity", lambda pid: "current-birth" if pid == os.getpid() else None)
    workspace = tmp_path / "workspace"
    lease = workspace / ".koww-studio" / "server.json"
    with studio.workspace_lease(workspace):
        owner = lease.read_bytes()
        with pytest.raises(RuntimeError, match="이미 실행"):
            with studio.workspace_lease(workspace):
                pytest.fail("a second live server acquired the same workspace")
        assert lease.read_bytes() == owner
    assert not lease.exists()


@pytest.mark.parametrize("previous_identity", [None, "new-recycled-identity"])
def test_workspace_lease_archives_stale_or_recycled_previous_owner(tmp_path, monkeypatch, previous_identity):
    workspace = tmp_path / "workspace"
    lease = workspace / ".koww-studio" / "server.json"
    previous = {"pid": 12345, "creation_identity": "old-birth", "lease_id": "previous"}
    write_json(lease, previous)
    monkeypatch.setattr(studio, "process_identity", lambda pid: "current-birth" if pid == os.getpid() else previous_identity)
    with studio.workspace_lease(workspace):
        assert read_json(lease)["lease_id"] != "previous"
        archives = list((lease.parent / "previous_servers").glob("*.json"))
        assert len(archives) == 1 and read_json(archives[0]) == previous
    assert not lease.exists()


def test_prepared_pool_discard_is_saved_for_worker_and_never_blanket_approved(manager, monkeypatch):
    from korean_openwakeword import review
    from korean_openwakeword.studio_steps import _saved_review
    project = create(manager)
    first_audio, queue = reference_queue(manager, project["id"])
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    first = json.loads(queue.read_text(encoding="utf-8"))
    first.update(provenance="public_dataset", source_url="https://example.invalid/public-fixture", license="CC0-1.0")
    second_audio = first_audio.with_name("discard-fixture.wav")
    with wave.open(str(second_audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x01\x00" * 1600)
    second = {**first, "sample_id": "fixture-2", "path": str(second_audio.relative_to(root)), "sha256": digest(second_audio)}
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [first, second])
    queue.write_text(body, encoding="utf-8")
    pool = root / "review" / "public_pool.jsonl"
    pool.write_text(body, encoding="utf-8")
    descriptor = {"id": "fixture", "kind": "content", "queue": str(queue), "queue_sha256": digest(queue),
                  "pool_manifest": str(pool), "pool_sha256": digest(pool)}
    state["results"]["fixture_review"]["reviews"] = [descriptor]
    manager._save(state)
    decisions = {"fixture-1": {"decision": "keep", "listened": True},
                 "fixture-2": {"decision": "discard", "listened": False}}
    with pytest.raises(ValueError, match="동의"):
        manager.save_review(project["id"], "fixture", {"decisions": decisions, "policy_approved": False})
    def unexpected_review(*args, **kwargs):
        pytest.fail("pool preview saving must not perform full-source admission in the HTTP thread")
    monkeypatch.setattr(review, "apply_review", unexpected_review)
    result = manager.save_review(project["id"], "fixture", {"decisions": decisions, "policy_approved": True})
    assert result["state"] == "review_saved" and result["admission_pending"] is True
    assert "admission" not in result and result["kept"] == 1
    pending = manager._load(project["id"])
    assert pending["inputs"]["review_results"]["fixture"]["admission_pending"] is True
    answer = read_json(pending["inputs"]["review_answers"]["fixture"]["path"])
    assert answer["decisions"]["fixture-2"]["decision"] == "discard"
    assert _saved_review(root, descriptor, {"inputs": pending["inputs"]}) is True
    assert not (root / "review" / "content_admission.json").exists()
    assert not manager.workers.items


def test_public_pool_preview_without_provenance_saves_only_bound_listening_evidence(manager, monkeypatch):
    from korean_openwakeword import review
    project = create(manager)
    _, queue = reference_queue(manager, project["id"])
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    row = json.loads(queue.read_text(encoding="utf-8"))
    # Real prepared previews carry the selected source binding; the source
    # manifest, rather than every preview, carries its complete provenance.
    for key in ("voice_family_id", "generation_method", "tts_settings_id", "provenance", "source_url", "license"):
        row.pop(key, None)
    row["reviewed_source"] = {"path": row["path"], "sha256": row["sha256"]}
    queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
    pool = root / "review" / "public_sources.jsonl"
    pool.write_text(json.dumps({**row, "provenance": "public_dataset", "source_url": "https://example.invalid/source", "license": "CC0-1.0"}) + "\n", encoding="utf-8")
    descriptor = {"id": "fixture", "kind": "content", "queue": str(queue), "queue_sha256": digest(queue),
                  "pool_manifest": str(pool), "pool_sha256": digest(pool)}
    state["results"]["fixture_review"]["reviews"] = [descriptor]
    manager._save(state)
    monkeypatch.setattr(review, "apply_review", lambda *args, **kwargs: pytest.fail("preview saving must defer source admission"))
    result = manager.save_review(project["id"], "fixture", {
        "decisions": {"fixture-1": {"decision": "keep", "listened": True}}, "policy_approved": True})
    assert result["state"] == "review_saved" and result["admission_pending"] is True
    assert "admission" not in result
    saved = manager._load(project["id"])
    answer = read_json(saved["inputs"]["review_answers"]["fixture"]["path"])
    assert answer["source_manifest_sha256"] == digest(pool)
    assert answer["queued_audio_sha256"] == {"fixture-1": row["sha256"]}
    assert not (root / "review" / "content_admission.json").exists()


def test_foreign_mic_origin_is_rejected_before_model_initialization(manager, http_server, monkeypatch):
    project = create(manager)
    calls = []
    monkeypatch.setattr(manager, "microphone", lambda project_id: calls.append(project_id))
    status, _, _ = http_server(f"/mic/{project['id']}/api/start", "POST", {},
        headers={"Origin": "https://evil.invalid", "X-Koww-Token": manager.token})
    assert status == 403 and not calls


def publish_actual_owner(manager, project_id, *, actual_pid=42123, **overrides):
    job = manager._load(project_id)["job"]
    owner = {"task_id": job["task_id"], "request_sha256": job["request_sha256"],
             "pid": actual_pid, "creation_identity": "actual-python-birth",
             "parent_pid": job["pid"], "started_at": now(), **overrides}
    path = Path(job["directory"]) / "owner.json"
    write_json(path, owner)
    return path, owner


def test_actual_python_owner_keeps_job_live_after_venv_launcher_exits(manager):
    project = create(manager)
    manager.start(project["id"])
    launcher = manager.workers.items[0]
    launcher.returncode = 0
    manager.workers.identities.pop(launcher.pid)
    _, owner = publish_actual_owner(manager, project["id"])
    manager.workers.identities[owner["pid"]] = owner["creation_identity"]
    assert manager.snapshot()["projects"][0]["process_alive"] is True
    manager.tick()
    assert manager._load(project["id"])["status"] == "running"
    manager.start(project["id"])
    assert len(manager.workers.items) == 1
    restarted = studio.StudioManager(manager.workspace, popen=manager.workers)
    assert restarted._load(project["id"])["status"] == "running"
    restarted.start(project["id"])
    assert len(manager.workers.items) == 1
    restarted.close()


@pytest.mark.parametrize("field", ["task_id", "request_sha256"])
def test_actual_owner_from_another_request_is_rejected_without_relaunch(manager, field):
    project = create(manager)
    manager.start(project["id"])
    launcher = manager.workers.items[0]
    launcher.returncode = 0
    manager.workers.identities.pop(launcher.pid)
    path, owner = publish_actual_owner(manager, project["id"], **{field: "wrong-binding"})
    manager.workers.identities[owner["pid"]] = owner["creation_identity"]
    before = digest(path)
    with pytest.raises(ValueError, match="소유 기록|실행 요청"):
        manager.start(project["id"])
    assert digest(path) == before and len(manager.workers.items) == 1


def test_dead_actual_python_lock_is_recovered_only_for_bound_owner_pid(manager):
    project = create(manager)
    manager.start(project["id"])
    launcher = manager.workers.items[0]
    launcher.returncode = 9
    manager.workers.identities.pop(launcher.pid)
    _, owner = publish_actual_owner(manager, project["id"])
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    lock = root / "features" / ".writer.lock"
    write_json(lock, {"pid": owner["pid"], "fixture": "actual-python-owner"})
    # The actual PID has since been recycled: preserve its lock until the live
    # process is gone even though it is not the original worker anymore.
    manager.workers.identities[owner["pid"]] = "recycled-process-birth"
    with pytest.raises(ValueError, match="살아|재사용"):
        manager._recover_dead_locks(state)
    assert lock.exists()
    manager.workers.identities.pop(owner["pid"])
    manager._recover_dead_locks(state)
    assert not lock.exists()
    archived = list((root / ".studio" / "recovered_locks").glob("*.json"))
    assert len(archived) == 1 and read_json(archived[0])["record"]["pid"] == owner["pid"]
    write_json(lock, {"pid": owner["pid"] + 1, "fixture": "unrelated-dead-process"})
    with pytest.raises(ValueError, match="다른 작업"):
        manager._recover_dead_locks(state)
    assert lock.exists()


@pytest.mark.parametrize("suffix", [".writer.lock", ".canonical_lock/.writer.lock"])
def test_canonical_shared_locks_recover_bound_dead_worker_without_recursive_audio_scan(manager, monkeypatch, suffix):
    project = create(manager)
    manager.start(project["id"])
    launcher = manager.workers.items[0]
    launcher.returncode = 9
    manager.workers.identities.pop(launcher.pid)
    _, owner = publish_actual_owner(manager, project["id"])
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    shared = Path(state["storage_workspace"]) / "shared"
    lock = shared / "automatic_sources_v3/parts/fixture" / suffix
    record = {"pid": owner["pid"], "fixture": "canonical-actual-worker"}
    write_json(lock, record)
    # A future namespace or arbitrary deep audio folder is outside the exact
    # recovery patterns. Neither is recursively searched or changed.
    unrelated = shared / "automatic_sources_v4/parts/fixture" / suffix
    write_json(unrelated, {"pid": owner["pid"], "fixture": "future-namespace"})
    monkeypatch.setattr(Path, "rglob", lambda *args, **kwargs: pytest.fail("Recovery must not scan corpus audio recursively"))
    manager._recover_dead_locks(state)
    assert not lock.exists() and unrelated.exists()
    archived = [read_json(path) for path in (root / ".studio/recovered_locks").glob("*.json")]
    assert len(archived) == 1
    assert archived[0]["original"] == str(lock)
    assert archived[0]["record"] == record
    assert archived[0]["job"]["request_sha256"] == state["job"]["request_sha256"]


@pytest.mark.parametrize("suffix", [".writer.lock", ".canonical_lock/.writer.lock"])
def test_canonical_shared_locks_preserve_live_recycled_and_other_worker_owners(manager, suffix):
    project = create(manager)
    manager.start(project["id"])
    state = manager._load(project["id"])
    pid = state["job"]["pid"]
    lock = Path(state["storage_workspace"]) / "shared/automatic_sources_v3/parts/fixture" / suffix
    write_json(lock, {"pid": pid, "fixture": "actual-launcher"})
    before = lock.read_bytes()
    manager._recover_dead_locks(state)
    assert lock.read_bytes() == before  # Identified original worker is live.
    manager.workers.items[0].returncode = 9
    manager.workers.identities[pid] = "recycled-birth-identity"
    with pytest.raises(ValueError, match="살아|재사용"):
        manager._recover_dead_locks(state)
    assert lock.read_bytes() == before
    manager.workers.identities.pop(pid)
    write_json(lock, {"pid": pid + 99, "fixture": "another-dead-worker"})
    before = lock.read_bytes()
    with pytest.raises(ValueError, match="다른 작업"):
        manager._recover_dead_locks(state)
    assert lock.read_bytes() == before
    root = Path(state["project"]).parent
    assert not (root / ".studio/recovered_locks").exists()


def test_real_worker_subprocess_records_actual_python_identity_and_error_receipt(tmp_path):
    from korean_openwakeword.studio_steps import preset_settings
    path, request = worker_request(tmp_path, stage="fixture-unknown-no-download")
    request["settings"] = preset_settings("check")
    write_json(path, request)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    source = str(Path(studio.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen([sys.executable, "-m", "korean_openwakeword.studio_worker", "--request", str(path)],
        cwd=tmp_path, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        _, error = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        pytest.fail("fixture worker did not finish its bounded unknown-stage check")
    assert process.returncode == 1, error.decode("utf-8", errors="replace")
    owner = read_json(path.parent / "owner.json")
    assert owner["task_id"] == request["task_id"] and owner["request_sha256"] == digest(path)
    assert type(owner["pid"]) is int and owner["pid"] != os.getpid()
    assert owner["creation_identity"] and type(owner["parent_pid"]) is int
    if owner["pid"] != process.pid:
        # Windows venv python.exe is a launcher; its real child owns the work.
        assert owner["parent_pid"] == process.pid
    else:
        assert owner["parent_pid"] == os.getpid()
    result = read_json(request["result_path"])
    assert result["task_id"] == request["task_id"] and result["request_sha256"] == digest(path)
    assert "Unknown studio stage" in result["error"]["message"]


def test_evaluation_catalog_changes_are_part_of_worker_implementation_identity(tmp_path, monkeypatch):
    package = tmp_path / "fixture_package"
    package.mkdir()
    source = package / "studio.py"
    source.write_text("# isolated fingerprint fixture\n", encoding="utf-8")
    resources = package / "resources"
    resources.mkdir()
    catalog = resources / "evaluation_noise_v2.json"
    catalog.write_text('{"recipe_version":2,"assets":[]}', encoding="utf-8")
    monkeypatch.setattr(studio, "__file__", str(source))
    before = studio._implementation()
    catalog.write_text('{"recipe_version":2,"assets":["different held-out source"]}', encoding="utf-8")
    assert studio._implementation() != before


def test_unstarted_workflow_update_preserves_sources_and_records_exact_prior_state(manager):
    project = create(manager)
    workflow = manager._path(project["id"])
    state = manager._load(project["id"])
    previous_implementation = state["implementation_id"]
    original_workflow = workflow.read_bytes()
    root = Path(state["project"]).parent
    original = root / "data" / "existing-public-material.bin"
    original.write_bytes(b"previously downloaded material is preserved")
    protected = {path: digest(path) for path in root.rglob("*") if path.is_file() and path != workflow}
    manager._implementation_id = "new-tested-implementation"
    manager.start(project["id"])
    migrated = manager._load(project["id"])
    assert migrated["implementation_id"] == "new-tested-implementation"
    migration = migrated["migrations"][0]
    assert digest(migration["path"]) == migration["sha256"]
    audit = read_json(migration["path"])
    assert audit["old_implementation_id"] == previous_implementation
    assert audit["new_implementation_id"] == manager._implementation_id
    assert audit["config_sha256"] == state["config_sha256"]
    assert "No workflow execution" in audit["reason"]
    assert Path(audit["previous_workflow"]["path"]).read_bytes() == original_workflow
    assert digest(audit["previous_workflow"]["path"]) == audit["previous_workflow"]["sha256"]
    assert all(digest(path) == checksum for path, checksum in protected.items())
    request = read_json(manager.workers.items[0].request_path)
    assert request["implementation_id"] == manager._implementation_id
    # Starting the already live worker again must not create another migration.
    manager.start(project["id"])
    assert len(manager._load(project["id"])["migrations"]) == 1


@pytest.mark.parametrize("changes", [
    {"results": {"frontend": {"state": "completed"}}},
    {"inputs": {"dependency_backend": "cpu"}},
    {"history": [{"stage": "frontend", "sha256": "old"}]},
    {"last_job": {"task_id": "previous-worker"}}, {"job": {"task_id": "active-worker"}},
    {"status": "running"}, {"status": "paused"}, {"index": 1}])
def test_started_or_decided_workflow_is_never_silently_rebound(manager, changes):
    project = create(manager)
    state = put_state(manager, project["id"], **changes)
    workflow = manager._path(project["id"])
    original = workflow.read_bytes()
    project_hash = digest(state["project"])
    manager._implementation_id = "new-tested-implementation"
    with pytest.raises(ValueError, match="라이브러리 코드가 변경"):
        manager._guard_config(state)
    assert workflow.read_bytes() == original and digest(state["project"]) == project_hash
    assert not (Path(state["project"]).parent / ".studio" / "migrations").exists()
    assert not manager.workers.items


@pytest.mark.parametrize("obstruction", ["lock", "old_job", "config"])
def test_unstarted_update_refuses_locks_old_job_files_or_changed_config(manager, obstruction):
    project = create(manager)
    state = manager._load(project["id"])
    root = Path(state["project"]).parent
    if obstruction == "lock":
        path = root / "features" / ".writer.lock"
        write_json(path, {"pid": 12345})
    elif obstruction == "old_job":
        path = root / ".studio" / "jobs" / "previous" / "request.json"
        write_json(path, {"task_id": "old-attempt"})
    else:
        path = Path(state["project"])
        config = read_json(path)
        config["wakeword"] = "다른호출어"
        write_json(path, config)
    before = path.read_bytes()
    workflow = manager._path(project["id"])
    previous = workflow.read_bytes()
    manager._implementation_id = "new-tested-implementation"
    with pytest.raises(ValueError):
        manager._guard_config(state)
    assert path.read_bytes() == before and workflow.read_bytes() == previous
    assert not (root / ".studio" / "migrations").exists()
    assert not manager.workers.items


def evaluation_only_queue(manager, project_id):
    _, queue = reference_queue(manager, project_id)
    state = manager._load(project_id)
    root = Path(state["project"]).parent
    row = json.loads(queue.read_text(encoding="utf-8"))
    pool = root / "review" / "held_out_sources.jsonl"
    pool.write_text(json.dumps({**row, "provenance": "public_dataset", "split": "validation", "category": "environment"}) + "\n", encoding="utf-8")
    descriptor = {"id": "fixture", "kind": "content", "queue": str(queue), "queue_sha256": digest(queue),
                  "pool_manifest": str(pool), "pool_sha256": digest(pool), "evaluation_only": True, "pools": []}
    state["results"]["fixture_review"]["reviews"] = [descriptor]
    manager._save(state)
    return root, descriptor


def test_evaluation_only_review_consent_and_saved_evidence_never_approve_training(manager, http_server):
    project = create(manager)
    root, _ = evaluation_only_queue(manager, project["id"])
    approval_hash = digest(root / "approval.json")
    status, content, _ = http_server(f"/review/queue?project={project['id']}&queue=fixture")
    assert status == 200
    payload = json.loads(content)
    assert "학습에는 사용하지 않고" in payload["policy_text"]
    assert "소음 평가에만" in payload["policy_text"]
    assert payload["evaluation_only"] is True and payload["pools"] == []
    result = manager.save_review(project["id"], "fixture", {
        "decisions": {"fixture-1": {"decision": "keep", "listened": True}}, "policy_approved": True})
    assert result["evaluation_only"] is True and result["pools"] == []
    assert result["intended_use"] == "held_out_development_noise" and "admission" not in result
    state = manager._load(project["id"])
    evidence = read_json(state["inputs"]["review_answers"]["fixture"]["path"])
    assert evidence["source_scope"] == "Human dispositions entered through the local review screen"
    assert evidence["evaluation_only"] is True and evidence["pools"] == []
    assert evidence["intended_use"] == "held_out_development_noise"
    assert evidence["consent_scope"] == "Held-out Development noise evaluation only; not for training"
    assert digest(root / "approval.json") == approval_hash
    assert not (root / "review" / "content_admission.json").exists()


def test_evaluation_only_descriptor_cannot_request_training_pool_admission(manager):
    project = create(manager)
    _, descriptor = evaluation_only_queue(manager, project["id"])
    state = manager._load(project["id"])
    descriptor["pools"] = ["noise"]
    state["results"]["fixture_review"]["reviews"] = [descriptor]
    manager._save(state)
    before = manager._path(project["id"]).read_bytes()
    with pytest.raises(ValueError, match="평가 전용"):
        manager.save_review(project["id"], "fixture", {
            "decisions": {"fixture-1": {"decision": "keep", "listened": True}}, "policy_approved": True})
    assert manager._path(project["id"]).read_bytes() == before
