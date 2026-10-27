"""Studio stages must call real public operations and stop at human gates."""
from copy import deepcopy
from pathlib import Path

import pytest

from korean_openwakeword import studio_steps as steps
from korean_openwakeword.common import digest, identity, read_json, read_rows, write_json
from korean_openwakeword.hard_phrase_policy import validate_noncall_text
from korean_openwakeword.project import BATCH, create_project
from korean_openwakeword.review import apply_review


@pytest.fixture
def project(tmp_path):
    budget = steps.preset_settings("check")
    path = create_project(tmp_path / "bora", "보라야", "bora", positive_originals=budget["originals"],
                          augmentation_rounds=budget["rounds"], updates=budget["updates"],
                          generic_hours=budget["generic_hours"], korean_hours=budget["korean_hours"])
    return path


def run(project, stage, context=None, settings=None):
    return steps.run_stage(project, project.parent.parent, settings or steps.preset_settings("check"), stage, context)


def fake_audio(root, name, **metadata):
    path = root / "test_audio" / (name + ".wav")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes((name + " distinct fake audio").encode())
    return {"sample_id": name, "path": str(path.relative_to(root)), "sha256": digest(path), **metadata}


def approve(project, queue, kind):
    decisions = {r["sample_id"]: {"decision": "keep", "listened": True} for r in read_rows(queue)}
    return apply_review(project, queue, decisions, kind)


@pytest.mark.parametrize("name, counts", [("check", (1000, 1, 10000, 2000, 1)),
                                          ("trial", (20000, 3, 50000, 2000, 2000 / 3)),
                                          ("production", (100000, 5, 100000, 3500, 1500))])
def test_presets_are_explicit_not_changes_to_global_defaults(name, counts):
    settings = steps.preset_settings(name)
    assert tuple(settings[k] for k in ("originals", "rounds", "updates", "generic_hours", "korean_hours")) == counts
    assert settings["batch"] == BATCH and settings["final_locked"]
    assert settings["noise_material_policy"] == ("reuse_negative_waveforms" if name == "production" else "official_example")
    assert settings["material_review_policy"] == "library_fixed_sources_v1"
    assert "common_download_policy" not in settings
    settings["batch"]["positive"] = -1
    assert steps.preset_settings(name)["batch"]["positive"] == 100


def test_common_evaluation_is_identical_for_all_training_budgets_and_cannot_drift():
    settings = [steps.preset_settings(name) for name in ("check", "trial", "production")]
    assert all(s["evaluation_policy"] == steps.common_evaluation_policy() for s in settings)
    assert all((s["validation_references"], s["validation_originals"], s["noise_sections"]) == (30, 2000, 1) for s in settings)
    assert settings[0]["originals"] == 1000 and settings[-1]["originals"] == 100000
    changed = settings[0]
    changed["validation_originals"] = 100
    with pytest.raises(ValueError, match="공통 평가"):
        steps.workflow_plan(changed)
    changed.pop("evaluation_policy")
    changed["validation_references"] = 2
    assert steps.workflow_plan(changed)  # Saved legacy workflows retain their original protocol.


def test_plan_is_ordered_and_has_no_final_unlock():
    plan = steps.workflow_plan(steps.preset_settings())
    ids = [r["id"] for r in plan]
    assert len(ids) == len(set(ids))
    assert ids.index("references_review") < ids.index("pilot") < ids.index("pilot_review") < ids.index("positives")
    assert ids.index("hard_review") < ids.index("features") < ids.index("training") < ids.index("evaluation")
    assert all(r["kind"] in ("task", "review") and r["label"] for r in plan)
    assert not any("final" in i or "freeze" in i for i in ids)


@pytest.mark.parametrize("stage", ["data_download", "data_connect"])
@pytest.mark.parametrize("policy_location", ["settings", "config"])
def test_retired_common_download_policy_blocks_new_preparation_without_migration(project, monkeypatch, stage, policy_location):
    from korean_openwakeword import studio_data
    settings = steps.preset_settings("check")
    if policy_location == "settings":
        settings["common_download_policy"] = "selected_common_zip_v1"
    else:
        config = read_json(project)
        config["common_download_policy"] = "selected_common_zip_v1"
        write_json(project, config)
    before = project.read_bytes()
    monkeypatch.setattr(studio_data, "prepare_materials", lambda *a, **k: pytest.fail("Retired download must not start"))
    with pytest.raises(ValueError, match="0.16"):
        run(project, stage, settings=settings)
    assert project.read_bytes() == before
    assert not (project.parent / ".studio").exists()


def test_retired_saved_common_policy_keeps_completed_workflow_plan_readable():
    settings = steps.preset_settings("check")
    settings["common_download_policy"] = "selected_common_zip_v1"
    assert len(steps.workflow_plan(settings)) == 19


def test_fixed_material_policy_keeps_nineteen_stages_and_legacy_review_plan():
    settings = steps.preset_settings()
    plan = steps.workflow_plan(settings)
    legacy_settings = dict(settings)
    legacy_settings.pop("material_review_policy")
    legacy = steps.workflow_plan(legacy_settings)
    assert len(plan) == len(legacy) == 19
    assert [stage["id"] for stage in plan] == [stage["id"] for stage in legacy]
    current = {stage["id"]: stage for stage in plan}
    historical = {stage["id"]: stage for stage in legacy}
    assert current["data_review"] == {"id": "data_review", "label": "고정 자료 자동 확인",
                                       "title": "고정 자료 자동 확인", "kind": "task"}
    assert historical["data_review"]["kind"] == "review"
    assert historical["data_review"]["label"] == "학습 · 평가 자료 확인"
    assert current["data_connect"]["label"] == "자료를 용도별로 연결"
    assert historical["data_connect"]["label"] == "검수한 자료를 용도별로 연결"
    assert all(current[sid] == stage for sid, stage in historical.items()
               if sid not in ("data_review", "data_connect"))


def test_unknown_material_review_policy_fails_before_stage_work(project):
    settings = steps.preset_settings()
    settings["material_review_policy"] = "auto_accept_everything"
    with pytest.raises(ValueError, match="material review policy"):
        steps.workflow_plan(settings)
    with pytest.raises(ValueError, match="material review policy"):
        run(project, "data_review", settings=settings)
    assert not (project.parent / ".studio").exists()


@pytest.mark.parametrize("word", ["보라야", "춘식아", "헤이집사", "아", "가나"])
def test_phrase_suggestions_never_contain_complete_call_name(word):
    phrases = steps.propose_hard_phrases(word)
    assert len(phrases) == len(set(phrases)) == 8
    for phrase in phrases:
        validate_noncall_text(phrase, word)


def test_no_generation_before_explicit_phrase_choice(project):
    result = run(project, "hard_phrases")
    assert result["state"] == "waiting_input"
    assert result["input_kind"] == "hard_phrases"
    assert not (project.parent / "data/generated/hard").exists()
    result = run(project, "hard_generation")
    assert result["state"] == "waiting_input"


def test_explicit_phrase_choice_uses_existing_shared_guard(project):
    context = {"inputs": {"hard_phrases": ["보리야", "돌아와"], "hard_phrases_approved": True}}
    result = run(project, "hard_phrases", context)
    assert result["state"] == "completed" and result["result"]["phrases"] == ["보리야", "돌아와"]
    for bad in (["보라를"], ["보라야"], ["보리야", "보리야"]):
        context["inputs"]["hard_phrases"] = bad
        with pytest.raises(ValueError):
            run(project, "hard_phrases", context)


def test_workload_drift_and_final_freeze_stop_before_any_action(project):
    changed = steps.preset_settings()
    changed["originals"] = 999
    with pytest.raises(ValueError, match="frozen workload"):
        run(project, "frontend", settings=changed)
    write_json(project.parent / "final_lock.json", {"a": "real lock presence"})
    result = run(project, "references")
    assert result["state"] == "waiting_input" and "Final" in result["message"]


def test_frontend_install_failure_is_actionable_and_does_not_download(project, monkeypatch):
    from korean_openwakeword import assets, benchmark
    monkeypatch.setattr(benchmark, "doctor", lambda **kw: {"tts_check": {"ready": False, "reasons": ["missing"]}})
    monkeypatch.setattr(assets, "prepare_frontend", lambda *a: pytest.fail("not ready; must not download"))
    result = run(project, "frontend")
    assert result["state"] == "waiting_input" and result["check"]["tts_check"]["reasons"] == ["missing"]
    assert result["input_kind"] == "dependency_setup"


def test_dependency_plan_uses_matching_cpu_or_gpu_pair_and_never_replaces_library_code():
    versions = {"torch": "2.8.0+cu128", "torchaudio": "2.8.0+cu128", "qwen-tts": "0.1.1",
                "onnx": "1.20.0", "openwakeword": "0.6.0"}
    assert steps._dependency_plan(versions, "cu128") == []
    commands = steps._dependency_plan({}, "cpu")
    assert len(commands) == 3
    assert commands[0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    suffix = "" if steps.sys.platform == "darwin" else "+cpu"
    assert "torch==2.8.0" + suffix in commands[0] and "torchaudio==2.8.0" + suffix in commands[0]
    assert commands[-1][-2:] == ["openwakeword==0.6.0", "--no-deps"]
    assert not any("korean-openwakeword" in arg or arg == "." for cmd in commands for arg in cmd)
    with pytest.raises(ValueError):
        steps._dependency_plan({}, "arbitrary-shell-command")


def test_cpu_to_gpu_choice_requires_cuda_build_even_when_plain_version_matches():
    cpu = {"torch": "2.8.0+cpu", "torchaudio": "2.8.0+cpu", "qwen-tts": "0.1.1", "onnx": "1.20.0", "openwakeword": "0.6.0"}
    commands = steps._dependency_plan(cpu, "cu128")
    assert len(commands) == 1 and "torch==2.8.0+cu128" in commands[0] and "torchaudio==2.8.0+cu128" in commands[0]
    gpu = dict(cpu, torch="2.8.0+cu128", torchaudio="2.8.0+cu128")
    cpu_commands = steps._dependency_plan(gpu, "cpu")
    expected = "2.8.0" if steps.sys.platform == "darwin" else "2.8.0+cpu"
    assert len(cpu_commands) == 1 and "torch==" + expected in cpu_commands[0] and "torchaudio==" + expected in cpu_commands[0]


def test_automatic_install_refuses_global_python(project, monkeypatch):
    monkeypatch.setattr(steps.sys, "prefix", steps.sys.base_prefix)
    monkeypatch.setattr(steps.subprocess, "run", lambda *a, **k: pytest.fail("No global pip install"))
    result = steps._install_dependencies(project.parent, "cpu")
    assert result["state"] == "waiting_input" and result["virtual_environment_required"] is True


def test_explicit_install_uses_current_venv_and_project_cache_only(project, monkeypatch):
    import os
    from korean_openwakeword import gpu_runtime_install
    gpu_calls = []
    def gpu_setup(root):
        gpu_calls.append(root)
        return {"state": "ready", "target": str(root / "cache/gpu_runtime")}
    monkeypatch.setattr(gpu_runtime_install, "setup_gpu_runtime", gpu_setup)
    monkeypatch.setattr(steps.sys, "prefix", str(project.parent / ".venv"))
    monkeypatch.delenv("KOWW_CACHE_ROOT", raising=False)
    monkeypatch.setattr(steps, "_installed_dependencies", lambda: {})
    commands = []
    def pip(command, **kwargs):
        assert command[:3] == [steps.sys.executable, "-m", "pip"]
        assert kwargs == {"cwd": str(project.parent), "check": True}
        assert Path(os.environ["PIP_CACHE_DIR"]).is_relative_to(project.parent)
        assert Path(os.environ["TMP"]).is_relative_to(project.parent)
        commands.append(command)
    monkeypatch.setattr(steps.subprocess, "run", pip)
    result = steps._install_dependencies(project.parent, "cu128")
    assert gpu_calls == [project.parent] and result["evaluation_gpu_runtime"]["state"] == "ready"
    assert result["state"] == "completed" and len(commands) == 3
    assert commands[0][-1] == "https://download.pytorch.org/whl/cu128"


def test_dependency_setup_rechecks_after_explicit_choice_and_then_prepares_frontend(project, monkeypatch):
    from korean_openwakeword import assets, benchmark
    installed = []
    monkeypatch.setattr(steps, "_install_dependencies", lambda root, backend: installed.append(backend) or {"state": "completed"})
    monkeypatch.setattr(steps, "_installed_dependencies", lambda: {"onnx": "1.20.0"})
    monkeypatch.setattr(benchmark, "doctor", lambda **kw: {"tts_check": {"ready": True}, "reference_openwakeword_ready": True})
    monkeypatch.setattr(assets, "prepare_frontend", lambda path: {"pinned": True})
    result = run(project, "frontend", {"inputs": {"dependency_backend": "cu128"}})
    assert result["state"] == "completed" and installed == ["cu128"]


def test_references_are_combined_once_and_require_actual_saved_answers(project, monkeypatch):
    settings = steps.preset_settings()
    settings.pop("reference_admission_policy")  # Frozen legacy workflows keep the listening gate.
    from korean_openwakeword import tts
    calls = []
    shared = object()
    loads = []
    def load(*args):
        loads.append(args)
        return shared
    def generate(path, split, limit, *, model=None):
        assert model is shared
        calls.append((split, limit))
        rows = [fake_audio(path.parent, split + str(i), voice_family_id=split + str(i), split=split)
                for i in range(limit)]
        queue = steps._publish_rows(path.parent / f"data/references/{split}/references.jsonl", rows)
        return {"generated": len(rows), "review_queue": str(queue)}
    monkeypatch.setattr(tts, "generate_references", generate)
    monkeypatch.setattr(tts, "_load_model", load)
    result = run(project, "references")
    assert calls == [("train", 20), ("validation", 30)]
    assert len(loads) == 1
    assert result["result"]["generated"] == 50
    queue = Path(result["result"]["queue"])
    assert queue.parent == project.parent / ".studio"
    pending = run(project, "references_review", settings=settings)
    assert pending["state"] == "waiting_review" and pending["reviews"][0]["count"] == 50
    assert read_json(project.parent / "approval.json")["generation_policy_approved"] is False
    approve(project, queue, "references")
    assert run(project, "references_review", settings=settings)["state"] == "completed"
    # A changed exact queue cannot reuse the previous listening answers.
    queue.write_text(queue.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert run(project, "references_review", settings=settings)["state"] == "waiting_review"


def test_saved_boolean_is_not_a_review_or_generation_approval(project):
    settings = steps.preset_settings()
    settings.pop("reference_admission_policy")
    root = project.parent
    row = fake_audio(root, "ref", voice_family_id="voice")
    queue = steps._publish_rows(root / ".studio/reference_queue.jsonl", [row])
    write_json(root / "review/references_results.json", {"queue_sha256": digest(queue),
               "decisions": {"ref": {"decision": "keep", "listened": True}}})
    assert run(project, "references_review", settings=settings)["state"] == "waiting_review"


def test_discarded_reference_is_excluded_but_remaining_train_and_validation_can_proceed(project):
    settings = steps.preset_settings()
    settings.pop("reference_admission_policy")
    root = project.parent
    rows = [fake_audio(root, name, voice_family_id=name, split=split)
            for name, split in (("train1", "train"), ("train2", "train"), ("val", "validation"))]
    queue = steps._publish_rows(root / ".studio/reference_queue.jsonl", rows)
    decisions = {r["sample_id"]: {"decision": "keep", "listened": True} for r in rows}
    decisions["train2"] = {"decision": "discard", "listened": True}
    apply_review(project, queue, decisions, "references")
    assert run(project, "references_review", settings=settings)["state"] == "completed"
    assert "train2" not in read_json(root / "approval.json")["accepted_reference_families"]
    decisions["train1"] = {"decision": "discard", "listened": True}
    apply_review(project, queue, decisions, "references")
    assert run(project, "references_review", settings=settings)["state"] == "waiting_review"


def test_pilot_queue_contains_only_actual_technical_passes(project, monkeypatch):
    settings = steps.preset_settings()
    settings.pop("pilot_review_policy")  # Historical fixed-count queues retain their original gate.
    from korean_openwakeword import tts_runtime
    config = read_json(project)
    rows = [fake_audio(project.parent, f"pilot{i}", slot=i, qc={"passed": True},
                       tts_settings_id=identity(config["tts"]), generation_method="clone") for i in range(24)]
    rows.append(fake_audio(project.parent, "failed", slot=0, qc={"passed": False}))
    candidates = steps._publish_rows(project.parent / "candidates.jsonl", rows)
    calls = []
    def generate(*args):
        calls.append(args)
        return {"full_target_reached": True, "review_queue": str(candidates), "technical_pass": 24}
    monkeypatch.setattr(tts_runtime, "generate_with_runtime", generate)
    result = run(project, "pilot", settings=settings)
    assert calls == [(project, "train", 24, True, "clone")]
    queue = result["result"]["queue"]
    assert len(read_rows(queue)) == 24
    assert run(project, "pilot_review", settings=settings)["state"] == "waiting_review"
    approve(project, queue, "pilot")
    assert run(project, "pilot_review", settings=settings)["state"] == "completed"


def test_incomplete_generation_stops_without_approving_failed_slots(project, monkeypatch):
    settings = steps.preset_settings()
    settings.pop("pilot_review_policy")
    from korean_openwakeword import tts_runtime
    monkeypatch.setattr(tts_runtime, "generate_with_runtime", lambda *a: {"full_target_reached": False, "technical_pass": 23})
    result = run(project, "pilot", settings=settings)
    assert result["state"] == "waiting_input"
    assert not (project.parent / ".studio/pilot_queue.jsonl").exists()


def test_positive_generation_uses_standalone_train_and_disjoint_validation_only(project, monkeypatch):
    from korean_openwakeword import tts, tts_runtime
    calls = []
    models = []
    def generate(*args, **kwargs):
        calls.append(args)
        models.append(kwargs["model"])
        return {"full_target_reached": True, "technical_pass": args[2]}
    shared = object()
    monkeypatch.setattr(tts_runtime, "CloneRuntime", lambda *a: shared)
    monkeypatch.setattr(tts, "generate", generate)
    result = run(project, "positives")
    assert calls == [(project, "train", 1000, False, "clone"), (project, "validation", 2000, False, "clone")]
    assert models == [shared, shared]
    assert result["result"]["independent_STT_runs"] == 0
    assert result["result"]["validation_originals"] == 2000
    assert not (project.parent / "data/generated/final").exists()


def test_features_reuse_generic_and_resume_actual_contracts_without_reducing_n(project, monkeypatch):
    from korean_openwakeword import features
    config = read_json(project)
    generic = project.parent / config["pool_features"]["generic_negative"]
    write_json(generic / "metadata.json", {"feature_rows": 5625000})
    calls = []
    monkeypatch.setattr(features, "verify_cache", lambda p: {"feature_rows": 5625000})
    def build(path, role, **kwargs):
        calls.append((role, kwargs["resume"]))
        destination = path.parent / config["pool_features"][role]
        write_json(destination / "contract.json", {"role": role})
        return {"feature_rows": 1000 if role == "positive" else 100}
    monkeypatch.setattr(features, "build_features", build)
    result = run(project, "features")
    assert calls == [("korean_negative", False), ("positive", False), ("hard_negative", False)]
    assert result["result"]["positive_actual_rows"] == result["result"]["positive_expected_rows"] == 1000
    calls.clear()
    run(project, "features")
    assert calls == [("korean_negative", True), ("positive", True), ("hard_negative", True)]


def test_wrong_n_times_rounds_never_marks_features_completed(project, monkeypatch):
    from korean_openwakeword import features
    monkeypatch.setattr(features, "build_features", lambda *a, **k: {"feature_rows": 999})
    with pytest.raises(ValueError, match="N × rounds"):
        run(project, "features")


def test_training_resume_only_with_member_contracts(project, monkeypatch):
    from korean_openwakeword import train
    calls = []
    def train_run(path, directory, **kwargs):
        calls.append(kwargs)
        return {"directory": str(directory), "formal_training": True}
    monkeypatch.setattr(train, "train", train_run)
    assert run(project, "training")["state"] == "completed"
    assert calls[-1] == {"resume": False}
    write_json(project.parent / "runs/studio_run/member1_seed1/contract.json", {})
    run(project, "training")
    assert calls[-1] == {"resume": True}


def test_partial_immutable_outputs_are_retained_instead_of_overwritten(tmp_path):
    directory = tmp_path / "package"
    directory.mkdir()
    (directory / "partial.onnx").write_bytes(b"keep")
    next_path, complete = steps._next_directory(directory, "receipt.json")
    assert next_path == tmp_path / "package_attempt2" and complete is False
    assert (directory / "partial.onnx").read_bytes() == b"keep"


def test_old_completed_output_with_different_recipe_is_retained_as_separate_version(tmp_path):
    directory = tmp_path / "paired"
    directory.mkdir()
    (directory / "manifest.jsonl").write_text("legacy immutable evidence\n")
    write_json(directory / "recipe.json", {"noise_selection": "legacy"})
    next_path, complete = steps._next_directory(directory, "manifest.jsonl",
        lambda p: read_json(p / "recipe.json")["noise_selection"] == "balanced_sources_v1")
    assert next_path == tmp_path / "paired_attempt2" and not complete
    assert (directory / "manifest.jsonl").read_text() == "legacy immutable evidence\n"


@pytest.mark.parametrize("preset", ["check", "trial", "production"])
@pytest.mark.parametrize("policy_version", [2, 3, 4, 5])
def test_studio_development_selects_balanced_policy_and_preserves_old_pair_outputs(project, monkeypatch, preset, policy_version):
    from korean_openwakeword import paired
    from korean_openwakeword import evaluation_materials
    material_calls = []
    def coverage(rows, **kwargs):
        material_calls.append(kwargs)
        return {"target_met": True, "shortages": []}
    monkeypatch.setattr(evaluation_materials, "evaluation_material_coverage", coverage)
    settings = steps.preset_settings(preset)
    policy = steps.common_evaluation_policy(policy_version)
    settings.update(evaluation_policy=policy, **{key: policy[key] for key in
                    ("validation_references", "validation_originals", "noise_sections")})
    project = create_project(project.parent.parent / preset, "보라야", "bora", positive_originals=settings["originals"],
                             augmentation_rounds=settings["rounds"], updates=settings["updates"],
                             generic_hours=settings["generic_hours"], korean_hours=settings["korean_hours"])
    root = project.parent
    config = read_json(project)
    steps._publish_rows(root / config["manifests"]["positive_validation"],
                        [{"sample_id": f"validation_{i:06d}"} for i in range(settings["validation_originals"])], allow_initial_empty=True)
    negative = fake_audio(root, "heldout", kind="negative", split="validation")
    negative_manifest = steps._publish_rows(root / ".studio/negative.jsonl", [negative])
    old = root / "reports/studio_paired"
    old.mkdir()
    (old / "manifest.jsonl").write_text("old completed pair evidence\n")
    calls = []
    def prepare(path, output, **kwargs):
        calls.append(kwargs)
        output.mkdir()
        recipe = paired.evaluation_recipe(path, **kwargs)
        write_json(output / "recipe.json", recipe)
        manifest = steps._publish_rows(output / "manifest.jsonl", [])
        write_json(output / "noise_coverage.json", {"test": "coverage sidecar"})
        return {"manifest": str(manifest), "noise_selection_policy": recipe["noise_selection"],
                "noise_coverage": {"path": str(output / "noise_coverage.json"), "sha256": digest(output / "noise_coverage.json")}}
    monkeypatch.setattr(paired, "prepare_evaluation", prepare)
    result = run(project, "development", {"results": {"data_connect": {"result": {"negative_validation_manifest": str(negative_manifest)}}}}, settings=settings)
    assert result["state"] == "completed" and calls[0]["noise_selection"] == "balanced_sources"
    assert calls[0]["originals"] == settings["validation_originals"] and calls[0]["noise_sections"] == 1
    assert calls[0]["reverb_originals"] == settings["evaluation_policy"]["reverb_originals"]
    if policy_version >= 3:
        assert calls[0]["reverb_selection"] == "metadata_axes"
    else:
        assert "reverb_selection" not in calls[0]
    assert (old / "manifest.jsonl").read_text() == "old completed pair evidence\n"
    assert (root / "reports/studio_paired_attempt2/recipe.json").is_file()
    assert result["result"]["noise_coverage"]
    assert result["result"]["evaluation_policy_sha256"] == identity(settings["evaluation_policy"])
    assert calls[0]["snrs"] == (20, 15, 10, 5, 0)
    assert material_calls == ([{"strict_common_pool": True}] if policy_version >= 4 else [{}])


@pytest.mark.parametrize("policy_version", [4, 5])
def test_common_evaluation_refuses_a_short_positive_pool_before_building_wavs(project, monkeypatch, policy_version):
    from korean_openwakeword import paired
    negative = fake_audio(project.parent, "background", kind="negative", split="validation")
    manifest = steps._publish_rows(project.parent / ".studio/negative.jsonl", [negative])
    monkeypatch.setattr(paired, "prepare_evaluation", lambda *a, **k: pytest.fail("Short pool cannot start full common evaluation"))
    context = {"results": {"data_connect": {"result": {"negative_validation_manifest": str(manifest)}}}}
    settings = steps.preset_settings()
    policy = steps.common_evaluation_policy(policy_version)
    settings.update(evaluation_policy=policy, **{key: policy[key] for key in
                    ("validation_references", "validation_originals", "noise_sections")})
    config = read_json(project)
    steps._publish_rows(project.parent / config["manifests"]["positive_validation"],
        [{"sample_id": f"validation_{i:06d}"} for i in range(policy["validation_originals"] - 1)], allow_initial_empty=True)
    with pytest.raises(ValueError, match=f"{policy['validation_originals']:,}개"):
        run(project, "development", context, settings=settings)


def test_complete_common_evaluation_is_not_relabeled_as_a_smoke_run_for_check_budget(project, monkeypatch):
    from korean_openwakeword import evaluate
    bundle = project.parent / "models/fixture_bundle"
    bundle.mkdir(parents=True)
    write_json(bundle / "manifest.json", {"fixture": True})
    manifest = steps._publish_rows(project.parent / ".studio/fixture_development.jsonl", [])
    calls = []
    def score(path, selected_bundle, output, **kwargs):
        calls.append(kwargs)
        write_json(output / "report.json", {"models": [], "smoke_test_only": kwargs["smoke"]})
        write_json(output / "receipt.json", {})
        write_json(output / "clips.json", [])
        return read_json(output / "report.json")
    monkeypatch.setattr(evaluate, "evaluate", score)
    context = {"results": {"package": {"result": {"bundle": str(bundle)}},
                           "development": {"result": {"manifest": str(manifest)}}}}
    output = run(project, "evaluation", context)
    result = output["result"]
    assert calls[0]["smoke"] is False
    assert result["functional_check_only"] is True  # The training budget remains the check preset.
    assert result["evaluation_functional_check_only"] is False
    assert result["evaluation_policy_sha256"] == identity(steps.common_evaluation_policy())
    assert result["common_evaluation_assessment"]["coverage_targets_passed"] is False
    assert result["common_evaluation_assessment"]["independent_Final_completed"] is False
    assert any(Path(path).name.startswith("common_coverage_") and Path(path).is_file() for path in output["artifacts"])


def test_hard_representatives_cover_every_phrase_voice_pair(project):
    root = project.parent
    rows = []
    for i in range(12):
        rows.append(fake_audio(root, f"hard{i}", slot=i, technical_pass=True,
                    dataset_id="qwen_bora", text=("보리야", "돌아와")[i % 2],
                    voice_family_id=("voice1", "voice2")[(i // 2) % 2]))
    steps._publish_rows(root / "data/generated/hard/candidates.jsonl", rows)
    legacy = steps.preset_settings()
    legacy.pop("hard_review_policy")
    result = run(project, "hard_review", settings=legacy)
    descriptor = result["reviews"][0]
    assert result["state"] == "waiting_review"
    assert descriptor["count"] == 4
    assert len(read_rows(descriptor["pool_manifest"])) == 12
    assert not (root / "data/negative/hard.jsonl").stat().st_size


def material_queue(root, name):
    row = fake_audio(root, name)
    source = steps._publish_rows(root / ".studio" / (name + "_sources.jsonl"), [row])
    queue = steps._publish_rows(root / ".studio" / (name + "_queue.jsonl"), [row])
    return {"id": name, "kind": "content", "queue": str(queue), "pool_manifest": str(source),
            "preparation": str(root / ".studio" / (name + "_preparation.json")), "pools": ["noise"]}


def test_fixed_material_review_uses_verified_source_policy_without_inventing_listening(project, monkeypatch):
    from korean_openwakeword import studio_data
    root = project.parent
    fixed = material_queue(root, "fixed")
    prepared = {"review_queues": [fixed], "shortages": []}
    context = {"results": {"data_download": {"result": prepared}}, "inputs": {}}
    before = deepcopy(context)
    calls = []

    def manual_queues(path, data, policy=None):
        calls.append((path, data, policy))
        return []  # Source pin/technical admission verification belongs to studio_data.

    monkeypatch.setattr(studio_data, "material_review_queues", manual_queues, raising=False)
    result = run(project, "data_review", context)
    assert result["state"] == "completed"
    assert result["result"]["fixed_source_queue_count"] == 1
    assert result["result"]["human_listened_count"] == 0
    assert result["result"]["material_review_policy"] == "library_fixed_sources_v1"
    assert "reviewed_queues" not in result["result"]
    assert "reviews" not in result
    assert result["artifacts"] == [fixed["queue"]]
    assert calls == [(project, prepared, "library_fixed_sources_v1")]
    assert context == before
    assert not list((root / ".studio").glob("*answers*"))
    assert read_json(root / "approval.json")["state"] == "pending"


def test_fixed_material_policy_preserves_manual_gate_for_custom_sources(project, monkeypatch):
    from korean_openwakeword import studio_data
    root = project.parent
    fixed, custom = material_queue(root, "fixed"), material_queue(root, "custom")
    prepared = {"review_queues": [fixed, custom]}
    monkeypatch.setattr(studio_data, "material_review_queues",
                        lambda path, data, policy=None: [custom], raising=False)
    result = run(project, "data_review", {"results": {"data_download": {"result": prepared}}})
    assert result["state"] == "waiting_review"
    assert [queue["id"] for queue in result["reviews"]] == ["custom"]
    assert result["reviews"][0]["kind"] == "content"
    assert not list((root / ".studio").glob("*answers*"))


def test_legacy_material_policy_does_not_silently_remove_a_human_gate(project, monkeypatch):
    from korean_openwakeword import studio_data
    fixed = material_queue(project.parent, "fixed")
    prepared = {"review_queues": [fixed]}
    settings = steps.preset_settings()
    settings.pop("material_review_policy")
    policies = []

    def manual_queues(path, data, policy=None):
        policies.append(policy)
        return data["review_queues"]

    monkeypatch.setattr(studio_data, "material_review_queues", manual_queues, raising=False)
    result = run(project, "data_review", {"results": {"data_download": {"result": prepared}}}, settings=settings)
    assert result["state"] == "waiting_review"
    assert [queue["id"] for queue in result["reviews"]] == ["fixed"]
    assert policies == [None]


def test_fixed_material_policy_still_refuses_missing_preparation_queues(project, monkeypatch):
    from korean_openwakeword import studio_data
    monkeypatch.setattr(studio_data, "material_review_queues", lambda *a, **k: pytest.fail("No source exists to confirm"), raising=False)
    result = run(project, "data_review", {"results": {"data_download": {"result": {"review_queues": []}}}})
    assert result["state"] == "waiting_input"
    assert result["input_kind"] == "generation_policy"


def test_fixed_material_connection_never_bypasses_existing_incomplete_answers(project, monkeypatch):
    from korean_openwakeword import studio_data
    root = project.parent
    fixed = material_queue(root, "fixed")
    prepared = {"review_queues": [fixed], "shortages": [], "marker": "frozen downloaded preparation"}
    answer = root / ".studio" / "existing_answers.json"
    write_json(answer, {"decisions": {"fixed": {"decision": "discard", "listened": True}}})
    binding = {"path": str(answer), "sha256": digest(answer)}
    context = {"results": {"data_download": {"result": prepared}},
               "inputs": {"review_answers": {"fixed": binding}}}
    before = deepcopy(context)
    checks, admitted = [], []

    def manual_queues(path, data, policy=None):
        checks.append((path, deepcopy(data), policy))
        return []

    def admit(path, data, answers, **kwargs):
        admitted.append((path, deepcopy(data), answers, kwargs))
        return {"shortages": [], "state": "connected"}

    monkeypatch.setattr(studio_data, "material_review_queues", manual_queues, raising=False)
    monkeypatch.setattr(studio_data, "ensure_evaluation_materials", lambda p, w, data: data)
    monkeypatch.setattr(studio_data, "admit_materials", admit)
    result = run(project, "data_connect", context)
    assert result["state"] == "waiting_review"
    assert checks == [(project, prepared, "library_fixed_sources_v1")]
    assert admitted == []
    assert context == before
    assert digest(answer) == binding["sha256"]


def test_resolved_material_discard_requires_actual_hash_bound_admission(project):
    root = project.parent
    sources = [fake_audio(root, "source1"), fake_audio(root, "source2")]
    manifest = steps._publish_rows(root / ".studio/sources.jsonl", sources)
    previews = [{**r, "sample_id": "preview_" + r["sample_id"],
                 "reviewed_source": {"path": r["path"], "sha256": r["sha256"]}} for r in sources]
    queue = steps._publish_rows(root / ".studio/previews.jsonl", previews)
    descriptor = steps._review_descriptor(root, queue, "content", "materials", pool_manifest=str(manifest),
                                           preparation="actual-preparation-receipt.json")
    answer_path = root / ".studio/actual_answers.json"
    write_json(answer_path, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "test-user-save",
               "source_scope": "Human dispositions entered through the local review screen",
               "queue_sha256": digest(queue), "source_manifest_sha256": digest(manifest),
               "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews},
               "whole_pool_policy_approved": True,
               "decisions": {previews[0]["sample_id"]: {"decision": "keep", "listened": True},
                             previews[1]["sample_id"]: {"decision": "discard", "listened": True}}})
    context = {"inputs": {"review_answers": {"materials": {"path": str(answer_path), "sha256": digest(answer_path)}}}}
    assert steps._saved_review(root, descriptor, context) is False
    context["inputs"]["review_results"] = {"materials": {"state": "review_saved", "admission_pending": True}}
    assert steps._saved_review(root, descriptor, context) is True
    admission = root / ".studio/actual_admission.json"
    record = {"state": "accepted", "content_review": "accepted",
              "human_answers": {"path": str(answer_path), "sha256": digest(answer_path)},
              "source_manifest": {"path": str(manifest), "sha256": digest(manifest)},
              "representative_queue": {"path": str(queue), "sha256": digest(queue)},
              "accepted_source_sha256": [sources[0]["sha256"]]}
    write_json(admission, record)
    context["inputs"]["review_results"] = {"materials": {"admission": {"path": str(admission), "sha256": digest(admission)}}}
    assert steps._saved_review(root, descriptor, context) is True
    # A receipt that puts the explicitly discarded source back cannot advance.
    record["accepted_source_sha256"].append(sources[1]["sha256"])
    write_json(admission, record)
    context["inputs"]["review_results"]["materials"]["admission"]["sha256"] = digest(admission)
    assert steps._saved_review(root, descriptor, context) is False


@pytest.mark.parametrize("reuse", [False, True])
def test_additional_raw_sources_are_prepared_once_then_reviewed_and_admitted_in_worker(project, monkeypatch, reuse):
    from korean_openwakeword import studio_data
    root = project.parent
    row = fake_audio(root, "additional_original")
    sources = steps._publish_rows(root / ".studio/additional_sources.jsonl", [row])
    previews = [{**row, "reviewed_source": {"path": row["path"], "sha256": row["sha256"]}}]
    queue = steps._publish_rows(root / ".studio/additional_queue.jsonl", previews)
    preparation = root / ".studio/additional_preparation.json"
    write_json(preparation, {"fixture": "actual provider/preparation API covered in studio_data tests"})
    descriptor = {"id": "extra", "kind": "content", "queue": str(queue), "pool_manifest": str(sources),
                  "preparation": str(preparation), "pools": ["korean_negative", "noise"] if reuse else ["korean_negative"]}
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"review_queue": descriptor, "preparation": {"path": str(preparation), "sha256": digest(preparation)}}
    admitted = []
    def admit(path, prepared, answers, **kwargs):
        admitted.append((prepared, answers, kwargs))
        return {"shortages": [], "state": "connected"}
    monkeypatch.setattr(studio_data, "prepare_additional_materials", prepare)
    monkeypatch.setattr(studio_data, "admit_materials", admit)
    monkeypatch.setattr(studio_data, "ensure_evaluation_materials", lambda p, w, prepared: prepared, raising=False)
    context = {"results": {"data_download": {"result": {"review_queues": [], "shortages": ["needs Korean"]}}},
               "inputs": {"additional_sources": [{"directory": str(root / "raw"), "dataset": "zeroth", "pool": "korean_negative", "hours": 1}]}}
    old_cache = root / ".studio" / ("additional_" + identity(context["inputs"]["additional_sources"][0]) + ".json")
    if reuse:
        context["results"]["data_download"]["result"]["noise_material_policy"] = "reuse_negative_waveforms"
        write_json(old_cache, {"source": context["inputs"]["additional_sources"][0],
                              "result": {"review_queue": {**descriptor, "pools": ["korean_negative"]}}})
        before = old_cache.read_bytes()
    pending = run(project, "data_connect", context)
    assert pending["state"] == "waiting_review" and pending["reviews"][0]["id"] == "extra"
    assert len(calls) == 1 and not admitted
    assert calls[0][1] == ({"noise_material_policy": "reuse_negative_waveforms"} if reuse else {})
    if reuse:
        assert old_cache.read_bytes() == before
    answer = root / ".studio/additional_actual_answers.json"
    write_json(answer, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "test-human-save",
               "source_scope": "Human dispositions entered through the local review screen",
               "queue_sha256": digest(queue), "source_manifest_sha256": digest(sources),
               "queued_audio_sha256": {row["sample_id"]: row["sha256"]}, "whole_pool_policy_approved": True,
               "decisions": {row["sample_id"]: {"decision": "keep", "listened": True}}})
    binding = {"path": str(answer), "sha256": digest(answer)}
    context["inputs"].update(review_answers={"extra": binding}, review_results={"extra": {"state": "review_saved", "admission_pending": True}})
    result = run(project, "data_connect", context)
    assert result["state"] == "completed" and len(calls) == 1
    assert admitted[0][0]["review_queues"] == [descriptor]
    assert admitted[0][1] == {str(preparation): binding}


@pytest.mark.parametrize("saved_policy", [None, "reuse_negative_waveforms"])
def test_download_worker_preserves_legacy_policy_and_binds_new_reuse_policy(project, monkeypatch, saved_policy):
    from korean_openwakeword import studio_data
    calls = []
    def prepare(path, workspace, **kwargs):
        calls.append(kwargs)
        return {"review_queues": [], "noise_material_policy": kwargs["noise_material_policy"]}
    monkeypatch.setattr(studio_data, "prepare_materials", prepare)
    settings = steps.preset_settings("check")
    if saved_policy is None:
        settings.pop("noise_material_policy")
    else:
        settings["noise_material_policy"] = saved_policy
    result = run(project, "data_download", settings=settings)
    assert result["state"] == "completed"
    assert calls[0]["noise_material_policy"] == (saved_policy or "official_example")


def test_hard_worker_excludes_whole_discarded_phrase_and_resume_keeps_exact_manifest(project, monkeypatch):
    from korean_openwakeword import pool_review
    root = project.parent
    rows = [fake_audio(root, name, slot=i, technical_pass=True, category="competing_speech", speech=True,
                       dataset_id="qwen_fixture", source_id=name, canonical_source_id=name, split="train",
                       text=text, voice_family_id=voice, provenance="qwen_tts", human_reference_used=False)
            for i, (name, text, voice) in enumerate((("hard0", "도리야", "v0"), ("hard1", "기다려", "v0"), ("hard2", "기다려", "v1")))]
    steps._publish_rows(root / "data/generated/hard/candidates.jsonl", rows)
    legacy = steps.preset_settings()
    legacy.pop("hard_review_policy")
    descriptor = run(project, "hard_review", settings=legacy)["reviews"][0]
    previews = read_rows(descriptor["queue"])
    decisions = {r["sample_id"]: {"decision": "keep" if r["text"] == "도리야" else "discard", "listened": True} for r in previews}
    answers = root / ".studio/hard_actual_answers.json"
    write_json(answers, {"kind": "content_listening_review", "wakeword": "보라야", "saved_at": "test-human-save",
               "source_scope": "Human dispositions entered through the local review screen", "queue_sha256": descriptor["queue_sha256"],
               "source_manifest_sha256": digest(descriptor["pool_manifest"]), "whole_pool_policy_approved": True,
               "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews}, "decisions": decisions})
    context = {"inputs": {"review_answers": {"hard": {"path": str(answers), "sha256": digest(answers)}},
                          "review_results": {"hard": {"state": "review_saved", "admission_pending": True}}}}
    calls = []
    def approve(path, **kwargs):
        assert kwargs["excluded_hard_phrases"] == ["기다려"]
        calls.append(kwargs)
        record = {"state": "accepted", "content_review": "accepted", "wakeword": "보라야",
                  "source_manifest": kwargs["source_manifest"], "representative_queue": kwargs["representative_queue"],
                  "human_answers": kwargs["answers"], "accepted_source_sha256": [rows[0]["sha256"]],
                  "excluded_hard_phrases": kwargs["excluded_hard_phrases"]}
        output = Path(kwargs["output"])
        write_json(output, record)
        return {"admission": {"path": output.relative_to(root).as_posix(), "sha256": digest(output)}}
    monkeypatch.setattr(pool_review, "approve_pool", approve)
    result = run(project, "hard_connect", context, settings=legacy)
    assert result["state"] == "completed" and result["result"]["originals"] == 1
    manifest = root / read_json(project)["manifests"]["hard_negative"]
    before = manifest.read_bytes()
    assert [r["text"] for r in read_rows(manifest)] == ["도리야"]
    assert run(project, "hard_connect", context, settings=legacy)["state"] == "completed"
    assert manifest.read_bytes() == before and len(calls) == 1


def test_mixed_hard_phrase_dispositions_return_queue_without_rewriting_answers(project):
    root = project.parent
    rows = [fake_audio(root, f"mixed{i}", slot=i, technical_pass=True, text="기다려", voice_family_id=f"v{i}", dataset_id="fixture")
            for i in range(2)]
    steps._publish_rows(root / "data/generated/hard/candidates.jsonl", rows)
    legacy = steps.preset_settings()
    legacy.pop("hard_review_policy")
    descriptor = run(project, "hard_review", settings=legacy)["reviews"][0]
    previews = read_rows(descriptor["queue"])
    answers = root / ".studio/hard_actual_answers.json"
    write_json(answers, {"source_scope": "Human dispositions entered through the local review screen", "saved_at": "user-save",
               "queue_sha256": descriptor["queue_sha256"], "source_manifest_sha256": digest(descriptor["pool_manifest"]),
               "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in previews}, "whole_pool_policy_approved": True,
               "decisions": {previews[0]["sample_id"]: {"decision": "keep", "listened": True},
                             previews[1]["sample_id"]: {"decision": "discard", "listened": True}}})
    context = {"inputs": {"review_answers": {"hard": {"path": str(answers), "sha256": digest(answers)}},
                          "review_results": {"hard": {"state": "review_saved", "admission_pending": True}}}}
    before = answers.read_bytes()
    result = run(project, "hard_connect", context, settings=legacy)
    assert result["state"] == "waiting_review" and "판정이 다릅니다" in result["message"]
    assert answers.read_bytes() == before


def test_development_requires_real_heldout_negative_audio(project):
    result = run(project, "development", {"results": {"data_connect": {"result": {}}}})
    assert result["state"] == "waiting_input"
    assert not (project.parent / "reports/studio_paired").exists()


def test_unknown_stages_and_invalid_settings_fail_before_work(project):
    with pytest.raises(ValueError, match="Unknown studio stage"):
        run(project, "freeze")
    with pytest.raises(ValueError):
        steps.preset_settings("mystery")
    settings = steps.preset_settings()
    settings["batch"]["hard_negative"] = 0
    with pytest.raises(ValueError, match="768/256/100/100"):
        steps.workflow_plan(settings)


def test_actual_small_features_train_package_evaluate_site_without_downloads(tmp_path, monkeypatch):
    """Exercise the real post-admission pipeline, never a performance claim.

    Procedural tones stand in for reviewed training audio only in this isolated
    fixture. All real hashes, frontend operations, optimizer updates, exported
    ONNX networks, scoring receipts and exact-WAV site copies are exercised.
    """
    import numpy as np
    import soundfile as sf
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("openwakeword")
    from korean_openwakeword import assets
    from korean_openwakeword.runtime import verify_bundle
    from korean_openwakeword.storage import storage_environment
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("No network in the integration fixture"))
    settings = steps.preset_settings()
    settings.update(originals=2, updates=1, generic_hours=.0001, korean_hours=.0001, validation_originals=1)
    settings.pop("evaluation_policy")  # Explicit tiny integration fixture, not the full common protocol.
    path = create_project(tmp_path / "tiny", "보라야", "bora", positive_originals=2,
                          augmentation_rounds=1, updates=1, generic_hours=.0001, korean_hours=.0001,
                          training_overrides={"device": "cpu", "cpu_threads": 1, "log_every": 1})
    root = path.parent
    config = read_json(path)
    config["augmentation"].update(clean_probability=1, moderate_probability=0, rir_probability=0)
    write_json(path, config)
    assets.prepare_frontend(root / config["frontend_dir"])

    def wave(name, seconds=1, **extra):
        wav = root / "data" / (name + ".wav")
        samples = np.arange(round(seconds * 16000)) / 16000
        frequency = 170 + sum(name.encode()) % 500
        sf.write(wav, .11 * np.sin(2 * np.pi * frequency * samples), 16000, subtype="PCM_16")
        return {"sample_id": name, "source_id": name, "path": str(wav), "sha256": digest(wav), **extra}

    for role in BATCH:
        rows = []
        for i in range(2):
            row = wave(role + str(i), split="train", speech=True)
            if role == "positive":
                row.update(provenance="qwen_tts", human_reference_used=False, text="보라야",
                           voice_family_id="train_fixture", call_start_s=0, call_end_s=1)
            elif role == "hard_negative":
                row.update(provenance="qwen_tts", human_reference_used=False, text="보리야",
                           voice_family_id="train_fixture", category="competing_speech", start_s=0, end_s=1)
            else:
                row.update(provenance="public_dataset", license="isolated-test-fixture",
                           source_url="https://example.test/fixture", category="competing_speech")
            rows.append(row)
        admission = root / "review" / (role + "_fixture.json")
        write_json(admission, {"state": "accepted", "wakeword": "보라야", "human_reference_used": False,
                   "generation_policy_approved": True, "content_review": "accepted",
                   "accepted_source_sha256": [r["sha256"] for r in rows], "scope": "isolated procedural fixture"})
        for row in rows:
            row["admission"] = {"path": str(admission), "sha256": digest(admission)}
        pool = "positive_train" if role == "positive" else role
        steps._publish_rows(root / config["manifests"][pool], rows, allow_initial_empty=True)

    validation = wave("validation_call", split="validation", provenance="qwen_tts", human_reference_used=False,
                      text="보라야", voice_family_id="validation_fixture", call_start_s=0, call_end_s=1)
    steps._publish_rows(root / config["manifests"]["positive_validation"], [validation], allow_initial_empty=True)
    noises = [wave("validation_" + category, split="validation", provenance="public_dataset",
                   license="isolated-test-fixture", source_url="https://example.test/fixture", category=category)
              for category in ("environment", "music", "competing_speech")]
    steps._publish_rows(root / config["manifests"]["noise_validation"], noises, allow_initial_empty=True)
    negative = wave("validation_continuous", seconds=3, split="validation", provenance="public_dataset",
                    license="isolated-test-fixture", source_url="https://example.test/fixture",
                    kind="negative", category="environment", calls=[])
    negative_manifest = steps._publish_rows(root / ".studio/negative_validation.jsonl", [negative])
    context = {"results": {"data_connect": {"result": {"negative_validation_manifest": str(negative_manifest)}}}}
    with storage_environment(root):
        for stage in ("features", "training", "package", "development", "evaluation", "results"):
            result = steps.run_stage(path, tmp_path, settings, stage, context)
            assert result["state"] == "completed", (stage, result)
            context["results"][stage] = result
    result = context["results"]["results"]["result"]
    assert Path(result["site"]).is_file()
    assert len(verify_bundle(result["bundle"])["members"]) == 3
    assert Path(result["selected_bundle"], "manifest.json").is_file()
    report = context["results"]["evaluation"]["result"]["report"]
    assert report["smoke_test_only"] is True and report["stage"] == "development"
    assert report["sessions"] == 17 and len(report["models"]) == 4
    assert context["results"]["development"]["result"]["noise_selection_policy"] == "balanced_sources_v1"
    assert read_json(result["noise_coverage"]["path"])["originals_used"] == 1
    assert result["independent_Final_completed"] is False and not (root / "final_lock.json").exists()
    assert context["results"]["features"]["result"]["positive_actual_rows"] == 2
    assert all(r["exposure"]["positive"]["drawn"] == 100 for r in context["results"]["training"]["result"]["members"])
