"""Real, resumable operations used by the local training studio.

This module chooses a sequence, not an alternative training implementation.
Every audio/model operation delegates to the public library API.  A review
stage only describes a queue; it never invents a human listening decision.
"""
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, write_json
from .hard_phrase_policy import validate_noncall_text
from .project import BATCH, MAX_TRAIN_VOICE_FAMILIES, load_project, require_standard_common_download
from .budgets import DEFAULT_GENERIC_HOURS, DEFAULT_KOREAN_HOURS


_PRESETS = {
    "check": {"title": "동작 확인용", "originals": 1000, "rounds": 1, "updates": 10000,
              "generic_hours": 2000, "korean_hours": 1, "train_references": 20,
              "hard_originals": 100, "functional_check_only": True, "noise_material_policy": "official_example"},
    "trial": {"title": "시험 모델용", "originals": 20000, "rounds": 3, "updates": 50000,
              "generic_hours": 2000, "korean_hours": 2000 / 3, "train_references": 100,
              "hard_originals": 1000, "functional_check_only": False, "noise_material_policy": "official_example"},
    "production": {"title": "상용 모델용", "originals": 100000, "rounds": 5, "updates": 100000,
              "generic_hours": DEFAULT_GENERIC_HOURS, "korean_hours": DEFAULT_KOREAN_HOURS, "train_references": 500,
              "hard_originals": 2000, "functional_check_only": False, "noise_material_policy": "reuse_negative_waveforms"},
}


def common_evaluation_policy(version=5):
    """One declared Development protocol, independent of the training budget.

    These are test coverage targets, not a claim that background-only hours or
    independent Final evidence have already been collected. The policy is
    frozen in each new workflow; old workflows keep their recorded settings.
    """
    previous = {"id": "common_development_v1", "label": "공통 성능 평가",
            "validation_references": 10, "validation_originals": 5000,
            "noise_sections": 1, "snrs": [20, 15, 10, 5, 0],
            "noise_selection": "balanced_sources", "positive_sessions": 80000,
            "development_negative_target_hours": [20, 30],
            "independent_final_negative_min_hours": 100,
            "scope": "Shared across training budgets; Final remains a separate locked holdout"}
    if version == 1:
        return previous
    if version not in (2, 3, 4, 5):
        raise ValueError("Unsupported common evaluation policy")
    expanded = {**previous, "id": "common_development_v2", "validation_references": 30,
            "positive_sessions": 100000, "dry_positive_sessions": 80000,
            "reverb_originals": 625, "reverb_groups": ["moderate", "strong"],
            "reverb_positive_sessions": 20000,
            "development_negative_target_hours": [30, 30],
            "negative_hours_by_bucket": {"korean_conversation": 10, "korean_interview": 5,
                "korean_read": 3, "other_speech": 4, "music": 3, "environment": 4, "electronic": 1},
            "evaluation_materials_recipe": 5, "rir_rooms": 20, "rir_files": 80,
            "evaluation_workers": "auto"}
    if version == 2:
        return expanded
    # New workflows use more representative calls for each RIR group. The
    # earlier 100k protocol remains byte-for-byte reproducible through v2.
    representative = {**expanded, "id": "common_development_v3", "positive_sessions": 112000,
            "reverb_originals": 1000, "reverb_positive_sessions": 32000,
            "reverb_selection": "metadata_axes"}
    if version == 3:
        return representative
    # The 30h benchmark is one corpus, not an addition to the legacy 193
    # clips or training-source Validation reserves. Keep old contracts intact.
    common_only = {**representative, "id": "common_development_v4",
                   "evaluation_materials_recipe": 6, "evaluation_source_scope": "common_30h_only"}
    if version == 4:
        return common_only
    # New workflows use 2,000 originals across all 16 dry conditions, plus
    # the same balanced 500-call subset for both reverberation groups. All
    # backgrounds, RIR sources, voice families and saved v1-v4 policies stay.
    return {**common_only, "id": "common_development_v5", "validation_originals": 2000,
            "positive_sessions": 48000, "dry_positive_sessions": 32000,
            "reverb_originals": 500, "reverb_positive_sessions": 16000}

_STAGES = (
    ("frontend", "환경 점검 · 공통 음성 처리 모델", "task"),
    ("data_download", "공개 자료 다운로드 · 전처리", "task"),
    ("data_review", "학습 · 평가 자료 확인", "review"),
    ("data_connect", "검수한 자료를 용도별로 연결", "task"),
    ("references", "학습 · 평가용 합성 음색 생성", "task"),
    ("references_review", "합성 음색 듣기", "review"),
    ("pilot", "호출어 파일럿 생성", "task"),
    ("pilot_review", "호출어 발음 확인", "review"),
    ("positives", "호출어 음성 생성", "task"),
    ("hard_phrases", "비호출 유사어 문구 확인", "review"),
    ("hard_generation", "비호출 유사어 음성 생성", "task"),
    ("hard_review", "비호출 유사어 듣기", "review"),
    ("hard_connect", "검수한 유사어를 학습에 연결", "task"),
    ("features", "소음 · 울림 증강과 특징 추출", "task"),
    ("training", "모델 학습", "task"),
    ("package", "시험할 모델 묶음 준비", "task"),
    ("development", "공통 평가 · 같은 호출어의 소음별 시험 준비", "task"),
    ("evaluation", "공통 평가 · 단일 모델과 앙상블 채점", "task"),
    ("results", "결과 · 오디오 · 마이크 시험 준비", "task"),
)


def preset_settings(name="check"):
    if name not in _PRESETS:
        raise ValueError("규모는 check / trial / production 중에서 선택하세요.")
    policy = common_evaluation_policy()
    return {"preset": name, **deepcopy(_PRESETS[name]),
            **{key: policy[key] for key in ("validation_references", "validation_originals", "noise_sections")},
            "evaluation_policy": policy, "batch": dict(BATCH),
            "evaluation_backend": "auto", "gpu_batch_size": 128, "gpu_boundary_margin": .001,
            "gpu_numerical_differences_approved": True,
            "material_acquisition_policy": "automatic_v3",
            "material_review_policy": "library_fixed_sources_v1",
            "audio_storage_policy": "canonical_flac_only_v1",
            **({"official_noise_acquisition_policy": "selective_fma_small_v1"} if name in ("check", "trial") else {}),
            **({"fma_acquisition_policy": "per_track_flac_prefetch_v1"} if name == "production" else {}),
            "final_locked": True, "positive_text_policy": "standalone_only"}


def workflow_plan(settings):
    _validate_settings(settings)
    plan = [{"id": sid, "label": title, "title": title, "kind": kind} for sid, title, kind in _STAGES]
    if settings.get("material_review_policy") == "library_fixed_sources_v1":
        for stage in plan:
            if stage["id"] == "data_review":
                stage.update(label="고정 자료 자동 확인", title="고정 자료 자동 확인", kind="task")
            elif stage["id"] == "data_connect":
                stage.update(label="자료를 용도별로 연결", title="자료를 용도별로 연결")
    return plan


def _validate_settings(settings):
    backend = settings.get("evaluation_backend", "cpu")
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("Evaluation backend must be auto, cpu or cuda")
    if backend in ("auto", "cuda"):
        if backend == "cuda" and settings.get("gpu_numerical_differences_approved") is not True:
            raise ValueError("CUDA evaluation requires explicit numerical-difference approval")
        batch = settings.get("gpu_batch_size", 128)
        margin = settings.get("gpu_boundary_margin", .001)
        if type(batch) is not int or not 1 <= batch <= 128:
            raise ValueError("GPU batch size must be an integer from 1 to 128")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or not 0 < margin < 1:
            raise ValueError("GPU boundary margin must be positive and finite below one")
    for key in ("originals", "updates", "train_references", "validation_references",
                "validation_originals", "hard_originals", "noise_sections"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings.get("rounds") not in (1, 3, 5):
        raise ValueError("증강 횟수는 R1 / R3 / R5입니다.")
    if settings.get("noise_material_policy") not in (None, "official_example", "reuse_negative_waveforms"):
        raise ValueError("Unsupported training-noise material policy")
    if settings.get("material_acquisition_policy") not in (None, "automatic_v1", "automatic_v2", "automatic_v3"):
        raise ValueError("Unsupported material acquisition policy")
    if settings.get("material_review_policy") not in (None, "library_fixed_sources_v1"):
        raise ValueError("Unsupported material review policy")
    # Saved 0.16 workflows remain readable; preparation refuses their retired
    # transport at its entry point instead of silently changing the request.
    if settings.get("common_download_policy") not in (None, "selected_common_zip_v1"):
        raise ValueError("Unsupported saved common material download policy")
    if settings.get("audio_storage_policy") not in (None, "canonical_flac_only_v1"):
        raise ValueError("Unsupported canonical audio storage policy")
    if settings.get("fma_acquisition_policy") not in (None, "per_track_flac_prefetch_v1"):
        raise ValueError("Unsupported FMA acquisition policy")
    if settings.get("fma_acquisition_policy") and settings.get("material_acquisition_policy") != "automatic_v3":
        raise ValueError("Per-track FMA requires canonical FLAC storage")
    if settings.get("batch", BATCH) != BATCH:
        raise ValueError("Studio baseline batch must remain 768/256/100/100")
    if settings["train_references"] > MAX_TRAIN_VOICE_FAMILIES or settings["validation_references"] > 30:
        raise ValueError(f"The synthetic family bank supports up to {MAX_TRAIN_VOICE_FAMILIES} Train and 30 Validation candidates")
    policy = settings.get("evaluation_policy")
    if policy is not None:
        versions = {"common_development_v1": 1, "common_development_v2": 2,
                    "common_development_v3": 3, "common_development_v4": 4,
                    "common_development_v5": 5}
        version = versions.get(policy.get("id")) if isinstance(policy, dict) else None
        if version is None:
            raise ValueError("지원하지 않는 공통 평가 설정입니다. 저장된 실험 설정을 임의로 변경하지 마세요.")
        expected = common_evaluation_policy(version)
        if policy != expected or any(settings.get(key) != expected[key]
                                     for key in ("validation_references", "validation_originals", "noise_sections")):
            raise ValueError("공통 평가 설정은 학습 규모와 별개로 고정됩니다. 기존 실험의 평가량을 변경하지 않습니다.")
    for key in ("generic_hours", "korean_hours"):
        value = settings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")


def propose_hard_phrases(wakeword, limit=8):
    """Conservative text suggestions, never automatically approved phrases.

    Change one Hangul syllable, and reject any whole call-name overlap with
    the shared import/generation guard. Pronunciation/word suitability remains
    visible for the user to edit before Qwen is invoked.
    """
    if not isinstance(wakeword, str) or not wakeword or not all("가" <= c <= "힣" for c in wakeword):
        raise ValueError("A Hangul wakeword is required")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    proposals = []
    # Real short Korean non-call expressions are useful even for a one-syllable
    # call, where vocative/name perturbations alone can all be protected.
    candidates = [wakeword[:i] + c + wakeword[i + 1:]
                  for i in range(max(1, len(wakeword) - 1)) for c in "도누가리모서하미"]
    candidates += ["누구야", "어디야", "기다려", "그만해", "다음으로", "그러네", "고마워", "오늘은"]
    for text in candidates:
        if text in proposals:
            continue
        try:
            validate_noncall_text(text, wakeword)
        except ValueError:
            continue
        proposals.append(text)
        if len(proposals) == limit:
            break
    return proposals


def _publish_rows(path, rows, *, allow_initial_empty=False):
    """Idempotent exact bytes, with no mutation of an existing frozen queue."""
    path = Path(path)
    body = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode("utf-8")
    if path.exists():
        prior = path.read_bytes()
        if prior == body:
            return path
        if not (allow_initial_empty and not prior):
            raise ValueError(f"기존 자료가 달라졌습니다. 새 프로젝트를 사용하세요: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)
    return path


def _publish_reference_queues(workspace, rows, previous=None):
    """Append review work without rewriting a queue bound to human answers."""
    base = workspace / "reference_queue.jsonl"
    by_id = {row["sample_id"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError("합성 목소리 목록은 중복 없는 실제 음원이어야 합니다.")
    if not base.exists():
        if previous:
            raise ValueError("기존 합성 목소리 검수 목록이 없습니다.")
        return [_publish_rows(base, rows)]
    queues = [Path(path).resolve() for path in (previous or [base])]
    if (not queues or queues[0] != base.resolve() or len(set(queues)) != len(queues)
            or any(path.parent != workspace.resolve() for path in queues)):
        raise ValueError("기존 합성 목소리 검수 목록 경로가 달라졌습니다.")
    seen = set()
    for queue in queues:
        old = read_rows(queue)
        if not old:
            raise ValueError("기존 합성 목소리 검수 목록이 비어 있습니다.")
        for row in old:
            sample_id = row["sample_id"]
            if sample_id in seen or by_id.get(sample_id) != row:
                raise ValueError("기존 합성 목소리 음원 또는 검수 목록이 달라졌습니다.")
            seen.add(sample_id)
    added = [row for row in rows if row["sample_id"] not in seen]
    if added:
        key = identity([{key: row[key] for key in ("sample_id", "sha256")} for row in added])[:16]
        queues.append(_publish_rows(workspace / f"reference_queue_added_{key}.jsonl", added))
    return queues


def _completed(result=None, artifacts=(), **extra):
    return {"state": "completed", "result": result or {}, "artifacts": [str(p) for p in artifacts], **extra}


def _waiting(message, **extra):
    return {"state": "waiting_input", "message": message, **extra}


def _installed_dependencies():
    versions = {}
    for name in ("torch", "torchaudio", "qwen-tts", "onnx", "openwakeword"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _recommended_dependency_backend():
    """Detect NVIDIA hardware before a CUDA build of PyTorch is installed."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode == 0 and result.stdout.strip():
            return "cu128"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "cpu"


def _onnx_ready(version):
    try:
        major, minor = (int(x) for x in version.split(".")[:2])
        return (major, minor) >= (1, 16)
    except (AttributeError, ValueError):
        return False


def _dependency_plan(versions, backend):
    """Exact argv lists; no shell interpolation or global package installs."""
    if backend not in ("cpu", "cu128"):
        raise ValueError("설치 방식은 cpu 또는 cu128을 선택하세요.")
    commands = []
    common = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    torch_version = "2.8.0" if backend == "cpu" and sys.platform == "darwin" else "2.8.0+" + backend
    if any(versions.get(name) != torch_version for name in ("torch", "torchaudio")):
        # A plain ==2.8.0 spec also accepts an already-installed +cpu wheel.
        # Pin the chosen build so an explicit CPU/GPU switch cannot be a no-op.
        commands.append(common + ["torch==" + torch_version, "torchaudio==" + torch_version, "--index-url",
                                  "https://download.pytorch.org/whl/" + backend])
    extras = []
    if versions.get("qwen-tts") != "0.1.1":
        extras.append("qwen-tts==0.1.1")
    if not _onnx_ready(versions.get("onnx")):
        extras.append("onnx>=1.16")
    if extras:
        commands.append(common + extras)
    if versions.get("openwakeword") != "0.6.0":
        commands.append(common + ["openwakeword==0.6.0", "--no-deps"])
    return commands


def _install_dependencies(root, backend):
    if sys.prefix == sys.base_prefix:
        return _waiting("기본 Python에는 자동 설치하지 않습니다. 라이브러리를 가상환경에 설치해 웹 화면을 다시 여세요.",
                        input_kind="dependency_setup", virtual_environment_required=True)
    # Plan and install before doctor imports torch. On Windows an imported
    # extension can keep its DLL open and prevent replacing a mismatched wheel.
    commands = _dependency_plan(_installed_dependencies(), backend)
    from .storage import storage_environment
    with storage_environment(root):
        for command in commands:
            print("필요한 학습 패키지 설치: " + " ".join(command[6:]), flush=True)
            subprocess.run(command, cwd=str(root), check=True)
        gpu_runtime = None
        if backend == "cu128":
            gpu_runtime = _prepare_evaluation_runtime(root)
    importlib.invalidate_caches()
    return {"state": "completed", "backend": backend, "commands_run": commands,
            "evaluation_gpu_runtime": gpu_runtime,
            "scope": "Current virtual environment only; project-owned download/temp cache"}


def _prepare_evaluation_runtime(root):
    from .gpu_runtime_install import setup_gpu_runtime
    try:
        return setup_gpu_runtime(root)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        # Training CUDA and evaluation CUDA have different native requirements.
        # Auto evaluation records a CPU reason if its optional runtime fails;
        # an explicit CUDA evaluation still fails its own capability check.
        reason = "평가용 GPU 도구를 준비하지 못했습니다: " + str(error)
        print(reason + "; 공통 평가는 사용 가능한 연산 장치를 다시 확인합니다.", flush=True)
        return {"state": "unavailable", "reason": reason, "default_evaluation_backend": "auto"}


def _result(context, key):
    result = context.get("results", {}).get(key, {})
    return result.get("result", result)


def _review_descriptor(root, queue, kind, review_id, **extra):
    rows = read_rows(queue)
    if not rows:
        raise ValueError("검수할 음원이 없습니다.")
    for row in rows:
        bound_file(root, row)
    return {"id": review_id, "kind": kind, "queue": str(Path(queue).resolve()),
            "queue_sha256": digest(queue), "count": len(rows), **extra}


def _saved_review(root, descriptor, context):
    entry = context.get("inputs", {}).get("review_answers", {}).get(descriptor["id"])
    if entry:
        path = bound_file(root, entry) if isinstance(entry, dict) else resolved(root, entry)
        report = read_json(path)
    else:
        path = root / "review" / (descriptor["kind"] + "_results.json")
        if not path.exists():
            return False
        report = read_json(path)
        if (descriptor["kind"] == "references"
                and report.get("queue_sha256") != descriptor["queue_sha256"]):
            # CLI reviews have no Studio answer binding. A later supplemental
            # review replaces the mutable summary, while its hash-bound archive
            # keeps the original human decisions available for the old queue.
            approval = read_json(root / "approval.json")
            for evidence in reversed(approval.get("reference_review_evidence", [])):
                archived = read_json(bound_file(root, evidence))
                if archived.get("queue_sha256") == descriptor["queue_sha256"]:
                    report = archived
                    break
    if report.get("queue_sha256") != descriptor["queue_sha256"]:
        return False
    rows = read_rows(descriptor["queue"])
    decisions = report.get("decisions", {})
    if (report.get("source_scope") != "Human dispositions entered through the local review screen"
            or not report.get("saved_at")
            or report.get("queued_audio_sha256") != {r["sample_id"]: r["sha256"] for r in rows}):
        return False
    if descriptor.get("pool_manifest") and report.get("source_manifest_sha256") != digest(descriptor["pool_manifest"]):
        return False
    approved = (set(decisions) == {r["sample_id"] for r in rows}
            and all(a.get("decision") in ("keep", "discard") for a in decisions.values())
            and all(a.get("listened") is True for a in decisions.values() if a.get("decision") == "keep")
            and (not descriptor.get("pool_manifest") or report.get("whole_pool_policy_approved") is True))
    if approved and descriptor["kind"] == "references":
        approval = read_json(root / "approval.json")
        kept = [r for r in rows if decisions[r["sample_id"]]["decision"] == "keep"]
        approved = ({r["voice_family_id"] for r in kept} <= set(approval.get("accepted_reference_families", []))
                    and {r.get("split") for r in kept} == {r.get("split") for r in rows})
    elif approved and descriptor["kind"] == "content" and descriptor.get("pool_manifest"):
        reviewed = context.get("inputs", {}).get("review_results", {}).get(descriptor["id"], {})
        if reviewed.get("state") == "review_saved" and reviewed.get("admission_pending") is True:
            # Saving human answers is quick HTTP work. Whole-source validation
            # and formal admission happen later in the isolated data/Hard worker.
            return any(a.get("decision") == "keep" for a in decisions.values())
        if not reviewed.get("admission"):
            return False
        admission = read_json(bound_file(root, reviewed["admission"]))
        if (admission.get("state") != "accepted" or admission.get("content_review") != "accepted"
                or admission.get("human_answers", {}).get("sha256") != digest(path)
                or admission.get("source_manifest", {}).get("sha256") != digest(descriptor["pool_manifest"])
                or admission.get("representative_queue", {}).get("sha256") != descriptor["queue_sha256"]):
            return False
        accepted = set(admission.get("accepted_source_sha256", []))
        approved = bool(accepted)
        for row in rows:
            binding = row.get("reviewed_source", row)
            decision = decisions[row["sample_id"]]["decision"]
            if (binding["sha256"] in accepted) != (decision == "keep"):
                return False
    elif approved:
        # Pilot policy changes and excluded public/Hard sources must first be
        # resolved by the corresponding admission API; never silently keep a
        # source the reviewer discarded.
        approved = all(a["decision"] == "keep" for a in decisions.values())
    return approved


def _review_gate(root, reviews, context):
    pending = [r for r in reviews if not _saved_review(root, r, context)]
    return ({"state": "waiting_review", "reviews": pending,
             "message": "음성을 확인하고 저장하면 다음 작업을 이어갑니다."} if pending
            else _completed({"reviewed_queues": len(reviews)},
                            artifacts=[r["queue"] for r in reviews]))


def _material_review_queues(project, prepared, context, policy):
    """Keep existing human decisions authoritative when enabling fixed sources."""
    from .studio_data import material_review_queues
    selected = material_review_queues(project, prepared, policy=policy)
    inputs = context.get("inputs", {})
    selected_ids = {q["id"] for q in selected}
    for queue in prepared.get("review_queues", []):
        preparation = queue.get("preparation")
        preparation = preparation.get("path") if isinstance(preparation, dict) else preparation
        has_answers = (queue["id"] in inputs.get("review_answers", {})
                       or queue["id"] in inputs.get("review_results", {})
                       or str(preparation) in inputs.get("material_answers", {}))
        if has_answers and queue["id"] not in selected_ids:
            selected.append(queue)
            selected_ids.add(queue["id"])
    return selected


def _generation_complete(summary):
    if summary.get("full_target_reached") is not True:
        return _waiting("자동 기술 검사를 통과하지 못해 필요한 음성 수를 채우지 못했습니다. 실행 기록의 실패 사유를 확인하세요. "
                        "승인한 발음·생성 기준을 자동으로 낮추지 않습니다. 생성 설정을 바꾸려면 기존 자료를 보존하고 새 프로젝트에서 파일럿부터 확인해야 합니다.",
                        input_kind="generation_policy", result=summary,
                        exhausted_slots=summary.get("exhausted_slots", []))
    return None


def _next_directory(base, marker, accept=None):
    """Retain an interrupted immutable output and use a new attempt directory."""
    base = Path(base)
    candidate = base
    index = 0
    while candidate.exists() and any(candidate.iterdir()):
        if (candidate / marker).is_file() and (accept is None or accept(candidate)):
            return candidate, True
        index += 1
        candidate = base.with_name(base.name + f"_attempt{index + 1}")
    return candidate, False


def _hard_queue(project, workspace):
    root, _ = load_project(project)
    candidates = read_rows(root / "data/generated/hard/candidates.jsonl")
    latest = {r["slot"]: r for r in candidates if r.get("technical_pass") is True}
    rows = sorted(latest.values(), key=lambda r: r["slot"])
    if not rows:
        raise ValueError("No technically usable Hard Negative originals")
    sources = _publish_rows(workspace / "hard_sources.jsonl", rows)
    seen, representatives = set(), []
    for row in rows:
        group = (row["dataset_id"], row["text"], row["voice_family_id"])
        if group in seen:
            continue
        seen.add(group)
        representatives.append({**row, "sample_id": "preview_" + row["sample_id"],
            "reviewed_source": {"path": row["path"], "sha256": row["sha256"]},
            "review_instruction": "표시된 비호출 문구가 명확하고 실제 호출어가 들리지 않으면 보존하세요.",
            "review_group": row["text"] + " · " + row["voice_family_id"]})
    queue = _publish_rows(workspace / "hard_representatives.jsonl", representatives)
    return _review_descriptor(root, queue, "content", "hard", pool_manifest=str(sources))


def run_stage(project, workspace, settings, stage_id, context=None):
    """Execute one actual operation; call in an isolated durable worker.

    ``context`` contains prior stage results and explicit user input. The
    manager, not this function, owns process locks and stage receipts. Stages
    never generate Final data, loosen a target, or invent human listening.
    """
    _validate_settings(settings)
    context = context or {}
    root, config = load_project(project)
    selected_workspace = Path(workspace).resolve()
    workspace = root / ".studio"
    if stage_id not in {r[0] for r in _STAGES}:
        raise ValueError("Unknown studio stage: " + str(stage_id))
    if stage_id in ("data_download", "data_connect"):
        require_standard_common_download(settings.get("common_download_policy"))
        require_standard_common_download(config.get("common_download_policy"))
    if config["batch"] != BATCH:
        raise ValueError("Studio requires the declared 768/256/100/100 batch; no silent migration")
    expected = (settings["originals"], settings["rounds"], settings["updates"],
                settings["generic_hours"], settings["korean_hours"])
    actual = (config["targets"]["positive_originals"], config["augmentation_rounds"], config["training"]["updates"],
              config["targets"]["generic_hours"], config["targets"]["korean_hours"])
    if actual != expected:
        raise ValueError("Studio settings differ from the project's frozen workload")
    if (root / "final_lock.json").exists():
        return _waiting("Final 시험이 잠긴 프로젝트입니다. 학습 입력을 변경하려면 새 프로젝트를 만드세요.",
                        input_kind="generation_policy")
    workspace.mkdir(parents=True, exist_ok=True)

    if stage_id == "frontend":
        from .assets import prepare_frontend
        from .benchmark import doctor
        installation = None
        backend = context.get("inputs", {}).get("dependency_backend")
        if backend:
            installation = _install_dependencies(root, backend)
            if installation["state"] != "completed":
                return installation
        try:
            checked = doctor(include_tts=True)
        except (ImportError, OSError, RuntimeError) as error:
            checked = {"versions": _installed_dependencies(), "reference_openwakeword_ready": False,
                       "tts_check": {"ready": False, "reasons": [str(error)]}}
        checked["onnx_export_ready"] = _onnx_ready(_installed_dependencies().get("onnx"))
        if (not checked.get("tts_check", {}).get("ready") or not checked.get("reference_openwakeword_ready")
                or not checked["onnx_export_ready"]):
            return _waiting("학습 패키지 설치 방식에서 CPU 또는 NVIDIA GPU를 선택하세요. 설치 후 다시 확인합니다.",
                            input_kind="dependency_setup", check=checked,
                            recommended_backend=_recommended_dependency_backend(),
                            required_dependencies=["torch==2.8.0", "torchaudio==2.8.0", "qwen-tts==0.1.1", "onnx>=1.16", "openwakeword==0.6.0"],
                            installation=installation)
        if settings.get("evaluation_backend") == "auto" and checked.get("cuda_available") and backend != "cpu":
            # Manually prepared CUDA training packages need the same managed
            # evaluation runtime as packages installed through this screen.
            if not installation or not installation.get("evaluation_gpu_runtime"):
                runtime = _prepare_evaluation_runtime(root)
                installation = dict(installation or {}, evaluation_gpu_runtime=runtime)
        destination = resolved(root, config["frontend_dir"])
        receipt = prepare_frontend(destination)
        return _completed({"check": checked, "frontend": receipt, "installation": installation}, [destination / "receipt.json"])

    if stage_id == "data_download":
        from .studio_data import prepare_materials
        download_files = {}
        def progress(record):
            # Keep the complete download list visible during later extraction,
            # source selection and conversion progress, without inventing rates.
            if isinstance(record.get("files"), list):
                download_files.update((entry["id"], dict(entry)) for entry in record["files"])
            if download_files:
                record = {**record, "files": list(download_files.values())}
            write_json(workspace / "progress.json", {"stage": stage_id, "updated_at": now(), **record})
        prepared = prepare_materials(project, selected_workspace, preset=settings["preset"], progress=progress,
                                     evaluation_policy=settings.get("evaluation_policy"),
                                     noise_material_policy=settings.get("noise_material_policy", "official_example"),
                                     **({"material_acquisition_policy": settings["material_acquisition_policy"]}
                                        if settings.get("material_acquisition_policy") else {}),
                                     **({"official_noise_acquisition_policy": settings["official_noise_acquisition_policy"]}
                                        if settings.get("official_noise_acquisition_policy") else {}))
        # Downloaded candidates and reviewed/admitted sources are distinct.
        return _completed(prepared, [q["queue"] for q in prepared.get("review_queues", [])])

    if stage_id == "data_review":
        prepared = _result(context, "data_download")
        queues = prepared.get("review_queues", [])
        if not queues:
            return _waiting("다운로드한 자료의 검수 큐가 없습니다. 자료 준비 결과를 확인하세요.",
                            input_kind="generation_policy")
        policy = settings.get("material_review_policy")
        manual_queues = _material_review_queues(project, prepared, context, policy)
        if policy == "library_fixed_sources_v1" and not manual_queues:
            return _completed({"material_review_policy": policy, "fixed_source_queue_count": len(queues),
                               "human_listened_count": 0,
                               "scope": "Library-fixed source integrity checked; no human listening decisions were created"},
                              artifacts=[q["queue"] for q in queues])
        reviews = [_review_descriptor(root, q["queue"], "content", q.get("id", f"material_{i}"),
                    **{k: v for k, v in q.items() if k not in ("queue", "kind", "id")})
                   for i, q in enumerate(manual_queues)]
        return _review_gate(root, reviews, context)

    if stage_id == "data_connect":
        from .studio_data import admit_materials, prepare_additional_materials, ensure_evaluation_materials
        prepared = deepcopy(_result(context, "data_download"))
        prepared = ensure_evaluation_materials(project, selected_workspace, prepared)
        inputs = context.get("inputs", {})
        additions = inputs.get("additional_sources", [])
        if not isinstance(additions, list) or any(not isinstance(source, dict) for source in additions):
            raise ValueError("추가 자료는 폴더와 자료 종류를 지정한 목록이어야 합니다.")
        additional_artifacts = []
        existing_ids = {q["id"] for q in prepared.get("review_queues", [])}
        for source in additions:
            policy = prepared.get("noise_material_policy")
            reuse = policy == "reuse_negative_waveforms"
            selection = {"source": source, "noise_material_policy": policy} if reuse else source
            cached = workspace / ("additional_" + identity(selection) + ".json")
            if cached.exists():
                record = read_json(cached)
                if record.get("source") != source:
                    raise ValueError("Frozen additional source selection changed")
                addition = record["result"]
            else:
                addition = prepare_additional_materials(project, selected_workspace, source,
                    **({"noise_material_policy": policy} if reuse else {}))
                write_json(cached, {"source": source, "result": addition})
            if isinstance(addition.get("preparation"), dict):
                bound_file(root, addition["preparation"])
            queue = addition["review_queue"]
            if queue["id"] not in existing_ids:
                prepared.setdefault("review_queues", []).append(queue)
                existing_ids.add(queue["id"])
            additional_artifacts.append(cached)
        review_policy = settings.get("material_review_policy")
        manual_queues = _material_review_queues(project, prepared, context, review_policy)
        reviews = [_review_descriptor(root, q["queue"], "content", q["id"],
                    **{k: v for k, v in q.items() if k not in ("queue", "kind", "id")})
                   for q in manual_queues]
        gate = _review_gate(root, reviews, context)
        if gate["state"] != "completed":
            gate["artifacts"] = [str(p) for p in additional_artifacts]
            return gate
        answers = dict(inputs.get("material_answers", {}))
        for i, queue in enumerate(prepared.get("review_queues", [])):
            key = queue.get("id", f"material_{i}")
            answer = inputs.get("review_answers", {}).get(key)
            if answer:
                preparation = queue.get("preparation")
                preparation = preparation.get("path") if isinstance(preparation, dict) else preparation
                answers[str(preparation)] = answer
        result = admit_materials(project, prepared, answers, admissions=inputs.get("material_admissions", ()),
                                 **({"material_review_policy": review_policy} if review_policy else {}))
        shortages = result.get("shortages", prepared.get("shortages", []))
        if shortages:
            return _waiting("선택한 규모에 필요한 자료가 부족합니다. 준비된 자료는 보존하며 추가 자료를 연결해야 합니다.",
                            result=result, shortages=shortages, input_kind="material_admissions")
        return _completed(result, additional_artifacts)

    if stage_id == "references":
        from .tts import generate_references, _load_model
        families = read_json(root / "voices.json")["families"]
        pending = False
        for split, count in (("train", settings["train_references"]), ("validation", settings["validation_references"])):
            candidates = [f for f in families if f["split"] == split]
            if len(candidates) < count:
                raise ValueError(f"{split} 기준 목소리 후보가 {len(candidates)}/{count}개입니다. 기존 음원과 검수 기록을 보존하는 목소리 설정 업데이트가 필요합니다.")
            selected = candidates[:count]
            completed = {r["voice_family_id"] for r in read_rows(root / "data/references" / split / "references.jsonl")}
            pending = pending or any(f["family_id"] not in completed for f in selected)
        # Both partitions use identical VoiceDesign weights. Keep one model
        # in this worker; the original function still seeds every family and
        # validates the immutable per-split journal before generation.
        model = _load_model(config["tts"]["design_model"], config["tts"], root) if pending else None
        outputs = [generate_references(project, "train", settings["train_references"], model=model),
                   generate_references(project, "validation", settings["validation_references"], model=model)]
        rows = [r for output in outputs for r in read_rows(output["review_queue"])]
        queues = _publish_reference_queues(workspace, rows, _result(context, "references").get("queues"))
        return _completed({"outputs": outputs, "queue": str(queues[0]),
                           "queues": [str(queue) for queue in queues], "generated": len(rows)}, queues)

    if stage_id == "references_review":
        queues = _result(context, "references").get("queues") or [workspace / "reference_queue.jsonl"]
        reviews = []
        for queue in queues:
            queue = Path(queue)
            if queue.resolve().parent != workspace.resolve():
                raise ValueError("합성 목소리 검수 목록은 현재 모델 폴더 안에 있어야 합니다.")
            review_id = "references" if queue.name == "reference_queue.jsonl" else queue.stem.replace("reference_queue_", "references_", 1)
            reviews.append(_review_descriptor(root, queue, "references", review_id))
        return _review_gate(root, reviews, context)

    if stage_id == "pilot":
        from .tts_runtime import generate_with_runtime
        summary = generate_with_runtime(project, "train", 24, True, "clone")
        incomplete = _generation_complete(summary)
        if incomplete:
            return incomplete
        passed = {r["slot"]: r for r in read_rows(summary["review_queue"]) if r["qc"]["passed"]}
        queue = _publish_rows(workspace / "pilot_queue.jsonl", sorted(passed.values(), key=lambda r: r["slot"]))
        return _completed({**summary, "queue": str(queue)}, [queue])

    if stage_id == "pilot_review":
        descriptor = _review_descriptor(root, workspace / "pilot_queue.jsonl", "pilot", "pilot")
        result = _review_gate(root, [descriptor], context)
        if result["state"] == "completed":
            approval = read_json(root / "approval.json")
            if approval.get("state") != "accepted" or approval.get("generation_policy_approved") is not True:
                return _waiting("파일럿에 예외가 있어 생성 정책을 확인해야 합니다. 검수 결과만으로 자동 승인하지 않습니다.",
                                input_kind="generation_policy")
        return result

    if stage_id == "positives":
        from .tts import generate
        from .tts_runtime import CloneRuntime
        # Lazy construction does not load/download until generate's existing
        # approval checks pass. Share weights across the two disjoint banks;
        # keep the original per-utterance seed/QC/provenance implementation.
        model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root,
                             "auto", config["training"]["cpu_threads"])
        outputs = []
        for split, target in (("train", settings["originals"]), ("validation", settings["validation_originals"])):
            summary = generate(project, split, target, False, "clone", model=model)
            incomplete = _generation_complete(summary)
            if incomplete:
                return incomplete
            outputs.append(summary)
        return _completed({"outputs": outputs, "train_originals": settings["originals"],
                           "validation_originals": settings["validation_originals"], "independent_STT_runs": 0},
                          [resolved(root, config["manifests"]["positive_" + split]) for split in ("train", "validation")])

    if stage_id == "hard_phrases":
        inputs = context.get("inputs", {})
        phrases = inputs.get("hard_phrases", [])
        if inputs.get("hard_phrases_approved") is not True or not phrases:
            return _waiting("실제 호출어와 같은 낱말이 들어 있지 않은 비호출 문구인지 확인하세요.",
                            input_kind="hard_phrases", suggestions=propose_hard_phrases(config["wakeword"]))
        if not isinstance(phrases, list) or any(not isinstance(p, str) for p in phrases) or len(set(phrases)) != len(phrases):
            raise ValueError("유사어는 중복 없는 문구 목록이어야 합니다.")
        for phrase in phrases:
            validate_noncall_text(phrase, config["wakeword"])
        return _completed({"phrases": phrases, "user_approved": True})

    if stage_id == "hard_generation":
        from .hard_tts import generate_hard
        from .tts_runtime import CloneRuntime
        approved = _result(context, "hard_phrases")
        if approved.get("user_approved") is not True or not approved.get("phrases"):
            return _waiting("비호출 문구를 먼저 확인하고 저장하세요.", input_kind="hard_phrases")
        model = CloneRuntime(config["tts"]["clone_model"], config["tts"], root, "auto", config["training"]["cpu_threads"])
        summary = generate_hard(project, approved["phrases"], settings["hard_originals"], model=model)
        incomplete = _generation_complete(summary)
        if incomplete:
            return incomplete
        descriptor = _hard_queue(project, workspace)
        return _completed({**summary, "review": descriptor}, [descriptor["queue"], descriptor["pool_manifest"]])

    if stage_id == "hard_review":
        return _review_gate(root, [_hard_queue(project, workspace)], context)

    if stage_id == "hard_connect":
        from .data import validate_admission
        descriptor = _hard_queue(project, workspace)
        if not _saved_review(root, descriptor, context):
            return {"state": "waiting_review", "reviews": [descriptor]}
        reviewed = context.get("inputs", {}).get("review_results", {}).get("hard", {})
        if not reviewed.get("admission"):
            from .pool_review import approve_pool
            answers = context.get("inputs", {}).get("review_answers", {}).get("hard")
            if not answers:
                return {"state": "waiting_review", "reviews": [descriptor]}
            answer_path = bound_file(root, answers) if isinstance(answers, dict) else resolved(root, answers)
            raw = read_json(answer_path)
            phrases = {}
            for preview in read_rows(descriptor["queue"]):
                phrases.setdefault(preview["text"], []).append(raw["decisions"][preview["sample_id"]]["decision"])
            if any("discard" in decisions and "keep" in decisions for decisions in phrases.values()):
                return {"state": "waiting_review", "reviews": [descriptor],
                        "message": "같은 유사어의 음색별 판정이 다릅니다. 해당 문구를 학습에서 빼려면 그 문구의 모든 음색을 폐기하세요. 보존 판정을 자동으로 변경하지 않습니다."}
            excluded = sorted(text for text, decisions in phrases.items() if all(d == "discard" for d in decisions))
            admission_path = workspace / ("hard_admission_" + digest(answer_path) + ".json")
            if admission_path.exists():
                admission = {"path": admission_path.relative_to(root).as_posix(), "sha256": digest(admission_path)}
                existing = read_json(admission_path)
                if (existing.get("human_answers", {}).get("sha256") != digest(answer_path)
                        or existing.get("excluded_hard_phrases") != excluded):
                    raise ValueError("Frozen Hard admission no longer matches saved listening decisions")
            else:
                approved = approve_pool(project,
                    source_manifest={"path": descriptor["pool_manifest"], "sha256": digest(descriptor["pool_manifest"])},
                    representative_queue={"path": descriptor["queue"], "sha256": descriptor["queue_sha256"]},
                    answers={"path": str(answer_path), "sha256": digest(answer_path)}, output=admission_path,
                    excluded_hard_phrases=excluded)
                admission = approved["admission"]
        else:
            admission = reviewed["admission"]
        record = read_json(bound_file(root, admission))
        if record.get("source_manifest", {}).get("sha256") != digest(descriptor["pool_manifest"]):
            raise ValueError("Hard admission does not bind the current sources")
        rows = []
        for source in read_rows(descriptor["pool_manifest"]):
            if source["sha256"] not in record["accepted_source_sha256"]:
                continue
            rows.append({**source, "admission": admission})
        validate_admission(root, rows, config, "hard_negative")
        destination = resolved(root, config["manifests"]["hard_negative"])
        if destination.exists() and not destination.stat().st_size and (
                any((root / "runs").rglob("contract.json")) or any((root / "features").rglob("contract.json"))):
            raise ValueError("A started experiment's Hard manifest cannot be changed")
        destination = _publish_rows(destination, rows, allow_initial_empty=True)
        return _completed({"originals": len(rows), "admission": admission}, [destination])

    if stage_id == "features":
        from .features import build_features, verify_cache
        outputs = {}
        for role in BATCH:
            directory = resolved(root, config["pool_features"][role])
            if role == "generic_negative" and (directory / "metadata.json").is_file():
                outputs[role] = verify_cache(directory)
            else:
                outputs[role] = build_features(project, role, resume=(directory / "contract.json").exists())
        positive = outputs["positive"]
        expected = settings["originals"] * settings["rounds"]
        if positive.get("feature_rows") != expected:
            raise ValueError("N × rounds does not match actual Positive feature rows")
        return _completed({"pools": outputs, "positive_expected_rows": expected,
                           "positive_actual_rows": positive["feature_rows"]},
                          [resolved(root, config["pool_features"][role]) / "metadata.json" for role in BATCH])

    if stage_id == "training":
        from .train import train
        directory = root / "runs" / "studio_run"
        resume = any(directory.glob("member*/contract.json"))
        result = train(project, directory, resume=resume)
        return _completed(result, [p for p in directory.glob("member*/receipt.json")])

    if stage_id == "package":
        from .bundle import package_run
        from .runtime import verify_bundle
        destination, existing = _next_directory(root / "models/studio_bundle", "manifest.json")
        result = verify_bundle(destination) if existing else package_run(project, root / "runs/studio_run", destination)
        if result.get("project_config_sha256") != digest(project):
            raise ValueError("Existing bundle does not bind this project")
        return _completed({"bundle": str(destination), "manifest": result}, [destination / "manifest.json"])

    if stage_id == "development":
        from .paired import prepare_evaluation, evaluation_recipe
        connected = _result(context, "data_connect")
        negatives = connected.get("negative_validation_manifest")
        if not negatives:
            return _waiting("연속 오호출 시험용 원본 비호출 음원이 없습니다. 학습 자료 준비 기록에서 Validation 분리 결과를 확인하세요.",
                            input_kind="generation_policy")
        negatives = bound_file(root, negatives) if isinstance(negatives, dict) else resolved(root, negatives)
        negative_rows = read_rows(negatives)
        if not negative_rows or any(r.get("kind") != "negative" or r.get("split") != "validation" for r in negative_rows):
            raise ValueError("Held-out continuous Negative recordings are required")
        if settings.get("evaluation_policy"):
            positives = read_rows(resolved(root, config["manifests"]["positive_validation"]))
            if len(positives) < settings["validation_originals"]:
                raise ValueError(f"공통 평가용 호출어 {settings['validation_originals']:,}개가 준비되지 않았습니다. 적은 표본으로 공통 평가 완료를 기록하지 않습니다.")
        selection = "balanced_sources"
        pair_settings = {"originals": settings["validation_originals"],
                         "noise_sections": settings["noise_sections"], "noise_selection": selection}
        if settings.get("evaluation_policy"):
            pair_settings["snrs"] = tuple(settings["evaluation_policy"]["snrs"])
            if settings["evaluation_policy"].get("reverb_originals"):
                pair_settings["reverb_originals"] = settings["evaluation_policy"]["reverb_originals"]
                if settings["evaluation_policy"].get("reverb_selection"):
                    pair_settings["reverb_selection"] = settings["evaluation_policy"]["reverb_selection"]
                from .evaluation_materials import evaluation_material_coverage
                material_coverage = evaluation_material_coverage(negative_rows,
                    **({"strict_common_pool": True} if settings["evaluation_policy"].get("evaluation_materials_recipe") == 6 else {}))
                if not material_coverage["target_met"]:
                    raise ValueError("공통 평가 자료가 부족합니다: " + str(material_coverage["shortages"]))
        recipe = evaluation_recipe(project, **pair_settings)
        def matches(directory):
            return ((directory / "recipe.json").is_file() and read_json(directory / "recipe.json") == recipe
                    and (selection == "legacy" or (directory / "noise_coverage.json").is_file()))
        directory, existing = _next_directory(root / "reports/studio_paired", "manifest.jsonl", matches)
        if existing:
            paired = {"manifest": str(directory / "manifest.jsonl"), "noise_selection_policy": recipe["noise_selection"],
                      "recipe": {"path": str(directory / "recipe.json"), "sha256": digest(directory / "recipe.json")}}
            if selection != "legacy":
                paired["noise_coverage"] = {"path": str(directory / "noise_coverage.json"), "sha256": digest(directory / "noise_coverage.json")}
        else:
            paired = prepare_evaluation(project, directory, **pair_settings)
        rows = read_rows(paired["manifest"]) + negative_rows
        manifest = _publish_rows(workspace / ("development_" + identity([recipe, digest(negatives), digest(paired["manifest"])])[:16] + ".jsonl"), rows)
        artifacts = [manifest, directory / "recipe.json"]
        if paired.get("noise_coverage"):
            artifacts.append(paired["noise_coverage"]["path"])
        return _completed({**paired, "manifest": str(manifest), "positive_sessions": len(rows) - len(negative_rows),
                           "negative_recordings": len(negative_rows), "functional_check_only": settings["functional_check_only"],
                           "evaluation_policy": deepcopy(settings.get("evaluation_policy")),
                           "evaluation_policy_sha256": identity(settings["evaluation_policy"]) if settings.get("evaluation_policy") else None,
                           "independent_Final_completed": False}, artifacts)

    if stage_id == "evaluation":
        from .evaluate import evaluate, evaluation_binding
        from .selection import verified_evaluation
        bundle = _result(context, "package")["bundle"]
        manifest = _result(context, "development")["manifest"]
        scoring_smoke = settings["functional_check_only"] and not settings.get("evaluation_policy")
        backend = settings.get("evaluation_backend", "cpu")
        gpu_batch_size = settings.get("gpu_batch_size", 128)
        boundary_margin = settings.get("gpu_boundary_margin", .001)
        import_path = settings.get("gpu_cpu_prefix_import")
        cpu_prefix = read_json(import_path)["prefix"] if import_path else None
        expected = (evaluation_binding(project, bundle, manifest, smoke=scoring_smoke, backend=backend,
            gpu_batch_size=gpu_batch_size, boundary_margin=boundary_margin, cpu_prefix=cpu_prefix)
            if backend == "cuda" else None)
        directory, existing = _next_directory(root / "reports/studio_evaluation", "receipt.json",
            lambda p: (read_json(p / "report.json").get("manifest_sha256") == digest(manifest)
                       and read_json(p / "report.json").get("bundle_manifest_sha256") == digest(Path(bundle) / "manifest.json")
                       and (backend == "auto" or backend == "cpu" and read_json(p / "report.json").get("execution", {}).get("backend", "cpu") == "cpu"
                            or read_json(p / "report.json").get("evaluation_binding") == expected)))
        if existing:
            verified_evaluation(project, bundle, directory)
            report = read_json(directory / "report.json")
        else:
            resume = False
            base = root / "reports/studio_evaluation"
            attempts = [base] + sorted((p for p in base.parent.glob(base.name + "_attempt*")
                if p.name[len(base.name + "_attempt"):].isdigit()),
                key=lambda p: int(p.name[len(base.name + "_attempt"):]))
            partials = [p for p in attempts if not (p / "receipt.json").exists()
                        and (p / "_evaluation_work/checkpoint.json").is_file()]
            if partials:
                if expected is None:
                    expected = evaluation_binding(project, bundle, manifest, smoke=scoring_smoke)
                for partial in partials:
                    try:
                        checkpoint = read_json(partial / "_evaluation_work/checkpoint.json")
                    except (OSError, ValueError):
                        continue  # Preserve malformed/unrelated work; never claim it is reusable.
                    for selected_backend in (("cpu", "cuda") if backend == "auto" else (backend,)):
                        selected_binding = (evaluation_binding(project, bundle, manifest, smoke=scoring_smoke,
                            backend=selected_backend, gpu_batch_size=gpu_batch_size, boundary_margin=boundary_margin,
                            cpu_prefix=cpu_prefix if selected_backend == "cuda" else None) if backend == "auto" else expected)
                        if isinstance(checkpoint, dict) and checkpoint.get("schema_version") == 1 and checkpoint.get("binding") == selected_binding:
                            directory, resume, backend = partial, True, selected_backend
                            break
                    if resume:
                        break
            report = evaluate(project, bundle, directory, manifest=manifest,
                              smoke=scoring_smoke, workers="auto", resume=resume, backend=backend,
                              gpu_batch_size=gpu_batch_size, boundary_margin=boundary_margin,
                              cpu_prefix_import=import_path,
                              numerical_differences_approved=settings.get("gpu_numerical_differences_approved"))
        assessment = None
        artifacts = [directory / "receipt.json", directory / "report.json", directory / "clips.json"]
        if settings.get("evaluation_policy"):
            from .evaluation_coverage import assess_common_evaluation
            coverage = _result(context, "development").get("noise_coverage")
            coverage = bound_file(root, coverage) if isinstance(coverage, dict) else coverage
            assessment = assess_common_evaluation(settings["evaluation_policy"], report, manifest, coverage)
            assessment_path = directory / ("common_coverage_" + identity(assessment)[:16] + ".json")
            if assessment_path.exists() and read_json(assessment_path) != assessment:
                raise ValueError("Frozen common evaluation coverage evidence changed")
            if not assessment_path.exists():
                write_json(assessment_path, assessment)
            artifacts.append(assessment_path)
        return _completed({"evaluation": str(directory), "report": report,
                           "functional_check_only": settings["functional_check_only"], "Final_accessed": False,
                           "evaluation_functional_check_only": settings["functional_check_only"] and not bool(settings.get("evaluation_policy")),
                           "evaluation_policy": deepcopy(settings.get("evaluation_policy")),
                           "evaluation_policy_sha256": identity(settings["evaluation_policy"]) if settings.get("evaluation_policy") else None,
                           "common_evaluation_assessment": assessment,
                           "noise_coverage": _result(context, "development").get("noise_coverage")},
                          artifacts)

    if stage_id == "results":
        from .site import export_site
        from .selection import select_candidate
        evaluation = Path(_result(context, "evaluation")["evaluation"])
        report = read_json(evaluation / "report.json")
        directory, existing = _next_directory(root / "reports/studio_results", "provenance.json",
            lambda p: read_json(p / "provenance.json").get("evaluation_receipt_sha256") == digest(evaluation / "receipt.json"))
        site = {"site": str(directory / "index.html")} if existing else export_site(evaluation, directory)
        candidates = report.get("candidate_adoptions", {})
        eligible = [r for r in report["models"] if candidates.get(r["id"], {}).get("passed")]
        ranked = sorted(eligible or report["models"], key=lambda r: (
            -(r.get("noise_cell_mean") if r.get("noise_cell_mean") is not None else -1),
            -(r["clean"]["recall"] if r["clean"]["recall"] is not None else -1),
            r["negative"]["fa_per_hour"] if r["negative"]["fa_per_hour"] is not None else math.inf, r["id"]))
        result = {**site, "bundle": _result(context, "package")["bundle"],
                  "ranking": [r["id"] for r in ranked], "Final_accessed": False,
                  "functional_check_only": settings["functional_check_only"],
                  "evaluation_policy": deepcopy(settings.get("evaluation_policy")),
                  "evaluation_policy_sha256": identity(settings["evaluation_policy"]) if settings.get("evaluation_policy") else None,
                  "common_evaluation_assessment": _result(context, "evaluation").get("common_evaluation_assessment"),
                  "meets_development_targets": bool(eligible), "independent_Final_completed": False,
                  "noise_coverage": _result(context, "development").get("noise_coverage")}
        if ranked:
            chosen = ranked[0]["id"]
            selection_directory, selected = _next_directory(root / "models/studio_selected", "receipt.json",
                lambda p: (read_json(p / "selection.json").get("source_report_sha256") == digest(evaluation / "report.json")
                           and read_json(p / "selection.json").get("candidate", {}).get("id") == chosen))
            if not selected:
                selection = select_candidate(project, result["bundle"], evaluation, selection_directory, chosen)
                result["selection"] = selection
            else:
                from .selection import verified_evaluation
                verified_evaluation(project, selection_directory / "bundle", selection_directory / "evaluation")
                selected_evidence = read_json(selection_directory / "selection.json")
                if (selected_evidence.get("source_report_sha256") != digest(evaluation / "report.json")
                        or selected_evidence.get("candidate", {}).get("id") != chosen):
                    raise ValueError("Previous candidate selection differs from this Development report")
            result["selected_candidate"] = chosen
            result["selected_bundle"] = str(selection_directory / "bundle")
            result["selection_scope"] = "Development recommendation only; not Final approval"
        return _completed(result, [directory / "provenance.json"])
