"""Generate replacement references and read the reviewed bank without editing its source."""

from pathlib import Path

from .clone_reference import GeneratedReferenceQualityError, verify_generated_reference
from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, writer_lock
from .project import load_project
from .reference_attempt import check_reference_timing, prepare_reference_attempt, publish_reference_attempt
from .reference_qc import _check_wave
from .storage import project_storage


REVIEW_KIND = "reference_voice_review_v1"
_DESIGN_FIELDS = ("voice_family_id", "split", "text", "model", "description", "style", "speed", "intonation", "vocal_effort")
_REPLACEMENT_ATTEMPTS = 4
_REPLACEMENT_GENERATION = "reference_review_replacement"


def _jobs(root):
    return root / ".studio" / "reference-review" / "jobs"


def _project_binding(root, entry):
    if not resolved(root, entry["path"]).is_relative_to(root):
        raise ValueError("기준 목소리 검수 파일이 모델 폴더 밖을 가리킵니다.")
    return bound_file(root, entry)


def _reference(row):
    return (row.get("split") in ("train", "validation")
            and row.get("provenance") == "qwen_synthetic_reference"
            and row.get("human_reference_used") is False)


def _verify_replacement(root, original, replacement):
    if (not _reference(replacement)
            or any(replacement.get(field) != original.get(field) for field in _DESIGN_FIELDS)
            or replacement.get("replaces_reference_sha256") != original.get("replaces_reference_sha256", original["sha256"])
            or replacement.get("sample_id") == original.get("sample_id")
            or replacement.get("sha256") == original.get("sha256")):
        raise ValueError("교체한 기준 목소리의 설정 또는 출처가 다릅니다.")
    path = resolved(root, replacement["path"])
    if not path.is_relative_to(_jobs(root)):
        raise ValueError("교체한 기준 목소리의 저장 위치가 다릅니다.")
    bound_file(root, replacement)
    if "reference_replacement_attempt" in replacement:
        attempt = replacement["reference_replacement_attempt"]
        preparation = replacement["reference_preparation"]
        generation = preparation["generation"]
        if (type(attempt) is not int or not 0 <= attempt < _REPLACEMENT_ATTEMPTS
                or generation.get("kind") != _REPLACEMENT_GENERATION
                or generation.get("source_reference_sha256") != original["sha256"]
                or generation.get("attempt") != attempt or generation.get("seed") != replacement["seed"]
                or generation.get("text") != replacement["text"]
                or preparation["prepared"]["sha256"] != replacement["sha256"]
                or preparation["technical_metrics"]["decoded_pcm_sha256"] != replacement["decoded_pcm_sha256"]
                or not verify_generated_reference(root, preparation).samefile(path)
                or replacement.get("reference_timing_check") != check_reference_timing(path, replacement["text"])):
            raise ValueError("교체한 기준 목소리의 생성·정리 기록이 다릅니다.")


def _original_pcm_hashes(root):
    binding = read_json(root / "approval.json").get("reference_technical_admission")
    if not binding:
        return set()
    receipt = read_json(_project_binding(root, binding))
    return {pcm for check in read_rows(_project_binding(root, receipt["quality_journal"]))
            if (pcm := check.get("metrics", {}).get("decoded_pcm_sha256"))}


def _replacement_generation(tts, original, job_key, attempt):
    from .tts import _reference_instruction

    family = original["voice_family_id"]
    seed = int(identity([tts["seed"], "reference-review", job_key, family, attempt])[:8], 16)
    return {"kind": _REPLACEMENT_GENERATION, "model": tts["design_model"], "seed": seed,
            "text": original["text"], "instruction": _reference_instruction(original),
            "source_reference_sha256": original["sha256"], "attempt": attempt,
            "tts_settings_id": identity(tts), "job_key": job_key}


@project_storage
def generate_replacements(project, output, job_key, rows, all_hashes, *, load_model=None):
    """Create fresh samples using one design model for all selected voices."""
    from .tts import _load_model

    root, config = load_project(project)
    output = resolved(root, output)
    if not output.is_relative_to(_jobs(root)) or output == _jobs(root):
        raise ValueError("기준 목소리 다시 생성 폴더가 올바르지 않습니다.")
    family_ids = {row["voice_family_id"] for row in rows}
    if not rows or len(family_ids) != len(rows) or any(not _reference(row) for row in rows):
        raise ValueError("다시 생성할 기준 목소리를 확인하세요.")
    tts = config["tts"]
    hashes = set(all_hashes) | _original_pcm_hashes(root) | {row["sha256"] for row in rows}
    for row in rows:
        _, metrics = _check_wave(_project_binding(root, row))
        if pcm := metrics.get("decoded_pcm_sha256"):
            hashes.add(pcm)

    replacements, model = {}, None

    def resident_model():
        nonlocal model
        if model is None:
            model = (load_model or _load_model)(tts["design_model"], tts, root)
        return model

    with writer_lock(output):
        for original in rows:
            family = original["voice_family_id"]
            token = identity([job_key, family, original["sha256"]])[:20]
            record = output / f"{token}.json"
            if record.exists():
                replacement = read_json(record)
                _verify_replacement(root, original, replacement)
                attempt = replacement.get("reference_replacement_attempt")
                target = output / (f"{token}.wav" if attempt is None else f"{token}_attempt{attempt}.wav")
                if (replacement["path"] != target.relative_to(root).as_posix()
                        or replacement.get("previous_reference_sha256") != original["sha256"]
                        or attempt is not None and replacement["reference_preparation"]["generation"] !=
                            _replacement_generation(tts, original, job_key, attempt)
                        or replacement["sha256"] in hashes or replacement["decoded_pcm_sha256"] in hashes):
                    raise ValueError("이전에 생성한 기준 목소리 기록이 다릅니다.")
            else:
                for attempt in range(_REPLACEMENT_ATTEMPTS):
                    generation = _replacement_generation(tts, original, job_key, attempt)
                    seed = generation["seed"]
                    if seed == original.get("seed"):
                        continue
                    name = f"{token}_attempt{attempt}.wav"
                    try:
                        prepared, timing_check, timing = prepare_reference_attempt(root, output / "raw" / name,
                            output / "prepared" / name, generation, tts["generation"], resident_model)
                    except GeneratedReferenceQualityError:
                        continue
                    sha = prepared["prepared"]["sha256"]
                    pcm = prepared["technical_metrics"]["decoded_pcm_sha256"]
                    if sha in hashes or pcm in hashes:
                        if timing is not None:
                            timing.committed(False)
                        continue
                    preparation = publish_reference_attempt(root, output / name, prepared)
                    replacement = {key: value for key, value in original.items()
                                   if key not in ("reference_generation_update", "reference_preparation", "reference_timing_check")}
                    replacement.update(sample_id=f"{family}_{token}", path=preparation["prepared"]["path"],
                        sha256=preparation["prepared"]["sha256"], seed=seed, created_at=now(), decoded_pcm_sha256=pcm,
                        previous_reference_sha256=original["sha256"],
                        replaces_reference_sha256=original.get("replaces_reference_sha256", original["sha256"]),
                        reference_preparation=preparation, reference_timing_check=timing_check,
                        reference_replacement_attempt=attempt)
                    if "technical_metrics" in replacement:
                        replacement["technical_metrics"] = preparation["technical_metrics"]
                    write_json(record, replacement)
                    if timing is not None:
                        timing.committed(True)
                    break
                else:
                    raise ValueError("기술 검사를 통과한 다른 기준 목소리를 생성하지 못했습니다.")
            replacements[family] = replacement
            hashes.update((replacement["sha256"], replacement["decoded_pcm_sha256"]))
    return replacements


def overlay_reference_banks(root, banks, binding):
    """Use a frozen human selection only while every original and replacement still matches."""
    if binding is None:
        return banks
    root = Path(root).resolve()
    manifest = read_json(_project_binding(root, binding))
    if not isinstance(manifest, dict):
        raise ValueError("저장한 기준 목소리 검수 목록을 읽을 수 없습니다.")
    if manifest.get("kind") == "reference_voice_review_v2":
        from .reference_candidates import overlay_candidate_banks
        return overlay_candidate_banks(root, banks, manifest)
    if manifest.get("kind") == "reference_voice_review_v3":
        from .reference_candidates import overlay_planned_banks
        return overlay_planned_banks(root, banks, manifest)
    originals = {row["voice_family_id"]: row for bank in banks.values() for row in bank}
    original_hashes = {family: row["sha256"] for family, row in originals.items()}
    replacements = manifest.get("replacements")
    if (manifest.get("kind") != REVIEW_KIND or manifest.get("originals") != original_hashes
            or len(originals) != sum(map(len, banks.values()))
            or not isinstance(replacements, dict) or replacements.keys() - originals.keys()):
        raise ValueError("저장한 검수 목록이 기존 기준 목소리와 일치하지 않습니다.")
    hashes = set(original_hashes.values())
    sample_ids = {row["sample_id"] for row in originals.values()}
    for family, replacement in replacements.items():
        original = originals[family]
        _project_binding(root, original)
        _verify_replacement(root, original, replacement)
        if replacement["sha256"] in hashes or replacement["sample_id"] in sample_ids:
            raise ValueError("검수에서 선택한 교체 음원이 서로 중복됩니다.")
        hashes.add(replacement["sha256"])
        sample_ids.add(replacement["sample_id"])
    return {split: [replacements.get(row["voice_family_id"], row) for row in bank] for split, bank in banks.items()}
