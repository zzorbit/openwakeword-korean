"""Five-at-a-time synthetic reference candidates with explicit human selection.

Candidate plans, waveforms and accepted selections are separate from the frozen
source bank.  No source WAV, journal or voice-family definition is rewritten.
"""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from .task_progress import emit as report_progress, generation_attempt
import random
import re
import uuid

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, snapshot_json, write_immutable_rows, write_json, writer_lock
from .project import load_project
from .reference_qc import _check_wave
from .storage import project_storage


CANDIDATE_POLICY = "candidate_pool_v2"
CANDIDATE_REVIEW_KIND = "reference_voice_review_v2"
PLANNED_CANDIDATE_POLICY = "candidate_pool_v3"
PLANNED_REVIEW_KIND = "reference_voice_review_v3"
SLOT_PLAN_KIND = "reference_candidate_slot_plan_v3"
PREPARATION_KIND = "reference_initial_candidate_preparation_v1"
OPTIONAL_LISTENING_POLICY = "optional_candidate_listening_v1"
_TERMINAL_CANDIDATE_QC_ERROR = "기술 검사를 통과한 서로 다른 목소리 후보를 만들지 못했습니다. 다시 시도하세요."
BATCH_KIND = "reference_candidate_batch_v2"
BATCH_SIZE = 5
GENDERS = ("man", "woman")
AGES = ("child", "teenager", "young adult", "middle-aged", "older adult")
GENDER_LABELS = {"man": "남성", "woman": "여성"}
AGE_LABELS = dict(zip(AGES, ("어린이", "청소년", "청년", "중년", "노년")))
OPTIONS = read_json(files("korean_openwakeword") / "resources/voice_design_options.json")
FIELDS = tuple(field["id"] for field in OPTIONS["fields"])


def candidate_jobs(root):
    return Path(root).resolve() / ".studio/reference-candidates/jobs"


def _inside(root, entry, directory=None):
    path = resolved(root, entry["path"])
    if not path.is_relative_to(Path(root).resolve()) or directory and not path.is_relative_to(directory):
        raise ValueError("기준 목소리 후보 파일의 저장 위치가 올바르지 않습니다.")
    return bound_file(root, entry)


def _trait(row, name):
    value = row.get("voice_design", {}).get(name, {})
    return value.get("value") if isinstance(value, dict) else value


def source_slots(project, settings):
    """Describe intended family slots without generating or inventing an audio hash."""
    root, config = load_project(project)
    root = Path(root).resolve()
    project = Path(project).resolve()
    if project.parent != root:
        raise ValueError("기준 목소리 설정 파일은 현재 모델 폴더 안에 있어야 합니다.")
    counts = {split: settings[f"{split}_references"] for split in ("train", "validation")}
    if any(type(count) is not int for count in counts.values()):
        raise ValueError("기준 목소리 분류별 목표 수량을 확인하세요.")
    source = {"project": {"path": project.relative_to(root).as_posix(), "sha256": digest(project)},
              "voice_bank": {"path": "voices.json", "sha256": digest(root / "voices.json")}, "counts": counts}
    families = read_json(root / "voices.json")["families"]
    from .tts import REFERENCE_TEXT
    slots = []
    for split, count in counts.items():
        selected = [family for family in families if family["split"] == split][:count]
        if len(selected) != count:
            raise ValueError("기준 목소리 설정에 필요한 화자 슬롯이 부족합니다.")
        for number, family in enumerate(selected, 1):
            row = {"sample_id": family["family_id"], "voice_family_id": family["family_id"], "split": split,
                   "text": config["tts"].get("reference_text", REFERENCE_TEXT), "model": config["tts"]["design_model"],
                   "description": family["description"], "voice_number": number, "source_slot": True,
                   "source_binding": source, "voice_design": {field: {"value": family.get(field)}
                       for field in ("gender", "age_group")}}
            row["slot_id"] = identity(row)
            slots.append(row)
    candidate_groups(slots)
    return slots


candidate_source_rows = source_slots


def _slot_descriptor(binding, slots):
    return {"kind": SLOT_PLAN_KIND, "slot_plan": binding, "slot_count": len(slots), "group_count": 20,
            "source_binding": slots[0]["source_binding"], "generated": 0, "requires_candidate_selection": True}


def prepare_candidate_slots(project, settings):
    root, _ = load_project(project)
    slots = source_slots(project, settings)
    value = {"kind": SLOT_PLAN_KIND, "slots": slots, "groups": candidate_groups(slots),
             "scope": "Metadata only; no waveform generated, listened to, or admitted"}
    binding = snapshot_json(root, value, "reference_candidate_slots")
    return _slot_descriptor(binding, slots)


def verify_candidate_slots(project, settings, binding):
    root, _ = load_project(project)
    slots = source_slots(project, settings)
    value = read_json(_inside(root, binding))
    expected = {"kind": SLOT_PLAN_KIND, "slots": slots, "groups": candidate_groups(slots),
                "scope": "Metadata only; no waveform generated, listened to, or admitted"}
    if value != expected:
        raise ValueError("기준 목소리 후보 선택 계획의 설정 또는 목표 수량이 다릅니다.")
    return _slot_descriptor(binding, slots)


def _planned(saved):
    return saved.get("kind") == PLANNED_REVIEW_KIND


def _verify_source(root, source):
    for key in ("project", "voice_bank"):
        _inside(root, source[key])


def _source_groups(root, originals, binding):
    """Reconstruct family membership from the unchanged, hash-bound source bank."""
    bank = read_json(_inside(root, binding))
    families = {family["family_id"]: family for family in bank["families"]}
    enriched = []
    for row in originals:
        family = families.get(row["voice_family_id"])
        if family is None or family["split"] != row["split"]:
            raise ValueError("기준 목소리 원본의 화자 분류가 바뀌었습니다.")
        attributes = {**family, **row}
        description = row.get("description") or family.get("description", "")
        for field, pattern in (("gender", r"\b(man|woman|male|female)\b"),
                               ("age_group", r"\b(child|teenager|young adult|middle-aged|older adult)\b")):
            if not attributes.get(field):
                match = re.search(pattern, description)
                attributes[field] = match.group(1) if match else None
        attributes["gender"] = {"male": "man", "female": "woman"}.get(attributes["gender"], attributes["gender"])
        enriched.append({**row, "voice_design": {
            field: {"value": attributes[field]} for field in ("gender", "age_group")}})
    return candidate_groups(enriched)


def candidate_groups(originals):
    rows = list(originals)
    if not rows or len({row["voice_family_id"] for row in rows}) != len(rows):
        raise ValueError("기준 목소리 원본 목록이 비었거나 중복되었습니다.")
    groups = []
    for split in ("train", "validation"):
        targets = []
        for gender in GENDERS:
            for age in AGES:
                matching = [row for row in rows if row.get("split") == split
                            and _trait(row, "gender") == gender and _trait(row, "age_group") == age]
                if not matching:
                    raise ValueError("성별·연령별 기준 목소리 원본이 부족합니다.")
                target = len(matching)
                targets.append(target)
                groups.append({"key": f"{split}:{gender}:{age}", "split": split,
                               "gender": gender, "age": age, "target": target,
                               "label": f"{'학습용' if split == 'train' else '공통 평가용'} · {GENDER_LABELS[gender]} · {AGE_LABELS[age]}",
                               "families": sorted(row["voice_family_id"] for row in matching)})
        if len(set(targets)) != 1:
            raise ValueError("성별·연령별 기준 목소리 목표 수량이 균일하지 않습니다.")
        if split == "train" and targets[0] not in (2, 10, 50) or split == "validation" and targets[0] != 3:
            raise ValueError("학습용 또는 공통 평가용 기준 목소리 목표 수량을 확인하세요.")
    if sum(group["target"] for group in groups) != len(rows):
        raise ValueError("기준 목소리에 분류할 수 없는 성별·연령이 있습니다.")
    return groups


def create_candidate_review(originals):
    originals = list(originals)
    planned = bool(originals) and all(row.get("source_slot") is True for row in originals)
    if not planned and any(row.get("source_slot") for row in originals):
        raise ValueError("기준 목소리 원본과 생성 전 슬롯을 섞을 수 없습니다.")
    saved = {"schema_version": 3 if planned else 2, "kind": PLANNED_REVIEW_KIND if planned else CANDIDATE_REVIEW_KIND,
            "originals": {} if planned else {row["voice_family_id"]: row["sha256"] for row in originals},
            "groups": candidate_groups(originals), "candidates": {}, "batches": {},
            "current_batch": {}, "selected": {}, "heard": {}, "approved": [],
            "job": None, "revision": 0, "created_at": now()}
    if planned:
        source = originals[0]["source_binding"]
        if any(row["source_binding"] != source or "sha256" in row or "path" in row for row in originals):
            raise ValueError("생성 전 기준 목소리 슬롯의 출처가 다릅니다.")
        saved.update(slot_ids={row["voice_family_id"]: row["slot_id"] for row in originals}, source_binding=deepcopy(source))
    return saved


def validate_candidate_review(saved, originals):
    expected = create_candidate_review(originals)
    if (any(saved.get(key) != expected.get(key) for key in
            ("kind", "schema_version", "originals", "groups", "slot_ids", "source_binding"))):
        raise ValueError("기준 목소리 후보 선택 기록이 원본 또는 수량과 일치하지 않습니다.")
    return saved


def active_candidate_group(saved):
    return next((group["key"] for group in saved["groups"] if group["key"] not in saved["approved"]),
                saved["groups"][-1]["key"])


def _group(saved, key):
    group = next((group for group in saved["groups"] if group["key"] == key), None)
    if group is None:
        raise ValueError("현재 기준 목소리 분류를 확인하세요.")
    return group


def _editable_group(saved, key):
    group = _group(saved, key)
    if key != active_candidate_group(saved) or key in saved["approved"]:
        raise ValueError("현재 검수할 기준 목소리 분류가 아닙니다.")
    return group


def _choices(age, selected):
    group = OPTIONS["ages"][age]
    if not isinstance(selected, dict) or set(selected) != set(FIELDS):
        raise ValueError("목소리 후보의 세부 설정을 확인하세요.")
    found = {}
    for field in FIELDS:
        option = next((option for option in group["controls"][field] if option["id"] == selected[field]), None)
        if option is None:
            raise ValueError("다른 연령의 목소리 설정이 후보에 섞여 있습니다.")
        found[field] = option["promptEn"]
    return found


def candidate_instruction(gender, age, selected):
    if gender not in GENDERS or age not in AGES:
        raise ValueError("목소리 후보의 성별·연령을 확인하세요.")
    group = OPTIONS["ages"][age]
    desc = _choices(age, selected)
    base = group["basePrompt"].format(speaker=group["speakers"][gender])
    return (f"{base} The voice has {desc['pitch']} relative to this age and gender, "
            f"{desc['resonance']}, and {desc['timbre']}. "
            f"The delivery uses {desc['pace']} with {desc['intonation']}. "
            "Maintain natural Korean pronunciation and realistic conversational vocal control.")


def plan_candidate_batch(project, saved, group_key, *, rng=None):
    group = _editable_group(saved, group_key)
    if len(saved["selected"].get(group_key, [])) >= group["target"]:
        raise ValueError("이 분류의 목소리 수량을 채웠습니다. 분류를 확정하세요.")
    return _plan_candidate_batch(project, saved, group, rng=rng)


def _plan_candidate_batch(project, saved, group, *, rng=None):
    group_key = group["key"]
    _, config = load_project(project)
    from .tts import REFERENCE_TEXT
    source = rng or random.SystemRandom()
    batch_id = uuid.uuid4().hex
    previous_seeds = {seed for batches in saved["batches"].values() for batch in batches
                      for row in batch["candidates"] for seed in row["attempt_seeds"]}
    candidates = []
    sequence_start = len(saved["batches"].get(group_key, [])) * BATCH_SIZE + 1
    for number in range(sequence_start, sequence_start + BATCH_SIZE):
        controls = {field: source.choice(OPTIONS["ages"][group["age"]]["controls"][field])["id"] for field in FIELDS}
        attempt_seeds = []
        while len(attempt_seeds) != 4:
            seed = source.randrange(2**31)
            if seed not in previous_seeds:
                previous_seeds.add(seed)
                attempt_seeds.append(seed)
        candidates.append({"candidate_id": f"candidate_{batch_id}_{number}", "number": number,
                           "seed": attempt_seeds[0], "attempt_seeds": attempt_seeds,
                           "voice_controls": controls, "design_instruction": candidate_instruction(group["gender"], group["age"], controls),
                           "text": config["tts"].get("reference_text", REFERENCE_TEXT), "model": config["tts"]["design_model"]})
    plan = {"kind": BATCH_KIND, "batch_id": batch_id, "group_key": group_key,
            "split": group["split"], "gender": group["gender"], "age": group["age"],
            "originals": saved["originals"], "options_sha256": identity(OPTIONS),
            "sequence_start": sequence_start,
            "generation": deepcopy(config["tts"]["generation"]),
            "candidates": candidates, "created_at": now()}
    if _planned(saved):
        plan.update(source_binding=deepcopy(saved["source_binding"]), slot_ids=deepcopy(saved["slot_ids"]))
    return plan


def register_candidate_batch(saved, plan):
    group = _editable_group(saved, plan["group_key"])
    if len(saved["selected"].get(group["key"], [])) >= group["target"]:
        raise ValueError("이 분류의 목소리 수량을 채웠습니다. 분류를 확정하세요.")
    _register_candidate_batch(saved, plan, group)


def _register_candidate_batch(saved, plan, group, *, initial=False):
    if (plan.get("kind") != BATCH_KIND or plan.get("originals") != saved["originals"]
            or any(plan.get(key) != saved.get(key) for key in ("source_binding", "slot_ids"))
            or any(plan.get(key) != group[key] for key in ("split", "gender", "age"))
            or len(plan.get("candidates", [])) != BATCH_SIZE):
        raise ValueError("기준 목소리 후보 묶음의 실행 계획이 다릅니다.")
    batches = saved["batches"].setdefault(group["key"], [])
    if any(batch["batch_id"] == plan["batch_id"] for batch in batches):
        raise ValueError("이미 등록한 목소리 후보 묶음입니다.")
    batches.append(deepcopy(plan))
    if not initial or group["key"] not in saved["current_batch"]:
        saved["current_batch"][group["key"]] = plan["batch_id"]
    saved["revision"] += 1


def _validate_plan(plan, config):
    if (plan.get("kind") != BATCH_KIND or plan.get("gender") not in GENDERS or plan.get("age") not in AGES
            or plan.get("split") not in ("train", "validation") or plan.get("options_sha256") != identity(OPTIONS)
            or plan.get("group_key") != f"{plan['split']}:{plan['gender']}:{plan['age']}"
            or len(plan.get("candidates", [])) != BATCH_SIZE
            or len({row["candidate_id"] for row in plan["candidates"]}) != BATCH_SIZE):
        raise ValueError("목소리 후보 생성 계획을 확인하세요.")
    if (not re.fullmatch(r"[a-f0-9]{32}", plan.get("batch_id", ""))
            or type(plan.get("sequence_start")) is not int or plan["sequence_start"] < 1
            or (plan["sequence_start"] - 1) % BATCH_SIZE
            or plan.get("generation") != config["tts"]["generation"]):
        raise ValueError("목소리 후보의 묶음 식별자 또는 생성 설정이 다릅니다.")
    from .tts import REFERENCE_TEXT
    all_seeds = []
    for number, row in enumerate(plan["candidates"], plan["sequence_start"]):
        if (row["model"] != config["tts"]["design_model"]
                or row["number"] != number
                or row["candidate_id"] != f"candidate_{plan['batch_id']}_{number}"
                or row["text"] != config["tts"].get("reference_text", REFERENCE_TEXT)
                or row["design_instruction"] != candidate_instruction(plan["gender"], plan["age"], row["voice_controls"])
                or len(row.get("attempt_seeds", [])) != 4 or row["seed"] != row["attempt_seeds"][0]
                or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in row["attempt_seeds"])):
            raise ValueError("목소리 후보의 설정 또는 난수 기록이 다릅니다.")
        all_seeds.extend(row["attempt_seeds"])
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("목소리 후보의 난수가 서로 중복됩니다.")


@project_storage
def generate_candidate_batch(project, output, job_key, plan, all_hashes, *, load_model=None):
    """Generate a persisted batch, reusing a caller-supplied resident loader."""
    import soundfile as sf
    from .tts import _load_model, _seed
    from .reference_voice_review import _original_pcm_hashes
    from .clone_reference import prepare_generated_reference, GeneratedReferenceQualityError
    root, config = load_project(project)
    output = resolved(root, output)
    if not output.is_relative_to(candidate_jobs(root)) or output == candidate_jobs(root):
        raise ValueError("목소리 후보는 현재 모델의 후보 작업 폴더에 저장해야 합니다.")
    _validate_plan(plan, config)
    if plan.get("source_binding"):
        _verify_source(root, plan["source_binding"])
    hashes = set(all_hashes) | set(plan["originals"].values())
    try:
        hashes |= _original_pcm_hashes(root)
    except FileNotFoundError:
        pass
    rows, model = [], None
    with writer_lock(output):
        plan_path = output / "plan.json"
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise ValueError("저장한 목소리 후보 실행 계획이 바뀌었습니다.")
        else:
            write_json(plan_path, plan)
        plan_binding = {"path": plan_path.relative_to(root).as_posix(), "sha256": digest(plan_path)}
        for item in plan["candidates"]:
            target, record = output / f"{item['candidate_id']}.wav", output / f"{item['candidate_id']}.json"
            if record.exists():
                evidence = read_json(record)
                row = evidence["candidate"]
                _verify_candidate(root, {**row, "candidate_record": {"path": record.relative_to(root).as_posix(), "sha256": digest(record)}})
                if row["plan_binding"] != plan_binding or row["sha256"] in hashes or row["decoded_pcm_sha256"] in hashes:
                    raise ValueError("재개할 목소리 후보의 기록이 다르거나 중복되었습니다.")
            else:
                raw_directory = output / "raw"
                raw_directory.mkdir(exist_ok=True)
                for attempt, seed in enumerate(item["attempt_seeds"]):
                    raw = raw_directory / f"{item['candidate_id']}_attempt{attempt}.wav"
                    raw_base, recovery = raw, 0
                    while raw.exists() and not raw.with_name(raw.name + ".raw.json").exists():
                        recovery += 1
                        raw = raw_base.with_name(raw_base.stem + f"_recovery{recovery}.wav")
                    attempt_target = output / "prepared" / f"{item['candidate_id']}_attempt{attempt}.wav"
                    generation = {"kind": BATCH_KIND, "candidate_id": item["candidate_id"],
                                  "batch_id": plan["batch_id"], "plan_binding": plan_binding,
                                  "model": item["model"], "text": item["text"], "seed": seed,
                                  "attempt": attempt, "instruction": item["design_instruction"],
                                  "voice_controls": item["voice_controls"], "generation": plan["generation"]}
                    timing = None
                    if not raw.exists():
                        if model is None:
                            if load_model is None:
                                report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
                            model = (load_model or _load_model)(config["tts"]["design_model"], config["tts"], root)
                        _seed(seed)
                        with generation_attempt() as timing:
                            waves, rate = model.generate_voice_design(text=item["text"], language="Korean",
                                instruct=item["design_instruction"], **plan["generation"])
                        if len(waves) != 1:
                            raise ValueError("목소리 후보 생성 결과의 수량이 다릅니다.")
                        sf.write(raw, waves[0], rate, subtype="FLOAT")
                    try:
                        preparation = prepare_generated_reference(root,
                            {"path": raw.relative_to(root).as_posix(), "sha256": digest(raw), "generation": generation}, attempt_target)
                    except GeneratedReferenceQualityError:
                        if timing is not None:
                            timing.committed(False)
                        continue
                    metrics = preparation["technical_metrics"]
                    sha, pcm = preparation["prepared"]["sha256"], metrics.get("decoded_pcm_sha256")
                    if pcm and sha not in hashes and pcm not in hashes:
                        # Only a quality- and duplicate-checked choice is exposed
                        # as a published candidate WAV. Retain attempted inputs
                        # separately so an interrupted batch is fully auditable.
                        preparation = prepare_generated_reference(root,
                            {"path": raw.relative_to(root).as_posix(), "sha256": digest(raw), "generation": generation}, target)
                        metrics = preparation["technical_metrics"]
                        sha, pcm = preparation["prepared"]["sha256"], metrics["decoded_pcm_sha256"]
                        row = {"candidate_id": item["candidate_id"], "sample_id": item["candidate_id"],
                                   "voice_family_id": item["candidate_id"], "source_id": "qwen:" + item["candidate_id"],
                                   "split": plan["split"], "gender": plan["gender"], "age_group": plan["age"],
                                   "group_key": plan["group_key"], "batch_id": plan["batch_id"], "number": item["number"],
                                   "path": target.relative_to(root).as_posix(), "sha256": sha, "decoded_pcm_sha256": pcm,
                                   "seed": seed, "attempt": attempt, "voice_controls": item["voice_controls"],
                                   "design_instruction": item["design_instruction"], "description": item["design_instruction"],
                                   "text": item["text"], "model": item["model"], "style": "reference_candidate",
                                   "speed": item["voice_controls"]["pace"], "intonation": item["voice_controls"]["intonation"],
                                   "vocal_effort": "normal", "call_length": "reference_style",
                                   "provenance": "qwen_synthetic_reference", "human_reference_used": False,
                                   "technical_pass": True, "technical_metrics": metrics, "plan_binding": plan_binding,
                                   "reference_preparation": preparation,
                                   "generation_job": job_key, "created_at": now()}
                        write_json(record, {"kind": BATCH_KIND, "candidate": row})
                        if timing is not None:
                            timing.committed(True)
                        break
                    if timing is not None:
                        timing.committed(False)
                else:
                    raise ValueError("기술 검사를 통과한 서로 다른 목소리 후보를 만들지 못했습니다. 다시 시도하세요.")
            published = {**row, "candidate_record": {"path": record.relative_to(root).as_posix(), "sha256": digest(record)}}
            rows.append(published)
            hashes.update((row["sha256"], row["decoded_pcm_sha256"]))
            write_json(output.parent / "progress.json", {"kind": BATCH_KIND, "phase": "generating", "completed": len(rows),
                "total": BATCH_SIZE, "message": f"기준 목소리 후보 {len(rows)} / {BATCH_SIZE}개 생성"})
    return rows


def _verify_candidate(root, row):
    jobs = candidate_jobs(root)
    audio = _inside(root, row, jobs)
    record = read_json(_inside(root, row["candidate_record"], jobs))
    candidate = {key: value for key, value in row.items() if key != "candidate_record"}
    if record.get("kind") != BATCH_KIND or record.get("candidate") != candidate:
        raise ValueError("목소리 후보의 생성 기록이 음원과 일치하지 않습니다.")
    plan = read_json(_inside(root, row["plan_binding"], jobs))
    item = next((item for item in plan["candidates"] if item["candidate_id"] == row["candidate_id"]), None)
    if (plan.get("kind") != BATCH_KIND or item is None or row["batch_id"] != plan["batch_id"] or row["group_key"] != plan["group_key"]
            or any(row[key] != plan[key] for key in ("split", "gender")) or row["age_group"] != plan["age"]
            or row["seed"] not in item["attempt_seeds"]
            or row["attempt"] != item["attempt_seeds"].index(row["seed"])
            or row["technical_metrics"].get("decoded_pcm_sha256") != row["decoded_pcm_sha256"]
            or any(row[key] != item[key] for key in ("text", "model", "voice_controls", "design_instruction"))
            or row["description"] != row["design_instruction"] or row.get("technical_pass") is not True
            or row.get("provenance") != "qwen_synthetic_reference" or row.get("human_reference_used") is not False):
        raise ValueError("목소리 후보의 생성 설정 또는 출처가 다릅니다.")
    if "reference_preparation" in row:
        from .clone_reference import verify_generated_reference
        preparation = row["reference_preparation"]
        prepared = verify_generated_reference(root, preparation)
        expected_generation = {"kind": BATCH_KIND, "candidate_id": row["candidate_id"],
            "batch_id": row["batch_id"], "plan_binding": row["plan_binding"], "model": row["model"],
            "text": row["text"], "seed": row["seed"], "attempt": row["attempt"],
            "instruction": row["design_instruction"], "voice_controls": row["voice_controls"],
            "generation": plan["generation"]}
        if (not prepared.samefile(audio) or preparation["prepared"]["sha256"] != row["sha256"]
                or preparation["generation"] != expected_generation
                or preparation["technical_metrics"] != row["technical_metrics"]):
            raise ValueError("목소리 후보의 원본·검수용 음원 또는 정리 기록이 다릅니다.")
    return audio


def ingest_candidate_batch(root, saved, plan, rows):
    group = _editable_group(saved, plan["group_key"])
    return _ingest_candidate_batch(root, saved, plan, rows, group)


def _ingest_candidate_batch(root, saved, plan, rows, group):
    registered = next((batch for batch in saved["batches"].get(group["key"], []) if batch["batch_id"] == plan["batch_id"]), None)
    if (registered != plan or len(rows) != BATCH_SIZE
            or {row["candidate_id"] for row in rows} != {item["candidate_id"] for item in plan["candidates"]}):
        raise ValueError("요청한 다섯 후보와 생성 결과가 다릅니다.")
    incoming = {row["candidate_id"] for row in rows}
    hashes = set(saved["originals"].values()) | {value for cid, row in saved["candidates"].items() if cid not in incoming
              for value in (row["sha256"], row["decoded_pcm_sha256"])}
    for row in rows:
        _verify_candidate(root, row)
        previous = saved["candidates"].get(row["candidate_id"])
        if previous is not None and previous != row:
            raise ValueError("이미 받은 목소리 후보의 기록이 바뀌었습니다.")
        if row["batch_id"] != plan["batch_id"] or row["group_key"] != plan["group_key"]:
            raise ValueError("다른 묶음의 목소리 후보가 포함되어 있습니다.")
        if row["sha256"] in hashes or row["decoded_pcm_sha256"] in hashes:
            raise ValueError("다른 분류 또는 기존 후보와 중복된 음원이 있습니다.")
        hashes.update((row["sha256"], row["decoded_pcm_sha256"]))
    if any(cid not in saved["candidates"] for cid in incoming):
        saved["candidates"].update({row["candidate_id"]: deepcopy(row) for row in rows})
        saved["revision"] += 1


def _finished_initial_batch(root, saved, group, *, verify=True):
    for plan in saved["batches"].get(group["key"], []):
        ids = [item["candidate_id"] for item in plan["candidates"]]
        if len(ids) != BATCH_SIZE or len(set(ids)) != BATCH_SIZE:
            raise ValueError("기준 목소리 후보 묶음의 수량이 다릅니다.")
        if not all(cid in saved["candidates"] for cid in ids):
            continue
        for cid in ids:
            row = saved["candidates"][cid]
            if row["group_key"] != group["key"] or row["batch_id"] != plan["batch_id"]:
                raise ValueError("기준 목소리 후보 묶음의 연결이 다릅니다.")
            if verify:
                _verify_candidate(root, row)
        return plan
    return None


def preparation_status(root, saved=None, *, verify=True):
    root = Path(root).resolve()
    if saved is None:
        try:
            saved = read_json(root / ".studio/reference-candidates/state.json")
        except FileNotFoundError:
            saved = {"groups": [], "batches": {}, "candidates": {}}
    ready = {group["key"]: _finished_initial_batch(root, saved, group, verify=verify)
             for group in saved["groups"]}
    rows = [{"split": split, "label": label,
             "completed": BATCH_SIZE * sum(bool(ready[group["key"]]) for group in saved["groups"] if group["split"] == split),
             "total": 50} for split, label in (("train", "학습용 목소리 후보"), ("validation", "공통 평가용 목소리 후보"))]
    current, phase = None, {}
    try:
        phase = read_json(root / ".studio/reference-candidates/preparation/state.json")
        current = phase.get("current")
        if current and not ready.get(current["group"]):
            output = resolved(root, current["output"])
            if not output.is_relative_to(candidate_jobs(root)):
                raise ValueError("준비 중인 목소리 후보의 저장 위치가 다릅니다.")
            progress = read_json(output.parent / "progress.json")
            amount = progress.get("completed", 0)
            if type(amount) is int and 0 <= amount <= BATCH_SIZE:
                split = _group(saved, current["group"])["split"]
                next(row for row in rows if row["split"] == split)["completed"] += amount
    except FileNotFoundError:
        pass
    groups_done = sum(bool(plan) for plan in ready.values())
    completed = sum(row["completed"] for row in rows)
    reused = min(completed, phase.get("reused_candidates", 0))
    receipt = phase.get("receipt") if groups_done == 20 and phase.get("status") == "completed" else None
    if receipt and verify:
        _inside(root, receipt)
    return {"kind": PREPARATION_KIND, "status": "completed" if receipt else phase.get("status", "pending"),
            "completed": completed, "total": 100, "reused_candidates": reused, "generated_candidates": completed - reused,
            "completed_groups": groups_done, "total_groups": 20, "rows": rows, "current": current,
            **({"receipt": receipt} if receipt else {})}


def _preparation_output(root, saved, plan):
    matches = []
    jobs = candidate_jobs(root)
    prior = saved.get("job") or {}
    if prior.get("group") == plan["group_key"] and prior.get("request_sha256"):
        directory = resolved(root, prior["directory"]) if prior.get("directory") else jobs / prior["task_id"]
        request_path = _inside(root, {"path": str(directory / "request.json"),
                                     "sha256": prior["request_sha256"]}, jobs)
        prior_plan = read_json(request_path).get("plan")
        if (not isinstance(prior_plan, dict) or prior_plan.get("batch_id") == plan["batch_id"] and prior_plan != plan):
            raise ValueError("기존 후보 작업의 요청과 저장한 계획이 다릅니다.")
    for path in jobs.glob("*/audio/plan.json"):
        stored = read_json(path)
        if stored.get("batch_id") == plan["batch_id"] and stored != plan:
            raise ValueError("기존 후보 묶음의 생성 계획이 변경되었습니다. 새 폴더로 대체하지 않습니다.")
        if stored == plan:
            matches.append(path.parent)
    for path in jobs.glob("*/request.json"):
        request = read_json(path)
        stored = request.get("plan")
        if isinstance(stored, dict) and stored.get("batch_id") == plan["batch_id"] and stored != plan:
            raise ValueError("기존 후보 작업의 생성 계획이 변경되었습니다. 새 폴더로 대체하지 않습니다.")
        if request.get("kind") == BATCH_KIND and request.get("plan") == plan:
            matches.append(path.parent / "audio")
    unique = set(path.resolve() for path in matches)
    if len(unique) > 1:
        raise ValueError("같은 후보 생성 계획이 여러 작업 폴더에 있어 자동 재개할 수 없습니다.")
    return next(iter(unique), jobs / ("initial_" + plan["batch_id"]) / "audio")


@project_storage
def prepare_initial_candidate_batches(project, originals, *, load_model=None, state_path=None,
                                      progress_path=None, pause_path=None, should_pause=None):
    """Prepare one complete stored batch for every group without changing human decisions."""
    root, config = load_project(project)
    root = Path(root).resolve()
    originals = list(originals)
    state_path = resolved(root, state_path or ".studio/reference-candidates/state.json")
    directory = root / ".studio/reference-candidates/preparation"
    phase_path = directory / "state.json"
    progress_path = resolved(root, progress_path or directory / "progress.json")
    pause_path = resolved(root, pause_path or directory / "pause.json")
    for path in (state_path, progress_path, pause_path):
        if not path.is_relative_to(root):
            raise ValueError("기준 목소리 준비 파일은 현재 모델 폴더에 저장해야 합니다.")
    try:
        saved = read_json(state_path)
        expected_sha = digest(state_path)
    except FileNotFoundError:
        saved, expected_sha = create_candidate_review(originals), None
    validate_candidate_review(saved, originals)
    if (saved.get("job") or {}).get("status") == "running":
        raise ValueError("진행 중인 후보 생성 작업을 확인한 뒤 준비를 시작하세요.")
    protected = {key: deepcopy(saved[key]) for key in ("selected", "heard", "approved", "current_batch")}
    old_candidates = deepcopy(saved["candidates"])
    def save():
        nonlocal expected_sha
        actual = digest(state_path) if state_path.exists() else None
        if actual != expected_sha:
            raise ValueError("준비 중 검수 기록이 외부에서 변경되었습니다. 기존 기록을 덮어쓰지 않습니다.")
        if any(saved[key] != protected[key] for key in ("selected", "heard", "approved")):
            raise ValueError("목소리 준비가 기존 검수 판정을 변경했습니다.")
        if any(saved["current_batch"].get(key) != value for key, value in protected["current_batch"].items()):
            raise ValueError("목소리 준비가 기존 후보 목록 위치를 변경했습니다.")
        if any(saved["candidates"].get(cid) != row for cid, row in old_candidates.items()):
            raise ValueError("목소리 준비가 기존 음원 기록을 변경했습니다.")
        write_json(state_path, saved)
        expected_sha = digest(state_path)
    model = None
    def cached_model(*args):
        nonlocal model
        if model is None:
            report_progress("tts_model_loading", "음성 생성 모델 준비", eta_eligible=False)
            from .tts import _load_model
            model = (load_model or _load_model)(*args)
        return model
    with writer_lock(directory / "lock"):
        save()
        preservation = snapshot_json(root, {"kind": PREPARATION_KIND, "protected": protected,
                                           "candidate_records": old_candidates}, "reference_preparation_preserved")
        source_id = identity({key: saved.get(key) for key in ("kind", "originals", "slot_ids", "source_binding")})
        try:
            previous_phase = read_json(phase_path)
        except FileNotFoundError:
            previous_phase = {}
        reuse = preparation_status(root, saved, verify=False)["completed_groups"] * BATCH_SIZE
        if previous_phase.get("source_id") == source_id and previous_phase.get("status") in ("running", "paused"):
            reuse = previous_phase["reused_candidates"]
        phase = {"kind": PREPARATION_KIND, "status": "running", "preserved": preservation,
                 "source_id": source_id, "reused_candidates": reuse, "current": None,
                 "terminal_failures": deepcopy(previous_phase.get("terminal_failures", {}))
                    if previous_phase.get("source_id") == source_id else {}}
        write_json(phase_path, phase)
        for group in saved["groups"]:
            if _finished_initial_batch(root, saved, group):
                continue
            if group["key"] in saved["approved"]:
                raise ValueError("확정한 분류의 기존 후보가 누락되었습니다. 새로 만들지 않습니다.")
            if (should_pause and should_pause()) or pause_path.exists():
                status = {**preparation_status(root, saved, verify=False), "status": "paused"}
                write_json(phase_path, {**phase, "status": "paused"})
                write_json(progress_path, status)
                return status
            plans = saved["batches"].get(group["key"], [])
            if plans:
                plan = plans[-1]
                failed = phase["terminal_failures"].get(plan["batch_id"])
                old_job = saved.get("job") or {}
                old_output = _preparation_output(root, saved, plan)
                if not failed and (old_job.get("group") == group["key"] and old_job.get("status") == "failed"
                                   and old_job.get("resumable") is False):
                    request_path = old_output.parent / "request.json"
                    result_path = old_output.parent / "result.json"
                    if (request_path.exists() and result_path.exists()
                            and digest(request_path) == old_job.get("request_sha256")):
                        request, result = read_json(request_path), read_json(result_path)
                        if (request.get("plan") == plan and result.get("request_sha256") == digest(request_path)
                                and result.get("error") == _TERMINAL_CANDIDATE_QC_ERROR):
                            failed = snapshot_json(root, {"kind": PREPARATION_KIND, "plan": plan,
                                "reason": _TERMINAL_CANDIDATE_QC_ERROR,
                                "worker_result": {"path": result_path.relative_to(root).as_posix(), "sha256": digest(result_path)}},
                                "reference_preparation_failed_batch")
                            phase["terminal_failures"][plan["batch_id"]] = failed
                if failed:
                    evidence = read_json(_inside(root, failed))
                    if evidence.get("plan") != plan or evidence.get("reason") != _TERMINAL_CANDIDATE_QC_ERROR:
                        raise ValueError("다시 시도할 후보 묶음의 최종 실패 기록이 다릅니다.")
                    # Only a new explicit preparation invocation reaches here.
                    # Exhausted seeds never trigger an automatic inference loop.
                    plan = _plan_candidate_batch(project, saved, group)
                    _register_candidate_batch(saved, plan, group, initial=True)
                    save()
            else:
                plan = _plan_candidate_batch(project, saved, group)
                _register_candidate_batch(saved, plan, group, initial=True)
                save()  # Random options and every retry seed are durable before inference.
            output = _preparation_output(root, saved, plan)
            phase["current"] = {"group": group["key"], "batch_id": plan["batch_id"],
                                "output": output.relative_to(root).as_posix()}
            write_json(phase_path, phase)
            write_json(progress_path, {**preparation_status(root, saved, verify=False), "status": "running"})
            hashes = set(saved["originals"].values()) | {value for row in saved["candidates"].values()
                      for value in (row["sha256"], row["decoded_pcm_sha256"])}
            for prior_plans in saved["batches"].values():
                for prior_plan in prior_plans:
                    if prior_plan["batch_id"] not in phase["terminal_failures"]:
                        continue
                    prior_output = _preparation_output(root, saved, prior_plan)
                    for item in prior_plan["candidates"]:
                        record_path = prior_output / f"{item['candidate_id']}.json"
                        if record_path.exists():
                            row = read_json(record_path)["candidate"]
                            _verify_candidate(root, {**row, "candidate_record": {
                                "path": record_path.relative_to(root).as_posix(), "sha256": digest(record_path)}})
                            hashes.update((row["sha256"], row["decoded_pcm_sha256"]))
            try:
                rows = generate_candidate_batch(project, output, "initial_" + plan["batch_id"], plan, hashes,
                                                load_model=cached_model)
            except ValueError as error:
                if str(error) == _TERMINAL_CANDIDATE_QC_ERROR:
                    failed = snapshot_json(root, {"kind": PREPARATION_KIND, "plan": plan,
                        "reason": str(error), "output": output.relative_to(root).as_posix()},
                        "reference_preparation_failed_batch")
                    phase["terminal_failures"][plan["batch_id"]] = failed
                    phase.update(status="failed", message=str(error), resumable=False)
                    write_json(phase_path, phase)
                    write_json(progress_path, {**preparation_status(root, saved, verify=False),
                        "status": "failed", "message": str(error), "resumable": False})
                raise
            _ingest_candidate_batch(root, saved, plan, rows, group)
            if (saved.get("job") or {}).get("group") == group["key"]:
                # Old request and result files remain immutable; only the completed job pointer is released.
                saved["job"] = None
            save()
        report_progress("reference_preparation_finalizing", "준비한 기준 목소리 검증", eta_eligible=False)
        status = preparation_status(root, saved)
        if status["completed_groups"] != 20:
            raise ValueError("모든 분류의 첫 후보 다섯 개가 준비되지 않았습니다.")
        batches = {group["key"]: _finished_initial_batch(root, saved, group, verify=False)["batch_id"] for group in saved["groups"]}
        receipt = snapshot_json(root, {"kind": PREPARATION_KIND, "originals": saved["originals"],
            "slot_ids": saved.get("slot_ids"), "source_binding": saved.get("source_binding"),
            "initial_batches": batches, "scope": "Stored candidates only; no human judgment or reference admission"},
            "reference_initial_candidates")
        result = {**status, "status": "completed", "receipt": receipt, "current": None}
        write_json(phase_path, {**phase, "status": "completed", "current": None, "receipt": receipt})
        write_json(progress_path, result)
        return result


def verify_initial_candidate_preparation(project, originals, binding):
    root, _ = load_project(project)
    saved = read_json(root / ".studio/reference-candidates/state.json")
    validate_candidate_review(saved, originals)
    receipt = read_json(_inside(root, binding))
    batches = {group["key"]: _finished_initial_batch(root, saved, group) for group in saved["groups"]}
    expected = {"kind": PREPARATION_KIND, "originals": saved["originals"], "slot_ids": saved.get("slot_ids"),
        "source_binding": saved.get("source_binding"),
        "initial_batches": {key: plan["batch_id"] for key, plan in batches.items() if plan},
        "scope": "Stored candidates only; no human judgment or reference admission"}
    if len(expected["initial_batches"]) != 20 or receipt != expected:
        raise ValueError("준비한 기준 목소리 후보 묶음의 검증 기록이 다릅니다.")
    return {**preparation_status(root, saved), "receipt": binding}


def candidate_action(root, saved, action, *, group_key=None, candidate_id=None, sha256=None):
    group = _editable_group(saved, group_key or active_candidate_group(saved))
    key = group["key"]
    if action in ("hear", "toggle"):
        row = saved["candidates"].get(candidate_id)
        if not row or row["group_key"] != key or row["sha256"] != sha256:
            raise ValueError("목소리 후보가 바뀌었습니다. 목록을 다시 불러오세요.")
        _verify_candidate(root, row)
        if action == "hear":
            saved["heard"][candidate_id] = sha256
        else:
            selected = saved["selected"].setdefault(key, [])
            if candidate_id in selected:
                selected.remove(candidate_id)
            else:
                if not _listening_matches(saved["heard"], candidate_id, sha256, optional=True):
                    raise ValueError("목소리 후보의 청취 기록이 음원과 다릅니다.")
                if len(selected) >= group["target"]:
                    raise ValueError("이 분류에서 선택할 수 있는 최대 수량을 채웠습니다.")
                selected.append(candidate_id)
    elif action == "approve":
        selected = saved["selected"].get(key, [])
        if len(selected) != group["target"]:
            raise ValueError("이 분류의 필요한 목소리 수량을 채운 뒤 확정하세요.")
        for cid in selected:
            row = saved["candidates"][cid]
            _verify_candidate(root, row)
            if row["group_key"] != key or not _listening_matches(saved["heard"], cid, row["sha256"], optional=True):
                raise ValueError("선택한 목소리의 청취 또는 분류 기록이 다릅니다.")
        saved["approved"].append(key)
    else:
        raise ValueError("목소리 후보 선택 요청을 확인하세요.")
    saved["revision"] += 1


def _optional_listening(manifest):
    """Unmarked frozen manifests retain their original all-listened contract."""
    if "listening_policy" not in manifest:
        return False
    if manifest["listening_policy"] != OPTIONAL_LISTENING_POLICY:
        raise ValueError("기준 목소리 선택의 청취 정책을 확인하세요.")
    return True


def _listening_matches(heard, candidate_id, sha256, *, optional):
    if not isinstance(heard, dict):
        raise ValueError("기준 목소리 청취 기록의 형식이 다릅니다.")
    # Missing listening evidence is permitted only by the explicit new policy;
    # an existing observation must still match the immutable candidate WAV.
    return heard[candidate_id] == sha256 if candidate_id in heard else optional


def candidate_review_view(root, saved):
    key = active_candidate_group(saved)
    current = saved["current_batch"].get(key)
    selected = set(saved["selected"].get(key, []))
    def public(row):
        return {**deepcopy(row), "id": row["candidate_id"], "selected": row["candidate_id"] in selected,
                "heard": saved["heard"].get(row["candidate_id"]) == row["sha256"]}
    items = [public(row) for row in saved["candidates"].values() if row["group_key"] == key and row["batch_id"] == current]
    accepted = [public(saved["candidates"][cid]) for cid in saved["selected"].get(key, [])]
    groups = [{**group, "selected_count": len(saved["selected"].get(group["key"], [])),
               "approved": group["key"] in saved["approved"]} for group in saved["groups"]]
    return {"mode": CANDIDATE_POLICY, "groups": groups, "active_group": key, "items": items,
            "accepted": accepted, "review_complete": len(saved["approved"]) == len(groups),
            "revision": saved["revision"], "batch_size": BATCH_SIZE}


def candidate_audio(root, saved, candidate_id, revision):
    row = saved["candidates"].get(candidate_id)
    if row is None or revision not in (None, row["sha256"]):
        raise ValueError("목소리 후보가 바뀌었습니다. 목록을 다시 불러오세요.")
    return _verify_candidate(root, row)


def _replacement(original, candidate):
    return {**{key: value for key, value in original.items() if key not in ("voice_design", "voice_number")},
            **deepcopy(candidate), "voice_family_id": original["voice_family_id"],
            "previous_reference_sha256": original["sha256"],
            "replaces_reference_sha256": original.get("replaces_reference_sha256", original["sha256"])}


def freeze_candidate_review(root, saved, originals):
    root = Path(root).resolve()
    if _planned(saved):
        return _freeze_planned_review(root, saved, originals)
    validate_candidate_review(saved, originals)
    voice_bank = {"path": "voices.json", "sha256": digest(root / "voices.json")}
    if _source_groups(root, originals, voice_bank) != saved["groups"]:
        raise ValueError("선택한 기준 목소리의 원본 성별·연령 분류가 다릅니다.")
    if set(saved["approved"]) != {group["key"] for group in saved["groups"]}:
        raise ValueError("모든 성별·연령 분류의 목소리를 선택하고 확정하세요.")
    by_family = {row["voice_family_id"]: row for row in originals}
    replacements = {}
    seen = set(saved["originals"].values())
    for group in saved["groups"]:
        selected = saved["selected"].get(group["key"], [])
        if len(selected) != group["target"] or len(set(selected)) != len(selected):
            raise ValueError("확정한 기준 목소리 수량이 분류 목표와 다릅니다.")
        for family, cid in zip(group["families"], selected):
            row = saved["candidates"][cid]
            _verify_candidate(root, row)
            if (row["group_key"] != group["key"] or not _listening_matches(saved["heard"], cid, row["sha256"], optional=True)
                    or row["sha256"] in seen or row["decoded_pcm_sha256"] in seen):
                raise ValueError("선택한 기준 목소리가 중복되거나 검수 기록이 다릅니다.")
            original = by_family[family]
            _inside(root, original)
            if row["text"] != original["text"] or row["model"] != original["model"]:
                raise ValueError("선택한 기준 목소리의 문장 또는 모델이 원본 계약과 다릅니다.")
            replacements[family] = _replacement(original, row)
            seen.update((row["sha256"], row["decoded_pcm_sha256"]))
    return {"kind": CANDIDATE_REVIEW_KIND, "listening_policy": OPTIONAL_LISTENING_POLICY, "originals": saved["originals"],
            "replacements": replacements, "heard": saved["heard"], "voice_bank": voice_bank,
            "approved_groups": list(saved["approved"]), "groups": saved["groups"],
            "selections": saved["selected"], "saved_at": now(),
            "source_scope": "Human candidate selections entered through the local reference review screen"}


def overlay_candidate_banks(root, banks, manifest):
    root = Path(root).resolve()
    optional_listening = _optional_listening(manifest)
    originals = [row for bank in banks.values() for row in bank]
    hashes = {row["voice_family_id"]: row["sha256"] for row in originals}
    replacements = manifest.get("replacements")
    if (manifest.get("kind") != CANDIDATE_REVIEW_KIND or manifest.get("originals") != hashes
            or len(hashes) != len(originals) or not isinstance(replacements, dict)
            or set(replacements) != set(hashes)
            or manifest.get("source_scope") != "Human candidate selections entered through the local reference review screen"):
        raise ValueError("확정한 목소리 후보 목록이 원본 기준 목소리와 다릅니다.")
    groups = manifest.get("groups", [])
    if (groups != _source_groups(root, originals, manifest["voice_bank"])
            or len(groups) != 20 or set(manifest.get("approved_groups", [])) != {g["key"] for g in groups}
            or set(manifest.get("selections", {})) != {g["key"] for g in groups}):
        raise ValueError("확정한 목소리 후보 분류 기록이 불완전합니다.")
    by_family = {row["voice_family_id"]: row for row in originals}
    seen, selected_ids, mapped = set(hashes.values()), set(), set()
    for group in groups:
        ids = manifest["selections"].get(group["key"], [])
        if len(ids) != group["target"] or len(group["families"]) != group["target"]:
            raise ValueError("확정한 목소리 후보 분류 수량이 다릅니다.")
        for family, cid in zip(group["families"], ids):
            row = replacements.get(family)
            if not row or family in mapped or cid in selected_ids or family not in by_family:
                raise ValueError("기준 목소리 후보가 다른 분류에 중복 배정되었습니다.")
            original = by_family[family]
            _inside(root, original)
            record = read_json(_inside(root, row["candidate_record"], candidate_jobs(root)))
            candidate = {**record["candidate"], "candidate_record": row["candidate_record"]}
            _verify_candidate(root, candidate)
            if (row != _replacement(original, candidate) or candidate["candidate_id"] != cid
                    or row["group_key"] != group["key"] or row["split"] != original["split"]
                    or row["text"] != original["text"] or row["model"] != original["model"]
                    or not _listening_matches(manifest.get("heard"), cid, row["sha256"], optional=optional_listening)
                    or row["sha256"] in seen or row["decoded_pcm_sha256"] in seen):
                raise ValueError("선택한 기준 목소리의 출처 또는 검수 연결이 다릅니다.")
            selected_ids.add(cid)
            mapped.add(family)
            seen.update((row["sha256"], row["decoded_pcm_sha256"]))
    if mapped != set(hashes):
        raise ValueError("선택한 기준 목소리가 모든 원본 슬롯을 채우지 못했습니다.")
    return {split: [replacements[row["voice_family_id"]] for row in bank] for split, bank in banks.items()}


def _canonical_reference(slot, candidate):
    """Assign a chosen waveform to a real family slot without claiming a prior waveform."""
    return {**deepcopy(candidate), "sample_id": slot["voice_family_id"],
            "voice_family_id": slot["voice_family_id"], "source_id": "qwen:" + slot["voice_family_id"],
            "status": "accepted", "selection_slot_id": slot["slot_id"],
            "reference_selection_policy": PLANNED_CANDIDATE_POLICY}


def _planned_slots(root, source):
    _verify_source(root, source)
    settings = {f"{split}_references": count for split, count in source["counts"].items()}
    return source_slots(_inside(root, source["project"]), settings)


def _planned_manifest_rows(root, manifest):
    """Verify the full selection against planned identities and original candidate receipts."""
    if manifest.get("kind") != PLANNED_REVIEW_KIND:
        raise ValueError("선택한 기준 목소리 은행의 형식이 다릅니다.")
    optional_listening = _optional_listening(manifest)
    slots = _planned_slots(root, manifest["source_binding"])
    groups = candidate_groups(slots)
    expected_ids = {slot["voice_family_id"]: slot["slot_id"] for slot in slots}
    if (manifest.get("slot_ids") != expected_ids or manifest.get("groups") != groups
            or manifest.get("approved_groups") != [group["key"] for group in groups]
            or set(manifest.get("selections", {})) != {group["key"] for group in groups}
            or set(manifest.get("references", {})) != set(expected_ids)
            or manifest.get("source_scope") != "Human candidate selections entered through the local reference review screen"):
        raise ValueError("선택한 기준 목소리가 원본 슬롯 및 분류와 일치하지 않습니다.")
    by_family = {slot["voice_family_id"]: slot for slot in slots}
    seen, candidate_ids, rows = set(), set(), {}
    for group in groups:
        selected = manifest["selections"][group["key"]]
        if len(selected) != group["target"]:
            raise ValueError("선택한 기준 목소리의 분류별 수량이 다릅니다.")
        for family, cid in zip(group["families"], selected):
            row = manifest["references"][family]
            record = read_json(_inside(root, row["candidate_record"], candidate_jobs(root)))
            candidate = {**record["candidate"], "candidate_record": row["candidate_record"]}
            _verify_candidate(root, candidate)
            plan = read_json(_inside(root, candidate["plan_binding"], candidate_jobs(root)))
            slot = by_family[family]
            if (cid in candidate_ids or row != _canonical_reference(slot, candidate)
                    or candidate["candidate_id"] != cid or candidate["group_key"] != group["key"]
                    or candidate["split"] != group["split"] or candidate["gender"] != group["gender"]
                    or candidate["age_group"] != group["age"]
                    or candidate["text"] != slot["text"] or candidate["model"] != slot["model"]
                    or plan.get("source_binding") != manifest["source_binding"]
                    or plan.get("slot_ids") != expected_ids
                    or not _listening_matches(manifest.get("heard"), cid, candidate["sha256"], optional=optional_listening)
                    or candidate["sha256"] in seen or candidate["decoded_pcm_sha256"] in seen):
                raise ValueError("선택한 기준 목소리의 음원, 설정 또는 청취 기록이 다릅니다.")
            seen.update((candidate["sha256"], candidate["decoded_pcm_sha256"]))
            candidate_ids.add(cid)
            rows[family] = row
    if set(manifest["heard"]) - candidate_ids:
        raise ValueError("선택한 기준 목소리 외의 청취 기록이 포함되어 있습니다.")
    body = {key: manifest[key] for key in ("source_binding", "slot_ids", "groups", "approved_groups",
                                          "selections", "heard", "references", "source_scope")}
    if optional_listening:
        body["listening_policy"] = manifest["listening_policy"]
    if manifest.get("selection_id") != identity(body):
        raise ValueError("선택한 기준 목소리 묶음의 식별자가 다릅니다.")
    return slots, rows


def _freeze_planned_review(root, saved, slots):
    """Publish only human-chosen WAVs, after every selected WAV passes fresh QC."""
    validate_candidate_review(saved, slots)
    if saved.get("approved") != [group["key"] for group in saved["groups"]]:
        raise ValueError("모든 성별·연령 분류의 목소리를 선택하고 확정하세요.")
    _verify_source(root, saved["source_binding"])
    by_family = {slot["voice_family_id"]: slot for slot in slots}
    references, heard = {}, {}
    for group in saved["groups"]:
        selected = saved["selected"].get(group["key"], [])
        if len(selected) != group["target"]:
            raise ValueError("선택한 기준 목소리 수량이 분류 목표와 다릅니다.")
        for family, cid in zip(group["families"], selected):
            candidate = saved["candidates"][cid]
            if not _listening_matches(saved["heard"], cid, candidate["sha256"], optional=True):
                raise ValueError("선택한 기준 목소리의 청취 기록이 음원과 다릅니다.")
            references[family] = _canonical_reference(by_family[family], candidate)
            if cid in saved["heard"]:
                heard[cid] = saved["heard"][cid]
    manifest = {"kind": PLANNED_REVIEW_KIND, "listening_policy": OPTIONAL_LISTENING_POLICY, "source_binding": deepcopy(saved["source_binding"]),
                "slot_ids": deepcopy(saved["slot_ids"]), "groups": deepcopy(saved["groups"]),
                "approved_groups": list(saved["approved"]), "selections": deepcopy(saved["selected"]),
                "heard": heard, "references": references,
                "source_scope": "Human candidate selections entered through the local reference review screen"}
    manifest["selection_id"] = identity({key: value for key, value in manifest.items() if key != "kind"})
    actual_slots, actual_rows = _planned_manifest_rows(root, manifest)
    from .reference_qc import REFERENCE_QC_POLICY, _RULES
    checks = []
    seen_pcm = set()
    # Complete verification before publishing any journal or admitting a family.
    for slot in actual_slots:
        row = actual_rows[slot["voice_family_id"]]
        reasons, metrics = _check_wave(_inside(root, row, candidate_jobs(root)))
        pcm = metrics.get("decoded_pcm_sha256")
        if (reasons or not pcm or pcm in seen_pcm or metrics != row["technical_metrics"]
                or pcm != row["decoded_pcm_sha256"] or digest(resolved(root, row["path"])) != row["sha256"]):
            raise ValueError("선택한 기준 목소리의 기술 검사 또는 중복 검사에 실패했습니다.")
        seen_pcm.add(pcm)
        checks.append({"voice_family_id": row["voice_family_id"], "sample_id": row["sample_id"],
                       "split": row["split"], "path": row["path"], "sha256": row["sha256"],
                       "actual_sha256": row["sha256"], "reasons": [], "metrics": metrics, "passed": True})
    with writer_lock(root / ".studio/reference-candidates/publication"):
        selected_banks = {split: [actual_rows[slot["voice_family_id"]] for slot in actual_slots if slot["split"] == split]
                          for split in ("train", "validation")}
        for split, selected in selected_banks.items():
            path = root / "data/references" / split / "references.jsonl"
            if path.exists() and path.stat().st_size and read_rows(path) != selected:
                raise FileExistsError("Existing immutable reference bank differs from the chosen candidates")
        journals = []
        for split, selected in selected_banks.items():
            path = root / "data/references" / split / "references.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_immutable_rows(path, selected, replace_empty=True)
            journals.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path)})
        directory = root / "review/reference_qc"
        directory.mkdir(parents=True, exist_ok=True)
        fingerprint = identity({"selection_id": manifest["selection_id"], "checks": checks, "journals": journals})
        quality_path = directory / f"{fingerprint}.jsonl"
        write_immutable_rows(quality_path, checks)
        quality_binding = {"path": quality_path.relative_to(root).as_posix(), "sha256": digest(quality_path)}
        source_config = read_json(_inside(root, saved["source_binding"]["project"]))
        receipt = {"schema_version": 1, "kind": "reference_technical_admission", "policy_id": REFERENCE_QC_POLICY,
                   "state": "accepted", "rules": _RULES, "selection_policy": PLANNED_CANDIDATE_POLICY,
                   "wakeword": source_config.get("wakeword"), "tts_settings_id": identity(source_config["tts"]),
                   "selection_id": manifest["selection_id"], "selected_counts": saved["source_binding"]["counts"],
                   "voice_bank": saved["source_binding"]["voice_bank"], "generation_journals": journals,
                   "quality_journal": quality_binding, "checked_audio_sha256": {row["voice_family_id"]: row["sha256"] for row in checks},
                   "accepted_reference_families": [row["voice_family_id"] for row in checks],
                   "human_listened_count": len(heard), "independent_STT_runs": 0, "acoustic_diversity_measured": False}
        receipt_path = directory / f"{fingerprint}.json"
        if receipt_path.exists():
            if read_json(receipt_path) != receipt:
                raise ValueError("기준 목소리 기술 검사 기록이 바뀌었습니다.")
        else:
            write_json(receipt_path, receipt)
        receipt_binding = {"path": receipt_path.relative_to(root).as_posix(), "sha256": digest(receipt_path)}
        approval_path = root / "approval.json"
        approval = read_json(approval_path)
        updated = {**approval, "accepted_reference_families": receipt["accepted_reference_families"],
                   "reference_technical_admission": receipt_binding}
        if updated != approval:
            write_json(approval_path, updated)
    return {**manifest, "journals": journals, "technical_admission": receipt_binding,
            "quality_journal": quality_binding, "saved_at": now()}


def overlay_planned_banks(root, banks, manifest):
    """Read a frozen chosen bank for metadata slots or the canonical admitted bank."""
    root = Path(root).resolve()
    slots, rows = _planned_manifest_rows(root, manifest)
    expected_ids = {slot["voice_family_id"] for slot in slots}
    provided = [row for bank in banks.values() for row in bank]
    if len(provided) != len(expected_ids) or {row["voice_family_id"] for row in provided} != expected_ids:
        raise ValueError("기준 목소리 은행의 화자 슬롯 수량이 다릅니다.")
    by_slot = {slot["voice_family_id"]: slot for slot in slots}
    for split, bank in banks.items():
        for row in bank:
            expected = by_slot[row["voice_family_id"]] if row.get("source_slot") else rows[row["voice_family_id"]]
            if row != expected or row["split"] != split:
                raise ValueError("기준 목소리 원본 슬롯 또는 선택 은행이 바뀌었습니다.")
    expected_journals = []
    for split in ("train", "validation"):
        journal = root / "data/references" / split / "references.jsonl"
        expected = [rows[slot["voice_family_id"]] for slot in slots if slot["split"] == split]
        if read_rows(journal) != expected:
            raise ValueError("선택한 기준 목소리의 공개 은행이 바뀌었습니다.")
        expected_journals.append({"path": journal.relative_to(root).as_posix(), "sha256": digest(journal)})
    receipt = read_json(_inside(root, manifest["technical_admission"]))
    quality = read_rows(_inside(root, manifest["quality_journal"]))
    from .reference_qc import REFERENCE_QC_POLICY, _RULES
    source_config = read_json(_inside(root, manifest["source_binding"]["project"]))
    expected_hashes = {family: row["sha256"] for family, row in rows.items()}
    expected_checks = [{"voice_family_id": row["voice_family_id"], "sample_id": row["sample_id"],
        "split": row["split"], "path": row["path"], "sha256": row["sha256"], "actual_sha256": row["sha256"],
        "reasons": [], "metrics": row["technical_metrics"], "passed": True}
        for slot in slots for row in (rows[slot["voice_family_id"]],)]
    if (manifest.get("journals") != expected_journals or receipt.get("generation_journals") != expected_journals
            or receipt.get("kind") != "reference_technical_admission" or receipt.get("policy_id") != REFERENCE_QC_POLICY
            or receipt.get("state") != "accepted" or receipt.get("selection_policy") != PLANNED_CANDIDATE_POLICY
            or receipt.get("selection_id") != manifest["selection_id"] or receipt.get("quality_journal") != manifest["quality_journal"]
            or receipt.get("rules") != _RULES or receipt.get("voice_bank") != manifest["source_binding"]["voice_bank"]
            or receipt.get("selected_counts") != manifest["source_binding"]["counts"]
            or receipt.get("wakeword") != source_config.get("wakeword")
            or receipt.get("tts_settings_id") != identity(source_config["tts"])
            or type(receipt.get("human_listened_count")) is not int
            or receipt.get("human_listened_count") != len(manifest["heard"])
            or receipt.get("checked_audio_sha256") != expected_hashes
            or set(receipt.get("accepted_reference_families", [])) != expected_ids
            or quality != expected_checks):
        raise ValueError("선택한 기준 목소리의 기술 검사 승인 기록이 다릅니다.")
    return {split: [rows[row["voice_family_id"]] for row in bank] for split, bank in banks.items()}
