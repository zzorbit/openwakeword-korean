"""Signal-quality admission for synthetic reference voices; never listening."""
import hashlib

import numpy as np
import soundfile as sf

from .common import bound_file, digest, identity, now, read_json, read_rows, resolved, write_json, write_immutable_rows, writer_lock
from .project import load_project


REFERENCE_QC_POLICY = "reference_signal_quality_v1"
_RULES = {"min_duration_s": .25, "max_duration_s": 30., "min_rms": .001,
          "max_abs_sample": 1., "clipping_level": .999, "max_clipping_fraction": .01,
          "min_sample_rate": 8000, "max_sample_rate": 96000}


def _human_answers(root, approval):
    """Use actual hash-bound answers; a later keep can resolve an old veto."""
    reports = []
    for binding in approval.get("reused_reference_review_evidence", []) + approval.get("reference_review_evidence", []):
        reports.append(read_json(bound_file(root, binding)))
    summary = root / "review/references_results.json"
    if approval.get("reference_review_sha256"):
        if not summary.exists() or digest(summary) != approval["reference_review_sha256"]:
            raise ValueError("Existing reference listening evidence changed")
        reports.append(read_json(summary))
    answers = {}
    for report in sorted(reports, key=lambda value: value.get("saved_at", "")):
        if report.get("source_scope") != "Human dispositions entered through the local review screen":
            raise ValueError("Existing reference decisions have no human provenance")
        for sample_id, answer in report.get("decisions", {}).items():
            sha = report.get("queued_audio_sha256", {}).get(sample_id)
            if sha:
                answers[(sample_id, sha)] = answer.get("decision")
    return answers


def _check_wave(path):
    """Read the reference's native samples; never resample or modify it."""
    reasons, metrics = [], {}
    try:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as error:
        return ["unreadable_audio"], {"error": str(error)}
    frames, channels = samples.shape
    metrics.update(sample_rate=rate, channels=channels, frames=frames,
                   duration_s=frames / rate if rate else 0.)
    if channels != 1:
        reasons.append("nonmono_audio")
    if not _RULES["min_sample_rate"] <= rate <= _RULES["max_sample_rate"]:
        reasons.append("unsupported_sample_rate")
    if not _RULES["min_duration_s"] <= metrics["duration_s"] <= _RULES["max_duration_s"]:
        reasons.append("duration_outlier")
    if not frames or not np.isfinite(samples).all():
        return reasons + ["empty_or_nonfinite_audio"], metrics
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    clipped = float(np.mean(np.abs(samples) >= _RULES["clipping_level"]))
    metrics.update(peak=peak, rms=rms, clipping_fraction=clipped)
    if rms < _RULES["min_rms"]:
        reasons.append("near_silence")
    if peak > _RULES["max_abs_sample"] or clipped > _RULES["max_clipping_fraction"]:
        reasons.append("severe_clipping")
    canonical = np.asarray(samples, dtype="<f4")
    canonical[canonical == 0] = 0  # normalize negative zero without touching WAV
    metrics["decoded_pcm_sha256"] = hashlib.sha256(
        f"{rate}:{frames}:{channels}:".encode() + canonical.tobytes()).hexdigest()
    return reasons, metrics


def admit_references(project, counts=None, queues=None):
    """Check the selected banks and admit only a completely passing selection.

    Counts choose prefixes of the saved bank, not arbitrary accepted rows.
    Existing waveform/journal/contract/voice files and human answers are read
    only. Automatic evidence is separate from both listening and the new-word
    pronunciation policy, which remains unapproved until its pilot is heard.
    """
    root, config = load_project(project)
    counts = counts or {split: len(read_rows(root / f"data/references/{split}/references.jsonl"))
                        for split in ("train", "validation")}
    if (not isinstance(counts, dict) or not counts or set(counts) - {"train", "validation"}
            or any(type(count) is not int or count < 1 for count in counts.values())):
        raise ValueError("Automatic reference checks need positive Train/common counts; Final remains locked")
    bank_path = root / "voices.json"
    bank = read_json(bank_path)
    expected_text = config["tts"].get("reference_text")
    if expected_text is None:
        from .tts import REFERENCE_TEXT
        expected_text = REFERENCE_TEXT
    selected, journals = [], []
    seen = set()
    for split, count in counts.items():
        families = [family for family in bank["families"] if family["split"] == split][:count]
        journal = root / f"data/references/{split}/references.jsonl"
        rows = read_rows(journal)
        by_id = {row["voice_family_id"]: row for row in rows}
        if len(by_id) != len(rows) or len(families) != count:
            raise ValueError("Reference bank/journal is incomplete or contains duplicate identities")
        journals.append({"path":str(journal.relative_to(root)), "sha256":digest(journal)})
        for family in families:
            family_id = family["family_id"]
            if family_id in seen or family_id not in by_id:
                raise ValueError("Selected reference family is missing or reused across splits")
            seen.add(family_id)
            selected.append((family, by_id[family_id]))
    if queues is not None:
        queued = [row for queue in queues for row in read_rows(queue)]
        if (len(queued) != len(selected) or
                {row["sample_id"]:row for row in queued} != {row["sample_id"]:row for _,row in selected}):
            raise ValueError("Reference playback queues differ from the selected generated voices")
    folder = root / "review/reference_qc"
    folder.mkdir(parents=True, exist_ok=True)
    with writer_lock(folder):
        approval = read_json(root / "approval.json")
        answers = _human_answers(root, approval)
        checks, pcm = [], {}
        for family, row in selected:
            reasons = []
            if (row.get("provenance") != "qwen_synthetic_reference" or row.get("human_reference_used") is not False
                    or row.get("split") != family["split"] or row.get("sample_id") != family["family_id"]
                    or row.get("model") != config["tts"]["design_model"]):
                reasons.append("invalid_synthetic_reference_provenance")
            if row.get("text") != expected_text or row.get("description") != family.get("description"):
                reasons.append("reference_text_or_design_changed")
            path = resolved(root, row["path"])
            actual_sha = digest(path) if path.is_file() else None
            metrics = {}
            if actual_sha != row.get("sha256"):
                reasons.append("audio_missing_or_changed")
            else:
                audio_reasons, metrics = _check_wave(path)
                reasons.extend(audio_reasons)
                if digest(path) != actual_sha:
                    reasons.append("audio_changed_during_check")
            if answers.get((row["sample_id"], row.get("sha256"))) in ("discard", "review"):
                reasons.append("human_discard_or_review_pending")
            check = {"voice_family_id":family["family_id"], "sample_id":row["sample_id"],
                "split":family["split"], "path":row["path"], "sha256":row.get("sha256"),
                "actual_sha256":actual_sha, "reasons":reasons, "metrics":metrics}
            checks.append(check)
            key = metrics.get("decoded_pcm_sha256")
            if key:
                pcm.setdefault(key, []).append(check)
        for duplicates in pcm.values():
            if len(duplicates) > 1:
                for check in duplicates:
                    check["reasons"].append("exact_decoded_audio_duplicate")
        for check in checks:
            check["passed"] = not check["reasons"]
        fingerprint = identity({"policy":REFERENCE_QC_POLICY, "rules":_RULES,
            "tts_settings":config["tts"], "bank_sha256":digest(bank_path), "generation_journals":journals, "checks":checks})
        journal_path = folder / (fingerprint + ".jsonl")
        write_immutable_rows(journal_path, checks)
        receipt_path = folder / (fingerprint + ".json")
        passed = all(check["passed"] for check in checks)
        receipt = {"schema_version":1, "kind":"reference_technical_admission",
                "policy_id":REFERENCE_QC_POLICY, "state":"accepted" if passed else "blocked",
                "wakeword":config["wakeword"], "rules":_RULES, "selected_counts":counts,
                "voice_bank":{"path":"voices.json", "sha256":digest(bank_path)}, "generation_journals":journals,
                "quality_journal":{"path":str(journal_path.relative_to(root)), "sha256":digest(journal_path)},
                "checked_audio_sha256":{check["voice_family_id"]:check["sha256"] for check in checks},
                "accepted_reference_families":[check["voice_family_id"] for check in checks] if passed else [],
                "human_listened_count":0, "independent_STT_runs":0, "acoustic_diversity_measured":False,
                "scope":"Automatic provenance/signal-quality admission only; pronunciation pilot is still required"}
        if receipt_path.exists():
            existing = read_json(receipt_path)
            existing.pop("created_at", None)
            if existing != receipt:
                raise ValueError("Immutable reference technical receipt changed")
        else:
            write_json(receipt_path, dict(receipt, created_at=now()))
        binding = {"path":str(receipt_path.relative_to(root)), "sha256":digest(receipt_path)}
        if passed:
            approval["accepted_reference_families"] = sorted(set(approval.get("accepted_reference_families", [])) | seen)
            approval["reference_technical_admission"] = binding
            if approval != read_json(root / "approval.json"):
                write_json(root / "approval.json", approval)
        return {"state":"accepted" if passed else "blocked", "checked":len(checks),
                "passed":sum(check["passed"] for check in checks), "failed":[check for check in checks if not check["passed"]],
                "receipt":binding, "quality_journal":{"path":str(journal_path),"sha256":digest(journal_path)},
                "human_listened_count":0, "new_word_pilot_still_required":True}


def admitted_reference_rows(root, approval):
    """Return only voices proved by a bound, passing automatic QC journal."""
    binding = approval.get("reference_technical_admission")
    if not binding:
        return {}
    receipt = read_json(bound_file(root, binding))
    checks = read_rows(bound_file(root, receipt["quality_journal"]))
    admitted = {row["voice_family_id"]:row["sha256"] for row in checks}
    if (receipt.get("kind") != "reference_technical_admission" or receipt.get("policy_id") != REFERENCE_QC_POLICY
            or receipt.get("state") != "accepted" or not checks or len(admitted) != len(checks)
            or any(row.get("passed") is not True or row.get("reasons") for row in checks)
            or admitted != receipt.get("checked_audio_sha256")
            or set(admitted) != set(receipt.get("accepted_reference_families", []))):
        raise ValueError("Source bank has invalid automatic reference evidence")
    return admitted
