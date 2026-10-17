"""Measured common-benchmark coverage, separate from model scores and Final."""
from collections import Counter
from pathlib import Path
import math

from .common import digest, identity, iter_rows, read_json


def assess_common_evaluation(policy, report, manifest, noise_coverage):
    """Assess declared coverage targets; never interpret missing evidence as passed.

    Targets are a documented benchmark contract, not a universal sufficient-data
    theorem or a guarantee of real-speaker or deployment performance.
    """
    coverage = read_json(noise_coverage) if isinstance(noise_coverage, (str, Path)) else noise_coverage
    originals, families, conditions = set(), set(), Counter()
    measured = {key: {"sources": set(), "types": set(), "music_sources": set()} for key in
                ("environment", "music", "competing_speech")}
    for row in iter_rows(manifest):
        if row.get("kind") != "positive":
            continue
        originals.add(row.get("original_id", row["sample_id"]))
        if row.get("voice_family_id"):
            families.add(row["voice_family_id"])
        category, snr = row.get("category"), row.get("snr_db")
        conditions[category, snr] += 1
        if category in measured:
            group = row.get("noise_source_group_id", row.get("noise_source_id"))
            subtype = row.get("noise_eval_type")
            if group:
                measured[category]["sources"].add(group)
            if subtype:
                measured[category]["types"].add(subtype)
                if category == "music" and subtype.startswith("music_") and group:
                    measured[category]["music_sources"].add(group)
    targets = {"environment": {"types": 30, "sources": 100},
               "music": {"types": 6, "sources": 12, "music_sources": 12},
               "competing_speech": {"types": 6, "sources": 12}}
    shortages = []
    if not coverage:
        shortages.append("배경 원본·종류별 사용 분포 근거 없음")
    count = policy["validation_originals"]
    if len(originals) < count:
        shortages.append(f"호출어 원본 {len(originals):,}/{count:,}개")
    family_target = policy["validation_references"]
    if len(families) < family_target:
        shortages.append(f"평가 음색 계열 {len(families)}/{family_target}개")
    expected = [("clean", None)] + [(category, snr) for category in measured for snr in policy["snrs"]]
    for condition in expected:
        if conditions[condition] < count * (1 if condition[0] == "clean" else policy["noise_sections"]):
            shortages.append(f"조건별 표본 부족: {condition[0]} / {condition[1]} dB")
    categories = {}
    category_names = {"environment": "환경음", "music": "음악·전자음", "competing_speech": "한국어 간섭 음성"}
    measure_names = {"types": "종류", "sources": "원본", "music_sources": "실제 음악 원본"}
    for category, values in measured.items():
        actual = {key: len(value) for key, value in values.items()}
        categories[category] = {"used": actual, "minimum": targets[category]}
        for key, minimum in targets[category].items():
            if actual[key] < minimum:
                shortages.append(f"{category_names[category]} {measure_names[key]}: {actual[key]}/{minimum}")
    hours = [m.get("negative", {}).get("hours") for m in report.get("models", [])]
    finite_hours = [h for h in hours if type(h) in (int, float) and math.isfinite(h) and h >= 0]
    minimum_hours = min(finite_hours) if finite_hours and len(finite_hours) == len(hours) else None
    target_hours = policy["development_negative_target_hours"][0]
    if minimum_hours is None or minimum_hours < target_hours:
        shortages.append(f"연속 비호출 평가 {minimum_hours if minimum_hours is not None else '미측정'}/{target_hours}시간")
    return {"kind": "common_evaluation_coverage_v1", "policy": policy, "policy_sha256": identity(policy),
        "manifest_sha256": digest(manifest), "positive_originals": len(originals), "synthetic_voice_families": len(families),
        "condition_counts": [{"category": k[0], "snr_db": k[1], "samples": v} for k, v in sorted(conditions.items(), key=str)],
        "categories": categories, "negative_hours_per_candidate_minimum": minimum_hours,
        "background_coverage_available": bool(coverage), "coverage_targets_passed": not shortages,
        "shortages": shortages, "independent_Final_completed": False,
        "scope": "Declared common-benchmark coverage only; model success/FA targets are checked separately. No population or independent Final claim."}
