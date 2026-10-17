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
    reverb_conditions, reverb_originals, reverb_sources, reverb_rooms = Counter(), {}, {}, {}
    measured = {key: {"sources": set(), "types": set(), "music_sources": set()} for key in
                ("environment", "music", "competing_speech")}
    for row in iter_rows(manifest):
        if row.get("kind") != "positive":
            continue
        originals.add(row.get("original_id", row["sample_id"]))
        if row.get("voice_family_id"):
            families.add(row["voice_family_id"])
        category, snr = row.get("category"), row.get("snr_db")
        group = row.get("reverb_condition", "none")
        reverb_conditions[group, category, snr] += 1
        if group == "none":
            conditions[category, snr] += 1
        else:
            reverb_originals.setdefault(group, set()).add(row.get("original_id", row["sample_id"]))
            if row.get("rir_sha256"):
                reverb_sources.setdefault(group, set()).add(row["rir_sha256"])
            if row.get("canonical_room_id"):
                reverb_rooms.setdefault(group, set()).add(row["canonical_room_id"])
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
    reverb_target = policy.get("reverb_originals", 0)
    if reverb_target:
        for group in policy.get("reverb_groups", ("moderate", "strong")):
            if len(reverb_originals.get(group, set())) < reverb_target:
                shortages.append(f"잔향 {group} 호출어 원본 {len(reverb_originals.get(group, set()))}/{reverb_target}개")
            for category, snr in expected:
                required = reverb_target * (1 if category == "clean" else policy["noise_sections"])
                if reverb_conditions[group, category, snr] < required:
                    shortages.append(f"잔향 조건별 표본 부족: {group} / {category} / {snr} dB")
        actual_rirs = len(set().union(*reverb_sources.values())) if reverb_sources else 0
        actual_rooms = len(set().union(*reverb_rooms.values())) if reverb_rooms else 0
        for label, actual, minimum in (("잔향 원본", actual_rirs, policy.get("rir_files", policy.get("rir_min_sources", 50))),
                                       ("잔향 공간", actual_rooms, policy.get("rir_rooms", policy.get("rir_min_rooms", 20)))):
            if actual < minimum:
                shortages.append(f"{label} {actual}/{minimum}개")
        if policy.get("id") == "common_development_v2":
            for group in policy.get("reverb_groups", ("moderate", "strong")):
                minimum = policy.get("rir_rooms", 20) // 2
                if len(reverb_rooms.get(group, set())) < minimum:
                    shortages.append(f"잔향 {group} 공간 {len(reverb_rooms.get(group, set()))}/{minimum}개")
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
    negative_materials = None
    if policy.get("id") == "common_development_v2":
        from .evaluation_materials import evaluation_material_coverage
        negative_materials = evaluation_material_coverage(row for row in iter_rows(manifest) if row.get("kind") == "negative")
        # The material helper owns interval union/source deduplication; neither
        # per-model replay nor looping a short recording creates new coverage.
        if not negative_materials.get("targets_met", negative_materials.get("target_met", False)):
            shortages.extend(negative_materials.get("shortages", ["공통 비호출 자료 종류별 목표 시간 부족"]))
    return {"kind": "common_evaluation_coverage_v1", "policy": policy, "policy_sha256": identity(policy),
        "manifest_sha256": digest(manifest), "positive_originals": len(originals), "synthetic_voice_families": len(families),
        "condition_counts": [{"category": k[0], "snr_db": k[1], "samples": v} for k, v in sorted(conditions.items(), key=str)],
        "reverberation": {"originals_per_group": {g: len(v) for g, v in reverb_originals.items()},
            "sources_per_group": {g: len(v) for g, v in reverb_sources.items()},
            "rooms_per_group": {g: len(v) for g, v in reverb_rooms.items()},
            "condition_counts": [{"reverb_condition": k[0], "category": k[1], "snr_db": k[2], "samples": v}
                                  for k, v in sorted(reverb_conditions.items(), key=str)],
            "scope": "Declared room identities can include provider simulated rooms; not a count of measured real rooms"},
        "categories": categories, "negative_hours_per_candidate_minimum": minimum_hours,
        "negative_materials": negative_materials,
        "background_coverage_available": bool(coverage), "coverage_targets_passed": not shortages,
        "shortages": shortages, "independent_Final_completed": False,
        "scope": "Declared common-benchmark coverage only; model success/FA targets are checked separately. No population or independent Final claim."}
