"""Voice requests balance small and large prefixes without changing old banks."""
from collections import Counter
from itertools import product
import json

import pytest

from korean_openwakeword.common import read_json
from korean_openwakeword.project import create_project, voice_candidates
from korean_openwakeword.voice_design import (
    BALANCED_VOICE_POLICY, VOICE_AXES, balanced_voice_candidates, summarize_voice_design,
)


def profiles(bank, split, count=None):
    return [row for row in bank["families"] if row["split"] == split][:count]


@pytest.mark.parametrize("split,count", [("train", 20), ("train", 100), ("train", 500),
                                        ("validation", 30), ("final", 10)])
def test_every_selected_budget_balances_all_requested_attributes(split, count):
    rows = profiles(balanced_voice_candidates(), split, count)
    assert len(rows) == count
    for axis, values in VOICE_AXES.items():
        counts = Counter(row[axis] for row in rows)
        allocation = [counts[value] for value in values]
        assert max(allocation) - min(allocation) <= 1, (split, count, axis, allocation)
    # Neither sex is assigned only one age, speaking style or effort category.
    for axis in ("age_group", "style", "vocal_effort"):
        counts = Counter((row["gender"], row[axis]) for row in rows)
        combinations = list(product(VOICE_AXES["gender"], VOICE_AXES[axis]))
        assert all(counts[pair] >= 1 for pair in combinations)
        assert max(counts[pair] for pair in combinations) - min(counts[pair] for pair in combinations) <= 1


def test_all_prefixes_keep_marginals_balanced_not_just_full_bank():
    bank = balanced_voice_candidates(validation_count=100)
    for split in ("train", "validation", "final"):
        counts = {axis: Counter() for axis in VOICE_AXES}
        for row in profiles(bank, split):
            for axis, values in VOICE_AXES.items():
                counts[axis][row[axis]] += 1
                allocation = [counts[axis][value] for value in values]
                assert max(allocation) - min(allocation) <= 1


def test_resizing_training_never_reassigns_evaluation_or_existing_prefix():
    banks = [balanced_voice_candidates(train_count=count, validation_count=30) for count in (20, 100, 500)]
    assert profiles(banks[0], "train") == profiles(banks[2], "train", 20)
    assert profiles(banks[1], "train") == profiles(banks[2], "train", 100)
    assert profiles(banks[0], "validation") == profiles(banks[1], "validation") == profiles(banks[2], "validation")
    assert profiles(balanced_voice_candidates(validation_count=10), "final") == profiles(banks[2], "final")
    assert balanced_voice_candidates(train_count=500) == balanced_voice_candidates(train_count=500)


def test_split_ids_and_full_design_combinations_are_distinct():
    bank = balanced_voice_candidates(validation_count=100)
    rows = bank["families"]
    assert len(rows) == 610
    assert len({row["family_id"] for row in rows}) == 610
    assert len({tuple(row[axis] for axis in VOICE_AXES) for row in rows}) == 610
    applied_axes = [axis for axis in VOICE_AXES if axis != "call_length"]
    assert len({tuple(row[axis] for axis in applied_axes) for row in rows}) == 610
    legacy_ids = {row["family_id"] for row in voice_candidates(100, train_count=500)["families"]}
    assert not legacy_ids.intersection(row["family_id"] for row in rows)
    assert bank["policy"] == BALANCED_VOICE_POLICY
    assert all("pitch variation" not in row["description"] for row in rows)
    assert all(("male" if row["gender"] == "man" else "female") in row["description"]
               and row["timbre"] in row["description"] for row in rows)
    assert all(row["voice_policy"] == BALANCED_VOICE_POLICY for row in rows)


def test_new_projects_select_balanced_bank_without_rewriting_legacy_choice(tmp_path):
    project = create_project(tmp_path / "new", "보라야", "bora")
    bank = read_json(project.parent / "voices.json")
    assert bank == balanced_voice_candidates()
    before = (project.parent / "voices.json").read_bytes()
    with pytest.raises(ValueError, match="new empty"):
        create_project(project.parent, "보라야", "bora")
    assert (project.parent / "voices.json").read_bytes() == before
    legacy = create_project(tmp_path / "legacy", "보라야", "bora", voice_policy=None)
    assert read_json(legacy.parent / "voices.json") == voice_candidates(30, train_count=500)


def test_summary_reports_balanced_counts_and_does_not_invent_legacy_attributes():
    rows = profiles(balanced_voice_candidates(), "validation")
    summary = summarize_voice_design(rows)
    assert summary["count"] == 30
    assert summary["gender"] == {"woman": 15, "man": 15}
    assert summary["age"] == {"child": 6, "teenager": 6, "young adult": 6, "middle-aged": 6, "older adult": 6}
    assert summary["vocal_effort"] == {"strong": 10, "soft": 10, "normal": 10}
    legacy = voice_candidates(30)["families"]
    legacy_summary = summarize_voice_design([row for row in legacy if row["split"] == "train"][:20])
    assert legacy_summary["age"] == {"young adult": 20}
    assert legacy_summary["gender"] == {"man": 10, "woman": 10}
    assert legacy_summary["vocal_effort"] == {"soft": 20}
    unknown = summarize_voice_design([{"description": "A synthetic voice request."}])
    assert unknown["gender"] == unknown["age"] == unknown["timbre"] == {"unspecified": 1}


@pytest.mark.parametrize("choices", [{"train_count": True}, {"train_count": 501}, {"validation_count": 0},
                                     {"validation_count": 101}, {"train_count": 20.0}])
def test_unsupported_counts_fail_instead_of_silently_changing_voice_allocation(choices):
    with pytest.raises(ValueError):
        balanced_voice_candidates(**choices)
