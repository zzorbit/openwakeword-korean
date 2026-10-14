"""Discarded call-name sources can be excluded without rewriting user evidence."""
import pytest

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.data import validate_admission
from korean_openwakeword.pool_review import approve_pool
from korean_openwakeword.project import load_project
from test_korean_library_pool_review import (
    pool, synthetic_pool, _bound, _refresh, _refresh_contract,
)


def _legacy_discarded_call_name(synthetic_pool):
    old = synthetic_pool["contract"]["texts"][0]
    synthetic_pool["contract"]["texts"][0] = "보라"
    for row in synthetic_pool["rows"]:
        if row.get("text") == old:
            row["text"] = "보라"
    _refresh_contract(synthetic_pool)
    excluded_sha = {row["sha256"] for row in synthetic_pool["rows"] if row.get("text") == "보라"}
    answers = read_json(synthetic_pool["answers"])
    for preview in synthetic_pool["representatives"]:
        if preview["reviewed_source"]["sha256"] in excluded_sha:
            answers["decisions"][preview["sample_id"]] = {"decision": "discard", "listened": True}
    write_json(synthetic_pool["answers"], answers)
    return excluded_sha


def _approve_excluding(fixture, excluded=("보라",)):
    return approve_pool(
        fixture["project"], source_manifest=_bound(fixture["manifest"]),
        representative_queue=_bound(fixture["queue"]), answers=_bound(fixture["answers"]),
        output=fixture["output"], excluded_hard_phrases=excluded,
    )


def test_exclude_every_call_name_original_preserving_full_user_discard_evidence(synthetic_pool):
    excluded_sha = _legacy_discarded_call_name(synthetic_pool)
    frozen = {name: digest(synthetic_pool[name]) for name in ("manifest", "queue", "answers", "contract_path")}
    result = _approve_excluding(synthetic_pool)
    record = read_json(synthetic_pool["output"])
    assert result["source_count"] == 12 and result["representative_preview_count"] == 6
    assert record["full_frozen_source_count"] == 16 and record["full_frozen_preview_count"] == 8
    assert record["excluded_hard_phrases"] == ["보라"]
    assert record["excluded_source_count"] == 4
    assert set(record["excluded_source_sha256"]) == excluded_sha
    assert not (set(record["accepted_source_sha256"]) & excluded_sha)
    assert len(record["excluded_listened_discard_previews"]) == 2
    assert all(row["saved_human_disposition"] == {"decision": "discard", "listened": True}
               for row in record["excluded_listened_discard_previews"])
    assert all(row["training_admitted"] is False for row in record["excluded_sources"])
    assert {row["text"] for row in record["synthetic_phrase_family_strata"]} == {"보리야"}
    assert {name: digest(synthetic_pool[name]) for name in frozen} == frozen
    root, config = load_project(synthetic_pool["project"])
    kept_hard = [{**row, "admission": result["admission"]} for row in synthetic_pool["rows"]
                 if row["provenance"] == "qwen_tts" and row["sha256"] not in excluded_sha]
    validate_admission(root, kept_hard, config, "hard_negative")


@pytest.mark.parametrize("decision,listened", [("keep", True), ("review", True), ("discard", False)])
def test_explicit_exclusion_cannot_override_user_keep_review_or_unheard(synthetic_pool, decision, listened):
    excluded_sha = _legacy_discarded_call_name(synthetic_pool)
    answers = read_json(synthetic_pool["answers"])
    preview = next(row for row in synthetic_pool["representatives"]
                   if row["reviewed_source"]["sha256"] in excluded_sha)
    answers["decisions"][preview["sample_id"]] = {"decision": decision, "listened": listened}
    write_json(synthetic_pool["answers"], answers)
    with pytest.raises(ValueError, match="actual listened/discard"):
        _approve_excluding(synthetic_pool)
    assert not synthetic_pool["output"].exists()


@pytest.mark.parametrize("decision", ["discard", "review"])
def test_other_representative_dispositions_still_block_selected_pool(synthetic_pool, decision):
    _legacy_discarded_call_name(synthetic_pool)
    answers = read_json(synthetic_pool["answers"])
    public_preview = synthetic_pool["representatives"][0]
    answers["decisions"][public_preview["sample_id"]] = {"decision": decision, "listened": True}
    write_json(synthetic_pool["answers"], answers)
    with pytest.raises(ValueError, match="resolution"):
        _approve_excluding(synthetic_pool)
    assert not synthetic_pool["output"].exists()


def test_excluded_phrase_cannot_be_invented_without_a_source(synthetic_pool):
    _legacy_discarded_call_name(synthetic_pool)
    with pytest.raises(ValueError, match="exist in the frozen"):
        _approve_excluding(synthetic_pool, ("보라", "없는단어"))


def test_excluded_phrase_requires_saved_discard_representative(synthetic_pool):
    _legacy_discarded_call_name(synthetic_pool)
    excluded_sha = {row["sha256"] for row in synthetic_pool["rows"] if row.get("text") == "보라"}
    synthetic_pool["representatives"] = [row for row in synthetic_pool["representatives"]
                                         if row["reviewed_source"]["sha256"] not in excluded_sha]
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="saved listened/discard representative"):
        _approve_excluding(synthetic_pool)


def test_public_source_cannot_be_filtered_by_a_hard_phrase_parameter(synthetic_pool):
    _legacy_discarded_call_name(synthetic_pool)
    synthetic_pool["rows"][0]["text"] = "보라"
    _refresh(synthetic_pool)
    with pytest.raises(ValueError, match="only narrow Qwen"):
        _approve_excluding(synthetic_pool)


@pytest.mark.parametrize("excluded", ["보라", [""], [" 보라"], [True], ["보라", "보라"]])
def test_exclusion_argument_is_exact_and_explicit(synthetic_pool, excluded):
    _legacy_discarded_call_name(synthetic_pool)
    with pytest.raises(ValueError, match="excluded_hard_phrases"):
        _approve_excluding(synthetic_pool, excluded)


def test_existing_noncall_guard_still_applies_to_selected_rows(synthetic_pool):
    _legacy_discarded_call_name(synthetic_pool)
    with pytest.raises(ValueError, match="complete call-name"):
        _approve_excluding(synthetic_pool, ())


@pytest.mark.parametrize("declaration", [[None, "보리야"], ["보라", ""], [7, "보리야"]])
def test_unused_legacy_contract_phrase_declarations_still_require_strings(synthetic_pool, declaration):
    _legacy_discarded_call_name(synthetic_pool)
    synthetic_pool["contract"]["texts"] = declaration
    _refresh_contract(synthetic_pool)
    with pytest.raises(ValueError, match="nonempty strings"):
        _approve_excluding(synthetic_pool)
