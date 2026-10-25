"""Hard CLI budgets distinguish per-phrase generation from legacy totals."""
import json

import pytest

from korean_openwakeword import cli, hard_tts, tts_runtime
from korean_openwakeword.common import read_json, write_json
from korean_openwakeword.project import create_project


@pytest.fixture
def hard_cli(tmp_path, monkeypatch):
    project = create_project(tmp_path / "bora", "보라야", "bora", positive_originals=1000,
                             train_voice_families=20)
    reference_path = project.parent / "data/references/train/references.jsonl"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    # The configured design bank has 20 voices, but only these two Train
    # references are accepted; a third generated reference is still pending.
    references = [
        {"voice_family_id": "by_saved_approval", "status": "pending_review"},
        {"voice_family_id": "legacy_accepted", "status": "accepted"},
        {"voice_family_id": "unapproved", "status": "pending_review"},
    ]
    reference_path.write_text("".join(json.dumps(row) + "\n" for row in references), encoding="utf-8")
    approval_path = project.parent / "approval.json"
    approval = read_json(approval_path)
    approval["accepted_reference_families"] = ["by_saved_approval"]
    write_json(approval_path, approval)
    runtime = object()
    runtime_calls, generation_calls = [], []

    def create_runtime(*args, **kwargs):
        runtime_calls.append((args, kwargs))
        return runtime

    def generate(*args, **kwargs):
        generation_calls.append((args, kwargs))
        return {"generation_dispatched": True}

    monkeypatch.setattr(tts_runtime, "CloneRuntime", create_runtime)
    monkeypatch.setattr(hard_tts, "generate_hard", generate)
    return project, runtime, runtime_calls, generation_calls


def _args(project, *budget):
    return cli.parser().parse_args([
        "generate-hard", "--project", str(project), "--phrases", "보리야", "돌아와", *budget,
    ])


def test_explicit_per_phrase_cli_budget_is_forwarded_as_keyword(hard_cli):
    project, runtime, runtime_calls, generation_calls = hard_cli
    result = cli.dispatch(_args(project, "--samples-per-phrase", "20"))
    assert result == {"generation_dispatched": True}
    assert len(runtime_calls) == 1
    assert generation_calls == [((project, ["보리야", "돌아와"]),
                                 {"model": runtime, "samples_per_phrase": 20})]


def test_explicit_legacy_count_keeps_total_budget_semantics(hard_cli):
    project, runtime, _, generation_calls = hard_cli
    cli.dispatch(_args(project, "--count", "100"))
    assert generation_calls == [((project, ["보리야", "돌아와"], 100), {"model": runtime})]


def test_cli_parser_does_not_allow_both_budget_modes(hard_cli):
    project, _, runtime_calls, generation_calls = hard_cli
    with pytest.raises(SystemExit) as error:
        _args(project, "--count", "100", "--samples-per-phrase", "20")
    assert error.value.code == 2
    assert not runtime_calls and not generation_calls


def test_omitted_budget_uses_actual_accepted_train_bank_not_design_count(hard_cli):
    project, runtime, _, generation_calls = hard_cli
    assert sum(row["split"] == "train" for row in read_json(project.parent / "voices.json")["families"]) == 20
    cli.dispatch(_args(project))
    assert generation_calls == [((project, ["보리야", "돌아와"]),
                                 {"model": runtime, "samples_per_phrase": 2})]
