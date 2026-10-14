"""Hard Negative text admission must never label a direct call name negative."""
import json

import pytest

from korean_openwakeword.common import read_rows
from korean_openwakeword.data import (
    audit_project, prepare_windows, validate_admission, validate_hard_negative,
)
from korean_openwakeword.features import build_features
from korean_openwakeword.hard_phrase_policy import protected_call_terms
from korean_openwakeword.project import load_project
from test_korean_library import project


def public_row(text):
    return {"text": text, "speech": True, "provenance": "public_dataset",
            "source_url": "https://example.org/noncall-fixture", "license": "test-only"}


@pytest.mark.parametrize("text", [
    "보라", "보라가", "보라를", "보라는", "보라야", "보라야불켜줘",
    "보라야 불켜줘", "보 라 를", "예쁜보라", "보라색",
])
@pytest.mark.parametrize("synthetic", [False, True])
def test_full_call_name_is_rejected_in_both_public_and_synthetic_rows(text, synthetic):
    row = public_row(text)
    if synthetic:
        row.update(provenance="qwen_tts", human_reference_used=False, voice_family_id="fixture_train")
    with pytest.raises(ValueError, match="complete call-name"):
        validate_hard_negative(row, {"wakeword": "보라야"})


@pytest.mark.parametrize("wakeword,text", [
    ("보라야", "보리야"), ("보라야", "보람아"), ("보라야", "돌아와"),
    ("춘식아", "춘천"), ("도아", "도우미"), ("도아", "도착"),
])
def test_distinct_words_with_shared_syllables_remain_valid(wakeword, text):
    validate_hard_negative(public_row(text), {"wakeword": wakeword})


def test_vocative_policy_is_explicit_and_does_not_ban_every_shared_character():
    assert protected_call_terms("보라야") == ["보라야", "보라"]
    assert protected_call_terms("춘식아") == ["춘식아", "춘식"]
    assert protected_call_terms("도아") == ["도아"]
    with pytest.raises(ValueError, match="complete call-name"):
        validate_hard_negative(public_row("춘식이를불러"), {"wakeword": "춘식아"})


def test_existing_speech_and_provenance_requirements_are_retained():
    config = {"wakeword": "보라야"}
    with pytest.raises(ValueError, match="speech"):
        validate_hard_negative({**public_row("보람아"), "speech": False}, config)
    with pytest.raises(ValueError, match="public source"):
        validate_hard_negative({**public_row("보람아"), "source_url": ""}, config)
    with pytest.raises(ValueError, match="Qwen-only"):
        validate_hard_negative({**public_row("보람아"), "provenance": "qwen_tts",
                                "human_reference_used": True, "voice_family_id": "fixture"}, config)


class UnusedFrontend:
    identity = {"fixture": "never invoked for prohibited non-call text"}

    def embed(self, clips):
        pytest.fail("Rejected phrase reached feature computation")


def test_audit_windows_and_existing_feature_cache_cannot_bypass_guard(project):
    root, config = load_project(project)
    manifest = root / config["manifests"]["hard_negative"]
    rows = read_rows(manifest)
    rows[0]["text"] = "보라를"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    audit = audit_project(project, check_files=False)
    assert not audit["passed"]
    assert any("complete call-name" in e for e in audit["errors"])
    with pytest.raises(ValueError, match="complete call-name"):
        validate_admission(root, rows, config, "hard_negative", smoke=True)
    with pytest.raises(ValueError, match="complete call-name"):
        prepare_windows(project, "hard_negative", root / "data/negative/forbidden_windows.jsonl")
    with pytest.raises(ValueError, match="complete call-name"):
        build_features(project, "hard_negative", smoke=True, resume=True, frontend=UnusedFrontend())
