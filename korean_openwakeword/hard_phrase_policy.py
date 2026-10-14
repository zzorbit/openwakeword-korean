"""Shared text guard for generated, imported and resumed Hard Negatives."""
import re
import unicodedata


def protected_call_terms(wakeword):
    """Protect the complete call and its multi-syllable vocative base.

    A Korean name followed by 아/야 also occurs without that ending or with
    a particle (보라, 보라를). Labeling those direct partial calls negative
    would teach rejection of the same name. This is a conservative text
    guard, not phonetic similarity matching or Korean morphology analysis.
    Do not ban a shared single syllable, e.g. 보 in 보리야 or 도 in 도우미.
    """
    terms = [wakeword]
    if wakeword.endswith(("아", "야")) and len(wakeword[:-1]) >= 2:
        terms.append(wakeword[:-1])
    return terms


def validate_noncall_text(text, wakeword):
    """Reject the same call name even when spacing/punctuation was added."""
    normalized = "".join(re.findall(r"[가-힣]", unicodedata.normalize("NFC", text)))
    if not normalized:
        raise ValueError("Hard Negative phrase must contain Korean Hangul text")
    for term in protected_call_terms(wakeword):
        if term in normalized:
            raise ValueError(
                f"Hard Negative phrase '{text}' contains the wakeword or its complete "
                f"call-name '{term}' (호출어·동일 낱말·부분 호출 금지); "
                "choose a different non-call phrase"
            )

