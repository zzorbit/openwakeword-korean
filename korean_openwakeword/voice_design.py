"""Deterministic, balanced synthetic voice requests, never certified speakers.

Every prefix balances each requested attribute. Pair counts break ties so that
gender, age and delivery do not collapse into a fixed set of combinations.
"""
from collections import Counter
from functools import lru_cache
import random
import re


BALANCED_VOICE_POLICY = "balanced_v1"
VOICE_AXES = {
    "gender": ("man", "woman"),
    "age_group": ("child", "teenager", "young adult", "middle-aged", "older adult"),
    "timbre": ("clear light", "warm rounded", "low resonant", "bright nasal",
               "slightly husky", "soft airy", "crisp focused", "deep mellow",
               "smooth balanced", "rich textured", "gentle thin", "firm dry",
               "light breathy", "dense midrange", "clear ringing"),
    "style": ("neutral", "friendly", "urgent", "projected"),
    "vocal_effort": ("soft", "normal", "strong"),
    "speed": ("slow", "natural", "quick"),
    "intonation": ("level", "rising", "falling"),
    "call_length": ("short", "natural", "drawn_out"),
}
_AGE_DESCRIPTIONS = {
    "child": "child aged 8 to 12",
    "teenager": "teenager aged 13 to 19",
    "young adult": "young adult aged 20 to 39",
    "middle-aged": "middle-aged adult aged 40 to 59",
    "older adult": "older adult aged 60 to 79",
}
_RESERVED_COUNTS = (("train", 500), ("validation", 100), ("final", 10))


@lru_cache(maxsize=1)
def _profiles():
    """Freeze full banks before slicing; Train size cannot move the holdout."""
    axes = tuple(VOICE_AXES)
    seen = set()
    banks = []
    for split_index, (split, total) in enumerate(_RESERVED_COUNTS):
        rng = random.Random(73129 + split_index * 104729)
        counts = {axis: Counter() for axis in axes}
        pairs = Counter()
        profiles = []
        for _ in range(total):
            # Retry only a whole-profile collision. Minimum-count choices keep
            # every marginal balanced, including a 20-voice smoke-test prefix.
            for attempt in range(128):
                selected = {}
                for axis in axes:
                    minimum = min(counts[axis][value] for value in VOICE_AXES[axis])
                    choices = [value for value in VOICE_AXES[axis]
                               if counts[axis][value] == minimum]
                    rng.shuffle(choices)
                    selected[axis] = min(choices, key=lambda value: (
                        pairs[("gender", selected["gender"], axis, value)] if "gender" in selected else 0,
                        sum(pairs[(prior, prior_value, axis, value)]
                            for prior, prior_value in selected.items())))
                # Reference generation applies seven axes; call_length is only
                # planning metadata and cannot make identical prompts distinct.
                key = tuple(selected[axis] for axis in axes if axis != "call_length")
                if key not in seen:
                    break
            else:
                raise RuntimeError("Unable to construct a unique balanced voice profile")
            seen.add(key)
            profiles.append(tuple(selected[axis] for axis in axes))
            for axis, value in selected.items():
                counts[axis][value] += 1
            for position, axis in enumerate(axes):
                for other in axes[position + 1:]:
                    pairs[(axis, selected[axis], other, selected[other])] += 1
        banks.append((split, tuple(profiles)))
    return tuple(banks)


def balanced_voice_candidates(validation_count=30, *, train_count=500):
    """Return fresh request records, with stable balanced prefixes and IDs."""
    if type(train_count) is not int or not 1 <= train_count <= 500:
        raise ValueError("Training synthetic voice families must be an integer from 1 to 500")
    if type(validation_count) is not int or not 1 <= validation_count <= 100:
        raise ValueError("Validation synthetic voice families must be an integer from 1 to 100")
    limits = {"train": train_count, "validation": validation_count, "final": 10}
    axes = tuple(VOICE_AXES)
    families = []
    for split, profiles in _profiles():
        for number, profile in enumerate(profiles[:limits[split]]):
            attributes = dict(zip(axes, profile))
            gender_description = "male" if attributes["gender"] == "man" else "female"
            description = (f"A {gender_description} native Korean speaker, a "
                           f"{_AGE_DESCRIPTIONS[attributes['age_group']]}, "
                           f"with a {attributes['timbre']} timbre. "
                           "Natural dry speech.")
            families.append({"family_id": f"{split}_voice_v2_{number:03d}", "split": split,
                "description": description, **attributes,
                "provenance": "qwen_synthetic", "status": "candidate",
                "voice_policy": BALANCED_VOICE_POLICY})
    return {"schema_version": 1, "policy": BALANCED_VOICE_POLICY, "families": families,
        "note": "Balanced requested attributes, not measured acoustic diversity or certified distinct speakers."}


def summarize_voice_design(families):
    """Summarize requested attributes, without treating them as measurements."""
    rows = list(families)
    fields = {"gender": "gender", "age": "age_group", "timbre": "timbre", "style": "style",
              "vocal_effort": "vocal_effort", "speed": "speed", "intonation": "intonation"}
    normalized = []
    for row in rows:
        attributes = dict(row)
        description = row.get("description", "")
        if "gender" not in attributes:
            found = re.search(r"\b(man|woman|male|female)\b", description)
            gender = found.group(1) if found else "unspecified"
            attributes["gender"] = {"male": "man", "female": "woman"}.get(gender, gender)
        if "age_group" not in attributes:
            found = re.search(r"\b(child|teenager|young adult|middle-aged|older adult|adult)\b", description)
            attributes["age_group"] = found.group(1) if found else "unspecified"
        if "timbre" not in attributes:
            found = re.search(r"with a (.*?) timbre\b", description)
            attributes["timbre"] = found.group(1) if found else "unspecified"
        normalized.append(attributes)
    return {"count": len(rows), **{name: dict(Counter(row.get(field) or "unspecified" for row in normalized))
                                   for name, field in fields.items()}}
