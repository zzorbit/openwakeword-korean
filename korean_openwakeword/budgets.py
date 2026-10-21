"""New-project preparation budgets, separate from saved experiment settings.

These are user-selected starting budgets, not measured optimal data quantities.
Changing them must never migrate a previously saved project's targets.
"""

DEFAULT_GENERIC_HOURS = 3500
DEFAULT_KOREAN_HOURS = 1500
GENERIC_SOURCE_WEIGHTS = (
    ("audioset_unbalanced", 3),
    ("mls", 3),
    ("fma_full", 1),
)


def generic_source_hours(generic_hours):
    """Distribute a caller-validated hour budget without rounding source hours."""
    total_weight = sum(weight for _, weight in GENERIC_SOURCE_WEIGHTS)
    return {source: generic_hours * weight / total_weight
            for source, weight in GENERIC_SOURCE_WEIGHTS}


DEFAULT_FMA_HOURS = generic_source_hours(DEFAULT_GENERIC_HOURS)["fma_full"]
