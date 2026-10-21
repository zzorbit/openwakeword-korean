"""Reject invalid extraction budgets before admission or frontend work."""
import pytest

from korean_openwakeword import features
from korean_openwakeword.project import create_project


@pytest.mark.parametrize("rounds", [0, -1, True, False, 1.0, "3", 2])
def test_invalid_rounds_are_not_replaced_by_defaults(tmp_path, monkeypatch, rounds):
    project = create_project(tmp_path / "word", "보라야", "bora")
    monkeypatch.setattr(features, "_manifest_inventory", lambda *args: pytest.fail("Validate before reading sources"))
    with pytest.raises(ValueError, match="Positive R1/R3/R5"):
        features.build_features(project, "positive", rounds=rounds, smoke=True)


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.0, "32"])
def test_invalid_batch_size_is_rejected_before_source_work(tmp_path, monkeypatch, batch_size):
    project = create_project(tmp_path / "word", "보라야", "bora")
    monkeypatch.setattr(features, "_manifest_inventory", lambda *args: pytest.fail("Validate before reading sources"))
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        features.build_features(project, "positive", batch_size=batch_size, smoke=True)
