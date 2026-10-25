"""CLI pilot defaults use the same voice coverage and runtime choices as Web."""
import pytest

from korean_openwakeword import cli, pilot, tts_runtime
from korean_openwakeword.project import create_project
from korean_openwakeword.common import digest


@pytest.mark.parametrize("extra,count,backend,threads", [
    ([], None, "auto", 4),
    (["--tts-backend", "eager", "--tts-cpu-threads", "8"], None, "eager", 8),
    (["--limit", "100"], 100, "auto", 4),
])
def test_cli_pilot_uses_one_per_voice_and_preserves_runtime_flags(tmp_path, monkeypatch, extra, count, backend, threads):
    project = create_project(tmp_path / "word", "보라야", "bora")
    before = digest(project.parent / "approval.json")
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"review_queue": "immutable-selected-queue", "human_listened": 0}
    monkeypatch.setattr(pilot, "prepare_pilot", prepare)
    monkeypatch.setattr(tts_runtime, "generate_with_runtime", lambda *args, **kwargs: pytest.fail("Fixed-count pilot route is not used"))
    args = cli.parser().parse_args(["generate", "--project", str(project), "--pilot", *extra])
    assert cli.dispatch(args)["review_queue"] == "immutable-selected-queue"
    assert calls == [((project,), {"train_count": count, "tts_backend": backend, "tts_cpu_threads": threads})]
    assert digest(project.parent / "approval.json") == before


def test_bulk_generation_keeps_its_existing_runtime_and_approval_gate(tmp_path, monkeypatch):
    project = create_project(tmp_path / "word", "보라야", "bora")
    calls = []
    monkeypatch.setattr(pilot, "prepare_pilot", lambda *args, **kwargs: pytest.fail("Bulk cannot become a pilot"))
    monkeypatch.setattr(tts_runtime, "generate_with_runtime", lambda *args: calls.append(args) or {"scope": "bulk"})
    args = cli.parser().parse_args(["generate", "--project", str(project), "--limit", "1000"])
    assert cli.dispatch(args) == {"scope": "bulk"}
    assert calls == [(project, "train", 1000, False, "clone", "auto", 4)]
