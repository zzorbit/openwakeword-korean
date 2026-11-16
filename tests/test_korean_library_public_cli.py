import argparse

import pytest

from korean_openwakeword import cli, studio


def test_public_cli_offers_web_and_model_utilities():
    commands = next(action.choices for action in cli.parser()._actions
                    if isinstance(action, argparse._SubParsersAction))
    assert set(commands) == {"web", "check", "doctor", "sample", "demo", "infer", "mic", "datasets"}


@pytest.mark.parametrize("command", (
    "init", "configure", "settings", "benchmark", "download-generic", "negative-plan",
    "download-source", "download-fma", "download-data", "import-generic", "index-data",
    "prepare-materials", "admit-materials", "connect-materials", "status", "audit",
    "prepare-frontend", "storage", "audit-final", "windows", "references", "generate",
    "check-references", "generate-hard", "review", "features", "verify-cache", "compare-rounds",
    "train", "package", "prepare-eval", "setup-gpu", "evaluate", "select-candidate", "tune",
    "recommend", "select-runtime", "import-references", "freeze", "site", "serve",
))
def test_retired_commands_stop_before_dispatch(command, monkeypatch):
    def unexpected_dispatch(args):
        pytest.fail(f"A removed command reached dispatch: {args.command}")

    monkeypatch.setattr(cli, "dispatch", unexpected_dispatch)
    with pytest.raises(SystemExit) as error:
        cli.main([command])
    assert error.value.code == 2


def test_web_command_launches_studio(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(studio, "serve_studio", lambda *args: calls.append(args))
    cli.main(["web", "--workspace", str(tmp_path), "--port", "8914", "--no-browser"])
    assert calls == [(tmp_path, 8914, False)]
