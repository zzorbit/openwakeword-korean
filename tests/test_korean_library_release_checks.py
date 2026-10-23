"""Source releases and CI must retain every browser contract, without local data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_tree(tmp_path):
    packager = _script("package_library")
    root = tmp_path / "repository"
    root.mkdir()
    for name in packager.ROOT_FILES:
        (root / name).write_text("public release\n", encoding="utf-8")
    for directory in ("korean_openwakeword", ".github", "examples/library_demo_site", "docs", "scripts", "tests", "validation"):
        (root / directory).mkdir(parents=True)
    for name in packager.GUIDES:
        (root / "docs" / name).write_text("public guide\n", encoding="utf-8")
    for name in ("Use-ProjectStorage.ps1", "Start-Web.ps1", "Start-Web.cmd", "package_library.py", "check_browser.py"):
        shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
    for source in (ROOT / "tests").glob("check_*.cjs"):
        shutil.copyfile(source, root / "tests" / source.name)
    # A newly added check follows the convention without another release list.
    (root / "tests/check_future.cjs").write_text("process.exitCode = 0;\n", encoding="utf-8")
    (root / "tests/test_korean_library_future.py").write_text("def test_future(): pass\n", encoding="utf-8")
    (root / "korean_openwakeword/public.py").write_text("VALUE = 1\n", encoding="utf-8")
    return packager, root


def test_source_release_retains_all_browser_checks_and_only_allowlisted_data(source_tree, tmp_path):
    packager, root = source_tree
    for relative in (".env", "cache/private.txt", "reports/private.json", "scripts/local_private.py",
                     "tests/local_private.py", "tests/check_private.cjs.tmp", "korean_openwakeword/__pycache__/private.pyc",
                     "korean_openwakeword/private.tmp", "korean_openwakeword/.writer.lock"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not distributable", encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    output = tmp_path / "release"
    result = packager.stage_library(root, output, inventory)
    expected = {path.name for path in (root / "tests").glob("check_*.cjs")}
    assert len(expected) >= 7
    assert {path.name for path in (output / "tests").glob("check_*.cjs")} == expected
    assert (output / "scripts/check_browser.py").read_bytes() == (root / "scripts/check_browser.py").read_bytes()
    assert (output / "tests/test_korean_library_future.py").is_file()
    paths = {entry["path"] for entry in result["files"]}
    assert not any("private" in path or path.endswith(".writer.lock") for path in paths)
    assert not inventory.is_relative_to(output)
    assert json.loads(inventory.read_text(encoding="utf-8")) == result
    for entry in result["files"]:
        data = (output / entry["path"]).read_bytes()
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()


def test_source_release_still_rejects_private_text(source_tree, tmp_path):
    packager, root = source_tree
    # Construct a fake credential; never embed even a fake scanner match in a release.
    (root / "korean_openwakeword/public.py").write_text("github_" + "pat_" + "a" * 25, encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    with pytest.raises(ValueError, match="Private path or credential"):
        packager.stage_library(root, tmp_path / "release", inventory)
    assert not inventory.exists()


def test_browser_runner_discovers_checks_and_records_failure_without_skipping_later_checks(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for actual browser runner execution")
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    (root / "tests/check_b_fail.cjs").write_text("console.error('fixture failure'); process.exitCode = 7;", encoding="utf-8")
    (root / "tests/check_a_pass.cjs").write_text("console.log('first fixture');", encoding="utf-8")
    (root / "tests/check_c_pass.cjs").write_text("console.log('last fixture');", encoding="utf-8")
    (root / "tests/private.cjs").write_text("throw Error('must not run');", encoding="utf-8")
    output = tmp_path / "logs"
    report = _script("check_browser").run_browser_checks(root, output, node=node)
    assert report["passed"] is False
    assert [row["name"] for row in report["checks"]] == ["check_a_pass.cjs", "check_b_fail.cjs", "check_c_pass.cjs"]
    assert [row["returncode"] for row in report["checks"]] == [0, 7, 0]
    assert "fixture failure" in (output / "check_b_fail.log").read_text(encoding="utf-8")
    assert "last fixture" in (output / "check_c_pass.log").read_text(encoding="utf-8")
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    cli = subprocess.run([__import__("sys").executable, str(ROOT / "scripts/check_browser.py"),
                          "--repository", str(root), "--output", str(tmp_path / "cli")], capture_output=True)
    assert cli.returncode == 1


@pytest.mark.parametrize("error", [OSError("Node could not start"), subprocess.TimeoutExpired(["node"], 120)])
def test_browser_runner_reports_launch_or_timeout_errors(tmp_path, monkeypatch, error):
    runner = _script("check_browser")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/check_fixture.cjs").write_text("", encoding="utf-8")

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner.subprocess, "run", failed)
    report = runner.run_browser_checks(tmp_path, tmp_path / "logs")
    assert report["passed"] is False
    assert report["checks"][0]["returncode"] is None
    assert str(error) in (tmp_path / "logs/check_fixture.log").read_text(encoding="utf-8")


def test_browser_runner_rejects_missing_checks_instead_of_reporting_success(tmp_path):
    with pytest.raises(ValueError, match="No browser checks"):
        _script("check_browser").run_browser_checks(tmp_path, tmp_path / "logs")


def test_standard_source_release_and_ci_use_the_complete_browser_check_convention():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include tests test_korean_library*.py check_*.cjs" in manifest
    assert "include scripts/check_browser.py" in manifest
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python scripts/check_browser.py --output .test-tmp/browser-checks" in workflow
    assert ".test-tmp/browser-checks/*" in workflow
    assert "python -m build" in workflow
