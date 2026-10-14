"""Build a portable, allowlisted source tree; preserve every prior release."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT_FILES = (
    "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE",
    "THIRD_PARTY_NOTICES.md", "plan.md", "CHANGELOG.md", "pytest.ini", ".gitignore",
)
GUIDES = (
    "LIBRARY_GUIDE.md", "DATASETS.md", "DATA_FORMAT.md", "VALIDATION.md",
    "TRAINING_RESULTS_SITE.md", "STORAGE_POLICY.md",
)
PRIVATE_TEXT = re.compile(
    r"C:[\\/]+Users[\\/]+|F:[\\/]+project-wakeword|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
)
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp", ".writer.lock")


def stage_library(repository, output, inventory):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    inventory = Path(inventory).resolve()
    if output == repository or repository.is_relative_to(output):
        raise ValueError("Source output cannot replace or contain the working repository")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty release directory")
    if inventory.is_relative_to(output):
        raise ValueError("Inventory belongs outside the distributable source tree")
    output.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FILES:
        shutil.copyfile(repository / name, output / name)
    for name in ("korean_openwakeword", ".github", "examples/library_demo_site"):
        shutil.copytree(repository / name, output / name, ignore=IGNORE)
    (output / "docs").mkdir()
    for name in GUIDES:
        shutil.copyfile(repository / "docs" / name, output / "docs" / name)
    (output / "scripts").mkdir()
    for name in ("Use-ProjectStorage.ps1", "package_library.py"):
        shutil.copyfile(repository / "scripts" / name, output / "scripts" / name)
    (output / "tests").mkdir()
    for source in sorted((repository / "tests").glob("test_korean_library*.py")):
        shutil.copyfile(source, output / "tests" / source.name)
    node_tests = repository / "tests/check_microphone_ui.cjs"
    if node_tests.is_file():
        shutil.copyfile(node_tests, output / "tests" / node_tests.name)
    (output / "validation").mkdir()
    for source in sorted((repository / "validation").glob("*.json")):
        shutil.copyfile(source, output / "validation" / source.name)
    records = []
    for source in sorted(output.rglob("*")):
        if not source.is_file():
            continue
        data = source.read_bytes()
        relative = source.relative_to(output).as_posix()
        if source.suffix.lower() not in (".onnx", ".wav"):
            if PRIVATE_TEXT.search(data.decode("utf-8-sig")):
                raise ValueError("Private path or credential in release: " + relative)
        records.append({"path": relative, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "git_blob_sha": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
    result = {"source": str(output), "files": records, "file_count": len(records),
              "bytes": sum(r["bytes"] for r in records), "raw_training_pools_included": False,
              "private_path_and_token_scan_passed": True}
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Stage reusable library sources without local training history")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    result = stage_library(args.repository, args.output, args.inventory)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
