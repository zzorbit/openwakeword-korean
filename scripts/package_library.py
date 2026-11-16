"""Build a portable, allowlisted source tree; preserve every prior release."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat


ROOT_FILES = (
    "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE",
    "THIRD_PARTY_NOTICES.md", "plan.md", "CHANGELOG.md", "pytest.ini", ".gitignore",
)
GUIDES = (
    "LIBRARY_GUIDE.md", "DATASETS.md", "DATA_FORMAT.md", "VALIDATION.md",
    "TRAINING_RESULTS_SITE.md", "STORAGE_POLICY.md",
    "WEB_GUIDE.md", "WEB_LAYOUT.md", "CODE_IMPROVEMENTS.md",
)
PRIVATE_TEXT = re.compile(
    r"C:[\\/]+Users[\\/]+|F:[\\/]+project-(?:test_)?wakeword|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    re.IGNORECASE,
)


def _release_paths(repository):
    """List the distributable tree without traversing caches or workspaces."""
    yield from (Path(name) for name in ROOT_FILES)
    yield from (Path("docs", name) for name in GUIDES)
    yield from (Path("scripts", name) for name in (
        "Use-ProjectStorage.ps1", "Start-Web.ps1", "Start-Web.cmd",
        "package_library.py", "check_browser.py"))
    patterns = (
        "korean_openwakeword/*.py",
        *("korean_openwakeword/resources/*." + ext for ext in ("json", "html", "js", "css", "svg", "txt")),
        *("korean_openwakeword/resources/sample_chunsik/*." + ext for ext in ("json", "md")),
        "korean_openwakeword/resources/sample_chunsik/licenses/*.txt",
        "korean_openwakeword/resources/sample_chunsik/models/*.onnx",
        "korean_openwakeword/resources/sample_chunsik/frontend/*.onnx",
        "korean_openwakeword/resources/sample_chunsik/audio/*.wav",
        ".github/workflows/*.yml", ".github/workflows/*.yaml",
        *("examples/library_demo_site/*." + ext for ext in ("html", "js", "css", "json")),
        "examples/library_demo_site/audio/*.wav",
        "tests/test_*.py", "tests/check_*.cjs", "tests/fixtures/*.json", "validation/*.json",
    )
    for pattern in patterns:
        for path in sorted(repository.glob(pattern)):
            yield path.relative_to(repository)


def _require_release_file(repository, relative):
    path = repository
    for part in relative.parts:
        path /= part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or (
                getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("Linked source in release: " + relative.as_posix())
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Expected a regular source file: " + relative.as_posix())
    return path


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
    records = []
    for relative in sorted(set(_release_paths(repository))):
        source = _require_release_file(repository, relative)
        data = source.read_bytes()
        name = relative.as_posix()
        if source.suffix.lower() not in (".onnx", ".wav"):
            if PRIVATE_TEXT.search(data.decode("utf-8-sig")):
                raise ValueError("Private path or credential in release: " + name)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": name, "bytes": len(data),
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
