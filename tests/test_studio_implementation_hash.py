"""The source freshness scan remains byte-exact without oversized read buffers."""
import os
from pathlib import Path

from korean_openwakeword import studio
from korean_openwakeword.common import digest, identity


def original_implementation(root):
    paths = sorted(root.glob("*.py")) + sorted((root / "resources").glob("*.json"))
    return identity({path.relative_to(root).as_posix(): digest(path) for path in paths})


def test_source_scan_is_identical_to_existing_artifact_hashes_for_the_whole_package():
    root = Path(studio.__file__).parent
    assert studio._implementation() == original_implementation(root)


def test_streaming_scan_covers_large_boundaries_and_ignores_unclassified_assets(tmp_path, monkeypatch):
    package = tmp_path / "package"
    resources = package / "resources"
    resources.mkdir(parents=True)
    source = package / "studio.py"
    source.write_bytes(b"# fixture\n" + b"x" * (128 * 1024 * 3 + 17))
    catalog = resources / "choices.json"
    catalog.write_bytes(b'{"fixture":"' + b"y" * (128 * 1024 + 5) + b'"}')
    (package / "empty.py").write_bytes(b"")
    (resources / "display.html").write_text("presentation one", encoding="utf-8")
    monkeypatch.setattr(studio, "__file__", str(source))
    before = studio._implementation()
    assert before == original_implementation(package)
    (resources / "display.html").write_text("presentation two", encoding="utf-8")
    assert studio._implementation() == before
    with source.open("r+b") as output:
        output.seek(128 * 1024 * 2 + 1)
        output.write(b"z")
    assert studio._implementation() == original_implementation(package) != before


def test_same_size_and_restored_mtime_edits_are_detected_without_caching(tmp_path, monkeypatch):
    package = tmp_path / "package"
    resources = package / "resources"
    resources.mkdir(parents=True)
    source = package / "studio.py"
    source.write_bytes(b"# version one\n")
    catalog = resources / "catalog.json"
    catalog.write_text('{"version":1}', encoding="utf-8")
    monkeypatch.setattr(studio, "__file__", str(source))
    initial = studio._implementation()
    for target, replacement in ((source, b"# version two\n"), (catalog, b'{"version":2}')):
        before = studio._implementation()
        stat = target.stat()
        assert len(replacement) == stat.st_size
        target.write_bytes(replacement)
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert target.stat().st_size == stat.st_size and target.stat().st_mtime_ns == stat.st_mtime_ns
        assert studio._implementation() == original_implementation(package) != before
    assert studio._implementation() != initial
    extra = package / "new_module.py"
    extra.write_text("# new dependency\n", encoding="utf-8")
    added = studio._implementation()
    assert added == original_implementation(package)
    extra.unlink()
    assert studio._implementation() == original_implementation(package) != added


def test_scan_rechecks_deleted_or_unreadable_source_instead_of_reusing_a_prior_hash(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "studio.py"
    source.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setattr(studio, "__file__", str(source))
    before = studio._implementation()
    actual_open = Path.open
    def denied(path, *args, **kwargs):
        if path == source:
            raise PermissionError("fixture denied source")
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", denied)
    import pytest
    with pytest.raises(PermissionError, match="denied source"):
        studio._implementation()
    monkeypatch.setattr(Path, "open", actual_open)
    source.unlink()
    assert studio._implementation() == original_implementation(package) != before
