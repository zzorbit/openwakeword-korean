"""Hash-bound artifacts, atomic publication, and single-writer ownership."""
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def _retry_windows_sharing(operation, *, read_access=False):
    """Bound brief Windows file-access races without masking persistent errors."""
    for attempt in range(8):
        try:
            return operation()
        except OSError as error:
            # CPython file reads can report a Windows sharing race as EACCES
            # without winerror. Permanent denial still fails after the bound.
            transient = (getattr(error, "winerror", None) in (5, 32, 33)
                         or read_access and os.name == "nt" and isinstance(error, PermissionError)
                         and error.errno == errno.EACCES)
            if not transient or attempt == 7:
                raise
            time.sleep(min(0.01 * 2**attempt, 0.25))


def read_json(path):
    text = _retry_windows_sharing(lambda: Path(path).read_text(encoding="utf-8-sig"), read_access=True)
    return json.loads(text)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers/scanners can briefly deny replacement of an open file.
    # Retry the same atomic publish; never truncate the destination or relax
    # non-sharing failures. A final failure preserves both files for recovery.
    _retry_windows_sharing(lambda: os.replace(temp, path))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def iter_rows(path):
    """Read JSONL in order without retaining a large window or feature journal."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_rows(path):
    return list(iter_rows(path))


def append_row(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def resolved(root, path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path(root) / p).resolve()


def bound_file(root, entry):
    path = resolved(root, entry["path"])
    if not path.is_file() or digest(path) != entry["sha256"]:
        raise ValueError(f"Missing or changed bound file: {path}")
    return path


def snapshot_json(root, value, prefix):
    """Keep reusable admission evidence independent of mutable UI summaries."""
    root = Path(root).resolve()
    path = root / "review" / "evidence" / f"{prefix}_{identity(value)}.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable evidence snapshot changed")
    else:
        write_json(path, value)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def require_final_lock(project):
    """Check the actual frozen configuration before opening Final generation."""
    project = Path(project).resolve()
    path = project.parent / "final_lock.json"
    if not path.is_file():
        raise ValueError("Final stays locked/reserved until model/runtime freeze")
    lock = read_json(path)
    if lock.get("project_config_sha256") != digest(project):
        raise ValueError("Final project configuration changed after freeze")
    bundle = resolved(project.parent, lock["bundle"])
    if digest(bundle / "manifest.json") != lock["bundle_manifest_sha256"]:
        raise ValueError("Final bundle changed after freeze")
    from .runtime import verify_bundle, deployment_candidate
    package = verify_bundle(bundle)
    if lock.get("runtime", package["runtime"]) != package["runtime"]:
        raise ValueError("Final runtime changed after freeze")
    if lock.get("selected_candidate", deployment_candidate(package)) != deployment_candidate(package):
        raise ValueError("Final deployed candidate changed after freeze")
    for name, entry in lock.get("input_bindings", {}).items():
        try:
            bound_file(project.parent, entry)
        except ValueError as exc:
            raise ValueError(f"Frozen Train/common/Final input changed after freeze: {name}") from exc
    return lock


@contextmanager
def writer_lock(directory):
    """A failed owner leaves evidence; do not steal a possibly live job's lock."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".writer.lock"
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": now()}, f)
    except FileExistsError:
        raise RuntimeError(f"Writer lock exists: {path}. Check its actual PID before recovery.")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
