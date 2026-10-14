"""Hash-bound artifacts, atomic publication, and single-writer ownership."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


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
