"""Durable publication for resumable feature and training stages."""
import json
import os

from .common import _retry_windows_sharing


def publish_checkpoint(path, save):
    """Write and sync a checkpoint before atomically publishing it."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        save(stream)
        stream.flush()
        os.fsync(stream.fileno())
    _retry_windows_sharing(lambda: os.replace(temporary, path))
    if os.name != "nt":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def write_checkpoint_json(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    publish_checkpoint(path, lambda stream: stream.write(encoded))
