"""Project-owned download/temp caches, never an implicit Codex/user-home cache.

Process environment is scoped and restored. Concurrent projects should run in
separate processes; the lock prevents overlapping environment changes here.
"""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import sys
import tempfile
import threading

_LOCK = threading.RLock()


def storage_paths(project_root):
    project_root = Path(project_root).resolve()
    selected = os.environ.get("KOWW_CACHE_ROOT")
    cache = Path(selected).expanduser().resolve() if selected else project_root / "cache"
    if any(part.lower() == ".codex" for part in cache.parts):
        raise ValueError("Wakeword bulk cache must not be stored inside .codex")
    return {"root": cache, "hf_home": cache / "huggingface", "hf_hub": cache / "huggingface" / "hub",
        "hf_xet": cache / "huggingface" / "xet", "torch": cache / "torch",
        "temp": cache / "tmp", "pip": cache / "pip", "xdg": cache / "xdg",
        "torchinductor": cache / "torchinductor", "triton": cache / "triton",
        "cuda": cache / "cuda", "torch_extensions": cache / "torch_extensions", "numba": cache / "numba"}


@contextmanager
def storage_environment(project_root):
    paths = storage_paths(project_root)
    environment = {"HF_HOME": paths["hf_home"], "HF_HUB_CACHE": paths["hf_hub"],
        "HUGGINGFACE_HUB_CACHE": paths["hf_hub"], "HF_XET_CACHE": paths["hf_xet"],
        "TORCH_HOME": paths["torch"], "TMP": paths["temp"], "TEMP": paths["temp"],
        "TMPDIR": paths["temp"], "PIP_CACHE_DIR": paths["pip"], "XDG_CACHE_HOME": paths["xdg"],
        "TORCHINDUCTOR_CACHE_DIR": paths["torchinductor"], "TRITON_CACHE_DIR": paths["triton"],
        "CUDA_CACHE_PATH": paths["cuda"], "TORCH_EXTENSIONS_DIR": paths["torch_extensions"], "NUMBA_CACHE_DIR": paths["numba"]}
    with _LOCK:
        for directory in paths.values():
            directory.mkdir(parents=True, exist_ok=True)
        previous = {key: os.environ.get(key) for key in environment}
        previous_temp = tempfile.tempdir
        # HF constants may already have been imported by a caller. Preserve
        # and scope them too, so xet chunk caching cannot use an old C path.
        hf_constants = sys.modules.get("huggingface_hub.constants")
        previous_hf = {key: getattr(hf_constants, key) for key in environment
                       if hf_constants is not None and hasattr(hf_constants, key)}
        try:
            os.environ.update({key: str(value) for key, value in environment.items()})
            # Python may have cached its system temp folder before this call.
            tempfile.tempdir = str(paths["temp"])
            for key in previous_hf:
                setattr(hf_constants, key, str(environment[key]))
            yield paths
        finally:
            tempfile.tempdir = previous_temp
            for key, value in previous_hf.items():
                setattr(hf_constants, key, value)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def project_storage(function):
    @wraps(function)
    def wrapped(project, *args, **kwargs):
        with storage_environment(Path(project).resolve().parent):
            return function(project, *args, **kwargs)
    return wrapped


def local_tts_snapshot(model_path, config, project_root):
    """Explicit hub cache survives earlier imports of HF constants.

Qwen 0.1.1 does not forward cache_dir/revision to its processor. Loading the
complete pinned local snapshot keeps all sub-loaders out of the user cache.
"""
    local = Path(model_path).expanduser()
    if local.is_dir():
        return str(local.resolve())
    revision = config.get("model_revisions", {}).get(model_path)
    if not revision:
        raise ValueError("Remote TTS model needs a pinned revision")
    from huggingface_hub import snapshot_download
    cache = storage_paths(project_root)["hf_hub"]
    _prepare_hub_cache(model_path, cache)
    return snapshot_download(repo_id=model_path, revision=revision,
        cache_dir=str(cache))


def _prepare_hub_cache(repo_id, cache):
    """Finish HF's link-capability probe before its parallel file workers.

    HF 0.36 temporarily caches True before testing the filesystem. Another
    snapshot worker can observe that provisional value and attempt a Windows
    symlink without privileges. Probe the exact repository directory first;
    HF then uses its normal move/copy fallback when links are unsupported.
    """
    from huggingface_hub.file_download import are_symlinks_supported, repo_folder_name
    with _LOCK:
        are_symlinks_supported(str(Path(cache) / repo_folder_name(repo_id=repo_id, repo_type="model")))


def describe_storage(project):
    root = Path(project).resolve().parent
    return {"project_root": str(root), "paths": {k: str(v) for k, v in storage_paths(root).items()},
        "override": os.environ.get("KOWW_CACHE_ROOT"),
        "scope": "Project defaults; optional explicit shared cache. Windows/Codex global settings unchanged."}
