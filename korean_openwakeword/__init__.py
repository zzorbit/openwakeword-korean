"""Stable public API; training/TTS dependencies are loaded only when requested."""
__version__ = "0.33.2"

from .project import create_project, load_project, save_configuration, workload_summary

__all__ = ["WakeWordDetector", "sample_bundle", "create_project", "load_project", "save_configuration", "workload_summary", "__version__"]


def __getattr__(name):
    if name in ("WakeWordDetector", "sample_bundle"):
        from . import runtime
        value = getattr(runtime, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(__all__))
