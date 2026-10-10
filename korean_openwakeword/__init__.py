"""Stable public API; training/TTS dependencies are loaded only when requested."""
__version__ = "0.3.0"

from .project import create_project, load_project, save_configuration, workload_summary
from .runtime import WakeWordDetector, sample_bundle

__all__ = ["WakeWordDetector", "sample_bundle", "create_project", "load_project", "save_configuration", "workload_summary", "__version__"]
