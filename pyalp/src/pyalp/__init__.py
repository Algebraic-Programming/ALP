"""pyalp Python package init.

Expose a small Python surface and import compiled extension if available.
"""
from importlib import metadata
try:
    # compiled extension will be available after installation or build
    from . import _pyalp  # type: ignore
except Exception:  # pragma: no cover - fallback for source tree
    _pyalp = None

__all__ = ["_pyalp"]

def version():
    try:
        return metadata.version("pyalp")
    except Exception:
        return "0.0.0"
