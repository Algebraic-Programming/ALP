"""pyalp Python package init.

Expose a small Python surface and import compiled extension if available.
"""
from importlib import metadata
import importlib
import pathlib
import sys

# compiled extension will be available after installation or build
try:
    from . import _pyalp  # type: ignore
except Exception:  # pragma: no cover - fallback for source tree
    # Fallback: try to discover any compiled extension in the package directory
    _pyalp = None
    try:
        pkgdir = pathlib.Path(__file__).parent
        for p in pkgdir.iterdir():
            if p.suffix == ".so":
                # PEP 3149 allows ABI tags in the filename (e.g. _pyalp.cpython-311-x86_64-linux-gnu.so)
                # The module name is the part before the first dot.
                modname = p.name.split(".", 1)[0]
                try:
                    # Use absolute import to avoid import-time package-relative issues
                    m = importlib.import_module(f"{__package__}.{modname}")
                    _pyalp = m
                    break
                except Exception:
                    # ignore and try next candidate
                    continue
    except Exception:
        _pyalp = None

# compiled metadata will be available after installation or build
try:
    from ._metadata import get_build_metadata, get_algorithm_metadata
except ImportError:  # pragma: no cover - fallback for source tree

    def get_build_metadata():
        """Return an empty dictionary if metadata is not available."""
        return {}

    def get_algorithm_metadata():
        """Return an empty dictionary if metadata is not available."""
        return {}


__all__ = ["_pyalp", "version", "get_build_metadata", "get_algorithm_metadata"]


def version():
    try:
        return metadata.version("pyalp")
    except Exception:
        return "0.0.0"
