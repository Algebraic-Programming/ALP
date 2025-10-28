"""pyalp Python package init.

Expose a small Python surface and import compiled extension if available.
"""
from importlib import metadata
import importlib
import pathlib
import sys
import os

# Do NOT auto-import any compiled backend at package import time.
# Importing compiled extension modules here could cause pybind11 type
# registration conflicts if multiple backends are present. Users should
# explicitly select a backend via `get_backend()` or import a specific
# submodule (e.g. `import pyalp.pyalp_ref`).
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


__all__ = ["version", "get_build_metadata", "get_algorithm_metadata", "get_backend", "list_backends"]


def version():
    try:
        return metadata.version("pyalp")
    except Exception:
        return "0.0.0"


# Expose available backend submodules (if present in the installed wheel) so users
# Backend discovery and selection helpers.
import pkgutil


def list_backends():
    """Return a sorted list of backend module names available in the package.

    This inspects the package directory for compiled extension modules with
    expected names (e.g. pyalp_ref, pyalp_omp, pyalp_nonblocking, _pyalp).
    """
    pkgdir = pathlib.Path(__file__).parent
    found = set()
    # Use pkgutil.iter_modules on the package path to discover installed modules
    try:
        for mod in pkgutil.iter_modules([str(pkgdir)]):
            name = mod.name
            if name in ("_pyalp",) or name.startswith("pyalp_"):
                found.add(name)
    except Exception:
        # fallback: scan filenames
        for p in pkgdir.iterdir():
            if p.is_file() and p.suffix in (".so", ".pyd"):
                stem = p.name.split(".", 1)[0]
                if stem == "_pyalp" or stem.startswith("pyalp_"):
                    found.add(stem)
    return sorted(found)


def import_backend(name: str):
    """Import and return the backend module `pyalp.<name>`.

    Raises ImportError with a helpful message if the backend is not present.
    """
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except Exception as e:
        raise ImportError(f"Backend module '{name}' is not available: {e}") from e


def get_backend(name: str | None = None, preferred=("pyalp_omp", "pyalp_nonblocking", "pyalp_ref", "_pyalp")):
    """Return an imported backend module.

    Selection order:
    - If ``name`` is provided, import that backend or raise ImportError.
    - If environment variable PYALP_BACKEND is set, try to import that.
    - Otherwise iterate over ``preferred`` and return the first available.

    Raises ImportError if no backend is available.
    """
    # explicit name wins
    if name:
        return import_backend(name)

    # environment override
    env = os.environ.get("PYALP_BACKEND")
    if env:
        return import_backend(env)

    # try preferred list
    available = set(list_backends())
    for pref in preferred:
        if pref in available:
            return import_backend(pref)

    raise ImportError(f"No pyalp backend available. Found: {sorted(available)}")
