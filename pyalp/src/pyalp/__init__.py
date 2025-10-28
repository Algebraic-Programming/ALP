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


# Expose available backend submodules (if present in the installed wheel) so users
# can import them as `from pyalp import pyalp_ref` or access `pyalp.pyalp_ref`.
_backend_candidates = ["pyalp_ref", "pyalp_omp", "pyalp_nonblocking", "_pyalp"]
for _b in _backend_candidates:
    try:
        _m = importlib.import_module(f"{__package__}.{_b}")
        globals()[_b] = _m
        if _b not in __all__:
            __all__.append(_b)
    except Exception:
        # ignore missing backends
        continue
    else:
        # if imported successfully, also register a top-level alias so
        # `import pyalp_ref` can work for users expecting the former layout.
        try:
            # ensure the module object is in globals
            _mod = globals().get(_b)
            if _mod is not None:
                # register top-level module name to point to the submodule
                sys.modules[_b] = _mod
        except Exception:
            pass
