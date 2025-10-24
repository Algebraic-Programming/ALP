#!/usr/bin/env python3
"""Smoke-test installer for the pyalp wheel.

Usage: python tools/smoke_test_pyalp.py

This script imports `pyalp`, checks for presence of `_pyalp` extension and
tries to call `backend_name()` if present. It exits non-zero on failure and
prints helpful tracebacks.
"""
import sys
import traceback


def main():
    try:
        import pyalp
    except Exception as e:
        print("ERROR: importing pyalp failed:", e, file=sys.stderr)
        traceback.print_exc()
        return 2
    ext = getattr(pyalp, "_pyalp", None)
    ok_ext = ext is not None
    print("pyalp import OK, compiled ext loaded:", ok_ext)
    if not ok_ext:
        # Try to import the extension directly and print diagnostics
        try:
            import importlib
            ext = importlib.import_module("pyalp._pyalp")
            print("Direct import succeeded after fallback.")
            ok_ext = True
        except Exception as e:
            print("Extension import failed:", e, file=sys.stderr)
            traceback.print_exc()
            try:
                import importlib.util
                import pathlib
                spec = importlib.util.find_spec("pyalp")
                pkgdir = None
                if spec and spec.submodule_search_locations:
                    pkgdir = pathlib.Path(list(spec.submodule_search_locations)[0])
                else:
                    pkgdir = pathlib.Path(__import__("pyalp").__file__).parent  # type: ignore[attr-defined]
                print("pyalp dir:", pkgdir)
                print(".so files:")
                for p in pkgdir.iterdir():
                    if p.suffix == ".so":
                        print(" -", p)
            except Exception:
                pass
    if ok_ext:
        try:
            if ext is not None and hasattr(ext, "backend_name"):
                name = ext.backend_name()
                print("backend_name:", name)
            else:
                print("Extension module loaded but missing backend_name()", file=sys.stderr)
                return 3
        except Exception as e:
            print("calling backend failed:", e, file=sys.stderr)
            traceback.print_exc()
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
