#!/usr/bin/env python3
"""
Run a small conjugate-gradient smoke test for a single pyalp backend.

This script is intended to be invoked as a subprocess by tests so each backend
is exercised in a fresh interpreter (avoiding pybind11 registration conflicts).

Usage:
    python backend_smoke_runner.py pyalp_ref

It prints the iterations, residual, and resulting solution vector to stdout.
"""
import sys
import importlib
import argparse
import numpy as np


def run_smoke(backend_name: str) -> int:
    # Import backend module as pyalp.<backend_name>, fallback to top-level name
    try:
        m = importlib.import_module(f"pyalp.{backend_name}")
    except Exception:
        try:
            m = importlib.import_module(backend_name)
        except Exception as e:
            print(f"Failed to import backend '{backend_name}': {e}", file=sys.stderr)
            return 2

    idata = np.array([0, 1, 2, 3, 3, 4, 2, 3, 3, 4, 1, 4, 1, 4, 4], dtype=np.int32)
    jdata = np.array([0, 1, 2, 3, 2, 2, 1, 4, 1, 1, 0, 3, 0, 3, 4], dtype=np.int32)
    vdata = np.array([1, 1, 1, 1, 0.5, 2, 1, 4, 4.4, 1, 0, 3.5, 0, 3, 1], dtype=np.float64)
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    x = np.array([1.0, 1.0, 0.0, 0.3, -1.0], dtype=np.float64)
    r = np.zeros(5, dtype=np.float64)
    u = np.zeros(5, dtype=np.float64)
    tmp = np.zeros(5, dtype=np.float64)

    try:
        A = m.Matrix(5, 5, idata, jdata, vdata)
        xv = m.Vector(5, x)
        bv = m.Vector(5, b)
        rv = m.Vector(5, r)
        uv = m.Vector(5, u)
        tv = m.Vector(5, tmp)

        iterations, residual = m.conjugate_gradient(A, xv, bv, rv, uv, tv, 2000, 0)
        print("iterations=", iterations, "residual=", residual)
        print("x_result=", xv.to_numpy())
    except Exception as e:
        print("Backend test failed:", e, file=sys.stderr)
        return 3
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run pyalp backend smoke test")
    parser.add_argument("backend", help="backend module name (e.g. pyalp_ref)")
    args = parser.parse_args(argv)
    return run_smoke(args.backend)


if __name__ == "__main__":
    sys.exit(main())
