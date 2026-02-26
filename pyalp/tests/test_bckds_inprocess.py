"""
Simple in-process test to verify multiple pyalp backend extension modules
can be imported and used in the same Python interpreter without pybind11
duplicate-type registration collisions.

Usage (when building locally):

  # configure & build top-level project with pyalp enabled
  cmake -S . -B build -DENABLE_PYALP=ON
  cmake --build build --target pyalp

  # run the test pointing PYTHONPATH to the build output
  PYTHONPATH=build/pyalp/src python3 pyalp/tests/test_bckds_inprocess.py

If the build places extensions elsewhere, adjust PYTHONPATH to include that
directory.
"""

import sys
import importlib
import numpy as np

# Optionally prepend a build directory. If you're running inside the repo and
# built into ../build, uncomment and adjust the path below.
# sys.path.insert(0, '/path/to/your/build/pyalp/src')

BACKENDS = ['pyalp_ref', 'pyalp_omp', 'pyalp_nonblocking']

def make_simple_matrix():
    # Create arrays for a single non-zero entry at (0,0) with value 1.0
    i = np.array([0], dtype=np.int64)
    j = np.array([0], dtype=np.int64)
    v = np.array([1.0], dtype=np.float64)
    return 1, 1, i, j, v


def main():
    m,n,i,j,v = make_simple_matrix()
    exercised = 0

    # If the installed package exposes a `pyalp` package, prefer to query
    # it for the list of available backends and skip any that aren't present
    # (useful for platform-specific wheels that omit some backends).
    installed_backends = None
    try:
        pkg = importlib.import_module('pyalp')
        try:
            installed_backends = set(pkg.list_backends())
        except Exception:
            installed_backends = None
    except ModuleNotFoundError:
        installed_backends = None

    for backend in BACKENDS:
        # If we detected an installed pyalp package and it doesn't list this
        # backend, skip it rather than failing the whole test.
        if installed_backends is not None and backend not in installed_backends:
            print(f"Backend {backend} not present in installed package, skipping")
            continue

        # Try importing the module as a top-level module first (old-style),
        # then as a submodule of the installed `pyalp` package. This mirrors
        # how the wheel packages the compiled extensions under the `pyalp`
        # package (pyalp.pyalp_ref, etc.). We also attach the imported
        # submodule to the `pyalp` package object for convenience.
        mod = None
        try:
            mod = importlib.import_module(backend)
        except ModuleNotFoundError:
            try:
                fq = f"pyalp.{backend}"
                mod = importlib.import_module(fq)
                # Attach to pyalp package so attribute access works
                try:
                    pkg = importlib.import_module('pyalp')
                    setattr(pkg, backend, mod)
                except Exception:
                    pass
            except Exception as e:
                print(f"FAILED IMPORT {backend}: {e}")
                raise
        print(f"Imported {backend}: {mod}")
        try:
            Matrix = getattr(mod, 'Matrix')
        except AttributeError:
            print(f"{backend} does not expose Matrix")
            raise
        # Construct an instance
        try:
            mat = Matrix(m, n, i, j, v)
            exercised += 1
            print(f"Constructed Matrix from {backend}:", type(mat))
        except Exception as e:
            print(f"FAILED TO CONSTRUCT Matrix from {backend}: {e}")
            raise

    print('\nALL BACKENDS IMPORTED AND INSTANCES CREATED SUCCESSFULLY')
    if exercised == 0:
        print('ERROR: no backends were exercised (none installed).', file=sys.stderr)
        raise SystemExit(2)
    else:
        print(f'SUCCESS: exercised {exercised} backend(s).')

if __name__ == '__main__':
    main()
