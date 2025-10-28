
# pyalp (packaged)

This directory contains the Python package layout for the `pyalp` bindings
that expose parts of the ALP GraphBLAS project via pybind11.

Quick start
-----------

Create and activate a virtual environment, then install the package (example
using TestPyPI):

```bash
python -m venv venv
source venv/bin/activate
pip install --index-url https://test.pypi.org/simple/ --no-deps pyalp
```

Basic usage
-----------
# pyalp (packaged)

This directory contains the Python package layout for the `pyalp` bindings
that expose parts of the ALP GraphBLAS project via pybind11.

Quick start
-----------

Create and activate a virtual environment, then install the package (example
using TestPyPI):

```bash
python -m venv venv
source venv/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyalp
```

Basic usage
-----------

The package exposes a small set of helpers and one or more compiled backend
modules. Use these helpers to list and select available backends and to read
runtime build metadata:

```python
import pyalp
print('pyalp build metadata:', pyalp.get_build_metadata())
print('available backends:', pyalp.list_backends())

# Import a specific backend module (returns the compiled module)
backend = pyalp.get_backend('pyalp_ref')  # or 'pyalp_omp', 'pyalp_nonblocking'
print('backend module:', backend)
```

Backends and import caveat
--------------------------

Wheels may include multiple compiled backend modules (for example
`pyalp_ref`, `pyalp_omp`, `pyalp_nonblocking`). Historically, importing
multiple different compiled backends in the same Python process could raise
pybind11 registration errors (types duplicated). The bindings now use
`py::module_local()` for core wrapper types, which reduces collisions, but if
you encounter issues importing more than one backend in-process, prefer
testing each backend in a separate process (the supplied test runner does
this).

Runtime metadata
----------------

The package provides a metadata module generated at build time by CMake. Use
`pyalp.get_build_metadata()` to access keys such as:

- `version` — pyalp package version
- `build_type` — CMake build type used (e.g., Release)
- `alp_version` — ALP repository version or tag used to build
- `alp_git_commit` / `alp_git_branch` — Git information captured by CI
- `license` — detected repository license (e.g. Apache-2.0)

`pyalp.get_algorithm_metadata()` contains algorithm/backends info and a
`readme` key with packaged README contents.

Minimal example — conjugate gradient (small test)
------------------------------------------------

Save the following as `test_cg.py` and run `python test_cg.py` after installing
`pyalp` and `numpy`. The example shows selecting a backend explicitly via
`pyalp.get_backend()` and then using the backend's `Matrix`, `Vector`, and
`conjugate_gradient` API.

```python
#!/usr/bin/env python3
"""
Test script for the pyalp backend (example uses the OpenMP backend name
`pyalp_omp`, but you can use `pyalp_ref` or another available backend).

Usage:
		python test_cg.py

Dependencies:
		- numpy
		- pyalp (installed and providing a backend such as pyalp_omp)
"""

import numpy as np
import pyalp

# Choose the backend module (change name if you want a different backend)
pyalp = pyalp.get_backend('pyalp_omp')  # or 'pyalp_ref', 'pyalp_nonblocking'

# Generate a small sparse linear system using numpy arrays
N, M = 5, 5
idata = np.array([0, 1, 2, 3, 3, 4, 2, 3, 3, 4, 1, 4, 1, 4, 4], dtype=np.int32)
jdata = np.array([0, 1, 2, 3, 2, 2, 1, 4, 1, 1, 0, 3, 0, 3, 4], dtype=np.int32)
vdata = np.array([1, 1, 1, 1, 0.5, 2, 1, 4, 4.4, 1, 0, 3.5, 0, 3, 1], dtype=np.float64)
b = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
x = np.array([1.0, 1.0, 0.0, 0.3, -1.0], dtype=np.float64)
r = np.zeros(5, dtype=np.float64)
u = np.zeros(5, dtype=np.float64)
tmp = np.zeros(5, dtype=np.float64)

# Create the pyalp Matrix and Vector objects
alpmatrixA = pyalp.Matrix(5, 5, idata, jdata, vdata)
alpvectorx = pyalp.Vector(5, x)
alpvectorb = pyalp.Vector(5, b)
alpvectorr = pyalp.Vector(5, r)
alpvectoru = pyalp.Vector(5, u)
alpvectortmp = pyalp.Vector(5, tmp)

maxiterations = 2000
verbose = 1

# Solve the linear system using the conjugate gradient method in the backend
iterations, residual = pyalp.conjugate_gradient(
		alpmatrixA,
		alpvectorx,
		alpvectorb,
		alpvectorr,
		alpvectoru,
		alpvectortmp,
		maxiterations,
		verbose,
)
print('iterations =', iterations)
print('residual =', residual)

# Convert the result vector to a numpy array and print it
x_result = alpvectorx.to_numpy()
print('x_result =', x_result)

# Check if the result is close to the expected solution
assert np.allclose(x_result, np.array([1.0, 1.0, 0.0, 0.13598679, -0.88396565])), 'solution mismatch'
```

Packaging notes (for maintainers)
--------------------------------

- The CI uses a top-level CMake configure/build to produce the native shared
	object and a CMake-configured `_metadata.py`. The packaging `setup.py` then
	copies the built `.so` and `_metadata.py` into the wheel.
- The CI passes Git/version information into CMake so the generated metadata
	is populated even in detached/CI environments.

If you modify the metadata template, update `pyalp/src/pyalp/_metadata.py.in`.

License
-------

See the repository `LICENSE` at the project root; the packaging pipeline
attempts to detect and embed the license string in runtime metadata.
