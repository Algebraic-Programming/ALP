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



for backendname in ['pyalp_ref','pyalp_omp','pyalp_nonblocking']:

    import pyalp
    # Choose the backend module (change name if you want a different backend)
    pyalp = pyalp.get_backend(backendname)  # or 'pyalp_ref', 'pyalp_nonblocking'

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

    print("backend ", backendname, " OK")
