"""
Test script for the pyalp_ref (GraphBLAS-like) Python module.

This script sets up a small sparse linear system and solves it using the
conjugate gradient method implemented in pyalp_ref. It verifies the solution
against an expected result using numpy's allclose.

Steps performed:
- Defines a 5x5 sparse matrix in coordinate (COO) format.
- Initializes vectors for the right-hand side (b), initial guess (x), and workspace.
- Constructs pyalp_ref Matrix and Vector objects.
- Runs the conjugate gradient solver.
- Prints the number of iterations, residual, and resulting solution vector.
- Asserts that the computed solution is close to the expected values.

Usage:
    python test.py

Dependencies:
    - numpy
    - pyalp_ref (should be available in the Python path)
"""

import pyalp_ref as pyalp
import numpy as np

# Gnerate a small sparse linear system using numpy arrays
N, M = 5 , 5
idata = np.array([ 0, 1, 2, 3, 3, 4, 2, 3, 3, 4, 1, 4, 1, 4, 4 ],dtype=np.int32)
jdata = np.array([ 0, 1, 2, 3, 2, 2, 1, 4, 1, 1, 0, 3, 0, 3, 4 ],dtype=np.int32)
vdata = np.array([ 1, 1, 1, 1, .5, 2, 1, 4, 4.4, 1, 0, 3.5, 0, 3, 1 ], dtype=np.float64)
b = np.array([ 1., 1., 1., 1., 1. ], dtype=np.float64)
x = np.array([ 1,  1., 0., 0.3, -1. ], dtype=np.float64)
r = np.zeros(5)
u = np.zeros(5)
tmp = np.zeros(5)

A=np.zeros((M,N))
for i,j,v in zip(idata,jdata,vdata):
    A[i,j]=v

pyalp.print_my_numpy_array(b)

maxiterations = 2000
verbose = 1

#########################
# Create the pyalp_ref Matrix and Vector objects
alpmatrixA = pyalp.Matrix(5,5,idata,jdata,vdata)
alpvectorx = pyalp.Vector(5,x)
alpvectorb = pyalp.Vector(5,b)
alpvectorr   = pyalp.Vector(5,r)
alpvectoru   = pyalp.Vector(5,u)
alpvectortmp = pyalp.Vector(5,tmp)

#solve the linear system using conjugate gradient method in pyalp_ref
iterations,residual = pyalp.conjugate_gradient( alpmatrixA, alpvectorx, alpvectorb, alpvectorr, alpvectoru, alpvectortmp, maxiterations, verbose )
print(" iterations = ", iterations )
print(" residual = ", residual )

# Convert the result vector to a numpy array and print it
x_result=alpvectorx.to_numpy()
print(x_result)
# Check if the result is close to the expected solution
assert(np.allclose(x_result,np.array([ 1., 1.,  0., 0.13598679, -0.88396565])))
