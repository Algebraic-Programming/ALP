"""
Test script for the pyalp_ref (GraphBLAS-like) Python module, specifically the simulated annealing Replica-Exchange (SARE) stuff.

This file should also work as a simple example of usage of such features.

Usage:
    python sare_test.py

Dependencies:
    - numpy
    - scipy.sparse
    - pyalp_ref (should be available in the Python path)
"""

import numpy as np
from scipy.sparse import dok_array
from time import time

import pyalp
# select backend
backendname = "pyalp_omp" # or "pyalp_omp" "pyalp_nonblocking"
pyalp = pyalp.get_backend(backendname)

# set here your parameters

n_replicas = 32
seed = 0
np.random.seed(seed)

# This is the data for the problem matrix
# note that data types have been fixed at compile time!
N = 5
J = dok_array((N, N), dtype=np.float64)

J[0,1] = 2
J[1,3] = 4
J[2,3] = -1
J[4,1] = 3
J[4,1] = -2
h = np.zeros( N )

# You can also import a matrix as follows:
from scipy.io import mmread
# J = mmread(matpath).astype(np.float64)

# read the local fields from a file
# h = np.loadtxt(vecpath, dtype=np.float64)

# or get the vector from the diagonal of the matrix
# if np.any(J.diagonal() != 0):
    # h = J.diagonal()
    # J.setdiag(0)
N = J.shape[0]

# Make sure the matrix is symmetric with zero diagonal, as that is a precondition!
J = (J + J.T)/2

# if diagonal is not zero we can call
assert np.all(J.diagonal() == 0)


# create coordinates and value arrays
J = J.todok()
idata,jdata = np.array([[x,y] for x,y in J.keys()], dtype=np.int32).T
vdata = np.array( list(J.values()), dtype=np.float64 )
idata, jdata, vdata = np.sort(idata), jdata[np.argsort(idata)], vdata[np.argsort(idata)]

# initial states
states_numpy = np.random.randint(0,2, (n_replicas, N), dtype=np.int8 )

# Initialize temperatures
betas_numpy = np.logspace( 1e-2, 1e+2, n_replicas, dtype=np.float64 )

# Initialize energies
energies_numpy = np.diag(states_numpy@J@states_numpy.T)/2 + np.dot(states_numpy, h)

niterations = 100
verbose = 0

#########################
# Create the pyalp_ref Matrix and Vector objects
J_alp = pyalp.Matrix( N, N, idata, jdata, vdata )
h_alp = pyalp.Vector( N, h )
states =  [pyalp.State(N, state) for state in states_numpy ]
betas = pyalp.Vector( n_replicas, betas_numpy )
# energies will be initialized by the library
energies   = pyalp.Vector( n_replicas, energies_numpy  )

# Preallocate best state
best_state = pyalp.State( N, np.zeros(N, dtype=np.int8) )

t0 = time()

# For Ising solver
best_energy = pyalp.SARE_Ising( J_alp, h_alp, states, energies, betas, best_state, niterations, seed, verbose )

# There is also a QUBO solver, with the only difference being that h is fixed 0
# best_energy = pyalp.SARE_Ising( J_alp, states, energies, betas, best_state, niterations, seed, verbose )
t1 = time()

best_state_numpy = best_state.to_numpy()

print( best_energy )
print( best_state_numpy )
print( f"Solver time is: {(t1-t0)*1000} ms" )

# check that energy is correct:
assert best_state_numpy@(J@best_state_numpy/2 + h )  == best_energy



