"""
Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from more_itertools import distribute
from itertools import repeat
from multiprocessing import Pool
from os import cpu_count

import numpy as np
from typing import List, Tuple, Any, Callable, Optional, NoReturn


Vector = List[float]
# Matrix = List[List[float]] instead for List[List[float]] we will use sparse matrix from scipy
from scipy.sparse import csr_matrix
# Define general type
dtype = 'float'
# set random seed
random_seed = 8

Vector = List[float]
Matrix = csr_matrix
State = List[float]
RefProbFun = Callable[[Vector, Optional[int]], float]
EnergyFunction = Callable[[Matrix, Vector, State], float]
UpdateSpinFunRef = Callable[
    [Matrix, Vector, Vector, int, float, float, RefProbFun], float]
UpdateSpinFunction = Callable[[Matrix, Vector, Vector, int, float, float],
                              float]
SweepFunction = Callable[[UpdateSpinFunction, Matrix, Vector, Vector, float],
                         float]

def partition_rows_by_independent_sets(couplings: csr_matrix, *, method: str = "welsh_powell"):
    """
    Partition the rows of a sparse symmetric matrix into independent sets (color classes)
    so that within each set no two rows share a non-zero coupling (i.e. they are
    pairwise non-adjacent in the row-interaction graph).

    This implements a greedy graph-coloring (Welsh-Powell) style algorithm on the
    graph implied by the sparsity pattern of `couplings`. The intent is to produce
    row blocks that can be updated in parallel (or in any order) without violating
    sequential Metropolis dependencies.

    Parameters
    - couplings: csr_matrix (shape [n, n]) sparse symmetric coupling matrix (only sparsity matters)
    - method: currently only "welsh_powell" supported

    Returns
    - blocks: list of lists, each inner list is a color class containing row indices
    - colors: numpy array of length n with integer color for each row

    Complexity: O(n log n + m) where m is number of nonzeros (greedy ordering cost)
    """

    if not isinstance(couplings, csr_matrix):
        couplings = csr_matrix(couplings)

    n = couplings.shape[0]

    # Build adjacency lists from the sparsity pattern (ignore diagonal)
    indptr = couplings.indptr
    indices = couplings.indices

    degrees = np.diff(indptr)

    # Order vertices by decreasing degree (Welsh-Powell)
    order = np.argsort(-degrees)

    colors = -1 * np.ones(n, dtype=int)
    blocks = []

    for v in order:
        if colors[v] != -1:
            continue

        # Try to place v into the first existing color where it's independent
        placed = False
        for c_idx, block in enumerate(blocks):
            # Check independence: v must not be adjacent to any node in block
            # We can test by scanning neighbors of v and see if any have colors == c_idx
            neigh = indices[indptr[v]:indptr[v+1]]
            if not np.intersect1d(neigh, np.array(block)).size:
                block.append(int(v))
                colors[v] = c_idx
                placed = True
                break

        if not placed:
            # Create new color
            colors[v] = len(blocks)
            blocks.append([int(v)])

    return blocks, colors


def compute_row_batches(couplings: csr_matrix, *, order: str = "color", return_row_blocks: bool = False):
    """
    Compute a sequential list of batches (sizes) that cover all rows exactly once.

    The simplest use is to partition rows by independent sets using
    `partition_rows_by_independent_sets` and then produce a list of batch sizes
    (and optionally the row indices per batch) that should be executed
    sequentially. This is useful for the lazy-accumulator sweep which wants
    to update entire independent blocks before flushing or proceeding.

    Parameters
    - couplings: csr_matrix
    - order: currently only 'color' supported (return blocks in color order)
    - return_row_blocks: if True return (batches, row_blocks) else return batches

    Returns
    - batches: list[int] of batch sizes whose sum equals n
    - row_blocks (optional): list[list[int]] of row indices per batch
    """

    blocks, colors = partition_rows_by_independent_sets(couplings)

    # Optionally reorder blocks; for now 'color' is the natural order
    row_blocks = blocks
    batches = [len(b) for b in row_blocks]

    if return_row_blocks:
        return batches, row_blocks
    return batches



def masked_row_dot(couplings: Matrix, D: np.ndarray, pos: int, bs: int) -> float:
    """
    Compute the contribution of the accumulated deltas `D` to the local field
    at row `pos`, i.e. return (J[row=pos] dot D). This models a masked-mxv
    primitive that computes only the scalar contribution required for the
    Metropolis decision at `pos` without materializing the whole h += J.dot(D).

    In Python we use the sparse row matvec; in C++ this should map to a
    fast masked-mxv primitive or a pipeline read for the single scalar.
    """
    # For CSR matrices getrow returns a 1xN sparse matrix; dot(D) returns a 1-elem array
    # we use todense to simulate a fast masked-mxv primitive
    return (couplings.todense()[pos:pos+bs, :].dot(D))


def get_energy(couplings: Matrix, local_fields: Vector, state: State) -> float:
    """
    Compute energy given couplings and local fields.
    """
    # Ensure state is 0/1
    assert np.all((state == 0) | (state == 1)), "State must contain only 0 or 1 values."
    # Ensure shapes are compatible
    assert couplings.shape[0] == couplings.shape[1], "Couplings must be a square matrix."
    assert couplings.shape[0] == state.shape[0], "State and couplings must have compatible shapes."
    assert local_fields.shape[0] == state.shape[0], "Local fields and state must have compatible shapes."
    return state.dot(couplings.dot(state) / 2 + local_fields)


def sequential_sweep_x(couplings: Matrix, local_fields: Vector,
                     state: State, beta: float) -> float:
    """
    Metropolis sweep that preserves sequential Metropolis semantics but avoids
    expensive per-row dense conversions by precomputing the local field
    h = J.dot(state) + local_fields and updating h incrementally when a spin
    flip is accepted. Works with CSR sparse `couplings`.
    """

    n = len(state)

    # Precompute local fields h = J.dot(state) + local_fields. In a lazy
    # evaluation framework you'd typically start a pipeline here that
    # represents h but we materialize once for correctness in the Python
    # prototype.
    h = couplings.dot(state) + local_fields

    # Random numbers (log uniform)
    log_r = np.log(np.random.random(size=n))

    delta_energy = 0.0

    # D is the lazy accumulator vector: it holds pending 0/1 deltas that we
    # haven't yet materialized into h. The lazy evaluator in your framework
    # would instead record these into a pipeline and only execute when a
    # reduction-to-scalar (dot, min, sum) requires it.
    D = np.zeros(n, dtype=h.dtype)

    # iterate sequentially (Metropolis semantics). We allow D to accumulate
    # multiple nonzeros. For each pos we compute the scalar contribution from
    # D to h[pos] via a masked row dot (row J[pos,:] dot D) so we don't have
    # to apply the whole matvec until we need to.

    # Use precomputed batches/row_blocks attached to the matrix when available
    # (compute_row_batches attaches them to the matrix as _row_blocks/_batches)
    if hasattr(couplings, '_row_blocks') and hasattr(couplings, '_batches'):
        row_blocks = getattr(couplings, '_row_blocks')
        batches = getattr(couplings, '_batches')
    else:
        # Compute batches (row blocks) from the matrix sparsity pattern. Each
        # block is an independent set and can be updated without internal
        # dependencies. compute_row_batches returns sizes; request row blocks too.
        batches, row_blocks = compute_row_batches(couplings, return_row_blocks=True)

    # Iterate sequentially over blocks. Inside each block rows are independent
    # so they can be processed in any order (or in parallel) while preserving
    # Metropolis semantics between blocks.
    for block_idx, rows in enumerate(row_blocks):
        # test_states for debugging
        test_states = state.copy()

        # If there are pending deltas compute their effect on these rows only
        # using a masked row dot over the block. masked_row_dot expects a
        # starting pos and block size; we call it for each row in the block.
        #row_contrib = np.asarray([masked_row_dot(couplings, D, r, 1).item() for r in rows])
        # rewritten as a dense matrix vector product for the block to simulate efficent code
        print("rows:",rows)
        row_contrib = couplings[rows, :].dot(D)

        # compute delta energies for the block
        state_slice = np.array(state)[rows]
        h_slice = np.array(h)[rows]
        dn = (2.0 * state_slice - 1.0) * (h_slice + row_contrib)

        # Vectorized Metropolis decision for rows in this block
        accept = (dn >= 0) | (log_r[rows] < beta * dn)
        old = np.array(state)[rows]
        new = np.where(accept, 1 - old, old)
        delta_energy += -np.sum(dn * accept)
        # Update state and D in place
        for idx, r in enumerate(rows):
            state[r] = new[idx]
            D[r] += (new[idx] - old[idx])


    # Flush any remaining accumulated deltas into h before returning. This
    # materializes the lazy pipeline; the framework could do this lazily at
    # a later synchronization point instead.
    # if np.any(D):
    #     apply_accumulated(couplings, D, h)

    return float(delta_energy)


def sequential_sweep_immediate(couplings: Matrix, local_fields: Vector,
                               state: State, beta: float, printinfo: bool = False) -> float:
    """
    Immediate-update Metropolis sweep: on each accepted flip we update the
    local-field vector `h` immediately by iterating the nonzeros of the
    flipped row (neighbor updates). This is the standard efficient approach
    for CSR matrices and is provided here for performance comparison against
    the lazy `D`-accumulator approach.

    For simplicity in this prototype, we convert the sparse matrix to dense internally.
    """
    n = len(state)
    # Convert couplings to dense numpy array
    dense_couplings = couplings.toarray()
    h = dense_couplings.dot(state) + local_fields
    log_r = np.log(np.random.random(size=n))
    delta_energy = 0.0

    # Use same batching mechanism as sequential_sweep_x to improve locality.
    if hasattr(couplings, '_row_blocks') and hasattr(couplings, '_batches'):
        row_blocks = getattr(couplings, '_row_blocks')
        batches = getattr(couplings, '_batches')
    else:
        batches, row_blocks = compute_row_batches(couplings, return_row_blocks=True)

    for block_idx, rows in enumerate(row_blocks):
        # process rows in this independent block (vectorized)
        if printinfo:
            print("rows = ", rows)
        rows = np.array(rows)
        hi = h[rows]
        si = state[rows]
        dn = (2.0 * si - 1.0) * hi

        # Vectorized Metropolis decision
        accept = (dn >= 0) | (log_r[rows] < beta * dn)
        old = si
        new = np.where(accept, 1 - old, old)
        delta = new - old

        # Update state and delta_energy
        state[rows] = new
        delta_energy += -np.sum(dn * accept)

        # Update h for all spins (dense update)
        if np.any(delta):
            h += dense_couplings[:, rows].dot(delta)

    return float(delta_energy)



def pt(states: List[State], energies: List[float], beta_idx: List[int],
       betas: List[float]) -> NoReturn:
    """
  Parallel tempering move.
    states: [n_replicas, ...]  Array of replicas
    energies: [n_replicas] Array of energies of each replica
    beta_idx: [n_replicas] The replica index currently assigned to each beta,
        i.e. inverse temperature K is currently used for simulating replica beta_idx[K]
    betas: [n_replicas] Sequential array of inverse temperatures.

    This function only modifies the order of beta_idx.
  """
    print("pt(in):",energies[beta_idx], "  beta(in):", betas[beta_idx])

    # Get number of replicas
    n_replicas = len(states)

    # Apply PT for each pair of replicas
    for k in range(n_replicas - 1):

        # Get first index
        k1 = n_replicas - k - 1

        # Get second index
        k2 = n_replicas - k - 2

        # Compute delta energy
        de = (energies[beta_idx[k1]] - energies[beta_idx[k2]]) * (betas[k1] - betas[k2])

        # Accept/reject following Metropolis
        if de >= 0 or np.random.random() < np.exp(de):
            beta_idx[k1], beta_idx[k2] = beta_idx[k2], beta_idx[k1]

    print("pt(out):",energies[beta_idx], "  beta(out):", betas[beta_idx])



def simulation_parallel_x(
                        sweep: SweepFunction,
                        couplings: Matrix,
                        local_fields: Vector,
                        states: List[State],
                        energies: List[float],
                        beta_idx: List[int],
                        betas: List[float],
                        n_sweeps: int,
                        get_part_fun: bool = False,
                        use_pt: bool = True) -> Tuple[State, float, int, int]:
    """
  Apply simulation.
  """

    # Get number of replicas
    n_replicas = len(states)

    # Best configuration/energy
    _best_energy = np.copy(energies)
    _best_state = np.copy(states)
    _best_sweeps = np.zeros(n_replicas, dtype=np.int32)
    betas_sorted = np.empty_like(betas)
    log_omegas = np.zeros(n_sweeps)

    # For each run ...
    for s in range(n_sweeps):
        for k in range(n_replicas):
            betas_sorted[beta_idx[k]] = betas[k]
        # ... apply sweep for each replica ...
        # interate k in random order to avoid bias
        #print("betas_sorted: ",betas_sorted)
        perm = np.random.permutation(n_replicas)
        print("perm: ",perm)
        for k in perm:  # numba.prange(n_replicas):

            # Apply sweep
            tmp = sweep(couplings, local_fields,states[k], betas_sorted[k])
            energies[k] += tmp
            print("Replica ",k," energy=",energies[k])

            # Store best state
            if energies[k] < _best_energy[k]:
                _best_energy[k] = energies[k]
                _best_state[k] = np.copy(states[k])
                _best_sweeps[k] = s

        # ... and pt move.
        #print("beta_idx after sweep: ",beta_idx)
        if use_pt:
            pt(states, energies, beta_idx, betas)
        #print("beta_idx after pt:    ",beta_idx)
        # Calculate the weights for the partition function

    # Get lowest energy
    best_pos = np.argmin(_best_energy)
    best_state = _best_state[best_pos]
    best_energy = _best_energy[best_pos]
    best_sweeps = _best_sweeps[best_pos]

    # Return states and energies
    return ((states, energies, beta_idx, log_omegas), (best_state, best_energy,
                                                       best_sweeps, s + 1))



def get_min_energy(couplings: Matrix, local_fields: Vector):

    # Get number of variables
    n_vars = couplings.shape[0]
    # assert square matrix
    assert couplings.shape[0] == couplings.shape[1], "Couplings must be a square matrix."
    # assert compatible shapes
    assert local_fields.shape[0] == n_vars, "Local fields must have the same number of variables as couplings."
    min_energy = np.inf
    best_state = np.array([0]*n_vars, dtype=dtype)

    # Find minimum energy by bruteforce
    for state in range(2**n_vars):
        # Transform state
        spin_state_unsigned = np.array([int(x) for x in bin(state)[2:].zfill(n_vars)], dtype=dtype)
        # Get energy for the state
        assert np.all(np.isin(spin_state_unsigned, [0, 1])), "State contains values other than 0 and 1"
        energy = get_energy(couplings, local_fields, spin_state_unsigned)
        # Store only the minimum energy
        if energy < min_energy:
            min_energy = energy
            best_state = np.copy(spin_state_unsigned)

    return min_energy, best_state


def gen_random_problem(n_vars: int,
                       dtype: Any = 'float', nzratio = 0.1, test_dense: bool = False, printinfo: bool = False) -> Tuple[Matrix, Vector]:

    # Generate random problem
    if (test_dense):
        couplings = 2 * np.random.random((n_vars, n_vars)).astype(dtype) - 1
        couplings = (couplings + couplings.T) / 2
        vals = couplings.flatten()
        row = np.array([i for i in range(n_vars) for j in range(n_vars)])
        col = np.array([j for i in range(n_vars) for j in range(n_vars)])
    else:
        # couplings are random sparse matrix instead of dense with nz none zero elements
        nz = int(nzratio * n_vars * n_vars)
        row = np.sort(np.random.randint(0, n_vars, nz))
        col = np.sort(np.random.randint(0, n_vars, nz))
        # make sure there are no duplicate entries ins same i,j pairs in (row,col)
        unique = np.unique(np.array([row, col]).T)
        row = row[unique]
        col = col[unique]
        vals = 2 * np.random.random(len(unique)).astype(dtype) - 1

    couplings = csr_matrix((vals, (row, col)), shape=(n_vars, n_vars))
    couplings = (couplings + couplings.T) / 2
    diag_couplings = couplings.diagonal()
    #set diagonal to zero
    couplings = couplings - csr_matrix((diag_couplings, (np.arange(n_vars), np.arange(n_vars))), shape=(n_vars, n_vars))

    if printinfo:
        # print sparse matrix structure
        # ie
        # 0 0 0 1 0
        # 0 0 1 0 0
        # 0 1 0 0 0
        # print actual value for the unit test
        print("Couplings matrix (indices and nonzero values in COO):")
        print("Row indices:", couplings.nonzero()[0])
        print("Column indices:", couplings.nonzero()[1])
        print("Nonzero values:", couplings.data)

        print("Couplings matrix structure (*=nonzero, .=zero):")
        dense_couplings = couplings.toarray()
        for i in range(n_vars):
            row_str = ""
            for j in range(n_vars):
                if dense_couplings[i,j] != 0:
                    row_str += "* "
                else:
                    row_str += ". "
            print("[{}]".format(i),"\t",row_str)

    # Split in couplings and local_fields
    #local_fields = np.copy(np.diagonal(couplings))
    #make local_fields random instead of from diagonal
    local_fields = 2 * np.random.random(n_vars).astype(dtype) - 1
    #print local_fields
    if printinfo:
        print("Local fields (random):", local_fields)

    return couplings, local_fields


def test_sequential_sweep_simulation_qubo(n_vars: int):

    n_replicas = 3
    print("n_vars =",n_vars)
    print("n_replicas =",n_replicas)


    # Generate random problem
    couplings, local_fields = gen_random_problem(n_vars, dtype=dtype,printinfo=True)

    # Find minimum energy by bruteforce
    min_energy,best_state_bruteforce = get_min_energy(couplings, local_fields)

    # Fix temperature
    betas = np.array([10]*n_replicas, dtype=dtype)
    print("Betas =",betas)
    beta_idx = np.arange(n_replicas)
    print("Initial beta_idx =",beta_idx)
    # Get initial state
    states = np.random.randint(2, size=(n_replicas, n_vars)).astype(dtype)
    print("Initial states =")
    for s in states:
        print(s)

    # Compute energies
    for s in states:
        assert np.all(np.isin(s, [0, 1])), "State contains values other than 0 and 1"
    energies = np.array(
        [get_energy(couplings, local_fields, state) for state in states])
    print("Initial energies=",energies)

    # Simulate
    print("beta_idx =",beta_idx)
    nsweeps = 2
    (state, energy, _, _), (best_state, best_energy, _, _) = simulation_parallel_x(
        sequential_sweep_immediate, 
        couplings, 
        local_fields,
        states, 
        energies, 
        beta_idx, 
        betas, 
        nsweeps)

    # Check that best energy is correct
    #ref_best_energy = -7.9322789708332255 # dense
    ref_best_energy = -5.079571790854985 # dense
    ref_nsweeps = 2
    ref_n_vars = 16
    ref_random_seed = 8
    # make sure parameters match reference
    assert n_vars == ref_n_vars, f"n_vars {n_vars} does not match reference {ref_n_vars}"
    assert nsweeps == ref_nsweeps, f"nsweeps {nsweeps} does not match reference {ref_nsweeps}"  
    assert random_seed == ref_random_seed, f"random_seed {random_seed} does not match reference {ref_random_seed}"
    # best_energy 
    assert np.isclose(best_energy, ref_best_energy), f"best_energy {best_energy} does not match reference {ref_best_energy}"

    assert np.all(np.isin(best_state, [0, 1])), "State contains values other than 0 and 1"
    qubo_energy = get_energy(couplings, local_fields, best_state)

    print("beta_idx(out) =",beta_idx)

    print("energy      =",energy)
    for e in energy:
        print(e)
    print("best_energy by solver     = ",best_energy)
    print("best_energy by bruteforce = ",min_energy)

    print("state(out)      =")
    for s in state:
        print(s)
    print("best state by solver     = ",best_state)
    print("best state by bruteforce = ",best_state_bruteforce)


    assert (np.isclose(best_energy,
                       get_energy(couplings, local_fields, best_state)))

    # Best energy should be always larger than the minimum energy
    assert (np.round(best_energy, 6) >= np.round(min_energy, 6))
    #assert (np.isclose(qubo_energy, ref_energy))



# ## test semantics of row-wise vs immediate update sweeps
# def test_rowwise_vs_immediate_equivalence(
#     nzratio=0.25,
#     seed0 = 12345,
#     seed1 = 20241010,
#     n_vars = 815,
#     beta = 10.0):
#     """
#     Cross-check that a row-wise sequential sweep implemented via
#     `sequential_sweep_rowwise` (with an `update_spin` that follows the
#     immediate-neighbor-update semantics) produces the same delta_energy and
#     final state as `sequential_sweep_immediate` when both use the same RNG
#     seed.
#     """
#     def sequential_sweep_rowwise(couplings: Matrix, local_fields: Vector,
#                         state: State, beta: float) -> float:
#         """
#     Metropolis update.
#     """
#         def update_spin(couplings: Matrix, local_fields: Vector, state: State, pos: int, beta: float, log_r: float) -> float:
#             """
#             Update spin accordingly to Metropolis update.
#             """
#             # Ensure state is 0/1
#             assert np.all((state == 0) | (state == 1)), "State must contain only 0 or 1 values."
#             # Ensure pos is valid
#             assert 0 <= pos < state.shape[0], "pos index out of bounds."
#             # Ensure shapes are compatible
#             assert couplings.shape[0] == couplings.shape[1], "Couplings must be a square matrix."
#             assert couplings.shape[0] == state.shape[0], "State and couplings must have compatible shapes."
#             assert local_fields.shape[0] == state.shape[0], "Local fields and state must have compatible shapes."

#             # Get the negate delta energy (qubo)
#             # delta_n_energy = (2. * state[pos] - 1.) * (couplings[pos].dot(state) + local_fields[pos]) 
#             # # we need to rewrite this for sparse matrix couplings, ie couplings[pos] for dense matrix becomes couplings.getrow(pos).toarray()[0]
#             delta_n_energy = (2. * state[pos] - 1.) * (couplings.getrow(pos).toarray()[0].dot(state) + local_fields[pos])

#             # Metropolis update
#             if delta_n_energy >= 0 or log_r < beta * delta_n_energy:
#                 # Update spin (qubo)
#                 state[pos] = 0 if state[pos] else 1
#                 # Return delta energy
#                 return -delta_n_energy
#             else:
#                 # Otherwise, return no change in energy
#                 return 0.

#         # Get random numbers
#         log_r = np.log(np.random.random(size=len(state)))

#         # Try to update every spin
#         delta_energy = 0.
#         for pos in range(len(state)):
#             delta_energy += update_spin(couplings, local_fields, state, pos, beta,log_r[pos])

#         return delta_energy

#     np.random.seed(seed0)
#     couplings, local_fields = gen_random_problem(n_vars, dtype=dtype, nzratio=nzratio)


#     # initial state
#     init_state = np.random.randint(2, size=n_vars).astype(dtype)

#     # copy for both runs
#     state_immediate = init_state.copy()
#     state_rowwise = init_state.copy()

#     np.random.seed(seed1)
#     delta_immediate = sequential_sweep_immediate(couplings, local_fields, state_immediate, beta)

#     # run rowwise with the same RNG sequence
#     np.random.seed(seed1)
#     delta_rowwise = sequential_sweep_rowwise(couplings, local_fields, state_rowwise, beta)

#     # Compare energies and final states
#     assert np.isclose(delta_immediate, delta_rowwise), f"delta_energy differs: immediate={delta_immediate}, rowwise={delta_rowwise}"
#     assert np.array_equal(state_immediate, state_rowwise), f"states differ after sweep: immediate={state_immediate}, rowwise={state_rowwise}"
#     print("Rowwise and Immediate sweeps are equivalent.")

# for i in range(5):
#     test_rowwise_vs_immediate_equivalence(seed1=i)

# run several time to check parallelism
for _ in range(4):
    np.random.seed(random_seed)
    test_sequential_sweep_simulation_qubo(n_vars=16)
