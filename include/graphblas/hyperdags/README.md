
This backend gathers meta-data while user programs execute. The actual compute
logic is executed by a compile-time selected secondary backend, which by default
is the `reference` backend. The meta-data will be used to generate, at program
exit, a HyperDAG representation of the executed computation. We foresee two
possible HyperDAG representations:

 1. a coarse-grain representation where vertices correspond to a) source
    containers (vectors or matrices-- not scalars), b) output containers, or
    c) ALP/GraphBLAS primitives (such as grb::mxv or grb::dot). Hyperedges
    capture which vertices act as source to operations or outputs in other
    vertices. Each hyperedge has exactly one source vertex only.

 2. a fine-grain representation where source vertices correspond to nonzeroes
    in a source container, not the container as a whole, and likewise for output
    vertices that correspond to individual elements of output containers. Also
    there are now many fine-grained operation vertices that are executed by a
    single ALP/GraphBLAS primitive. For example, a call to grb::vxm will emit
    two hyperedges for every nonzero in the sparse input matrix.

