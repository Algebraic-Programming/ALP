
Version 0.7.0
=============

This is a summary of changes. For full details, see the publicly available Git
history prior to the v0.7 tag.

Highlights:

 1. This release re-implements the nonblocking ALP/GraphBLAS backend by Mastoras
    et al. (GrAPL/IPDPSW '22, TACO '23) on the latest ALP code base. The use of
    the nonblocking backend for some algorithms results in multiple-factor
    speedups versus standard blocking execution as well as versus external
    industry-standard frameworks. This includes Eigen, which, like nonblocking
    ALP/GraphBLAS, perform cross-operation fusion. Simply compile your ALP
    programs using `grbcxx -b nonblocking`, and enjoy the speedups!

 2. We also introduce a new programming interface to the ALP software stack that
    allows vertex-centric programming in addition to programming using
    generalised sparse linear algebra. This new interface, ALP/Pregel,
    translates vertex-centric programs to standard ALP/GraphBLAS primitives
    during compilation, and thus benefits of all automatic optimisations
    included with the ALP software stack.

 3. Support for software prefetching during `vxm` and `mxv` has been added to
    the `reference` and `reference_omp` backends. Since optimal prefetch
    settings and its overall effectiveness relies strongly on 1) the structure
    of the sparse matrices and graphs considered as well as on 2) the algorithms
    used on those data, this new feature is turned off by default. To use it,
    please enable it via `include/graphblas/reference/config.hpp` and tune the
    there-defined prefetch distances.

 4. Finally, this release includes another new backend, the `hyperdags` backend.
    A program compiled with this backend will, after execution, dump a HyperDAG
    representation of the ALP computations that the program executed.

Changes to the specification:

 1. Any ALP primitive with ALP container output now takes a Phase argument.

 2. Clarify that the use of the `dense` descriptor also implies that the output
    containers on entry must be dense. This applies also for out-of-place
    primitives.

Algorithms:
 - [new] a vertex-centric PageRank-like algorithm implemented on top of the new
   ALP/Pregel has been added;
 - [new] a vertex-centric algorithm for strongly connected components on
   undirected graphs implemented on top of ALP/Pregel has been added;
 - [new] the algebraic k-core decomposition algorithm by Li et al. (HPEC '21)
   has been added;
 - [bug] the mpv algorithm performed one too many iterations, while all
   associated tests used an ALP/GraphBLAS baseline-- v0.7 now instead verifies
   against external ground truths;
 - [bug] the label propagation algorithm relied on a bugged implementation of
   `grb::set`, now fixed, while it now and when possible relies on `std::swap`
   instead of performing explicit and expensive copies;
 - [bug] the CG algorithm returned `SUCCESS` even it failed to converge within
   the given number of maximum iterations.

Operators:
 - [new] v0.7 (re-)introduces the four less-than(-or-equal) and
   greater-than(-or-equal) operators;

All backends:
 - [bug] fixed the behaviour of ALP containers under copy-assignment and
   copy-construction;
 - [bug] all variants of `foldl` and `foldr` previously could erroneously return
   `ILLEGAL` in the presence of sparse vectors and/or masks;
 - [bug] several primitives would not return `ILLEGAL` in the presence of the
   `dense` descriptor when faced with sparse containers;
 - [bug] all backends missed the implementation of at least one `eWiseMul`
   variant;
 - [bug] all backends missed the implementation of at least two `eWiseApply`
   variants where both inputs are scalar;
 - [feature] improved `_DEBUG` tracing and code style throughout.

Reference and reference_omp backends:
 - [bug] overlap detection of the output and output mask was erroneously
   disabled for the `vxm` and `mxv` primitives, herewith fixed;
 - [bug] `foldl` and `foldr` previously have employed unexpected casting
   behaviour;
 - [bug] multiple copy-assignment of the same vector could fail;
 - [bug] the vector<-scalar<-vector `eWiseApply` using operators was in-place;
 - [bug] the `eWiseApply` using sparse vector inputs and/or masks could in some
   rare cases depending on structure and vector lengths generate incorrect
   output;
 - [bug] the implementation of the vector `grb::set` where the output container
   was not already dense was in-place, while out-of-place semantics were
   defined;
 - [bug] the output-masked `eWiseMul` was bugged in the case where one of the
   inputs was scalar;
 - [bug] matrix containers with initial requested capacity zero could attempt
   to access uninitialised memory, including even after a successful subsequent
   `resize`;
 - [performance] `foldl` and `foldr` using sparse vectors and/or masks were
   previously not always following asymptotically optimal behaviour;
 - [performance] `set` previously did not exploit information such as whether
   the `dense` descriptor was present, whether vectors need only touch
   coordinate data to generate correct output, or whether it never needs to
   touch coordinate data;
 - [performance] `eWiseApply` detects more cases of trivial operations on empty
   vectors, and completes those faster;
 - [performance] optimised `eWiseMul` with scalar inputs.

BSP1D and hybrid backends:
 - [bug] the output-masked `vxm` and various `foldl` and `foldr` were missing;
 - [bug] copy-assignment operator for vectors was missing.

Testing, development, and documentation:
 - the unit test suite has been hardened to detect all aforementioned bugs;
 - outdated documentation was revised-- in particular, all user-facing
   documentation has been checked and can now be generated via the new make
   target `make userdocs`;
 - developer documentation is now built via `make devdocs`, while the older
   `make docs` target now builds both the user and developer documentation;
 - new developers can now enjoy an updated developer guide;
 - the test suite now prints an error when the automatic detection of the number
   of sockets fails, and then auto-selects one instead of zero (which caused the
   test scripts to fail);
 - added performance tests for the sparse matrix--vector, sparse matrix--sparse
   vector, and sparse matrix--sparse matrix multiplication kernels;
 - improved both the GitHub and internal CI scripts.


Version 0.6.0
=============

This is a summary of changes. For full details, see the publicly available Git
history prior to the v0.6 tag.

Highlights and changes to the specification:
 - Deprecated `grb::init` and `grb::finalize` in favour of grb::Launcher.
   Existing code should migrate to using the Launcher as any later release may
   remove the now-deprecated primitives.
 - If you wish to rely on ALP/GraphBLAS for more standard sparse linear
   algebra but if you cannot, or do not wish to, adapt your existing sources
   to the C++ ALP/GraphBLAS API, then v0.6 onwards generates libraries that
   implement a subset of the standard C/C++ SparseBLAS and SpBLAS interfaces.
   After installation, these libraries are found in
     - `<install path>/lib/sequential/libsparseblas.a` (sequential) and
     - `<install path>/lib/sequential/libsparseblas_omp.a` (shared-memory
       parallel).
   The headers are found in
     - `<install path>/include/transition/sparseblas.h` and
     - `<install path>/include/transition/spblas.h`.
 - Input iterators passed to `grb::buildMatrixUnique` that happen to be random
   access will now lead to shared-memory parallel ingestion when using the
   reference_omp or hybrid backends.
 - `grb::Phase` is now only accepted for primitives with non-scalar
   ALP/GraphBLAS output containers.

Algorithms:
 - Feature: the CG algorithm has been adapted to work with complex-valued
   matrices making use of the standard `std::complex` type. A corresponding
   smoke test is added.
 - Bugfix: BiCGstab erroneously relied on `grb::utils::equals`, and could
   (rarely) lead to false orthogonality detection and an unnecessary abort.

Utilities:
 - The parser that reads MatrixMarket files, `grb::utils::MatrixFileReader`, now
   can parse complex values and load Hermitian matrices.
 - What constitutes an ALP/GraphBLAS sparse matrix iterator has been formalised
   with a novel type trait, `grb::utils::is_alp_matrix_iterator`. ALP/GraphBLAS
   matrix output iterators now also adhere to these requirements.
 - A `grb::utils::is_complex` type trait has been added, and is used by the CG
   algorithm so as to not materialise unnecessary buffers and code paths.
 - Bugfixes to the `grb::utils::equals` routine, as well as better
   documentation. A unit test has been added for it.

Testing, development, and documentation:
 - Documentation has been adapted to include GitHub for reporting issues.
 - Documentation of various ALP/GraphBLAS primitives and concepts have been
   improved.
 - Documentation detailing the compiler warning suppressions and their rationale
   have been moved to the code repository at `docs/Suppressions.md`.
 - Add basic CI tests (build, install, and smoke tests) for GitHub.
 - More thorough testing of output matrix iterators, input iterators, and of
   `grb::buildMatrixUnique`.
 - The `dense_spmv.cpp` smoke test did not correctly verify output, and could
   fail for SpMV multiplications that yield very small nonzero values.
 - Improvements to various tests and scripts.

Reference and reference_omp backends:
 - Bugfix: matrix output iterators failed if all nonzeroes were on the last row
   and no nonzeroes existed anywhere else.
 - Bugfix: copying and immediately dereferencing a matrix output iterator led to
   use of uninitialised values.
 - Bugfix: `grb::foldl` with a dense descriptor would accept sparse inputs while
   it should return `ILLEGAL`. This behaviour, as well as for other error codes,
   are now also (unit) tested for, including with masks and inverted masks.
 - Bugfix: `grb::set` was moved from `reference/blas1.hpp` to
   `reference/io.hpp`, but the macros that guard parallelisation were not
   properly updated.
 - Bugfix: the OpenMP `schedule( static, chunk_size )` has a dynamic (run-time)
   component that was not intended.
 - Bugfix: some OpenMP `schedule( dynamic, chunk_size )` operate on regular
   loops and should employ a static schedule instead.

BSP1D backend:
 - Bugfix: too thorough sanity checking disallowed building dense matrices.
 - Bugfix: `grb::set` on vectors with non-fundamental value types would not
   compile (due to code handling the use_index descriptor).
 - Bugfix: `grb::clear` could leave the vector in an undefined state if it
   immediately followed an operation that left said vector dense.
 - Code improvement: PinnedVector constructors now throws exceptions on errors.

All backends:
 - Bugfix: an input-masked variant of `grb::foldr` was missing. These are now
   also added to the unit test suite.
 - Bugfix: matrix constructors that throw an exception could segfault on
   destruction.
 - Bugfix: use of PinnedVectors that pin sparse or empty vectors could segfault.
 - Code improvements: noexcept additions, const-correctness, code style fixes,
   removal of compiler warnings (on some compiler versions), dead code removal,
   improved `_DEBUG` tracing, additional debug-mode sanity checks, and reduced
   code duplication.


Version 0.5.0
=============

Only changes that may affect end-user codes are summarised. For full details,
see the publicly available Git history before the v0.5 tag.

New features and specification changes:
 - New feature: all ALP/GraphBLAS containers now expose their capacities. The
   default capacity is the minimum of the container dimension(s) and may be
   overridden during construction; e.g., to construct a vector of size `n>>1`
   which will hold only one nonzero, `grb::Vector< SomeType > x( n, 1 )` may
   now be used. Capacities can be resized through `grb::resize` which already
   existed for matrices, but now also is specified for vectors. Current
   capacities may be inspected via the newly introduced `grb::capacity`.
   Backends shall guarantee *at least* the requested capacity is made available;
   if not possible, container construction or resizing shall fail. All current
   backends implement these new primitives.
 - Spec change: apart from work, number of operator applications, intra-process
   data movement, inter-process data movement, inter-process synchronisations,
   and whether system calls may be made, backends must now also specify the
   memory storage requirements of containers and internal buffers.
 - New feature: re-worked `grb::Phase` to now define two phases `RESIZE` and
   `EXECUTE`, instead of the previous more classic `SYMBOLIC` and `NUMERICAL`.
   This change 1) allows backends to implement a wider range of single-stage and
   two-stage approaches without the interface implying the classical two-stage
   approach, and 2) makes it clear that only the `RESIZE` stage is allowed to
   increase (and never decrease) the capacities of output containers.
 - Spec change: all primitives with ALP/GraphBLAS container output(s) now
   require a `grb::Phase` argument. The default is `EXECUTE`. This notably
   includes primitives with vector outputs. Note that if default vector capacities
   are used, however, existing code can remain unchanged.
 - Spec change: executing a primitive in `EXECUTE` (previously `NUMERICAL`)
   mode while the output container did not have sufficient capacity, now results
   in that primitive returning `FAILED` and the output containers being cleared.
 - All current algorithms implicitly assumed vectors have capacities equal to
   their size. Algorithms for which the assumption *must* hold have been
   modified to check the assumption holds through the new `grb::capacity`. Note
   that if relying on default vector capacities this assumption automatically
   holds.
 - Spec change: `grb::PinnedVector` can now iterate over sparse vectors in
   `Theta(nz)` time instead of `Theta(n)`, as originally intended. All backends
   now implement the updated specification, while all code that relied on the
   PinnedVector has been updated.
 - New feature: non-empty ALP/GraphBLAS containers are now assigned a unique
   identifier that may be retrieved via `grb::getID`. For deterministic
   programs, this identifier shall be --and for all current backends indeed is--
   consistent across different runs.
 - Deprecated `grb::{init,finalize}` in favour of `grb::Launcher`.
 - Deprecated `grb::{eWiseAdd,eWiseMulAdd}` in favour of `grb::foldl` and
   `grb::foldr` using (additive) monoids in both cases, followed by `eWiseMul`
   for the `eWiseMulAdd`. We recommend using a nonblocking backend to effect the
   same fusion opportunities otherwise lost with the deprecated primitives.
 - Spec change: `grb::wait` was specified for nonblocking backends.
 - Bugfix: one `grb::mxm` variant had the additive monoid and multiplicative
   operator arguments switched. More level-3 primitives will be added with the
   next release, which will complete their specification.

Algorithms:
 - BiCGstab had been requested and has been implemented. A smoke test has been
   added that verifies the algorithm's correctness on the `gyro_m` matrix. The
   algorithm location is `include/graphblas/algorithms/bicgstab.hpp`.
 - The CG algorithm has been updated to support the `no_casting` descriptor, and
   has been updated to, if given sparse initial guesses or sparse right-hand
   sides, to ensure that vectors used during the performance-critical section
   nonetheless become dense. This, in turn, allows for the addition of the
   `dense` descriptor to all performance-critical primitives, thus further
   increasing performance.
 - k-nearest-neighbours (kNN) computation requires only one buffer, while
   previously two were used.
 - The HPCG benchmark is now available as an ALP/GraphBLAS algorithm, rather
   than as a test.
 - Bugfix: label propagation's convergence detection was broken due to an
   erroneous adaptation to in-place semantics. The corresponding smoke test did
   not properly check for the correct number of iterations. Both issues are
   fixed.

Testing and development:
 - Pushed our gitlab CI configuration into the repository in hopes it is useful
   for others. Docker images for LPF-based tests may appear in future -- at
   current, only CI testing without LPF is enabled.
 - Most code style issues due to a clang-format misadventure are resolved with
   this release.
 - Documentation on the project directory structure has been added in
   `docs/Directory_structure.md`, which `README.md` now points to.
 - Bugfix: `tests/utils/output_verification.hpp` now properly handles NaNs.

Build system:
 - We now support arbitrary choices for a build directory. To this end, the
   older `configure` script has been removed and replaced with a `bootstrap.sh`
   script. The latter takes the same arguments as the former. On how to use the
   new `bootstrap.sh` script, please see the updated `README.md`.
   Note that with this change, the ALP build system is now equivalent to that of
   LPF.

BSP1D and hybrid backends:
 - Clearer instructions on how to run the test suite when relying on other MPI
   implementations than MPICH were added to `README.md`.
 - Bugfix: when used with an MPI-based LPF engine (which is the default), and
   if the underlying MPI implementation was OpenMPI, then (commented) commands
   in the `tests/pase_env.sh` and the `tests/performance/performancetests.sh`
   did not work with the latest versions of OpenMPI, now fixed.
 - Bugfix: should forward-declare internal getters that were previously first
   declared as part of BSP1D friend declarations. Curiously, many compilers
   accepted the previous erroneous code.
 - Bugfix: empty BSP1D containers could previously leave process-local matrices
   uninitialised.

Reference and reference_omp backends:
 - Bugfix: matrix construction did not use the `alloc.hpp` mechanisms. This
   had negative impact on the correct handling of out-of-memory errors, on
   NUMA-aware allocation, and on code maintainability.

All backends:
 - Bugfix: `grb::Launcher` (as well as the benchmarker) did not always properly
   finalise the ALP/GraphBLAS context after exec completed. This caused some
   memory to not be properly freed on program exits.
 - Bugfix: the out-of-place versions of `grb::operators::{argmin,argmax}` were
   incorrect. All code within the repository was unaffected by this bug. The
   corresponding unit tests were updated to also test out-of-place application.

Various other bugfixes, performance bug fixes, documentation improvements,
default configuration updates, and code structure improvements -- see Gitee MRs
!21, !22, !38, !39, !40, and !42 for details.


Version 0.4.1
=============

 - The CG algorithm assumed out-of-place behaviour of grb::dot, while the
   specification since v0.1 defines it to be in-place. Implementations of
   grb::dot were erroneously out-of-place until v0.4, but the CG algorithm
   was erroneously not updated. This hotfix rectifies this.


Version 0.4.0
=============

Reference and reference_omp backends:
 - Removed memory allocations in eWiseApply on matrices; closes Gitee issue #4.
 - Fixed issue where some exotic operators could lead to uninitialised values.
 - Fixed bug in copy-constructor of grb::Matrix.
 - Fixed issues where Vector::operator[] was unnecessarily used.
 - Fixed slow clearing of vectors when there are many threads but few elements.
 - Fixed slow initialisation of empty vectors for faster hybrid performance.

BSP1D and hybrid backends:
 - Refactored the polyalgorithm for combining and synchronising vectors. The
   implementation is now back in line with the paper.
 - Fixed issue where SpMV with a dense descriptor could fail if the process-
   local submatrices happened to be hypersparse.
 - Fixed issue where grb::Matrix copy-constructor did not exist.
 - Fixed issue where internal collectives did not guarantee sufficient buffer.
 - Fixed algorithm verification issues due to races on stdout.
 - Fixed issue where vector iterator construction could fail in debug mode.
 - Fixed issue where collectives could fail due to an erroneous assertion.

All backends and some algorithms:
 - Fixed issue where grb::dot erroneously remained out-of-place after the
   specification since v0.1 has all operations except apply (and variants)
   defined as in-place.

Utilities:
 - Fixed issue where hpparser could segfault in unit test mode.

Others:
 - Switch to CMake as the build infrastructure.
 - Consistently have code and documentation refer to itself as ALP/GraphBLAS.
 - Cleaner separation of test categories: unit, smoke, and performance.
 - Now always compiles and runs unit tests both in release and debug modes.
 - Refactored some tests to become backend-agnostic.
 - Improved test coverage overall.
 - Fixed issue where hook-compatible tests may not always propagate errors.
 - Introduced targeted suppressions of GCC compiler warnings. While tested
   on various GCC versions, some warnings may yet appear for others during
   compilation of the library, tests, and / or programs. If so, please do
   report them so we can add them to the suppression list.
 - Code style fixes, dead code removal, and improved logging.


Version 0.3.0
=============

Reference and reference_omp backends:
 - Fixed issue where grb::set, grb::vxm, and grb::mxv could fail for more
   exotic data types.
 - Fixed issue that prevented std::move on matrices, both from assignment and
   construction.
 - Optimised masked grb::set to now reach optimal complexity in all cases.
 - Optimised grb::eWiseLambda over matrices to avoid atomics.

BSP1D backend:
 - Fixed issue where iterating over empty matrices could fail in the BSP1D
   backend.
 - Fixed issue in BSP1D backend that caused dynamic allocations where they were
   not allowed.
 - Fixed issue where the automatic-mode launcher and benchmarker could, in rare
   cases, fail.
 - Fixed issue where, under rare conditions, the stack-based combine could fail.
 - Fixed performance bug in the BSP1D backend causing spurious calls to
   lpf_sync.

Level-3 functionality, all backends:
 - Fixed issue where a masked set-to-value on matrices would fail.
 - Fixed issue where mxm could work with uninitialised values when more exotic
   semirings are used.
 - Fixed issue that prevented std::move on matrices, both from assignment and
   construction.
 - New level-3 function: eWiseApply.

(Note that the interface of level-3 functionality remains experimental.)

Algorithms and utilities:
 - Fixed issue where MatrixFileReader would store uninitialised values when
   reading pattern matrices.
 - Updated the sparse neural network inference algorithm.
 - New algorithm added: spy.

Others:
 - Fixed issue where a `make clean` would miss some object files.
 - Added new unit and performance tests, including those for detecting the
   above-described bug fixes and added functionality.
 - Documentation update in line with the upcoming revision of the C++ GraphBLAS
   paper.
 - Added some missing documentation.
 - Code style fixes and some dead code removal.


Version 0.2.0
=============

Fix some issues in the Banshee backend that appeared after refactoring for the
0.1.0 release.

Removes --deps option from ./configure as it was no longer used.

