
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
   unitialised.

Reference and reference_omp backends:
 - Bugfix: matrix construction did not use the `alloc.hpp` mechanisms. This
   had negative impact on the correct handling of out-of-memory errors, on
   NUMA-aware allocation, and on code maintainability.

All backends:
 - Bugfix: `grb::Launcher` (as well as the benchmarker) did not always properly
   finalize the ALP/GraphBLAS context after exec completed. This caused some
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
   was errouneously not updated. This hotfix rectifies this.


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
 - Fixed issue where grb::set, grb::vxm, and grb::mxv could fail for more exotic data types.
 - Fixed issue that prevented std::move on matrices, both from assignment and construction.
 - Optimised masked grb::set to now reach optimal complexity in all cases.
 - Optimised grb::eWiseLambda over matrices to avoid atomics.

BSP1D backend:
 - Fixed issue where iterating over empty matrices could fail in the BSP1D backend.
 - Fixed issue in BSP1D backend that caused dynamic allocations where they were not allowed.
 - Fixed issue where the automatic-mode launcher and benchmarker could, in rare cases, fail.
 - Fixed issue where, under rare conditions, the stack-based combine could fail.
 - Fixed performance bug in the BSP1D backend causing spurious calls to lpf_sync.

Level-3 functionality, all backends:
 - Fixed issue where a masked set-to-value on matrices would fail.
 - Fixed issue where mxm could work with unitialised values when more exotic semirings are used.
 - Fixed issue that prevented std::move on matrices, both from assignment and construction.
 - New level-3 function: eWiseApply.

(Note that the interface of level-3 functionality remains experimental.)

Algorithms and utilities:
 - Fixed issue where MatrixFileReader would store unitialised values when reading pattern matrices.
 - Updated the sparse neural network inference algorithm.
 - New algorithm added: spy.

Others:
 - Fixed issue where a `make clean` would miss some object files.
 - Added new unit and performance tests, including those for detecting the above-described bug
   fixes and added functionality.
 - Documentation update in line with the upcoming revision of the C++ GraphBLAS paper.
 - Added some missing documentation.
 - Code style fixes and some dead code removal.


Version 0.2.0
=============

Fix some issues in the Banshee backend that appeared after refactoring for the 0.1.0 release.

Removes --deps option from ./configure as it was no longer used.

