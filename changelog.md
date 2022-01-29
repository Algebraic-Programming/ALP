
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

