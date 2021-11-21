
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

Fix some issues in the Banshee backend that appeared after refactoring for the 0.1.0 release

Removes --deps option from ./configure as it was no longer used

