
// Auto-generated hardware parameters for INTEL-WS
// Allocation policies: close, spread
// Base levels: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
// Generated ALL subset combinations (16 total)
// Generated on 2025-10-07 15:55:48

#ifndef HW_PARAMS_INTEL_WS_HPP
#define HW_PARAMS_INTEL_WS_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations and level subsets
const std::vector<cost_models::HW_model::HWParameters> hw_models_vector = {

    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {4.361676e-11},
    /* ls           */ {3.120099e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.428106e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 3.120099e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.584451e-11, 4.361676e-11},
    /* ls           */ {6.324624e-10, 3.120099e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.824054e-11, 4.361676e-11},
    /* ls           */ {1.787022e-09, 3.120099e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {4.361676e-11, 2.503395e-07},
    /* ls           */ {3.120099e-09, 2.384186e-07},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.428106e-11, 1.584451e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 6.324624e-10, 3.120099e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.428106e-11, 2.824054e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 1.787022e-09, 3.120099e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.428106e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.584451e-11, 2.824054e-11, 4.361676e-11},
    /* ls           */ {6.324624e-10, 1.787022e-09, 3.120099e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.584451e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {6.324624e-10, 3.120099e-09, 2.384186e-07},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.428106e-11, 1.584451e-11, 2.824054e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 6.324624e-10, 1.787022e-09, 3.120099e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.428106e-11, 1.584451e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 6.324624e-10, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.428106e-11, 2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.584451e-11, 2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {6.324624e-10, 1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.428106e-11, 1.584451e-11, 2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 6.324624e-10, 1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {3.805493e-11},
    /* ls           */ {2.872106e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.208603e-12, 3.805493e-11},
    /* ls           */ {3.388982e-10, 2.872106e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.379596e-11, 3.805493e-11},
    /* ls           */ {7.385124e-10, 2.872106e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.331419e-11, 3.805493e-11},
    /* ls           */ {1.590890e-09, 2.872106e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {3.805493e-11, 5.114079e-06},
    /* ls           */ {2.872106e-09, 5.424022e-06},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.208603e-12, 1.379596e-11, 3.805493e-11},
    /* ls           */ {3.388982e-10, 7.385124e-10, 2.872106e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.208603e-12, 2.331419e-11, 3.805493e-11},
    /* ls           */ {3.388982e-10, 1.590890e-09, 2.872106e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.208603e-12, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.379596e-11, 2.331419e-11, 3.805493e-11},
    /* ls           */ {7.385124e-10, 1.590890e-09, 2.872106e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.379596e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {7.385124e-10, 2.872106e-09, 5.424022e-06},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.208603e-12, 1.379596e-11, 2.331419e-11, 3.805493e-11},
    /* ls           */ {3.388982e-10, 7.385124e-10, 1.590890e-09, 2.872106e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.208603e-12, 1.379596e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 7.385124e-10, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.208603e-12, 2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.379596e-11, 2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {7.385124e-10, 1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.208603e-12, 1.379596e-11, 2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 7.385124e-10, 1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {2.625016e-11},
    /* ls           */ {1.895809e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.531737e-12, 2.625016e-11},
    /* ls           */ {1.845807e-10, 1.895809e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.800376e-12, 2.625016e-11},
    /* ls           */ {4.616326e-10, 1.895809e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.296028e-11, 2.625016e-11},
    /* ls           */ {9.143188e-10, 1.895809e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {2.625016e-11, 6.890297e-06},
    /* ls           */ {1.895809e-09, 7.736683e-06},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.531737e-12, 7.800376e-12, 2.625016e-11},
    /* ls           */ {1.845807e-10, 4.616326e-10, 1.895809e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.531737e-12, 1.296028e-11, 2.625016e-11},
    /* ls           */ {1.845807e-10, 9.143188e-10, 1.895809e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.531737e-12, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.800376e-12, 1.296028e-11, 2.625016e-11},
    /* ls           */ {4.616326e-10, 9.143188e-10, 1.895809e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.800376e-12, 2.625016e-11, 6.890297e-06},
    /* ls           */ {4.616326e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.531737e-12, 7.800376e-12, 1.296028e-11, 2.625016e-11},
    /* ls           */ {1.845807e-10, 4.616326e-10, 9.143188e-10, 1.895809e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.531737e-12, 7.800376e-12, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 4.616326e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.531737e-12, 1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.800376e-12, 1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {4.616326e-10, 9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.531737e-12, 7.800376e-12, 1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 4.616326e-10, 9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {2.027756e-11},
    /* ls           */ {1.661482e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.163243e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 1.661482e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.863353e-12, 2.027756e-11},
    /* ls           */ {2.271028e-10, 1.661482e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.068442e-12, 2.027756e-11},
    /* ls           */ {5.081777e-10, 1.661482e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {2.027756e-11, 1.112223e-05},
    /* ls           */ {1.661482e-09, 1.325607e-05},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.163243e-12, 3.863353e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 2.271028e-10, 1.661482e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.163243e-12, 7.068442e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 5.081777e-10, 1.661482e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.163243e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.863353e-12, 7.068442e-12, 2.027756e-11},
    /* ls           */ {2.271028e-10, 5.081777e-10, 1.661482e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.863353e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {2.271028e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.163243e-12, 3.863353e-12, 7.068442e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 2.271028e-10, 5.081777e-10, 1.661482e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.163243e-12, 3.863353e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 2.271028e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.163243e-12, 7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.863353e-12, 7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {2.271028e-10, 5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.163243e-12, 3.863353e-12, 7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 2.271028e-10, 5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.755491e-11},
    /* ls           */ {1.405836e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.063670e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 1.405836e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.426981e-12, 1.755491e-11},
    /* ls           */ {6.640225e-11, 1.405836e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.568575e-12, 1.755491e-11},
    /* ls           */ {3.659156e-10, 1.405836e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.755491e-11, 3.293753e-05},
    /* ls           */ {1.405836e-09, 2.822876e-05},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.063670e-12, 1.426981e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 6.640225e-11, 1.405836e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.063670e-12, 5.568575e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 3.659156e-10, 1.405836e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.063670e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.426981e-12, 5.568575e-12, 1.755491e-11},
    /* ls           */ {6.640225e-11, 3.659156e-10, 1.405836e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.426981e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {6.640225e-11, 1.405836e-09, 2.822876e-05},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.063670e-12, 1.426981e-12, 5.568575e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 6.640225e-11, 3.659156e-10, 1.405836e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.063670e-12, 1.426981e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 6.640225e-11, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.063670e-12, 5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.426981e-12, 5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {6.640225e-11, 3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.063670e-12, 1.426981e-12, 5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 6.640225e-11, 3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {4.361676e-11},
    /* ls           */ {3.120099e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.428106e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 3.120099e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.584451e-11, 4.361676e-11},
    /* ls           */ {6.324624e-10, 3.120099e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.824054e-11, 4.361676e-11},
    /* ls           */ {1.787022e-09, 3.120099e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {4.361676e-11, 2.503395e-07},
    /* ls           */ {3.120099e-09, 2.384186e-07},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.428106e-11, 1.584451e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 6.324624e-10, 3.120099e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.428106e-11, 2.824054e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 1.787022e-09, 3.120099e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.428106e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.584451e-11, 2.824054e-11, 4.361676e-11},
    /* ls           */ {6.324624e-10, 1.787022e-09, 3.120099e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.584451e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {6.324624e-10, 3.120099e-09, 2.384186e-07},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.428106e-11, 1.584451e-11, 2.824054e-11, 4.361676e-11},
    /* ls           */ {4.842258e-10, 6.324624e-10, 1.787022e-09, 3.120099e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.428106e-11, 1.584451e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 6.324624e-10, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.428106e-11, 2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.584451e-11, 2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {6.324624e-10, 1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.428106e-11, 1.584451e-11, 2.824054e-11, 4.361676e-11, 2.503395e-07},
    /* ls           */ {4.842258e-10, 6.324624e-10, 1.787022e-09, 3.120099e-09, 2.384186e-07},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {3.805493e-11},
    /* ls           */ {2.872106e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.208603e-12, 3.805493e-11},
    /* ls           */ {3.388982e-10, 2.872106e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.379596e-11, 3.805493e-11},
    /* ls           */ {7.385124e-10, 2.872106e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.331419e-11, 3.805493e-11},
    /* ls           */ {1.590890e-09, 2.872106e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {3.805493e-11, 5.114079e-06},
    /* ls           */ {2.872106e-09, 5.424022e-06},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.208603e-12, 1.379596e-11, 3.805493e-11},
    /* ls           */ {3.388982e-10, 7.385124e-10, 2.872106e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.208603e-12, 2.331419e-11, 3.805493e-11},
    /* ls           */ {3.388982e-10, 1.590890e-09, 2.872106e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.208603e-12, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.379596e-11, 2.331419e-11, 3.805493e-11},
    /* ls           */ {7.385124e-10, 1.590890e-09, 2.872106e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.379596e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {7.385124e-10, 2.872106e-09, 5.424022e-06},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.208603e-12, 1.379596e-11, 2.331419e-11, 3.805493e-11},
    /* ls           */ {3.388982e-10, 7.385124e-10, 1.590890e-09, 2.872106e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.208603e-12, 1.379596e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 7.385124e-10, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.208603e-12, 2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.379596e-11, 2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {7.385124e-10, 1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.208603e-12, 1.379596e-11, 2.331419e-11, 3.805493e-11, 5.114079e-06},
    /* ls           */ {3.388982e-10, 7.385124e-10, 1.590890e-09, 2.872106e-09, 5.424022e-06},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {2.625016e-11},
    /* ls           */ {1.895809e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.531737e-12, 2.625016e-11},
    /* ls           */ {1.845807e-10, 1.895809e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.800376e-12, 2.625016e-11},
    /* ls           */ {4.616326e-10, 1.895809e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.296028e-11, 2.625016e-11},
    /* ls           */ {9.143188e-10, 1.895809e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {2.625016e-11, 6.890297e-06},
    /* ls           */ {1.895809e-09, 7.736683e-06},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.531737e-12, 7.800376e-12, 2.625016e-11},
    /* ls           */ {1.845807e-10, 4.616326e-10, 1.895809e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.531737e-12, 1.296028e-11, 2.625016e-11},
    /* ls           */ {1.845807e-10, 9.143188e-10, 1.895809e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.531737e-12, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.800376e-12, 1.296028e-11, 2.625016e-11},
    /* ls           */ {4.616326e-10, 9.143188e-10, 1.895809e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.800376e-12, 2.625016e-11, 6.890297e-06},
    /* ls           */ {4.616326e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.531737e-12, 7.800376e-12, 1.296028e-11, 2.625016e-11},
    /* ls           */ {1.845807e-10, 4.616326e-10, 9.143188e-10, 1.895809e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.531737e-12, 7.800376e-12, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 4.616326e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.531737e-12, 1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.800376e-12, 1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {4.616326e-10, 9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.531737e-12, 7.800376e-12, 1.296028e-11, 2.625016e-11, 6.890297e-06},
    /* ls           */ {1.845807e-10, 4.616326e-10, 9.143188e-10, 1.895809e-09, 7.736683e-06},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {2.027756e-11},
    /* ls           */ {1.661482e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.163243e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 1.661482e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.863353e-12, 2.027756e-11},
    /* ls           */ {2.271028e-10, 1.661482e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.068442e-12, 2.027756e-11},
    /* ls           */ {5.081777e-10, 1.661482e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {2.027756e-11, 1.112223e-05},
    /* ls           */ {1.661482e-09, 1.325607e-05},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.163243e-12, 3.863353e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 2.271028e-10, 1.661482e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.163243e-12, 7.068442e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 5.081777e-10, 1.661482e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.163243e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.863353e-12, 7.068442e-12, 2.027756e-11},
    /* ls           */ {2.271028e-10, 5.081777e-10, 1.661482e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.863353e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {2.271028e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.163243e-12, 3.863353e-12, 7.068442e-12, 2.027756e-11},
    /* ls           */ {8.582038e-11, 2.271028e-10, 5.081777e-10, 1.661482e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.163243e-12, 3.863353e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 2.271028e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.163243e-12, 7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.863353e-12, 7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {2.271028e-10, 5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.163243e-12, 3.863353e-12, 7.068442e-12, 2.027756e-11, 1.112223e-05},
    /* ls           */ {8.582038e-11, 2.271028e-10, 5.081777e-10, 1.661482e-09, 1.325607e-05},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.755491e-11},
    /* ls           */ {1.405836e-09},
    /* m            */ {67027689472},
    /* p            */ {16},
    /* kmax         */ {999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.063670e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 1.405836e-09},
    /* m            */ {49152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.426981e-12, 1.755491e-11},
    /* ls           */ {6.640225e-11, 1.405836e-09},
    /* m            */ {2097152, 67027689472},
    /* p            */ {2, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.568575e-12, 1.755491e-11},
    /* ls           */ {3.659156e-10, 1.405836e-09},
    /* m            */ {37748736, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.755491e-11, 3.293753e-05},
    /* ls           */ {1.405836e-09, 2.822876e-05},
    /* m            */ {67027689472, 67027689472},
    /* p            */ {16, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.063670e-12, 1.426981e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 6.640225e-11, 1.405836e-09},
    /* m            */ {49152, 2097152, 67027689472},
    /* p            */ {2, 1, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.063670e-12, 5.568575e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 3.659156e-10, 1.405836e-09},
    /* m            */ {49152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.063670e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.426981e-12, 5.568575e-12, 1.755491e-11},
    /* ls           */ {6.640225e-11, 3.659156e-10, 1.405836e-09},
    /* m            */ {2097152, 37748736, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.426981e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {6.640225e-11, 1.405836e-09, 2.822876e-05},
    /* m            */ {2097152, 67027689472, 67027689472},
    /* p            */ {2, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {37748736, 67027689472, 67027689472},
    /* p            */ {16, 1, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.063670e-12, 1.426981e-12, 5.568575e-12, 1.755491e-11},
    /* ls           */ {4.138110e-11, 6.640225e-11, 3.659156e-10, 1.405836e-09},
    /* m            */ {49152, 2097152, 37748736, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.063670e-12, 1.426981e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 6.640225e-11, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 2097152, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.063670e-12, 5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.426981e-12, 5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {6.640225e-11, 3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.063670e-12, 1.426981e-12, 5.568575e-12, 1.755491e-11, 3.293753e-05},
    /* ls           */ {4.138110e-11, 6.640225e-11, 3.659156e-10, 1.405836e-09, 2.822876e-05},
    /* m            */ {49152, 2097152, 37748736, 67027689472, 67027689472},
    /* p            */ {2, 1, 8, 1, 1},
    /* kmax         */ {999, 999, 999, 999, 999}}

};

// HWParameter_configurations struct
const cost_models::HW_model::HWParameter_configurations dis_system_params = {
    /* threads_options_num */ {1, 2, 4, 8, 16},
    /* policy_options_str */ {"close", "spread"},
    /* level_options_str  */ {"L1Cache", "L2Cache", "L3Cache", "NodeMem", "GLOBAL_SYNC"},
    /* hw_model_thread_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    /* hw_model_policy_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    /* hw_model_level_bithash */ {8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31},
    /* hw_models */ hw_models_vector
};

#endif // HW_PARAMS_INTEL_WS_HPP
