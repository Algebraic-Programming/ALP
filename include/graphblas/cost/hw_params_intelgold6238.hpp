
// Auto-generated hardware parameters for IntelGold6238
// Allocation policies: close
// Base levels: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
// ALL subset combinations (16 total)
// Generated on 2025-11-03 17:05:30

#ifndef HW_PARAMS_INTELGOLD6238_HPP
#define HW_PARAMS_INTELGOLD6238_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations and level subsets
const std::vector<cost_models::HW_model::HWParameters> hw_models_vector = {

    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.168424e-10},
    /* ls           */ {6.704420e-09},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.875623e-11, 1.168424e-10},
    /* ls           */ {1.419005e-09, 6.704420e-09},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {7.470224e-11, 1.168424e-10},
    /* ls           */ {3.472347e-09, 6.704420e-09},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.069839e-10, 1.168424e-10},
    /* ls           */ {5.803513e-09, 6.704420e-09},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.168424e-10, 0.000000e+00},
    /* ls           */ {6.704420e-09, 1.815955e-06},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {4.875623e-11, 7.470224e-11, 1.168424e-10},
    /* ls           */ {1.419005e-09, 3.472347e-09, 6.704420e-09},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.875623e-11, 1.069839e-10, 1.168424e-10},
    /* ls           */ {1.419005e-09, 5.803513e-09, 6.704420e-09},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.875623e-11, 1.168424e-10, 0.000000e+00},
    /* ls           */ {1.419005e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {7.470224e-11, 1.069839e-10, 1.168424e-10},
    /* ls           */ {3.472347e-09, 5.803513e-09, 6.704420e-09},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {7.470224e-11, 1.168424e-10, 0.000000e+00},
    /* ls           */ {3.472347e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.069839e-10, 1.168424e-10, 0.000000e+00},
    /* ls           */ {5.803513e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.875623e-11, 7.470224e-11, 1.069839e-10, 1.168424e-10},
    /* ls           */ {1.419005e-09, 3.472347e-09, 5.803513e-09, 6.704420e-09},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.875623e-11, 7.470224e-11, 1.168424e-10, 0.000000e+00},
    /* ls           */ {1.419005e-09, 3.472347e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.875623e-11, 1.069839e-10, 1.168424e-10, 0.000000e+00},
    /* ls           */ {1.419005e-09, 5.803513e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {7.470224e-11, 1.069839e-10, 1.168424e-10, 0.000000e+00},
    /* ls           */ {3.472347e-09, 5.803513e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.875623e-11, 7.470224e-11, 1.069839e-10, 1.168424e-10, 0.000000e+00},
    /* ls           */ {1.419005e-09, 3.472347e-09, 5.803513e-09, 6.704420e-09, 1.815955e-06},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {5.865522e-11},
    /* ls           */ {3.332479e-09},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.348678e-11, 5.865522e-11},
    /* ls           */ {5.330843e-10, 3.332479e-09},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {3.791343e-11, 5.865522e-11},
    /* ls           */ {1.749554e-09, 3.332479e-09},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.283538e-11, 5.865522e-11},
    /* ls           */ {2.898976e-09, 3.332479e-09},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.865522e-11, 0.000000e+00},
    /* ls           */ {3.332479e-09, 1.815955e-06},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.348678e-11, 3.791343e-11, 5.865522e-11},
    /* ls           */ {5.330843e-10, 1.749554e-09, 3.332479e-09},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.348678e-11, 5.283538e-11, 5.865522e-11},
    /* ls           */ {5.330843e-10, 2.898976e-09, 3.332479e-09},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.348678e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {5.330843e-10, 3.332479e-09, 1.815955e-06},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.791343e-11, 5.283538e-11, 5.865522e-11},
    /* ls           */ {1.749554e-09, 2.898976e-09, 3.332479e-09},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.791343e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {1.749554e-09, 3.332479e-09, 1.815955e-06},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.283538e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {2.898976e-09, 3.332479e-09, 1.815955e-06},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.348678e-11, 3.791343e-11, 5.283538e-11, 5.865522e-11},
    /* ls           */ {5.330843e-10, 1.749554e-09, 2.898976e-09, 3.332479e-09},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.348678e-11, 3.791343e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {5.330843e-10, 1.749554e-09, 3.332479e-09, 1.815955e-06},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.348678e-11, 5.283538e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {5.330843e-10, 2.898976e-09, 3.332479e-09, 1.815955e-06},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.791343e-11, 5.283538e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {1.749554e-09, 2.898976e-09, 3.332479e-09, 1.815955e-06},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.348678e-11, 3.791343e-11, 5.283538e-11, 5.865522e-11, 0.000000e+00},
    /* ls           */ {5.330843e-10, 1.749554e-09, 2.898976e-09, 3.332479e-09, 1.815955e-06},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.986029e-11},
    /* ls           */ {1.708615e-09},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.176404e-11, 2.986029e-11},
    /* ls           */ {2.679966e-10, 1.708615e-09},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.884714e-11, 2.986029e-11},
    /* ls           */ {8.729781e-10, 1.708615e-09},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.627025e-11, 2.986029e-11},
    /* ls           */ {1.443445e-09, 1.708615e-09},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.986029e-11, 0.000000e+00},
    /* ls           */ {1.708615e-09, 1.815955e-06},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.176404e-11, 1.884714e-11, 2.986029e-11},
    /* ls           */ {2.679966e-10, 8.729781e-10, 1.708615e-09},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.176404e-11, 2.627025e-11, 2.986029e-11},
    /* ls           */ {2.679966e-10, 1.443445e-09, 1.708615e-09},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.176404e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {2.679966e-10, 1.708615e-09, 1.815955e-06},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.884714e-11, 2.627025e-11, 2.986029e-11},
    /* ls           */ {8.729781e-10, 1.443445e-09, 1.708615e-09},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.884714e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {8.729781e-10, 1.708615e-09, 1.815955e-06},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.627025e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {1.443445e-09, 1.708615e-09, 1.815955e-06},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.176404e-11, 1.884714e-11, 2.627025e-11, 2.986029e-11},
    /* ls           */ {2.679966e-10, 8.729781e-10, 1.443445e-09, 1.708615e-09},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.176404e-11, 1.884714e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {2.679966e-10, 8.729781e-10, 1.708615e-09, 1.815955e-06},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.176404e-11, 2.627025e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {2.679966e-10, 1.443445e-09, 1.708615e-09, 1.815955e-06},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.884714e-11, 2.627025e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {8.729781e-10, 1.443445e-09, 1.708615e-09, 1.815955e-06},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.176404e-11, 1.884714e-11, 2.627025e-11, 2.986029e-11, 0.000000e+00},
    /* ls           */ {2.679966e-10, 8.729781e-10, 1.443445e-09, 1.708615e-09, 1.815955e-06},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.706080e-11},
    /* ls           */ {1.001663e-09},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {5.877538e-12, 1.706080e-11},
    /* ls           */ {1.423543e-10, 1.001663e-09},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {9.556236e-12, 1.706080e-11},
    /* ls           */ {4.532497e-10, 1.001663e-09},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.709113e-11, 1.706080e-11},
    /* ls           */ {9.720056e-10, 1.001663e-09},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.706080e-11, 0.000000e+00},
    /* ls           */ {1.001663e-09, 2.419949e-06},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {5.877538e-12, 9.556236e-12, 1.706080e-11},
    /* ls           */ {1.423543e-10, 4.532497e-10, 1.001663e-09},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.877538e-12, 1.709113e-11, 1.706080e-11},
    /* ls           */ {1.423543e-10, 9.720056e-10, 1.001663e-09},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.877538e-12, 1.706080e-11, 0.000000e+00},
    /* ls           */ {1.423543e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.556236e-12, 1.709113e-11, 1.706080e-11},
    /* ls           */ {4.532497e-10, 9.720056e-10, 1.001663e-09},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.556236e-12, 1.706080e-11, 0.000000e+00},
    /* ls           */ {4.532497e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.709113e-11, 1.706080e-11, 0.000000e+00},
    /* ls           */ {9.720056e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {5.877538e-12, 9.556236e-12, 1.709113e-11, 1.706080e-11},
    /* ls           */ {1.423543e-10, 4.532497e-10, 9.720056e-10, 1.001663e-09},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {5.877538e-12, 9.556236e-12, 1.706080e-11, 0.000000e+00},
    /* ls           */ {1.423543e-10, 4.532497e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {5.877538e-12, 1.709113e-11, 1.706080e-11, 0.000000e+00},
    /* ls           */ {1.423543e-10, 9.720056e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {9.556236e-12, 1.709113e-11, 1.706080e-11, 0.000000e+00},
    /* ls           */ {4.532497e-10, 9.720056e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {5.877538e-12, 9.556236e-12, 1.709113e-11, 1.706080e-11, 0.000000e+00},
    /* ls           */ {1.423543e-10, 4.532497e-10, 9.720056e-10, 1.001663e-09, 2.419949e-06},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.449961e-11},
    /* ls           */ {9.074610e-10},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.022741e-12, 1.449961e-11},
    /* ls           */ {9.079104e-11, 9.074610e-10},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {6.304809e-12, 1.449961e-11},
    /* ls           */ {2.963546e-10, 9.074610e-10},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.413030e-11, 1.449961e-11},
    /* ls           */ {8.446864e-10, 9.074610e-10},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.449961e-11, 0.000000e+00},
    /* ls           */ {9.074610e-10, 3.337860e-06},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {4.022741e-12, 6.304809e-12, 1.449961e-11},
    /* ls           */ {9.079104e-11, 2.963546e-10, 9.074610e-10},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.022741e-12, 1.413030e-11, 1.449961e-11},
    /* ls           */ {9.079104e-11, 8.446864e-10, 9.074610e-10},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.022741e-12, 1.449961e-11, 0.000000e+00},
    /* ls           */ {9.079104e-11, 9.074610e-10, 3.337860e-06},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {6.304809e-12, 1.413030e-11, 1.449961e-11},
    /* ls           */ {2.963546e-10, 8.446864e-10, 9.074610e-10},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {6.304809e-12, 1.449961e-11, 0.000000e+00},
    /* ls           */ {2.963546e-10, 9.074610e-10, 3.337860e-06},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.413030e-11, 1.449961e-11, 0.000000e+00},
    /* ls           */ {8.446864e-10, 9.074610e-10, 3.337860e-06},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.022741e-12, 6.304809e-12, 1.413030e-11, 1.449961e-11},
    /* ls           */ {9.079104e-11, 2.963546e-10, 8.446864e-10, 9.074610e-10},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.022741e-12, 6.304809e-12, 1.449961e-11, 0.000000e+00},
    /* ls           */ {9.079104e-11, 2.963546e-10, 9.074610e-10, 3.337860e-06},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.022741e-12, 1.413030e-11, 1.449961e-11, 0.000000e+00},
    /* ls           */ {9.079104e-11, 8.446864e-10, 9.074610e-10, 3.337860e-06},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {6.304809e-12, 1.413030e-11, 1.449961e-11, 0.000000e+00},
    /* ls           */ {2.963546e-10, 8.446864e-10, 9.074610e-10, 3.337860e-06},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.022741e-12, 6.304809e-12, 1.413030e-11, 1.449961e-11, 0.000000e+00},
    /* ls           */ {9.079104e-11, 2.963546e-10, 8.446864e-10, 9.074610e-10, 3.337860e-06},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.444315e-11},
    /* ls           */ {9.211918e-10},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.247108e-12, 1.444315e-11},
    /* ls           */ {5.134671e-11, 9.211918e-10},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {3.882754e-12, 1.444315e-11},
    /* ls           */ {1.886740e-10, 9.211918e-10},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.285863e-11, 1.444315e-11},
    /* ls           */ {8.204070e-10, 9.211918e-10},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.444315e-11, 0.000000e+00},
    /* ls           */ {9.211918e-10, 5.114079e-06},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.247108e-12, 3.882754e-12, 1.444315e-11},
    /* ls           */ {5.134671e-11, 1.886740e-10, 9.211918e-10},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.247108e-12, 1.285863e-11, 1.444315e-11},
    /* ls           */ {5.134671e-11, 8.204070e-10, 9.211918e-10},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.247108e-12, 1.444315e-11, 0.000000e+00},
    /* ls           */ {5.134671e-11, 9.211918e-10, 5.114079e-06},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.882754e-12, 1.285863e-11, 1.444315e-11},
    /* ls           */ {1.886740e-10, 8.204070e-10, 9.211918e-10},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.882754e-12, 1.444315e-11, 0.000000e+00},
    /* ls           */ {1.886740e-10, 9.211918e-10, 5.114079e-06},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.285863e-11, 1.444315e-11, 0.000000e+00},
    /* ls           */ {8.204070e-10, 9.211918e-10, 5.114079e-06},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.247108e-12, 3.882754e-12, 1.285863e-11, 1.444315e-11},
    /* ls           */ {5.134671e-11, 1.886740e-10, 8.204070e-10, 9.211918e-10},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.247108e-12, 3.882754e-12, 1.444315e-11, 0.000000e+00},
    /* ls           */ {5.134671e-11, 1.886740e-10, 9.211918e-10, 5.114079e-06},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.247108e-12, 1.285863e-11, 1.444315e-11, 0.000000e+00},
    /* ls           */ {5.134671e-11, 8.204070e-10, 9.211918e-10, 5.114079e-06},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.882754e-12, 1.285863e-11, 1.444315e-11, 0.000000e+00},
    /* ls           */ {1.886740e-10, 8.204070e-10, 9.211918e-10, 5.114079e-06},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.247108e-12, 3.882754e-12, 1.285863e-11, 1.444315e-11, 0.000000e+00},
    /* ls           */ {5.134671e-11, 1.886740e-10, 8.204070e-10, 9.211918e-10, 5.114079e-06},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.798469e-12},
    /* ls           */ {6.221578e-10},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.472711e-12, 9.798469e-12},
    /* ls           */ {3.391029e-11, 6.221578e-10},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.085993e-12, 9.798469e-12},
    /* ls           */ {8.988997e-11, 6.221578e-10},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.399441e-12, 9.798469e-12},
    /* ls           */ {4.634705e-10, 6.221578e-10},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.798469e-12, 0.000000e+00},
    /* ls           */ {6.221578e-10, 1.131296e-05},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.472711e-12, 2.085993e-12, 9.798469e-12},
    /* ls           */ {3.391029e-11, 8.988997e-11, 6.221578e-10},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472711e-12, 7.399441e-12, 9.798469e-12},
    /* ls           */ {3.391029e-11, 4.634705e-10, 6.221578e-10},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472711e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {3.391029e-11, 6.221578e-10, 1.131296e-05},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.085993e-12, 7.399441e-12, 9.798469e-12},
    /* ls           */ {8.988997e-11, 4.634705e-10, 6.221578e-10},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.085993e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {8.988997e-11, 6.221578e-10, 1.131296e-05},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.399441e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {4.634705e-10, 6.221578e-10, 1.131296e-05},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.472711e-12, 2.085993e-12, 7.399441e-12, 9.798469e-12},
    /* ls           */ {3.391029e-11, 8.988997e-11, 4.634705e-10, 6.221578e-10},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.472711e-12, 2.085993e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {3.391029e-11, 8.988997e-11, 6.221578e-10, 1.131296e-05},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472711e-12, 7.399441e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {3.391029e-11, 4.634705e-10, 6.221578e-10, 1.131296e-05},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.085993e-12, 7.399441e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {8.988997e-11, 4.634705e-10, 6.221578e-10, 1.131296e-05},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.472711e-12, 2.085993e-12, 7.399441e-12, 9.798469e-12, 0.000000e+00},
    /* ls           */ {3.391029e-11, 8.988997e-11, 4.634705e-10, 6.221578e-10, 1.131296e-05},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.823679e-12},
    /* ls           */ {6.278725e-10},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.123883e-12, 9.823679e-12},
    /* ls           */ {2.590913e-11, 6.278725e-10},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.513345e-12, 9.823679e-12},
    /* ls           */ {5.948391e-11, 6.278725e-10},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.284549e-12, 9.823679e-12},
    /* ls           */ {4.008977e-10, 6.278725e-10},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.823679e-12, 0.000000e+00},
    /* ls           */ {6.278725e-10, 1.703501e-05},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.123883e-12, 1.513345e-12, 9.823679e-12},
    /* ls           */ {2.590913e-11, 5.948391e-11, 6.278725e-10},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.123883e-12, 6.284549e-12, 9.823679e-12},
    /* ls           */ {2.590913e-11, 4.008977e-10, 6.278725e-10},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.123883e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {2.590913e-11, 6.278725e-10, 1.703501e-05},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.513345e-12, 6.284549e-12, 9.823679e-12},
    /* ls           */ {5.948391e-11, 4.008977e-10, 6.278725e-10},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.513345e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {5.948391e-11, 6.278725e-10, 1.703501e-05},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.284549e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {4.008977e-10, 6.278725e-10, 1.703501e-05},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.123883e-12, 1.513345e-12, 6.284549e-12, 9.823679e-12},
    /* ls           */ {2.590913e-11, 5.948391e-11, 4.008977e-10, 6.278725e-10},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.123883e-12, 1.513345e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {2.590913e-11, 5.948391e-11, 6.278725e-10, 1.703501e-05},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.123883e-12, 6.284549e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {2.590913e-11, 4.008977e-10, 6.278725e-10, 1.703501e-05},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.513345e-12, 6.284549e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {5.948391e-11, 4.008977e-10, 6.278725e-10, 1.703501e-05},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.123883e-12, 1.513345e-12, 6.284549e-12, 9.823679e-12, 0.000000e+00},
    /* ls           */ {2.590913e-11, 5.948391e-11, 4.008977e-10, 6.278725e-10, 1.703501e-05},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.728446e-12},
    /* ls           */ {6.228078e-10},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {9.597317e-13, 9.728446e-12},
    /* ls           */ {3.527172e-11, 6.228078e-10},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.591014e-12, 9.728446e-12},
    /* ls           */ {8.885077e-11, 6.228078e-10},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.036710e-12, 9.728446e-12},
    /* ls           */ {4.453786e-10, 6.228078e-10},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.728446e-12, 0.000000e+00},
    /* ls           */ {6.228078e-10, 2.363920e-05},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {9.597317e-13, 1.591014e-12, 9.728446e-12},
    /* ls           */ {3.527172e-11, 8.885077e-11, 6.228078e-10},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.597317e-13, 7.036710e-12, 9.728446e-12},
    /* ls           */ {3.527172e-11, 4.453786e-10, 6.228078e-10},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.597317e-13, 9.728446e-12, 0.000000e+00},
    /* ls           */ {3.527172e-11, 6.228078e-10, 2.363920e-05},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.591014e-12, 7.036710e-12, 9.728446e-12},
    /* ls           */ {8.885077e-11, 4.453786e-10, 6.228078e-10},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.591014e-12, 9.728446e-12, 0.000000e+00},
    /* ls           */ {8.885077e-11, 6.228078e-10, 2.363920e-05},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.036710e-12, 9.728446e-12, 0.000000e+00},
    /* ls           */ {4.453786e-10, 6.228078e-10, 2.363920e-05},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {9.597317e-13, 1.591014e-12, 7.036710e-12, 9.728446e-12},
    /* ls           */ {3.527172e-11, 8.885077e-11, 4.453786e-10, 6.228078e-10},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {9.597317e-13, 1.591014e-12, 9.728446e-12, 0.000000e+00},
    /* ls           */ {3.527172e-11, 8.885077e-11, 6.228078e-10, 2.363920e-05},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {9.597317e-13, 7.036710e-12, 9.728446e-12, 0.000000e+00},
    /* ls           */ {3.527172e-11, 4.453786e-10, 6.228078e-10, 2.363920e-05},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.591014e-12, 7.036710e-12, 9.728446e-12, 0.000000e+00},
    /* ls           */ {8.885077e-11, 4.453786e-10, 6.228078e-10, 2.363920e-05},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {9.597317e-13, 1.591014e-12, 7.036710e-12, 9.728446e-12, 0.000000e+00},
    /* ls           */ {3.527172e-11, 8.885077e-11, 4.453786e-10, 6.228078e-10, 2.363920e-05},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.975225e-12},
    /* ls           */ {2.967564e-10},
    /* m            */ {201863462912},
    /* p            */ {84},
    /* kmax         */ {999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {8.251363e-13, 9.975225e-12},
    /* ls           */ {3.075218e-11, 2.967564e-10},
    /* m            */ {1048576, 201863462912},
    /* p            */ {2, 42},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.269087e-12, 9.975225e-12},
    /* ls           */ {1.327258e-10, 2.967564e-10},
    /* m            */ {31457280, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.626116e-12, 9.975225e-12},
    /* ls           */ {2.957186e-10, 2.967564e-10},
    /* m            */ {100931731456, 201863462912},
    /* p            */ {42, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.975225e-12, 0.000000e+00},
    /* ls           */ {2.967564e-10, 1.182675e-04},
    /* m            */ {201863462912, 201863462912},
    /* p            */ {84, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {8.251363e-13, 2.269087e-12, 9.975225e-12},
    /* ls           */ {3.075218e-11, 1.327258e-10, 2.967564e-10},
    /* m            */ {1048576, 31457280, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.251363e-13, 4.626116e-12, 9.975225e-12},
    /* ls           */ {3.075218e-11, 2.957186e-10, 2.967564e-10},
    /* m            */ {1048576, 100931731456, 201863462912},
    /* p            */ {2, 21, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.251363e-13, 9.975225e-12, 0.000000e+00},
    /* ls           */ {3.075218e-11, 2.967564e-10, 1.182675e-04},
    /* m            */ {1048576, 201863462912, 201863462912},
    /* p            */ {2, 42, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.269087e-12, 4.626116e-12, 9.975225e-12},
    /* ls           */ {1.327258e-10, 2.957186e-10, 2.967564e-10},
    /* m            */ {31457280, 100931731456, 201863462912},
    /* p            */ {42, 1, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.269087e-12, 9.975225e-12, 0.000000e+00},
    /* ls           */ {1.327258e-10, 2.967564e-10, 1.182675e-04},
    /* m            */ {31457280, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.626116e-12, 9.975225e-12, 0.000000e+00},
    /* ls           */ {2.957186e-10, 2.967564e-10, 1.182675e-04},
    /* m            */ {100931731456, 201863462912, 201863462912},
    /* p            */ {42, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.251363e-13, 2.269087e-12, 4.626116e-12, 9.975225e-12},
    /* ls           */ {3.075218e-11, 1.327258e-10, 2.957186e-10, 2.967564e-10},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912},
    /* p            */ {2, 21, 1, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.251363e-13, 2.269087e-12, 9.975225e-12, 0.000000e+00},
    /* ls           */ {3.075218e-11, 1.327258e-10, 2.967564e-10, 1.182675e-04},
    /* m            */ {1048576, 31457280, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.251363e-13, 4.626116e-12, 9.975225e-12, 0.000000e+00},
    /* ls           */ {3.075218e-11, 2.957186e-10, 2.967564e-10, 1.182675e-04},
    /* m            */ {1048576, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.269087e-12, 4.626116e-12, 9.975225e-12, 0.000000e+00},
    /* ls           */ {1.327258e-10, 2.957186e-10, 2.967564e-10, 1.182675e-04},
    /* m            */ {31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {42, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.251363e-13, 2.269087e-12, 4.626116e-12, 9.975225e-12, 0.000000e+00},
    /* ls           */ {3.075218e-11, 1.327258e-10, 2.957186e-10, 2.967564e-10, 1.182675e-04},
    /* m            */ {1048576, 31457280, 100931731456, 201863462912, 201863462912},
    /* p            */ {2, 21, 1, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}}

};

// HWParameter_configurations struct
const cost_models::HW_model::HWParameter_configurations dis_system_params = {
    /* threads_options_num */ {1, 2, 4, 8, 12, 21, 32, 42, 64, 84},
    /* policy_options_str */ {"close"},
    /* level_options_str  */ {"L2Cache", "L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"},
    /* hw_model_thread_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
    /* hw_model_policy_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    /* hw_model_level_bithash */ {8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31},
    /* hw_models */ hw_models_vector
};

#endif // HW_PARAMS_INTELGOLD6238_HPP
