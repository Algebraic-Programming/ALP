
// Auto-generated hardware parameters for ARM920
// Allocation policies: close, spread
// Base levels: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
// Generated ALL subset combinations (64 total)
// Generated on 2025-10-07 17:39:56

#ifndef HW_PARAMS_ARM920_HPP
#define HW_PARAMS_ARM920_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations and level subsets
const std::vector<cost_models::HW_model::HWParameters> hw_models_vector = {

    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.857639e-10},
    /* ls           */ {7.502419e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.300177e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 7.502419e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.704307e-11, 1.857639e-10},
    /* ls           */ {1.220061e-09, 7.502419e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.869834e-11, 1.857639e-10},
    /* ls           */ {3.151055e-09, 7.502419e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.183435e-10, 1.857639e-10},
    /* ls           */ {4.377526e-09, 7.502419e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.338132e-10, 1.857639e-10},
    /* ls           */ {4.949459e-09, 7.502419e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.857639e-10, 2.503395e-06},
    /* ls           */ {7.502419e-09, 2.217293e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {9.475240e-11},
    /* ls           */ {3.959072e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.133604e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 3.959072e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.325483e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 3.959072e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.120760e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 3.959072e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.620053e-11, 9.475240e-11},
    /* ls           */ {2.896586e-09, 3.959072e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.732910e-11, 9.475240e-11},
    /* ls           */ {2.798050e-09, 3.959072e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {9.475240e-11, 1.931191e-06},
    /* ls           */ {3.959072e-09, 2.443791e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 2.325483e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 3.959072e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 5.120760e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 5.120760e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.896586e-09, 3.959072e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {4.977566e-11},
    /* ls           */ {2.692659e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.065737e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.692659e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.216898e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.692659e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.141160e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.692659e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.745975e-11, 4.977566e-11},
    /* ls           */ {2.088403e-09, 2.692659e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.660023e-11, 4.977566e-11},
    /* ls           */ {2.567747e-09, 2.692659e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {4.977566e-11, 3.719330e-06},
    /* ls           */ {2.692659e-09, 3.588200e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.692659e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.692659e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {2.604668e-11},
    /* ls           */ {1.472085e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.308637e-12, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.472085e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.220618e-12, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.472085e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.643502e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.472085e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.193186e-11, 2.604668e-11},
    /* ls           */ {1.295675e-09, 1.472085e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.635965e-11, 2.604668e-11},
    /* ls           */ {1.436174e-09, 1.472085e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {2.604668e-11, 4.613400e-06},
    /* ls           */ {1.472085e-09, 4.911422e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.472085e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.472085e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.851089e-11},
    /* ls           */ {1.146673e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.533423e-12, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.146673e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.265756e-12, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.146673e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.189379e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.146673e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.750302e-11, 1.851089e-11},
    /* ls           */ {1.041810e-09, 1.146673e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.042966e-11, 1.851089e-11},
    /* ls           */ {1.288429e-09, 1.146673e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.851089e-11, 7.688997e-06},
    /* ls           */ {1.146673e-09, 6.961822e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.146673e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.146673e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.146673e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 1.750302e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.579307e-11},
    /* ls           */ {1.061221e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.653410e-12, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.061221e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.294345e-12, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.061221e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.015133e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.061221e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.543769e-11, 1.579307e-11},
    /* ls           */ {1.004934e-09, 1.061221e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.842458e-11, 1.579307e-11},
    /* ls           */ {1.223974e-09, 1.061221e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.579307e-11, 2.183914e-05},
    /* ls           */ {1.061221e-09, 2.493858e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.061221e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.061221e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.061221e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.315468e-11},
    /* ls           */ {8.750512e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.775661e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 8.750512e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.280146e-12, 1.315468e-11},
    /* ls           */ {9.300533e-11, 8.750512e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.955418e-12, 1.315468e-11},
    /* ls           */ {4.417624e-10, 8.750512e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.255653e-11, 1.315468e-11},
    /* ls           */ {7.994690e-10, 8.750512e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.604392e-11, 1.315468e-11},
    /* ls           */ {1.054863e-09, 8.750512e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.315468e-11, 8.952619e-06},
    /* ls           */ {8.750512e-10, 8.249284e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 8.750512e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 8.750512e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 8.750512e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 7.994690e-10, 8.750512e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.219101e-11},
    /* ls           */ {8.239890e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.182685e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 8.239890e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.501807e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 8.239890e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.604989e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 8.239890e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.450134e-12, 1.219101e-11},
    /* ls           */ {5.349837e-10, 8.239890e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {9.866466e-12, 1.219101e-11},
    /* ls           */ {6.667878e-10, 8.239890e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.219101e-11, 3.409386e-05},
    /* ls           */ {8.239890e-10, 3.490448e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 1.501807e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 8.239890e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 3.604989e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 3.604989e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 5.349837e-10, 8.239890e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.237560e-11},
    /* ls           */ {8.563975e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.869306e-13, 1.237560e-11},
    /* ls           */ {1.826369e-11, 8.563975e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.169244e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 8.563975e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.155775e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 8.563975e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.195628e-12, 1.237560e-11},
    /* ls           */ {4.529260e-10, 8.563975e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {9.605758e-12, 1.237560e-11},
    /* ls           */ {6.765814e-10, 8.563975e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.237560e-11, 6.256103e-05},
    /* ls           */ {8.563975e-10, 6.915331e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 1.169244e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 8.563975e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 3.155775e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 3.155775e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 4.529260e-10, 8.563975e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {9.129807e-12},
    /* ls           */ {6.495532e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.644030e-13, 9.129807e-12},
    /* ls           */ {1.373541e-11, 6.495532e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.885908e-13, 9.129807e-12},
    /* ls           */ {3.665152e-11, 6.495532e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.108379e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 6.495532e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.605207e-12, 9.129807e-12},
    /* ls           */ {3.587535e-10, 6.495532e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {9.629964e-12, 9.129807e-12},
    /* ls           */ {7.137248e-10, 6.495532e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {9.129807e-12, 9.292364e-05},
    /* ls           */ {6.495532e-10, 1.113296e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 6.495532e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 3.587535e-10, 6.495532e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {9.184656e-12},
    /* ls           */ {6.728599e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.527422e-13, 9.184656e-12},
    /* ls           */ {9.486443e-12, 6.728599e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.345705e-13, 9.184656e-12},
    /* ls           */ {2.705177e-11, 6.728599e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.439972e-12, 9.184656e-12},
    /* ls           */ {1.141888e-10, 6.728599e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.554426e-12, 9.184656e-12},
    /* ls           */ {2.908892e-10, 6.728599e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.131008e-11, 9.184656e-12},
    /* ls           */ {8.342130e-10, 6.728599e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {9.184656e-12, 3.641129e-04},
    /* ls           */ {6.728599e-10, 2.368450e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 6.345705e-13, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 6.728599e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 1.439972e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 1.439972e-12, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 4.554426e-12, 9.184656e-12},
    /* ls           */ {2.705177e-11, 2.908892e-10, 6.728599e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.857639e-10},
    /* ls           */ {7.502419e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.300177e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 7.502419e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.704307e-11, 1.857639e-10},
    /* ls           */ {1.220061e-09, 7.502419e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.869834e-11, 1.857639e-10},
    /* ls           */ {3.151055e-09, 7.502419e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.183435e-10, 1.857639e-10},
    /* ls           */ {4.377526e-09, 7.502419e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.338132e-10, 1.857639e-10},
    /* ls           */ {4.949459e-09, 7.502419e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.857639e-10, 2.503395e-06},
    /* ls           */ {7.502419e-09, 2.217293e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.300177e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.704307e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.300177e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.704307e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.300177e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 4.704307e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.300177e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {4.300177e-11, 4.704307e-11, 7.869834e-11, 1.183435e-10, 1.338132e-10, 1.857639e-10, 2.503395e-06},
    /* ls           */ {8.841671e-10, 1.220061e-09, 3.151055e-09, 4.377526e-09, 4.949459e-09, 7.502419e-09, 2.217293e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {9.475240e-11},
    /* ls           */ {3.959072e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.133604e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 3.959072e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.325483e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 3.959072e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.120760e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 3.959072e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.620053e-11, 9.475240e-11},
    /* ls           */ {2.896586e-09, 3.959072e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.732910e-11, 9.475240e-11},
    /* ls           */ {2.798050e-09, 3.959072e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {9.475240e-11, 1.931191e-06},
    /* ls           */ {3.959072e-09, 2.443791e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 2.325483e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 3.959072e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 5.120760e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.133604e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 5.120760e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.896586e-09, 3.959072e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.325483e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 2.325483e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.133604e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.325483e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.133604e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 2.325483e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.133604e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {2.133604e-11, 2.325483e-11, 5.120760e-11, 6.620053e-11, 6.732910e-11, 9.475240e-11, 1.931191e-06},
    /* ls           */ {4.433552e-10, 6.895785e-10, 2.697968e-09, 2.896586e-09, 2.798050e-09, 3.959072e-09, 2.443791e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {4.977566e-11},
    /* ls           */ {2.692659e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.065737e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.692659e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.216898e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.692659e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.141160e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.692659e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.745975e-11, 4.977566e-11},
    /* ls           */ {2.088403e-09, 2.692659e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.660023e-11, 4.977566e-11},
    /* ls           */ {2.567747e-09, 2.692659e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {4.977566e-11, 3.719330e-06},
    /* ls           */ {2.692659e-09, 3.588200e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.692659e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.065737e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.692659e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.216898e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.065737e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.216898e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 1.216898e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.065737e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.065737e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {1.065737e-11, 1.216898e-11, 3.141160e-11, 3.745975e-11, 4.660023e-11, 4.977566e-11, 3.719330e-06},
    /* ls           */ {2.206169e-10, 4.205849e-10, 1.986984e-09, 2.088403e-09, 2.567747e-09, 2.692659e-09, 3.588200e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {2.604668e-11},
    /* ls           */ {1.472085e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.308637e-12, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.472085e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.220618e-12, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.472085e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.643502e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.472085e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.193186e-11, 2.604668e-11},
    /* ls           */ {1.295675e-09, 1.472085e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.635965e-11, 2.604668e-11},
    /* ls           */ {1.436174e-09, 1.472085e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {2.604668e-11, 4.613400e-06},
    /* ls           */ {1.472085e-09, 4.911422e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.472085e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.308637e-12, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.472085e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.220618e-12, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.308637e-12, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.220618e-12, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.308637e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 6.220618e-12, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {5.308637e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {5.308637e-12, 6.220618e-12, 1.643502e-11, 2.193186e-11, 2.635965e-11, 2.604668e-11, 4.613400e-06},
    /* ls           */ {1.100022e-10, 2.238497e-10, 1.183197e-09, 1.295675e-09, 1.436174e-09, 1.472085e-09, 4.911422e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.851089e-11},
    /* ls           */ {1.146673e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.533423e-12, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.146673e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.265756e-12, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.146673e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.189379e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.146673e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.750302e-11, 1.851089e-11},
    /* ls           */ {1.041810e-09, 1.146673e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.042966e-11, 1.851089e-11},
    /* ls           */ {1.288429e-09, 1.146673e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.851089e-11, 7.688997e-06},
    /* ls           */ {1.146673e-09, 6.961822e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.146673e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.146673e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.533423e-12, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.146673e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 1.750302e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.265756e-12, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.533423e-12, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.265756e-12, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 4.265756e-12, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.533423e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.533423e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {3.533423e-12, 4.265756e-12, 1.189379e-11, 1.750302e-11, 2.042966e-11, 1.851089e-11, 7.688997e-06},
    /* ls           */ {7.301589e-11, 1.797765e-10, 8.689776e-10, 1.041810e-09, 1.288429e-09, 1.146673e-09, 6.961822e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.579307e-11},
    /* ls           */ {1.061221e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.653410e-12, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.061221e-09},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.294345e-12, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.061221e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.015133e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.061221e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.543769e-11, 1.579307e-11},
    /* ls           */ {1.004934e-09, 1.061221e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.842458e-11, 1.579307e-11},
    /* ls           */ {1.223974e-09, 1.061221e-09},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.579307e-11, 2.183914e-05},
    /* ls           */ {1.061221e-09, 2.493858e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.061221e-09},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.061221e-09},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.653410e-12, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.061221e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.294345e-12, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.653410e-12, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.294345e-12, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.653410e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.653410e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {2.653410e-12, 3.294345e-12, 1.015133e-11, 1.543769e-11, 1.842458e-11, 1.579307e-11, 2.183914e-05},
    /* ls           */ {5.488045e-11, 1.484338e-10, 7.595400e-10, 1.004934e-09, 1.223974e-09, 1.061221e-09, 2.493858e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.315468e-11},
    /* ls           */ {8.750512e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.775661e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 8.750512e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.280146e-12, 1.315468e-11},
    /* ls           */ {9.300533e-11, 8.750512e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.955418e-12, 1.315468e-11},
    /* ls           */ {4.417624e-10, 8.750512e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.255653e-11, 1.315468e-11},
    /* ls           */ {7.994690e-10, 8.750512e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.604392e-11, 1.315468e-11},
    /* ls           */ {1.054863e-09, 8.750512e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.315468e-11, 8.952619e-06},
    /* ls           */ {8.750512e-10, 8.249284e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 8.750512e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 8.750512e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.775661e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 8.750512e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 7.994690e-10, 8.750512e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.280146e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.775661e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.280146e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.775661e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 2.280146e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.775661e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {1.775661e-12, 2.280146e-12, 5.955418e-12, 1.255653e-11, 1.604392e-11, 1.315468e-11, 8.952619e-06},
    /* ls           */ {3.662919e-11, 9.300533e-11, 4.417624e-10, 7.994690e-10, 1.054863e-09, 8.750512e-10, 8.249284e-06},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.219101e-11},
    /* ls           */ {8.239890e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.182685e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 8.239890e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.501807e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 8.239890e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.604989e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 8.239890e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.450134e-12, 1.219101e-11},
    /* ls           */ {5.349837e-10, 8.239890e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {9.866466e-12, 1.219101e-11},
    /* ls           */ {6.667878e-10, 8.239890e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.219101e-11, 3.409386e-05},
    /* ls           */ {8.239890e-10, 3.490448e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 1.501807e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 8.239890e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 3.604989e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.182685e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 3.604989e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 5.349837e-10, 8.239890e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.501807e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 1.501807e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.182685e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.501807e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 1.501807e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.182685e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 1.501807e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.182685e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {1.182685e-12, 1.501807e-12, 3.604989e-12, 8.450134e-12, 9.866466e-12, 1.219101e-11, 3.409386e-05},
    /* ls           */ {2.449485e-11, 6.269940e-11, 2.887813e-10, 5.349837e-10, 6.667878e-10, 8.239890e-10, 3.490448e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {1.237560e-11},
    /* ls           */ {8.563975e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.869306e-13, 1.237560e-11},
    /* ls           */ {1.826369e-11, 8.563975e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.169244e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 8.563975e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {3.155775e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 8.563975e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {7.195628e-12, 1.237560e-11},
    /* ls           */ {4.529260e-10, 8.563975e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {9.605758e-12, 1.237560e-11},
    /* ls           */ {6.765814e-10, 8.563975e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {1.237560e-11, 6.256103e-05},
    /* ls           */ {8.563975e-10, 6.915331e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 1.169244e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 8.563975e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 3.155775e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.869306e-13, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 3.155775e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 4.529260e-10, 8.563975e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.169244e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 1.169244e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.869306e-13, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.169244e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 1.169244e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.869306e-13, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 1.169244e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.869306e-13, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {8.869306e-13, 1.169244e-12, 3.155775e-12, 7.195628e-12, 9.605758e-12, 1.237560e-11, 6.256103e-05},
    /* ls           */ {1.826369e-11, 4.908825e-11, 2.369889e-10, 4.529260e-10, 6.765814e-10, 8.563975e-10, 6.915331e-05},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {9.129807e-12},
    /* ls           */ {6.495532e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.644030e-13, 9.129807e-12},
    /* ls           */ {1.373541e-11, 6.495532e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {8.885908e-13, 9.129807e-12},
    /* ls           */ {3.665152e-11, 6.495532e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {2.108379e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 6.495532e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {5.605207e-12, 9.129807e-12},
    /* ls           */ {3.587535e-10, 6.495532e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {9.629964e-12, 9.129807e-12},
    /* ls           */ {7.137248e-10, 6.495532e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {9.129807e-12, 9.292364e-05},
    /* ls           */ {6.495532e-10, 1.113296e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 6.495532e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.644030e-13, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 3.587535e-10, 6.495532e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {8.885908e-13, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.644030e-13, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {8.885908e-13, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 8.885908e-13, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.644030e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 8.885908e-13, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.644030e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {6.644030e-13, 8.885908e-13, 2.108379e-12, 5.605207e-12, 9.629964e-12, 9.129807e-12, 9.292364e-05},
    /* ls           */ {1.373541e-11, 3.665152e-11, 1.684369e-10, 3.587535e-10, 7.137248e-10, 6.495532e-10, 1.113296e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* g            */ {9.184656e-12},
    /* ls           */ {6.728599e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.527422e-13, 9.184656e-12},
    /* ls           */ {9.486443e-12, 6.728599e-10},
    /* m            */ {65536, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {6.345705e-13, 9.184656e-12},
    /* ls           */ {2.705177e-11, 6.728599e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.439972e-12, 9.184656e-12},
    /* ls           */ {1.141888e-10, 6.728599e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* g            */ {4.554426e-12, 9.184656e-12},
    /* ls           */ {2.908892e-10, 6.728599e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* g            */ {1.131008e-11, 9.184656e-12},
    /* ls           */ {8.342130e-10, 6.728599e-10},
    /* m            */ {270582939648, 541165879296},
    /* p            */ {48, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* g            */ {9.184656e-12, 3.641129e-04},
    /* ls           */ {6.728599e-10, 2.368450e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 6.345705e-13, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 6.728599e-10},
    /* m            */ {65536, 524288, 541165879296},
    /* p            */ {1, 1, 96},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 1.439972e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.527422e-13, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 1.439972e-12, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 4.554426e-12, 9.184656e-12},
    /* ls           */ {2.705177e-11, 2.908892e-10, 6.728599e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 270582939648, 541165879296},
    /* p            */ {1, 48, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {6.345705e-13, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {25165824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* g            */ {4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {135291469824, 270582939648, 541165879296},
    /* p            */ {24, 2, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* g            */ {1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {270582939648, 541165879296, 541165879296},
    /* p            */ {48, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 270582939648, 541165879296},
    /* p            */ {1, 1, 48, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 6.345705e-13, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 541165879296, 541165879296},
    /* p            */ {1, 1, 96, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.527422e-13, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {6.345705e-13, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* g            */ {1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* g            */ {4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 48, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {4.527422e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* g            */ {1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 6.345705e-13, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {4.527422e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* g            */ {6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L1Cache, L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 7,
    /* g            */ {4.527422e-13, 6.345705e-13, 1.439972e-12, 4.554426e-12, 1.131008e-11, 9.184656e-12, 3.641129e-04},
    /* ls           */ {9.486443e-12, 2.705177e-11, 1.141888e-10, 2.908892e-10, 8.342130e-10, 6.728599e-10, 2.368450e-04},
    /* m            */ {65536, 524288, 25165824, 135291469824, 270582939648, 541165879296, 541165879296},
    /* p            */ {1, 1, 24, 1, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999, 999}}

};

// HWParameter_configurations struct
const cost_models::HW_model::HWParameter_configurations dis_system_params = {
    /* threads_options_num */ {1, 2, 4, 8, 12, 16, 24, 36, 48, 64, 96},
    /* policy_options_str */ {"close", "spread"},
    /* level_options_str  */ {"L1Cache", "L2Cache", "L3Cache", "NUMANode", "Socket", "NodeMem", "GLOBAL_SYNC"},
    /* hw_model_thread_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
    /* hw_model_policy_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    /* hw_model_level_bithash */ {32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127, 32, 33, 34, 36, 40, 48, 96, 35, 37, 41, 49, 97, 38, 42, 50, 98, 44, 52, 100, 56, 104, 112, 39, 43, 51, 99, 45, 53, 101, 57, 105, 113, 46, 54, 102, 58, 106, 114, 60, 108, 116, 120, 47, 55, 103, 59, 107, 115, 61, 109, 117, 121, 62, 110, 118, 122, 124, 63, 111, 119, 123, 125, 126, 127},
    /* hw_models */ hw_models_vector
};

#endif // HW_PARAMS_ARM920_HPP
