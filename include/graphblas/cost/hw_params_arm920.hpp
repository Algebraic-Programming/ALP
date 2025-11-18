
// Auto-generated hardware parameters for ARM920
// Allocation policies: close, spread
// Base levels: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
// ALL subset combinations (16 total)
// Generated on 2025-11-05 09:22:15
//
// Command used to generate this file:
// python3 runner.py --threads=96,64,48,36,32,24,16,12,8,4,2,1 --core-policy=close,spread --create-sync-level empirical_monotony --keep-levels=L2Cache,L3Cache,NUMANode,NodeMem,GLOBAL_SYNC

#ifndef HW_PARAMS_ARM920_HPP
#define HW_PARAMS_ARM920_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations and level subsets
const std::vector<cost_models::HW_model::HWParameters> hw_models_vector = {

    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.909695e-10},
    /* ls           */ {7.449206e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.659538e-11, 1.909695e-10},
    /* ls           */ {1.262599e-09, 7.449206e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {7.877291e-11, 1.909695e-10},
    /* ls           */ {3.145208e-09, 7.449206e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.184568e-10, 1.909695e-10},
    /* ls           */ {4.308728e-09, 7.449206e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.909695e-10, 0.000000e+00},
    /* ls           */ {7.449206e-09, 3.226598e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {4.659538e-11, 7.877291e-11, 1.909695e-10},
    /* ls           */ {1.262599e-09, 3.145208e-09, 7.449206e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.659538e-11, 1.184568e-10, 1.909695e-10},
    /* ls           */ {1.262599e-09, 4.308728e-09, 7.449206e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.659538e-11, 1.909695e-10, 0.000000e+00},
    /* ls           */ {1.262599e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {7.877291e-11, 1.184568e-10, 1.909695e-10},
    /* ls           */ {3.145208e-09, 4.308728e-09, 7.449206e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {7.877291e-11, 1.909695e-10, 0.000000e+00},
    /* ls           */ {3.145208e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.184568e-10, 1.909695e-10, 0.000000e+00},
    /* ls           */ {4.308728e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.659538e-11, 7.877291e-11, 1.184568e-10, 1.909695e-10},
    /* ls           */ {1.262599e-09, 3.145208e-09, 4.308728e-09, 7.449206e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.659538e-11, 7.877291e-11, 1.909695e-10, 0.000000e+00},
    /* ls           */ {1.262599e-09, 3.145208e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.659538e-11, 1.184568e-10, 1.909695e-10, 0.000000e+00},
    /* ls           */ {1.262599e-09, 4.308728e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {7.877291e-11, 1.184568e-10, 1.909695e-10, 0.000000e+00},
    /* ls           */ {3.145208e-09, 4.308728e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.659538e-11, 7.877291e-11, 1.184568e-10, 1.909695e-10, 0.000000e+00},
    /* ls           */ {1.262599e-09, 3.145208e-09, 4.308728e-09, 7.449206e-09, 3.226598e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.800125e-11},
    /* ls           */ {4.128642e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.306680e-11, 9.800125e-11},
    /* ls           */ {7.485876e-10, 4.128642e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.859121e-11, 9.800125e-11},
    /* ls           */ {3.284496e-09, 4.128642e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.611856e-11, 9.800125e-11},
    /* ls           */ {3.191796e-09, 4.128642e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.800125e-11, 0.000000e+00},
    /* ls           */ {4.128642e-09, 3.226598e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.306680e-11, 4.859121e-11, 9.800125e-11},
    /* ls           */ {7.485876e-10, 3.284496e-09, 4.128642e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.306680e-11, 6.611856e-11, 9.800125e-11},
    /* ls           */ {7.485876e-10, 3.191796e-09, 4.128642e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.306680e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {7.485876e-10, 4.128642e-09, 3.226598e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.859121e-11, 6.611856e-11, 9.800125e-11},
    /* ls           */ {3.284496e-09, 3.191796e-09, 4.128642e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.859121e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {3.284496e-09, 4.128642e-09, 3.226598e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.611856e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {3.191796e-09, 4.128642e-09, 3.226598e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.306680e-11, 4.859121e-11, 6.611856e-11, 9.800125e-11},
    /* ls           */ {7.485876e-10, 3.284496e-09, 3.191796e-09, 4.128642e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.306680e-11, 4.859121e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {7.485876e-10, 3.284496e-09, 4.128642e-09, 3.226598e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.306680e-11, 6.611856e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {7.485876e-10, 3.191796e-09, 4.128642e-09, 3.226598e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.859121e-11, 6.611856e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {3.284496e-09, 3.191796e-09, 4.128642e-09, 3.226598e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.306680e-11, 4.859121e-11, 6.611856e-11, 9.800125e-11, 0.000000e+00},
    /* ls           */ {7.485876e-10, 3.284496e-09, 3.191796e-09, 4.128642e-09, 3.226598e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {5.672049e-11},
    /* ls           */ {3.676243e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.259281e-11, 5.672049e-11},
    /* ls           */ {5.263005e-10, 3.676243e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.310630e-11, 5.672049e-11},
    /* ls           */ {3.170920e-09, 3.676243e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.532850e-11, 5.672049e-11},
    /* ls           */ {3.084812e-09, 3.676243e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.672049e-11, 0.000000e+00},
    /* ls           */ {3.676243e-09, 3.226598e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.259281e-11, 4.310630e-11, 5.672049e-11},
    /* ls           */ {5.263005e-10, 3.170920e-09, 3.676243e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.259281e-11, 4.532850e-11, 5.672049e-11},
    /* ls           */ {5.263005e-10, 3.084812e-09, 3.676243e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.259281e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {5.263005e-10, 3.676243e-09, 3.226598e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.310630e-11, 4.532850e-11, 5.672049e-11},
    /* ls           */ {3.170920e-09, 3.084812e-09, 3.676243e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.310630e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {3.170920e-09, 3.676243e-09, 3.226598e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.532850e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {3.084812e-09, 3.676243e-09, 3.226598e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.259281e-11, 4.310630e-11, 4.532850e-11, 5.672049e-11},
    /* ls           */ {5.263005e-10, 3.170920e-09, 3.084812e-09, 3.676243e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.259281e-11, 4.310630e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {5.263005e-10, 3.170920e-09, 3.676243e-09, 3.226598e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.259281e-11, 4.532850e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {5.263005e-10, 3.084812e-09, 3.676243e-09, 3.226598e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.310630e-11, 4.532850e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {3.170920e-09, 3.084812e-09, 3.676243e-09, 3.226598e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.259281e-11, 4.310630e-11, 4.532850e-11, 5.672049e-11, 0.000000e+00},
    /* ls           */ {5.263005e-10, 3.170920e-09, 3.084812e-09, 3.676243e-09, 3.226598e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.988310e-11},
    /* ls           */ {1.948085e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {6.563498e-12, 2.988310e-11},
    /* ls           */ {2.492739e-10, 1.948085e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.038517e-11, 2.988310e-11},
    /* ls           */ {1.534892e-09, 1.948085e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.746561e-11, 2.988310e-11},
    /* ls           */ {1.791342e-09, 1.948085e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.988310e-11, 0.000000e+00},
    /* ls           */ {1.948085e-09, 4.541874e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {6.563498e-12, 2.038517e-11, 2.988310e-11},
    /* ls           */ {2.492739e-10, 1.534892e-09, 1.948085e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {6.563498e-12, 2.746561e-11, 2.988310e-11},
    /* ls           */ {2.492739e-10, 1.791342e-09, 1.948085e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {6.563498e-12, 2.988310e-11, 0.000000e+00},
    /* ls           */ {2.492739e-10, 1.948085e-09, 4.541874e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.038517e-11, 2.746561e-11, 2.988310e-11},
    /* ls           */ {1.534892e-09, 1.791342e-09, 1.948085e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.038517e-11, 2.988310e-11, 0.000000e+00},
    /* ls           */ {1.534892e-09, 1.948085e-09, 4.541874e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.746561e-11, 2.988310e-11, 0.000000e+00},
    /* ls           */ {1.791342e-09, 1.948085e-09, 4.541874e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {6.563498e-12, 2.038517e-11, 2.746561e-11, 2.988310e-11},
    /* ls           */ {2.492739e-10, 1.534892e-09, 1.791342e-09, 1.948085e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {6.563498e-12, 2.038517e-11, 2.988310e-11, 0.000000e+00},
    /* ls           */ {2.492739e-10, 1.534892e-09, 1.948085e-09, 4.541874e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {6.563498e-12, 2.746561e-11, 2.988310e-11, 0.000000e+00},
    /* ls           */ {2.492739e-10, 1.791342e-09, 1.948085e-09, 4.541874e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.038517e-11, 2.746561e-11, 2.988310e-11, 0.000000e+00},
    /* ls           */ {1.534892e-09, 1.791342e-09, 1.948085e-09, 4.541874e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {6.563498e-12, 2.038517e-11, 2.746561e-11, 2.988310e-11, 0.000000e+00},
    /* ls           */ {2.492739e-10, 1.534892e-09, 1.791342e-09, 1.948085e-09, 4.541874e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.102634e-11},
    /* ls           */ {1.408417e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.490876e-12, 2.102634e-11},
    /* ls           */ {2.047731e-10, 1.408417e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.275688e-11, 2.102634e-11},
    /* ls           */ {9.883669e-10, 1.408417e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.200847e-11, 2.102634e-11},
    /* ls           */ {1.360610e-09, 1.408417e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.102634e-11, 0.000000e+00},
    /* ls           */ {1.408417e-09, 1.204014e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {4.490876e-12, 1.275688e-11, 2.102634e-11},
    /* ls           */ {2.047731e-10, 9.883669e-10, 1.408417e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.490876e-12, 2.200847e-11, 2.102634e-11},
    /* ls           */ {2.047731e-10, 1.360610e-09, 1.408417e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.490876e-12, 2.102634e-11, 0.000000e+00},
    /* ls           */ {2.047731e-10, 1.408417e-09, 1.204014e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.275688e-11, 2.200847e-11, 2.102634e-11},
    /* ls           */ {9.883669e-10, 1.360610e-09, 1.408417e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.275688e-11, 2.102634e-11, 0.000000e+00},
    /* ls           */ {9.883669e-10, 1.408417e-09, 1.204014e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.200847e-11, 2.102634e-11, 0.000000e+00},
    /* ls           */ {1.360610e-09, 1.408417e-09, 1.204014e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.490876e-12, 1.275688e-11, 2.200847e-11, 2.102634e-11},
    /* ls           */ {2.047731e-10, 9.883669e-10, 1.360610e-09, 1.408417e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.490876e-12, 1.275688e-11, 2.102634e-11, 0.000000e+00},
    /* ls           */ {2.047731e-10, 9.883669e-10, 1.408417e-09, 1.204014e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.490876e-12, 2.200847e-11, 2.102634e-11, 0.000000e+00},
    /* ls           */ {2.047731e-10, 1.360610e-09, 1.408417e-09, 1.204014e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.275688e-11, 2.200847e-11, 2.102634e-11, 0.000000e+00},
    /* ls           */ {9.883669e-10, 1.360610e-09, 1.408417e-09, 1.204014e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.490876e-12, 1.275688e-11, 2.200847e-11, 2.102634e-11, 0.000000e+00},
    /* ls           */ {2.047731e-10, 9.883669e-10, 1.360610e-09, 1.408417e-09, 1.204014e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.796711e-11},
    /* ls           */ {1.233585e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {3.404019e-12, 1.796711e-11},
    /* ls           */ {1.481933e-10, 1.233585e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {9.245633e-12, 1.796711e-11},
    /* ls           */ {7.172145e-10, 1.233585e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.943085e-11, 1.796711e-11},
    /* ls           */ {1.239793e-09, 1.233585e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.796711e-11, 0.000000e+00},
    /* ls           */ {1.233585e-09, 2.062321e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {3.404019e-12, 9.245633e-12, 1.796711e-11},
    /* ls           */ {1.481933e-10, 7.172145e-10, 1.233585e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.404019e-12, 1.943085e-11, 1.796711e-11},
    /* ls           */ {1.481933e-10, 1.239793e-09, 1.233585e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.404019e-12, 1.796711e-11, 0.000000e+00},
    /* ls           */ {1.481933e-10, 1.233585e-09, 2.062321e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.245633e-12, 1.943085e-11, 1.796711e-11},
    /* ls           */ {7.172145e-10, 1.239793e-09, 1.233585e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.245633e-12, 1.796711e-11, 0.000000e+00},
    /* ls           */ {7.172145e-10, 1.233585e-09, 2.062321e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.943085e-11, 1.796711e-11, 0.000000e+00},
    /* ls           */ {1.239793e-09, 1.233585e-09, 2.062321e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {3.404019e-12, 9.245633e-12, 1.943085e-11, 1.796711e-11},
    /* ls           */ {1.481933e-10, 7.172145e-10, 1.239793e-09, 1.233585e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {3.404019e-12, 9.245633e-12, 1.796711e-11, 0.000000e+00},
    /* ls           */ {1.481933e-10, 7.172145e-10, 1.233585e-09, 2.062321e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.404019e-12, 1.943085e-11, 1.796711e-11, 0.000000e+00},
    /* ls           */ {1.481933e-10, 1.239793e-09, 1.233585e-09, 2.062321e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {9.245633e-12, 1.943085e-11, 1.796711e-11, 0.000000e+00},
    /* ls           */ {7.172145e-10, 1.239793e-09, 1.233585e-09, 2.062321e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {3.404019e-12, 9.245633e-12, 1.943085e-11, 1.796711e-11, 0.000000e+00},
    /* ls           */ {1.481933e-10, 7.172145e-10, 1.239793e-09, 1.233585e-09, 2.062321e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.635115e-11},
    /* ls           */ {1.133943e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.478075e-12, 1.635115e-11},
    /* ls           */ {9.845543e-11, 1.133943e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {5.647574e-12, 1.635115e-11},
    /* ls           */ {4.512578e-10, 1.133943e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.722779e-11, 1.635115e-11},
    /* ls           */ {1.115377e-09, 1.133943e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.635115e-11, 0.000000e+00},
    /* ls           */ {1.133943e-09, 2.062321e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.478075e-12, 5.647574e-12, 1.635115e-11},
    /* ls           */ {9.845543e-11, 4.512578e-10, 1.133943e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.478075e-12, 1.722779e-11, 1.635115e-11},
    /* ls           */ {9.845543e-11, 1.115377e-09, 1.133943e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.478075e-12, 1.635115e-11, 0.000000e+00},
    /* ls           */ {9.845543e-11, 1.133943e-09, 2.062321e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.647574e-12, 1.722779e-11, 1.635115e-11},
    /* ls           */ {4.512578e-10, 1.115377e-09, 1.133943e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.647574e-12, 1.635115e-11, 0.000000e+00},
    /* ls           */ {4.512578e-10, 1.133943e-09, 2.062321e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.722779e-11, 1.635115e-11, 0.000000e+00},
    /* ls           */ {1.115377e-09, 1.133943e-09, 2.062321e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.478075e-12, 5.647574e-12, 1.722779e-11, 1.635115e-11},
    /* ls           */ {9.845543e-11, 4.512578e-10, 1.115377e-09, 1.133943e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.478075e-12, 5.647574e-12, 1.635115e-11, 0.000000e+00},
    /* ls           */ {9.845543e-11, 4.512578e-10, 1.133943e-09, 2.062321e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.478075e-12, 1.722779e-11, 1.635115e-11, 0.000000e+00},
    /* ls           */ {9.845543e-11, 1.115377e-09, 1.133943e-09, 2.062321e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {5.647574e-12, 1.722779e-11, 1.635115e-11, 0.000000e+00},
    /* ls           */ {4.512578e-10, 1.115377e-09, 1.133943e-09, 2.062321e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.478075e-12, 5.647574e-12, 1.722779e-11, 1.635115e-11, 0.000000e+00},
    /* ls           */ {9.845543e-11, 4.512578e-10, 1.115377e-09, 1.133943e-09, 2.062321e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.542167e-11},
    /* ls           */ {1.049554e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.805895e-12, 1.542167e-11},
    /* ls           */ {7.409751e-11, 1.049554e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {3.469091e-12, 1.542167e-11},
    /* ls           */ {3.164791e-10, 1.049554e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.183266e-11, 1.542167e-11},
    /* ls           */ {7.712585e-10, 1.049554e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.542167e-11, 0.000000e+00},
    /* ls           */ {1.049554e-09, 4.085898e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.805895e-12, 3.469091e-12, 1.542167e-11},
    /* ls           */ {7.409751e-11, 3.164791e-10, 1.049554e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.805895e-12, 1.183266e-11, 1.542167e-11},
    /* ls           */ {7.409751e-11, 7.712585e-10, 1.049554e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.805895e-12, 1.542167e-11, 0.000000e+00},
    /* ls           */ {7.409751e-11, 1.049554e-09, 4.085898e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.469091e-12, 1.183266e-11, 1.542167e-11},
    /* ls           */ {3.164791e-10, 7.712585e-10, 1.049554e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.469091e-12, 1.542167e-11, 0.000000e+00},
    /* ls           */ {3.164791e-10, 1.049554e-09, 4.085898e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.183266e-11, 1.542167e-11, 0.000000e+00},
    /* ls           */ {7.712585e-10, 1.049554e-09, 4.085898e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.805895e-12, 3.469091e-12, 1.183266e-11, 1.542167e-11},
    /* ls           */ {7.409751e-11, 3.164791e-10, 7.712585e-10, 1.049554e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.805895e-12, 3.469091e-12, 1.542167e-11, 0.000000e+00},
    /* ls           */ {7.409751e-11, 3.164791e-10, 1.049554e-09, 4.085898e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.805895e-12, 1.183266e-11, 1.542167e-11, 0.000000e+00},
    /* ls           */ {7.409751e-11, 7.712585e-10, 1.049554e-09, 4.085898e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.469091e-12, 1.183266e-11, 1.542167e-11, 0.000000e+00},
    /* ls           */ {3.164791e-10, 7.712585e-10, 1.049554e-09, 4.085898e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.805895e-12, 3.469091e-12, 1.183266e-11, 1.542167e-11, 0.000000e+00},
    /* ls           */ {7.409751e-11, 3.164791e-10, 7.712585e-10, 1.049554e-09, 4.085898e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.562589e-11},
    /* ls           */ {1.064283e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.587220e-12, 1.562589e-11},
    /* ls           */ {6.611071e-11, 1.064283e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.715279e-12, 1.562589e-11},
    /* ls           */ {2.625493e-10, 1.064283e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.041626e-11, 1.562589e-11},
    /* ls           */ {6.766786e-10, 1.064283e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.562589e-11, 0.000000e+00},
    /* ls           */ {1.064283e-09, 4.085898e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.587220e-12, 2.715279e-12, 1.562589e-11},
    /* ls           */ {6.611071e-11, 2.625493e-10, 1.064283e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.587220e-12, 1.041626e-11, 1.562589e-11},
    /* ls           */ {6.611071e-11, 6.766786e-10, 1.064283e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.587220e-12, 1.562589e-11, 0.000000e+00},
    /* ls           */ {6.611071e-11, 1.064283e-09, 4.085898e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.715279e-12, 1.041626e-11, 1.562589e-11},
    /* ls           */ {2.625493e-10, 6.766786e-10, 1.064283e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.715279e-12, 1.562589e-11, 0.000000e+00},
    /* ls           */ {2.625493e-10, 1.064283e-09, 4.085898e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.041626e-11, 1.562589e-11, 0.000000e+00},
    /* ls           */ {6.766786e-10, 1.064283e-09, 4.085898e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.587220e-12, 2.715279e-12, 1.041626e-11, 1.562589e-11},
    /* ls           */ {6.611071e-11, 2.625493e-10, 6.766786e-10, 1.064283e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.587220e-12, 2.715279e-12, 1.562589e-11, 0.000000e+00},
    /* ls           */ {6.611071e-11, 2.625493e-10, 1.064283e-09, 4.085898e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.587220e-12, 1.041626e-11, 1.562589e-11, 0.000000e+00},
    /* ls           */ {6.611071e-11, 6.766786e-10, 1.064283e-09, 4.085898e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.715279e-12, 1.041626e-11, 1.562589e-11, 0.000000e+00},
    /* ls           */ {2.625493e-10, 6.766786e-10, 1.064283e-09, 4.085898e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.587220e-12, 2.715279e-12, 1.041626e-11, 1.562589e-11, 0.000000e+00},
    /* ls           */ {6.611071e-11, 2.625493e-10, 6.766786e-10, 1.064283e-09, 4.085898e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.601515e-11},
    /* ls           */ {1.100323e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.240932e-12, 1.601515e-11},
    /* ls           */ {4.945936e-11, 1.100323e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.835931e-12, 1.601515e-11},
    /* ls           */ {2.252136e-10, 1.100323e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.681732e-12, 1.601515e-11},
    /* ls           */ {5.604646e-10, 1.100323e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.601515e-11, 0.000000e+00},
    /* ls           */ {1.100323e-09, 5.775690e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.240932e-12, 2.835931e-12, 1.601515e-11},
    /* ls           */ {4.945936e-11, 2.252136e-10, 1.100323e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.240932e-12, 8.681732e-12, 1.601515e-11},
    /* ls           */ {4.945936e-11, 5.604646e-10, 1.100323e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.240932e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {4.945936e-11, 1.100323e-09, 5.775690e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.835931e-12, 8.681732e-12, 1.601515e-11},
    /* ls           */ {2.252136e-10, 5.604646e-10, 1.100323e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.835931e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {2.252136e-10, 1.100323e-09, 5.775690e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {8.681732e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {5.604646e-10, 1.100323e-09, 5.775690e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.240932e-12, 2.835931e-12, 8.681732e-12, 1.601515e-11},
    /* ls           */ {4.945936e-11, 2.252136e-10, 5.604646e-10, 1.100323e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.240932e-12, 2.835931e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {4.945936e-11, 2.252136e-10, 1.100323e-09, 5.775690e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.240932e-12, 8.681732e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {4.945936e-11, 5.604646e-10, 1.100323e-09, 5.775690e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.835931e-12, 8.681732e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {2.252136e-10, 5.604646e-10, 1.100323e-09, 5.775690e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.240932e-12, 2.835931e-12, 8.681732e-12, 1.601515e-11, 0.000000e+00},
    /* ls           */ {4.945936e-11, 2.252136e-10, 5.604646e-10, 1.100323e-09, 5.775690e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.373831e-12},
    /* ls           */ {6.465906e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {9.203036e-13, 9.373831e-12},
    /* ls           */ {3.694605e-11, 6.465906e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.752926e-12, 9.373831e-12},
    /* ls           */ {1.576825e-10, 6.465906e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.173943e-12, 9.373831e-12},
    /* ls           */ {3.980040e-10, 6.465906e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.373831e-12, 0.000000e+00},
    /* ls           */ {6.465906e-10, 8.903742e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {9.203036e-13, 1.752926e-12, 9.373831e-12},
    /* ls           */ {3.694605e-11, 1.576825e-10, 6.465906e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.203036e-13, 6.173943e-12, 9.373831e-12},
    /* ls           */ {3.694605e-11, 3.980040e-10, 6.465906e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {9.203036e-13, 9.373831e-12, 0.000000e+00},
    /* ls           */ {3.694605e-11, 6.465906e-10, 8.903742e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.752926e-12, 6.173943e-12, 9.373831e-12},
    /* ls           */ {1.576825e-10, 3.980040e-10, 6.465906e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.752926e-12, 9.373831e-12, 0.000000e+00},
    /* ls           */ {1.576825e-10, 6.465906e-10, 8.903742e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.173943e-12, 9.373831e-12, 0.000000e+00},
    /* ls           */ {3.980040e-10, 6.465906e-10, 8.903742e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {9.203036e-13, 1.752926e-12, 6.173943e-12, 9.373831e-12},
    /* ls           */ {3.694605e-11, 1.576825e-10, 3.980040e-10, 6.465906e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {9.203036e-13, 1.752926e-12, 9.373831e-12, 0.000000e+00},
    /* ls           */ {3.694605e-11, 1.576825e-10, 6.465906e-10, 8.903742e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {9.203036e-13, 6.173943e-12, 9.373831e-12, 0.000000e+00},
    /* ls           */ {3.694605e-11, 3.980040e-10, 6.465906e-10, 8.903742e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.752926e-12, 6.173943e-12, 9.373831e-12, 0.000000e+00},
    /* ls           */ {1.576825e-10, 3.980040e-10, 6.465906e-10, 8.903742e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {9.203036e-13, 1.752926e-12, 6.173943e-12, 9.373831e-12, 0.000000e+00},
    /* ls           */ {3.694605e-11, 1.576825e-10, 3.980040e-10, 6.465906e-10, 8.903742e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.172742e-12},
    /* ls           */ {6.712515e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {6.195075e-13, 9.172742e-12},
    /* ls           */ {2.668596e-11, 6.712515e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.429457e-12, 9.172742e-12},
    /* ls           */ {1.139079e-10, 6.712515e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.548665e-12, 9.172742e-12},
    /* ls           */ {2.906231e-10, 6.712515e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.172742e-12, 0.000000e+00},
    /* ls           */ {6.712515e-10, 2.831340e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {6.195075e-13, 1.429457e-12, 9.172742e-12},
    /* ls           */ {2.668596e-11, 1.139079e-10, 6.712515e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {6.195075e-13, 4.548665e-12, 9.172742e-12},
    /* ls           */ {2.668596e-11, 2.906231e-10, 6.712515e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {6.195075e-13, 9.172742e-12, 0.000000e+00},
    /* ls           */ {2.668596e-11, 6.712515e-10, 2.831340e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.429457e-12, 4.548665e-12, 9.172742e-12},
    /* ls           */ {1.139079e-10, 2.906231e-10, 6.712515e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.429457e-12, 9.172742e-12, 0.000000e+00},
    /* ls           */ {1.139079e-10, 6.712515e-10, 2.831340e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.548665e-12, 9.172742e-12, 0.000000e+00},
    /* ls           */ {2.906231e-10, 6.712515e-10, 2.831340e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {6.195075e-13, 1.429457e-12, 4.548665e-12, 9.172742e-12},
    /* ls           */ {2.668596e-11, 1.139079e-10, 2.906231e-10, 6.712515e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {6.195075e-13, 1.429457e-12, 9.172742e-12, 0.000000e+00},
    /* ls           */ {2.668596e-11, 1.139079e-10, 6.712515e-10, 2.831340e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {6.195075e-13, 4.548665e-12, 9.172742e-12, 0.000000e+00},
    /* ls           */ {2.668596e-11, 2.906231e-10, 6.712515e-10, 2.831340e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.429457e-12, 4.548665e-12, 9.172742e-12, 0.000000e+00},
    /* ls           */ {1.139079e-10, 2.906231e-10, 6.712515e-10, 2.831340e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {6.195075e-13, 1.429457e-12, 4.548665e-12, 9.172742e-12, 0.000000e+00},
    /* ls           */ {2.668596e-11, 1.139079e-10, 2.906231e-10, 6.712515e-10, 2.831340e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.805582e-10},
    /* ls           */ {7.555633e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.749077e-11, 1.805582e-10},
    /* ls           */ {1.177522e-09, 7.555633e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {7.862377e-11, 1.805582e-10},
    /* ls           */ {3.156902e-09, 7.555633e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.182302e-10, 1.805582e-10},
    /* ls           */ {4.446325e-09, 7.555633e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.805582e-10, 0.000000e+00},
    /* ls           */ {7.555633e-09, 3.612042e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {4.749077e-11, 7.862377e-11, 1.805582e-10},
    /* ls           */ {1.177522e-09, 3.156902e-09, 7.555633e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.749077e-11, 1.182302e-10, 1.805582e-10},
    /* ls           */ {1.177522e-09, 4.446325e-09, 7.555633e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.749077e-11, 1.805582e-10, 0.000000e+00},
    /* ls           */ {1.177522e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.862377e-11, 1.182302e-10, 1.805582e-10},
    /* ls           */ {3.156902e-09, 4.446325e-09, 7.555633e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.862377e-11, 1.805582e-10, 0.000000e+00},
    /* ls           */ {3.156902e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.182302e-10, 1.805582e-10, 0.000000e+00},
    /* ls           */ {4.446325e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.749077e-11, 7.862377e-11, 1.182302e-10, 1.805582e-10},
    /* ls           */ {1.177522e-09, 3.156902e-09, 4.446325e-09, 7.555633e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.749077e-11, 7.862377e-11, 1.805582e-10, 0.000000e+00},
    /* ls           */ {1.177522e-09, 3.156902e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.749077e-11, 1.182302e-10, 1.805582e-10, 0.000000e+00},
    /* ls           */ {1.177522e-09, 4.446325e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.862377e-11, 1.182302e-10, 1.805582e-10, 0.000000e+00},
    /* ls           */ {3.156902e-09, 4.446325e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.749077e-11, 7.862377e-11, 1.182302e-10, 1.805582e-10, 0.000000e+00},
    /* ls           */ {1.177522e-09, 3.156902e-09, 4.446325e-09, 7.555633e-09, 3.612042e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.150356e-11},
    /* ls           */ {3.789501e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.344286e-11, 9.150356e-11},
    /* ls           */ {6.305693e-10, 3.789501e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {5.382400e-11, 9.150356e-11},
    /* ls           */ {2.111440e-09, 3.789501e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.628249e-11, 9.150356e-11},
    /* ls           */ {2.601375e-09, 3.789501e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.150356e-11, 0.000000e+00},
    /* ls           */ {3.789501e-09, 4.291535e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {2.344286e-11, 5.382400e-11, 9.150356e-11},
    /* ls           */ {6.305693e-10, 2.111440e-09, 3.789501e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.344286e-11, 6.628249e-11, 9.150356e-11},
    /* ls           */ {6.305693e-10, 2.601375e-09, 3.789501e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.344286e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {6.305693e-10, 3.789501e-09, 4.291535e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.382400e-11, 6.628249e-11, 9.150356e-11},
    /* ls           */ {2.111440e-09, 2.601375e-09, 3.789501e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.382400e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {2.111440e-09, 3.789501e-09, 4.291535e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.628249e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {2.601375e-09, 3.789501e-09, 4.291535e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.344286e-11, 5.382400e-11, 6.628249e-11, 9.150356e-11},
    /* ls           */ {6.305693e-10, 2.111440e-09, 2.601375e-09, 3.789501e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.344286e-11, 5.382400e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {6.305693e-10, 2.111440e-09, 3.789501e-09, 4.291535e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.344286e-11, 6.628249e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {6.305693e-10, 2.601375e-09, 3.789501e-09, 4.291535e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.382400e-11, 6.628249e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {2.111440e-09, 2.601375e-09, 3.789501e-09, 4.291535e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.344286e-11, 5.382400e-11, 6.628249e-11, 9.150356e-11, 0.000000e+00},
    /* ls           */ {6.305693e-10, 2.111440e-09, 2.601375e-09, 3.789501e-09, 4.291535e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {4.283082e-11},
    /* ls           */ {1.709075e-09},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.174514e-11, 4.283082e-11},
    /* ls           */ {3.148693e-10, 1.709075e-09},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.971691e-11, 4.283082e-11},
    /* ls           */ {8.030478e-10, 1.709075e-09},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.959100e-11, 4.283082e-11},
    /* ls           */ {1.091995e-09, 1.709075e-09},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.283082e-11, 0.000000e+00},
    /* ls           */ {1.709075e-09, 7.259846e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.174514e-11, 1.971691e-11, 4.283082e-11},
    /* ls           */ {3.148693e-10, 8.030478e-10, 1.709075e-09},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.174514e-11, 2.959100e-11, 4.283082e-11},
    /* ls           */ {3.148693e-10, 1.091995e-09, 1.709075e-09},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.174514e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {3.148693e-10, 1.709075e-09, 7.259846e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.971691e-11, 2.959100e-11, 4.283082e-11},
    /* ls           */ {8.030478e-10, 1.091995e-09, 1.709075e-09},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.971691e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {8.030478e-10, 1.709075e-09, 7.259846e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.959100e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {1.091995e-09, 1.709075e-09, 7.259846e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.174514e-11, 1.971691e-11, 2.959100e-11, 4.283082e-11},
    /* ls           */ {3.148693e-10, 8.030478e-10, 1.091995e-09, 1.709075e-09},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.174514e-11, 1.971691e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {3.148693e-10, 8.030478e-10, 1.709075e-09, 7.259846e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.174514e-11, 2.959100e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {3.148693e-10, 1.091995e-09, 1.709075e-09, 7.259846e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.971691e-11, 2.959100e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {8.030478e-10, 1.091995e-09, 1.709075e-09, 7.259846e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.174514e-11, 1.971691e-11, 2.959100e-11, 4.283082e-11, 0.000000e+00},
    /* ls           */ {3.148693e-10, 8.030478e-10, 1.091995e-09, 1.709075e-09, 7.259846e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.221025e-11},
    /* ls           */ {9.960855e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {5.877739e-12, 2.221025e-11},
    /* ls           */ {1.984254e-10, 9.960855e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.248487e-11, 2.221025e-11},
    /* ls           */ {8.315024e-10, 9.960855e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.639812e-11, 2.221025e-11},
    /* ls           */ {8.000075e-10, 9.960855e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.221025e-11, 0.000000e+00},
    /* ls           */ {9.960855e-10, 8.511543e-06},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {5.877739e-12, 1.248487e-11, 2.221025e-11},
    /* ls           */ {1.984254e-10, 8.315024e-10, 9.960855e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.877739e-12, 1.639812e-11, 2.221025e-11},
    /* ls           */ {1.984254e-10, 8.000075e-10, 9.960855e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.877739e-12, 2.221025e-11, 0.000000e+00},
    /* ls           */ {1.984254e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.248487e-11, 1.639812e-11, 2.221025e-11},
    /* ls           */ {8.315024e-10, 8.000075e-10, 9.960855e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.248487e-11, 2.221025e-11, 0.000000e+00},
    /* ls           */ {8.315024e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.639812e-11, 2.221025e-11, 0.000000e+00},
    /* ls           */ {8.000075e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {5.877739e-12, 1.248487e-11, 1.639812e-11, 2.221025e-11},
    /* ls           */ {1.984254e-10, 8.315024e-10, 8.000075e-10, 9.960855e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {5.877739e-12, 1.248487e-11, 2.221025e-11, 0.000000e+00},
    /* ls           */ {1.984254e-10, 8.315024e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.877739e-12, 1.639812e-11, 2.221025e-11, 0.000000e+00},
    /* ls           */ {1.984254e-10, 8.000075e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.248487e-11, 1.639812e-11, 2.221025e-11, 0.000000e+00},
    /* ls           */ {8.315024e-10, 8.000075e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {5.877739e-12, 1.248487e-11, 1.639812e-11, 2.221025e-11, 0.000000e+00},
    /* ls           */ {1.984254e-10, 8.315024e-10, 8.000075e-10, 9.960855e-10, 8.511543e-06},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.599544e-11},
    /* ls           */ {8.849290e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.040636e-12, 1.599544e-11},
    /* ls           */ {1.547798e-10, 8.849290e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.103070e-11, 1.599544e-11},
    /* ls           */ {7.495883e-10, 8.849290e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.299757e-11, 1.599544e-11},
    /* ls           */ {7.230107e-10, 8.849290e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.599544e-11, 0.000000e+00},
    /* ls           */ {8.849290e-10, 1.308918e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {4.040636e-12, 1.103070e-11, 1.599544e-11},
    /* ls           */ {1.547798e-10, 7.495883e-10, 8.849290e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.040636e-12, 1.299757e-11, 1.599544e-11},
    /* ls           */ {1.547798e-10, 7.230107e-10, 8.849290e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.040636e-12, 1.599544e-11, 0.000000e+00},
    /* ls           */ {1.547798e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.103070e-11, 1.299757e-11, 1.599544e-11},
    /* ls           */ {7.495883e-10, 7.230107e-10, 8.849290e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.103070e-11, 1.599544e-11, 0.000000e+00},
    /* ls           */ {7.495883e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.299757e-11, 1.599544e-11, 0.000000e+00},
    /* ls           */ {7.230107e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.040636e-12, 1.103070e-11, 1.299757e-11, 1.599544e-11},
    /* ls           */ {1.547798e-10, 7.495883e-10, 7.230107e-10, 8.849290e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.040636e-12, 1.103070e-11, 1.599544e-11, 0.000000e+00},
    /* ls           */ {1.547798e-10, 7.495883e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.040636e-12, 1.299757e-11, 1.599544e-11, 0.000000e+00},
    /* ls           */ {1.547798e-10, 7.230107e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.103070e-11, 1.299757e-11, 1.599544e-11, 0.000000e+00},
    /* ls           */ {7.495883e-10, 7.230107e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 12 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.040636e-12, 1.103070e-11, 1.299757e-11, 1.599544e-11, 0.000000e+00},
    /* ls           */ {1.547798e-10, 7.495883e-10, 7.230107e-10, 8.849290e-10, 1.308918e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.361904e-11},
    /* ls           */ {8.888564e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {3.184671e-12, 1.361904e-11},
    /* ls           */ {1.486742e-10, 8.888564e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.105703e-11, 1.361904e-11},
    /* ls           */ {8.018654e-10, 8.888564e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.144454e-11, 1.361904e-11},
    /* ls           */ {7.700748e-10, 8.888564e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.361904e-11, 0.000000e+00},
    /* ls           */ {8.888564e-10, 1.476407e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {3.184671e-12, 1.105703e-11, 1.361904e-11},
    /* ls           */ {1.486742e-10, 8.018654e-10, 8.888564e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.184671e-12, 1.144454e-11, 1.361904e-11},
    /* ls           */ {1.486742e-10, 7.700748e-10, 8.888564e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.184671e-12, 1.361904e-11, 0.000000e+00},
    /* ls           */ {1.486742e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.105703e-11, 1.144454e-11, 1.361904e-11},
    /* ls           */ {8.018654e-10, 7.700748e-10, 8.888564e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.105703e-11, 1.361904e-11, 0.000000e+00},
    /* ls           */ {8.018654e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.144454e-11, 1.361904e-11, 0.000000e+00},
    /* ls           */ {7.700748e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {3.184671e-12, 1.105703e-11, 1.144454e-11, 1.361904e-11},
    /* ls           */ {1.486742e-10, 8.018654e-10, 7.700748e-10, 8.888564e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {3.184671e-12, 1.105703e-11, 1.361904e-11, 0.000000e+00},
    /* ls           */ {1.486742e-10, 8.018654e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.184671e-12, 1.144454e-11, 1.361904e-11, 0.000000e+00},
    /* ls           */ {1.486742e-10, 7.700748e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.105703e-11, 1.144454e-11, 1.361904e-11, 0.000000e+00},
    /* ls           */ {8.018654e-10, 7.700748e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 16 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {3.184671e-12, 1.105703e-11, 1.144454e-11, 1.361904e-11, 0.000000e+00},
    /* ls           */ {1.486742e-10, 8.018654e-10, 7.700748e-10, 8.888564e-10, 1.476407e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.958215e-12},
    /* ls           */ {6.161590e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.082217e-12, 9.958215e-12},
    /* ls           */ {8.755524e-11, 6.161590e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {6.263263e-12, 9.958215e-12},
    /* ls           */ {4.322670e-10, 6.161590e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.885280e-12, 9.958215e-12},
    /* ls           */ {4.835609e-10, 6.161590e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.958215e-12, 0.000000e+00},
    /* ls           */ {6.161590e-10, 1.476407e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {2.082217e-12, 6.263263e-12, 9.958215e-12},
    /* ls           */ {8.755524e-11, 4.322670e-10, 6.161590e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.082217e-12, 7.885280e-12, 9.958215e-12},
    /* ls           */ {8.755524e-11, 4.835609e-10, 6.161590e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.082217e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {8.755524e-11, 6.161590e-10, 1.476407e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {6.263263e-12, 7.885280e-12, 9.958215e-12},
    /* ls           */ {4.322670e-10, 4.835609e-10, 6.161590e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {6.263263e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {4.322670e-10, 6.161590e-10, 1.476407e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.885280e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {4.835609e-10, 6.161590e-10, 1.476407e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.082217e-12, 6.263263e-12, 7.885280e-12, 9.958215e-12},
    /* ls           */ {8.755524e-11, 4.322670e-10, 4.835609e-10, 6.161590e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.082217e-12, 6.263263e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {8.755524e-11, 4.322670e-10, 6.161590e-10, 1.476407e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.082217e-12, 7.885280e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {8.755524e-11, 4.835609e-10, 6.161590e-10, 1.476407e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {6.263263e-12, 7.885280e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {4.322670e-10, 4.835609e-10, 6.161590e-10, 1.476407e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.082217e-12, 6.263263e-12, 7.885280e-12, 9.958215e-12, 0.000000e+00},
    /* ls           */ {8.755524e-11, 4.322670e-10, 4.835609e-10, 6.161590e-10, 1.476407e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.069817e-12},
    /* ls           */ {6.102702e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.617303e-12, 9.069817e-12},
    /* ls           */ {7.463461e-11, 6.102702e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {5.419625e-12, 9.069817e-12},
    /* ls           */ {3.901082e-10, 6.102702e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.044473e-12, 9.069817e-12},
    /* ls           */ {4.492068e-10, 6.102702e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.069817e-12, 0.000000e+00},
    /* ls           */ {6.102702e-10, 3.756285e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.617303e-12, 5.419625e-12, 9.069817e-12},
    /* ls           */ {7.463461e-11, 3.901082e-10, 6.102702e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.617303e-12, 7.044473e-12, 9.069817e-12},
    /* ls           */ {7.463461e-11, 4.492068e-10, 6.102702e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.617303e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {7.463461e-11, 6.102702e-10, 3.756285e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.419625e-12, 7.044473e-12, 9.069817e-12},
    /* ls           */ {3.901082e-10, 4.492068e-10, 6.102702e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.419625e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {3.901082e-10, 6.102702e-10, 3.756285e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.044473e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {4.492068e-10, 6.102702e-10, 3.756285e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.617303e-12, 5.419625e-12, 7.044473e-12, 9.069817e-12},
    /* ls           */ {7.463461e-11, 3.901082e-10, 4.492068e-10, 6.102702e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.617303e-12, 5.419625e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {7.463461e-11, 3.901082e-10, 6.102702e-10, 3.756285e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.617303e-12, 7.044473e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {7.463461e-11, 4.492068e-10, 6.102702e-10, 3.756285e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.419625e-12, 7.044473e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {3.901082e-10, 4.492068e-10, 6.102702e-10, 3.756285e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.617303e-12, 5.419625e-12, 7.044473e-12, 9.069817e-12, 0.000000e+00},
    /* ls           */ {7.463461e-11, 3.901082e-10, 4.492068e-10, 6.102702e-10, 3.756285e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {8.756127e-12},
    /* ls           */ {5.836949e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.416395e-12, 8.756127e-12},
    /* ls           */ {5.928809e-11, 5.836949e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.494698e-12, 8.756127e-12},
    /* ls           */ {3.150132e-10, 5.836949e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.484011e-12, 8.756127e-12},
    /* ls           */ {3.932887e-10, 5.836949e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.756127e-12, 0.000000e+00},
    /* ls           */ {5.836949e-10, 3.756285e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.416395e-12, 4.494698e-12, 8.756127e-12},
    /* ls           */ {5.928809e-11, 3.150132e-10, 5.836949e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.416395e-12, 6.484011e-12, 8.756127e-12},
    /* ls           */ {5.928809e-11, 3.932887e-10, 5.836949e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.416395e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {5.928809e-11, 5.836949e-10, 3.756285e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.494698e-12, 6.484011e-12, 8.756127e-12},
    /* ls           */ {3.150132e-10, 3.932887e-10, 5.836949e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.494698e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {3.150132e-10, 5.836949e-10, 3.756285e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.484011e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {3.932887e-10, 5.836949e-10, 3.756285e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.416395e-12, 4.494698e-12, 6.484011e-12, 8.756127e-12},
    /* ls           */ {5.928809e-11, 3.150132e-10, 3.932887e-10, 5.836949e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.416395e-12, 4.494698e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {5.928809e-11, 3.150132e-10, 5.836949e-10, 3.756285e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.416395e-12, 6.484011e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {5.928809e-11, 3.932887e-10, 5.836949e-10, 3.756285e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.494698e-12, 6.484011e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {3.150132e-10, 3.932887e-10, 5.836949e-10, 3.756285e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 36 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.416395e-12, 4.494698e-12, 6.484011e-12, 8.756127e-12, 0.000000e+00},
    /* ls           */ {5.928809e-11, 3.150132e-10, 3.932887e-10, 5.836949e-10, 3.756285e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {8.736049e-12},
    /* ls           */ {6.124715e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.097556e-12, 8.736049e-12},
    /* ls           */ {4.871714e-11, 6.124715e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {3.475620e-12, 8.736049e-12},
    /* ls           */ {2.487641e-10, 6.124715e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.709525e-12, 8.736049e-12},
    /* ls           */ {3.453875e-10, 6.124715e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.736049e-12, 0.000000e+00},
    /* ls           */ {6.124715e-10, 8.285046e-05},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.097556e-12, 3.475620e-12, 8.736049e-12},
    /* ls           */ {4.871714e-11, 2.487641e-10, 6.124715e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.097556e-12, 5.709525e-12, 8.736049e-12},
    /* ls           */ {4.871714e-11, 3.453875e-10, 6.124715e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.097556e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {4.871714e-11, 6.124715e-10, 8.285046e-05},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.475620e-12, 5.709525e-12, 8.736049e-12},
    /* ls           */ {2.487641e-10, 3.453875e-10, 6.124715e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.475620e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {2.487641e-10, 6.124715e-10, 8.285046e-05},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.709525e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {3.453875e-10, 6.124715e-10, 8.285046e-05},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.097556e-12, 3.475620e-12, 5.709525e-12, 8.736049e-12},
    /* ls           */ {4.871714e-11, 2.487641e-10, 3.453875e-10, 6.124715e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.097556e-12, 3.475620e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {4.871714e-11, 2.487641e-10, 6.124715e-10, 8.285046e-05},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.097556e-12, 5.709525e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {4.871714e-11, 3.453875e-10, 6.124715e-10, 8.285046e-05},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.475620e-12, 5.709525e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {2.487641e-10, 3.453875e-10, 6.124715e-10, 8.285046e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 48 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.097556e-12, 3.475620e-12, 5.709525e-12, 8.736049e-12, 0.000000e+00},
    /* ls           */ {4.871714e-11, 2.487641e-10, 3.453875e-10, 6.124715e-10, 8.285046e-05},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {8.885783e-12},
    /* ls           */ {6.525157e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {8.568781e-13, 8.885783e-12},
    /* ls           */ {3.635699e-11, 6.525157e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.463833e-12, 8.885783e-12},
    /* ls           */ {1.791912e-10, 6.525157e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.036471e-12, 8.885783e-12},
    /* ls           */ {3.195031e-10, 6.525157e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.885783e-12, 0.000000e+00},
    /* ls           */ {6.525157e-10, 1.384139e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {8.568781e-13, 2.463833e-12, 8.885783e-12},
    /* ls           */ {3.635699e-11, 1.791912e-10, 6.525157e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {8.568781e-13, 5.036471e-12, 8.885783e-12},
    /* ls           */ {3.635699e-11, 3.195031e-10, 6.525157e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {8.568781e-13, 8.885783e-12, 0.000000e+00},
    /* ls           */ {3.635699e-11, 6.525157e-10, 1.384139e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.463833e-12, 5.036471e-12, 8.885783e-12},
    /* ls           */ {1.791912e-10, 3.195031e-10, 6.525157e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.463833e-12, 8.885783e-12, 0.000000e+00},
    /* ls           */ {1.791912e-10, 6.525157e-10, 1.384139e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.036471e-12, 8.885783e-12, 0.000000e+00},
    /* ls           */ {3.195031e-10, 6.525157e-10, 1.384139e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {8.568781e-13, 2.463833e-12, 5.036471e-12, 8.885783e-12},
    /* ls           */ {3.635699e-11, 1.791912e-10, 3.195031e-10, 6.525157e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {8.568781e-13, 2.463833e-12, 8.885783e-12, 0.000000e+00},
    /* ls           */ {3.635699e-11, 1.791912e-10, 6.525157e-10, 1.384139e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {8.568781e-13, 5.036471e-12, 8.885783e-12, 0.000000e+00},
    /* ls           */ {3.635699e-11, 3.195031e-10, 6.525157e-10, 1.384139e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.463833e-12, 5.036471e-12, 8.885783e-12, 0.000000e+00},
    /* ls           */ {1.791912e-10, 3.195031e-10, 6.525157e-10, 1.384139e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 64 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {8.568781e-13, 2.463833e-12, 5.036471e-12, 8.885783e-12, 0.000000e+00},
    /* ls           */ {3.635699e-11, 1.791912e-10, 3.195031e-10, 6.525157e-10, 1.384139e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.196569e-12},
    /* ls           */ {6.744683e-10},
    /* m            */ {541165879296},
    /* p            */ {96},
    /* kmax         */ {999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {6.496336e-13, 9.196569e-12},
    /* ls           */ {2.741757e-11, 6.744683e-10},
    /* m            */ {524288, 541165879296},
    /* p            */ {1, 96},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.450486e-12, 9.196569e-12},
    /* ls           */ {1.144697e-10, 6.744683e-10},
    /* m            */ {25165824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.560187e-12, 9.196569e-12},
    /* ls           */ {2.911553e-10, 6.744683e-10},
    /* m            */ {135291469824, 541165879296},
    /* p            */ {24, 4},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.196569e-12, 0.000000e+00},
    /* ls           */ {6.744683e-10, 2.831340e-04},
    /* m            */ {541165879296, 541165879296},
    /* p            */ {96, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {6.496336e-13, 1.450486e-12, 9.196569e-12},
    /* ls           */ {2.741757e-11, 1.144697e-10, 6.744683e-10},
    /* m            */ {524288, 25165824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {6.496336e-13, 4.560187e-12, 9.196569e-12},
    /* ls           */ {2.741757e-11, 2.911553e-10, 6.744683e-10},
    /* m            */ {524288, 135291469824, 541165879296},
    /* p            */ {1, 24, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {6.496336e-13, 9.196569e-12, 0.000000e+00},
    /* ls           */ {2.741757e-11, 6.744683e-10, 2.831340e-04},
    /* m            */ {524288, 541165879296, 541165879296},
    /* p            */ {1, 96, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.450486e-12, 4.560187e-12, 9.196569e-12},
    /* ls           */ {1.144697e-10, 2.911553e-10, 6.744683e-10},
    /* m            */ {25165824, 135291469824, 541165879296},
    /* p            */ {24, 1, 4},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.450486e-12, 9.196569e-12, 0.000000e+00},
    /* ls           */ {1.144697e-10, 6.744683e-10, 2.831340e-04},
    /* m            */ {25165824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.560187e-12, 9.196569e-12, 0.000000e+00},
    /* ls           */ {2.911553e-10, 6.744683e-10, 2.831340e-04},
    /* m            */ {135291469824, 541165879296, 541165879296},
    /* p            */ {24, 4, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {6.496336e-13, 1.450486e-12, 4.560187e-12, 9.196569e-12},
    /* ls           */ {2.741757e-11, 1.144697e-10, 2.911553e-10, 6.744683e-10},
    /* m            */ {524288, 25165824, 135291469824, 541165879296},
    /* p            */ {1, 24, 1, 4},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {6.496336e-13, 1.450486e-12, 9.196569e-12, 0.000000e+00},
    /* ls           */ {2.741757e-11, 1.144697e-10, 6.744683e-10, 2.831340e-04},
    /* m            */ {524288, 25165824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {6.496336e-13, 4.560187e-12, 9.196569e-12, 0.000000e+00},
    /* ls           */ {2.741757e-11, 2.911553e-10, 6.744683e-10, 2.831340e-04},
    /* m            */ {524288, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.450486e-12, 4.560187e-12, 9.196569e-12, 0.000000e+00},
    /* ls           */ {1.144697e-10, 2.911553e-10, 6.744683e-10, 2.831340e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 96 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {6.496336e-13, 1.450486e-12, 4.560187e-12, 9.196569e-12, 0.000000e+00},
    /* ls           */ {2.741757e-11, 1.144697e-10, 2.911553e-10, 6.744683e-10, 2.831340e-04},
    /* m            */ {524288, 25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {1, 24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999, 999}}

};

// HWParameter_configurations struct
const cost_models::HW_model::HWParameter_configurations dis_system_params = {
    /* threads_options_num */ {1, 2, 4, 8, 12, 16, 24, 32, 36, 48, 64, 96},
    /* policy_options_str */ {"close", "spread"},
    /* level_options_str  */ {"L2Cache", "L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"},
    /* hw_model_thread_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11},
    /* hw_model_policy_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    /* hw_model_level_bithash */ {8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31, 8, 9, 10, 12, 24, 11, 13, 25, 14, 26, 28, 15, 27, 29, 30, 31},
    /* hw_models */ hw_models_vector
};

#endif // HW_PARAMS_ARM920_HPP
