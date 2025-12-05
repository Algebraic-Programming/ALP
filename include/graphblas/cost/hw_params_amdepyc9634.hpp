
// Auto-generated hardware parameters for AMDEPYC9634
// Allocation policies: close, spread
// Base levels: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
// ALL subset combinations (32 total)
// Generated on 2025-11-28 15:59:09
//
// Command used to generate this file:
// python3 runner.py --threads 168,84,42,32,24,21,14,8,7,4,2,1 --core-policy close,spread --keep-levels L2Cache,L3Cache,NUMANode,Socket,NodeMem,GLOBAL_SYNC --create-sync-level empirical_poly

#ifndef HW_PARAMS_AMDEPYC9634_HPP
#define HW_PARAMS_AMDEPYC9634_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations and level subsets
const std::vector<cost_models::HW_model::HWParameters> hw_models_vector = {

    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11},
    /* ls           */ {3.433720e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 3.433720e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 3.433720e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 8.964311e-11},
    /* ls           */ {2.755147e-09, 3.433720e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 8.964311e-11},
    /* ls           */ {3.821781e-09, 3.433720e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 0.000000e+00},
    /* ls           */ {3.433720e-09, 2.896786e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 1.230436e-09, 3.433720e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.433720e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.433720e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {2.755147e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {2.755147e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 1.230436e-09, 2.755147e-09, 3.433720e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 1.230436e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 1.230436e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {2.755147e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11},
    /* ls           */ {1.230436e-09, 1.230436e-09, 2.755147e-09, 3.821781e-09, 3.433720e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 1.230436e-09, 2.755147e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 1.230436e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 2.755147e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 8.964311e-11, 0.000000e+00},
    /* ls           */ {1.230436e-09, 1.230436e-09, 2.755147e-09, 3.821781e-09, 3.433720e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11},
    /* ls           */ {2.227291e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 2.227291e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 2.227291e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 4.540568e-11},
    /* ls           */ {1.851919e-09, 2.227291e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 4.540568e-11},
    /* ls           */ {2.217130e-09, 2.227291e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 0.000000e+00},
    /* ls           */ {2.227291e-09, 3.038251e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 4.982741e-10, 2.227291e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.227291e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 2.217130e-09, 2.227291e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.227291e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 2.217130e-09, 2.227291e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 2.227291e-09, 3.038251e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {1.851919e-09, 2.217130e-09, 2.227291e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {1.851919e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 4.982741e-10, 1.851919e-09, 2.227291e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 4.982741e-10, 2.217130e-09, 2.227291e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 4.982741e-10, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.217130e-09, 2.227291e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.217130e-09, 2.227291e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {1.851919e-09, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11},
    /* ls           */ {4.982741e-10, 4.982741e-10, 1.851919e-09, 2.217130e-09, 2.227291e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 4.982741e-10, 1.851919e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 4.982741e-10, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 1.851919e-09, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 4.540568e-11, 0.000000e+00},
    /* ls           */ {4.982741e-10, 4.982741e-10, 1.851919e-09, 2.217130e-09, 2.227291e-09, 3.038251e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {3.093825e-11},
    /* ls           */ {2.041519e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 2.041519e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 2.041519e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.748049e-11, 3.093825e-11},
    /* ls           */ {1.645678e-09, 2.041519e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.748049e-11, 3.093825e-11},
    /* ls           */ {1.727633e-09, 2.041519e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {3.093825e-11, 0.000000e+00},
    /* ls           */ {2.041519e-09, 3.332065e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 2.939975e-10, 2.041519e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 1.645678e-09, 2.041519e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 1.727633e-09, 2.041519e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 1.645678e-09, 2.041519e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 1.727633e-09, 2.041519e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 2.041519e-09, 3.332065e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {1.645678e-09, 1.727633e-09, 2.041519e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {1.645678e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 2.939975e-10, 1.645678e-09, 2.041519e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 2.939975e-10, 1.727633e-09, 2.041519e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 2.939975e-10, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 1.645678e-09, 1.727633e-09, 2.041519e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 1.645678e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 1.645678e-09, 1.727633e-09, 2.041519e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 1.645678e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {1.645678e-09, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11},
    /* ls           */ {2.939975e-10, 2.939975e-10, 1.645678e-09, 1.727633e-09, 2.041519e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 2.939975e-10, 1.645678e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 2.939975e-10, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 1.645678e-09, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 1.645678e-09, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {2.748049e-11, 2.748049e-11, 2.748049e-11, 2.748049e-11, 3.093825e-11, 0.000000e+00},
    /* ls           */ {2.939975e-10, 2.939975e-10, 1.645678e-09, 1.727633e-09, 2.041519e-09, 3.332065e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {3.113831e-11},
    /* ls           */ {2.005319e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 2.005319e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 3.113831e-11},
    /* ls           */ {2.083172e-10, 2.005319e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.658701e-11, 3.113831e-11},
    /* ls           */ {1.710630e-09, 2.005319e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.813480e-11, 3.113831e-11},
    /* ls           */ {1.800228e-09, 2.005319e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {3.113831e-11, 0.000000e+00},
    /* ls           */ {2.005319e-09, 3.799990e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 2.083172e-10, 2.005319e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 1.710630e-09, 2.005319e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 1.800228e-09, 2.005319e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 3.113831e-11},
    /* ls           */ {2.083172e-10, 1.710630e-09, 2.005319e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {2.083172e-10, 1.800228e-09, 2.005319e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {2.083172e-10, 2.005319e-09, 3.799990e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.658701e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {1.710630e-09, 1.800228e-09, 2.005319e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.658701e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.710630e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 2.658701e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 2.083172e-10, 1.710630e-09, 2.005319e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 2.083172e-10, 1.800228e-09, 2.005319e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 2.083172e-10, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 1.710630e-09, 1.800228e-09, 2.005319e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 1.710630e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {2.083172e-10, 1.710630e-09, 1.800228e-09, 2.005319e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {2.083172e-10, 1.710630e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {2.083172e-10, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {2.658701e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.710630e-09, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 2.658701e-11, 2.813480e-11, 3.113831e-11},
    /* ls           */ {1.794973e-10, 2.083172e-10, 1.710630e-09, 1.800228e-09, 2.005319e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 2.658701e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 2.083172e-10, 1.710630e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 2.083172e-10, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 1.710630e-09, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.688592e-11, 2.658701e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {2.083172e-10, 1.710630e-09, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {1.688592e-11, 1.688592e-11, 2.658701e-11, 2.813480e-11, 3.113831e-11, 0.000000e+00},
    /* ls           */ {1.794973e-10, 2.083172e-10, 1.710630e-09, 1.800228e-09, 2.005319e-09, 3.799990e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.457851e-11},
    /* ls           */ {1.595225e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.595225e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.595225e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.134946e-11, 2.457851e-11},
    /* ls           */ {1.398847e-09, 1.595225e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.326069e-11, 2.457851e-11},
    /* ls           */ {1.495247e-09, 1.595225e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.457851e-11, 0.000000e+00},
    /* ls           */ {1.595225e-09, 3.963220e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.595225e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.595225e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.495247e-09, 1.595225e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.595225e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.495247e-09, 1.595225e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.595225e-09, 3.963220e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.134946e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.398847e-09, 1.495247e-09, 1.595225e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.134946e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.398847e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.134946e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.398847e-09, 1.595225e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.495247e-09, 1.595225e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.495247e-09, 1.595225e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.495247e-09, 1.595225e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {2.134946e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.398847e-09, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.134946e-11, 2.326069e-11, 2.457851e-11},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.398847e-09, 1.495247e-09, 1.595225e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.134946e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.398847e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.472733e-11, 2.134946e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.398847e-09, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {1.472733e-11, 1.472733e-11, 2.134946e-11, 2.326069e-11, 2.457851e-11, 0.000000e+00},
    /* ls           */ {1.525035e-10, 1.525035e-10, 1.398847e-09, 1.495247e-09, 1.595225e-09, 3.963220e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.814913e-11},
    /* ls           */ {1.160466e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.160466e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11},
    /* ls           */ {1.285168e-10, 1.160466e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.617438e-11, 1.814913e-11},
    /* ls           */ {1.045735e-09, 1.160466e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.814913e-11, 1.814913e-11},
    /* ls           */ {1.160466e-09, 1.160466e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.814913e-11, 0.000000e+00},
    /* ls           */ {1.160466e-09, 5.018774e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.160466e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.045735e-09, 1.160466e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.160466e-09, 1.160466e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11},
    /* ls           */ {1.285168e-10, 1.045735e-09, 1.160466e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11, 1.814913e-11},
    /* ls           */ {1.285168e-10, 1.160466e-09, 1.160466e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.285168e-10, 1.160466e-09, 5.018774e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.617438e-11, 1.814913e-11, 1.814913e-11},
    /* ls           */ {1.045735e-09, 1.160466e-09, 1.160466e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.617438e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.045735e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.617438e-11, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.045735e-09, 1.160466e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.814913e-11, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.160466e-09, 1.160466e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.045735e-09, 1.160466e-09, 1.160466e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.045735e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11, 1.814913e-11},
    /* ls           */ {1.285168e-10, 1.045735e-09, 1.160466e-09, 1.160466e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.285168e-10, 1.045735e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.285168e-10, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {1.617438e-11, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.045735e-09, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.617438e-11, 1.814913e-11, 1.814913e-11},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.045735e-09, 1.160466e-09, 1.160466e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.617438e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.045735e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.045735e-09, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {8.059711e-12, 1.617438e-11, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {1.285168e-10, 1.045735e-09, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {8.059711e-12, 8.059711e-12, 1.617438e-11, 1.814913e-11, 1.814913e-11, 0.000000e+00},
    /* ls           */ {7.573487e-11, 1.285168e-10, 1.045735e-09, 1.160466e-09, 1.160466e-09, 5.018774e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.643051e-11},
    /* ls           */ {1.054824e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11},
    /* ls           */ {3.121964e-11, 1.054824e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11},
    /* ls           */ {1.148908e-10, 1.054824e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.458452e-11, 1.643051e-11},
    /* ls           */ {9.334660e-10, 1.054824e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.643051e-11, 1.643051e-11},
    /* ls           */ {1.054824e-09, 1.054824e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.643051e-11, 0.000000e+00},
    /* ls           */ {1.054824e-09, 6.415300e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.643051e-11},
    /* ls           */ {3.121964e-11, 1.148908e-10, 1.054824e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11},
    /* ls           */ {3.121964e-11, 9.334660e-10, 1.054824e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11, 1.643051e-11},
    /* ls           */ {3.121964e-11, 1.054824e-09, 1.054824e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11},
    /* ls           */ {1.148908e-10, 9.334660e-10, 1.054824e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11, 1.643051e-11},
    /* ls           */ {1.148908e-10, 1.054824e-09, 1.054824e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11, 0.000000e+00},
    /* ls           */ {1.148908e-10, 1.054824e-09, 6.415300e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.458452e-11, 1.643051e-11, 1.643051e-11},
    /* ls           */ {9.334660e-10, 1.054824e-09, 1.054824e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.458452e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {9.334660e-10, 1.054824e-09, 6.415300e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.458452e-11, 1.643051e-11},
    /* ls           */ {3.121964e-11, 1.148908e-10, 9.334660e-10, 1.054824e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.643051e-11, 1.643051e-11},
    /* ls           */ {3.121964e-11, 1.148908e-10, 1.054824e-09, 1.054824e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 1.148908e-10, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11, 1.643051e-11},
    /* ls           */ {3.121964e-11, 9.334660e-10, 1.054824e-09, 1.054824e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 9.334660e-10, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11, 1.643051e-11},
    /* ls           */ {1.148908e-10, 9.334660e-10, 1.054824e-09, 1.054824e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {1.148908e-10, 9.334660e-10, 1.054824e-09, 6.415300e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {1.148908e-10, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {1.458452e-11, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {9.334660e-10, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.458452e-11, 1.643051e-11, 1.643051e-11},
    /* ls           */ {3.121964e-11, 1.148908e-10, 9.334660e-10, 1.054824e-09, 1.054824e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.458452e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 1.148908e-10, 9.334660e-10, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 1.148908e-10, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 9.334660e-10, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {4.316901e-12, 1.458452e-11, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {1.148908e-10, 9.334660e-10, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {4.316901e-12, 4.316901e-12, 1.458452e-11, 1.643051e-11, 1.643051e-11, 0.000000e+00},
    /* ls           */ {3.121964e-11, 1.148908e-10, 9.334660e-10, 1.054824e-09, 1.054824e-09, 6.415300e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.314770e-11},
    /* ls           */ {8.472081e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11},
    /* ls           */ {2.705472e-11, 8.472081e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11},
    /* ls           */ {5.214102e-11, 8.472081e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.161746e-11, 1.314770e-11},
    /* ls           */ {7.455466e-10, 8.472081e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.314770e-11, 1.314770e-11},
    /* ls           */ {8.472081e-10, 8.472081e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.314770e-11, 0.000000e+00},
    /* ls           */ {8.472081e-10, 7.068222e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.314770e-11},
    /* ls           */ {2.705472e-11, 5.214102e-11, 8.472081e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11},
    /* ls           */ {2.705472e-11, 7.455466e-10, 8.472081e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11, 1.314770e-11},
    /* ls           */ {2.705472e-11, 8.472081e-10, 8.472081e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11},
    /* ls           */ {5.214102e-11, 7.455466e-10, 8.472081e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11, 1.314770e-11},
    /* ls           */ {5.214102e-11, 8.472081e-10, 8.472081e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11, 0.000000e+00},
    /* ls           */ {5.214102e-11, 8.472081e-10, 7.068222e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.161746e-11, 1.314770e-11, 1.314770e-11},
    /* ls           */ {7.455466e-10, 8.472081e-10, 8.472081e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.161746e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {7.455466e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.161746e-11, 1.314770e-11},
    /* ls           */ {2.705472e-11, 5.214102e-11, 7.455466e-10, 8.472081e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.314770e-11, 1.314770e-11},
    /* ls           */ {2.705472e-11, 5.214102e-11, 8.472081e-10, 8.472081e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 5.214102e-11, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11, 1.314770e-11},
    /* ls           */ {2.705472e-11, 7.455466e-10, 8.472081e-10, 8.472081e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 7.455466e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11, 1.314770e-11},
    /* ls           */ {5.214102e-11, 7.455466e-10, 8.472081e-10, 8.472081e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {5.214102e-11, 7.455466e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {5.214102e-11, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {1.161746e-11, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {7.455466e-10, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.161746e-11, 1.314770e-11, 1.314770e-11},
    /* ls           */ {2.705472e-11, 5.214102e-11, 7.455466e-10, 8.472081e-10, 8.472081e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.161746e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 5.214102e-11, 7.455466e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 5.214102e-11, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 7.455466e-10, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {3.356358e-12, 1.161746e-11, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {5.214102e-11, 7.455466e-10, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {3.356358e-12, 3.356358e-12, 1.161746e-11, 1.314770e-11, 1.314770e-11, 0.000000e+00},
    /* ls           */ {2.705472e-11, 5.214102e-11, 7.455466e-10, 8.472081e-10, 8.472081e-10, 7.068222e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {8.775680e-12},
    /* ls           */ {5.613886e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 5.613886e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12},
    /* ls           */ {3.794211e-11, 5.613886e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.803350e-12, 8.775680e-12},
    /* ls           */ {5.011089e-10, 5.613886e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.775680e-12, 8.775680e-12},
    /* ls           */ {5.613886e-10, 5.613886e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {8.775680e-12, 0.000000e+00},
    /* ls           */ {5.613886e-10, 8.968950e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.613886e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 5.011089e-10, 5.613886e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 5.613886e-10, 5.613886e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12},
    /* ls           */ {3.794211e-11, 5.011089e-10, 5.613886e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {3.794211e-11, 5.613886e-10, 5.613886e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {3.794211e-11, 5.613886e-10, 8.968950e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.803350e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {5.011089e-10, 5.613886e-10, 5.613886e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.803350e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {5.011089e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 7.803350e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.011089e-10, 5.613886e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.613886e-10, 5.613886e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 5.011089e-10, 5.613886e-10, 5.613886e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 5.011089e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {3.794211e-11, 5.011089e-10, 5.613886e-10, 5.613886e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {3.794211e-11, 5.011089e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {3.794211e-11, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {7.803350e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {5.011089e-10, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 7.803350e-12, 8.775680e-12, 8.775680e-12},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.011089e-10, 5.613886e-10, 5.613886e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 7.803350e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.011089e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 5.011089e-10, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {2.236258e-12, 7.803350e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {3.794211e-11, 5.011089e-10, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {2.236258e-12, 2.236258e-12, 7.803350e-12, 8.775680e-12, 8.775680e-12, 0.000000e+00},
    /* ls           */ {1.995653e-11, 3.794211e-11, 5.011089e-10, 5.613886e-10, 5.613886e-10, 8.968950e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {7.837442e-12},
    /* ls           */ {5.045217e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 5.045217e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12},
    /* ls           */ {5.724380e-11, 5.045217e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.962023e-12, 7.837442e-12},
    /* ls           */ {4.472180e-10, 5.045217e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.837442e-12, 7.837442e-12},
    /* ls           */ {5.045217e-10, 5.045217e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.837442e-12, 0.000000e+00},
    /* ls           */ {5.045217e-10, 1.167132e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 5.724380e-11, 5.045217e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 4.472180e-10, 5.045217e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 5.045217e-10, 5.045217e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12},
    /* ls           */ {5.724380e-11, 4.472180e-10, 5.045217e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {5.724380e-11, 5.045217e-10, 5.045217e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {5.724380e-11, 5.045217e-10, 1.167132e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.962023e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {4.472180e-10, 5.045217e-10, 5.045217e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.962023e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {4.472180e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 6.962023e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 5.724380e-11, 4.472180e-10, 5.045217e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 5.724380e-11, 5.045217e-10, 5.045217e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 5.724380e-11, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 4.472180e-10, 5.045217e-10, 5.045217e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 4.472180e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {5.724380e-11, 4.472180e-10, 5.045217e-10, 5.045217e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {5.724380e-11, 4.472180e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {5.724380e-11, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {6.962023e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {4.472180e-10, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 6.962023e-12, 7.837442e-12, 7.837442e-12},
    /* ls           */ {1.516521e-11, 5.724380e-11, 4.472180e-10, 5.045217e-10, 5.045217e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 6.962023e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 5.724380e-11, 4.472180e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 5.724380e-11, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 4.472180e-10, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.487938e-12, 6.962023e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {5.724380e-11, 4.472180e-10, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {1.487938e-12, 1.487938e-12, 6.962023e-12, 7.837442e-12, 7.837442e-12, 0.000000e+00},
    /* ls           */ {1.516521e-11, 5.724380e-11, 4.472180e-10, 5.045217e-10, 5.045217e-10, 1.167132e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {3.648506e-12},
    /* ls           */ {2.342909e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.648506e-12},
    /* ls           */ {9.401769e-12, 2.342909e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.648506e-12},
    /* ls           */ {3.024337e-11, 2.342909e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {3.189642e-12, 3.648506e-12},
    /* ls           */ {2.043914e-10, 2.342909e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {3.304159e-12, 3.648506e-12},
    /* ls           */ {2.122640e-10, 2.342909e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {3.648506e-12, 0.000000e+00},
    /* ls           */ {2.342909e-10, 2.698239e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.648506e-12},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.342909e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.189642e-12, 3.648506e-12},
    /* ls           */ {9.401769e-12, 2.043914e-10, 2.342909e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.304159e-12, 3.648506e-12},
    /* ls           */ {9.401769e-12, 2.122640e-10, 2.342909e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.189642e-12, 3.648506e-12},
    /* ls           */ {3.024337e-11, 2.043914e-10, 2.342909e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.304159e-12, 3.648506e-12},
    /* ls           */ {3.024337e-11, 2.122640e-10, 2.342909e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.648506e-12, 0.000000e+00},
    /* ls           */ {3.024337e-11, 2.342909e-10, 2.698239e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {3.189642e-12, 3.304159e-12, 3.648506e-12},
    /* ls           */ {2.043914e-10, 2.122640e-10, 2.342909e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {3.189642e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {2.043914e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.189642e-12, 3.648506e-12},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.043914e-10, 2.342909e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.304159e-12, 3.648506e-12},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.122640e-10, 2.342909e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.189642e-12, 3.304159e-12, 3.648506e-12},
    /* ls           */ {9.401769e-12, 2.043914e-10, 2.122640e-10, 2.342909e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.189642e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 2.043914e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.189642e-12, 3.304159e-12, 3.648506e-12},
    /* ls           */ {3.024337e-11, 2.043914e-10, 2.122640e-10, 2.342909e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.189642e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {3.024337e-11, 2.043914e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {3.024337e-11, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {3.189642e-12, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {2.043914e-10, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.189642e-12, 3.304159e-12, 3.648506e-12},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.043914e-10, 2.122640e-10, 2.342909e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.189642e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.043914e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {2.784493e-13, 3.189642e-12, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 2.043914e-10, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {5.939225e-13, 3.189642e-12, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {3.024337e-11, 2.043914e-10, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {2.784493e-13, 5.939225e-13, 3.189642e-12, 3.304159e-12, 3.648506e-12, 0.000000e+00},
    /* ls           */ {9.401769e-12, 3.024337e-11, 2.043914e-10, 2.122640e-10, 2.342909e-10, 2.698239e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.243808e-12},
    /* ls           */ {1.402536e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 2.243808e-12},
    /* ls           */ {4.910691e-12, 1.402536e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 2.243808e-12},
    /* ls           */ {1.467725e-11, 1.402536e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.117451e-12, 2.243808e-12},
    /* ls           */ {7.111146e-11, 1.402536e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.098726e-12, 2.243808e-12},
    /* ls           */ {1.323775e-10, 1.402536e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.243808e-12, 0.000000e+00},
    /* ls           */ {1.402536e-10, 7.680059e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 2.243808e-12},
    /* ls           */ {4.910691e-12, 1.467725e-11, 1.402536e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 1.117451e-12, 2.243808e-12},
    /* ls           */ {4.910691e-12, 7.111146e-11, 1.402536e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 2.098726e-12, 2.243808e-12},
    /* ls           */ {4.910691e-12, 1.323775e-10, 1.402536e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 1.117451e-12, 2.243808e-12},
    /* ls           */ {1.467725e-11, 7.111146e-11, 1.402536e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 2.098726e-12, 2.243808e-12},
    /* ls           */ {1.467725e-11, 1.323775e-10, 1.402536e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 2.243808e-12, 0.000000e+00},
    /* ls           */ {1.467725e-11, 1.402536e-10, 7.680059e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.117451e-12, 2.098726e-12, 2.243808e-12},
    /* ls           */ {7.111146e-11, 1.323775e-10, 1.402536e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.117451e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {7.111146e-11, 1.402536e-10, 7.680059e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 1.117451e-12, 2.243808e-12},
    /* ls           */ {4.910691e-12, 1.467725e-11, 7.111146e-11, 1.402536e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 2.098726e-12, 2.243808e-12},
    /* ls           */ {4.910691e-12, 1.467725e-11, 1.323775e-10, 1.402536e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 1.467725e-11, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 1.117451e-12, 2.098726e-12, 2.243808e-12},
    /* ls           */ {4.910691e-12, 7.111146e-11, 1.323775e-10, 1.402536e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 1.117451e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 7.111146e-11, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 1.117451e-12, 2.098726e-12, 2.243808e-12},
    /* ls           */ {1.467725e-11, 7.111146e-11, 1.323775e-10, 1.402536e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 1.117451e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {1.467725e-11, 7.111146e-11, 1.402536e-10, 7.680059e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {1.467725e-11, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {1.117451e-12, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {7.111146e-11, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 1.117451e-12, 2.098726e-12, 2.243808e-12},
    /* ls           */ {4.910691e-12, 1.467725e-11, 7.111146e-11, 1.323775e-10, 1.402536e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 1.117451e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 1.467725e-11, 7.111146e-11, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 1.467725e-11, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {1.388669e-13, 1.117451e-12, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 7.111146e-11, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ 2,
    /* g            */ {2.833715e-13, 1.117451e-12, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {1.467725e-11, 7.111146e-11, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: close, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ 3,
    /* g            */ {1.388669e-13, 2.833715e-13, 1.117451e-12, 2.098726e-12, 2.243808e-12, 0.000000e+00},
    /* ls           */ {4.910691e-12, 1.467725e-11, 7.111146e-11, 1.323775e-10, 1.402536e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10},
    /* ls           */ {3.339447e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 3.339447e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 3.339447e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 1.282384e-10},
    /* ls           */ {2.680885e-09, 3.339447e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 1.282384e-10},
    /* ls           */ {3.476004e-09, 3.339447e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 0.000000e+00},
    /* ls           */ {3.339447e-09, 2.896786e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 1.339229e-09, 3.339447e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.339447e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.339447e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {2.680885e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {2.680885e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 1.339229e-09, 2.680885e-09, 3.339447e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 1.339229e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 1.339229e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {2.680885e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10},
    /* ls           */ {1.339229e-09, 1.339229e-09, 2.680885e-09, 3.476004e-09, 3.339447e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 1.339229e-09, 2.680885e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 1.339229e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 2.680885e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 1 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 1.282384e-10, 0.000000e+00},
    /* ls           */ {1.339229e-09, 1.339229e-09, 2.680885e-09, 3.476004e-09, 3.339447e-09, 2.896786e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11},
    /* ls           */ {1.683937e-09},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.683937e-09},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.683937e-09},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 5.008890e-11},
    /* ls           */ {1.226384e-09, 1.683937e-09},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 5.008890e-11},
    /* ls           */ {1.309244e-09, 1.683937e-09},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 0.000000e+00},
    /* ls           */ {1.683937e-09, 3.339323e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.683937e-09},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.683937e-09},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.309244e-09, 1.683937e-09},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.683937e-09},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.309244e-09, 1.683937e-09},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.683937e-09, 3.339323e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {1.226384e-09, 1.309244e-09, 1.683937e-09},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {1.226384e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.226384e-09, 1.683937e-09},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.309244e-09, 1.683937e-09},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.309244e-09, 1.683937e-09},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.309244e-09, 1.683937e-09},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {1.226384e-09, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.226384e-09, 1.309244e-09, 1.683937e-09},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.226384e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 1.226384e-09, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 2 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 5.008890e-11, 0.000000e+00},
    /* ls           */ {5.335141e-10, 5.335141e-10, 1.226384e-09, 1.309244e-09, 1.683937e-09, 3.339323e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11},
    /* ls           */ {7.210155e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 7.210155e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11},
    /* ls           */ {2.301395e-10, 7.210155e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 2.047614e-11},
    /* ls           */ {5.788822e-10, 7.210155e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 2.047614e-11},
    /* ls           */ {6.054815e-10, 7.210155e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 0.000000e+00},
    /* ls           */ {7.210155e-10, 4.224399e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 2.301395e-10, 7.210155e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 5.788822e-10, 7.210155e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.301395e-10, 5.788822e-10, 7.210155e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.301395e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.301395e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {5.788822e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {5.788822e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 2.301395e-10, 5.788822e-10, 7.210155e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 2.301395e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 2.301395e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 5.788822e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 5.788822e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.301395e-10, 5.788822e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.301395e-10, 5.788822e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.301395e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {5.788822e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11},
    /* ls           */ {2.429460e-10, 2.301395e-10, 5.788822e-10, 6.054815e-10, 7.210155e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 2.301395e-10, 5.788822e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 2.301395e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 5.788822e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.301395e-10, 5.788822e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 4 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 2.047614e-11, 0.000000e+00},
    /* ls           */ {2.429460e-10, 2.301395e-10, 5.788822e-10, 6.054815e-10, 7.210155e-10, 4.224399e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12},
    /* ls           */ {5.079041e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 5.079041e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12},
    /* ls           */ {1.420377e-10, 5.079041e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 9.327227e-12},
    /* ls           */ {3.077096e-10, 5.079041e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 9.327227e-12},
    /* ls           */ {3.987527e-10, 5.079041e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 0.000000e+00},
    /* ls           */ {5.079041e-10, 5.552012e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 1.420377e-10, 5.079041e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 3.077096e-10, 5.079041e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 3.987527e-10, 5.079041e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {1.420377e-10, 3.077096e-10, 5.079041e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {1.420377e-10, 3.987527e-10, 5.079041e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {1.420377e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {3.077096e-10, 3.987527e-10, 5.079041e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {3.077096e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 1.420377e-10, 3.077096e-10, 5.079041e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 1.420377e-10, 3.987527e-10, 5.079041e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 1.420377e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 3.077096e-10, 3.987527e-10, 5.079041e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 3.077096e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {1.420377e-10, 3.077096e-10, 3.987527e-10, 5.079041e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {1.420377e-10, 3.077096e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {1.420377e-10, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {3.077096e-10, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12},
    /* ls           */ {9.044258e-11, 1.420377e-10, 3.077096e-10, 3.987527e-10, 5.079041e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 1.420377e-10, 3.077096e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 1.420377e-10, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 3.077096e-10, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {1.420377e-10, 3.077096e-10, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 7 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 9.327227e-12, 0.000000e+00},
    /* ls           */ {9.044258e-11, 1.420377e-10, 3.077096e-10, 3.987527e-10, 5.079041e-10, 5.552012e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12},
    /* ls           */ {4.404983e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 4.404983e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12},
    /* ls           */ {1.241595e-10, 4.404983e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 7.677262e-12},
    /* ls           */ {2.585035e-10, 4.404983e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 7.677262e-12},
    /* ls           */ {3.523095e-10, 4.404983e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 0.000000e+00},
    /* ls           */ {4.404983e-10, 5.994550e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 1.241595e-10, 4.404983e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 2.585035e-10, 4.404983e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 3.523095e-10, 4.404983e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {1.241595e-10, 2.585035e-10, 4.404983e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {1.241595e-10, 3.523095e-10, 4.404983e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {1.241595e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {2.585035e-10, 3.523095e-10, 4.404983e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {2.585035e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 1.241595e-10, 2.585035e-10, 4.404983e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 1.241595e-10, 3.523095e-10, 4.404983e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 1.241595e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 2.585035e-10, 3.523095e-10, 4.404983e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 2.585035e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {1.241595e-10, 2.585035e-10, 3.523095e-10, 4.404983e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {1.241595e-10, 2.585035e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {1.241595e-10, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {2.585035e-10, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12},
    /* ls           */ {7.806378e-11, 1.241595e-10, 2.585035e-10, 3.523095e-10, 4.404983e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 1.241595e-10, 2.585035e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 1.241595e-10, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 2.585035e-10, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {1.241595e-10, 2.585035e-10, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 8 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 7.677262e-12, 0.000000e+00},
    /* ls           */ {7.806378e-11, 1.241595e-10, 2.585035e-10, 3.523095e-10, 4.404983e-10, 5.994550e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {4.982002e-12},
    /* ls           */ {3.726846e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 3.726846e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12},
    /* ls           */ {1.971958e-10, 3.726846e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.711439e-12, 4.982002e-12},
    /* ls           */ {3.213639e-10, 3.726846e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.982002e-12, 4.982002e-12},
    /* ls           */ {3.726846e-10, 3.726846e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.982002e-12, 0.000000e+00},
    /* ls           */ {3.726846e-10, 8.649776e-06},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.726846e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 3.213639e-10, 3.726846e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 3.726846e-10, 3.726846e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12},
    /* ls           */ {1.971958e-10, 3.213639e-10, 3.726846e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {1.971958e-10, 3.726846e-10, 3.726846e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {1.971958e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.711439e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {3.213639e-10, 3.726846e-10, 3.726846e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.711439e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {3.213639e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.711439e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.213639e-10, 3.726846e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.726846e-10, 3.726846e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 3.213639e-10, 3.726846e-10, 3.726846e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 3.213639e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {1.971958e-10, 3.213639e-10, 3.726846e-10, 3.726846e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {1.971958e-10, 3.213639e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {1.971958e-10, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {4.711439e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {3.213639e-10, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.711439e-12, 4.982002e-12, 4.982002e-12},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.213639e-10, 3.726846e-10, 3.726846e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.711439e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.213639e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 3.213639e-10, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {4.313032e-12, 4.711439e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {1.971958e-10, 3.213639e-10, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 14 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {4.313032e-12, 4.313032e-12, 4.711439e-12, 4.982002e-12, 4.982002e-12, 0.000000e+00},
    /* ls           */ {4.555968e-11, 1.971958e-10, 3.213639e-10, 3.726846e-10, 3.726846e-10, 8.649776e-06},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {4.410604e-12},
    /* ls           */ {2.889222e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 2.889222e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.410604e-12},
    /* ls           */ {1.921332e-10, 2.889222e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {3.809472e-12, 4.410604e-12},
    /* ls           */ {2.439424e-10, 2.889222e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.044588e-12, 4.410604e-12},
    /* ls           */ {2.564929e-10, 2.889222e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.410604e-12, 0.000000e+00},
    /* ls           */ {2.889222e-10, 1.174754e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.889222e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 2.439424e-10, 2.889222e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 2.564929e-10, 2.889222e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.410604e-12},
    /* ls           */ {1.921332e-10, 2.439424e-10, 2.889222e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {1.921332e-10, 2.564929e-10, 2.889222e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {1.921332e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {3.809472e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {2.439424e-10, 2.564929e-10, 2.889222e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {3.809472e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {2.439424e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 3.809472e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.439424e-10, 2.889222e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.564929e-10, 2.889222e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 2.439424e-10, 2.564929e-10, 2.889222e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 2.439424e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {1.921332e-10, 2.439424e-10, 2.564929e-10, 2.889222e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {1.921332e-10, 2.439424e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {1.921332e-10, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {3.809472e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {2.439424e-10, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 3.809472e-12, 4.044588e-12, 4.410604e-12},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.439424e-10, 2.564929e-10, 2.889222e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 3.809472e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.439424e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 2.439424e-10, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {3.354710e-12, 3.809472e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {1.921332e-10, 2.439424e-10, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 21 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {3.354710e-12, 3.354710e-12, 3.809472e-12, 4.044588e-12, 4.410604e-12, 0.000000e+00},
    /* ls           */ {3.031309e-11, 1.921332e-10, 2.439424e-10, 2.564929e-10, 2.889222e-10, 1.174754e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {4.509745e-12},
    /* ls           */ {3.895867e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 3.895867e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.509745e-12},
    /* ls           */ {2.586154e-10, 3.895867e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.350120e-12, 4.509745e-12},
    /* ls           */ {3.403620e-10, 3.895867e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.100768e-12, 4.509745e-12},
    /* ls           */ {3.895867e-10, 3.895867e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {4.509745e-12, 0.000000e+00},
    /* ls           */ {3.895867e-10, 1.307515e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.895867e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.350120e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 3.403620e-10, 3.895867e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 3.895867e-10, 3.895867e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.350120e-12, 4.509745e-12},
    /* ls           */ {2.586154e-10, 3.403620e-10, 3.895867e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {2.586154e-10, 3.895867e-10, 3.895867e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {2.586154e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.350120e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {3.403620e-10, 3.895867e-10, 3.895867e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.350120e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {3.403620e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.350120e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.403620e-10, 3.895867e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.895867e-10, 3.895867e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.350120e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 3.403620e-10, 3.895867e-10, 3.895867e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.350120e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 3.403620e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.350120e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {2.586154e-10, 3.403620e-10, 3.895867e-10, 3.895867e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.350120e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {2.586154e-10, 3.403620e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {2.586154e-10, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {4.350120e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {3.403620e-10, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.350120e-12, 4.100768e-12, 4.509745e-12},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.403620e-10, 3.895867e-10, 3.895867e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.350120e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.403620e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {4.156891e-12, 4.350120e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 3.403620e-10, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {3.985053e-12, 4.350120e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {2.586154e-10, 3.403620e-10, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 24 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {4.156891e-12, 3.985053e-12, 4.350120e-12, 4.100768e-12, 4.509745e-12, 0.000000e+00},
    /* ls           */ {5.272319e-11, 2.586154e-10, 3.403620e-10, 3.895867e-10, 3.895867e-10, 1.307515e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {5.700102e-12},
    /* ls           */ {3.186687e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 3.186687e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.700102e-12},
    /* ls           */ {2.487301e-10, 3.186687e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.220252e-12, 5.700102e-12},
    /* ls           */ {3.321574e-10, 3.186687e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.700102e-12, 5.700102e-12},
    /* ls           */ {3.625127e-10, 3.186687e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {5.700102e-12, 0.000000e+00},
    /* ls           */ {3.186687e-10, 1.661546e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.186687e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.220252e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 3.321574e-10, 3.186687e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 3.625127e-10, 3.186687e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.220252e-12, 5.700102e-12},
    /* ls           */ {2.487301e-10, 3.321574e-10, 3.186687e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {2.487301e-10, 3.625127e-10, 3.186687e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {2.487301e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.220252e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {3.321574e-10, 3.625127e-10, 3.186687e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.220252e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.321574e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.220252e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.321574e-10, 3.186687e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.625127e-10, 3.186687e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.220252e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 3.321574e-10, 3.625127e-10, 3.186687e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.220252e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 3.321574e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.220252e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {2.487301e-10, 3.321574e-10, 3.625127e-10, 3.186687e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.220252e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {2.487301e-10, 3.321574e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {2.487301e-10, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {5.220252e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.321574e-10, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.220252e-12, 5.700102e-12, 5.700102e-12},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.321574e-10, 3.625127e-10, 3.186687e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.220252e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.321574e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {2.991093e-12, 5.220252e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 3.321574e-10, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {3.920657e-12, 5.220252e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {2.487301e-10, 3.321574e-10, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 32 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {2.991093e-12, 3.920657e-12, 5.220252e-12, 5.700102e-12, 5.700102e-12, 0.000000e+00},
    /* ls           */ {3.853621e-11, 2.487301e-10, 3.321574e-10, 3.625127e-10, 3.186687e-10, 1.661546e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {6.379357e-12},
    /* ls           */ {3.985064e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 3.985064e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.379357e-12},
    /* ls           */ {2.679529e-10, 3.985064e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.305477e-12, 6.379357e-12},
    /* ls           */ {3.901496e-10, 3.985064e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.950726e-12, 6.379357e-12},
    /* ls           */ {4.302226e-10, 3.985064e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {6.379357e-12, 0.000000e+00},
    /* ls           */ {3.985064e-10, 2.104083e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 2.679529e-10, 3.985064e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.305477e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 3.901496e-10, 3.985064e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 4.302226e-10, 3.985064e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.305477e-12, 6.379357e-12},
    /* ls           */ {2.679529e-10, 3.901496e-10, 3.985064e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {2.679529e-10, 4.302226e-10, 3.985064e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {2.679529e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.305477e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {3.901496e-10, 4.302226e-10, 3.985064e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.305477e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.901496e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.305477e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 2.679529e-10, 3.901496e-10, 3.985064e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 2.679529e-10, 4.302226e-10, 3.985064e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 2.679529e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.305477e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 3.901496e-10, 4.302226e-10, 3.985064e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.305477e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 3.901496e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.305477e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {2.679529e-10, 3.901496e-10, 4.302226e-10, 3.985064e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.305477e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {2.679529e-10, 3.901496e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {2.679529e-10, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {6.305477e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.901496e-10, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.305477e-12, 6.950726e-12, 6.379357e-12},
    /* ls           */ {3.994820e-11, 2.679529e-10, 3.901496e-10, 4.302226e-10, 3.985064e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.305477e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 2.679529e-10, 3.901496e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 2.679529e-10, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {2.930038e-12, 6.305477e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 3.901496e-10, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {4.416975e-12, 6.305477e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {2.679529e-10, 3.901496e-10, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 42 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {2.930038e-12, 4.416975e-12, 6.305477e-12, 6.950726e-12, 6.379357e-12, 0.000000e+00},
    /* ls           */ {3.994820e-11, 2.679529e-10, 3.901496e-10, 4.302226e-10, 3.985064e-10, 2.104083e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.588359e-12},
    /* ls           */ {1.442927e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.442927e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.588359e-12},
    /* ls           */ {1.195558e-10, 1.442927e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.390053e-12, 2.588359e-12},
    /* ls           */ {1.450849e-10, 1.442927e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.588359e-12, 2.588359e-12},
    /* ls           */ {1.578142e-10, 1.442927e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.588359e-12, 0.000000e+00},
    /* ls           */ {1.442927e-10, 3.962742e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.442927e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.390053e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.450849e-10, 1.442927e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.578142e-10, 1.442927e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.390053e-12, 2.588359e-12},
    /* ls           */ {1.195558e-10, 1.450849e-10, 1.442927e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {1.195558e-10, 1.578142e-10, 1.442927e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.195558e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.390053e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {1.450849e-10, 1.578142e-10, 1.442927e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.390053e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.450849e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.390053e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.450849e-10, 1.442927e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.578142e-10, 1.442927e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.390053e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.450849e-10, 1.578142e-10, 1.442927e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.390053e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.450849e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.390053e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {1.195558e-10, 1.450849e-10, 1.578142e-10, 1.442927e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.390053e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.195558e-10, 1.450849e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.195558e-10, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {2.390053e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.450849e-10, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.390053e-12, 2.588359e-12, 2.588359e-12},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.450849e-10, 1.578142e-10, 1.442927e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.390053e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.450849e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {1.577008e-12, 2.390053e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.450849e-10, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {1.913179e-12, 2.390053e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {1.195558e-10, 1.450849e-10, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 84 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {1.577008e-12, 1.913179e-12, 2.390053e-12, 2.588359e-12, 2.588359e-12, 0.000000e+00},
    /* ls           */ {2.218720e-11, 1.195558e-10, 1.450849e-10, 1.578142e-10, 1.442927e-10, 3.962742e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: NodeMem
   {
    /* d            */ 1,
    /* d_numa       */ 0,
    /* g            */ {2.239198e-12},
    /* ls           */ {1.410487e-10},
    /* m            */ {605590388736},
    /* p            */ {168},
    /* kmax         */ {999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 2.239198e-12},
    /* ls           */ {4.915842e-12, 1.410487e-10},
    /* m            */ {1048576, 605590388736},
    /* p            */ {1, 168},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 2.239198e-12},
    /* ls           */ {1.492092e-11, 1.410487e-10},
    /* m            */ {33554432, 605590388736},
    /* p            */ {7, 24},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: NUMANode, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {1.116973e-12, 2.239198e-12},
    /* ls           */ {7.163592e-11, 1.410487e-10},
    /* m            */ {100931731456, 605590388736},
    /* p            */ {21, 8},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: Socket, NodeMem
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.098825e-12, 2.239198e-12},
    /* ls           */ {1.328118e-10, 1.410487e-10},
    /* m            */ {302795194368, 605590388736},
    /* p            */ {84, 2},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: NodeMem, GLOBAL_SYNC
   {
    /* d            */ 2,
    /* d_numa       */ 0,
    /* g            */ {2.239198e-12, 0.000000e+00},
    /* ls           */ {1.410487e-10, 7.680059e-05},
    /* m            */ {605590388736, 605590388736},
    /* p            */ {168, 1},
    /* kmax         */ {999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 2.239198e-12},
    /* ls           */ {4.915842e-12, 1.492092e-11, 1.410487e-10},
    /* m            */ {1048576, 33554432, 605590388736},
    /* p            */ {1, 7, 24},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 1.116973e-12, 2.239198e-12},
    /* ls           */ {4.915842e-12, 7.163592e-11, 1.410487e-10},
    /* m            */ {1048576, 100931731456, 605590388736},
    /* p            */ {1, 21, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 2.098825e-12, 2.239198e-12},
    /* ls           */ {4.915842e-12, 1.328118e-10, 1.410487e-10},
    /* m            */ {1048576, 302795194368, 605590388736},
    /* p            */ {1, 84, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 605590388736, 605590388736},
    /* p            */ {1, 168, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 1.116973e-12, 2.239198e-12},
    /* ls           */ {1.492092e-11, 7.163592e-11, 1.410487e-10},
    /* m            */ {33554432, 100931731456, 605590388736},
    /* p            */ {7, 3, 8},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 2.098825e-12, 2.239198e-12},
    /* ls           */ {1.492092e-11, 1.328118e-10, 1.410487e-10},
    /* m            */ {33554432, 302795194368, 605590388736},
    /* p            */ {7, 12, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 2.239198e-12, 0.000000e+00},
    /* ls           */ {1.492092e-11, 1.410487e-10, 7.680059e-05},
    /* m            */ {33554432, 605590388736, 605590388736},
    /* p            */ {7, 24, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.116973e-12, 2.098825e-12, 2.239198e-12},
    /* ls           */ {7.163592e-11, 1.328118e-10, 1.410487e-10},
    /* m            */ {100931731456, 302795194368, 605590388736},
    /* p            */ {21, 4, 2},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {1.116973e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {7.163592e-11, 1.410487e-10, 7.680059e-05},
    /* m            */ {100931731456, 605590388736, 605590388736},
    /* p            */ {21, 8, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 3,
    /* d_numa       */ 0,
    /* g            */ {2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {302795194368, 605590388736, 605590388736},
    /* p            */ {84, 2, 1},
    /* kmax         */ {999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 1.116973e-12, 2.239198e-12},
    /* ls           */ {4.915842e-12, 1.492092e-11, 7.163592e-11, 1.410487e-10},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736},
    /* p            */ {1, 7, 3, 8},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 2.098825e-12, 2.239198e-12},
    /* ls           */ {4.915842e-12, 1.492092e-11, 1.328118e-10, 1.410487e-10},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736},
    /* p            */ {1, 7, 12, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 1.492092e-11, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 605590388736, 605590388736},
    /* p            */ {1, 7, 24, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 1.116973e-12, 2.098825e-12, 2.239198e-12},
    /* ls           */ {4.915842e-12, 7.163592e-11, 1.328118e-10, 1.410487e-10},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 21, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 1.116973e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 7.163592e-11, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 21, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 84, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 1.116973e-12, 2.098825e-12, 2.239198e-12},
    /* ls           */ {1.492092e-11, 7.163592e-11, 1.328118e-10, 1.410487e-10},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 1.116973e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {1.492092e-11, 7.163592e-11, 1.410487e-10, 7.680059e-05},
    /* m            */ {33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {1.492092e-11, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 4,
    /* d_numa       */ 0,
    /* g            */ {1.116973e-12, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {7.163592e-11, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 1.116973e-12, 2.098825e-12, 2.239198e-12},
    /* ls           */ {4.915842e-12, 1.492092e-11, 7.163592e-11, 1.328118e-10, 1.410487e-10},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736},
    /* p            */ {1, 7, 3, 4, 2},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 1.116973e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 1.492092e-11, 7.163592e-11, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 100931731456, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 8, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 1.492092e-11, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 12, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {1.384990e-13, 1.116973e-12, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 7.163592e-11, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 21, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 5,
    /* d_numa       */ -2,
    /* g            */ {2.852375e-13, 1.116973e-12, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {1.492092e-11, 7.163592e-11, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999}},
    // Hardware parameters for 168 thread(s), policy: spread, subset: L2Cache, L3Cache, NUMANode, Socket, NodeMem, GLOBAL_SYNC
   {
    /* d            */ 6,
    /* d_numa       */ -3,
    /* g            */ {1.384990e-13, 2.852375e-13, 1.116973e-12, 2.098825e-12, 2.239198e-12, 0.000000e+00},
    /* ls           */ {4.915842e-12, 1.492092e-11, 7.163592e-11, 1.328118e-10, 1.410487e-10, 7.680059e-05},
    /* m            */ {1048576, 33554432, 100931731456, 302795194368, 605590388736, 605590388736},
    /* p            */ {1, 7, 3, 4, 2, 1},
    /* kmax         */ {999, 999, 999, 999, 999, 999}}

};

// HWParameter_configurations struct
const cost_models::HW_model::HWParameter_configurations dis_system_params = {
    /* threads_options_num */ {1, 2, 4, 7, 8, 14, 21, 24, 32, 42, 84, 168},
    /* policy_options_str */ {"close", "spread"},
    /* level_options_str  */ {"L2Cache", "L3Cache", "NUMANode", "Socket", "NodeMem", "GLOBAL_SYNC"},
    /* hw_model_thread_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11},
    /* hw_model_policy_id */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    /* hw_model_level_bithash */ {16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63, 16, 17, 18, 20, 24, 48, 19, 21, 25, 49, 22, 26, 50, 28, 52, 56, 23, 27, 51, 29, 53, 57, 30, 54, 58, 60, 31, 55, 59, 61, 62, 63},
    /* hw_models */ hw_models_vector
};

#endif // HW_PARAMS_AMDEPYC9634_HPP
