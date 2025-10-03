
// Auto-generated hardware parameters for ARM920 with filtered levels: L3Cache, NUMANode, NodeMem, GLOBAL_SYNC
// Generated on 2025-10-02 12:28:03

#ifndef HW_PARAMS_ARM920_HPP
#define HW_PARAMS_ARM920_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations
const std::vector<cost_models::HW_model::HWParameters> dis_system_params = {

    // Hardware parameters for 1 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 1,
    /* g            */ {7.862377e-11, 1.182302e-10, 1.805582e-10, 4.529953e-07},
    /* ls           */ {3.156902e-09, 4.446325e-09, 7.555633e-09, 5.722046e-07},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 2 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 2,
    /* g            */ {5.382400e-11, 6.628249e-11, 9.150356e-11, 4.112720e-06},
    /* ls           */ {2.111440e-09, 2.601375e-09, 3.789501e-09, 1.443624e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 4 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 4,
    /* g            */ {1.971691e-11, 2.959100e-11, 4.283082e-11, 1.133084e-05},
    /* ls           */ {8.030478e-10, 1.091995e-09, 1.709075e-09, 2.974868e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 8 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 8,
    /* g            */ {1.248487e-11, 1.639812e-11, 2.221025e-11, 2.232492e-05},
    /* ls           */ {8.315024e-10, 8.000075e-10, 9.960855e-10, 1.741052e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 12 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 12,
    /* g            */ {1.103070e-11, 1.299757e-11, 1.599544e-11, 1.335541e-05},
    /* ls           */ {7.495883e-10, 7.230107e-10, 8.849290e-10, 1.320640e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 16 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 16,
    /* g            */ {1.105703e-11, 1.144454e-11, 1.361904e-11, 1.130104e-05},
    /* ls           */ {8.018654e-10, 7.700748e-10, 8.888564e-10, 1.333207e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 24 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 24,
    /* g            */ {6.263263e-12, 7.885280e-12, 9.958215e-12, 2.566576e-05},
    /* ls           */ {4.322670e-10, 4.835609e-10, 6.161590e-10, 2.819796e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 32 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 32,
    /* g            */ {5.419625e-12, 7.044473e-12, 9.069817e-12, 5.444661e-05},
    /* ls           */ {3.901082e-10, 4.492068e-10, 6.102702e-10, 4.574992e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 48 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 48,
    /* g            */ {3.475620e-12, 5.709525e-12, 8.736049e-12, 5.992055e-05},
    /* ls           */ {2.487641e-10, 3.453875e-10, 6.124715e-10, 4.929155e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 64 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 64,
    /* g            */ {2.463833e-12, 5.036471e-12, 8.885783e-12, 9.840652e-05},
    /* ls           */ {1.791912e-10, 3.195031e-10, 6.525157e-10, 6.764457e-05},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 96 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 96,
    /* g            */ {1.450486e-12, 4.560187e-12, 9.196569e-12, 1.305498e-04},
    /* ls           */ {1.144697e-10, 2.911553e-10, 6.744683e-10, 1.413514e-04},
    /* m            */ {25165824, 135291469824, 541165879296, 541165879296},
    /* p            */ {24, 1, 4, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "NUMANode", "NodeMem", "GLOBAL_SYNC"}}

};

#endif // HW_PARAMS_ARM920_HPP
