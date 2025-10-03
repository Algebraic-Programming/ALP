
// Auto-generated hardware parameters for ARM920 with filtered levels: L3Cache, Socket, NodeMem, GLOBAL_SYNC
// Generated on 2025-09-26 11:06:45

#ifndef HW_PARAMS_ARM920_HPP
#define HW_PARAMS_ARM920_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations
const std::vector<cost_models::HW_model::HWParameters> dis_system_params = {

    // Hardware parameters for 1 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 1,
    /* g            */ {7.926437e-11, 1.308842e-10, 1.822264e-10, 4.053116e-07},
    /* ls           */ {3.147774e-09, 4.933732e-09, 7.531626e-09, 4.291534e-07},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 2 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 2,
    /* g            */ {4.933195e-11, 7.068031e-11, 9.834070e-11, 8.487701e-06},
    /* ls           */ {3.282291e-09, 3.327598e-09, 4.156659e-09, 7.700920e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 4 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 4,
    /* g            */ {4.349034e-11, 4.711087e-11, 5.766357e-11, 1.757741e-05},
    /* ls           */ {3.169343e-09, 3.086671e-09, 3.675988e-09, 7.367134e-06},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 8 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 8,
    /* g            */ {2.084298e-11, 2.585017e-11, 2.991737e-11, 1.290143e-05},
    /* ls           */ {1.540131e-09, 1.658305e-09, 1.950987e-09, 1.540780e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 12 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 12,
    /* g            */ {1.377165e-11, 2.159372e-11, 2.113758e-11, 1.498461e-05},
    /* ls           */ {1.043762e-09, 1.460243e-09, 1.410074e-09, 1.694361e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 24 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 24,
    /* g            */ {6.904198e-12, 1.981243e-11, 1.645436e-11, 1.988908e-05},
    /* ls           */ {5.264774e-10, 1.331178e-09, 1.134929e-09, 2.686779e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 48 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 48,
    /* g            */ {3.731182e-12, 8.850154e-12, 1.594113e-11, 9.877135e-05},
    /* ls           */ {2.960106e-10, 5.890118e-10, 1.084965e-09, 9.626945e-05},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}},
    // Hardware parameters for 96 thread(s)
   {
    /* d            */ 4,
    /* threads      */ 96,
    /* g            */ {4.485143e-12, 5.248954e-12, 6.700621e-12, 2.418131e-04},
    /* ls           */ {3.135494e-10, 3.787572e-10, 4.705854e-10, 2.748723e-04},
    /* m            */ {25165824, 270582939648, 541165879296, 541165879296},
    /* p            */ {24, 2, 2, 1},
    /* kmax         */ {999, 999, 999, 999},
    /* level_names  */ {"L3Cache", "Socket", "NodeMem", "GLOBAL_SYNC"}}

};

#endif // HW_PARAMS_ARM920_HPP
