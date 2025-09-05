
// Auto-generated hardware parameters for ARM920 with filtered levels: L3Cache, Socket, NodeMem
// Generated on 2025-09-02 16:43:41

#ifndef HW_PARAMS_ARM920_HPP
#define HW_PARAMS_ARM920_HPP

#include "cost_models.hpp"

// Array of hardware parameters for different thread configurations
const std::vector<cost_models::HW_model::HWParameters> dis_system_params = {
    // Hardware parameters for 1 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 1,
        /* g            */ {7.888400e-11, 1.429237e-10, 1.769150e-10},
        /* ls           */ {3.804431e-09, 5.943001e-09, 8.474460e-09},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 2 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 2,
        /* g            */ {7.939308e-11, 2.629301e-10, 3.323472e-10},
        /* ls           */ {3.329684e-09, 1.040319e-08, 1.571885e-08},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 4 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 4,
        /* g            */ {1.279385e-10, 1.865067e-10, 2.238136e-10},
        /* ls           */ {7.977480e-09, 9.995450e-09, 1.243209e-08},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 8 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 8,
        /* g            */ {1.557452e-10, 1.882778e-10, 2.253047e-10},
        /* ls           */ {1.054513e-08, 1.152198e-08, 1.374962e-08},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 12 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 12,
        /* g            */ {1.447166e-10, 2.366003e-10, 2.376501e-10},
        /* ls           */ {1.136634e-08, 1.518419e-08, 1.538356e-08},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 24 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 24,
        /* g            */ {1.358497e-10, 4.465584e-10, 3.729654e-10},
        /* ls           */ {1.063378e-08, 2.857819e-08, 2.440605e-08},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 48 thread(s)
    {
        /* d            */ 3,
        /* threads      */ 48,
        /* g            */ {2.084867e-10, 4.164649e-10, 7.333445e-10},
        /* ls           */ {1.401557e-08, 2.516966e-08, 4.810795e-08},
        /* m            */ {25165824, 270582939648, 541165879296},
        /* p            */ {24, 2, 2},
        /* kmax         */ {999, 999, 999}},

    // Hardware parameters for 96 thread(s)
    {
        /* d            */ 2,
        /* threads      */ 96,
        /* g            */ {5.658618e-10, 8.464927e-10},
        /* ls           */ {4.448333e-08, 6.001962e-08},
        /* m            */ {25165824, 541165879296},
        /* p            */ {24, 2},
        /* kmax         */ {999, 999}}
};

#endif // HW_PARAMS_ARM920_HPP
