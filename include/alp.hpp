
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef __DOXYGEN__
/**
 * Define this macro to disable libnuma use.
 */
#define _ALP_NO_LIBNUMA

/**
 * Define this macro to disable thread pinning.
 */
#define _ALP_NO_PINNING

/**
 * Defie this macro to compile with PlatformBSP support.
 */
#define _ALP_WITH_LPF

/**
 * Which GraphBLAS backend should be default.
 *
 * Known single user-process options:
 *  -# reference
 *  -# reference_omp
 *
 * Known multiple user-process options:
 *  -# BSP1D
 */
#define _ALP_BACKEND reference

/**
 * Which GraphBLAS backend the BSP1D backend should use within a single user
 * process. For possible values, see the single user process options for
 * #_ALP_BACKEND.
 */
#define _ALP_BSP1D_BACKEND
#endif

#ifndef _H_ALP
#define _H_ALP

// do NOT remove this #if, in order to protect this header from
// clang-format re-ordering
#if 1
// load active configuration
// #include <alp/config.hpp> //defines _ALP_BACKEND and _WITH_BSP
#endif

// #pragma message "Included ALP.hpp"

// collects the user-level includes
// #include <alp/benchmark.hpp>
#include <alp/blas0.hpp>
#include <alp/blas1.hpp>
#include <alp/blas2.hpp>
#include <alp/blas3.hpp>
// #include <alp/collectives.hpp>
#include <alp/exec.hpp>
#include <alp/init.hpp>
// #include <alp/io.hpp>
// #include <alp/ops.hpp>
// #include <alp/pinnedvector.hpp>
// #include <alp/properties.hpp>
// #include <alp/semiring.hpp>
// #include <alp/spmd.hpp>

#ifdef _ALP_BACKEND
// #pragma message "_ALP_BACKEND defined"
// include also the main data types in order to have the default definitions
// but ONLY if a default backend is define; otherwise, the previous headers
// contain the relevant definitions (without defaults)
#include <alp/matrix.hpp>
#include <alp/vector.hpp>
#endif

#endif // end ``_H_ALP''
