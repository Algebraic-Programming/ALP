
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

/*
 * @author: A. N. Yzelman.
 * @date: 8th of August, 2016.
 */

/** \mainpage
 *
 * This is a GraphBLAS implementation in ANSI C++11. Authors:
 *   -# A. N. Yzelman, Huawei Technologies France; 2016-2020.
 *   -# A. N. Yzelman, Huawei Technologies Switzerland AG; 2020-current.
 *   -# Aristeidis Mastoras, Huawei Technologies Switzerland AG; 2020-current.
 *   -# Alberto Scolari, Huawei Technologies Switzerland AG; 2021-current.
 *   -# Verner Vlacic, Huawei Technologies Switzerland AG; 2021-current.
 *   -# Auke Booij, Huawei Technologies Switzerland AG; 2021.
 *   -# Dan Iorga, Huawei Technologies Switzerland AG; 2021.
 *   -# Daniel Di Nardo, Huawei Technologies France; 2017.
 *   -# Jonathan M. Nash, Huawei Technologies France; 2017.
 *
 * Contact: albertjan.yzelman@huawei.com
 *
 * This API exposes only two containers: a #grb::Vector and a #grb::Matrix.
 *
 * All primitives defined on these containers must be given a (binary)
 * operator, a #grb::Monoid, or a #grb::Semiring. These monoid and semiring are
 * generalised from their mathematical counterpart in that they holds multiple
 * domains. The monoid consists of one binary operator and a corresponding
 * identity. The semiring consists of one additive operator, one multiplicative
 * operator, one identity under addition, and one identity under multiplication.
 *
 * Monoids and semirings must comply with their regular axioms-- a type system
 * assists users by checking for incorrect operators acting as additive or
 * multiplicative operators. Standard operators and identities are found in
 * their respective namespaces, #grb::operators and #grb::identities,
 * respectively.
 *
 * Monoids and semirings must be supplied with the domain(s) it will operate
 * on. These must be available at compile time. Also the element type of
 * GraphBLAS containers must be set at compile time. The size of a container is
 * set at run-time, but may not change during its life time.
 *
 * This implementation provides various \ref BLAS1 and \ref BLAS2 primitives. To
 * simplify writing generalised algebraic routines, it also provides \ref BLAS0
 * primitives.
 *
 * The three aforementioned ingredients, namely, containers, algebraic relations
 * (such as semirings), and level-{1,2,3} primitives make up the full interface
 * of this DSL.
 *
 * An example is provided within examples/sp.cpp. It demonstrates usage of this
 * API. We now follow with some code snippets from that example. First, the
 * example dataset:
 *
 * \snippet sp.cpp Example Data
 *
 * Matrix creation (5-by-5 matrix, 10 nonzeroes):
 *
 * \snippet sp.cpp Example matrix allocation
 *
 * Vector creation:
 *
 * \snippet sp.cpp Example vector allocation
 *
 * Matrix assignment:
 *
 * \snippet sp.cpp Example matrix assignment
 *
 * Vector assignment:
 *
 * \snippet sp.cpp Example vector assignment
 *
 * Example semiring definition:
 *
 * \snippet sp.cpp Example semiring definition
 *
 * Example semiring use:
 *
 * \snippet sp.cpp Example semiring use: sparse vector times matrix multiplication
 *
 * Example function taking arbitrary semirings:
 *
 * \snippet sp.cpp Example function taking arbitrary semirings
 *
 * Example use of a function taking arbitrary semirings:
 *
 * \snippet sp.cpp Example function call while passing a semiring
 *
 * Full example use case:
 *
 * \snippet sp.cpp Example shortest-paths with semiring adapted to find the most reliable route instead
 *
 * Any GraphBLAS code may execute using any of the backends this implementation
 * defines. Currently, the following backends are stable:
 *   -# #grb::reference, a single-process, auto-vectorising, sequential backend;
 *   -# #grb::reference_omp, a single-process, auto-parallelising, shared-memory
 *      parallel backend based on OpenMP and the aforementioned vectorising
 *      backend;
 *   -# #grb::BSP1D, an auto-parallelising, distributed-memory parallel
 *      backend based on the Lightweight Parallel Foundations (LPF). This is a
 *      multi-process backend and may rely on any single-process backend for
 *      process-local computations. Its combination with the #grb::reference_omp
 *      backend results in a fully hybrid shared- and distributed-memory
 *      GraphBLAS implementation.
 *
 * Backends that are currently under development:
 *   -# #grb::banshee, a single-process, reference-based backend for the Banshee
 *      RISC-V hardware simulator making use of indirection stream semantic
 *      registers (ISSR, in collaboration with Prof. Benini at ETHZ);
 *
 * @author A. N. Yzelman, Huawei Technologies France (2016-2020)
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 */

#ifdef __DOXYGEN__
/**
 * Define this macro to disable libnuma use.
 */
#define _GRB_NO_LIBNUMA

/**
 * Define this macro to disable thread pinning.
 */
#define _GRB_NO_PINNING

/**
 * Defie this macro to compile with PlatformBSP support.
 */
#define _GRB_WITH_LPF

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
#define _GRB_BACKEND reference

/**
 * Which GraphBLAS backend the BSP1D backend should use within a single user
 * process. For possible values, see the single user process options for
 * #_GRB_BACKEND.
 */
#define _GRB_BSP1D_BACKEND
#endif

#ifndef _H_GRAPHBLAS
#define _H_GRAPHBLAS

// load active configuration
#include <graphblas/config.hpp> //defines _GRB_BACKEND and _WITH_BSP

// collects the user-level includes
// the order of these includes matter--
//    do not modify without proper consideration!

// First include all algebraic structures, which have the benefit of not
// depending on anything else
#include <graphblas/ops.hpp>
#include <graphblas/monoid.hpp>
#include <graphblas/semiring.hpp>

// Then include containers. If containers rely on ALP/GraphBLAS primitives that
// are defined as free functions, then container implementations must forward-
// declare those.
#include <graphblas/vector.hpp>
#include <graphblas/matrix.hpp>

// The aforementioned forward declarations must be in sync with the
// declarations of the user primitives defined as free functions in the below.
// The below relies on both algebraic structures/relations as well as container
// definitions. By maintaining the current order, these do not require forward
// declarations.
#include <graphblas/io.hpp>
#include <graphblas/benchmark.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/blas1.hpp>
#include <graphblas/blas2.hpp>
#include <graphblas/blas3.hpp>
#include <graphblas/collectives.hpp>
#include <graphblas/exec.hpp>
#include <graphblas/init.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/pinnedvector.hpp>
#include <graphblas/properties.hpp>
#include <graphblas/spmd.hpp>

#ifdef _GRB_WITH_LPF
 // collects various BSP utilities
 #include <graphblas/bsp/spmd.hpp>
#endif

#endif // end ``_H_GRAPHBLAS''

