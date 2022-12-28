
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
 * The Algebraic Programming (ALP) project is a modern and humble C++
 * programming framework that achieves scalable and high performance.
 *
 * With ALP, programmers are encouraged to express programs using algebraic
 * concepts directly. ALP is a humble programming model in that it hides all
 * optimisations pertaining to parallelisation, vectorisation, and other
 * complexities with programming large-scale and heterogeneous systems.
 *
 * ALP presently exposes the following interfaces:
 *  -# generalised sparse linear algebra, \ref GraphBLAS;
 *  -# vertex-centric programming, ALP/Pregel.
 *
 * Several other programming interfaces are under design at present.
 *
 * For authors who contributed to ALP, please see the NOTICE file.
 *
 * Contact:
 *  - https://github.com/Algebraic-Programming/ALP
 *  - https://gitee.com/CSL-ALP/graphblas/
 *  - albertjan.yzelman@huawei.com
 *
 * @author A. N. Yzelman, Huawei Technologies France (2016-2020)
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 *
 * \defgroup GraphBLAS ALP/GraphBLAS
 * @{
 *
 * @brief ALP/GraphBLAS enables sparse linear algebraic programming.
 *
 * \par API introduction
 *
 * ALP/GraphBLAS is an ANSI C++11 variant of the C GraphBLAS standard with a few
 * different choices and an emphasis on portability and auto-parallelisation. It
 * exposes only two containers: #grb::Vector and #grb::Matrix. A template
 * argument controls the type of the values contained within a container.
 *
 * A container may have between \f$ 0 \f$ and \f$ c \f$ values, and each such
 * value has a coordinate. The value \f$ c \f$ is the \em capacity of a
 * container, and at most equals the \em size of that container. The size of a
 * matrix is the product of its number of rows and its number of columns.
 * Containers with fewer values than their size are considered \em sparse, while
 * those with as many values as their size are considered \em dense. Scalars
 * correspond to the standard C++ plain-old-data types, and, as such, have size,
 * capacity, and number of values equal to one-- scalars are always dense. 
 *
 * For matrices, their size can be derived from #grb::nrows and #grb::ncols,
 * while for vectors their size may be immediately retrieved via #grb::size.
 * For both vectors and matrices, their capacity and current number of values
 * may be retrieved via #grb::capacity and #grb::nnz, respectively. Finally,
 * containers have a unique identifier that may be retrieved via #grb::getID.
 * These identifiers are assigned in a deterministic fashion, so that for
 * deterministic programs executed with the same number of processes, the same
 * containers will be assigned the same IDs.
 *
 * Containers may be populated using #grb::set or by using dedicated I/O
 * routines such as #grb::buildVectorUnique or #grb::buildMatrixUnique. Here,
 * \em unique refers to the collection of values that should be ingested having
 * no duplicate coordinates; i.e., there are no two values that map to the same
 * coordinate. The first argument to either function is the output container,
 * which is followed by an iterator pair that points to a collection of values
 * to be ingested into the output container.
 *
 * ALP/GraphBLAS supports multiple user processes \f$ P \f$. If \f$ P > 1 \f$,
 * there is a difference between #grb::SEQUENTIAL and #grb::PARALLEL I/O. The
 * default I/O mode is #grb::PARALLEL, which may be overridden by supplying
 * #grb::SEQUENTIAL as a fourth and final argument to the input routines. In
 * sequential I/O, the iterator pair must point to the exact same collection
 * of input values on each of the \f$ P \f$ user processes. In the parallel
 * mode, however, each iterator pair points to disjoint value sets at each of
 * the processes, while their union is what is logically ingested into the
 * output container.
 *
 * Output iteration is done using the standard STL-style iterators. ALP,
 * however, only supports const_iterators on output. Output iterators default
 * to sequential mode also.
 *
 * Primitives perform algebraic operations on containers while using explicitly
 * supplied algebraic structures. Primitives may be as simple as the
 * element-wise application of a binary operator to two input vectors,
 * generating values in a third output vector (\f$ z = x \odot y \f$,
 * #grb::eWiseApply), or may be as rich as multiplying two matrices together
 * whose result is to be added in-place to a third matrix
 * (\f$ C \into C + AB \f$, #grb::mxm). The latter is typically deemed richer
 * since it requires a semiring structure rather than a more basic binary
 * operator.
 *
 * Primitives are grouped according to their classical BLAS levels:
 *  - \ref BLAS0
 *  - \ref BLAS1
 *  - \ref BLAS2
 *  - \ref BLAS3
 *
 * The "level-0" primitives operate on scalars, and in terms of arithmetic
 * intensity match those of level-1 primitives-- however, since standard BLAS
 * need not define scalar operations this specification groups them separately.
 * All primitives except for #grb::set and #grb::eWiseApply are \em in-place,
 * meaning that new output values are "added" to any pre-existing contents in
 * output containers. The operator used for addition is derived from the
 * algebraic structure that the primitive is called with.
 *
 * ALP/GraphBLAS defines three types of algebra structures, namely, a
 *  -# binary operator such as #grb::operators::add (numerical addition),
 *  -# #grb::Monoid, and
 *  -# #grb::Semiring.
 *
 * Binary operators are parametrised in two input domains and one output domain,
 * \f$ D_1 \times D_2 \to D_3 \f$. The \f$ D_i \f$ are given as template
 * arguments to the operator. A #grb::monoid is composed from a binary operator
 * coupled with an identity. For example, the additive monoid is defined as
 * \code
 *  grb::Monoid<
 *    grb::operators::add< double >,
 *    grb::identities::zero
 *  >
 * \endcode
 * Note that passing a single domain as a template argument to a binary operator
 * is a short-hand for an operator with \f$ D_{\{1,2,3\}} \f$ equal to the same
 * domain.
 *
 * Likewise, a #grb::Semiring is composed from two monoids, where the first,
 * the so-called additive monoid, furthermore must be commutative. The classic
 * semiring over integers taught in elementary school, for example, reads
 * \code
 *  grb::Semiring<
 *    grb::operators::add< unsigned int >,
 *    grb::operators::mul< unsigned int >,
 *    grb::identities::zero,
 *    grb::identities::one
 *  >
 * \endcode
 *
 * Monoids and semirings must comply with their regular axioms-- a type system
 * assists users by checking for incorrect operators acting as additive or
 * multiplicative monoids. Errors are reported <em>at compile time</em>, through
 * the use of <em>algebraic type traits</em> such as #grb::is_associative.
 *
 * Standard operators and identities are found in their respective namespaces,
 * #grb::operators and #grb::identities, respectively. The ALP monoids and
 * semirings are generalised from their standard mathematical definitions in
 * that they hold multiple domains. The description of #grb::Semiring details
 * the underlying mathematical structure that nevertheless can be identified.
 *
 * \par ALP/GraphBLAS by example
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
 * @author A. N. Yzelman, Huawei Technologies France (2016-2020)
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 * @}
 *
 * \defgroup Backends ALP backends
 * @{
 *
 * ALP code is compiled using a compiler wrapper, which optionally takes a
 * backend parameter as an argument. The backend selection controls for which
 * use case the code is compiled. Options that are always included are:
 *   -# #grb::reference, a single-process, auto-vectorising, sequential backend;
 *   -# #grb::reference_omp, a single-process, auto-parallelising, shared-memory
 *      parallel backend based on OpenMP and the aforementioned vectorising
 *      backend;
 *   -# grb::hyperdags, a backend that captures the meta-data of computations
 *      while delegating the actual work to the #grb::reference backend. At
 *      program exit, the #grb::hyperdags backend dumps a HyperDAG of the
 *      computations performed.
 *
 * Additionally, the following backends may be enabled by providing their
 * dependences before building ALP:
 *   -# #grb::BSP1D, an auto-parallelising, distributed-memory parallel
 *      backend based on the Lightweight Parallel Foundations (LPF). This is a
 *      multi-process backend and may rely on any single-process backend for
 *      process-local computations, which by default is #grb::reference.
 *      Distributed-memory auto-parallelisation is achieved using a row-wise
 *      one-dimensional block-cyclic distributon.
 *      Its combination with the #grb::reference_omp
 *      backend results in a fully hybrid shared- and distributed-memory
 *      GraphBLAS implementation.
 *   -# #grb::hybrid, essentially the same backend as #grb::BSP1D, but now
 *      composed with the #grb::reference_omp backend for process-local
 *      computations. This backend facilitates full hybrid shared- and
 *      distributed-memory parallelisation.
 *   -# #grb::banshee, a single-process, reference-based backend for the Banshee
 *      RISC-V hardware simulator making use of indirection stream semantic
 *      registers (ISSR). Written by Dan Iorga in collaboration with ETHZ. This
 *      backend is outdated, but, last tested, remained functional.
 *
 * @author A. N. Yzelman, Huawei Technologies France (2016-2020)
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 * @}
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

