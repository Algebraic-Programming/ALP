
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

/**
 * @file
 *
 * The main header to include in order to use the ALP/GraphBLAS API.
 *
 * @author A. N. Yzelman.
 * @date 8th of August, 2016.
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
 *  -# vertex-centric programming, \ref Pregel.
 *
 * Additionally, to ease integration with existing software, ALP defines
 * so-called \ref TRANS libraries, which presently includes (partial)
 * implementations of the \ref SPARSEBLAS and \ref SPBLAS (de-facto) standards,
 * as well as an interface for numerical \ref TRANS_SOLVERS.
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
 * \parblock
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
 * (\f$ C \leftarrow C + AB \f$, #grb::mxm). The latter is typically deemed
 * richer since it requires a semiring structure rather than a more basic binary
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
 * ALP requires that every primitive is \em parallelisable. Every backend that
 * implements primitive for a specific system furthermore must specify
 * <em>performance semantics</em>. Contrary to functional semantics that this
 * reference specifies, performance semantics guarantee certain observable
 * behaviours when it comes to the amount of work, data movement,
 * synchronisation across parallel systems, and/or memory use.
 *
 * @see perfSemantics
 * \endparblock
 *
 * \parblock
 * \par Algebraic Structures
 *
 * ALP/GraphBLAS defines three types of algebra structures, namely, a
 *  -# binary operator such as #grb::operators::add (numerical addition),
 *  -# #grb::Monoid, and
 *  -# #grb::Semiring.
 *
 * Binary operators are parametrised in two input domains and one output domain,
 * \f$ D_1 \times D_2 \to D_3 \f$. The \f$ D_i \f$ are given as template
 * arguments to the operator. A #grb::Monoid is composed from a binary operator
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
 * @see typeTraits
 *
 * Standard operators and identities are found in their respective namespaces,
 * #grb::operators and #grb::identities, respectively. The ALP monoids and
 * semirings are generalised from their standard mathematical definitions in
 * that they hold multiple domains. The description of #grb::Semiring details
 * the underlying mathematical structure that nevertheless can be identified.
 * \endparblock
 *
 * \parblock
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
 * \endparblock
 *
 * @author A. N. Yzelman, Huawei Technologies France (2016-2020)
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 * @}
 *
 * \defgroup typeTraits Algebraic Type Traits
 * @{
 *
 * Algebraic type traits allows compile-time reasoning on algebraic structures.
 *
 * Under <em>algebraic type traits</em>, ALP defines two classes of type traits:
 *  1. classical type traits, akin to, e.g., <tt>std::is_integral</tt>, defined
 *     over the ALP-specific algebraic objects such as #grb::Semiring, and
 *  2. algebraic type traits that allow for the compile-time introspection of
 *     algebraic structures.
 *
 * Under the first class, the following type traits are defined by ALP:
 *  - #grb::is_operator, #grb::is_monoid, and #grb::is_semiring, but also
 *  - #grb::is_container and #grb::is_object.
 *
 * Under the second class, the following type traits are defined by ALP:
 *  - #grb::is_associative, #grb::is_commutative, #grb::is_idempotent, and
 *    #grb::has_immutable_nonzeroes.
 *
 * Algebraic type traits are a central concept to ALP; depending on algebraic
 * properties, ALP applies different optimisations. Properties such as
 * associativity furthermore often define whether primitives may be
 * automatically parallelised. Therefore, some primitives only allow algebraic
 * structures with certain properties.
 *
 * Since algebraic type traits are compile-time, the composition of invalid
 * structures (e.g., composing a monoid out of a non-associative binary
 * operator), or the calling of a primitive using an incompatible algebraic
 * structure, results in an <em>compile-time</em> error. Such errors are
 * furthermore accompanied by clear messages and suggestions.
 *
 * @}
 *
 * \defgroup backends Backends
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
 * The #grb::Backend enum lists all backends known to ALP. Properties of a
 * backend that may affect more advanced user code are collected in
 * #grb::Properties.
 *
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 * @}
 *
 * \defgroup perfSemantics Performance Semantics
 * @{
 *
 * Each ALP primitive, every constructor, and every destructor come with
 * <em>performance semantics</em>, in addition to functional semantics.
 *
 * Performance semantics may differ for different backends-- ALP stringently
 * mandates that backends defines them, thus imposing a significant degree of
 * predictability on implementations of ALP, but does not significantly limit
 * possible implementation choices.
 *
 * \warning Performance semantics should not be mistaken for performance
 *          \em guarantees. The vast majority of computing platforms exhibit
 *          performance variabilities that preclude defining stringent such
 *          guarantees.
 *
 * Performance semantics includes classical asymptotic work analysis in the
 * style of Cormen et alii, as commonly taught as part of basic computer science
 * courses. Aside from making the reasonable (although arguably too uncommon)
 * demand that ALP libraries must clearly document the work complexity of the
 * primitives it defines, ALP furthermore demands such analyses for the
 * following quantities:
 *  - how many times operator(s) may be applied,
 *  - intra-process data movement from main memory to processing units,
 *  - new dynamic memory allocations and/or releases of previously allocated
 *     memory, and
 *  - whether system calls may occur during a call to the given primitive.
 *
 * \note Typically (but not always) the amount of work is proportional to the
 *       number of operator applications.
 *
 * \note Typically (but not necessarily always) if primitives are allowed to
 *       allocate or free dynamic memory, then it may also thus make system
 *       calls.
 *
 * For backends that allow for more than one user process, the following
 * additional performance semantics must be defined:
 *  - inter-process data movement, and
 *  - how many synchronisation steps a primitive requires to complete.
 *
 * Defining such performance semantics are crucial to
 *  1. allow algorithm designers to design the best possible algorithms even if
 *     the target platforms and target use cases vary,
 *  2. allow users to determine scalability under increasing problem sizes, and
 *  3. allow system architects to determine the qualitative effect of scaling up
 *     system resources in an a-priori fashion.
 *
 * These advantages furthermore do not require expensive experimentation on the
 * part of algorithm designers, users, or system architects. However, it puts a
 * significant demand on the implementers and maintainers of ALP.
 *
 * @see backends
 *
 * @author A. N. Yzelman, Huawei Technologies Switzerland AG (2020-current)
 * @}
 */

#ifdef __DOXYGEN__

/**
 * Define this macro to disable the dependence on libnuma.
 *
 * \warning Defining this macro is discouraged and not tested thoroughly.
 *
 * \note The CMake bootstrap treats libnuma as a non-optional dependence.
 */
#define _GRB_NO_LIBNUMA

/**
 * \internal
 * Define this macro to disable thread pinning.
 * \todo Make sure this macro is taken into account for backends that perform
 *       automatic pinning.
 * \endinternal
 */
#define _GRB_NO_PINNING

/**
 * Define this macro to turn off standard input/output support.
 *
 * \warning This macro has only been fully supported within the #grb::banshee
 *          backend, where neither standard <tt>iostream</tt> nor
 *          <tt>stdio.h</tt> were available. If support through the full ALP
 *          implementation would be useful, please raise an issue through
 *          GitHub or Gitee so that we may consider and plan for supporting
 *          this macro more fully.
 */
#define _GRB_NO_STDIO

/**
 * Define this macro to turn off reliance on standard C++ exceptions.
 *
 * \deprecated Support for this macro is being phased out.
 *
 * \note Its intended use is to support ALP/GraphBLAS deployments on platforms
 *       that do not support C++ exceptions, such as some older Android SDK
 *       applications.
 *
 * \warning The safe usage of ALP/GraphBLAS while exceptions are disabled
 *          relies, at present, on the inspection of internal states and the
 *          usage of internal functions. We have no standardised exception-free
 *          way of using ALP/GraphBLAS at present and have no plans to
 *          (continue and/or extend) support for it.
 */
#define _GRB_NO_EXCEPTIONS

/**
 * Define this macro to compile with LPF support.
 *
 * \note The CMake bootstrap automatically defines this flag when a valid LPF
 *       installation is found. This flag is also defined by the ALP/GraphBLAS
 *       compiler wrapper whenever an LPF-enabled backend is selected.
 */
#define _GRB_WITH_LPF

/**
 * \internal
 * Which ALP/GraphBLAS backend should be the default.
 *
 * This flag is overridden by the compiler wrapper, and it is set by the base
 * config.hpp header.
 * \endinternal
 */
#define _GRB_BACKEND reference

/**
 * Which ALP/GraphBLAS backend the BSP1D backend should use for computations
 * within a single user process. The ALP/GraphBLAS compiler wrapper sets this
 * value automatically depending on the choice of backend-- compare, e.g., the
 * #grb::BSP1D backend versus the #grb::hybrid backend.
 */
#define _GRB_BSP1D_BACKEND

/**
 * The ALP/GraphBLAS namespace.
 *
 * All ALP/GraphBLAS primitives, container types, algebraic structures, and type
 * traits are defined within.
 */
namespace grb {

	/**
	 * The namespace for ALP/GraphBLAS algorithms.
	 */
	namespace algorithms {

		/**
		 * The namespace for ALP/Pregel algorithms.
		 */
		namespace pregel {}

	}

	/**
	 * The namespace for programming APIs that automatically translate to
	 * ALP/GraphBLAS.
	 */
	namespace interfaces {}

}

#endif // end ``#ifdef __DOXYGEN__''

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

