
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

/**
 * \mainpage Algebraic Programming (ALP) API Specification.
 *
 * This document specifies the ALP API.
 *
 * \par Containers
 *
 * ALP defines the following containers for users to interface with:
 *   -# alp::Scalar
 *   -# alp::Vector
 *   -# alp::Matrix
 *
 * Containers take as a template argument \a T the type that the container
 * stores. The type \a T can be any C++ plain-old-data type.
 *
 * ALP defines primitives for performing IO to and from containers in the
 * \ref IO module.
 *
 * \par Algebraic structures
 *
 * ALP defines the following algebraic structures to interface with:
 *   -# All binary operators defined in alp::operators;
 *   -# identities defined in alp::identities;
 *   -# alp::Monoid structures by combining binary operators and identities;
 *   -# alp::Semiring structures by combining two operators and two identites.
 *
 * For example, a real semiring is composed as follows:
 * \code
 * alp::Semiring<
 *    alp::operators::add< double >, alp::operators::mul< double >,
 *    alp::identities::zero, alp::identities::one
 * > reals;
 * \endcode
 * This semiring forms the basis of most numerical linear algebra.
 *
 * Our definition of monoid and semirings imply that the domains they operate
 * over are derived from the operators. For example, to perform half precision
 * multiplication and accumulate in single precision, the following semiring
 * may be defined:
 * \code
 * alp::Semiring<
 *     alp::operators::add< short float, float, float >,
 *     alp::operators::mul< short float >,
 *     alp::identities::zero, alp::identities::one
 * > mixedReals;
 * \endcode
 *
 * \par Algebraic primitives
 *
 * Operations on containers proceed by calling ALP primitives, which are
 * parametrised in the algebraic structure the operation should proceed with.
 * Primitives are grouped in modules that follow roughly the traditional BLAS
 * taxonomy:
 *   -# \ref BLAS0
 *   -# \ref BLAS1
 *   -# \ref BLAS2
 *   -# \ref BLAS3
 *
 * \par Algebraic structures and views
 *
 * Containers may have structures (e.g., symmetric) and views (e.g., transpose),
 * and may be sparse or dense as per alp::Density. Operations are in principle
 * defined for both sparse \em and dense containers, as well as mixtures of
 * sparse and dense containers, provided that the right algebraic structures are
 * provided -- for example, a sparse vector cannot be reduced into a scalar via
 * alp::foldl when an (associative) operator is given; instead, a monoid
 * structure is required in order to interpret any missing values in a sparse
 * vector.
 *
 * Views allow for the selection of submatrices from a larger matrix, such as
 * for example necessary to express Cholesky factorisation algorithms. Views are
 * constructed through alp::get_view. Please see the slides for concrete
 * examples.
 */
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
#include <alp/rels.hpp>
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

