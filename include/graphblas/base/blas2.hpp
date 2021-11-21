
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
 * Defines the GraphBLAS level 2 API.
 *
 * @author A. N. Yzelman
 * @date 30th of March 2017
 */

#ifndef _H_GRB_BLAS2_BASE
#define _H_GRB_BLAS2_BASE

#include <assert.h>

#include <graphblas/backends.hpp>
#include <graphblas/blas1.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include "config.hpp"
#include "matrix.hpp"
#include "vector.hpp"

namespace grb {

	/**
	 * \defgroup BLAS2 The Level-2 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that allow GraphBLAS operators, monoids, and
	 * semirings work on a mix of zero-dimensional, one-dimensional, and
	 * two-dimensional containers.
	 *
	 * That is, these functions allow various linear algebra operations on
	 * scalars, objects of type grb::Vector, and objects of type grb::Matrix.
	 *
	 * \note The backends of each opaque data type should match.
	 *
	 * @{
	 */

	/**
	 * Right-handed sparse matrix times vector multiplication, \f$ u = Av \f$.
	 *
	 * Let \f$ u \f$ and \f$ \mathit{mask} \f$ each be a #grb::Vector of #grb::size
	 * \f$ m \f$, \f$ v \f$ be a #grb::Vector of #grb::size \f$ n \f$, and let
	 * \f$ A \f$ be a #Matrix with #grb::nrows \f$ m \f$ and #grb::ncols \f$ n \f$.
	 * Let furthermore \f$ z \f$ be an interal vector of size \f$ m \f$.
	 * A call to this function first computes \f$ z = Av \f$ over the provided
	 * \a ring. It then left-folds \f$ z \f$ into \f$ u \f$ using the provided
	 * \a accumulator.
	 *
	 * @see Vector for an in-depth description of a GraphBLAS vector.
	 * @see size   for retrieving the length of a given GraphBLAS vector.
	 * @see Matrix for an in-depth description of a GraphBLAS matrix.
	 * @see nrows  for retrieving the number of rows of a given GraphBLAS matrix.
	 * @see ncols  for retrieving the number of columns of a given GraphBLAS
	 *             vector.
	 *
	 * Formally, the exact operation executed is
	 *  \f$ u_i^\mathit{out} = u_i^\mathit{in} \bigodot z_i, \f$
	 * for all \f$ i \in \{ 0, 1, \ldots, m-1 \} \f$ for which
	 * \f$ \mathit{mask}_i \f$ evaluates <tt>true</tt>. If there is a nonzero at
	 * \f$ z_i \f$ but no nonzero at \f$ u_i^\mathit{in} \f$ then the latter is interpreted as the additive
	 * identity \f$ \mathbf{0} \f$ of the given \a ring.
	 * For \f$ z \f$, we formally have:
	 *  \f$ z_i = \bigoplus{i=0}^{m-1} \left( A_{ij} \bigotimes v_j \right), \f$
	 * where \f$ \bigodot \f$ represents the \a accumulator, \f$ \bigoplus \f$
	 * represents the additive operator of the provided \a ring, and
	 * \f$ \bigotimes \f$ represents the multiplicative operator of \a ring. If here
	 * \f$ v_j \f$ does not exist, it is considered to be equal to the additive
	 * identity of the given \a ring.
	 *
	 * \note The additive identity of a given \a ring is an annihilator of
	 *       nonzeroes from \f$ A \f$ under the multiplicative operator of \a ring;
	 *       that is, \f$ z_i \f$ will be \f$ \mathbf{0} \f$ always. This can, of
	 *       course, be exploited during sparse matrix--sparse vector (SpMSpV)
	 *       multiplication.
	 *
	 * \note A good implementation is very careful about forming \f$ z \f$
	 *       explicitly and, even if it is formed already, is very careful about
	 *       making use of \f$ z \f$. Making use of an explicit buffer will result
	 *       in \f$ \Theta(m) \f$ data movement and may only be warrented when
	 *       \f$ A \f$ has many nonzeroes per row and \f$ v \f$ is dense.
	 *
	 * @tparam descr    Any combination of one or more #grb::descriptors. When
	 *                  ommitted, the default #grb::descriptors:no_operation will
	 *                  be assumed.
	 * @tparam Ring     The generalised semi-ring the matrix--vector multiplication
	 *                  is to be executed under.
	 * @tparam IOType   The type of the elements of the output vector \a u.
	 * @tparam InputType1 The type of the elements of the input vector \a v.
	 * @tparam InputType2 The type of the elements of the input matrix \a A.
	 * @tparam Operator The type of the \a accumulator. Must be a GraphBLAS
	 *                  operator; see also #grb::operators.
	 * @tparam InputType3 The type of the elements of the mask vector \a mask.
	 * @tparam implementation Which back-end the given vectors and matrices belong
	 *                        to. These must all belong to the same back-end.
	 *
	 * @param[in,out] u The output vector. Depending on the provided
	 *                  \a accumulator, old vector values may affect new values.
	 * @param[in]  mask The mask vector. The vector #grb::size must be equal to
	 *                  that of \a u, \em or it must be equal to zero. A \a mask
	 *                  of grb::size zero will be ignored (assumed <tt>true</tt>
	 *                  always.
	 * @param[in] accumulator The operator \f$ \bigodot \f$ in the above
	 *                        description.
	 * @param[in] A     The input matrix. Its #grb::nrows must equal the
	 *                  #grb::size of \a u.
	 * @param[in] v     The input vector. Its #grb::size must equal the
	 *                  #grb::ncols of \a A.
	 * @param[in] ring  The semiring to perform the matrix--vector multiplication
	 *                  under. Unless #grb::descriptors::no_casting is defined,
	 *                  elements from \a u, \a A, and \a v will be cast to the
	 *                  domains of the additive and multiplicative operators of
	 *                  \a ring as they are applied during the multiplication.
	 *
	 * \warning Even if #grb::operators::right_assign is provided as accumulator,
	 *          old values of \a u may \em not be overwritten if the computation
	 *          ends up not writing any new values to those values. To throw away
	 *          old vector values use grb::descriptors::explicit_zero (for dense
	 *          vectors only if you wish to retain sparsity of the output vector),
	 *          or first simply use grb::clear on \a u.
	 *
	 * The above semantics may be changed by the following descriptors:
	 *   * #descriptors::invert_mask: \f$ u_i^\mathit{out} \f$ will be written to
	 *     if and only if \f$ \mathit{mask}_i \f$ evaluates <tt>false</tt>.
	 *   * #descriptors::transpose_matrix: \f$ A \f$ is interpreted as \f$ A^T \f$
	 *     instead.
	 *   * #descriptors::structural: when evaluating \f$ \mathit{mask}_i \f$, only
	 *     the structure of \f$ \mathit{mask} \f$ is considered (as opposed to its
	 *     elements); if \f$ \mathit{mask} \f$ has a nonzero at its \f$ i \f$th
	 *     index, it is considered to evaluate <tt>true</tt> no matter what the
	 *     actual value of \f$ \mathit{mask}_i \f$ was.
	 *   * #descriptors::structural_complement: a combination of two descriptors:
	 *     #descriptors::structural and #descriptors::invert_mask (and thus
	 *     equivalent to <tt>structural | invert_mask</tt>). Its net effect is if
	 *     \f$ \mathit{mask} \f$ does \em not have a nonzero at the \f$ i \f$th
	 *     index, the mask is considered to evaluate <tt>true</tt>.
	 *   * #descriptors::add_identity: the matrix \f$ A \f$ is instead interpreted
	 *     as \f$ A + \mathbf{1} \f$, where \f$ \mathbf{1} \f$ is the
	 *     multiplicative identity of the given ring.
	 *   * #descriptors::use_index: when referencing \f$ v_i \f$, if assigned, then
	 *     instead of using the value itself, its index \f$ i \f$ is used instead.
	 *   * #descriptors::in_place: the \a accumulator is ignored; the additive
	 *     operator of the given \a ring is used in its place. Under certain
	 *     conditions, an implementation can exploit this semantic to active
	 *     faster computations.
	 *   * #descriptors::explicit_zero: if \f$ \mathbf{0} \f$ would be assigned to
	 *     a previously unassigned index, assign \f$ \mathbf{0} \f$ explicitly to
	 *     that index. Here, \f$ \mathbf{0} \f$ is the additive identity of the
	 *     provided \a ring.
	 *
	 * \parblock
	 * \par Performance semantics
	 * Performance semantics vary depending on whether a mask was provided, and on
	 * whether the input vector is sparse or dense. If the input vector \f$ v \f$
	 * is sparse, let \f$ J \f$ be its set of assigned indices. If a non-trivial
	 * mask \f$ \mathit{mask} \f$ is given, let \f$ I \f$ be the set of indices for
	 * which the corresponding \f$ \mathit{mask}_i \f$ evaluate <tt>true</tt>. Then:
	 *   -# For the performance guarantee on the amount of work this function
	 *      entails the following table applies:<br>
	 *      \f$ \begin{tabular}{cccc}
	 *           Masked & Dense input  & Sparse input \\
	 *           \noalign{\smallskip}
	 *           no  & $\Theta(2\mathit{nnz}(A))$      & $\Theta(2\mathit{nnz}(A_{:,J}))$ \\
	 *           yes & $\Theta(2\mathit{nnz}(A_{I,:})$ & $\Theta(\min\{2\mathit{nnz}(A_{I,:}),2\mathit{nnz}(A_{:,J})\})$
	 *          \end{tabular}. \f$
	 *   -# For the amount of data movements, the following table applies:<br>
	 *      \f$ \begin{tabular}{cccc}
	 *           Masked & Dense input  & Sparse input \\
	 *           \noalign{\smallskip}
	 *           no  & $\Theta(\mathit{nnz}(A)+\min\{m,n\}+m+n)$                         & $\Theta(\mathit{nnz}(A_{:,J}+\min\{m,2|J|\}+|J|)+\mathcal{O}(2m)$ \\
	 *           yes & $\Theta(\mathit{nnz}(A_{I,:})+\min\{|I|,n\}+2|I|)+\mathcal{O}(n)$ &
	 * $\Theta(\min\{\Theta(\mathit{nnz}(A_{I,:})+\min\{|I|,n\}+2|I|)+\mathcal{O}(n),\mathit{nnz}(A_{:,J}+\min\{m,|J|\}+2|J|)+\mathcal{O}(2m))$ \end{tabular}. \f$
	 *   -# A call to this function under no circumstance will allocate nor free
	 *      dynamic memory.
	 *   -# A call to this function under no circumstance will make system calls.
	 * The above performance bounds may be changed by the following desciptors:
	 *   * #descriptors::invert_mask: replaces \f$ \Theta(|I|) \f$ data movement
	 *     costs with a \f$ \mathcal{O}(2m) \f$ cost instead, or a
	 *     \f$ \mathcal{O}(m) \f$ cost if #descriptors::structural was defined as
	 *     well (see below). In other words, implementations are not required to
	 *     implement inverted operations efficiently (\f$ 2\Theta(m-|I|) \f$ data
	 *     movements would be optimal but costs another \f$ \Theta(m) \f$ memory
	 *     to maintain).
	 *   * #descriptors::structural: removes \f$ \Theta(|I|) \f$ data movement
	 *     costs as the mask values need no longer be touched.
	 *   * #descriptors::add_identity: adds, at most, the costs of grb::foldl
	 *     (on vectors) to all performance metrics.
	 *   * #descriptors::use_index: removes \f$ \Theta(n) \f$ or
	 *     \f$ \Theta(|J|) \f$ data movement costs as the input vector values need
	 *     no longer be touched.
	 *   * #descriptors::in_place (see also above): turns \f$ \mathcal{O}(2m) \f$
	 *     data movements into \f$ \mathcal{O}(m) \f$ instead; i.e., it halves the
	 *     amount of data movements for writing the output.
	 *   * #descriptors::dense: the input, output, and mask vectors are assumed to
	 *     be dense. This allows the implementation to skip checks or other code
	 *     blocks related to handling of sparse vectors. This may result in use of
	 *     unitialised memory if any of the provided vectors were, in fact,
	 *     sparse.
	 * Implementations that support multiple user processes must characterise data
	 * movement between then.
	 * \endparblock
	 *
	 * @returns grb::SUCCESS  If the computation completed successfully.
	 * @returns grb::MISMATCH If there is at least one mismatch between vector
	 *                        dimensions or between vectors and the given matrix.
	 * @returns grb::OVERLAP  If two or more provided vectors refer to the same
	 *                        vector.
	 *
	 * When a non-SUCCESS error code is returned, it shall be as though the call
	 * was never made. Note that all GraphBLAS functions may additionally return
	 * #grb::PANIC, which indicates the library has entered an undefined state; if
	 * this error code is returned, the only sensible thing a user can do is exit,
	 * or at least refrain from using any GraphBLAS functions for the remainder of
	 * the application.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords,
		enum Backend implementation = config::default_backend >
	RC mxv( Vector< IOType, implementation, Coords > & u,
		const Vector< InputType3, implementation, Coords > & mask,
		const Matrix< InputType2, implementation > & A,
		const Vector< InputType1, implementation, Coords > & v,
		const Ring & ring,
		typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::mxv (output-masked)\n";
 #endif
#endif
		(void)u;
		(void)mask;
		(void)A;
		(void)v;
		(void)ring;
		return UNSUPPORTED;
	}

	/**
	 * A short-hand for an unmasked #grb::mxv.
	 *
	 * @see grb::mxv for the full documentation.
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename IOType, typename InputType1, typename InputType2, typename Coords, Backend implementation = config::default_backend >
	RC mxv( Vector< IOType, implementation, Coords > & u,
		const Matrix< InputType2, implementation > & A,
		const Vector< InputType1, implementation, Coords > & v,
		const Ring & ring,
		typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * = NULL ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::mxv\n";
#else
		printf( "Selected backend does not implement grb::mxv\n" );
#endif
#endif
		(void)u;
		(void)A;
		(void)v;
		(void)ring;
		return UNSUPPORTED;
	}

	/**
	 * Left-handed sparse matrix times vector multiplication, \f$ u = vA \f$.
	 *
	 * If \a descr does not have #grb::descriptors::transpose_matrix defined, the
	 * semantics and performance semantics of this function are exactly that of
	 * grb::mxv with the #grb::descriptors::transpose_matrix set.
	 * In the other case, the functional and performance semantics of this function
	 * are exactly that of grb::mxv without the #grb::descriptors::transpose_matrix
	 * set.
	 *
	 * @see grb::mxv for the full documentation.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords,
		enum Backend implementation = config::default_backend >
	RC vxm( Vector< IOType, implementation, Coords > & u,
		const Vector< InputType3, implementation, Coords > & mask,
		const Vector< InputType1, implementation, Coords > & v,
		const Matrix< InputType2, implementation > & A,
		const Ring & ring,
		typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::vxm (output-masked)\n";
 #endif
#endif
		(void)u;
		(void)mask;
		(void)v;
		(void)A;
		(void)ring;
		return UNSUPPORTED;
	}

	/**
	 * A short-hand for an unmasked grb::vxm.
	 *
	 * @see grb::vxm for the full documentation.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename Coords,
		enum Backend implementation = config::default_backend >
	RC vxm( Vector< IOType, implementation, Coords > & u,
		const Vector< InputType1, implementation, Coords > & v,
		const Matrix< InputType2, implementation > & A,
		const Ring & ring,
		typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * = NULL ) {
#ifdef _DEBUG
  #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::vxm\n";
 #endif
#endif
		(void)u;
		(void)v;
		(void)A;
		(void)ring;
		return UNSUPPORTED;
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename Coords,
		Backend backend >
	RC vxm( Vector< IOType, backend, Coords > & u,
		const Vector< InputType3, backend, Coords > & mask,
		const Vector< InputType1, backend, Coords > & v,
		const Vector< InputType4, backend, Coords > & v_mask,
		const Matrix< InputType2, backend > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! grb::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement vxm (doubly-masked)\n";
 #endif
#endif
		(void)u;
		(void)mask;
		(void)v;
		(void)v_mask;
		(void)A;
		(void)add;
		(void)mul;
		return UNSUPPORTED;
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename Coords,
		Backend backend >
	RC mxv( Vector< IOType, backend, Coords > & u,
		const Vector< InputType3, backend, Coords > & mask,
		const Matrix< InputType2, backend > & A,
		const Vector< InputType1, backend, Coords > & v,
		const Vector< InputType4, backend, Coords > & v_mask,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! grb::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement mxv (doubly-masked)\n";
 #endif
#endif
		(void)u;
		(void)mask;
		(void)A;
		(void)v;
		(void)v_mask;
		(void)add;
		(void)mul;
		return UNSUPPORTED;
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords,
		Backend backend >
	RC mxv( Vector< IOType, backend, Coords > & u,
		const Vector< InputType3, backend, Coords > & mask,
		const Matrix< InputType2, backend > & A,
		const Vector< InputType1, backend, Coords > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		(void)u;
		(void)mask;
		(void)A;
		(void)v;
		(void)add;
		(void)mul;
		return UNSUPPORTED;
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename Coords,
		Backend backend >
	RC vxm( Vector< IOType, backend, Coords > & u,
		const Vector< InputType1, backend, Coords > & v,
		const Matrix< InputType2, backend > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement vxm (unmasked)\n";
 #endif
#endif
		(void)u;
		(void)v;
		(void)A;
		(void)add;
		(void)mul;
		return UNSUPPORTED;
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords,
		Backend implementation >
	RC vxm( Vector< IOType, implementation, Coords > & u,
		const Vector< InputType3, implementation, Coords > & mask,
		const Vector< InputType1, implementation, Coords > & v,
		const Matrix< InputType2, implementation > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::vxm (output-masked)\n";
 #endif
#endif
		(void)u;
		(void)mask;
		(void)v;
		(void)A;
		(void)add;
		(void)mul;
		return UNSUPPORTED;
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename Coords,
		Backend backend >
	RC mxv( Vector< IOType, backend, Coords > & u,
		const Matrix< InputType2, backend > & A,
		const Vector< InputType1, backend, Coords > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::mxv (unmasked)\n";
 #endif
#endif
		(void)u;
		(void)A;
		(void)v;
		(void)add;
		(void)mul;
		return UNSUPPORTED;
	}

	/**
	 * Executes an arbitrary element-wise user-defined function \a f on all
	 * nonzero elements of a given matrix \a A.
	 *
	 * The user-defined function is passed as a lambda which can capture whatever
	 * the user would like, including one or multiple grb::Vector instances, or
	 * multiple scalars. When capturing vectors, these should also be passed as a
	 * additional arguments to this functions so to make sure those vectors are
	 * synchronised for access on all row- and column- indices corresponding to
	 * locally stored nonzeroes of \a A.
	 *
	 * Only the elements of a single matrix may be iterated upon.
	 *
	 * \note Rationale: while it is reasonable to expect an implementation be able
	 *       to synchronise vector elements, it may be unreasonable to expect two
	 *       different matrices can be jointly accessed via arbitrary lambda
	 *       functions.
	 *
	 * \warning The lambda shall only be executed on the data local to the user
	 *          process calling this function! This is different from the various
	 *          fold functions, or grb::dot, in that the semantics of those
	 *          functions always result in globally synchronised result. To
	 *          achieve the same effect with user-defined lambdas, the users
	 *          should manually prescribe how to combine the local results into
	 *          global ones, for instance, by subsequent calls to
	 *          grb::collectives.
	 *
	 * \note This is an addition to the GraphBLAS. It is alike user-defined
	 *       operators, monoids, and semirings, except it allows execution on
	 *       arbitrarily many inputs and arbitrarily many outputs.
	 *
	 * @tparam Func     the user-defined lambda function type.
	 * @tparam DataType the type of the user-supplied matrix.
	 * @tparam backend  the backend type of the user-supplied vector example.
	 *
	 * @param[in] f The user-supplied lambda. This lambda should only capture
	 *              and reference vectors of the same length as either the row or
	 *              column dimension length of \a A. The lambda function should
	 *              prescribe the operations required to execute on a given
	 *              reference to a matrix nonzero of \a A (of type \a DataType) at
	 *              a given index \f$ (i,j) \f$. Captured GraphBLAS vectors can
	 *              access corresponding elements via Vector::operator[] or
	 *              Vector::operator(). It is illegal to access any element not at
	 *              position \a i if the vector length is equal to the row
	 *              dimension. It is illegal to access any element not at position
	 *              \a j if the vector length is equal to the column dimension.
	 *              Vectors of length neither equal to the column or row dimension
	 *              may \em not be referenced or undefined behaviour will occur. The
	 *              reference to the matrix nonzero is non \a const and may thus be
	 *              modified. New nonzeroes may \em not be added through this lambda
	 *              functionality. The function \a f must have the following
	 *              signature:
	 *              <code>(DataType &nz, const size_t i, const size_t j)</code>.
	 *              The GraphBLAS implementation decides which nonzeroes of \a A are
	 *              dereferenced, and thus also decides the values \a i and \a j the
	 *              user function is evaluated on.
	 * @param[in] A The matrix the lambda is to access the elements of.
	 * @param[in] args All vectors the lambda is to access elements of. Must be of
	 *                 the same length as \a nrows(A) or \a ncols(A). If this
	 *                 constraint is violated, grb::MISMATCH shall be returned. If
	 *                 the vector length equals \a nrows(A), the vector shall be
	 *                 synchronized for access on \a i. If the vector length equals
	 *                 \a ncols(A), the vector shall be synchronized for access on
	 *                 \a j. If \a A is square, the vectors will be synchronised for
	 *                 access on both \a x and \a y. <em>This is a variadic argument
	 *                 and can contain any number of containers of type grb::Vector,
	 *                 passed as though they were separate arguments.</em>
	 *
	 * \warning Using a grb::Vector inside a lambda passed to this function while
	 *          not passing that same vector into \a args, will result in undefined
	 *          behaviour.
	 *
	 * \warning Due to the constraints on \a f described above, it is illegal to
	 *          capture some vector \a y and have the following line in the body
	 *          of \a f: <code>x[i] += x[i+1]</code>. Vectors can only be
	 *          dereferenced at position \a i and \a i alone, and similarly for
	 *          access using \a j. For square matrices, however, the following
	 *          code in the body is accepted, however: <code>x[i] += x[j]</code>.
	 *
	 * @return grb::SUCCESS  When the lambda is successfully executed.
	 * @return grb::MISMATCH When two or more vectors passed to \a args are not of
	 *                       appropriate length.
	 *
	 * \warning Captured scalars will be local to the user process executing the
	 *          lambda. To retrieve the global dot product, an allreduce must
	 *          explicitly be called.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::operator()()
	 * @see Vector::lambda_reference
	 */
	template< typename Func, typename DataType, Backend implementation = config::default_backend, typename... Args >
	RC eWiseLambda( const Func f, const Matrix< DataType, implementation > & A, Args... /*args*/ ) {
#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
		std::cerr << "Selected backend does not implement grb::eWiseLambda (matrices)\n";
 #endif
#endif
		(void)f;
		(void)A;
		return UNSUPPORTED;
	}

	/** @} */

} // namespace grb

#endif // end _H_GRB_BLAS2_BASE
