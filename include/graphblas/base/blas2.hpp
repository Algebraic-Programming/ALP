
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
	 * Right-handed in-place doubly-masked sparse matrix times vector
	 * multiplication, \f$ u = u + Av \f$.
	 *
	 * Aliases to this function exist that do not include masks:
	 *  - #grb::mxv( u, u_mask, A, v, semiring );
	 *  - #grb::mxv( u, A, v, semiring );
	 * When masks are omitted, the semantics shall be the same as though a dense
	 * Boolean vector of the appropriate size with all elements set to
	 * <tt>true</tt> was given as a mask. We thus describe the semantics of the
	 * fully masked variant only.
	 *
	 * \note If only an input mask \a v_mask is intended to be given (and no output
	 *       mask \a u_mask), then \a u_mask must nonetheless be explicitly given.
	 *       Passing an empty Boolean vector for \a u_mask is sufficient.
	 *
	 * Let \f$ u, \mathit{u\_mask} \f$ be vectors of size \f$ m \f$, let
	 * \f$ v, \mathit{v\_mask} \f$ be vectors of size \f$ n \f$, and let
	 * \f$ A \f$ be an \f$ m \times n \f$ matrix. Then, a call to this function
	 * computes \f$ u = u + Av \f$ but:
	 *   1. only for the elements \f$ u_i \f$ for which \f$ \mathit{u\_mask}_i \f$
	 *      evaluates <tt>true</tt>; and
	 *   2. only considering the elements \f$ v_j \f$ for which
	 *      \f$ \mathit{v\_mask}_v \f$ evaluates <tt>true</tt>, and otherwise
	 *      substituting the zero element under the given semiring.
	 *
	 * When multiplying a matrix nonzero element \f$ a_{ij} \in A \f$, it shall
	 * be multiplied with an element \f$ x_j \f$ using the multiplicative operator
	 * of the given \a semiring.
	 *
	 * When accumulating multiple contributions of multiplications of nonzeroes on
	 * some row \f$ i \f$, the additive operator of the given \a semiring shall be
	 * used.
	 *
	 * Nonzero resulting from computing \f$ Av \f$ are accumulated into any pre-
	 * existing values in \f$ u \f$ by the additive operator of the given
	 * \a semiring.
	 *
	 * If elements from \f$ v \f$, \f$ A \f$, or \f$ u \f$ were missing, the zero
	 * identity of the given \a semiring is substituted.
	 *
	 * If nonzero values from \f$ A \f$ were missing, the one identity of the given
	 * semiring is substituted.
	 *
	 * \note A nonzero in \f$ A \f$ may not have a nonzero value in case it is
	 *       declared as <tt>grb::Matrix< void ></tt>.
	 *
	 * The following template arguments \em may be explicitly given:
	 *
	 * @tparam descr      Any combination of one or more #grb::descriptors. When
	 *                    ommitted, the default #grb::descriptors:no_operation will
	 *                    be assumed.
	 * @tparam Semiring   The generalised semiring the matrix--vector
	 *                    multiplication is to be executed under.
	 *
	 * The following template arguments will be inferred from the input arguments:
	 *
	 * @tparam IOType     The type of the elements of the output vector \a u.
	 * @tparam InputType1 The type of the elements of the input vector \a v.
	 * @tparam InputType2 The type of the elements of the input matrix \a A.
	 * @tparam InputType3 The type of the output mask (\a u_mask) elements.
	 * @tparam InputType4 The type of the input mask (\a v_mask) elements.
	 *
	 * \internal
	 * The following template arguments will be inferred from the input arguments
	 * and generally do not concern end-users:
	 *
	 * @tparam Coords  Which coordinate class is used to maintain sparsity
	 *                 structures.
	 * @tparam RIT     The integer type used for row indices.
	 * @tparam CIT     The integer type used for column indices.
	 * @tparam NIT     The integer type used for nonzero indices.
	 * @tparam backend The backend implementing the SpMV multiplication. The input
	 *                 containers must all refer to the same backend.
	 * \endinternal
	 *
	 * The following arguments are mandatory:
	 *
	 * @param[in,out] u    The output vector.
	 * @param[in]     A    The input matrix. Its #grb::nrows must equal the
	 *                     #grb::size of \a u.
	 * @param[in]     v    The input vector. Its #grb::size must equal the
	 *                     #grb::ncols of \a A.
	 * @param[in] semiring The semiring to perform the matrix--vector
	 *                     multiplication under. Unless
	 *                     #grb::descriptors::no_casting is defined, elements from
	 *                     \a u, \a A, and \a v will be cast to the domains of the
	 *                     additive and multiplicative operators of \a semiring.
	 *
	 * The vector \a v may not be the same as \a u.
	 *
	 * Instead of passing a \a semiring, users may opt to provide an additive
	 * commutative monoid and a binary multiplicative operator instead. In this
	 * case, \a A may not be a pattern matrix (that is, it must not be of type
	 * <tt>grb::Matrix< void ></tt>).
	 *
	 * The \a semiring (or the commutative monoid - binary operator pair) is
	 * optional if they are passed as a template argument instead.
	 *
	 * \note When providing a commutative monoid - binary operator pair, ALP
	 *       backends are precluded from employing distributative laws in
	 *       generating optimised codes.
	 *
	 * Non-mandatory arguments are:
	 *
	 * @param[in] u_mask The output mask. The vector must be of equal size as \a u,
	 *                   \em or it must be empty (have size zero).
	 * @param[in] v_mask The input mask. The vector must be of equal size as \a v,
	 *                   \em or it must be empty (have size zero).
	 * @param[in] phase  The requested phase for this primitive-- see
	 *                   #grb::Phase for details.
	 *
	 * The vectors \a u_mask and \a v_mask may never be the same as \a u.
	 *
	 * An empty \a u_mask will behave semantically the same as providing no mask;
	 * i.e., as a mask that evaluates <tt>true</tt> at every position.
	 *
	 * If \a phase is not given, it will be set to the default #grb::EXECUTE.
	 *
	 * If \a phase is #grb::EXECUTE, then the capacity of \a u must be greater than
	 * or equal to the capacity required to hold all output elements of the
	 * requested computation.
	 *
	 * The above semantics may be changed by the following descriptors:
	 *   - #descriptors::transpose_matrix: \f$ A \f$ is interpreted as \f$ A^T \f$
	 *     instead.
	 *   - #descriptors::add_identity: the matrix \f$ A \f$ is instead interpreted
	 *     as \f$ A + \mathbf{1} \f$, where \f$ \mathbf{1} \f$ is the one identity
	 *     (i.e., multiplicative identity) of the given \a semiring.
	 *   - #descriptors::invert_mask: \f$ u_i \f$ will be written to if and only if
	 *     \f$ \mathit{u\_mask}_i \f$ evaluates <tt>false</tt>, and \f$ v_j \f$
	 *     will be read from if and only if \f$ \mathit{v\_mask}_j \f$ evaluates
	 *     <tt>false</tt>.
	 *   - #descriptors::structural: when evaluating \f$ \mathit{mask}_i \f$, only
	 *     the structure of \f$ \mathit{u\_mask}, \mathit{v\_mask} \f$ is
	 *     considered, as opposed to considering their values.
	 *   - #descriptors::structural_complement: a combination of two descriptors:
	 *     #descriptors::structural and #descriptors::invert_mask.
	 *   - #descriptors::use_index: when reading \f$ v_i \f$, then, if there is
	 *     indeed a nonzero \f$ v_i \f$, use the value \f$ i \f$ instead. This
	 *     casts the index from <tt>size_t</tt> to the \a InputType1 of \a v.
	 *   - #descriptors::explicit_zero: if \f$ u_i \f$ was unassigned on entry and
	 *     if \f$ (Av)_i \f$ is \f$ \mathbf{0} \f$, then instead of leaving
	 *     \f$ u_i \f$ unassigned, it is set to \f$ \mathbf{0} \f$ explicitly.
	 *     Here, \f$ \mathbf{0} \f$ is the additive identity of the provided
	 *     \a semiring.
	 *   - #descriptors::safe_overlap: the vectors \a u and \a v may now be the
	 *     same container. The user guarantees that no race conditions exist during
	 *     the requested computation, however. The user may guarantee this due to a
	 *     a very specific structure of \a A and \a v, or via an intelligently
	 *     constructed \a u_mask, for example.
	 *
	 * \parblock
	 * \par Performance semantics
	 * Backends must specify performance semantics in the amount of work, intra-
	 * process data movement, inter-process data movement, and the number of
	 * user process synchronisations required. They should also specify whether
	 * any system calls may be made, in particularly those related to dynamic
	 * memory management. If new memory may be allocated, they must specify how
	 * much.
	 * \endparblock
	 *
	 * @returns grb::SUCCESS  If the computation completed successfully.
	 * @returns grb::MISMATCH If there is at least one mismatch between vector
	 *                        dimensions or between vectors and the given matrix.
	 * @returns grb::OVERLAP  If two or more provided vectors refer to the same
	 *                        container while this was not allowed.
	 *
	 * When any of the above non-SUCCESS error code is returned, it shall be as
	 * though the call was never made-- the state of all container arguments and
	 * of the application remain unchanged, save for the returned error code.
	 *
	 * @returns grb::PANIC Indicates that the application has entered an undefined
	 *                     state.
	 *
	 * \note Should this error code be returned, the only sensible thing to do is
	 *       exit the application as soon as possible, while refraining from using
	 *       any other ALP pritimives.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend backend
	>
	RC mxv(
		Vector< IOType, backend, Coords > &u,
		const Vector< InputType3, backend, Coords > &u_mask,
		const Matrix< InputType2, backend, RIT, CIT, NIT > &A,
		const Vector< InputType1, backend, Coords > &v,
		const Vector< InputType4, backend, Coords > &v_mask,
		const Semiring &semiring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Semiring >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement mxv "
			<< "(doubly-masked, semiring)\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_doubly_masked_mxv_sr = false;
		assert( selected_backed_does_not_support_doubly_masked_mxv_sr );
#endif
		(void) u;
		(void) u_mask;
		(void) A;
		(void) v;
		(void) v_mask;
		(void) semiring;
		return UNSUPPORTED;
	}

	/**
	 * Left-handed in-place doubly-masked sparse matrix times vector
	 * multiplication, \f$ u = u + vA \f$.
	 *
	 * A call to this function is exactly equivalent to calling
	 *   - #grb::vxm( u, u_mask, A, v, v_mask, semiring, phase )
	 * with the #descriptors::transpose_matrix flipped.
	 *
	 * See the documentation of #grb::mxv for the full semantics of this function.
	 * Like with #grb::mxv, aliases to this function exist that do not include
	 * masks:
	 *  - #grb::vxm( u, u_mask, v, A, semiring, phase );
	 *  - #grb::vxm( u, v, A, semiring, phase );
	 * Similarly, aliases to this function exist that take an additive commutative
	 * monoid and a multiplicative binary operator instead of a semiring.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Semiring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename Coords, typename RIT, typename CIT, typename NIT,
		enum Backend backend
	>
	RC vxm(
		Vector< IOType, backend, Coords > &u,
		const Vector< InputType3, backend, Coords > &u_mask,
		const Vector< InputType1, backend, Coords > &v,
		const Vector< InputType4, backend, Coords > &v_mask,
		const Matrix< InputType2, backend, RIT, CIT, NIT > &A,
		const Semiring &semiring = Semiring(),
		const Phase &phase = EXECUTE,
		typename std::enable_if<
			grb::is_semiring< Semiring >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!grb::is_object< IOType >::value,
		void >::type * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement doubly-masked grb::vxm\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_doubly_masked_vxm_sr = false;
		assert( selected_backend_does_not_support_doubly_masked_vxm_sr );
#endif
		(void) u;
		(void) u_mask;
		(void) v;
		(void) v_mask;
		(void) A;
		(void) semiring;
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
	template<
		typename Func, typename DataType,
		typename RIT, typename CIT, typename NIT,
		Backend implementation = config::default_backend,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, implementation, RIT, CIT, NIT > &A,
		Args... /*args*/
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::eWiseLambda (matrices)\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_matrix_eWiseLamba = false;
		assert( selected_backend_does_not_support_matrix_eWiseLamba );
#endif
		(void) f;
		(void) A;
		return UNSUPPORTED;
	}

	 // default (non-)implementations follow:

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3,
		typename RIT, typename CIT, typename NIT,
		typename Coords,
		enum Backend implementation = config::default_backend
	>
	RC mxv(
		Vector< IOType, implementation, Coords > &u,
		const Vector< InputType3, implementation, Coords > &mask,
		const Matrix< InputType2, implementation, RIT, CIT, NIT > &A,
		const Vector< InputType1, implementation, Coords > &v,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		typename std::enable_if<
			grb::is_semiring< Ring >::value,
		void >::type * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::mxv (output-masked)\n";
#endif
#ifndef NDEBUG
		const bool backend_does_not_support_output_masked_mxv = false;
		assert( backend_does_not_support_output_masked_mxv );
#endif
		(void) u;
		(void) mask;
		(void) A;
		(void) v;
		(void) ring;
		return UNSUPPORTED;
	}

	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend implementation = config::default_backend
	>
	RC mxv(
		Vector< IOType, implementation, Coords > &u,
		const Matrix< InputType2, implementation, RIT, CIT, NIT > &A,
		const Vector< InputType1, implementation, Coords > &v,
		const Ring &ring,
		typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::mxv\n";
#endif
#ifndef NDEBUG
		const bool backend_does_not_support_mxv = false;
		assert( backend_does_not_support_mxv );
#endif
		(void) u;
		(void) A;
		(void) v;
		(void) ring;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3,
		typename Coords, typename RIT, typename CIT, typename NIT,
		enum Backend implementation = config::default_backend
	>
	RC vxm(
		Vector< IOType, implementation, Coords > &u,
		const Vector< InputType3, implementation, Coords > &mask,
		const Vector< InputType1, implementation, Coords > &v,
		const Matrix< InputType2, implementation, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::vxm (output-masked)\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_output_masked_vxm = false;
		assert( selected_backend_does_not_support_output_masked_vxm );
#endif
		(void) u;
		(void) mask;
		(void) v;
		(void) A;
		(void) ring;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename Coords, typename RIT, typename CIT, typename NIT,
		enum Backend implementation = config::default_backend
	>
	RC vxm(
		Vector< IOType, implementation, Coords > &u,
		const Vector< InputType1, implementation, Coords > &v,
		const Matrix< InputType2, implementation, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::vxm\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_vxm = false;
		assert( selected_backend_does_not_support_vxm );
#endif
		(void) u;
		(void) v;
		(void) A;
		(void) ring;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend backend
	>
	RC vxm(
		Vector< IOType, backend, Coords > &u,
		const Vector< InputType3, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &v,
		const Vector< InputType4, backend, Coords > &v_mask,
		const Matrix< InputType2, backend, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement vxm (doubly-masked)\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_doubly_masked_vxm = false;
		assert( selected_backed_does_not_support_doubly_masked_vxm );
#endif
		(void) u;
		(void) mask;
		(void) v;
		(void) v_mask;
		(void) A;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend backend
	>
	RC mxv(
		Vector< IOType, backend, Coords > &u,
		const Vector< InputType3, backend, Coords > &mask,
		const Matrix< InputType2, backend, RIT, CIT, NIT > &A,
		const Vector< InputType1, backend, Coords > &v,
		const Vector< InputType4, backend, Coords > &v_mask,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!std::is_same< InputType2,
		void >::value, void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement mxv (doubly-masked)\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_doubly_masked_mxv = false;
		assert( selected_backed_does_not_support_doubly_masked_mxv );
#endif
		(void) u;
		(void) mask;
		(void) A;
		(void) v;
		(void) v_mask;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend backend
	>
	RC mxv(
		Vector< IOType, backend, Coords > &u,
		const Vector< InputType3, backend, Coords > &mask,
		const Matrix< InputType2, backend, RIT, NIT, CIT > &A,
		const Vector< InputType1, backend, Coords > &v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement "
			<< "singly-masked monoid-op mxv\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_masked_monop_mxv = false;
		assert( selected_backed_does_not_support_masked_monop_mxv );
#endif
		(void) u;
		(void) mask;
		(void) A;
		(void) v;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend backend
	>
	RC vxm(
		Vector< IOType, backend, Coords > &u,
		const Vector< InputType1, backend, Coords > &v,
		const Matrix< InputType2, backend, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement vxm "
			<< "(unmasked, monoid-op version )\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_monop_vxm = false;
		assert( selected_backed_does_not_support_monop_vxm );
#endif
		(void) u;
		(void) v;
		(void) A;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename InputType3,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend implementation
	>
	RC vxm(
		Vector< IOType, implementation, Coords > &u,
		const Vector< InputType3, implementation, Coords > &mask,
		const Vector< InputType1, implementation, Coords > &v,
		const Matrix< InputType2, implementation, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::vxm (output-masked)\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_masked_monop_vxm = false;
		assert( selected_backed_does_not_support_masked_monop_vxm );
#endif
		(void) u;
		(void) mask;
		(void) v;
		(void) A;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename Coords, typename RIT, typename CIT, typename NIT,
		Backend backend
	>
	RC mxv(
		Vector< IOType, backend, Coords > &u,
		const Matrix< InputType2, backend, RIT, CIT, NIT > &A,
		const Vector< InputType1, backend, Coords > &v,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::mxv (unmasked)\n";
#endif
#ifndef NDEBUG
		const bool selected_backed_does_not_support_monop_mxv = false;
		assert( selected_backed_does_not_support_monop_mxv );
#endif
		(void) u;
		(void) A;
		(void) v;
		(void) add;
		(void) mul;
		return UNSUPPORTED;
	}

	/** @} */

} // namespace grb

#endif // end _H_GRB_BLAS2_BASE

