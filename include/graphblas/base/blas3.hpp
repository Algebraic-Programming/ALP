
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
 * Defines the ALP/GraphBLAS level-3 API
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_BLAS3_BASE
#define _H_GRB_BLAS3_BASE

#include <graphblas/backends.hpp>
#include <graphblas/phase.hpp>

#include "matrix.hpp"
#include "vector.hpp"


namespace grb {

	/**
	 * \defgroup BLAS3 Level-3 Primitives
	 * \ingroup GraphBLAS
	 *
	 * A collection of functions that allow GraphBLAS semirings to work on
	 * one or more two-dimensional sparse containers (i.e, sparse matrices).
	 *
	 * @{
	 */

	/**
	 * Unmasked and in-place sparse matrix--sparse matrix multiplication (SpMSpM),
	 * \f$ C += A+B \f$.
	 *
	 * @tparam descr      The descriptors under which to perform the computation.
	 *                    Optional; default is #grb::descriptors::no_operation.
	 * @tparam OutputType The type of elements in the output matrix.
	 * @tparam InputType1 The type of elements in the left-hand side input
	 *                    matrix.
	 * @tparam InputType2 The type of elements in the right-hand side input
	 *                    matrix.
	 * @tparam Semiring   The semiring under which to perform the
	 *                    multiplication.
	 *
	 * @param[in,out] C The matrix into which the multiplication \f$ AB \f$ is
	 *                  accumulated.
	 * @param[in]   A   The left-hand side input matrix \f$ A \f$.
	 * @param[in]   B   The left-hand side input matrix \f$ B \f$.
	 *
	 * @param[in] ring  The semiring under which the computation should
	 *                  proceed.
	 * @param[in] phase The #grb::Phase the primitive should be executed with. This
	 *                  argument is optional; its default is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  If the computation completed as intended.
	 * @return #grb::FAILED   If the capacity of \a C was insufficient to store the
	 *                        output of multiplying \a A and \a B. If this code is
	 *                        returned, \a C on output appears cleared.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE and an out-of-error
	 *                        condition arose while resizing \a C.
	 *
	 * \note This specification does not account for #grb::TRY as that phase is
	 *       still experimental. See its documentation for details.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename CIT1, typename RIT1, typename NIT1,
		typename CIT2, typename RIT2, typename NIT2,
		typename CIT3, typename RIT3, typename NIT3,
		class Semiring,
		Backend backend
	>
	RC mxm(
		Matrix< OutputType, backend, CIT1, RIT1, NIT1 > &C,
		const Matrix< InputType1, backend, CIT2, RIT2, NIT2 > &A,
		const Matrix< InputType2, backend, CIT3, RIT3, NIT3 > &B,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::mxm "
			<< "(semiring version)\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_mxm = false;
		assert( selected_backend_does_not_support_mxm );
#endif
		(void) C;
		(void) A;
		(void) B;
		(void) ring;
		(void) phase;
		// this is the generic stub implementation
		return UNSUPPORTED;
	}

	/**
	 * The #grb::zip merges three vectors into a matrix.
	 *
	 * Interprets three input vectors \a x, \a y, and \a z as a series of row
	 * coordinates, column coordinates, and nonzeroes, respectively. The
	 * thus-defined nonzeroes of a matrix are then stored in a given output
	 * matrix \a A.
	 *
	 * The vectors \a x, \a y, and \a z must have equal length, as well as the same
	 * number of nonzeroes. If the vectors are sparse, all vectors must have the
	 * same sparsity structure.
	 *
	 * \note A variant of this function only takes \a x and \a y, and has that the
	 *       output matrix \a A has <tt>void</tt> element types.
	 *
	 * If this function does not return #grb::SUCCESS, the output \ a A will have
	 * no contents on function exit.
	 *
	 * The matrix \a A must have been pre-allocated to store the nonzero pattern
	 * that the three given vectors \a x, \a y, and \a z encode, or otherwise this
	 * function returns #grb::ILLEGAL.
	 *
	 * \note To ensure that the capacity of \a A is sufficient, a succesful call to
	 *       #grb::resize with #grb::nnz of \a x suffices. Alternatively, and with
	 *       the same effect, a succesful call to this function with \a phase equal
	 *       to #grb::RESIZE instead of #grb::SUCCESS suffices also.
	 *
	 * @param[out]  A   The output matrix.
	 * @param[in]   x   A vector of row indices.
	 * @param[in]   y   A vector of column indices.
	 * @param[in]   z   A vector of nonzero values.
	 * @param[in] phase The #grb::Phase in which the primitive is to proceed.
	 *                  Optional; the default is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  If \a A was constructed successfully.
	 * @return #grb::MISMATCH If \a y or \a z does not match the size of \a x.
	 * @return #grb::ILLEGAL  If \a y or \a z do not have the same number of
	 *                        nonzeroes as \a x.
	 * @return #grb::ILLEGAL  If \a y or \a z has a different sparsity pattern from
	 *                        \a x.
	 * @return #grb::FAILED   If the capacity of \a A was insufficient to store the
	 *                        given sparsity pattern and \a phase is #grb::EXECUTE.
	 * @return #grb::OUTOFMEM If the \a phase is #grb::RESIZE and \a A could not be
	 *                        resized to have sufficient capacity to complete this
	 *                        function due to out-of-memory conditions.
	 *
	 * \parblock
	 * \par Descriptors
	 *
	 * None allowed.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename InputType3, typename RIT, typename CIT, typename NIT,
		Backend backend, typename Coords
	>
	RC zip(
		Matrix< OutputType, backend, RIT, CIT, NIT > &A,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Vector< InputType3, backend, Coords > &z,
		const Phase &phase = EXECUTE
	) {
		(void) x;
		(void) y;
		(void) z;
		(void) phase;
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::zip (vectors into "
			<< "matrices, non-void)\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_zip_from_vectors_to_matrix
			= false;
		assert( selected_backend_does_not_support_zip_from_vectors_to_matrix );
#endif
		const RC ret = grb::clear( A );
		return ret == SUCCESS ? UNSUPPORTED : ret;
	}

	/**
	 * Merges two vectors into a <tt>void</tt> matrix.
	 *
	 * This is a specialisation of #grb::zip for pattern matrices. The two input
	 * vectors \a x and \a y represent coordinates of nonzeroes to be stored in
	 * \a A.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType1, typename InputType2, typename InputType3,
		typename RIT, typename CIT, typename NIT,
		Backend backend, typename Coords
	>
	RC zip(
		Matrix< void, backend, RIT, CIT, NIT > &A,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		(void) x;
		(void) y;
		(void) phase;
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::zip (vectors into "
			<< "matrices, void)\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_zip_from_vectors_to_void_matrix
			= false;
		assert( selected_backend_does_not_support_zip_from_vectors_to_void_matrix );
#endif
		const RC ret = grb::clear( A );
		return ret == SUCCESS ? UNSUPPORTED : ret;
	}

	/**
	 * Computes \f$ C = A \odot B \f$, out of place, monoid variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * matrix, \f$ C = A \odot B \f$, using the given monoid's operator. The input
	 * and output matrices must be of same dimension.
	 *
	 * Any old entries of \a C will be removed after a successful call to this
	 * primitive; that is, this is an out-of-place primitive.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a C
	 * will match that of the union of \a A and \a B. An implementing backend may
	 * skip processing rows \a i and columns \a j that are not in the union of the
	 * nonzero structure of \a A and \a B.
	 *
	 * \note When applying element-wise operators on sparse matrices using
	 *       semirings, there is a difference between interpreting missing
	 *       values as an annihilating identity or as a neutral identity--
	 *       intuitively, such identities are known as `zero' or `one',
	 *       respectively. As a consequence, this functionality is provided by
	 *       #grb::eWiseApply depending on whether a monoid or operator is
	 *       provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand matrix.
	 * @tparam InputType2 The value type of the right-hand matrix.
	 * @tparam OutputType The value type of the ouput matrix.
	 *
	 * @param[out]  C    The output matrix.
	 * @param[in]   A    The left-hand input matrix.
	 * @param[in]   B    The right-hand input matrix.
	 * @param[in] monoid The monoid structure containing \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output matrix
	 *                        \a z is cleared, and the call to this function has
	 *                        no further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 *
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3,
		Backend backend
	>
	RC eWiseApply(
		Matrix< OutputType, backend, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, backend, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, backend, RIT3, CIT3, NIT3 > &B,
		const Monoid &monoid,
		const Phase phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) phase;
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::eWiseApply\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_ewiseapply = false;
		assert( selected_backend_does_not_support_ewiseapply );
#endif
		const RC ret = grb::clear( A );
		return ret == SUCCESS ? UNSUPPORTED : ret;
	}

	/**
	 * Computes \f$ C = A \odot B \f$, out of place, operator variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * matrix, \f$ C = A \odot B \f$, using the given operator. The input and
	 * output matrices must be of same dimension.
	 *
	 * Any old entries of \a C will be removed after a successful call to this
	 * primitive; that is, this primitive is out-of-place.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a C
	 * will match that of the intersection of \a A and \a B. An implementing
	 * backend may skip processing rows \a i and columns \a j that are not in the
	 * intersection of the nonzero structure of \a A and \a B.
	 *
	 * \note When applying element-wise operators on sparse matrices using
	 *       semirings, there is a difference between interpreting missing
	 *       values as an annihilating identity or as a neutral identity--
	 *       intuitively, such identities are known as `zero' or `one',
	 *       respectively. As a consequence, this functionality is provided by
	 *       #grb::eWiseApply depending on whether a monoid or operator is
	 *       provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Operator   The operator to use.
	 * @tparam InputType1 The value type of the left-hand matrix.
	 * @tparam InputType2 The value type of the right-hand matrix.
	 * @tparam OutputType The value type of the ouput matrix.
	 *
	 * @param[out]  C      The output matrix.
	 * @param[in]   A      The left-hand input matrix.
	 * @param[in]   B      The right-hand input matrix.
	 * @param[in]   op     The operator.
	 * @param[in]   phase  The #grb::Phase the call should execute. Optional; the
	 *                     default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        be as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output
	 *                        matrix \a z is cleared, and the call to this function
	 *                        has no further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 *
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = grb::descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2,
		typename RIT3, typename CIT3, typename NIT3,
		Backend backend
	>
	RC eWiseApply(
		Matrix< OutputType, backend, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, backend, RIT2, CIT2, NIT2 > &A,
		const Matrix< InputType2, backend, RIT3, CIT3, NIT3 > &B,
		const Operator &op,
		const Phase phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
		(void) C;
		(void) A;
		(void) B;
		(void) phase;
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::eWiseApply\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_ewiseapply	= false;
		assert( selected_backend_does_not_support_ewiseapply );
#endif
		const RC ret = grb::clear( A );
		return ret == SUCCESS ? UNSUPPORTED : ret;
	}

	/**
	 * Selects elements from a matrix based on a given selection boolean operator.
	 *
	 * This function template is used to select elements from a given input matrix
	 * based on a provided selection operator. The selected elements are then stored
	 * in a different output matrix, making this an out-of-place operation. The
	 * output matrix is cleared before the selection.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a B
	 * will match the one of \a A without the elements that were not matched by
	 * the selection operator. All the elements of \a B will result $true$ when
	 * applied to the selection operator.
	 *
	 * Any old entries of \a B will be removed after a successful call to this
	 * primitive; that is, this primitive is out-of-place.
	 *
	 * \note See #grb::selectLambda for a variant of this function that uses a
	 *       lambda function instead of a selection operator.
	 *
	 * @tparam descr              The descriptor to be used. Optional; the default is
	 *                            #grb::descriptors::no_operation.
	 * @tparam SelectionOperator  The selection operator type, must be of a type
	 *                            equivalent to `bool(RIT*, CIT*, T*)` with:
	 *                            - RIT: The row index type of the input matrix,
	 *                                   or convertible to it.
	 *                            - CIT: The column index type of the input matrix,
	 *                                   or convertible to it.
	 *                            - T:   The value type of the input matrix, or
	 *                                   convertible to it.
	 * @tparam Tin                The value type of the input matrix.
	 * @tparam RITin              The row index type of the input matrix.
	 * @tparam CITin              The column index type of the input matrix.
	 * @tparam NITin              The nonzero index type of the input matrix.
	 * @tparam Tout	              The value type of the output matrix.
	 * @tparam RITout             The row index type of the output matrix.
	 * @tparam CITout             The column index type of the output matrix.
	 * @tparam NITout             The nonzero index type of the output matrix.
	 * @tparam backend            The backend to use for the operation.
	 *
	 * @param[out] B       The output matrix. Will be cleared before the selection.
	 * @param[in]  A       The input matrix.
	 * @param[in]  op      The selection boolean operator. Pre-defined selection
	 *                     operators can be found in the namespace
	 *                     #grb::operators::select.
	 * @param[in]  phase   The #grb::Phase the call should execute. Optional; the
	 *                     default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a A and \a B do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        be as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a B was insufficient. The output
	 *                        matrix \a B is cleared, and the call to this function
	 *                        has no further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a B is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 *
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class SelectionOperator,
		typename Tin,
		typename RITin, typename CITin, typename NITin,
		typename Tout,
		typename RITout, typename CITout, typename NITout,
		Backend backend
	>
	RC select(
		Matrix< Tout, backend, RITout, CITout, NITout > &B,
		const Matrix< Tin, backend, RITin, CITin, NITin > &A,
		const SelectionOperator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!is_object< Tin >::value &&
			!is_object< Tout >::value &&
			is_matrix_selection_operator< SelectionOperator >::value
		>::type * const = nullptr
	) {
		(void) descr;
		(void) B;
		(void) A;
		(void) op;
		(void) phase;
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::select\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_select = false;
		assert( selected_backend_does_not_support_select );
#endif
		return UNSUPPORTED;
	}

	/**
	 * Selects elements from a matrix based on a given selection function.
	 *
	 * This function template is used to select elements from a given input matrix
	 * based on a provided selection function. The selected elements are then stored
	 * in a different output matrix, making this an out-of-place operation. The
	 * output matrix is cleared before the selection.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a B
	 * will match the one of \a A without the elements that were not matched by
	 * the selection function. All the elements of \a B will result $true$ when
	 * applied to the selection function.
	 *
	 * Any old entries of \a B will be removed after a successful call to this
	 * primitive; that is, this primitive is out-of-place.
	 *
	 * \note See #grb::select for a variant of this function that uses pre-defined
	 *       selection operators instead of a lambda function.
	 *
	 * @tparam descr              The descriptor to be used. Optional; the default is
	 *                            #grb::descriptors::no_operation.
	 * @tparam SelectionLambda    The selection function type, must be of a type
	 *                            equivalent to `bool(RIT*, CIT*, T*)` with:
	 *                            - RIT: The row index type of the input matrix,
	 *                                   or convertible to it.
	 *                            - CIT: The column index type of the input matrix,
	 *                                   or convertible to it.
	 *                            - T:   The value type of the input matrix, or
	 *                                   convertible to it.
	 * @tparam Tin                The value type of the input matrix.
	 * @tparam RITin              The row index type of the input matrix.
	 * @tparam CITin              The column index type of the input matrix.
	 * @tparam NITin              The nonzero index type of the input matrix.
	 * @tparam Tout	              The value type of the output matrix.
	 * @tparam RITout             The row index type of the output matrix.
	 * @tparam CITout             The column index type of the output matrix.
	 * @tparam NITout             The nonzero index type of the output matrix.
	 * @tparam backend            The backend to use for the operation.
	 *
	 * @param[out] B       The output matrix. Will be cleared before the selection.
	 * @param[in]  A       The input matrix.
	 * @param[in]  lambda  The selection function returning a boolean value.
	 * @param[in]  phase   The #grb::Phase the call should execute. Optional; the
	 *                     default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a A and \a B do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        be as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a B was insufficient. The output
	 *                        matrix \a B is cleared, and the call to this function
	 *                        has no further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a B is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 *
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 *
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class SelectionLambda,
		typename Tin,
		typename RITin, typename CITin, typename NITin,
		typename Tout,
		typename RITout, typename CITout, typename NITout,
		Backend backend
	>
	RC selectLambda(
		Matrix< Tout, backend, RITout, CITout, NITout > &B,
		const Matrix< Tin, backend, RITin, CITin, NITin > &A,
		const SelectionLambda &lambda,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!is_object< Tin >::value &&
			!is_object< Tout >::value
		>::type * const = nullptr
	) {
		(void) descr;
		(void) B;
		(void) A;
		(void) lambda;
		(void) phase;
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement grb::selectLambda\n";
#endif
#ifndef NDEBUG
		const bool selected_backend_does_not_support_selectLambda = false;
		assert( selected_backend_does_not_support_selectLambda );
#endif
		return UNSUPPORTED;
	}

	/**
	 * @}
	 */

} // namespace grb

#endif // end _H_GRB_BLAS3_BASE

