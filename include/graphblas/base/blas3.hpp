
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
		typename CIT, typename RIT, typename NIT,
		class Semiring,
		Backend backend
	>
	RC mxm(
		Matrix< OutputType, backend, CIT, RIT, NIT > &C,
		const Matrix< InputType1, backend, CIT, RIT, NIT > &A,
		const Matrix< InputType2, backend, CIT, RIT, NIT > &B,
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
	 * @}
	 */

} // namespace grb

#endif // end _H_GRB_BLAS3_BASE

