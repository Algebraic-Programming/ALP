
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
	 * Unmaked sparse matrix--sparse matrix multiplication (SpMSpM).
	 *
	 * @tparam descr      The descriptors under which to perform the computation.
	 * @tparam OutputType The type of elements in the output matrix.
	 * @tparam InputType1 The type of elements in the left-hand side input
	 *                    matrix.
	 * @tparam InputType2 The type of elements in the right-hand side input
	 *                    matrix.
	 * @tparam Semiring   The semiring under which to perform the
	 *                    multiplication.
	 * @tparam Backend    The backend that should perform the computation.
	 *
	 * @returns SUCCESS If the computation completed as intended.
	 * @returns FAILED  If the call was not not preceded by one to
	 *                  #grb::resize( C, A, B ); \em and the current capacity of
	 *                  \a C was insufficient to store the multiplication of \a A
	 *                  and \a B. The contents of \a C shall be undefined (which
	 *                  is why #FAILED is returned instead of #ILLEGAL-- this
	 *                  error has side effects).
	 *
	 * @param[out] C The output matrix \f$ C = AB \f$ when the function returns
	 *               #SUCCESS.
	 * @param[in]  A The left-hand side input matrix \f$ A \f$.
	 * @param[in]  B The left-hand side input matrix \f$ B \f$.
	 *
	 * @param[in] ring (Optional.) The semiring under which the computation should
	 *                             proceed.
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
	 * Interprets three vectors x, y, and z as a series of row coordinates,
	 * column coordinates, and nonzeroes, respectively, and stores the thus
	 * defined nonzeroes in a given output matrix A.
	 *
	 * If this function does not return SUCCESS, A will have been cleared.
	 *
	 * A must have been pre-allocated to store the nonzero pattern the three
	 * given vectors x, y, and z encode, or ILLEGAL shall be returned.
	 *
	 * \note A call to this function hence must be preceded by a successful
	 *       call to grb::resize( matrix, nnz );
	 *
	 * @param[out] A The output matrix
	 * @param[in]  x A vector of row indices.
	 * @param[in]  y A vector of column indices.
	 * @param[in]  z A vector of nonzero values.
	 *
	 * If x, y, and z are sparse, they must have the exact same sparsity
	 * structure.
	 *
	 * \par Descriptors
	 *
	 * None allowed.
	 *
	 * @returns SUCCESS  If A was constructed successfully.
	 * @returns MISMATCH If y or z does not match the size of x.
	 * @returns ILLEGAL  If y or z do not have the same number of nonzeroes
	 *                   as x.
	 * @returns ILLEGAL  If y or z has a different sparsity pattern from x.
	 * @returns ILLEGAL  If the capacity of A was insufficient to store the
	 *                   given sparsity pattern.
	 *
	 * @see grb::resize
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
	 * Specialisation of grb::zip for void output matrices.
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

