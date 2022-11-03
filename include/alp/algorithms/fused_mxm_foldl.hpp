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

#include <iostream>

#include <alp.hpp>

namespace alp {

	namespace algorithms {

		/**
		 * Performs mxm followed by foldl: C += A*AT.
		 * The purpose of this function is to simulate operation fusion.
		 *
		 * @tparam MatrixC  Type of a symmetric ALP Matrix C
		 * @tparam MatrixA  Type of ALP Matrix A
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring passed to mxm
		 * @tparam Op       Type of the operator passed to foldl
		 *
		 * @param[inout] C  Matrix C
		 * @param[in] A     Matrix A
		 * @param[in] ring  Ring passed to mxm
		 * @param[in] op    Operator passed to foldl
		 *
		 * @return RC       SUCCESS if the execution was correct
		 * \note This does not support complex numbers at the moment.
		 */
		template<
			typename MatrixC,
			typename MatrixA,
			typename D = typename MatrixC::value_type,
			typename Ring, typename Op,
			std::enable_if_t<
				alp::is_matrix< MatrixC >::value &&
				alp::is_matrix< MatrixA >::value &&
				alp::is_semiring< Ring >::value &&
				alp::is_operator< Op >::value &&
				config::default_backend != Backend::dispatch
			> * = nullptr
		>
		RC fused_symm_mxm_foldl(
			MatrixC &C,
			MatrixA &A,
			const Ring &ring = Ring(),
			const Op &op = Op()
		) {

			// Verify that the C is of dimensions nrows(A) x nrows(A)
			const size_t m = ncols( A );
			if( ( nrows( C ) != m ) || ( ncols( C ) != m ) ) {
				return MISMATCH;
			}

			const auto AT = get_view< view::transpose >( A );

			Matrix< D, typename MatrixC::structure, Density::Dense > AAT( m );

			RC rc = SUCCESS;

			// AAT = 0
			rc = rc ? rc : set( AAT, Scalar< D >( ring.template getZero< D >() ) );
			assert( rc == SUCCESS );

			// AAT += A * AT
			rc = rc ? rc : mxm( AAT, AT, A, ring );
			assert( rc == SUCCESS );

			// C += AAT
			rc = rc ? rc : foldl( C, AAT, op );
			assert( rc == SUCCESS );

			return rc;
		}

		/**
		 * Specialization for dispatch backend. Offloads to syrk.
		 * Assumes that A is transposed.
		 */
		template<
			typename MatrixC,
			typename MatrixA,
			typename D = typename MatrixC::value_type,
			typename Ring, typename Op,
			std::enable_if_t<
				alp::is_matrix< MatrixC >::value &&
				alp::is_matrix< MatrixA >::value &&
				alp::is_semiring< Ring >::value &&
				alp::is_operator< Op >::value &&
				config::default_backend == Backend::dispatch
			> * = nullptr
		>
		RC fused_symm_mxm_foldl(
			MatrixC &C,
			MatrixA &A,
			const Ring &ring = Ring(),
			const Op &op = Op()
		) {
			(void) ring;
			(void) op;

			// Verify that the C is of dimensions nrows(A) x nrows(A)
			const size_t k = nrows( A );
			const size_t m = ncols( A );
			if( ( nrows( C ) != m ) || ( ncols( C ) != m ) ) {
				return MISMATCH;
			}

			RC rc = SUCCESS;

#ifdef _ALP_WITH_DISPATCH
			cblas_dsyrk(
				CblasRowMajor, CblasUpper, CblasTrans,
				m,
				k,
				-1,
				internal::getRawPointerToFirstElement( A ),
				internal::getLeadingDimension( A ),
				1,
				internal::getRawPointerToFirstElement( C ),
				internal::getLeadingDimension( C )
			);
#endif

			return rc;
		}

	} // namespace algorithms
} // namespace alp
