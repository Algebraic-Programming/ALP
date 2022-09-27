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
		 * @brief gemm_like example where a sub-matrix
		 *        \f$C_blk = \alpha \cdot At_blk \cdot B_blk + \beta \cdot C_blk\f$,
		 *        where \f$At_blk, B_blk, C_blk\f$ are sub-matrices (optionally at
		 *        a stride both row- and column-wise) of matrices
		 *        \f$A, B, C\f$, respectively, and \f$At_blk\f$ and \f$B_blk$ may be
		 *        transposed views over the \f$A\f$ and \f$B\f$ sub-matrices
		 *        depending on parameters \f$transposeA\f$ and \f$transposeB\f$, respectively.
		 *
		 * @tparam transposeA  Whether to transpose A
		 * @tparam transposeB  Whether to transpose B
		 * @tparam D         Data element type
		 * @tparam Ring      Type of the semiring used in computation
		 * @param m          Number of rows of matrices \a C_blk and \a At_blk
		 * @param n          Number of columns of matrices \a C_blk and \a B_blk
		 * @param k          Number of rows of matrix \a B_blk and columns of \a A_blk
		 * @param alpha      Alpha scalar parameter
		 * @param A          reference to matrix A
		 * @param startAr    Row offset of \a At_blk within \a A
		 * @param startAc    Column offset of \a At_blk within \a A
		 * @param B          reference to matrix B
		 * @param startBr    Row offset of \a B_blk within \a B
		 * @param startBc    Column offset of \a B_blk within \a B
		 * @param beta       Beta scalar parameter
		 * @param C          reference to matrix C
		 * @param startCr    Row offset of \a C_blk within \a C
		 * @param startCc    Column offset of \a C_blk within \a C
		 * @param ring       The semiring used for performing operations
		 * @return RC        SUCCESS if the execution was correct
		 */
		template<
			bool transposeA = false,
			bool transposeB = false,
			typename D = double,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >
		>
		RC gemm_like_example(
			const size_t m,
			const size_t n,
			const size_t k,
			const Scalar< D > &alpha,
			Matrix< D, structures::General, Dense > &A,
			const size_t startAr,
			const size_t strideAr,
			const size_t startAc,
			const size_t strideAc,
			Matrix< D, structures::General, Dense > &B,
			const size_t startBr,
			const size_t strideBr,
			const size_t startBc,
			const size_t strideBc,
			const Scalar< D > &beta,
			Matrix< D, structures::General, Dense > &C,
			const size_t startCr,
			const size_t strideCr,
			const size_t startCc,
			const size_t strideCc,
			const Ring &ring = Ring()
		) {

			// Ensure the compatibility of parameters
			const size_t endCr = startCr + m * strideCr;
			const size_t endCc = startCc + n * strideCc;
			const size_t endAr = transposeA ? startAr + k * strideAr : startAr + m * strideAr;
			const size_t endAc = transposeA ? startAc + m * strideAc : startAc + k * strideAc;
			const size_t endBr = transposeB ? startBr + n * strideBr : startBr + k * strideBr;
			const size_t endBc = transposeB ? startBc + k * strideBc : startBc + n * strideBc;

			if(
				( endAr > nrows( A ) ) || ( endAc > ncols( A ) ) ||
				( endBr > nrows( B ) ) || ( endBc > ncols( B ) ) ||
				( endCr > nrows( C ) ) || ( endCc > ncols( C ) )
			) {
				return MISMATCH;
			}

			const size_t mA = transposeA ? k : m;
			const size_t kA = transposeA ? m : k;
			auto A_blk_orig = get_view(
				A,
				utils::range( startAr, startAr + mA * strideAr, strideAr ),
				utils::range( startAc, startAc + kA * strideAc, strideAc )
			);

			auto A_blk = get_view< transposeA ? view::transpose : view::original >( A_blk_orig );

			const size_t kB = transposeB ? n : k;
			const size_t nB = transposeB ? k : n;
			auto B_blk_orig = get_view(
				B,
				utils::range( startBr, startBr + kB * strideBr, strideBr ),
				utils::range( startBc, startBc + nB * strideBc, strideBc )
			);

			auto B_blk = get_view< transposeB ? view::transpose : view::original >( B_blk_orig );

			auto C_blk = get_view(
				C,
				utils::range( startCr, startCr + m * strideCr, strideCr ),
				utils::range( startCc, startCc + n * strideCc, strideCc )
			);

			Matrix< D, structures::General, Dense > C_tmp( m, n );

			RC rc = SUCCESS;

			// C_blk = beta * C_blk
			rc = rc ? rc : foldr( beta, C_blk, ring.getMultiplicativeMonoid() );
			assert( rc == SUCCESS );

			// C_tmp = 0
			rc = rc ? rc : set( C_tmp, Scalar< D >( ring.template getZero< D >() ) );
			assert( rc == SUCCESS );
			// C_tmp += At_blk * B_blk
			rc = rc ? rc : mxm( C_tmp, A_blk, B_blk, ring );
			assert( rc == SUCCESS );

			// C_blk += alpha * C_tmp
			rc = rc ? rc : eWiseMul( C_blk, alpha, C_tmp, ring );
			assert( rc == SUCCESS );

			return rc;
		}

	} // namespace algorithms
} // namespace alp
