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
		 *        \f$A, B, C\f$, respectively, and \f$At_blk\f$ is a transposed view
		 *        over the \f$A\f$ matrix.
		 *
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
			typename D = double,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >
		>
		RC gemm_like_example(
			size_t m,
			size_t n,
			size_t k,
			Scalar< D > alpha,
			Matrix< D, structures::General, Dense > &A,
			size_t startAr,
			size_t strideAr,
			size_t startAc,
			size_t strideAc,
			Matrix< D, structures::General, Dense > &B,
			size_t startBr,
			size_t strideBr,
			size_t startBc,
			size_t strideBc,
			Scalar< D > beta,
			Matrix< D, structures::General, Dense > &C,
			size_t startCr,
			size_t strideCr,
			size_t startCc,
			size_t strideCc,
			const Ring &ring = Ring()
		) {

			auto At = get_view< view::transpose >( A ); // Transposed view of matrix A
			auto At_blk = get_view(
				At,
				utils::range( startAr, startAr + m, strideAr ),
				utils::range( startAc, startAc + k, strideAc )
			);

			auto B_blk = get_view(
				B,
				utils::range( startBr, startBr + k, strideBr ),
				utils::range( startBc, startBc + n, strideBc )
			);

			auto C_blk = get_view(
				C,
				utils::range( startCr, startCr + m, strideCr ),
				utils::range( startCc, startCc + n, strideCc )
			);

			Matrix< D, structures::General, Dense > C_tmp( m, n );

			RC rc = SUCCESS;

			// C_blk = beta * C_blk
			rc = rc ? rc : foldr( beta, C_blk, ring.getMultiplicativeMonoid() );

			// C_tmp = 0
			rc = rc ? rc : set( C_tmp, Scalar< D >( ring.template getZero< D >() ) );
			// C_tmp += At_blk * B_blk
			rc = rc ? rc : mxm( C_tmp, At_blk, B_blk, ring );

			// C_blk += alpha * C_tmp
			rc = rc ? rc : eWiseMul( C_blk, alpha, C_tmp, ring );

			return rc;
		}

	} // namespace algorithms
} // namespace alp
