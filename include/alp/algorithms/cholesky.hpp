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

#include <cmath>
#include <iostream>

#include <alp.hpp>

#define TEMP_DISABLE

namespace alp {

	namespace algorithms {

		/**
		 * @brief Computes the Cholesky decomposition LL^T = H of a real symmetric
		 *        positive definite (SPD) matrix H where \a L is lower triangular.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] L    output lower triangular matrix
		 * @param[in]  H    input real symmetric positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D = double,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC cholesky_lowtr(
			Matrix< D, structures::UpperTriangular, Dense > &L,
			const Matrix< D, structures::Symmetric, Dense > &H,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {


			RC rc = SUCCESS;

			const size_t n = nrows( H );

			// Out of place specification of the operation
			Matrix< D, structures::Symmetric, Dense > LL( n, n );
			rc = set( LL, H );

			for( size_t k = 0; k < n ; ++k ) {

				auto v = get_view( LL , utils::range( k, n) , k );

				// L[ k, k ] = alpha = sqrt( LL[ k, k ] )
				Scalar< D > alpha;
				rc = eWiseLambda(
					[ &v, &alpha, &ring ]( const size_t i ) {
						if ( i == 0 ) {
#ifdef TEMP_DISABLE
							*alpha = std::sqrt( v[ i ] );
							internal::setInitialized( alpha, true );
#else
							(void)set( alpha, std::sqrt( v[ i ] ) );
#endif
							v[ i ] = *alpha;
						}
					}, v
				);

				// LL[ k + 1: , k ] = LL[ k + 1: , k ] / alpha
				rc = eWiseLambda(
					[ &v, &alpha, &divide ]( const size_t i ) {
						if ( i > 0 ) {
							internal::foldl( v[ i ], *alpha, divide );
						}
					},
					v
				);


				// LL[ k+1: , k+1: ] -= v*v^T
				auto LLprim = get_view( LL, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vvt = outer( v, ring.getMultiplicativeOperator() );
#ifdef TEMP_DISABLE
				std::cout << "warning:  missing foldl( Mat, Mat, minus )";
				std::function< void( typename operators::subtract< D > &, const size_t, const size_t ) > lambda_f_minus =
					[ &LLprim, &vvt, &minus ]( typename operators::subtract< D >::D3 &result, const size_t i, const size_t j ) {
						result = LLprim[ i, j ];
						//result = minus( LLprim[ i, j ], vvt[ i, j ] );
						//internal::apply( result, x[ i ], y[ j ], minus );
					};
#else
				rc = foldl( LLprim, vvt, minus );
#endif

			}

			// Finally collect output into L matrix and return
			for( size_t k = 0; k < n ; ++k ) {

				// L[ k: , k ] = LL[ k: , k ]
				auto vL  = get_view( L  , utils::range( k, n), k  );
				auto vLL = get_view( LL , utils::range( k, n), k  );

				rc = set( vL, vLL );

			}

			return rc;
		}

	} // namespace algorithms
} // namespace alp
