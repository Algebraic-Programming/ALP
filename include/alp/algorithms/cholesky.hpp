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

namespace alp {

	namespace algorithms {

		/**
		 * @brief Computes the Cholesky decomposition LL^H = H of a real symmetric
		 *        positive definite (SPD) matrix H where \a L is lower triangular.
		 *        L^H  is equvalent to cojugate(transpose(L))
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] L    output lower triangular matrix
		 * @param[in]  H    input real symmetric (or complex hermitian) positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC cholesky_lowtr(
			Matrix< D, structures::LowerTriangular, Dense > &L,
			const Matrix< D, structures::HermitianPositiveDefinite, Dense > &H,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {


			RC rc = SUCCESS;

			const size_t n = nrows( H );

			// Out of place specification of the operation
			Matrix< D, structures::HermitianPositiveDefinite, Dense > LL( n, n );
			rc = set( LL, H );

			for( size_t k = 0; k < n ; ++k ) {

				const size_t m = n - k - 1;

				auto v = get_view( LL , utils::range( k, n) , k );

				// L[ k, k ] = alpha = sqrt( LL[ k, k ] )
				Scalar< D > alpha;
				rc = eWiseLambda(
							[ &v, &alpha, &ring ]( const size_t i ) {
								if ( i == 0 ) {
									v[ i ] = alpha = std::sqrt( v[ i ] );
								}
							},
							v );

				// LL[ k + 1: , k ] = LL[ k + 1: , k ] / alpha
				rc = eWiseLambda(
							[ &v, &alpha, &divide ]( const size_t i ) {
								if ( i > 0 ) {
									foldl( v[ i ], alpha, divide );
								}
							},
							v );


				// LL[ k+1: , k+1: ] -= v*v^H
				auto LLprim = get_view( LL, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vvt = outer( v, ring.getMultiplicativeOperator() );
				rc = foldl( LLprim, vvt, minus );

			}

			// Finally collect output into L matrix and return
			for( size_t k = 0; k < n ; ++k ) {

				// L[ k: , k ] = LL[ k: , k ]
				auto vL  = get_view( L  , utils::range( k, n) , k );
				auto vLL = get_view( LL , utils::range( k, n) , k );

				rc = set( vL, vLL );

			}

			return rc;
		}

		/**
		 * @brief Computes the blocked version Cholesky decomposition LL^H = H of a real symmetric
		 *        or complex hermitian positive definite (SPD) matrix H where \a L is lower triangular.
		 *        L^H  is equvalent to cojugate(transpose(L))
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] L    output lower triangular matrix
		 * @param[in]  H    input real symmetric (or complex hermitian) positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */		
		template<
			typename D,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC cholesky_lowtr_blk(
			Matrix< D, structures::LowerTriangular, Dense > &L,
			const Matrix< D, structures::HermitianPositiveDefinite, Dense > &H,
			const size_t & bs,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {

			RC rc = SUCCESS;

			const size_t n = nrows( L );

			Matrix< D, structures::HermitianPositiveDefinite, Dense > LL( n, n );
			rc = set( LL, H );

			//nb: number of blocks of (max) size bz 
			size_t nb = n / bs ;
			if( n % bs != 0 ){
				nb = nb + 1 ;
			}

			for( size_t i = 0; i < nb ; ++i ) {
				//A11=L[i*bs:(i+1)*bs,i*bs:(i+1)*bs]
				auto A11 = get_view( LL,
									 utils::range( i * bs, std::max( ( i+1 ) * bs, n ) ),
									 utils::range( i * bs, std::max( ( i+1 ) * bs, n ) )
									 );
				//A21=L[(i+1)*bs:,i*bs:(i+1)*bs]
				auto A21 = get_view( LL,
									 utils::range( i * bs, n ),
									 utils::range( i * bs, std::max( ( i+1 ) * bs, n ) )
									 );
				//A22=L[(i+1)*bs:,(i+1)*bs:]
				auto A22 = get_view( LL,
									 utils::range( ( i + 1 ) * bs, n ),
									 utils::range( ( i + 1 ) * bs, n )
									 );

				//A11=cholesky(A11)
				Matrix< D, structures::LowerTriangular, Dense > tmpM( std::min( bs, n - i * bs ),
																	  std::min( bs, n - i * bs )   );
				rc = cholesky_lowtr( tmpM, A11 );
				rc = set( A11, tmpM );

				//A21=TRSM(A11,conjugate(A21).T)
				auto A21ct = get_view<view::conjugate_transpose>( A21 );
				rc = trsm( A11, A21ct );
				
				//A22=A22-A21.dot(conjugate(A21).T)
				auto A21A21H = kronecker( A21 );
				rc = foldl( A22, A21A21H, minus );
				
			}

			for( size_t k = 0; k < n ; ++k ) {

				// L[ k: , k ] = LL[ k: , k ]
				auto vL  = get_view( L  , utils::range( k, n) , k );
				auto vLL = get_view( LL , utils::range( k, n) , k );
				
				rc = set( vL, vLL );

			}

			return rc;
		}
					


	} // namespace algorithms
} // namespace alp
