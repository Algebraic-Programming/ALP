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
#include <iomanip>

#include <alp.hpp>

#include "../../../tests/utils/print_alp_containers.hpp"

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
		RC cholesky_uptr(
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

			if( rc != SUCCESS ) {
				std::cerr << " set( LL, H ) failed\n";
				return rc;
			}
#ifdef DEBUG
			print_matrix( " -- LL --  " , LL );
#endif

			for( size_t k = 0; k < n ; ++k ) {
#ifdef DEBUG
				std::cout << "============ Iteration " << k << " ============" << std::endl;
#endif

				auto a = get_view( LL, k, utils::range( k, n ) );
#ifdef DEBUG
				print_vector( " -- a --  " , a );
#endif

				// L[ k, k ] = alpha = sqrt( LL[ k, k ] )
				Scalar< D > alpha;
				rc = eWiseLambda(
					[ &alpha, &ring ]( const size_t i, D &val ) {
						if ( i == 0 ) {
							(void)set( alpha, alp::Scalar< D >( std::sqrt( val ) ) );
							val = *alpha;
						}
					},
					a
				);

#ifdef DEBUG
				std::cout << "alpha " << *alpha << std::endl;
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (0) failed\n";
					return rc;
				}
#endif

				auto v = get_view( LL, k, utils::range( k + 1, n ) );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
#endif
				// LL[ k + 1: , k ] = LL[ k + 1: , k ] / alpha
				rc = foldl( v, alpha, divide );

#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (1) failed\n";
					return rc;
				}
#endif

				// LL[ k+1: , k+1: ] -= v*v^T
				auto LLprim = get_view( LL, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vvt = outer( v, ring.getMultiplicativeOperator() );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
				print_matrix( " vvt ", vvt );
#endif

				// this eWiseLambda should be replaced by foldl on matrices
				rc = alp::eWiseLambda(
					[ &vvt, &minus ]( const size_t i, const size_t j, D &val ) {
						internal::foldl(
							val,
							internal::access( vvt, internal::getStorageIndex( vvt, i, j ) ),
							minus
						);
					},
					LLprim
				);
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (2) failed\n";
					return rc;
				}
#endif
			}

			// Finally collect output into L matrix and return
			for( size_t k = 0; k < n ; ++k ) {

				// L[ k: , k ] = LL[ k: , k ]
				auto vL  = get_view( L, k, utils::range( k, n )  );
				auto vLL = get_view( LL, k, utils::range( k, n )  );

				rc = set( vL, vLL );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " set( view, view ) failed\n";
					return rc;
				}
#endif
			}

			assert( rc == SUCCESS );
			return rc;
		}

	} // namespace algorithms
} // namespace alp