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
#define DEBUG

namespace alp {

	namespace algorithms {
		template< typename V >
		void print_vector( std::string name, const V &v) {

			if( ! alp::internal::getInitialized( v ) ) {
				std::cout << "Vector " << name << " uninitialized.\n";
				return;
			}

			std::cout << name << ":" << std::endl;
			std::cout << "[\t";
			for( size_t i = 0; i < alp::getLength( v ); ++i ) {
					std::cout << std::setprecision(5) << v[ i ] << "\t";
				}
			std::cout << "]" << std::endl;
		}

		template< typename M >
		void print_matrix( std::string name, const M & A) {

			if( ! alp::internal::getInitialized( A ) ) {
				std::cout << "Matrix " << name << " uninitialized.\n";
				return;
			}

			std::cout << name << ":" << std::endl;
			for( size_t row = 0; row < alp::nrows( A ); ++row ) {
				std::cout << "[\t";
				for( size_t col = 0; col < alp::ncols( A ); ++col ) {
					if ( col < row ) {
						std::cout << 0 << "\t";
					} else {
						auto pos  = internal::getStorageIndex( A, row, col );
						std::cout << std::setprecision(5) << internal::access(A, pos ) << "\t";
					}
				}
				std::cout << "]" << std::endl;
			}
		}


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
#ifdef DEBUG
			if( rc != SUCCESS ) {
				std::cerr << " set( LL, H ) failed\n";
				return rc;
			}
#endif
			print_matrix( " -- LL --  " , LL );

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
#ifdef TEMP_DISABLE
							internal::setInitialized( alpha, true );
							*alpha = std::sqrt( val );
#else
							(void)set( alpha, std::sqrt( val ) );
#endif
							val = *alpha;
						}
					}, a
				);

				std::cout << "alpha " << *alpha << std::endl;
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (0) failed\n";
					return rc;
				}
#endif

				auto v = get_view( LL, k, utils::range( k + 1, n ) );
				print_vector( " -- v --  " , v );
				// LL[ k + 1: , k ] = LL[ k + 1: , k ] / alpha
				rc = eWiseLambda(
					[ &alpha, &divide ]( const size_t i, D &val ) {
						(void)i;
						internal::foldl( val, *alpha, divide );
					},
					v
				);

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
#ifdef TEMP_DISABLE
				rc = alp::eWiseLambda(
					[ &vvt, &minus, &divide ]( const size_t i, const size_t j, D &val ) {
						internal::foldl(
							val,
							internal::access( vvt, internal::getStorageIndex( vvt, i, j ) ),
							minus
						);
						std::cout << "lambda " << i << " " << j << " " << val << " " << internal::access( vvt, internal::getStorageIndex( vvt, i, j ) ) << std::endl;
					},
					LLprim
				);
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (2) failed\n";
					return rc;
				}
#endif
#else
				rc = foldl( LLprim, vvt, minus );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " foldl( view, outer, minus ) failed\n";
					return rc;
				}
#endif
#endif

				print_matrix( " -- LL --  " , LL );
			}

			// Finally collect output into L matrix and return
			for( size_t k = 0; k < n ; ++k ) {

				// L[ k: , k ] = LL[ k: , k ]
				auto vL  = get_view( L  , k, utils::range( k, n )  );


				auto vLL = get_view( LL , k, utils::range( k, n )  );

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
