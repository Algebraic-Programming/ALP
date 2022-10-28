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
#include <alp/algorithms/forwardsubstitution.hpp>
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
			typename ViewL,
			typename ImfRL,
			typename ImfCL,
			typename ViewH,
			typename ImfRH,
			typename ImfCH,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC cholesky_uptr(
			Matrix< D, structures::UpperTriangular, Dense, ViewL, ImfRL, ImfCL > &L,
			const Matrix< D, structures::Symmetric, Dense, ViewH, ImfRH, ImfCH > &H,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const size_t n = nrows( H );

			// Out of place specification of the operation
			Matrix< D, structures::Symmetric, Dense > LL( n, n );
			rc = rc ? rc : set( LL, H );

#ifdef DEBUG
			if( rc != SUCCESS ) {
				std::cerr << " set( LL, H ) failed\n";
				return rc;
			}
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
				rc = rc ? rc : eWiseLambda(
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
				rc = rc ? rc : foldl( v, alpha, divide );

#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (1) failed\n";
					return rc;
				}
#endif

				// LL[ k+1: , k+1: ] -= v*v^T
				auto Lprim = get_view( LL, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vvt = outer( v, ring.getMultiplicativeOperator() );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
				print_matrix( " vvt ", vvt );
#endif
				rc = rc ? rc : foldl( Lprim, vvt, minus );
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
		 * @param[out] L    output upper triangular matrix
		 * @param[in]  H    input real symmetric (or complex hermitian) positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D,
			typename ViewL,
			typename ImfRL,
			typename ImfCL,
			typename ViewH,
			typename ImfRH,
			typename ImfCH,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC cholesky_uptr_blk(
			Matrix< D, structures::UpperTriangular, Dense, ViewL, ImfRL, ImfCL > &L,
			const Matrix< D, structures::Symmetric, Dense, ViewH, ImfRH, ImfCH > &H,
			const size_t &bs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			(void) divide;
			const Scalar< D > zero( ring.template getZero< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( L );

			Matrix< D, structures::Symmetric, Dense > LL( n, n );
			rc = rc ? rc : set( LL, H );
			if( rc != SUCCESS ) {
				std::cout << "set failed\n";
				return rc;
			}

			//nb: number of blocks of (max) size bz
			size_t nb = n / bs ;
			if( n % bs != 0 ){
				nb = nb + 1 ;
			}


			for( size_t i = 0; i < nb ; ++i ) {
				size_t a = std::min( i * bs, n );
				size_t b = std::min( ( i + 1 ) * bs, n );
				size_t c = n;

				auto range1 = utils::range( a, b );
				auto range2 = utils::range( b, c );

				//A11=L[i*bs:(i+1)*bs,i*bs:(i+1)*bs]
				auto A11 = get_view( LL, range1, range1 );

				//A21=L[(i+1)*bs:,i*bs:(i+1)*bs]
				// for complex we should conjugate A21
				auto A21 = get_view< structures::General >( LL, range1, range2 );

				//A11=cholesky(A11)
				auto A11_out = get_view( L, range1, range1 );

				rc = rc ? rc : cholesky_uptr( A11_out, A11, ring );
				if( rc != SUCCESS ) {
					std::cout << "cholesky_uptr failed\n";
					return rc;
				}

				auto A21_out = get_view< structures::General >(	L, range1, range2 );
				auto A11_out_T = get_view< alp::view::transpose >( A11_out );

				rc = rc ? rc : algorithms::forwardsubstitution(
					A11_out_T,
					A21_out,
					A21,
					ring
				);
				if( rc != SUCCESS ) {
					std::cout << "Forwardsubstitution failed\n";
					return rc;
				}

				Matrix< D, structures::Symmetric, Dense > Reflector( ncols(A21_out), ncols(A21_out) );
				rc = rc ? rc : set( Reflector, zero );
				rc = rc ? rc : mxm( Reflector, get_view< alp::view::transpose >( A21_out ), A21_out, ring );

				auto A22 = get_view( LL, range2, range2 );
				rc = rc ? rc : foldl( A22, Reflector, minus );
			}

			return rc;
		}

		// inplace non-blocked versions
		template<
			typename D = double,
			typename ViewL,
			typename ImfRL,
			typename ImfCL,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D >
		>
		RC cholesky_uptr(
			Matrix< D, structures::Square, Dense, ViewL, ImfRL, ImfCL > &L,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			const Scalar< D > zero( ring.template getZero< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( L );

			for( size_t k = 0; k < n ; ++k ) {
#ifdef DEBUG
				std::cout << "============ Iteration " << k << " ============" << std::endl;
#endif

				auto a = get_view( L, k, utils::range( k, n ) );

				// L[ k, k ] = alpha = sqrt( LL[ k, k ] )
				Scalar< D > alpha;
				rc = rc ? rc : eWiseLambda(
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

				auto v = get_view( L, k, utils::range( k + 1, n ) );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
#endif
				// LL[ k + 1: , k ] = LL[ k + 1: , k ] / alpha
				rc = rc ? rc : foldl( v, alpha, divide );

#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (1) failed\n";
					return rc;
				}
#endif

				// LL[ k+1: , k+1: ] -= v*v^T
				auto Lprim = get_view( L, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vvt = outer( v, ring.getMultiplicativeOperator() );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
				print_matrix( " vvt ", vvt );
#endif

				rc = rc ? rc : foldl( Lprim, vvt, minus );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (2) failed\n";
					return rc;
				}
#endif
				auto vcol = get_view( L, utils::range( k + 1, n ), k );
				set( vcol, zero );

			}

			return rc;
		}


		// inplace blocked version
		template<
			typename D,
			typename ViewL,
			typename ImfRL,
			typename ImfCL,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC cholesky_uptr_blk(
			Matrix< D, structures::Square, Dense, ViewL, ImfRL, ImfCL > &L,
			const size_t &bs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			(void) divide;
			const Scalar< D > zero( ring.template getZero< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( L );

			//nb: number of blocks of (max) size bz
			size_t nb = n / bs ;
			if( n % bs != 0 ){
				nb = nb + 1 ;
			}


			for( size_t i = 0; i < nb ; ++i ) {
				size_t a = std::min( i * bs, n );
				size_t b = std::min( ( i + 1 ) * bs, n );
				size_t c = n;

				auto range1 = utils::range( a, b );
				auto range2 = utils::range( b, c );

				//A11=L[i*bs:(i+1)*bs,i*bs:(i+1)*bs]
				auto A11 = get_view< structures::Square >( L, range1, range1 );


				//A21=L[(i+1)*bs:,i*bs:(i+1)*bs]
				// for complex we should conjugate A21
				auto A21 = get_view< structures::General >( L, range1, range2 );


				Matrix< D, structures::General > A21_tmp( nrows(A21), ncols(A21) );
				rc = rc ? rc : set( A21_tmp, A21 );
				if( rc != SUCCESS ) {
					std::cout << "set failed\n";
					return rc;
				}


				rc = rc ? rc : cholesky_uptr( A11, ring );
				if( rc != SUCCESS ) {
					std::cout << "cholesky_uptr failed\n";
					return rc;
				}

				// this view cannot be used in the current foldl
				// i.e. foldl(Square,UpperTriangular) will update
				// only UpperTriangular of Square
				//auto A11_T = get_view< alp::view::transpose >( A11 );
				auto A11UT = get_view< structures::UpperTriangular >( L, range1, range1 );

				auto A11UT_T = get_view< alp::view::transpose >( A11UT );

				rc = rc ? rc : algorithms::forwardsubstitution(
					A11UT_T,
					A21,
					A21_tmp,
					ring
				);
				if( rc != SUCCESS ) {
					std::cout << "Forwardsubstitution failed\n";
					return rc;
				}

				//Matrix< D, structures::Symmetric, Dense > Reflector( ncols(A21), ncols(A21) );
				Matrix< D, structures::Square, Dense > Reflector( ncols(A21), ncols(A21) );
				rc = rc ? rc : set( Reflector, zero );
				if( rc != SUCCESS ) {
					std::cout << "set(2) failed\n";
					return rc;
				}
				rc = rc ? rc : mxm( Reflector, get_view< alp::view::transpose >( A21 ), A21, ring );
				if( rc != SUCCESS ) {
					std::cout << "mxm failed\n";
					return rc;
				}

				auto A22 = get_view< structures::Square >( L, range2, range2 );

				rc = rc ? rc : foldl( A22, Reflector, minus );
				if( rc != SUCCESS ) {
					std::cout << "foldl failed\n";
					return rc;
				}

				auto A12 = get_view< structures::General >( L, range2, range1 );
				rc = rc ? rc : set( A12, zero );

			}

			return rc;
		}


	} // namespace algorithms
} // namespace alp
