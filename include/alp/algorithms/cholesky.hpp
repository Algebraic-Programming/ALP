/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include <alp/algorithms/forwardsubstitution.hpp>
#include "../../../tests/utils/print_alp_containers.hpp"

namespace alp {

	namespace algorithms {

		/**
		 * Computes the Cholesky decomposition U^TU = H of a real symmetric
		 * positive definite (SPD) (or complex Hermitian positive definite)
		 * matrix H where \a U is upper triangular.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] U    output upper triangular matrix
		 * @param[in]  H    input real symmetric positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatU,
			typename MatH,
			typename D = typename MatU::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D >,
			std::enable_if_t<
				is_matrix< MatU >::value &&
				is_matrix< MatH >::value &&
				structures::is_a< typename MatU::structure, structures::UpperTriangular >::value &&
				// TODO: structures::SymmetricPositiveDefinite should be replced
				//       with structures::SymmetricPositiveDefinitePositiveDefinite
				(
					(
						!grb::utils::is_complex< D >::value &&
						structures::is_a< typename MatH::structure, structures::SymmetricPositiveDefinite >::value
					) || (
						grb::utils::is_complex< D >::value &&
						structures::is_a< typename MatH::structure, structures::HermitianPositiveDefinite >::value
					)
				) &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC cholesky_uptr(
			MatU &U,
			const MatH &H,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			if(
				( nrows( U ) != nrows( H ) ) ||
				( ncols( U ) != ncols( H ) )
			) {
				std::cerr << "Incompatible sizes in cholesky_uptr.\n";
				return FAILED;
			}

			const size_t n = nrows( H );

			// Out of place specification of the operation
			Matrix< D, typename MatH::structure > UU( n );
			rc = rc ? rc : set( UU, H );

#ifdef DEBUG
			if( rc != SUCCESS ) {
				std::cerr << " set( UU, H ) failed\n";
				return rc;
			}
			print_matrix( " -- UU --  " , UU );
#endif

			for( size_t k = 0; k < n; ++k ) {
#ifdef DEBUG
				std::cout << "============ Iteration " << k << " ============" << std::endl;
#endif

				auto a = get_view( UU, k, utils::range( k, n ) );
#ifdef DEBUG
				print_vector( " -- a --  " , a );
#endif

				// U[ k, k ] = alpha = sqrt( UU[ k, k ] )
				Scalar< D > alpha;
				rc = rc ? rc : eWiseLambda(
					[ &alpha, &ring ]( const size_t i, D &val ) {
						if( i == 0 ) {
							(void) set( alpha, alp::Scalar< D >( std::sqrt( val ) ) );
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

				auto v = get_view( UU, k, utils::range( k + 1, n ) );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
#endif
				// UU[ k + 1: , k ] = UU[ k + 1: , k ] / alpha
				rc = rc ? rc : foldl( v, alpha, divide );

#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (1) failed\n";
					return rc;
				}
#endif

				// UU[ k+1: , k+1: ] -= v*v^T
				auto Uprim = get_view( UU, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vstar = conjugate( v );
				auto vvt = outer( vstar, ring.getMultiplicativeOperator() );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
				print_matrix( " vvt ", vvt );
#endif
				rc = rc ? rc : foldl( Uprim, vvt, minus );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (2) failed\n";
					return rc;
				}
#endif
			}

			// Finally collect output into U matrix and return
			for( size_t k = 0; k < n; ++k ) {

				// U[ k, k: ] = UU[ k, k: ]
				auto vU  = get_view( U, k, utils::range( k, n )  );
				auto vUU = get_view( UU, k, utils::range( k, n )  );

				rc = set( vU, vUU );
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
		 * Computes the blocked version Cholesky decomposition U^TU = H of a real symmetric
		 * positive definite (SPD) matrix H where \a U is upper triangular.
		 * U^T  is equvalent to transpose(U)
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @param[out] U    output upper triangular matrix
		 * @param[in]  H    input real symmetric positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatU,
			typename MatH,
			typename D = typename MatU::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			std::enable_if_t<
				is_matrix< MatU >::value &&
				is_matrix< MatH >::value &&
				structures::is_a< typename MatU::structure, structures::UpperTriangular >::value &&
				structures::is_a< typename MatH::structure, structures::SymmetricPositiveDefinite >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value
			> * = nullptr
		>
		RC cholesky_uptr_blk(
			MatU &U,
			const MatH &H,
			const size_t &bs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus()
		) {
			const Scalar< D > zero( ring.template getZero< D >() );

			if(
				( nrows( U ) != nrows( H ) ) ||
				( ncols( U ) != ncols( H ) )
			) {
				std::cerr << "Incompatible sizes in cholesky_uptr_blk.\n";
				return FAILED;
			}

			RC rc = SUCCESS;

			const size_t n = nrows( U );

			Matrix< D, typename MatH::structure > UU( n );
			rc = rc ? rc : set( UU, H );
#ifdef DEBUG
			if( rc != SUCCESS ) {
				std::cout << "set failed\n";
				return rc;
			}
#endif

			//nb: number of blocks of (max) size bz
			if( ( bs == 0 ) || ( bs > n ) ) {
				std::cerr << "Block size has illegal value, bs =   " << bs << " .\n";
				std::cerr << "It should be from interval < 0,  " << n << "] .\n";
				return FAILED;
			}
			size_t nb = n / bs;
			if( n % bs != 0 ){
				nb = nb + 1;
			}


			for( size_t i = 0; i < nb; ++i ) {
				const size_t a = std::min( i * bs, n );
				const size_t b = std::min( ( i + 1 ) * bs, n );
				const size_t c = n;

				const utils::range range1( a, b );
				const utils::range range2( b, c );

				auto A11 = get_view( UU, range1, range1 );

				// for complex we should conjugate A12
				auto A12 = get_view< structures::General >( UU, range1, range2 );

				//A11=cholesky(A11)
				auto A11_out = get_view( U, range1, range1 );

				rc = rc ? rc : cholesky_uptr( A11_out, A11, ring );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "cholesky_uptr failed\n";
					return rc;
				}
#endif

				auto A12_out = get_view< structures::General >(	U, range1, range2 );
				auto A11_out_T = get_view< alp::view::transpose >( A11_out );

				rc = rc ? rc : algorithms::forwardsubstitution(
					A11_out_T,
					A12_out,
					A12,
					ring
				);
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "Forwardsubstitution failed\n";
					return rc;
				}
#endif

				Matrix< D, typename MatH::structure > Reflector( ncols( A12_out ) );
				rc = rc ? rc : set( Reflector, zero );
				rc = rc ? rc : mxm( Reflector, get_view< alp::view::transpose >( A12_out ), A12_out, ring );

				auto A22 = get_view( UU, range2, range2 );
				rc = rc ? rc : foldl( A22, Reflector, minus );
			}

			return rc;
		}

		/** inplace non-blocked versions, part below diagonal is not modified */
		template<
			typename MatU,
			typename D = typename MatU::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D >,
			std::enable_if_t<
				is_matrix< MatU >::value &&
				structures::is_a< typename MatU::structure, structures::Square >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC cholesky_uptr(
			MatU &U,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			const Scalar< D > zero( ring.template getZero< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( U );

			for( size_t k = 0; k < n; ++k ) {
#ifdef DEBUG
				std::cout << "============ Iteration " << k << " ============" << std::endl;
#endif

				auto a = get_view( U, k, utils::range( k, n ) );

				// U[ k, k ] = alpha = sqrt( UU[ k, k ] )
				Scalar< D > alpha;
				rc = rc ? rc : eWiseLambda(
					[ &alpha, &ring ]( const size_t i, D &val ) {
						if( i == 0 ) {
							(void) set( alpha, alp::Scalar< D >( std::sqrt( val ) ) );
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

				auto v = get_view( U, k, utils::range( k + 1, n ) );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
#endif
				// UU[ k, k + 1: ] = UU[ k, k + 1: ] / alpha
				rc = rc ? rc : foldl( v, alpha, divide );

#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (1) failed\n";
					return rc;
				}
#endif

				// UU[ k+1: , k+1: ] -= v*v^T
				auto Uprim = get_view( U, utils::range( k + 1, n ), utils::range( k + 1, n ) );

				auto vvt = outer( v, ring.getMultiplicativeOperator() );
#ifdef DEBUG
				print_vector( " -- v --  " , v );
				print_matrix( " vvt ", vvt );
#endif

				rc = rc ? rc : foldl( Uprim, vvt, minus );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, view ) (2) failed\n";
					return rc;
				}
#endif

			}

			return rc;
		}


		/** inplace blocked version, part below diagonal is not modified */
		template<
			typename MatU,
			typename D = typename MatU::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			std::enable_if_t<
				is_matrix< MatU >::value &&
				structures::is_a< typename MatU::structure, structures::Square >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value
			> * = nullptr
		>
		RC cholesky_uptr_blk(
			MatU &U,
			const size_t &bs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus()
		) {
			const Scalar< D > zero( ring.template getZero< D >() );

			RC rc = SUCCESS;

			const size_t n = nrows( U );

			//nb: number of blocks of (max) size bz
			if( ( bs == 0 ) || ( bs > n ) ) {
				std::cerr << "Block size has illegal value, bs =   " << bs << " .\n";
				std::cerr << "It should be from interval < 0,  " << n << "] .\n";
				return FAILED;
			}
			size_t nb = n / bs;
			if( n % bs != 0 ){
				nb = nb + 1;
			}


			for( size_t i = 0; i < nb; ++i ) {
				const size_t a = std::min( i * bs, n );
				const size_t b = std::min( ( i + 1 ) * bs, n );
				const size_t c = n;

				const utils::range range1( a, b );
				const utils::range range2( b, c );

				auto A11 = get_view< structures::Square >( U, range1, range1 );

				// for complex we should conjugate A12
				auto A12 = get_view< structures::General >( U, range1, range2 );

				rc = rc ? rc : cholesky_uptr( A11, ring );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "cholesky_uptr failed\n";
					return rc;
				}
#endif

				//auto A11_T = get_view< alp::view::transpose >( A11 );
				auto A11UT = get_view< structures::UpperTriangular >( U, range1, range1 );

				auto A11UT_T = get_view< alp::view::transpose >( A11UT );

				rc = rc ? rc : algorithms::forwardsubstitution(	A11UT_T, A12, ring );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "Forwardsubstitution failed\n";
					return rc;
				}
#endif

				Matrix< D, structures::SymmetricPositiveDefinite, Dense > Reflector( ncols( A12 ) );
				rc = rc ? rc : set( Reflector, zero );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "set(2) failed\n";
					return rc;
				}
#endif
				rc = rc ? rc : mxm( Reflector, get_view< alp::view::transpose >( A12 ), A12, ring );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "mxm failed\n";
					return rc;
				}
#endif
				auto A22UT = get_view< structures::SymmetricPositiveDefinite >( U, range2, range2 );

				rc = rc ? rc : foldl( A22UT, Reflector, minus );
#ifdef DEBUG
				if( rc != SUCCESS ) {
					std::cout << "foldl failed\n";
					return rc;
				}
#endif
			}

			return rc;
		}


	} // namespace algorithms
} // namespace alp
