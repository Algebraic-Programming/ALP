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

#include <numeric> //iota
#include <iostream>
#include <sstream>

#include <alp.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include <alp/algorithms/forwardsubstitution.hpp>
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

namespace alp {

	namespace algorithms {

		/**
		 * Computes Householder LU decomposition of general matrix \f$H = LU\f$
		 * where \a H is general (complex or real),
		 * \a L lower trapezoidal,
		 * \a U is upper trapezoidal.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] L    output lower trapezoidal matrix
		 * @param[out] U    output upper trapezoidal matrix
		 * @param[out] p    output permutation vector
		 * @param[in]  H    input general matrix
		 * @param[in]  ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatH,
			typename D = typename MatH::value_type,
			typename MatL,
			typename MatU,
			typename IndexType,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				std::is_integral< IndexType >::value &&
				is_matrix< MatH >::value &&
				is_matrix< MatL >::value &&
				is_matrix< MatU >::value &&
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatL::structure, structures::LowerTrapezoidal >::value &&
				structures::is_a< typename MatU::structure, structures::UpperTrapezoidal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC householder_lu(
			const MatH &H,
			MatL &L,
			MatU &U,
			Vector< IndexType > &p,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );
			const size_t kk = std::min( n, m );

			// initialize permutation vector to identity permutation
			alp::set< alp::descriptors::use_index >( p, alp::Scalar< IndexType >( 0 ) );

			// check sizes
			if(
				( nrows( L ) != nrows( H ) ) ||
				( ncols( U ) != ncols( H ) ) ||
				( nrows( U ) != kk ) ||
				( ncols( L ) != kk )
			) {
#ifdef DEBUG
				std::cerr << " n, kk, m = " << n << ", "  << kk << ", " << m << "\n";
				std::cerr << "Incompatible sizes in householder_lu.\n";
#endif
				return FAILED;
			}


			// L = identity( n )
			auto Ldiag = alp::get_view< alp::view::diagonal >( L );
			rc = rc ? rc : alp::set( Ldiag, one );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( L, I ) failed\n";
				return rc;
			}

			// Out of place specification of the computation
			alp::Matrix< D, structures::General > HWork( m, n );
			rc = rc ? rc : alp::set( HWork, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( HWork, H ) failed\n";
				return rc;
			}

			Vector< D > PivotVec( n );
			rc = rc ? rc : alp::set( PivotVec, zero );

			for( size_t k = 0; k < std::min( n, m ); ++k ) {
				// =====   algorithm  =====
				// a = H[ k, k ]
				// v = H[ k + 1 : , k ]
				// w = H[ k, k + 1 : ]
				// Ak = H[ k + 1 :, k + 1 : ]
				// v = v / a
				// Ak = Ak - outer(v,w)
				// scalar view should replace vector view of length 1 (issue #598)
				// besides here there are many places in the use cases where this should be changed
				auto a_view = alp::get_view( HWork, utils::range( k, k + 1 ), k );
				auto v_view = alp::get_view( HWork, utils::range( k + 1, m ), k );
				auto w_view = alp::get_view( HWork, k, utils::range( k + 1, n ) );
				auto Ak_view = alp::get_view( HWork, utils::range( k + 1, m ), utils::range( k + 1, n ) );

				Scalar< D > alpha( zero );
				rc = rc ? rc : alp::foldl( alpha, a_view, ring.getAdditiveMonoid() );

				// pivoting: find index ipivot
				size_t ipivot = k;
				rc = rc ? rc : eWiseLambda(
					[ &alpha, &ipivot, &k ]( const size_t i, D &val ) {
						if( std::abs( val ) > std::abs( *alpha ) ) {
							*alpha = val;
							ipivot = i + k + 1;
						}
					},
					v_view
				);
				// do pivoting if needed
				if( ipivot > k ) {
					//p[ ipivot ] <-> p[ k ]
					auto p1 = alp::get_view( p, utils::range( k, k + 1 ) );
					auto p2 = alp::get_view( p, utils::range( ipivot, ipivot + 1 ) );
					Vector< size_t > ptmp( 1 );
					rc = rc ? rc : alp::set( ptmp, p1 );
					rc = rc ? rc : alp::set( p1, p2 );
					rc = rc ? rc : alp::set( p2, ptmp );

					//HWork[ ipivot ] <-> HWork[ k ]
					auto v1 = alp::get_view( HWork, k, utils::range( 0, n ) );
					auto v2 = alp::get_view( HWork, ipivot, utils::range( 0, n ) );
					rc = rc ? rc : alp::set( PivotVec, v1 );
					rc = rc ? rc : alp::set( v1, v2 );
					rc = rc ? rc : alp::set( v2, PivotVec );
				}

				rc = rc ? rc : alp::foldl( v_view, alpha, divide );

				auto w_view_star = conjugate( w_view );
				auto Reflector = alp::outer( v_view, w_view_star, ring.getMultiplicativeOperator() );

				rc = rc ? rc : alp::foldl( Ak_view, Reflector, minus );

			}


			// save the result in L and U
			auto H_Utrapez = get_view< structures::UpperTrapezoidal >( HWork, utils::range( 0, kk ), utils::range( 0, n ) );
			rc = rc ? rc : alp::set( U, H_Utrapez );

			auto H_Ltrapez = get_view< structures::LowerTrapezoidal >( HWork, utils::range( 1, m ), utils::range( 0, kk ) );
			auto L_lowerTrapez = get_view( L, utils::range( 1, m ), utils::range( 0, kk ) );
			rc = rc ? rc : alp::set( L_lowerTrapez, H_Ltrapez );

			return rc;

		}

		/** version without pivoting */
		template<
			typename MatH,
			typename D = typename MatH::value_type,
			typename MatL,
			typename MatU,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				is_matrix< MatH >::value &&
				is_matrix< MatL >::value &&
				is_matrix< MatU >::value &&
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatL::structure, structures::LowerTrapezoidal >::value &&
				structures::is_a< typename MatU::structure, structures::UpperTrapezoidal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC householder_lu(
			const MatH &H,
			MatL &L,
			MatU &U,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );
			const size_t kk = std::min( n, m );

			// check sizes
			if(
				( nrows( L ) != nrows( H ) ) ||
				( ncols( U ) != ncols( H ) ) ||
				( nrows( U ) != kk ) ||
				( ncols( L ) != kk )
			) {
				std::cerr << " n, kk, m = " << n << ", "  << kk << ", " << m << "\n";
				std::cerr << "Incompatible sizes in householder_lu.\n";
				return FAILED;
			}

			// L = identity( n )
			auto Ldiag = alp::get_view< alp::view::diagonal >( L );
			rc = rc ? rc : alp::set( Ldiag, one );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( L, I ) failed\n";
				return rc;
			}

			// Out of place specification of the computation
			alp::Matrix< D, structures::General > HWork( m, n );
			rc = rc ? rc : alp::set( HWork, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( HWork, H ) failed\n";
				return rc;
			}

			Vector< D > PivotVec( n );
			rc = rc ? rc : alp::set( PivotVec, zero );

			for( size_t k = 0; k < std::min( n, m ); ++k ) {
				// =====   algorithm  =====
				// a = H[ k, k ]
				// v = H[ k + 1 : , k ]
				// w = H[ k, k + 1 : ]
				// Ak = H[ k + 1 :, k + 1 : ]
				// v = v / a
				// Ak = Ak - outer(v,w)
				// scalar view should replace vector view of length 1 (issue #598)
				// besides here there are many places in the use cases where this should be changed
				auto a_view = alp::get_view( HWork, utils::range( k, k + 1 ), k );
				auto v_view = alp::get_view( HWork, utils::range( k + 1, m ), k );
				auto w_view = alp::get_view( HWork, k, utils::range( k + 1, n ) );
				auto Ak_view = alp::get_view( HWork, utils::range( k + 1, m ), utils::range( k + 1, n ) );

				Scalar< D > alpha( zero );
				rc = rc ? rc : alp::foldl( alpha, a_view, ring.getAdditiveMonoid() );

				rc = rc ? rc : alp::foldl( v_view, alpha, divide );

				auto w_view_star = conjugate( w_view );
				auto Reflector = alp::outer( v_view, w_view_star, ring.getMultiplicativeOperator() );

				rc = rc ? rc : alp::foldl( Ak_view, Reflector, minus );
			}

			// save the result in L and U
			auto H_Utrapez = get_view< structures::UpperTrapezoidal >( HWork, utils::range( 0, kk ), utils::range( 0, n ) );
			rc = rc ? rc : alp::set( U, H_Utrapez );

			auto H_Ltrapez = get_view< structures::LowerTrapezoidal >( HWork, utils::range( 1, m ), utils::range( 0, kk ) );
			auto L_lowerTrapez = get_view( L, utils::range( 1, m ), utils::range( 0, kk ) );
			rc = rc ? rc : alp::set( L_lowerTrapez, H_Ltrapez );

			return rc;

		}

		/** blocked version without pivoting */
		template<
			typename MatH,
			typename D = typename MatH::value_type,
			typename MatL,
			typename MatU,
			typename IndexType,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				std::is_integral< IndexType >::value &&
				is_matrix< MatH >::value &&
				is_matrix< MatL >::value &&
				is_matrix< MatU >::value &&
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatL::structure, structures::LowerTrapezoidal >::value &&
				structures::is_a< typename MatU::structure, structures::UpperTrapezoidal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC householder_lu(
			const MatH &H,
			MatL &L,
			MatU &U,
			IndexType &bs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );
			const size_t kk = std::min( n, m );

			// check sizes
			if(
				( nrows( L ) != nrows( H ) ) ||
				( ncols( U ) != ncols( H ) ) ||
				( nrows( U ) != kk ) ||
				( ncols( L ) != kk )
			) {
				std::cerr << " n, kk, m = " << n << ", "  << kk << ", " << m << "\n";
				std::cerr << "Incompatible sizes in householder_lu.\n";
				return FAILED;
			}


			alp::set( L, zero );
			alp::set( U, zero );

			// Out of place specification of the computation
			MatH HWork( m, n );
			rc = rc ? rc : alp::set( HWork, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( HWork, H ) failed\n";
				return rc;
			}

			size_t nblocks_r = m / bs;
			if( m != nblocks_r * bs ) {
				nblocks_r += 1;
			}

			size_t nblocks_c = n / bs;
			if( n != nblocks_c * bs ) {
				nblocks_c += 1;
			}

			for( size_t k = 0; k < std::min( nblocks_r, nblocks_c ); ++k ) {

				auto range_a = utils::range( k * bs, std::min( ( k + 1) * bs, kk ) );

				auto range_c = utils::range( std::min( ( k + 1) * bs, kk ), nrows( L ) );
				auto range_d = utils::range( std::min( ( k + 1) * bs, kk ), ncols( U ) );

				auto A00 = alp::get_view< structures::General >( HWork, range_a, range_a );
				auto A01 = alp::get_view( HWork, range_a, range_d );
				auto A10 = alp::get_view( HWork, range_c, range_a );
				auto A11 = alp::get_view( HWork, range_c, range_d );

				auto L00 = alp::get_view< structures::LowerTrapezoidal >( L, range_a, range_a );
				auto L10 = alp::get_view< structures::General >( L, range_c, range_a );

				auto U00 = alp::get_view< structures::UpperTrapezoidal >( U, range_a, range_a );
				auto U01 = alp::get_view< structures::General >( U, range_a, range_d );

				rc = rc ? rc : algorithms::householder_lu( A00, L00, U00, ring );

				// U[k*bs:b,b:]=inv(L00).dot(A0[k*bs:b,b:][p00])
				// L00 X=A0[k*bs:b,b:]
				auto L00_LT = alp::get_view< structures::LowerTriangular >( L00 );
				rc = rc ? rc : algorithms::forwardsubstitution( L00_LT, U01, A01, ring );

				// L[b:,k*bs:b]=(A0[b:,k*bs:b]).dot(inv(U00))
				//  U00.T X.T =A0[b:,k*bs:b].T
				auto U00_UT = alp::get_view< structures::UpperTriangular >( U00 );
				auto U00_UT_T = alp::get_view< alp::view::transpose >( U00_UT );
				auto A10_T = alp::get_view< alp::view::transpose >( A10 );
				auto L10_T = alp::get_view< alp::view::transpose >( L10 );
				rc = rc ? rc : algorithms::forwardsubstitution( U00_UT_T, L10_T, A10_T, ring );

				// A11tmp=L[b:,k*bs:b].dot(U[k*bs:b,b:])
				// A0[b:,b:]=A0[b:,b:]-A11tmp
				Matrix< D, structures::General > A11tmp( nrows( L10 ), ncols( U01 ) );

				rc = rc ? rc : set( A11tmp, zero );
				rc = rc ? rc : mxm( A11tmp, L10, U01, ring );;
				rc = rc ? rc : foldl( A11, A11tmp, minus );
			}

			return rc;
		}


		/** blocked version with per-block pivoting */
		template<
			typename MatH,
			typename D = typename MatH::value_type,
			typename MatL,
			typename MatU,
			typename IndexType,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				std::is_integral< IndexType >::value &&
				is_matrix< MatH >::value &&
				is_matrix< MatL >::value &&
				is_matrix< MatU >::value &&
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatL::structure, structures::LowerTrapezoidal >::value &&
				structures::is_a< typename MatU::structure, structures::UpperTrapezoidal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value
			> * = nullptr
		>
		RC householder_lu(
			const MatH &H,
			MatL &L,
			MatU &U,
			Vector< IndexType > &p,
			IndexType &bs,
			const Ring &ring = Ring(),
			const Minus &minus = Minus()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );
			const size_t kk = std::min( n, m );

			// initialize permutation vector to identity permutation
			alp::set< alp::descriptors::use_index >( p, alp::Scalar< IndexType >( 0 ) );

			// check sizes
			if(
				( nrows( L ) != nrows( H ) ) ||
				( ncols( U ) != ncols( H ) ) ||
				( nrows( U ) != kk ) ||
				( ncols( L ) != kk )
			) {
				std::cerr << " n, kk, m = " << n << ", "  << kk << ", " << m << "\n";
				std::cerr << "Incompatible sizes in householder_lu.\n";
				return FAILED;
			}

			alp::set( L, zero );
			alp::set( U, zero );

			// Out of place specification of the computation
			MatH HWork( m, n );
			rc = rc ? rc : alp::set( HWork, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( HWork, H ) failed\n";
				return rc;
			}

			Vector< D > PivotVec( n );
			rc = rc ? rc : alp::set( PivotVec, zero );

			size_t nblocks_r = m / bs;
			if( m != nblocks_r * bs ) {
				nblocks_r += 1;
			}

			size_t nblocks_c = n / bs;
			if( n != nblocks_c * bs ) {
				nblocks_c += 1;
			}

			for( size_t k = 0; k < std::min( nblocks_r, nblocks_c ); ++k ) {
#ifdef DEBUG
				std::cout << "k = "<< kk << "\n";
#endif

				auto range_a = utils::range( k * bs, std::min( ( k + 1) * bs, kk ) );
				auto range_c = utils::range( std::min( ( k + 1) * bs, kk ), nrows( L ) );
				auto range_d = utils::range( std::min( ( k + 1) * bs, kk ), ncols( U ) );

				auto A00 = alp::get_view< structures::General >( HWork, range_a, range_a );
				auto A01 = alp::get_view( HWork, range_a, range_d );
				auto A10 = alp::get_view( HWork, range_c, range_a );
				auto A11 = alp::get_view( HWork, range_c, range_d );

				auto L00 = alp::get_view< structures::LowerTrapezoidal >( L, range_a, range_a );
				auto L10 = alp::get_view< structures::General >( L, range_c, range_a );

				auto U00 = alp::get_view< structures::UpperTrapezoidal >( U, range_a, range_a );
				auto U01 = alp::get_view< structures::General >( U, range_a, range_d );

				alp::Vector< size_t > pvec( nrows( A00 ) );
				alp::set< alp::descriptors::use_index >( pvec, alp::Scalar< size_t >( 0 ) );
				rc = rc ? rc : algorithms::householder_lu( A00, L00, U00, pvec, ring );

				// U[ k * bs : b, b : ] = inv( L00 ).dot( A0[ k * bs : b, b : ][ p00 ] )
				// is equivalent to:
				// L00 U[ k * bs : b, b : ] = A0[ k*bs : b, b : ][p00]
				auto L00_LT = alp::get_view< structures::LowerTriangular >( L00 );
				alp::Vector< size_t > no_permutation_vec( ncols( A01 ) );
				alp::set< alp::descriptors::use_index >( no_permutation_vec, alp::Scalar< size_t >( 0 ) );
				auto A01_perm = alp::get_view< alp::structures::General >( A01, pvec, no_permutation_vec );
				rc = rc ? rc : algorithms::forwardsubstitution( L00_LT, U01, A01_perm, ring );

				// L[ b : , k * bs : b ] = A0[ b : ,k * bs : b ].dot( inv( U00 ) )
				// is equivalent to:
				// U00.T L[ b : , k * bs : b ].T = A0[ b : , k * bs : b ].T
				auto U00_UT = alp::get_view< structures::UpperTriangular >( U00 );
				auto U00_UT_T = alp::get_view< alp::view::transpose >( U00_UT );
				auto A10_T = alp::get_view< alp::view::transpose >( A10 );
				auto L10_T = alp::get_view< alp::view::transpose >( L10 );
				rc = rc ? rc : algorithms::forwardsubstitution( U00_UT_T, L10_T, A10_T, ring );

				// update non-decomposed part for work matrix (A11)
				// A11tmp = L[ b : ,k * bs : b ].dot( U[ k * bs : b, b : ] )
				// A0[ b : ,b : ] = A0[ b : ,b : ] - A11tmp
				Matrix< D, structures::General > A11tmp( nrows( L10 ), ncols( U01 ) );
				rc = rc ? rc : set( A11tmp, zero );
				rc = rc ? rc : mxm( A11tmp, L10, U01, ring );
				rc = rc ? rc : foldl( A11, A11tmp, minus );

				//permute nondiagonal-L block
				//L[ k * bs : b, : k * bs ] = L[ k * bs : b, : k * bs ][p00]
				auto L00a = alp::get_view< structures::General >( L, range_a, utils::range( 0, k * bs ) );
				Matrix< D, structures::General > tmpL00a( nrows( L00a ), ncols( L00a ) );
				rc = rc ? rc : set( tmpL00a, L00a );
				alp::Vector< size_t > no_permutation_vec2( ncols( L00a ) );
				alp::set< alp::descriptors::use_index >( no_permutation_vec2, alp::Scalar< size_t >( 0 ) );
				auto tmpL00a_perm = alp::get_view< alp::structures::General >( tmpL00a, pvec, no_permutation_vec2 );
				rc = rc ? rc : set( L00a, tmpL00a_perm );

				//update permutation vector
				auto p_block = alp::get_view( p, range_a );
				alp::Vector< size_t > vec_tmp( size( p_block ) );
				rc = rc ? rc : set( vec_tmp, p_block );
				auto vec_tmp_perm = alp::get_view< alp::structures::General >( vec_tmp, pvec );
				rc = rc ? rc : set( p_block, vec_tmp_perm );

			}

			return rc;
		}


	} // namespace algorithms
} // namespace alp
