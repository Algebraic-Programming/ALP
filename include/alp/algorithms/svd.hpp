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
#include <alp/algorithms/householder_bidiag.hpp>
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

namespace alp {

	namespace algorithms {


		// for a more general purpose
		// a more stable implementations is needed
		// todo: move to utils?
		template<
			typename D = double,
			typename StruG,
			typename ViewG,
			typename ImfRG,
			typename ImfCG,
			typename Struv,
			typename Viewv,
			typename ImfRv,
			typename ImfCv,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC Givens(
			Matrix< D, StruG, Dense, ViewG, ImfRG, ImfCG > &G,
			Vector< D, Struv, Dense, Viewv, ImfRv, ImfCv > &v,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			(void) minus;
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			// c = abs(a) / sqrt(abs(a)**2 + abs(b)**2)
			// s = (a/abs(a)) * conjugate(b) / sqrt(abs(a)**2 + abs(b)**2)
			// return(array([[c,-conjugate(s)],[(s),c]]))
			Scalar< D > c( zero );
			Scalar< D > s( zero );
			Scalar< D > d( zero );
			rc = rc ? rc : alp::norm2( d, v, ring );
			auto a = get_view( v, utils::range( 0, 1 ) );
			auto b = get_view( v, utils::range( 1, 2 ) );

			rc = rc ? rc : alp::norm2( c, a, ring );
			rc = rc ? rc : alp::foldl( s, a, ring.getAdditiveMonoid() );

			rc = rc ? rc : alp::foldl( s, c, divide );
			rc = rc ? rc : alp::foldl( s, conjugate( b ), ring.getMultiplicativeMonoid() );

			// return(array([[c,-conjugate(s)],[(s),c]]),r)
			auto G11 = get_view( G, 0, utils::range( 0, 1 ) );
			auto G12 = get_view( G, 0, utils::range( 1, 2 ) );
			auto G21 = get_view( G, 1, utils::range( 0, 1 ) );
			auto G22 = get_view( G, 1, utils::range( 1, 2 ) );
			rc = rc ? rc : alp::set( G, zero );
			rc = rc ? rc : alp::foldl( G11, c, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( G22, c, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( G21, s, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::set( G12, conjugate( G21 ) );
			rc = rc ? rc : alp::foldl( G12, Scalar< D >( -1 ), ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::foldl( G, d, divide );
			return rc;
		}


		/** Golub-Kahan SVD step */
		template<
			typename D = double,
			typename StruB,
			typename ViewB,
			typename ImfRB,
			typename ImfCB,
			typename StruU,
			typename ViewU,
			typename ImfRU,
			typename ImfCU,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				structures::is_a< StruB, structures::General >::value &&
				// structures::is_a< StruU, structures::Orthogonal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC gk_svd_step(
			Matrix< D, StruU, Dense, ViewU, ImfRU, ImfCU > &U,
			Matrix< D, StruB, Dense, ViewB, ImfRB, ImfCB > &B,
			Matrix< D, StruU, Dense, ViewU, ImfRU, ImfCU > &V,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( B );
			const size_t n = ncols( B );
			const size_t k = std::min( m, n );

			// get lambda
			// calcualte eigenvalue llambda of
			// which is closer to t22
			auto BEnd = get_view( B, utils::range( k - 3, k ), utils::range( k - 2, k ) );
			Matrix< D, structures::Square, Dense > BEndSquare( 2, 2 );
			rc = rc ? rc : alp::set( BEndSquare, zero );
			auto BEndT = get_view< alp::view::transpose >( BEnd );
			auto BEndT_star = conjugate( BEndT );
			rc = rc ? rc : mxm( BEndSquare, BEndT_star, BEnd, ring );

			auto tdiag = get_view< alp::view::diagonal >( BEndSquare );
			auto t11 = get_view( BEndSquare, 0, utils::range( 0, 1 ) );
			auto t12 = get_view( BEndSquare, 0, utils::range( 1, 2 ) );
			auto t22 = get_view( BEndSquare, 1, utils::range( 1, 2 ) );

			Scalar< D > llabmda( zero );
			rc = rc ? rc : alp::foldl( llabmda, tdiag, ring.getAdditiveMonoid() );
			rc = rc ? rc : alp::foldl( llabmda, alp::Scalar< D >( 2 ), divide );

			Scalar< D > bb( zero );
			rc = rc ? rc : alp::foldl( bb, t11, ring.getAdditiveMonoid() );
			rc = rc ? rc : alp::foldl( bb, Scalar< D >( -1 ), ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::foldl( bb, t22, ring.getAdditiveMonoid() );
			rc = rc ? rc : alp::foldl( bb, alp::Scalar< D >( 2 ), divide );

			Scalar< D > cc( zero );
			rc = rc ? rc : alp::foldl( cc, conjugate( t12 ), ring.getAdditiveMonoid() );

			Vector< D > DD( 2 );
			rc = rc ? rc : alp::set( DD, zero );
			auto DD0 =  get_view( DD, utils::range( 0, 1 ) );
			auto DD1 =  get_view( DD, utils::range( 1, 2 ) );
			rc = rc ? rc : alp::foldl( DD0, bb, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( DD1, cc, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::set( bb, zero );
			rc = rc ? rc : alp::norm2( bb, DD, ring );

			Scalar< D > t11scal( zero );
			Scalar< D > t22scal( zero );
			rc = rc ? rc : alp::foldl( t11scal, t11, ring.getAdditiveMonoid() );
			rc = rc ? rc : alp::foldl( t22scal, t22, ring.getAdditiveMonoid() );

			if ( std::real( *t11scal ) > std::real( *t22scal ) ) {
				rc = rc ? rc : alp::foldl( llabmda, bb, minus );
			} else {
				rc = rc ? rc : alp::foldl( llabmda, bb, ring.getAdditiveOperator() );
			}
			// end of get lambda

			Vector< D > rotvec( 2 );
			auto Brow = get_view( B, 0, utils::range( 0, 2 ) );
			auto B00 = get_view( B, 0, utils::range( 0, 1 ) );
			Scalar< D > b00star( zero );
			rc = rc ? rc : alp::foldl( b00star, conjugate( B00 ), ring.getAdditiveMonoid() );
			rc = rc ? rc : alp::set( rotvec, Brow );
			rc = rc ? rc : alp::foldl( rotvec, b00star, ring.getMultiplicativeOperator() );

			auto rotvec0 = get_view( rotvec, utils::range( 0, 1 ) );
			rc = rc ? rc : alp::foldl( rotvec0, llabmda, minus );

			Matrix< D, structures::Square, Dense > G( 2, 2 );
			rc = rc ? rc : alp::set( G, zero );
			rc = rc ? rc : Givens( G, rotvec );
			auto Gdiag = get_view< alp::view::diagonal >( G );
			auto Gstar = conjugate( G );
			auto GT = get_view< alp::view::transpose >( G );
			auto GTstar = conjugate( GT );

			for( size_t i = 0; i < k - 1; ++i ){
				// B[max(i-1,0):i+2,i:i+2]=B[max(i-1,0):i+2,i:i+2].dot(G)
				auto Bblock1 = get_view( B, utils::range( ( i == 0 ? 0 : i - 1 ), i + 2 ), utils::range( i, i + 2 ) );
				Matrix< D, structures::General, Dense > TMP1( nrows( Bblock1 ), ncols( Bblock1 ) );
				rc = rc ? rc : alp::set( TMP1, Bblock1 );
				rc = rc ? rc : alp::set( Bblock1, zero );
				rc = rc ? rc : mxm( Bblock1, TMP1, G, ring );

				// update V
				// G2=G-identity(2).astype(complex)
				rc = rc ? rc : alp::foldl( Gdiag, one, minus );
				// V[i:i+2,:]=V[i:i+2,:] + conjugate(G2).dot(V[i:i+2,:])
				auto Vstrip = get_view< structures::General >( V, utils::range( i, i + 2 ), utils::range( 0, ncols( V ) ) );
				Matrix< D, structures::General, Dense > TMPStrip1( nrows( Vstrip ), ncols( Vstrip ) );
				rc = rc ? rc : alp::set( TMPStrip1, Vstrip );
				rc = rc ? rc : mxm( Vstrip, GTstar, TMPStrip1, ring );

				// B[i:i+2,i:i+3]=G.T.dot(B[i:i+2,i:i+3])
				auto Bblock2 = get_view( B, utils::range( i, i + 2 ), utils::range( i, std::min( i + 3, n ) ) );
				Matrix< D, structures::General, Dense > TMP2( nrows( Bblock2 ), ncols( Bblock2 ) );
				auto rotvec2 = get_view( B, utils::range( i, i + 2 ), i );
				rc = rc ? rc : Givens( G, rotvec2 );
				rc = rc ? rc : alp::set( TMP2, Bblock2 );
				rc = rc ? rc : alp::set( Bblock2, zero );
				rc = rc ? rc : mxm( Bblock2, GT, TMP2, ring );

				// update U
				// G2=G-identity(2).astype(complex)
				rc = rc ? rc : alp::foldl( Gdiag, one, minus );
				// U[:,k:k+2]=U[:,k:k+2]+U[:,k:k+2].dot(conjugate(G2))
				auto Ustrip = get_view< structures::General >( U, utils::range( 0, nrows( U ) ), utils::range( i, i + 2 ) );
				Matrix< D, structures::General, Dense > TMPStrip2( nrows( Ustrip ), ncols( Ustrip ) );
				rc = rc ? rc : alp::set( TMPStrip2, Ustrip );
				rc = rc ? rc : mxm( Ustrip, TMPStrip2, Gstar, ring );

				if( i + 2 < k ) {
					auto rotvec3 = get_view( B, i, utils::range( i + 1, i + 3 ) );
					rc = rc ? rc : Givens( G, rotvec3 );
				} else {
					rc = rc ? rc : Givens( G, rotvec2 );
				}
			}

			return rc;
		}

		/** Golub-Khan SVD algorithm */
		template<
			typename D = double,
			typename StruB,
			typename ViewB,
			typename ImfRB,
			typename ImfCB,
			typename StruU,
			typename ViewU,
			typename ImfRU,
			typename ImfCU,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				structures::is_a< StruB, structures::General >::value &&
				structures::is_a< StruU, structures::Orthogonal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC svd_solve(
			Matrix< D, StruU, Dense, ViewU, ImfRU, ImfCU > &U,
			Matrix< D, StruB, Dense, ViewB, ImfRB, ImfCB > &B,
			Matrix< D, StruU, Dense, ViewU, ImfRU, ImfCU > &V,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( B );
			const size_t n = ncols( B );
			const size_t k = std::min( m, n );

			const double tol = 1.e-12;
			const size_t maxit = k*5;

			auto Bsupsquare =  get_view( B, utils::range( 0, k - 1 ) , utils::range( 1, k ) );
			auto superdiagonal = get_view< alp::view::diagonal >( Bsupsquare );

			size_t i1 = 0;
			size_t i2 = k;

			rc = rc ? rc : algorithms::householder_bidiag( U, B, V, ring, minus, divide );
			//repeat while superdiagonal is not zero
			for( size_t i = 0; i < maxit; ++i ) {
				//todo: replace convergenve tests with absolute tolerance cehck
				//      with reltive tolerance checks

				//todo: check for zeroes in diagonal, if any do Givens rotatations
				//      to move the zero from diagonal to superdiagonal
				//      (no likely to affect randomly generated tests)


				//check for zeros in superdiagonaldiagonal, if any,
				//move i1 and i2 to bound non-zero part of superdiagonaldiagonal
				for( ; i1 < i2 ; ++i1 ) {
					auto B_l = get_view( superdiagonal, utils::range( i1, i1 + 1 ) );
					Scalar< D > bnorm( zero );
					rc = rc ? rc : alp::norm2( bnorm, B_l, ring );
					if( std::abs( *bnorm ) > tol ) {
						break;
					}
				}
				for( ; i2 > i1 ; --i2 ) {
					auto B_l = get_view( superdiagonal, utils::range( i2 - 2, i2 - 1 ) );
					Scalar< D > bnorm( zero );
					rc = rc ? rc : alp::norm2( bnorm, B_l, ring );
					if( std::abs( *bnorm ) > tol ) {
						break;
					}
				}
				if( i2 <= i1 ){
					break;
				}

				auto Bview = get_view( B, utils::range( i1, i2 ), utils::range( i1, i2 ) );
				auto Uview = get_view< structures::General >( U, utils::range( 0, nrows( U ) ), utils::range( i1, i2 ) );
				auto Vview = get_view< structures::General >( V, utils::range( i1, i2 ), utils::range( 0, ncols( V ) ) );

				rc = rc ? rc : algorithms::gk_svd_step( Uview, Bview, Vview, ring, minus, divide );

				// check convergence
				Scalar< D > sup_diag_norm( zero );
				rc = rc ? rc : alp::norm2( sup_diag_norm, superdiagonal, ring );

				if( std::abs( *sup_diag_norm ) < tol ) {
					break ;
				}
			}

			// Rotate diagonal elements in complex plane
			// in order to have them on real axis (positive singular values)
			auto BSquare = alp::get_view( B, utils::range( 0, k ), utils::range( 0, k ) );
			auto DiagBview = alp::get_view< alp::view::diagonal >( BSquare );
			for( size_t i = 0; i < size( DiagBview ); ++i ) {
				Scalar< D > sigmaiphase( zero );
				Scalar< D > sigmainorm( zero );
				auto U_vi = get_view< >( U, utils::range( 0, nrows( U ) ), i );
				auto B_vi = get_view< >( B, i, utils::range( 0, ncols( B ) ) );
				auto d_i = get_view< >( DiagBview, utils::range( i, i + 1 ) );
				rc = rc ? rc : alp::norm2( sigmainorm, d_i, ring );
				if( std::abs( *sigmainorm ) > tol ) {
					rc = rc ? rc : alp::foldl( sigmaiphase, d_i, ring.getAdditiveMonoid() );
					rc = rc ? rc : alp::foldl( sigmaiphase, sigmainorm, divide );
					rc = rc ? rc : alp::foldl( U_vi, sigmaiphase, ring.getMultiplicativeOperator() );
					rc = rc ? rc : alp::foldl( B_vi, sigmaiphase, divide );
				}
			}

			return rc;
		}



		/**
		 *        Computes singular value decomposition (inplace) of a
		 *        general matrix \f$H = U B V \f$
		 *        where \a H is general (complex or real),
		 *        \a U orthogonal and \a V are orthogonal, \a B is nonzero only on diagonal
		 *        and it contains positive singular values.
		 *        If convergenece is not reached B will contain nonzeros on superdiagonal.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out]      U orthogonal matrix
		 * @param[out]      V orthogonal matrix
		 * @param[in,out]   B input general matrix, output bidiagonal matrix
		 * @param[in]       ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatH,
			typename MatU,
			typename MatS,
			typename MatV,
			typename D = typename MatH::value_type,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >,
			std::enable_if_t<
				is_matrix< MatH >::value &&
				is_matrix< MatU >::value &&
				is_matrix< MatS >::value &&
				is_matrix< MatV >::value &&
				structures::is_a< typename MatH::structure, structures::General >::value &&
				structures::is_a< typename MatU::structure, structures::Orthogonal >::value &&
				structures::is_a< typename MatS::structure, structures::General >::value &&
				structures::is_a< typename MatV::structure, structures::Orthogonal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC svd(
			const MatH &H,
			MatU &U,
			MatS &S,
			MatV &V,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );

			//inplace work on B
			MatH B( m, n );
			rc = rc ? rc : set( B, H );

			rc = rc ? rc : set( U, zero );
			rc = rc ? rc : set( V, zero );

			// set U to Identity
			auto DiagU = alp::get_view< alp::view::diagonal >( U );
			rc = rc ? rc : alp::set( U, zero );
			rc = rc ? rc : alp::set( DiagU, one );
			// set V to Identity
			auto DiagV = alp::get_view< alp::view::diagonal >( V );
			rc = rc ? rc : alp::set( V, zero );
			rc = rc ? rc : alp::set( DiagV, one );

			if( n > m ) {
				auto UT = get_view< alp::view::transpose >( U );
				auto BT = get_view< alp::view::transpose >( B );
				auto VT = get_view< alp::view::transpose >( V );
				rc = rc ? rc : algorithms::svd_solve( VT, BT, UT, ring, minus, divide );
			} else {
				rc = rc ? rc : algorithms::svd_solve( U, B, V, ring, minus, divide );
			}

			//update S
			auto DiagS = alp::get_view< alp::view::diagonal >( S );
			auto DiagB = alp::get_view< alp::view::diagonal >( B );
			rc = rc ? rc : set( S, zero );
			rc = rc ? rc : set( DiagS, DiagB );

			return rc;
		}

	} // namespace algorithms
} // namespace alp
