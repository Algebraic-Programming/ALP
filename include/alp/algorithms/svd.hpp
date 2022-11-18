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


		// Golub-Kahan SVD step
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

			//auto Bsubmat = get_view( B, utils::range( 1, k ), utils::range( 0, k - 1 ) );
			//auto Bsubdiag = get_view< alp::view::diagonal >( Bsubmat );
			//rc = rc ? rc : alp::set( Bsubdiag, zero );

			// get lambda
			// calcualte eigenvalue llambda of BEndSquare
			// which is closer to t22
			Matrix< D, structures::Square, Dense > BSquare( k, k );
			rc = rc ? rc : alp::set( BSquare, zero );

			auto Bselect = get_view( B, utils::range( 0, k ), utils::range( 0, k ) );

			auto BT = get_view< alp::view::transpose >( Bselect );
			auto BT_star = conjugate( BT );
			rc = rc ? rc : mxm( BSquare, BT_star, Bselect, ring );
			auto BEndSquare = get_view( BSquare, utils::range( k - 2, k ), utils::range( k - 2, k ) );

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
			//rc = rc ? rc : alp::foldl( cc, t12, ring.getMultiplicativeMonoid() );

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
			auto Brow = get_view( BSquare, 0, utils::range( 0, 2 ) );
			rc = rc ? rc : alp::set( rotvec, Brow );

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
				auto Vstrip = get_view< structures::General >( V, utils::range( i, i + 2 ), utils::range( 0, n ) );
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


		// Docs
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
			const double tol = 1.e-12;
			const size_t maxit = 30;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( B );
			const size_t n = ncols( B );
			const size_t k = std::min( m, n );

			auto Bsupsquare =  get_view( B, utils::range( 0, k - 1 ) , utils::range( 1, k ) );
			auto superdiagonal = get_view< alp::view::diagonal >( Bsupsquare );

			size_t i1 = 0;
			size_t i2 = k;
#ifdef DEBUG
			std::cout << "---> (init) i1, i2 = " << i1 << ", " << i2 << "\n";
#endif

			rc = rc ? rc : algorithms::householder_bidiag( U, B, V, ring, minus, divide );
			//repeat while superdiagonal is not zero
			for( size_t i = 0; i < maxit; ++i ) {
				//TODO check for zeroes in diagonal, if any do Givens rotatations
				//         to move the zero from diagonal to superdiagonal


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
#ifdef DEBUG
				print_vector( "superdiagonal ", superdiagonal );
				print_matrix( "B ", B );
				std::cout << "---> i1, i2 = " << i1 << ", " << i2 << "\n";
#endif

				auto Bview = get_view( B, utils::range( i1, i2 ), utils::range( i1, i2 ) );
				Matrix< D, structures::Orthogonal, Dense > Utmp( nrows( Bview ), nrows( Bview ) );
				Matrix< D, structures::Orthogonal, Dense > Vtmp( ncols( Bview ), ncols( Bview ) );
				rc = rc ? rc : set( Utmp, zero );
				rc = rc ? rc : set( Vtmp, zero );
				// set Utmp to Identity
				auto DiagUtmp = alp::get_view< alp::view::diagonal >( Utmp );
				rc = rc ? rc : alp::set( Utmp, zero );
				rc = rc ? rc : alp::set( DiagUtmp, one );
				// set Vtmp to Identity
				auto DiagVtmp = alp::get_view< alp::view::diagonal >( Vtmp );
				rc = rc ? rc : alp::set( Vtmp, zero );
				rc = rc ? rc : alp::set( DiagVtmp, one );

				// sdv step
				rc = rc ? rc : algorithms::gk_svd_step( Utmp, Bview, Vtmp, ring, minus, divide );

				// rc = rc ? rc : algorithms::gk_svd_step( Uview, Bview, Vview, ring, minus, divide );

				Matrix< D, structures::Orthogonal, Dense > UtmpX( nrows( U ), ncols( U ) );
				Matrix< D, structures::Orthogonal, Dense > VtmpX( nrows( V ), ncols( V ) );
				{
					auto DiagUtmpX = alp::get_view< alp::view::diagonal >( UtmpX );
					rc = rc ? rc : alp::set( UtmpX, zero );
					rc = rc ? rc : alp::set( DiagUtmpX, one );
					auto DiagVtmpX = alp::get_view< alp::view::diagonal >( VtmpX );
					rc = rc ? rc : alp::set( VtmpX, zero );
					rc = rc ? rc : alp::set( DiagVtmpX, one );
					auto SUtmpX = get_view( UtmpX, utils::range( i1, i2 ), utils::range( i1, i2 ) );
					auto SVtmpX = get_view( VtmpX, utils::range( i1, i2 ), utils::range( i1, i2 ) );
					rc = rc ? rc : alp::set( SUtmpX, Utmp );
					rc = rc ? rc : alp::set( SVtmpX, Vtmp );
				}


				Matrix< D, structures::Orthogonal, Dense > Utmp2( nrows( U ), ncols( U ) );
				Matrix< D, structures::Orthogonal, Dense > Vtmp2( nrows( V ), ncols( V ) );
				rc = rc ? rc : set( Utmp2, U );
				rc = rc ? rc : set( Vtmp2, V );
				rc = rc ? rc : set( U, zero );
				rc = rc ? rc : set( V, zero );
				rc = rc ? rc : mxm( U, Utmp2, UtmpX, ring );
				rc = rc ? rc : mxm( V, VtmpX, Vtmp2, ring );


				// check convergence
				Scalar< D > sup_diag_norm( zero );
				rc = rc ? rc : alp::norm2( sup_diag_norm, superdiagonal, ring );
#ifdef DEBUG
				std::cout << " norm( superdiagonal B ) = " << *sup_diag_norm << "\n";
#endif
				if( std::abs( *sup_diag_norm ) < tol ) {
					break ;
				}
			}

			// Rotate diagonal elements in complex plane
			auto BSquare = alp::get_view( B, utils::range( 0, k ), utils::range( 0, k ) );
			auto DiagBview = alp::get_view< alp::view::diagonal >( BSquare );
			Matrix< D, structures::Square, Dense > RotMat( nrows( B ) );
			auto DiagRotMat = alp::get_view< alp::view::diagonal >( RotMat );
			rc = rc ? rc : alp::set( RotMat, zero );
			rc = rc ? rc : alp::set( DiagRotMat, one );
			auto d1 = alp::get_view( DiagRotMat, utils::range( 0, k ) );
			rc = rc ? rc : alp::set( d1, DiagBview );

			rc = rc ? rc : eWiseLambda(
				[ ]( const size_t i, D &val ) {
					val = val/std::abs(val);
				},
				d1
			);

			Matrix< D, structures::Orthogonal, Dense > UtmpRot( nrows( U ) );
			rc = rc ? rc : alp::set( UtmpRot, U );
			rc = rc ? rc : alp::set( U, zero );
#ifdef DEBUG
			print_matrix( "---->  RotMat(in) ", RotMat );
			print_matrix( "---->  UtmpRot(in) ", UtmpRot );
			print_matrix( "---->  U(in) ", U );
#endif
			rc = rc ? rc : mxm( U, UtmpRot, RotMat, ring );
#ifdef DEBUG
			print_matrix( "---->  U(out) ", U );
#endif

			Matrix< D, structures::General, Dense > BtmpRot( nrows( B ), ncols( B ) );
			rc = rc ? rc : alp::set( BtmpRot, B );
			rc = rc ? rc : alp::set( B, zero );
			rc = rc ? rc : eWiseLambda(
				[ &one ]( const size_t i, D &val ) {
					val = (*one) / val; //make it foldl
				},
				d1
			);
#ifdef DEBUG
			print_matrix( " RotMat(1) ", RotMat );
#endif
			rc = rc ? rc : mxm( B, RotMat, BtmpRot, ring );

			return rc;
		}



		// Docs
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
		RC svd(
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
			//const size_t k = std::min( m, n );

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

			return rc;
		}

	} // namespace algorithms
} // namespace alp
