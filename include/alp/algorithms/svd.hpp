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
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

namespace alp {

	namespace algorithms {


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
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			// c = abs(a) / sqrt(abs(a)**2 + abs(b)**2)
			// s = (a/abs(a)) * conjugate(b) / sqrt(abs(a)**2 + abs(b)**2)
			// return(array([[c,-conjugate(s)],[(s),c]]),r)
			Vector< D > c( 1 );
			Vector< D > s( 1 );
			Vector< D > d( 1 );
			Vector< D > aa( 1 );
			Vector< D > bb( 1 );
			rc = rc ? rc : alp::set( c, zero );
			rc = rc ? rc : alp::set( s, zero );
			rc = rc ? rc : alp::set( d, zero );
			rc = rc ? rc : alp::set( aa, zero );
			rc = rc ? rc : alp::set( bb, zero );
			auto a = get_view( v, utils::range( 0, 1 ) );
			auto b = get_view( v, utils::range( 1, 2 ) );
			rc = rc ? rc : alp::foldl( aa, a, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( aa, conjugate( a ), ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::foldl( bb, b, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( bb, conjugate( b ), ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::foldl( d, aa, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( d, bb, ring.getAdditiveOperator() );
			rc = rc ? rc : eWiseLambda(
				[  ]( const size_t i, D &val ) {
					(void) i;
					val = std::sqrt( val );
				},
				d
			);
			rc = rc ? rc : alp::set( c, a );
			rc = rc ? rc : alp::set( s, a );
			//
			rc = rc ? rc : eWiseLambda(
				[  ]( const size_t i, D &val ) {
					(void) i;
					val = std::abs( val );
				},
				c
			);
			rc = rc ? rc : alp::foldl( s, c, divide );
			rc = rc ? rc : alp::foldl( s, conjugate( b ), ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::foldl( s, d, divide );
			rc = rc ? rc : alp::foldl( c, d, divide );

			// return(array([[c,-conjugate(s)],[(s),c]]),r)
			auto G11 = get_view( G, 0, utils::range( 0, 1 ) );
			auto G12 = get_view( G, 0, utils::range( 1, 2 ) );
			auto G21 = get_view( G, 1, utils::range( 0, 1 ) );
			auto G22 = get_view( G, 1, utils::range( 1, 2 ) );
			rc = rc ? rc : alp::set( G11, c );
			rc = rc ? rc : alp::set( G12, conjugate( s ) );
			rc = rc ? rc : alp::foldl( G12, Scalar< D >( -1 ), ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::set( G21, s );
			rc = rc ? rc : alp::set( G22, c );
// #ifdef DEBUG
// 			print_vector( "Givens: v " , v );
// 			print_vector( "Givens: d " , d );
// 			print_vector( "Givens: c " , c );
// 			print_vector( "Givens: s " , s );
// 			print_matrix( "Givens: G " , G );
// #endif

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
				structures::is_a< StruU, structures::Orthogonal >::value &&
				is_semiring< Ring >::value &&
				is_operator< Minus >::value &&
				is_operator< Divide >::value
			> * = nullptr
		>
		RC gk_svd_step(
			Matrix< D, StruU, Dense, ViewU, ImfRU, ImfCU > &V,
			Matrix< D, StruB, Dense, ViewB, ImfRB, ImfCB > &B,
			Matrix< D, StruU, Dense, ViewU, ImfRU, ImfCU > &U,
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
			// calcualte eigenvalue llambda of BEndSquare
			// which is closer to t22
			auto BEnd2x2 = get_view( B, utils::range( k - 2, k ), utils::range( k - 2, k ) );

			Matrix< D, structures::Square, Dense > BEndSquare( 2, 2 );
			rc = rc ? rc : alp::set( BEndSquare, zero );
			auto BEnd2x2T = get_view< alp::view::transpose >( BEnd2x2 );
			auto BEnd2x2T_star = conjugate( BEnd2x2T );
			rc = rc ? rc : mxm( BEndSquare, BEnd2x2T_star, BEnd2x2, ring );


			auto t11 = get_view( BEndSquare, 0, utils::range( 0, 1 ) );
			auto t12 = get_view( BEndSquare, 0, utils::range( 1, 2 ) );
			auto t22 = get_view( BEndSquare, 1, utils::range( 1, 2 ) );
#ifdef DEBUG
			if( rc != SUCCESS ) {
				std::cerr << " mxmfailed\n";
				return rc;
			}
			print_matrix( "BEnd2x2 " , BEnd2x2 );
			print_matrix( "BEndSquare " , BEndSquare );
			print_vector( "t11 " , t11 );
			print_vector( "t12 " , t12 );
			print_vector( "t22 " , t22 );
#endif
			Vector< D > aa( 1 );
			rc = rc ? rc : alp::set( aa, t11 );
			rc = rc ? rc : alp::foldl( aa, t22, ring.getAdditiveOperator() );
			rc = rc ? rc : alp::foldl( aa, alp::Scalar< D >( 2 ), divide );

			Vector< D > bb( 1 );
			rc = rc ? rc : alp::set( bb, t11 );
			rc = rc ? rc : alp::foldl( bb, t22, minus );
			rc = rc ? rc : alp::foldl( bb, alp::Scalar< D >( 2 ), divide );
			rc = rc ? rc : alp::foldl( bb, conjugate( bb ), ring.getMultiplicativeOperator() );

			Vector< D > cc( 1 );
			rc = rc ? rc : alp::set( cc, conjugate( t12 ) );
			rc = rc ? rc : alp::foldl( cc, t12, ring.getMultiplicativeOperator() );
			rc = rc ? rc : alp::foldl( bb, cc, ring.getAdditiveOperator() );

			rc = rc ? rc : eWiseLambda(
				[ ]( const size_t i, D &val ) {
					(void) i;
					val = std::sqrt( val );
				},
				bb
			);

			alp::Scalar< D > t11scal( zero );
			alp::Scalar< D > t22scal( zero );
			rc = rc ? rc : alp::foldl( t11scal, t11, ring.getAdditiveMonoid() );
			rc = rc ? rc : alp::foldl( t22scal, t22, ring.getAdditiveMonoid() );

			if ( std::real( *t11scal ) > std::real( *t22scal ) ) {
				rc = rc ? rc : alp::foldl( aa, bb, minus );
			} else {
				rc = rc ? rc : alp::foldl( aa, bb, ring.getAdditiveOperator() );
			}
			// end of get lambda
#ifdef DEBUG
			// final eigenvalue-shift
			print_vector( "lambda " , aa );
#endif

			Vector< D > rotvec( 2 );
			auto Brow = get_view( B, 0, utils::range( 0, 2 ) );
			rc = rc ? rc : alp::set( rotvec, Brow );
			auto rotvec0 = get_view( rotvec, utils::range( 0, 1 ) );
			rc = rc ? rc : alp::foldl( rotvec0, aa, minus );

			Matrix< D, structures::Square, Dense > G( 2, 2 );
			rc = rc ? rc : alp::set( G, zero );
			rc = rc ? rc : Givens( G, rotvec );


			return rc;
		}

	} // namespace algorithms
} // namespace alp
