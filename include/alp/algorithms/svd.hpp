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

			//get lambda
			auto Bend2x2 = get_view( B, utils::range( k - 2, k ), utils::range( k - 2, k ) );

			Matrix< D, structures::Square, Dense > BendSquare( 2, 2 );
			rc = rc ? rc : alp::set( BendSquare, zero );
			auto Bend2x2T = get_view< alp::view::transpose >( Bend2x2 );
			auto Bend2x2T_star = conjugate( Bend2x2T );
			rc = rc ? rc : mxm( BendSquare, Bend2x2T_star, Bend2x2, ring );

			// calcualte eigenvalue llambda of BendSquare
			// which is closer to t22
			auto t11 = get_view( BendSquare, 0, utils::range( 0, 1 ) );
			auto t12 = get_view( BendSquare, 0, utils::range( 1, 2 ) );
			auto t22 = get_view( BendSquare, 1, utils::range( 1, 2 ) );
#ifdef DEBUG
			if( rc != SUCCESS ) {
				std::cerr << " mxmfailed\n";
				return rc;
			}
			print_matrix( "Bend2x2 " , Bend2x2 );
			print_matrix( "BendSquare " , BendSquare );
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
#ifdef DEBUG
			// final eigenvalue-shift
			print_vector( "lambda " , aa );
#endif

			


			return rc;
		}

	} // namespace algorithms
} // namespace alp
