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

		/**
		 * @brief Computes Householder LU decomposition of general matrix \f$H = LU\f$
		 *        where \a H is general (complex or real),
		 *        \a L lower triangular,
		 *        \a U is upper triangular.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] Q    output orthogonal matrix such that H = Q T Q^T
		 * @param[out] R    output same shape as H with zeros below diagonal
		 * @param[in]  H    input general matrix
		 * @param[in]  ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D,
			typename GeneralType,
			typename GenView,
			typename GenImfR,
			typename GenImfC,
			typename UType,
			typename UView,
			typename UImfR,
			typename UImfC,
			typename LType,
			typename LView,
			typename LImfR,
			typename LImfC,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC householder_lu(
			Matrix< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > &H,
			Matrix< D, LType, alp::Dense, LView, LImfR, LImfC > &L,
			Matrix< D, UType, alp::Dense, UView, UImfR, UImfC > &U,
			Vector< size_t > &p,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			const size_t m = nrows( H );
			const size_t n = ncols( H );
			const size_t k = std::min( n, m );

			// check sizes

#ifdef DEBUG
			std::cout << " n, k, m = " << n << ", "  << k << ", " << m << "\n";
#endif

			// L = identity( n )
			auto Ldiag = alp::get_view< alp::view::diagonal >( L );
			rc = rc ? rc : alp::set( Ldiag, one );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( L, I ) failed\n";
				return rc;
			}

			// Out of place specification of the computation
			Matrix< D, GeneralType, alp::Dense > HWork( m, n );
			rc = rc ? rc : alp::set( HWork, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( HWork, H ) failed\n";
				return rc;
			}

			Vector< D, GeneralType, alp::Dense > PivotVec( n );
			rc = rc ? rc : alp::set( PivotVec, zero );

			for( size_t k = 0; k < std::min( n, m ); ++k ) {
				// =====   algorithm  =====
				// a = H[ k, k ]
				// v = H[ k + 1 : , k ]
				// w = H[ k, k + 1 : ]
				// Ak = H[ k + 1 :, k + 1 : ]
				// v = v / a
				// Ak = Ak - outer(v,w)
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
					Vector< size_t, GeneralType, alp::Dense > ptmp( 1 );
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


			//save the result
			for( size_t k = 0; k < std::min( n, m ); ++k ) {
				auto Hcol = alp::get_view( HWork, utils::range( k + 1, m ), k );
				auto Lcol = alp::get_view( L, utils::range( k + 1, m ), k );
				rc = rc ? rc : alp::set( Lcol, Hcol );

				auto Urow = alp::get_view( U, k, utils::range( k, n ) );
				auto Hrow = alp::get_view( HWork, k, utils::range( k, n ) );
				rc = rc ? rc : alp::set( Urow, Hrow );
			}

			return rc;

		}
	} // namespace algorithms
} // namespace alp
