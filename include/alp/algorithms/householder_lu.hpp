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


			alp::Vector< size_t > no_permutation_vec( n );
			{
				// tmp data
				std::vector< size_t > vi( n );
				std::iota( std::begin( vi ), std::end( vi ), 0 );
				rc = rc ? rc : alp::buildVector( no_permutation_vec, vi.begin(), vi.end() );
			}

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

			rc = alp::set( HWork, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( HWork, H ) failed\n";
				return rc;
			}
#ifdef DEBUG
			print_matrix( " << HWork >> ", HWork );
#endif

			for( size_t k = 0; k < std::min( n, m ); ++k ) {
 				//const size_t m = n - k - 1;

				// ===== Begin Computing v =====
				// a = H[ k, k ]
				// v = H[ k + 1 : , k ]
				// w = H[ k, k + 1 : ]


// 				//TODO
// //#define _NO_DISABLE_PERMUTE
// #ifdef _NO_DISABLE_PERMUTE
// 				//
// 				auto h_col_view = alp::get_view( HWork, utils::range( 0, m ), k );
// 				// auto p_view = alp::get_view( p, utils::range( k, m ) );
// 				auto h_col_perm_view = alp::get_view< alp::structures::General >( h_col_view, p ); //use in sort

// 				auto p_view = alp::get_view( p, utils::range( k, m ) );
// 				// get current permutations from p
// 				std::vector< size_t > vj( m - k );
// 				for( size_t ll = 0; ll < m - k ; ++ll ) {
// 					vj[ ll ] = p_view[ ll ];
// 				}

// #ifdef DEBUG
// 				auto p1 = alp::get_view< alp::structures::General >( h_col_view, p );
// 				print_vector( " h_col_perm_view(in) ", p1 );
// 				std::cout << " vj(in) = [ ";
// 				for( size_t ll = 0; ll < m - k ; ++ll ) {
// 					std::cout << vj[ ll ] << ", ";
// 				}
// 				std::cout << "]\n";
// #endif

// 				// std::sort(
// 				// 	vj.begin(),
// 				// 	vj.end(),
// 				// 	[ &h_col_perm_view ]( const size_t &a, const size_t &b ) {
// 				// 		return (
// 				// 			std::abs( h_col_perm_view[ a ] ) > std::abs( h_col_perm_view[ b ] )
// 				// 		);
// 				// 	}
// 				// );
// 				for( size_t ll = 0; ll < m - k; ++ll ) {
// 					p_view[ ll ] = vj[ ll ];
// 				}
// #ifdef DEBUG
// 				std::cout << " vj(out) = [ ";
// 				for( size_t ll = 0; ll < m - k ; ++ll ) {
// 					std::cout << vj[ ll ] << ", ";
// 				}
// 				std::cout << "]\n";
// 				auto p2 = alp::get_view< alp::structures::General >( h_col_view, p );
// 				print_vector( " h_col_perm_view(in) ", p2 );
// #endif

// #endif //_NO_DISABLE_PERMUTE


				auto a_view = alp::get_view( HWork, utils::range( k, k + 1 ), k );
				auto v_view = alp::get_view( HWork, utils::range( k + 1, m ), k );
				auto w_view = alp::get_view( HWork, k, utils::range( k + 1, n ) );
				auto Ak_view = alp::get_view( HWork, utils::range( k + 1, m ), utils::range( k + 1, n ) );

				Scalar< D > alpha( zero );
				rc = rc ? rc : alp::foldl( alpha, a_view, ring.getAdditiveMonoid() );

				rc = rc ? rc : alp::foldl( v_view, alpha, divide );
#ifdef DEBUG
				std::string matname( " << Ak_view(" );
				matname = matname + std::to_string( k );
				matname = matname + std::string( ") >> " );
				print_matrix( matname, Ak_view );

				print_vector( " v  ", v_view );
				print_vector( " w  ", w_view );
				print_vector( " a_view  ", a_view );
				std::cout << " alpha = " << *alpha << "\n";
#endif

				auto w_view_star = conjugate( w_view );
				auto Reflector = alp::outer( v_view, w_view_star, ring.getMultiplicativeOperator() );
#ifdef DEBUG
				print_matrix( " R ", Reflector );
#endif
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
