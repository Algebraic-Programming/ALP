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

#include <iostream>
#include <sstream>

#include <alp.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include "../tests/utils/print_alp_containers.hpp"

namespace alp {

	namespace algorithms {

		/**
		 * @brief Computes Householder tridiagonalization \f$H = QTQ^T\f$
		 *        where \a H is real symmetric, \a T is symmetric tridiagonal, and
		 *        \a Q is orthogonal.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] Q    output orthogonal matrix such that H = Q T Q^T
		 * @param[out] T    output symmetric tridiagonal matrix such that H = Q T Q^T
		 * @param[in]  H    input symmetric matrix
		 * @param[in]  ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename D,
			typename SymmOrHermType,
			typename SymmOrHermTridiagonalType,
			typename OrthogonalType,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC householder_tridiag(
			Matrix< D, OrthogonalType, Dense > &Q,
			Matrix< D, SymmOrHermTridiagonalType, Dense > &T,
			Matrix< D, SymmOrHermType, Dense > &H,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {

			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );
			const size_t n = nrows( H );

			// Q = identity( n )
			rc = alp::set( Q, zero );
			auto Qdiag = alp::get_view< alp::view::diagonal >( Q );
			rc = rc ? rc : alp::set( Qdiag, one );
			if( rc != SUCCESS ) {
				std::cerr << " set( Q, I ) failed\n";
				return rc;
			}

			// Out of place specification of the computation
			Matrix< D, SymmOrHermType, Dense > RR( n );

			rc = set( RR, H );
			if( rc != SUCCESS ) {
				std::cerr << " set( RR, H ) failed\n";
				return rc;
			}
#ifdef DEBUG
			print_matrix( " << RR >> ", RR );
#endif

			// a temporary for storing the mxm result
			Matrix< D, OrthogonalType, Dense > Qtmp( n, n );

			for( size_t k = 0; k < n - 2; ++k ) {
#ifdef DEBUG
				std::string matname(" << RR(");
				matname = matname + std::to_string(k);
				matname = matname + std::string( ") >> ");
				print_matrix( matname , RR );
#endif

				const size_t m = n - k - 1;

				// ===== Begin Computing v =====
				// v = H[ k + 1 : , k ]
				// alpha = norm( v ) * v[ 0 ] / norm( v[ 0 ] )
				// v = v - alpha * e1
				// v = v / norm ( v )

				auto v_view = get_view( RR, k, utils::range( k + 1, n ) );
				Vector< D, structures::General, Dense > v( n - ( k + 1 ) );
				rc = set( v, v_view );
				if( rc != SUCCESS ) {
					std::cerr << " set( v, view ) failed\n";
					return rc;
				}

				Scalar< D > alpha( zero );
				rc = norm2( alpha, v, ring );
				if( rc != SUCCESS ) {
					std::cerr << " norm2( alpha, v, ring ) failed\n";
					return rc;
				}

				rc = eWiseLambda(
					[ &alpha, &ring, &divide, &minus ]( const size_t i, D &val ) {
						if ( i == 0 ) {
							Scalar< D > norm_v0( std::abs( val ) );
							Scalar< D > val_scalar( val );
							foldl( alpha, val_scalar, ring.getMultiplicativeOperator() );
							foldl( alpha, norm_v0, divide );
							foldl( val_scalar, alpha, minus );
							val = *val_scalar;
						}
					},
					v
				);
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, v ) failed\n";
					return rc;
				}

				Scalar< D > norm_v( zero );
				rc = norm2( norm_v, v, ring );
				if( rc != SUCCESS ) {
					std::cerr << " norm2( norm_v, v, ring ) failed\n";
					return rc;
				}

				rc = foldl(v, norm_v, divide );
#ifdef DEBUG
				print_vector( " v = ", v );
#endif
				// ===== End Computing v =====

				// ===== Calculate reflector Qk =====
				// Q_k = identity( n )
				Matrix< D, SymmOrHermType, Dense > Qk( n );
				rc = alp::set( Qk, zero );
				auto Qk_diag = alp::get_view< alp::view::diagonal >( Qk );
				rc = rc ? rc : alp::set( Qk_diag, one );

				// this part can be rewriten without temp matrix using functors
				Matrix< D, SymmOrHermType, Dense > vvt( m );

				rc = rc ? rc : set( vvt, outer( v, ring.getMultiplicativeOperator() ) );
				// vvt = 2 * vvt
				rc = rc ? rc : foldr( Scalar< D >( 2 ), vvt, ring.getMultiplicativeOperator() );


#ifdef DEBUG
				print_matrix( " vvt ", vvt );
#endif

				// Qk = Qk - vvt ( expanded: I - 2 * vvt )
				auto Qk_view = get_view< SymmOrHermType >( Qk, utils::range( k + 1, n ), utils::range( k + 1, n ) );
				if ( grb::utils::is_complex< D >::value ) {
					rc = rc ? rc : foldl( Qk_view, alp::get_view< alp::view::transpose >( vvt ), minus );
				} else {
					rc = rc ? rc : foldl( Qk_view, vvt, minus );
				}

#ifdef DEBUG
				print_matrix( " << Qk >> ", Qk );
#endif
				// ===== End of Calculate reflector Qk ====

				// ===== Update R =====
				// Rk = Qk * Rk * Qk

				// RRQk = RR * Qk
				Matrix< D, structures::Square, Dense > RRQk( n );
				rc = rc ? rc : set( RRQk, zero );
				rc = rc ? rc : mxm( RRQk, RR, Qk, ring );
				if( rc != SUCCESS ) {
					std::cerr << " mxm( RRQk, RR, Qk, ring ); failed\n";
					return rc;
				}
#ifdef DEBUG
				print_matrix( " << RR x Qk = >> ", RRQk );
#endif
				// RR = Qk * RRQk
				rc = rc ? rc : set( RR, zero );
				rc = rc ? rc : mxm( RR, Qk, RRQk, ring );

#ifdef DEBUG
				print_matrix( " << RR( updated ) >> ", RR );
#endif
				// ===== End of Update R =====

				// ===== Update Q =====
				// Q = Q * Qk

				// Qtmp = Q * Qk
				rc = rc ? rc : set( Qtmp, zero );
				rc = rc ? rc : mxm( Qtmp, Q, Qk, ring );

				// Q = Qtmp
				rc = rc ? rc : set( Q, Qtmp );
#ifdef DEBUG
				print_matrix( " << Q updated >> ", Q );
#endif
				// ===== End of Update Q =====
			}

			// T = RR

			rc = rc ? rc : set( T, get_view< SymmOrHermTridiagonalType > ( RR ) );
			return rc;
		}
	} // namespace algorithms
} // namespace alp
