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
			typename D = double,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D > >
		RC householder_tridiag(
			Matrix< D, structures::Orthogonal, Dense > &Q,
			Matrix< D, structures::SymmetricTridiagonal, Dense > &T, // Need to be add this once alp -> alp is done
			const Matrix< D, structures::Symmetric, Dense > &H,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {

			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const size_t n = nrows( H );

			// Q = identity( n )
			rc = set( Q , structures::constant::I( n ) );

			// Out of place specification of the computation
			Matrix< D, structures::Symmetric, Dense > RR( n );
			rc = set( RR, H );

			for( size_t k = 0; k < n - 2; ++k ) {

				const size_t m = n - k - 1;

				// ===== Begin Computing v =====
				// v = H[ k + 1 : , k ]
				// alpha = norm( v ) * v[ 0 ] / norm( v[ 0 ] )
				// v = v - alpha * e1
				// v = v / norm ( v )
				Vector< D, structures::General, Dense > v;
				rc = set( v, get_view( RR, utils::range( k + 1, n ), k ) );

				Scalar< D > alpha;
				rc = norm2( alpha, v, ring );

				rc = eWiseLambda(
							[ &v, &alpha ]( const size_t i ) {
								if ( i == 0 ) {
									Scalar< D > norm_v0( std::abs( v[ i ] ) );
									foldl(alpha, v [ i ], ring.getMultiplicativeOperator() );
									foldl(alpha, norm_v0, divide );
									foldl(v [ i ], alpha, minus );
								}
							},
							v );

				Scalar< D > norm_v;
				rc = norm2( norm_v, v, ring );

				rc = foldl(v, norm_v, divide );
				// ===== End Computing v =====


				// ===== Calculate reflector Qk =====
				// Q_k = identity( n )
				Matrix< D, structures::Symmetric, Dense > Qk( m );
				rc = set( Qk, structures::constant::I( m ) );

				Matrix< D, structures::Symmetric, Dense > vvt( m );
				rc = set(vvt, zero );
				// vvt = v * v^T
				rc = outer( vvt, v, v, ring );

				// vvt = 2 * vvt
				rc = foldr( Scalar< D >( 2 ), vvt, ring.getMultiplicativeOperator() );

				// Qk = Qk - vvt ( expanded: I - 2 * vvt )
				rc = foldl( Qk, vvt, minus );
				// ===== End of Calculate reflector Qk ====

				// ===== Update R =====
				// Rk = Qk * Rk * Qk^T

				// get a view over RR (temporary of R)
				auto Rk = get_view( RR, range( k + 1, n ), range( k + 1, n ) );

				// QkRk = Qk * Rk
				Matrix< D, structures::Square, Dense > QkRk( m );
				rc = set( QkRk, zero );
				rc = mxm( QkRk, Qk, Rk, ring );

				// Rk = QkRk * QkT
				rc = set( Rk, zero );
				rc = mxm( Rk, QkRk, Qk, ring );
				// ===== End of Update R =====

				// ===== Update Q =====
				// Q = Q * conjugate( QkT )
				// a temporary for storing the mxm result
				Matrix< D, structures::Orthogonal, Dense > Qtmp( m, m );
				// a view over smaller portion of Q
				auto Qprim = get_view( Q, range( k + 1, n ), range( k + 1, n ) );

				// Qtmp = Qprim * QkT
				rc = set( Qtmp, zero );
				rc = mxm( Qtmp, Qprim, Qk, ring );

				// Qprim = Qtmp
				rc = set( Qprim, Qtmp );
				// ===== End of Update Q =====
			}

			// T = RR
			rc = set( T, get_view< structures::SymmetricTridiagonal > ( RR ) );

			return rc;
		}
	} // namespace algorithms
} // namespace alp
