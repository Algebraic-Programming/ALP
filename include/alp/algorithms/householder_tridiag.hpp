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

#include "../tests/utils/print_alp_containers.hpp"


#define DEBUG

#define TEMPDISABLE

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
			// Matrix< D, structures::Orthogonal, Dense > &Q,
			Matrix< D, structures::Square, Dense > &Q,
			// Matrix< D, SymmetricTridiagonal, Dense > &T, // Need to be add this once alp -> alp is done
			Matrix< D, structures::Symmetric, Dense > &T, // Need to be add this once alp -> alp is done
			const Matrix< D, structures::Symmetric, Dense > &H,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide() ) {

			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );
			const size_t n = nrows( H );

			// Q = identity( n )
#ifdef TEMPDISABLE
			rc = set( Q, zero );
			rc = rc ? rc : alp::eWiseLambda(
				[ &one ]( const size_t i, const size_t j, D &val ) {
					if ( i == j ) {
						val = *one;
					}
				},
				Q
			);
#else
			rc = set( Q, alp::structures::constant::I< D >( n ) );
#endif
			if( rc != SUCCESS ) {
				std::cerr << " set( Q, I ) failed\n";
				return rc;
			}
#ifdef DEBUG
			print_matrix( " << Q(0) >> ", Q );
#endif

			// Out of place specification of the computation
			Matrix< D, structures::Symmetric, Dense > RR( n );

			// auto RR = get_view< view::transpose >( R0 ); 
			rc = set( RR, H );
			if( rc != SUCCESS ) {
				std::cerr << " set( RR, H ) failed\n";
				return rc;
			}
#ifdef DEBUG
			print_matrix( " << RR >> ", RR );
#endif

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

				// Vector< D, structures::General, Dense > v( n - ( k + 1 ) );
				// rc = set( v, get_view( RR, utils::range( k + 1, n ), k ) );
				auto v_view = get_view( RR, k, utils::range( k + 1, n ) );
				Vector< D, structures::General, Dense > v( n - ( k + 1 ) );
				rc = set( v, v_view );
				if( rc != SUCCESS ) {
					std::cerr << " set( v, view ) failed\n";
					return rc;
				}
// #ifdef DEBUG
// 				print_vector( " -- v_view -- ", v_view );
// 				print_vector( " -- v(0) -- ", v );
// #endif

				Scalar< D > alpha( zero );
				rc = norm2( alpha, v, ring );
				if( rc != SUCCESS ) {
					std::cerr << " norm2( alpha, v, ring ) failed\n";
					return rc;
				}
// #ifdef DEBUG
// 				std::cout << " >>>> alpha = " << *alpha << "\n";
// #endif

				rc = eWiseLambda(
					[ &alpha, &ring, &divide, &minus ]( const size_t i, D &val ) {
						if ( i == 0 ) {
							Scalar< D > norm_v0( std::abs( val ) );
							internal::foldl( *alpha, val, ring.getMultiplicativeOperator() );
							internal::foldl( *alpha, *norm_v0, divide );
							internal::foldl( val, *alpha, minus );
						}
					},
					v
				);
				if( rc != SUCCESS ) {
					std::cerr << " eWiseLambda( lambda, v ) failed\n";
					return rc;
				}
// #ifdef DEBUG
// 				std::cout << " <<<< = " << *alpha << "\n";
// 				print_vector( " -- v(1) -- ", v );
// #endif
				Scalar< D > norm_v( zero );
				rc = norm2( norm_v, v, ring );
				if( rc != SUCCESS ) {
					std::cerr << " norm2( norm_v, v, ring ) failed\n";
					return rc;
				}
// #ifdef DEBUG
// 				print_vector( " -- v(2) -- ", v );
// 				std::cout << " norm_v = " << *norm_v << "\n";
// #endif
				rc = foldl(v, norm_v, divide );
#ifdef DEBUG
				print_vector( " -- v(3) -- ", v );
#endif
				// ===== End Computing v =====

				// ===== Calculate reflector Qk =====
				// Q_k = identity( n )
				Matrix< D, structures::Symmetric, Dense > Qk( n );
#ifdef TEMPDISABLE
				rc = set( Qk, zero );
				rc = rc ? rc : alp::eWiseLambda(
					[ &one ]( const size_t i, const size_t j, D &val ) {
						if ( i == j ) {
							val = *one;
						}
					},
					Qk
				);
#else
				rc = set( Qk, structures::constant::I< D >( n ) );
#endif

#ifdef DEBUG
				print_matrix( " << Qk(0) >> ", Qk );
#endif

				// this part might be rewriten without temp matrix using functors
				Matrix< D, structures::Symmetric, Dense > vvt( m );

				// vvt = v * v^T
#ifndef TEMPDISABLE
				rc = set(vvt, outer( v, ring.getMultiplicativeOperator() ) );
#else
				rc = set(vvt, zero );
				auto vvt_outer = outer( v, ring.getMultiplicativeOperator());
				rc = rc ? rc : alp::eWiseLambda(
					[ &vvt_outer ]( const size_t i, const size_t j, D &val ) {
						val = internal::access( vvt_outer, internal::getStorageIndex( vvt_outer, i, j ) );
					},
					vvt
				);
#endif


#ifdef DEBUG
				print_matrix( " << vvt(0) >> ", vvt );
#endif

				// vvt = 2 * vvt
#ifndef TEMPDISABLE
				// disabled until fold(s) on Matrices are functional
				rc = foldr( Scalar< D >( 2 ), vvt, ring.getMultiplicativeOperator() );
#else
				rc = rc ? rc : alp::eWiseLambda(
					[ ]( const size_t i, const size_t j, D &val ) {
						(void) i;
						(void) j;
						val = 2 * val;
					},
					vvt
				);
#endif

#ifdef DEBUG
				print_matrix( " << vvt >> ", vvt );
#endif

				// Qk = Qk - vvt ( expanded: I - 2 * vvt )
#ifndef TEMPDISABLE
				auto Qk_view = get_view( Qk, utils::range( k + 1, n ), utils::range( k + 1, n ) );
				rc = foldl( Qk_view, vvt, minus );
#else
				auto Qk_view = get_view( Qk, utils::range( k + 1, n ), utils::range( k + 1, n ) );
				rc = rc ? rc : alp::eWiseLambda(
					[ &vvt, &minus ]( const size_t i, const size_t j, D &val ) {
						internal::foldl(
							val,
							internal::access( vvt, internal::getStorageIndex( vvt, i, j ) ),
							minus
						);
					},
					Qk_view
				);
#endif
#ifdef DEBUG
				print_matrix( " << Qk >> ", Qk );
#endif
				// ===== End of Calculate reflector Qk ====


				// ===== Update R =====
				// Rk = Qk * Rk * Qk^T

#ifdef DEBUG
				print_matrix( " << RR >> ", RR );
#endif
				// QkRR = Qk * RR
				Matrix< D, structures::Square, Dense > QkRR( n );
				rc = set( QkRR, zero );
#ifdef DEBUG
				print_matrix( " << QkRR = 0 >> ", QkRR );
#endif
				rc = mxm( QkRR, RR, Qk, ring );
#ifdef DEBUG
				print_matrix( " << QkRR >> ", QkRR );
#endif

				// RR = QkRR * QkT
				rc = set( RR, zero );
				rc = mxm( RR, Qk, QkRR, ring );

#ifdef DEBUG
				print_matrix( " << RR( updated ) >> ", RR );
#endif
				// ===== End of Update R =====

				// ===== Update Q =====
				// Q = Q * conjugate( QkT )
				// a temporary for storing the mxm result
#ifndef TEMPDISABLE
				Matrix< D, structures::Orthogonal, Dense > Qtmp( n, n );
#else
				Matrix< D, structures::Square, Dense > Qtmp( n, n );
#endif


#ifdef DEBUG
				print_matrix( " << Q in >> ", Q );
#endif
				// Qtmp = Q * QkT
				rc = set( Qtmp, zero );
				rc = mxm( Qtmp, Q, Qk, ring );

#ifdef DEBUG
				print_matrix( " << Qtmp >> ", Qtmp );
#endif

				// Q = Qtmp
				rc = set( Q, Qtmp );
#ifdef DEBUG
				print_matrix( " << Q out >> ", Q );
#endif
				// ===== End of Update Q =====
			}

			// T = RR
#ifndef TEMPDISABLE
			rc = set( T, get_view< SymmetricTridiagonal > ( RR ) );
#else
			rc = set( T, get_view< structures::Symmetric > ( RR ) );
#endif
			return rc;
		}
	} // namespace algorithms
} // namespace alp
