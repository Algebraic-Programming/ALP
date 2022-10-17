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

#define DEBUG
#include <alp.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include "../tests/utils/print_alp_containers.hpp"
// TEMPDISABLE enables workarounds for non-implemented features
// should be removed in the final version
#define TEMPDISABLE

namespace alp {

	namespace algorithms {

		/**
		 * @brief Computes Householder QR decomposition of general matrix \f$H = QR\f$
		 *        where \a H is general (complex or real),
		 *        \a R is upper triangular (if H is not square,
		 *        R is of the same shape with zeros below diagonal), and
		 *        \a Q is orthogonal.
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
			typename OrthogonalType,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC householder_qr(
			Matrix< D, GeneralType, Dense > &H,
			Matrix< D, OrthogonalType, Dense > &Q,
			Matrix< D, GeneralType, Dense > &R,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			RC rc = SUCCESS;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );
			const size_t n = nrows( H );
			const size_t m = ncols( H );

#ifdef DEBUG
			std::cout << " n, m= " << n << ", " << m << "\n";
#endif

			// Q = identity( n )
			rc = alp::set( Q, zero );
			auto Qdiag = alp::get_view< alp::view::diagonal >( Q );
			rc = rc ? rc : alp::set( Qdiag, one );
			if( rc != SUCCESS ) {
				std::cerr << " set( Q, I ) failed\n";
				return rc;
			}

			// Out of place specification of the computation
			Matrix< D, GeneralType, Dense > RR( n, m );

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

			for( size_t k = 0; k < std::min( n-1, m ); ++k ) {
#ifdef DEBUG
				std::string matname(" << RR(");
				matname = matname + std::to_string(k);
				matname = matname + std::string( ") >> ");
				print_matrix( matname , RR );
#endif

 				//const size_t m = n - k - 1;

				// ===== Begin Computing v =====
				// v = H[ k + 1 : , k ]
				// alpha = norm( v ) * v[ 0 ] / norm( v[ 0 ] )
				// v = v - alpha * e1
				// v = v / norm ( v )

				auto v_view = get_view( RR, utils::range( k, n ), k );
				Vector< D, GeneralType, Dense > v( n - k );
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
				typedef typename std::conditional<
					grb::utils::is_complex< D >::value,
					structures::Hermitian,
					structures::Symmetric
				>::type SymmOrHerm;
				Matrix<	D, SymmOrHerm, Dense > Qk( n );
				//Matrix<	D, GeneralType, Dense > Qk( n, n );
				rc = alp::set( Qk, zero );
				auto Qk_diag = alp::get_view< alp::view::diagonal >( Qk );
				rc = rc ? rc : alp::set( Qk_diag, one );
#ifdef DEBUG
				print_matrix( " << Qk(0) >> ", Qk );
#endif

				// this part can be rewriten without temp matrix using functors
				Matrix<	D, SymmOrHerm, Dense > vvt( n - k );
				//Matrix<	D, GeneralType, Dense > vvt( n - k, n - k );

				rc = rc ? rc : set( vvt, outer( v, ring.getMultiplicativeOperator() ) );
				// rc = rc ? rc : set( vvt, zero );
				// rc = rc ? rc : alp::eWiseLambda(
				// 	[ &v ]( const size_t i, const size_t j, D &val ) {
				// 		//val = v[ i ] * std::conj( v[ j ] ) ;
				// 		val = v[ i ] * v[ j ] ;
				// 	},
				// 	vvt
				// );
#ifdef DEBUG
				print_matrix( " vvt(0) ", vvt );
#endif
				// vvt = 2 * vvt
				rc = rc ? rc : foldr( Scalar< D >( 2 ), vvt, ring.getMultiplicativeOperator() );

#ifdef DEBUG
				print_matrix( " vvt(1) ", vvt );
#endif

				// Qk = Qk - vvt ( expanded: I - 2 * vvt )
				auto Qk_view = get_view< SymmOrHerm >(
					//auto Qk_view = get_view< GeneralType >(
					Qk,
					utils::range( k, n ),
					utils::range( k, n )
				);
				rc = rc ? rc : foldl( Qk_view, vvt, minus );

#ifdef DEBUG
				print_matrix( " << Qk(1) >> ", Qk );
#endif

// #ifdef TEMPDISABLE
// 				// workaround untill foldl( Gen, Symm, op ); is resolved
// 				// i.e. nonmatching structures reduction
// 				if ( !grb::utils::is_complex< D >::value ) {
// 					std::cout << "**** updating lover tringaular part ****\n";
// 					rc = rc ? rc : alp::eWiseLambda(
// 						[ &vvt, &minus ]( const size_t i, const size_t j, D &val ) {
// 							if ( j < i ) {
// 								internal::foldl(
// 									val,
// 									internal::access( vvt, internal::getStorageIndex( vvt, i, j ) ),
// 									minus
// 								);
// 							}
// 						},
// 						Qk_view
// 					);
// 				}
// #endif


#ifdef DEBUG
				print_matrix( " << Qk >> ", Qk );
#endif
				// ===== End of Calculate reflector Qk ====

				// ===== Update RR =====
				// RR = Qk * RR

				// QkRR = Qk * RR
				Matrix< D, GeneralType, Dense > QkRR( n, m );
				rc = rc ? rc : set( QkRR, zero );
				rc = rc ? rc : mxm( QkRR, Qk, RR, ring );
				if( rc != SUCCESS ) {
					std::cerr << " mxm( QkRR, Qk, RR, ring ); failed\n";
					return rc;
				}
#ifdef DEBUG
				print_matrix( " << Qk x RR  >> ", QkRR );
#endif
				rc = rc ? rc : set( RR, QkRR );

#ifdef DEBUG
				print_matrix( " << RR( updated ) >> ", RR );
#endif
				// ===== End of Update R =====

				// ===== Update Q =====
				// Q = Q * conjugate(transpose(Qk))

				// Qtmp = Q * conjugate(transpose(Qk))
				rc = rc ? rc : set( Qtmp, zero );
 				rc = rc ? rc : mxm(
					Qtmp,
					Q,
					conjugate( alp::get_view< alp::view::transpose >( Qk ) ),
					ring
				);

				// Q = Qtmp
				rc = rc ? rc : set( Q, Qtmp );
#ifdef DEBUG
				print_matrix( " << Q updated >> ", Q );
#endif
				// ===== End of Update Q =====
			}

			// R = RR
			rc = rc ? rc : set( R, RR );
			return rc;
		}
	} // namespace algorithms
} // namespace alp
