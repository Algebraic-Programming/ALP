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
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

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
			typename GenView,
			typename GenImfR,
			typename GenImfC,
			typename OrthogonalType,
			typename OrthogonalView,
			typename OrthogonalImfR,
			typename OrthogonalImfC,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC householder_qr(
			Matrix< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > &H,
			Matrix< D, OrthogonalType, alp::Dense, OrthogonalView, OrthogonalImfR, OrthogonalImfC > &Q,
			Matrix< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > &R,
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
				std::cerr << " alp::set( Q, I ) failed\n";
				return rc;
			}

			// Out of place specification of the computation
			Matrix< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > RR( n, m );

			rc = alp::set( RR, H );
			if( rc != SUCCESS ) {
				std::cerr << " alp::set( RR, H ) failed\n";
				return rc;
			}
#ifdef DEBUG
			print_matrix( " << RR >> ", RR );
#endif

			// a temporary for storing the alp::mxm result
			Matrix< D, OrthogonalType, alp::Dense, OrthogonalView, OrthogonalImfR, OrthogonalImfC > Qtmp( n, n );

			for( size_t k = 0; k < std::min( n-1, m ); ++k ) {
#ifdef DEBUG
				std::string matname( " << RR(" );
				matname = matname + std::to_string( k );
				matname = matname + std::string( ") >> " );
				print_matrix( matname, RR );
#endif

 				//const size_t m = n - k - 1;

				// ===== Begin Computing v =====
				// v = H[ k + 1 : , k ]
				// alpha = norm( v ) * v[ 0 ] / norm( v[ 0 ] )
				// v = v - alpha * e1
				// v = v / norm ( v )

				auto v_view = alp::get_view( RR, utils::range( k, n ), k );
				Vector< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > v( n - k );
				rc = alp::set( v, v_view );
				if( rc != SUCCESS ) {
					std::cerr << " alp::set( v, view ) failed\n";
					return rc;
				}

				Scalar< D > alpha( zero );
				rc = alp::norm2( alpha, v, ring );
				if( rc != SUCCESS ) {
					std::cerr << " alp::norm2( alpha, v, ring ) failed\n";
					return rc;
				}

				rc = alp::eWiseLambda(
					[ &alpha, &ring, &divide, &minus ]( const size_t i, D &val ) {
						if ( i == 0 ) {
							Scalar< D > norm_v0( std::abs( val ) );
							Scalar< D > val_scalar( val );
							alp::foldl( alpha, val_scalar, ring.getMultiplicativeOperator() );
							alp::foldl( alpha, norm_v0, divide );
							alp::foldl( val_scalar, alpha, minus );
							val = *val_scalar;
						}
					},
					v
				);
				if( rc != SUCCESS ) {
					std::cerr << " alp::eWiseLambda( lambda, v ) failed\n";
					return rc;
				}

				Scalar< D > norm_v( zero );
				rc = alp::norm2( norm_v, v, ring );
				if( rc != SUCCESS ) {
					std::cerr << " alp::norm2( norm_v, v, ring ) failed\n";
					return rc;
				}

				rc = alp::foldl( v, norm_v, divide );
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
				Matrix<	D, SymmOrHerm, alp::Dense > Qk( n );
				rc = alp::set( Qk, zero );
				auto Qk_diag = alp::get_view< alp::view::diagonal >( Qk );
				rc = rc ? rc : alp::set( Qk_diag, one );

				// this part can be rewriten without temp matrix using functors
				Matrix<	D, SymmOrHerm, alp::Dense > vvt( n - k );

				rc = rc ? rc : alp::set( vvt, alp::outer( v, ring.getMultiplicativeOperator() ) );
				rc = rc ? rc : alp::foldr( Scalar< D >( 2 ), vvt, ring.getMultiplicativeOperator() );

				// Qk = Qk - vvt ( expanded: I - 2 * vvt )
				auto Qk_view = alp::get_view< SymmOrHerm >(
					//auto Qk_view = alp::get_view< GeneralType >(
					Qk,
					utils::range( k, n ),
					utils::range( k, n )
				);
				rc = rc ? rc : alp::foldl( Qk_view, vvt, minus );

#ifdef DEBUG
				print_matrix( " << Qk >> ", Qk );
#endif
				// ===== End of Calculate reflector Qk ====

				// ===== Update RR =====
				// RR = Qk * RR

				// QkRR = Qk * RR
				Matrix< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > QkRR( n, m );
				rc = rc ? rc : alp::set( QkRR, zero );
				rc = rc ? rc : alp::mxm( QkRR, Qk, RR, ring );
				if( rc != SUCCESS ) {
					std::cerr << " alp::mxm( QkRR, Qk, RR, ring ); failed\n";
					return rc;
				}
#ifdef DEBUG
				print_matrix( " << Qk x RR  >> ", QkRR );
#endif
				rc = rc ? rc : alp::set( RR, QkRR );

#ifdef DEBUG
				print_matrix( " << RR( updated ) >> ", RR );
#endif
				// ===== End of Update R =====

				// ===== Update Q =====
				// Q = Q * conjugate(transpose(Qk))

				// Qtmp = Q * conjugate(transpose(Qk))
				rc = rc ? rc : alp::set( Qtmp, zero );
				if( grb::utils::is_complex< D >::value ) {
					rc = rc ? rc : alp::mxm(
						Qtmp,
						Q,
						alp::conjugate( alp::get_view< alp::view::transpose >( Qk ) ),
						ring
					);
				} else {
					rc = rc ? rc : alp::mxm( Qtmp, Q, Qk, ring );
				}

				// Q = Qtmp
				rc = rc ? rc : alp::set( Q, Qtmp );
#ifdef DEBUG
				print_matrix( " << Q updated >> ", Q );
#endif
				// ===== End of Update Q =====
			}

			// R = RR
			rc = rc ? rc : alp::set( R, RR );
			return rc;
		}
	} // namespace algorithms
} // namespace alp
