/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
#include <alp/algorithms/householder_qr.hpp>
#ifdef DEBUG
#include "../tests/utils/print_alp_containers.hpp"
#endif

// TEMPDISABLE should be removed in the final version
#define TEMPDISABLE


namespace alp {

	namespace algorithms {

		/**
		 * Calculate eigendecomposition of square matrix T
		 *        \f$T = Qdiag(d)Q^T\f$ where
		 *        \a T is real
		 *        \a Q is orthogonal (columns are eigenvectors).
		 *        \a d is vector containing eigenvalues.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type of minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[out] Q    output orthogonal matrix contaning eigenvectors
		 * @param[out] d    output vector containg eigenvalues
		 * @param[in]  T    input symmetric tridiagonal matrix
		 * @param[in]  ring A semiring for operations
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatA,
			typename MatQ,
			typename Vec,
			typename D = typename MatA::value_type,
			class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			class Minus = operators::subtract< D >,
			class Divide = operators::divide< D >
		>
		RC qr_eigensolver(
			MatA &A,
			MatQ &Q,
			Vec &d,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {
			(void) ring;
			(void) minus;
			(void) divide;

			const size_t max_it = 1.e+7;
			const D tol = 1.e-6;

			const Scalar< D > zero( ring.template getZero< D >() );
			const Scalar< D > one( ring.template getOne< D >() );

			RC rc = SUCCESS;

			rc = rc ? rc : alp::set( d, zero );

			const size_t n = nrows( A );

			alp::Matrix< D, structures::General > Atmp( n, n );
			rc = rc ? rc : alp::set( Atmp, zero );

			// auto A_diag = alp::get_view< alp::view::diagonal >( A );

			auto A_tmp_orig_view = alp::get_view< typename MatA::structure >( Atmp );

			auto A_tmp_diag = alp::get_view< alp::view::diagonal >( Atmp );

			auto A_tmp_supsquare = alp::get_view< alp::structures::Square >( Atmp, utils::range( 0, n - 1 ), utils::range( 1, n ) );
			auto A_tmp_supdiag = alp::get_view< alp::view::diagonal >( A_tmp_supsquare );

			auto A_tmp_subsquare = alp::get_view< alp::structures::Square >( Atmp, utils::range( 1, n ), utils::range( 0, n - 1 ) );
			auto A_tmp_subdiag = alp::get_view< alp::view::diagonal >( A_tmp_subsquare );

			rc = rc ? rc : alp::set( A_tmp_orig_view, A );
			rc = rc ? rc : alp::set( A_tmp_subdiag, A_tmp_supdiag );

// //#ifdef DEBUG
// 			print_matrix( "  A(input)     =  ", A );
// 			print_matrix( "  Atmp  =  ", Atmp );
// //#endif

			rc = rc ? rc : alp::set( Q, zero );
			auto Q_diag = alp::get_view< alp::view::diagonal >( Q );
			rc = rc ? rc : alp::set(
				Q_diag,
				one
			);

			alp::Matrix< D, structures::Orthogonal > qmat( n );
			alp::Matrix< D, structures::General > rmat( n, n );
			MatQ Q_tmp( n, n );

			size_t k1 = 0;
			size_t k2 = n;

			for( size_t i = 0; i < max_it; ++i ) {
// //#ifdef DEBUG
// 				print_vector( "  A_tmp_supdiag   ", A_tmp_supdiag );
// //#endif

				Scalar< D > sdiagnorm1( zero );
				auto sdiag1 = alp::get_view( A_tmp_supdiag, utils::range( k1, k1 + 1 ) );
				rc = rc ? rc : alp::norm2( sdiagnorm1, sdiag1, ring );
				if( std::abs( *sdiagnorm1 ) < tol ) {
					++k1;
				}
				if ( k1 >= k2 - 1 ){
					break;
				}

				Scalar< D > sdiagnorm2( zero );
				auto sdiag2 = alp::get_view( A_tmp_supdiag, utils::range( k2 - 2, k2 - 1 ) );
				rc = rc ? rc : alp::norm2( sdiagnorm2, sdiag2, ring );
				if( std::abs( *sdiagnorm2 ) < tol ) {
					--k2;
				}
				if ( k1 >= k2 - 1 ){
					break;
				}

				if( ( k2 - k1 ) != n ) {
					auto A_tmp_subprob = alp::get_view( Atmp, utils::range( k1, k2 ), utils::range( k1, k2 ) );
					MatQ qmat2( k2 - k1 );
					MatA A_sub_mat( k2 - k1 );
					Vec d_tmp( k2 - k1 );
					rc = rc ? rc : alp::set( A_sub_mat, zero );
					auto view_t1 = alp::get_view< typename MatA::structure >( A_tmp_subprob );
					rc = rc ? rc : alp::set( A_sub_mat, view_t1 );
					rc = rc ? rc : alp::set( d_tmp, zero );
// //#ifdef DEBUG
// 					print_matrix( "  Atmp   ", Atmp );
// 					print_matrix( "  A_tmp_subprob   ", A_tmp_subprob );
// 					print_matrix( "  A_sub_mat   ", A_sub_mat );
// //#endif
					rc = rc ? rc : alp::set( qmat2, zero );
					rc = rc ? rc : alp::algorithms::qr_eigensolver( A_sub_mat, qmat2, d_tmp );
// // #ifdef DEBUG
// 					std::cout << " d_tmp : \n";
// 					print_vector( "  ---> d_tmp   ", d_tmp );
// // #endif

					//Q[:,k1:k2]=Q[:,k1:k2].dot(q1)
					auto Q_update_view = alp::get_view< structures::OrthogonalColumns >( Q, utils::range( 0, n ), utils::range( k1, k2 ) );
					alp::Matrix< D, structures::OrthogonalColumns > Q_tmp2( n, k2 - k1 );
					rc = rc ? rc : alp::set( Q_tmp2, Q_update_view );
					rc = rc ? rc : alp::set( Q_update_view, zero );
					rc = rc ? rc : alp::mxm( Q_update_view, Q_tmp2, qmat2, ring );

					rc = rc ? rc : alp::set( A_tmp_subprob, zero );
					auto A_tmp_diag_update = alp::get_view< alp::view::diagonal >( A_tmp_subprob );
					rc = rc ? rc : alp::set( A_tmp_diag_update, d_tmp );

					break;
				} else {

					rc = rc ? rc : alp::set( qmat, zero );
					rc = rc ? rc : alp::set( rmat, zero );
					rc = rc ? rc : alp::algorithms::householder_qr( Atmp, qmat, rmat, ring );

					rc = rc ? rc : alp::set( Q_tmp, Q );
					rc = rc ? rc : alp::set( Q, zero );
					rc = rc ? rc : alp::mxm( Q, Q_tmp, qmat, ring );
					rc = rc ? rc : alp::set( Atmp, zero );
					rc = rc ? rc : alp::mxm( Atmp, rmat, qmat, ring );

				}

// //#ifdef DEBUG
// 				print_matrix( "  Atmp   ", Atmp );
// //#endif

//				if( i % ( n ) == 0 ) {
					Scalar< D > supdiagnorm( zero );
					rc = rc ? rc : alp::norm2( supdiagnorm, A_tmp_supdiag, ring );
					if( std::abs( *supdiagnorm ) < tol * tol ) {
						break;
					}
//				}
			}

			rc = rc ? rc : alp::set( d, A_tmp_diag );

			return rc;
		}
	} // namespace algorithms
} // namespace alp
