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

#include <alp/algorithms/cholesky.hpp>

namespace alp {

	namespace algorithms {

		/**
		 * Computes the Cholesky decomposition U^HU = H of a real symmetric
		 * positive definite (SPD) (or complex Hermitian positive definite)
		 * matrix H where \a U is upper triangular, and ^H is transpose in
		 * the real case and transpose + complex conjugate in the complex case.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @param[out] L    output lower triangular matrix
		 * @param[in]  H    input real symmetric positive definite matrix
		 *                  or complex hermitian positive definite matrix
		 * @param[in]  ring The semiring used in the computation
		 * @return RC        SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatH,
			typename D = typename MatH::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			std::enable_if_t<
				is_matrix< MatH >::value &&
				(
					(
						!grb::utils::is_complex< D >::value &&
						structures::is_a< typename MatH::structure, structures::SymmetricPositiveDefinite >::value
					) || (
						grb::utils::is_complex< D >::value &&
						structures::is_a< typename MatH::structure, structures::HermitianPositiveDefinite >::value
					)
				) &&
				is_semiring< Ring >::value
			> * = nullptr
		>
		RC symherm_posdef_inverse(
			MatH &Hinv,
			const MatH &H,
			const Ring &ring = Ring()
		) {
			RC rc = SUCCESS;

			const alp::Scalar< D > zero( ring.template getZero< D >() );
			const alp::Scalar< D > one( ring.template getOne< D >() );

			if( nrows( Hinv ) != nrows( H ) ) {
				std::cerr << "Incompatible sizes in symherm_posdef_inverse.\n";
				return FAILED;
			}

			const size_t N = nrows( H );

			alp::Matrix< D, structures::UpperTriangular, Dense > U( N );

			rc = rc ? rc : alp::set( U, zero );

			rc = rc ? rc : algorithms::cholesky_uptr( U, H, ring );
#ifdef DEBUG
			print_matrix( "  U ", U );
#endif
			// H = U^H U
			// H^-1 = U^-1 U^H-1
			alp::Matrix< D, structures::UpperTriangular, Dense > Uinv( N );
			rc = rc ? rc : alp::set( Uinv, zero );
			auto Uinvdiag = alp::get_view< alp::view::diagonal >( Uinv );
			auto UinvT = alp::get_view< alp::view::transpose >( Uinv );
			rc = rc ? rc : alp::set( Uinvdiag, one );
			auto UT = alp::get_view< alp::view::transpose >( U );
			for( size_t i = 0; i < N; ++i ){
				auto x = alp::get_view( UinvT, utils::range( i, N ), i );
				auto UT_submatview = alp::get_view( UT, utils::range( i, N ), utils::range( i, N ) );
				rc = rc ? rc : alp::algorithms::forwardsubstitution( UT_submatview, x, ring );
			}
#ifdef DEBUG
			print_matrix( "  Uinv  ", Uinv );
#endif
			rc = rc ? rc : alp::set( Hinv, zero );
			// conjugate(linv.T).dot(linv)
			auto UinvTvstar = conjugate( UinvT );
			rc = rc ? rc : alp::mxm( Hinv, Uinv, UinvTvstar, ring );
#ifdef DEBUG
			print_matrix( "  Hinv  ", Hinv );
#endif
			return rc;
		}

	} // namespace algorithms
} // namespace alp
