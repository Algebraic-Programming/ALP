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

#include <cmath>
#include <iostream>
#include <iomanip>

#include <alp.hpp>
#ifdef DEBUG
#include "../../../tests/utils/print_alp_containers.hpp"
#endif

namespace alp {

	namespace algorithms {

		/**
		 * Solves linear system Ax=b
		 * where A is LowerTriangular matrix, b is given RHS vector
		 * and x is the solution.
		 *
		 * @tparam D        Data element type
		 * @tparam Ring     Type of the semiring used in the computation
		 * @tparam Minus    Type minus operator used in the computation
		 * @tparam Divide   Type of divide operator used in the computation
		 * @param[in]  A    input upper trinagular matrix
		 * @param[in]  b    input RHS vector
		 * @param[out] x    solution vector
		 * @param[in]  ring The semiring used in the computation
		 * @return RC       SUCCESS if the execution was correct
		 *
		 */
		template<
			typename MatA,
			typename Vecx,
			typename Vecb,
			typename D = typename MatA::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D >,
			std::enable_if_t<
				is_vector< Vecx >::value &&
				is_vector< Vecb >::value &&
				is_matrix< MatA >::value &&
				structures::is_a< typename  MatA::structure, structures::LowerTriangular >::value
			> * = nullptr
		>
		RC forwardsubstitution(
			MatA &A,
			Vecx &x,
			Vecb &b,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {

			RC rc = SUCCESS;

			if( ( nrows( A ) != size( x ) ) || ( size( b ) != size( x ) ) ) {
				std::cerr << "Incompatible sizes in trsv.\n";
				return FAILED;
			}

			const size_t n = nrows( A );

			for( size_t i = 0; i < n ; ++i ) {
				Scalar< D > alpha( ring.template getZero< D >() );
				auto A_i = get_view( A, i, utils::range( 0, i ) );
				auto A_ii = get_view( A, i, utils::range( i, i + 1 ) );
				auto x_i = get_view( x, utils::range( i, i + 1 ) );
				auto b_i = get_view( b, utils::range( i, i + 1 ) );
				auto x_0_i = get_view( x, utils::range( 0, i ) );
				rc = rc ? rc : alp::dot( alpha, A_i, alp::conjugate( x_0_i ), ring );
				rc = rc ? rc : alp::set( x_i, b_i );
				rc = rc ? rc : alp::foldl( x_i, alpha, minus );
 				rc = rc ? rc : alp::set( alpha, Scalar< D >( ring.template getZero< D >() ) );
 				rc = rc ? rc : alp::foldl( alpha, A_ii, ring.getAdditiveMonoid() );
 				rc = rc ? rc : alp::foldl( x_i, alpha, divide );
			}

			return rc;
		}

		template<
			typename MatA,
			typename MatX,
			typename MatB,
			typename D = typename MatA::value_type,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D >,
			std::enable_if_t<
				is_matrix< MatA >::value &&
				is_matrix< MatX >::value &&
				is_matrix< MatB >::value &&
				structures::is_a< typename  MatA::structure, structures::LowerTriangular >::value &&
				structures::is_a< typename  MatX::structure, typename MatB::structure >::value
			> * = nullptr
		>
		RC forwardsubstitution(
			MatA &A,
			MatX &X,
			MatB &B,
			const Ring &ring = Ring(),
			const Minus &minus = Minus(),
			const Divide &divide = Divide()
		) {

			RC rc = SUCCESS;

			if (
				( nrows( X ) != nrows( B ) ) ||
				( ncols( X ) != ncols( B ) ) ||
				( ncols( A ) != nrows( X ) )
			) {
				std::cerr << "Incompatible sizes in trsm.\n";
				return FAILED;
			}

			const size_t m = nrows( X );
			const size_t n = ncols( X );

			for( size_t i = 0; i < n ; ++i ) {
				auto x = get_view( X, utils::range( 0, m ), i );
				auto b = get_view( B, utils::range( 0, m ), i );
				rc = rc ? rc : algorithms::forwardsubstitution( A, x, b, ring, minus, divide );
			}

			return rc;
		}

	} // namespace algorithms
} // namespace alp
