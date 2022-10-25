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
		 * @brief Solves linear system Ax=b
		 *        where A is UpperTriangular matrix, b is given RHS vector
		 *        and x is the solution.
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
			typename D = double,
			typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
			typename Minus = operators::subtract< D >,
			typename Divide = operators::divide< D > >
		RC backsubstitution(
			Matrix< D, structures::UpperTriangular, Dense > &A,
			Vector< D > &x,
			Vector< D > &b,
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide()
		) {

			RC rc = SUCCESS;

			const size_t n = nrows( A );

			for( size_t k = 0; k < n ; ++k ) {
				Scalar< D > alpha( ring.template getZero< D >() );
				const size_t i = n - k - 1;
				//x[i]=(b[i]-A[i,i:].dot(x[i:]))/A[i,i]
 				auto A_i  = get_view( A, i, utils::range( i, n ) );
				auto A_ii  = get_view( A, i, utils::range( i, i + 1 ) );
				auto x_i  = get_view( x, utils::range( i, i + 1 ) );
				auto b_i  = get_view( b, utils::range( i, i + 1 ) );
				auto x_i_n  = get_view( x, utils::range( i, n ) );
				rc = rc ? rc : alp::dot( alpha, A_i, alp::conjugate( x_i_n ), ring );
				rc = rc ? rc : alp::set( x_i, b_i );
				rc = rc ? rc : alp::foldl( x_i, alpha, minus );
 				rc = rc ? rc : alp::set( alpha, Scalar< D >( ring.template getZero< D >() ) );
 				rc = rc ? rc : alp::foldl( alpha, A_ii, ring.getAdditiveMonoid() );
 				rc = rc ? rc : alp::foldl( x_i, alpha, divide );
			}

			assert( rc == SUCCESS );
			return rc;
		}

	} // namespace algorithms
} // namespace alp
