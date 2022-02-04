
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

/*
 * @author A. N. Yzelman
 * @date 14th of January 2022
 */

#ifndef _H_GRB_DENSEREF_BLAS3
#define _H_GRB_DENSEREF_BLAS3

#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>

#include "io.hpp"
#include "matrix.hpp"

namespace grb {
	namespace internal {

		/**
		 * \internal general mxm implementation that all mxm variants refer to
		 */
		template<
			bool allow_void,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			class Operator, class Monoid
		>
		RC mxm_generic( Matrix< OutputType, reference_dense > &C,
			const Matrix< InputType1, reference_dense > &A,
			const Matrix< InputType2, reference_dense > &B,
			const Operator &oper,
			const Monoid &monoid,
			const MulMonoid &mulMonoid,
			const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value && !
				grb::is_object< InputType2 >::value &&
				grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL
		) {
			(void)oper;
			(void)monoid;
			(void)mulMonoid;
			static_assert( allow_void ||
				( !(
					std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value
				) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)"
			);

#ifdef _DEBUG
			std::cout << "In grb::internal::mxm_generic (reference_dense, unmasked)\n";
#endif

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = grb::nrows( A );
			const size_t k = grb::ncols( A );
			const size_t k_B = grb::nrows( B );
			const size_t n_B = grb::ncols( B );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto A_raw = grb::getRaw( A );
			const auto B_raw = grb::getRaw( B );
			auto C_raw = grb::getRaw( C );

			std::cout << "Multiplying dense matrices.\n";

			for( size_t row = 0; row < m; ++row ) {
				for( size_t col = 0; col < n; ++col ) {
					C_raw[ row * k + col] = monoid.template getIdentity< OutputType >();
					for( size_t i = 0; i < k; ++ i ) {
						OutputType temp = monoid.template getIdentity< OutputType >();
						(void)grb::apply( temp, A_raw[ row * k + i ], B_raw[ i * n_B + col ], oper );
						(void)grb::foldl( C_raw[ row * k + col], temp, monoid.getOperator() );
					}
				}
			}
			grb::internal::setInitialized( C, true );
			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * \internal grb::mxm, semiring version.
	 * Dispatches to internal::mxm_generic
	 */
	template< typename OutputType, typename InputType1, typename InputType2, class Semiring >
	RC mxm( Matrix< OutputType, reference_dense > & C,
		const Matrix< InputType1, reference_dense > & A,
		const Matrix< InputType2, reference_dense > & B,
		const Semiring & ring = Semiring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value,
			void >::type * const = NULL ) {

#ifdef _DEBUG
		std::cout << "In grb::mxm (reference_dense, unmasked, semiring)\n";
#endif

		return internal::mxm_generic< true >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid() );
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_BLAS3''

