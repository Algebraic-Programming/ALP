
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

/**
 * @file
 *
 * Implements the 2-norm.
 *
 * @author A. N. Yzelman
 * @date 17th of March 2022
 *
 * \internal
 * Factored out of graphblas/blas1.hpp, promoted to a (simple) algorithm since
 * semiring structures are insufficient to capture <tt>sqrt</tt>.
 *
 * \todo Provide implementations of other norms.
 * \endinternal
 */

#ifndef _H_GRB_ALGORITHMS_NORM
#define _H_GRB_ALGORITHMS_NORM

#include <graphblas.hpp>

#include <cmath> // for std::sqrt


namespace grb {

	namespace algorithms {

		/**
		 * An alias of std::sqrt where the input and output types are templated
		 * separately.
		 *
		 * @tparam OutputType The output type of the square-root operation.
		 * @tparam InputType The input type of the square-root operation.
		 *
		 * @param[in] x The value to take the square root of.
		 *
		 * @returns The square root of \a x, cast to \a OutputType.
		 *
		 * Relies on the standard std::sqrt implementation. If this is not available
		 * for \a InputType, the use of this operation will result in a compile-time
		 * error.
		 *
		 * This operation is used as a default to the #norm2 algorithm, as well as a
		 * default to algorithms that depend on it.
		 */
		template< typename OutputType, typename InputType >
		OutputType std_sqrt( const InputType x ) {
			return( static_cast< OutputType >( std::sqrt( x ) ) );
		};

		/**
		 * Provides a generic implementation of the 2-norm computation.
		 *
		 * Proceeds by computing a dot-product on itself and then taking the square
		 * root of the result.
		 *
		 * This function is only available when the output type is floating point.
		 *
		 * For return codes, exception behaviour, performance semantics, template
		 * and non-template arguments, @see grb::dot.
		 *
		 * @param[out]  x   The 2-norm of \a y. The input value of \a x will be
		 *                  ignored.
		 * @param[in]   y   The vector to compute the norm of.
		 * @param[in] ring  The Semiring under which the 2-norm is to be computed.
		 * @param[in] sqrtX The square root function which operates on real data
		 *                  type, as both input and output. If not explicitly
		 *                  provided, the std::sqrt() is used.
		 */
		template<
			Descriptor descr = descriptors::no_operation, class Ring,
			typename InputType, typename OutputType,
			Backend backend, typename Coords
		>
		RC norm2( OutputType &x,
			const Vector< InputType, backend, Coords > &y,
			const Ring &ring = Ring(),
			const std::function< OutputType( OutputType ) > sqrtX =
				std_sqrt< OutputType, OutputType >,
			const typename std::enable_if<
				std::is_floating_point< OutputType >::value,
			void >::type * = nullptr
		) {
			InputType yyt = ring.template getZero< InputType >();

			RC ret = grb::dot< descr >(
				yyt, y, y, ring.getAdditiveMonoid(),
				grb::operators::conjugate_mul< InputType, InputType, InputType >()
			);
			if( ret == SUCCESS ) {
				Semiring<
					grb::operators::add< OutputType >, grb::operators::mul< OutputType >,
					grb::identities::zero, grb::identities::one
				> ring_rtype;
				ret = ret ? ret : grb::foldl(
					x,
					sqrtX( grb::utils::is_complex< InputType >::modulus( yyt ) ),
					ring_rtype.getAdditiveOperator()
				);
			}
			return ret;
		}
	}
}

#endif // end ``_H_GRB_ALGORITHMS_NORM''

