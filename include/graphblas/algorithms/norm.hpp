
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

		template< typename OutputType, typename InputType >
		OutputType std_sqrt( InputType x ) {
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
		 * @param[out] x The 2-norm of \a y. The input value of \a x will be ignored.
		 * @param[in]  y The vector to compute the norm of.
		 * @param[in] ring The Semiring under which the 2-norm is to be computed.
		 *
		 * \warning This function computes \a x out-of-place. This is contrary to
		 *          standard ALP/GraphBLAS functions that are always in-place.
		 *
		 * \warning A \a ring is not sufficient for computing a two-norm. This
		 *          implementation assumes the standard <tt>sqrt</tt> function
		 *          must be applied on the result of a dot-product of \a y with
		 *          itself under the supplied semiring.
		 *
		 * \todo Make sqrt an argument to this function.
		 */
		template<
			Descriptor descr = descriptors::no_operation, class Ring,
			typename InputType, typename OutputType,
			Backend backend, typename Coords
		>
		RC norm2( OutputType &x,
			const Vector< InputType, backend, Coords > &y,
			const Ring &ring = Ring(),
			const std::function< OutputType( InputType ) > sqrtX = std_sqrt< OutputType, InputType >,
			const typename std::enable_if<
				std::is_floating_point< OutputType >::value,
			void >::type * const = nullptr
		) {
			RC ret = grb::dot< descr >( x, y, y, ring );
			if( ret == SUCCESS ) {
				x = sqrtX( x );
			}
			return ret;
		}

	}
}

#endif // end ``_H_GRB_ALGORITHMS_NORM''

