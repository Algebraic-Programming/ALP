
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
 */

#ifndef _H_GRB_BSP1D_BLAS3
#define _H_GRB_BSP1D_BLAS3

#include <graphblas/backends.hpp>
#include <graphblas/base/blas3.hpp>

#include "matrix.hpp"

namespace grb {

	/** \internal No implementation details; simply delegates */
	template< Descriptor descr = descriptors::no_operation, typename DataType1, typename DataType2 >
	RC set( Matrix< DataType1, BSP1D > & out, const Matrix< DataType2, BSP1D > & in ) noexcept {
		RC ret = grb::set< descr >( internal::getLocal( out ), internal::getLocal( in ) );
		/*(void) collectives< BSP1D >::allreduce<
			descriptors::no_casting,
			operators::any_or< RC >
		>( ret );*/ // <-- WARNING: if we allow ONCE as a mode for level-3 primitives,
		            //              we need to be wary of local allocation errors
		return ret;
	}

	/** \internal Simply delegates to process-local backend. */
	template< Descriptor descr = descriptors::no_operation, typename DataType1, typename DataType2, typename DataType3 >
	RC set( Matrix< DataType1, BSP1D > & out, const Matrix< DataType2, BSP1D > & mask, const DataType3 & val ) noexcept {
		RC ret = grb::set< descr >( internal::getLocal( out ), internal::getLocal( mask ), val );
		/*(void) collectives<>::allreduce<
			descriptors::no_casting,
			operators::any_or< RC >
		>( ret );*/ // <-- WARNING: if we allow ONCE as a mode for level-3 primitives,
		            //              we need to be wary of local allocation errors
		return ret;
	}

	/** \internal Simply delegates to process-local backend */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class MulMonoid >
	RC eWiseApply( Matrix< OutputType, BSP1D > &C,
		const Matrix< InputType1, BSP1D > &A,
		const Matrix< InputType2, BSP1D > &B,
		const MulMonoid &mul,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< MulMonoid >::value,
		void >::type * const = NULL
	) {
		return eWiseApply< descr >( internal::getLocal( C ), internal::getLocal( A ), internal::getLocal( B ), mul, phase );
	}

	/** \internal Simply delegates to process-local backend */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class Operator >
	RC eWiseApply( Matrix< OutputType, BSP1D > &C,
		const Matrix< InputType1, BSP1D > &A,
		const Matrix< InputType2, BSP1D > &B,
		const Operator &op,
		const PHASE phase = NUMERICAL,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value && !
			grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = NULL
	) {
		return eWiseApply< descr >( internal::getLocal( C ), internal::getLocal( A ), internal::getLocal( B ), op, phase );
	}

} // namespace grb

#endif

