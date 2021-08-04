
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

	/** \internal No implementation notes: a simple delegate yields correct behaviour. */
	template< typename IOType >
	RC clear( grb::Matrix< IOType, BSP1D > & A ) noexcept {
		return grb::clear( internal::getLocal( A ) );
	}

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

} // namespace grb

#endif
