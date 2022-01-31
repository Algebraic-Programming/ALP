
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
 * @author D. G. Spampinato
 * @date 16th of November 2021
 */

#if ! defined _H_GRB_MLIR_IO
#define _H_GRB_MLIR_IO

#include <graphblas/base/io.hpp>

namespace grb {

	/**
	 * \defgroup IO Data Ingestion
	 * @{
	 */

	/**
	 * Calls the other #buildMatrixUnique variant.
	 * @see grb::buildMatrixUnique for the user-level specification.
	 */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename fwd_iterator >
	RC buildMatrixUnique( Matrix< InputType, mlir > & A, fwd_iterator start, const fwd_iterator end, const IOMode mode ) {
		// parallel or sequential mode are equivalent for now
		assert( mode == PARALLEL || mode == SEQUENTIAL );
#ifdef NDEBUG
		(void)mode;
#endif
		return A.template buildMatrixUnique< descr >( start, end );
	}

	/** @} */

} // namespace grb

#endif // end ``_H_GRB_MLIR_IO
