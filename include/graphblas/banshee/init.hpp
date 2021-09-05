
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

#if ! defined _H_GRB_BANSHEE_INIT
#define _H_GRB_BANSHEE_INIT

#include <graphblas/base/init.hpp>

namespace grb {

	namespace internal {
		extern size_t * __restrict__ privateSizetOMP;
	}

	/**
	 * This function completes in \f$ \Theta(1) \f$, moves \f$ \Theta(1) \f$ data,
	 * does not allocate nor free any memory, and does not make any system calls.
	 *
	 * This implementation does not support multiple user processes.
	 *
	 * @see grb::init for the user-level specification.
	 */
	template<>
	RC init< banshee >( const size_t, const size_t, void * const );

	/**
	 * This function completes in \f$ \Theta(1) \f$, moves \f$ \Theta(1) \f$ data,
	 * does not allocate nor free any memory, and does not make any system calls.
	 *
	 * @see grb::finalize() for the user-level specification.
	 */
	template<>
	RC finalize< banshee >();

} // namespace grb

#endif //``end _H_GRB_BANSHEE_INIT''
