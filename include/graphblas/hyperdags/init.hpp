
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
 * Provides initialisers for the HyperDAGs backend
 *
 * @author A. N. Yzelman
 * @date 31st of January, 2022
 */

#ifndef _H_GRB_HYPERDAGS_INIT
#define _H_GRB_HYPERDAGS_INIT

#include <graphblas/hyperdags/hyperdags.hpp>


namespace grb {

	namespace internal {

		namespace hyperdags {

			/** Singleton generator instance. */
			extern HyperDAGGenerator generator;

		}

	}

	template<>
	RC init< hyperdags >( const size_t, const size_t, void * const );

	template<>
	RC finalize< hyperdags >();

} // end namespace grb

#endif

