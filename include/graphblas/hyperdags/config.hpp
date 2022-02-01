
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
 * Contains the configuration parameters for the HyperDAGs backend.
 *
 * @author A. N. Yzelman
 * @date 31st of January 2022.
 */

#ifndef _H_GRB_HYPERDAGS_CONFIG
#define _H_GRB_HYPERDAGS_CONFIG

#include <graphblas/config.hpp>

#ifndef _GRB_WITH_HYPERDAGS_USING
 #error "_GRB_WITH_HYPERDAGS_USING must be defined"
#endif


namespace grb {

	namespace config {

		template<>
		class IMPLEMENTATION< hyperdags > {

			public:

				// propagate the defaults of the underlying backend

				static constexpr ALLOC_MODE defaultAllocMode() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::defaultAllocMode();
				}

				static constexpr ALLOC_MODE sharedAllocMode() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::sharedAllocMode();
				}

				static constexpr Backend coordinatesBackend() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::coordinatesBackend();
				}

		};

	}

} // end namespace grb

#endif

