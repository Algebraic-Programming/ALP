
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
 * Provides the initialisation and finalisation routines for the ascend
 * backend.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_INIT
#define _H_GRB_ASCEND_INIT

#include <graphblas/base/init.hpp>
#include <graphblas/utils/DMapper.hpp>


namespace grb {

	template<>
	RC init< ascend >( const size_t, const size_t, void * const );

	template<>
	RC finalize< ascend >();

	namespace internal {

		/** Internal state of the ascend backend. */
		class ASCEND {

			friend RC init< ascend >( const size_t, const size_t, void * const );

			private:

			public:

		};

	}

} // namespace grb

#endif //``end _H_GRB_ASCEND_INIT''

