
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
 * Collects the hyperdags backend properties
 *
 * @author A. N. Yzelman
 * @date 23rd of March, 2023
 */

#ifndef _H_GRB_HYPERDAGS_PROPERTIES
#define _H_GRB_HYPERDAGS_PROPERTIES

#include <graphblas/base/properties.hpp>
#include <graphblas/hyperdags/config.hpp>


namespace grb {

	/** All properties are inherited from the underlying backend. */
	template<>
	class Properties< hyperdags > {

		public:

			static constexpr const bool writableCaptured =
				Properties< _GRB_WITH_HYPERDAGS_USING >::writableCaptured;

			static constexpr const bool isBlockingExecution =
				Properties< _GRB_WITH_HYPERDAGS_USING >::isBlockingExecution;

			static constexpr const bool isNonblockingExecution =
				Properties< _GRB_WITH_HYPERDAGS_USING >::isNonblockingExecution;

	};

} // namespace grb

#endif // end `_H_GRB_HYPERDAGS_PROPERTIES

