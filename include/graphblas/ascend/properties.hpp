
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
 * Collects the Ascend backend properties.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_PROPERTIES
#define _H_GRB_ASCEND_PROPERTIES

#include <graphblas/base/properties.hpp>


namespace grb {

	/** No implementation notes. */
	template<>
	class Properties< ascend > {

		public:

			/**
			 * This is a shared-memory parallel implementation and therefore captured
			 * scalars cannot be written to without causing data races.
			 */
			static constexpr const bool writableCaptured = false;

			/** This is a nonblocking backend. */
			static constexpr const bool isBlockingExecution = false;

			/** This is a nonblocking backend. */
			static constexpr const bool isNonblockingExecution = true;

	};

} // namespace grb

#endif // end `_H_GRB_ASCEND_PROPERTIES

