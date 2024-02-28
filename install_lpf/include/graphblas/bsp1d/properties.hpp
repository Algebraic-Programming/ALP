
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
 * Gathers the properties of the BSP1D and hybrid backends.
 *
 * @author A. N. Yzelman
 * @date 5th of May 2017
 */

#ifndef _H_GRB_BSP1D_PROPERTIES
#define _H_GRB_BSP1D_PROPERTIES

#include <graphblas/base/properties.hpp>

namespace grb {

	/** No implementation notes. */
	template<>
	class Properties< BSP1D > {

		public:

			/** This property is inherited from the backend it depends on. */
			static constexpr const bool writableCaptured =
				Properties< _GRB_BSP1D_BACKEND >::writableCaptured;

			/**
			 * This implementation at present only supports blocking execution.
			 */
			static constexpr const bool isBlockingExecution = true;

			/**
			 * This implementation at present only supports blocking execution.
			 */
			static constexpr const bool isNonblockingExecution = false;

			static_assert( Properties< _GRB_BSP1D_BACKEND >::isBlockingExecution,
					"This implementation assumes blocking behaviour of the underlying "
					"process-local backend"
				);

	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_PROPERTIES''

