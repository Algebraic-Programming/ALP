
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
 * Implements the various grb::config items for the grb::BSP1D backend.
 *
 * @author A. N. Yzelman
 * @date 5th of May, 2017
 */

#ifndef _H_GRB_BSP1D_CONFIG
#define _H_GRB_BSP1D_CONFIG

#include <assert.h>

#include <graphblas/reference/config.hpp>

// if not defined, we set the backend of the BSP1D implementation to the
// reference implementation
#ifndef _GRB_BSP1D_BACKEND
#define _GRB_BSP1D_BACKEND reference
#endif

namespace grb {

	/**
	 * \defgroup bsp1d The BSP1D backend implementation
	 *
	 * Groups all definitions and documentations corresponding to the #BSP1D
	 * backend.
	 * @{
	 */

	namespace config {

		/**
		 * Defaults for the BSP1D implementation
		 */
		template<>
		class IMPLEMENTATION< grb::Backend::BSP1D > {

		private:
			/**
			 * \a true if and only if \a mode was set. By default, value is \a false.
			 */
			static bool set;

			/**
			 * The selected mode. Only set if \a set is \a true.
			 */
			static grb::config::ALLOC_MODE mode;

			/** Attempts to automatically deduce the best value for \a mode. */
			static void deduce() noexcept;

		public:
			/**
			 * For private memory segments, which is the default, simply choose aligned
			 * allocations.
			 */
			static constexpr ALLOC_MODE defaultAllocMode() {
				return grb::config::ALLOC_MODE::ALIGNED;
			}

			/**
			 * For the BSP1D backend, a shared memory-segment should use interleaved
			 * alloc only if is running one process per compute node.
			 */
			static grb::config::ALLOC_MODE sharedAllocMode() noexcept;
		};

	} // namespace config

	/** @} */

} // namespace grb

#endif // end ``_H_GRB_BSP1D_CONFIG''
