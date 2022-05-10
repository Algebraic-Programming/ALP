
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
 * Contains the configuration parameters for the BSP1D backend
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
 #pragma message "_GRB_BSP1D_BACKEND was not set-- auto-selecting reference"
 #define _GRB_BSP1D_BACKEND reference
#endif


namespace grb {

	namespace config {

		/**
		 * \defgroup bsp1dConfig BSP1D backend configuration
		 * \ingroup config
		 *
		 * All configuration parameters for the #BSP1D and #hybrid backends.
		 *
		 * @{
		 */

		/**
		 * This class collects configuration parameters that are specific to the
		 * #grb::BSP1D and #grb::hybrid backends
		 *
		 * \ingroup bsp1d
		 */
		template<>
		class IMPLEMENTATION< BSP1D > {

			private:

				/**
				 * \internal
				 * \a true if and only if \a mode was set. By default, value is \a false.
				 * \endinternal
				 */
				static bool set;

				/**
				 * \internal
				 * The selected mode. Only set if \a set is \a true.
				 * \endinternal
				 */
				static grb::config::ALLOC_MODE mode;

				/**
				 * \internal
				 * Attempts to automatically deduce the best value for \a mode.
				 * \endinternal
				 */
				static void deduce() noexcept;


			public:

				/**
				 * @returns The default allocation strategy for private memory segments.
				 */
				static constexpr ALLOC_MODE defaultAllocMode() {
					return grb::config::ALLOC_MODE::ALIGNED;
				}

				/**
				 * \internal
				 * Whether the backend has vector capacities always fixed to their
				 * defaults.
				 * \endinternal
				 */
				static constexpr bool fixedVectorCapacities() {
					return IMPLEMENTATION< _GRB_BSP1D_BACKEND >::fixedVectorCapacities();
				}

				/**
				 * @returns The default allocation strategy for shared memory regions.
				 *
				 * By default, for the BSP1D backend, a shared memory-segment should use
				 * interleaved alloc only if is running one process per compute node. This
				 * implies a run-time component to this function, which is why for this
				 * backend this function is \em not <tt>constexpr</tt>.
				 *
				 * \warning This function does assume that the number of processes does not
				 *          change over the life time of a single application.
				 *
				 * \note While the above may seem a reasonably safe assumption, the use of
				 *       the launcher in #MANUAL mode may, in fact, make this a realistic
				 *       issue that could be encountered. In such cases the deduction should
				 *       be re-initiated. If you encounter this problem, please report it so
				 *       that such a fix can be implemented.
				 */
				static grb::config::ALLOC_MODE sharedAllocMode() noexcept;

				/**
				 * \internal
				 * Select the coordinates backend of the selected process-local backend.
				 * \endinternal
				 */
				static constexpr Backend coordinatesBackend() {
					return IMPLEMENTATION< _GRB_BSP1D_BACKEND >::coordinatesBackend();
				}

				/**
				 * The selected backend may perform nonblocking execution depending on the underlying backend.
				 */
				static constexpr bool isNonblockingExecution() {
					return IMPLEMENTATION< _GRB_BSP1D_BACKEND >::isNonblockingExecution();
				}
		};

		/** @} */

	} // namespace config

} // namespace grb

#endif // end ``_H_GRB_BSP1D_CONFIG''

