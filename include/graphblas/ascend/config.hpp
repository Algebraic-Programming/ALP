
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
 * Configuration settings for the Ascend backend.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_CONFIG
#define _H_GRB_ASCEND_CONFIG

#include <graphblas/base/config.hpp>
#include <graphblas/nonblocking/config.hpp>


namespace grb {

	/**
	 * \defgroup ascendConfig Nonblocking backend configuration
	 *
	 * \ingroup config
	 *
	 * All configuration parameters for the #grb::ascend backend.
	 *
	 * @{
	 */

	namespace config {

		/**
		 * The various supported Ascend boards.
		 */
		enum Ascend {
			Ascend_910A,
			Ascend_910B
		};

		/**
		 * Class with information about the Ascend cache/scratchpad hierarchy.
		 */
		//template< enum Ascend = _ASC_DEFAULT_TARGET > TODO FIXME no way to get this passed in from alpcxx / grbcxx
		template< enum Ascend = Ascend_910B > // Assuming 910B default instead
		class ASCEND_CACHE_HIERARCHY {};

		/**
		 * Cache hierarchy parameters for the 910A.
		 */
		template<>
		class ASCEND_CACHE_HIERARCHY< Ascend_910A > {
			public:
				static constexpr const size_t UB_SIZE = 8192;
		};

		/**
		 * Cache hierarchy parameters for the 910B.
		 */
		template<>
		class ASCEND_CACHE_HIERARCHY< Ascend_910B > {
			public:
				static constexpr const size_t UB_SIZE = 8192;
		};

		/**
		 * Implementation-dependent configuration parameters for the \a ascend
		 * backend.
		 *
		 * \note The user documentation only specifies the fields that under some
		 *       circumstances may benefit from a user adapting it. For viewing all
		 *       fields, please see the developer documentation.
		 *
		 * \note Adapting the fields should be done with care and may require
		 *       re-compilation and re-installation of the ALP framework.
		 *
		 * \ingroup ascendConfig
		 *
		 * @see grb::config::IMPLEMENTATION
		 */
		template<>
		class IMPLEMENTATION< ascend > {

			public:

				/**
				 * A private memory segment shall never be accessed by threads other than
				 * the thread who allocates it. Therefore we choose aligned mode here.
				 */
				static constexpr ALLOC_MODE defaultAllocMode() {
					return ALLOC_MODE::ALIGNED;
				}

				/**
				 * For the ascend backend, a shared memory-segment should use
				 * interleaved alloc so that any thread has uniform access on average.
				 */
				static constexpr ALLOC_MODE sharedAllocMode() {
					return ALLOC_MODE::INTERLEAVED;
				}

				/**
				 * \internal
				 * By default, use the coordinates of the selected backend.
				 *
				 * \note This is an extension that may, at some later stage, be used for
				 *       composability with the #grb::bsp1d and #grb::hybrid backends.
				 * \endinternal
				 */
				static constexpr Backend coordinatesBackend() {
					return IMPLEMENTATION< nonblocking >::coordinatesBackend();
				}

				/**
				 * \internal
				 * Whether the backend has vector capacities always fixed to their
				 * defaults.
				 * \endinternal
				 */
				static constexpr bool fixedVectorCapacities() {
					return true;
				}

		};

	} // namespace config

	/** @} */

} // namespace grb

#endif // end ``_H_GRB_ASCEND_CONFIG''

