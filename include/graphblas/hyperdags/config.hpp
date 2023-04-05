
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
 * Contains the configuration parameters for the HyperDAGs backend
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

		/**
		 * The implementation details of the #grb::hyperdag backend.
		 *
		 * Since the HyperDAGs backend simply intercepts primitive calls and relies
		 * on a second backend for its functional execution, this class simply
		 * delegates all fields to that underlying backend.
		 *
		 * \note The user documentation only specifies the fields that under some
		 *       circumstances may benefit from a user adapting it. For viewing all
		 *       fields, please see the developer documentation.
		 *
		 * \note Adapting the fields should be done with care and may require
		 *       re-compilation and re-installation of the ALP framework.
		 */
		template<>
		class IMPLEMENTATION< hyperdags > {

			public:

				/**
				 * @returns The default allocation policy for private memory regions of the
				 *          underlying backend.
				 */
				static constexpr ALLOC_MODE defaultAllocMode() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::defaultAllocMode();
				}

				/**
				 * @returns The default allocation policy for shared memory regions of the
				 *          underlying backend.
				 */
				static constexpr ALLOC_MODE sharedAllocMode() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::sharedAllocMode();
				}

				/**
				 * \internal
				 * @returns The default vector coordinates instance of the underlying
				 *          backend.
				 *
				 * \note This is an extension for compatability with the reference and BSP1D
				 *       backends.
				 * \endinternal
				 */
				static constexpr Backend coordinatesBackend() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::coordinatesBackend();
				}

				/**
				 * \internal
				 * @returns The fixed vector capacity property of the underlying
				 *          implementation.
				 * \endinternal
				 */
				static constexpr bool fixedVectorCapacities() {
					return IMPLEMENTATION< _GRB_WITH_HYPERDAGS_USING >::
						fixedVectorCapacities();
				}

		};

	}

} // end namespace grb

#endif

