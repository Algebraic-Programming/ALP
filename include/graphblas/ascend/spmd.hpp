
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
 * Provides the SPMD functions for the Ascend backend.
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_SPMD
#define _H_GRB_ASCEND_SPMD

#include <cstddef> //size_t

#include <graphblas/base/spmd.hpp>


namespace grb {

	/** The spmd class is based on that of the reference backend */
	template<>
	class spmd< ascend > {

		public:

			/** Refers back to the reference backend */
			static inline size_t nprocs() noexcept {
				return spmd< reference >::nprocs();
			}

			/** Refers back to the reference backend */
			static inline size_t pid() noexcept {
				return spmd< reference >::pid();
			}

			/** Refers back to the reference backend */
			static RC sync( const size_t msgs_in = 0, const size_t msgs_out = 0 ) noexcept {
				return spmd< reference >::sync( msgs_in, msgs_out );
			}

			/** Refers back to the reference backend */
			static RC barrier() noexcept {
				return spmd< reference >::barrier();
			}

	}; // end class ``spmd'' ascend implementation

} // namespace grb

#endif // end _H_GRB_ASCEND_SPMD

