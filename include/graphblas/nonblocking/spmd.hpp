
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

/*
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_SPMD
#define _H_GRB_NONBLOCKING_SPMD

#include <cstddef> //size_t

#include <graphblas/base/spmd.hpp>


namespace grb {

	/** The spmd class is based on that of the reference backend */
	template<>
	class spmd< nonblocking > {

		public:

			static inline size_t nprocs() noexcept {
				return spmd< reference >::nprocs();
			}

			static inline size_t pid() noexcept {
				return spmd< reference >::pid();
			}

			static RC sync( const size_t msgs_in = 0, const size_t msgs_out = 0 ) noexcept {
				return spmd< reference >::sync( msgs_in, msgs_out );
			}

			static RC barrier() noexcept {
				return spmd< reference >::barrier();
			}

	}; // end class ``spmd'' nonblocking implementation

} // namespace grb

#endif // end _H_GRB_NONBLOCKING_SPMD

