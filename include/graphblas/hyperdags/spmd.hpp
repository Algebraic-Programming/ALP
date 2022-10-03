
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
 * @author A. Karanasiou
 * @date 15th of March 2022
 */

#include <cstddef> //size_t

#include <graphblas/base/spmd.hpp>

namespace grb {

	template<>
	class spmd< hyperdags > {

		public:

			static inline size_t nprocs() noexcept {
				return spmd< _GRB_WITH_HYPERDAGS_USING >::nprocs();
			}

			static inline size_t pid() noexcept {
				return spmd< _GRB_WITH_HYPERDAGS_USING >::pid();
			}

			static RC sync(
				const size_t msgs_in = 0, const size_t msgs_out = 0
			) noexcept {
				return spmd< _GRB_WITH_HYPERDAGS_USING >::sync( msgs_in, msgs_out );
			}

			static RC barrier() noexcept {
				return spmd< _GRB_WITH_HYPERDAGS_USING >::barrier();
			}

	}; // end class ``spmd'' reference implementation

} // namespace grb

