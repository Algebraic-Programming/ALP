
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
 * @author A. N. Yzelman
 * @date 14th of January, 2022
 */

#ifndef _H_ALP_REFERENCE_SPMD
#define _H_ALP_REFERENCE_SPMD

#include <cstddef> //size_t

#include <alp/base/spmd.hpp>

namespace alp {

	/** \internal This is a single-process back-end. */
	template<>
	class spmd< reference > {

	public:

		/**
		 * @return The number of user processes in this GraphBLAS run.
		 *
		 * In this single-process backend, will always return 1.
		 */
		static inline size_t nprocs() noexcept {
			return 1;
		}

		/**
		 * @return The user process ID.
		 *
		 * In this single-process backend, will always return 0.
		 */
		static inline size_t pid() noexcept {
			return 0;
		}

		/**
		 * In this backend, corresponds to a no-op.
		 *
		 * @return alp::SUCCESS.
		 */
		static RC barrier() noexcept {
			return SUCCESS;
		}

	}; // end class ``spmd'' reference implementation

} // namespace alp

#endif // end _H_ALP_REFERENCE_SPMD

