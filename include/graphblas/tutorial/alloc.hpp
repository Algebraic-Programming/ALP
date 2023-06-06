
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
 * @date 22nd of January, 2021
 */

#ifndef _H_GRB_ALLOC_TUTORIAL
#define _H_GRB_ALLOC_TUTORIAL

#include <graphblas/base/alloc.hpp>
#include <graphblas/reference/alloc.hpp>

#include "config.hpp"


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * Provides standard allocation mechanisms using the POSIX and libnuma
			 *   -# posix_memalign() and
			 *   -# numa_alloc_interleaved()
			 * system calls.
			 *
			 * When one of these functions are not available a different allocation
			 * mechanism must be selected.
			 */
			template<>
			class AllocatorFunctions< tutorial > : public AllocatorFunctions< reference  > {

			};

			template<>
			class Allocator< tutorial > : public Allocator< reference > {};

		} // namespace internal

	}     // namespace utils

} // namespace grb

#endif

