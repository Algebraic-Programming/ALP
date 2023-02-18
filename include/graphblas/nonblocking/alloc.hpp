
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
 * Allocator functions for the nonblocking backend
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_ALLOC_NONBLOCKING
#define _H_GRB_ALLOC_NONBLOCKING

#include <iostream>

#include <graphblas/base/alloc.hpp>

#include "config.hpp"


namespace grb {

	namespace utils {

		namespace internal {

			template<>
			class Allocator< nonblocking > {

				private:

					/** Prevent initialisation. */
					Allocator();

				public:

					/** Refer to the standard allocation mechanism. */
					typedef AllocatorFunctions< reference > functions;

			};

		} // namespace internal

	} // namespace utils

} // namespace grb

#endif

