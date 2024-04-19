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
 *
 * @file This file registers (static) views on matrix containers that are either
 *       implemented, under implementation, or were at any point in time
 *       conceived and noteworthy enough to be recorded for future consideration.
 *       A static view represents a particular \em perspective on a container that
 *       can be defined at compile-time and that can always be applied to a container
 *       irrespective of features such as its dimensions.
 */

#ifndef _H_GRB_VIEWS
#define _H_GRB_VIEWS

#include <utility>

namespace grb {

	namespace view {

		template< typename OriginalType >
		struct Identity {

			using applied_to = OriginalType;

			static std::pair< size_t, size_t > dims( std::pair< size_t, size_t > dims_pair ) {
				return std::make_pair( dims_pair.first, dims_pair.second );
			}
		};

		template< typename OriginalType >
		struct Transpose {

			using applied_to = original_view;
			using dims_retval_type = std::pair< size_t, size_t >

				static dims_retval_type dims( std::pair< size_t, size_t > dims_pair ) {
				return std::make_pair( dims_pair.second, dims_pair.first );
			}
		};

	}; // namespace view

} // namespace grb

#endif // _H_GRB_VIEWS