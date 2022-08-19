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
 *       irrespective of dynamic features such as its dimensions.
 * 		 A View provides information about the structured matrix it is applied to,
 *       including its type (member type \a applied_to), or how read its dimensions.
 */

#ifndef _H_ALP_VIEWS
#define _H_ALP_VIEWS

#include <algorithm>
#include <utility>

namespace alp {

	namespace view {

		/**
		 * Lists the view types exposed to the user.
		 *
		 * \note View type "_internal" shall not be used by the user and
		 *       its use may result in an undefined behaviour.
		 *
		 * \note \internal "_internal" value is added so that all view types have
		 *                 a defined type_id field, which is used by internal
		 *                 type traits.
		 *
		 */
		enum Views {
			original,
			gather,
			transpose,
			diagonal,
			_internal
		};

		template< typename OriginalType >
		struct Original {

			using applied_to = OriginalType;

			static constexpr Views type_id = Views::original;

		};

		template< typename OriginalType >
		struct Gather {

			using applied_to = OriginalType;

			static constexpr Views type_id = Views::gather;

		};

		template< typename OriginalType >
		struct Transpose {

			using applied_to = OriginalType;

			static constexpr Views type_id = Views::transpose;

		};

		template< typename OriginalType >
		struct Diagonal {

			using applied_to = OriginalType;

			static constexpr Views type_id = Views::diagonal;

		};

		template< typename LambdaFunctionType >
		struct Functor {

			using applied_to = LambdaFunctionType;

			/** Functor views are not exposed to the user */
			static constexpr Views type_id = Views::_internal;

		};

	}; // namespace view

} // namespace alp

#endif // _H_ALP_VIEWS
