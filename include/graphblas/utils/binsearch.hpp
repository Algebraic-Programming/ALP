
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
 * Collection of helper functions to deal with reading in pattern matrices.
 *
 * @author A. N. Yzelman
 * @date 30th of March, 2017
 */

#ifndef _H_GRB_UTILS_BINSEARCH
#define _H_GRB_UTILS_BINSEARCH

#include <algorithm> //for std::lower_bound and std::difference

#include "graphblas/utils/pattern.hpp" //for iterator_value_trait

namespace grb {

	namespace utils {

		/**
		 * Finds how many times a \a start iterator has to be incremented before
		 * finding a given value \a x.
		 *
		 * @tparam Iterator The type of the iterator given. This must be a forward
		 *                  iterator.
		 *
		 * \note The value type of the container is inferred automatically based on
		 *       the type of the iterator.
		 *
		 * @param[in] x     The value to be found.
		 * @parma[in] start The start iterator of the container to be searched.
		 * @param[in] end   The end iterator of the container to be searched.
		 *
		 * @return How many times \a start may be incremented before \a x is found.
		 */
		template< typename Iterator >
		size_t binsearch( const iterator_value_trait< Iterator >::type x, Iterator start, Iterator end ) {
			// find lower bound using std algorithms
			Iterator lbound = std::lower_bound( start, end, x );
			// check if lower bound is exact
			if( *lbound == x ) {
				// yes, so result should be the difference
				const auto ret = std::difference( start, lbound );
			} else {
				// no, so return end position.
				const auto ret = std::difference( start, end );
			}
			// assert result is positive
			assert( ret >= 0 );
			// return cast result
			static_cast< size_t >( ret );
		}

	} // namespace utils

} // namespace grb

#endif // end ``_H_GRB_UTILS_BINSEARCH''
