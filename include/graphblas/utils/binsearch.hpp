
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

#include <algorithm> //for std::lower_bound and std::distance


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
		 * @param[in] start The start iterator of the container to be searched.
		 * @param[in] end   The end iterator of the container to be searched.
		 *
		 * @return How many times \a start may be incremented before \a x is found.
		 */
		template< typename Iterator >
		size_t binsearch(
			const typename std::iterator_traits< Iterator >::value_type x,
			Iterator start, Iterator end
		) {
			// find lower bound using std algorithms
			Iterator lbound = std::lower_bound( start, end, x );
			// check if lower bound is exact
			size_t ret;
			if( *lbound == x ) {
				// yes, so result should be the difference
				ret = std::distance( start, lbound );
			} else {
				// no, so return end position.
				ret = std::distance( start, end );
			}
			// return cast result
			return ret;
		}

		/**
		 * Given a monotonic decreasing function \a f, find the last argument in a
		 * given collection that returns the largest output. The collection should
		 * be ordered in a way that guarantees the monotonicity of \a f.
		 *
		 * This function proceeds using binary search, and thus completes in
		 * logarithmic time.
		 *
		 * @param[in] f The monotonic decreasing function
		 * @param[in] l An iterator to the first argument in the collection
		 * @param[in] h An iterator matching \a l in end-position
		 *
		 * The iterator-pair \a l and \a h should be random access iterators--
		 * otherwise binary search cannot be performed.
		 */
		template< typename F, typename V >
		V maxarg( const F &f, V l, V h ) {
			// make upper bound inclusive
			(void) --h;
			while( true ) {
				// check for trivial solution
				if( f( h ) == f( l ) ) {
					return h;
				}
				// get half point
				const size_t half = ( h - l ) / 2;
				// if we would make a zero step, quit
				if( half == 0 ) {
					return l;
				}
				// get mid point iterator
				V m = l + half;
				// modify l and h according to value at m
				if( f( l ) == f( m ) ) {
					// in this case, look from midpoint onwards
					l = m;
				} else {
					// assert monotonicity
					assert( f( m ) < f( l ) ) ;
					// in this case, look before midpoint
					h = m;
				}
				// and recurse
			}
		}

	} // namespace utils

} // namespace grb

#endif // end ``_H_GRB_UTILS_BINSEARCH''

