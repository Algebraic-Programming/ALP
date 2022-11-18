
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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

#include <cstddef>
#include <algorithm>
#include <cassert>

#ifndef H_GRB_UTILS_PARTITION_RANGE
#define H_GRB_UTILS_PARTITION_RANGE

namespace grb {
	namespace utils {

		template< typename T > void partition_nonzeroes(
				size_t num_procs,
				size_t this_proc,
				T num_elements,
				T& first_offset,
				T& last_offset
		) {
			const T per_process{ ( num_elements + num_procs - 1 ) / num_procs }; // round up
			first_offset = std::min( per_process * static_cast< T >( this_proc ), num_elements );
			last_offset = std::min( first_offset + per_process, num_elements );
		}

		template< typename IterT > void partition_iteration_range_on_procs(
			size_t num_procs,
			size_t this_proc,
			size_t num_nonzeroes,
			IterT &begin,
			IterT &end
		) {
			static_assert( std::is_base_of< std::random_access_iterator_tag,
				typename std::iterator_traits< IterT >::iterator_category >::value,
				"the given iterator is not a random access one" );
			assert( num_nonzeroes == static_cast< size_t >( end - begin ) );
			size_t first, last;
			partition_nonzeroes( num_procs, this_proc, num_nonzeroes, first, last );
			if( last < num_nonzeroes ) {
				end = begin;
				end += last;
			}
			begin += first;
		}

		template< typename IterT > void partition_iteration_range_on_procs(
			size_t num_nonzeroes,
			IterT &begin,
			IterT &end
		) {
			return partition_iteration_range_on_procs( spmd<>::nprocs(), spmd<>::pid(), num_nonzeroes, begin, end );
		}

	} // namespace utils
} // namespace grb

#endif // H_GRB_UTILS_PARTITION_RANGE
