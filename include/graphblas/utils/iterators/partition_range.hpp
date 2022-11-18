
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

/**
 * @file partition_range.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of utilities to partition iterators across processes.
 */

#include <cstddef>
#include <algorithm>
#include <cassert>

#ifndef H_GRB_UTILS_PARTITION_RANGE
#define H_GRB_UTILS_PARTITION_RANGE

namespace grb {
	namespace utils {

		/**
		 * Partitions the size of a collection across processes and computes the first offset
		 * and the size for the local partition.
		 *
		 * @tparam T size type
		 * @param[in] num_procs total number of processes
		 * @param[in] this_proc ID of current process
		 * @param[in] num_elements total number of elements in the collection
		 * @param[out] first_offset offset to the first element of the local partition
		 * @param[out] local_size size of the local partition
		 */
		template< typename T > void partition_collection_size(
				size_t num_procs,
				size_t this_proc,
				T num_elements,
				T& first_offset,
				T& local_size
		) {
			const T per_process = ( num_elements + num_procs - 1 ) / num_procs; // round up
			first_offset = std::min( per_process * static_cast< T >( this_proc ), num_elements );
			local_size = std::min( first_offset + per_process, num_elements );
		}

		/**
		 * Partitions an iteration range across processes according to the given information.
		 *
		 * With \p num_procs processes and \p this_proc < \p num_procs and a collection of \p num_elements
		 * elements across all processes, it partitions the collection evenly among processes and sets
		 * \p begin and \p end so that they iterate over the local partition designated by \p this_proc.
		 *
		 * It works also for a single-process scenario.
		 *
		 * Note: the number of processes and the ID of the current process is expected in input
		 * not to introduce dependencies on separate code paths.
		 *
		 * @tparam IterT iterator type
		 * @param[in] num_procs number of processes
		 * @param[in] this_proc Id of current process
		 * @param[in] num_elements number of elements of the collection; it can be computed as
		 *  \code std::distance( begin, end ) \endcode
		 * @param[out] begin beginning iterator to the whole collection
		 * @param[out] end end iterator
		 */
		template< typename IterT > void partition_iteration_range_on_procs(
			size_t num_procs,
			size_t this_proc,
			size_t num_elements,
			IterT &begin,
			IterT &end
		) {
			static_assert( std::is_base_of< std::random_access_iterator_tag,
				typename std::iterator_traits< IterT >::iterator_category >::value,
				"the given iterator is not a random access one" );
			assert( this_proc < num_procs );
			assert( num_elements == static_cast< size_t >( end - begin ) );
			if( num_procs == 1 ) {
				return;
			}
			size_t first, num_local_elements;
			partition_collection_size( num_procs, this_proc, num_elements, first, num_local_elements );
			if( num_local_elements < num_elements ) {
				end = begin;
				end += num_local_elements;
			}
			if( first > 0 ) {
				begin += first;
			}
		}

	} // namespace utils
} // namespace grb

#endif // H_GRB_UTILS_PARTITION_RANGE
