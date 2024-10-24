
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

#ifndef _H_GRB_OMP_CONFIG
#define _H_GRB_OMP_CONFIG

#include <cassert>
#include <cstddef>

#include <omp.h>

#include <graphblas/base/config.hpp>

#ifndef NDEBUG
 #include <cmath>
#endif


namespace grb {

	namespace config {

		/** OpenMP defaults and utilities for internal use. */
		class OMP {

			private:

				static inline size_t nblocks(
					const size_t start, const size_t end,
					const size_t block_size
				) {
					const size_t n = end - start;
					return n / block_size + ( n % block_size > 0 ? 1 : 0 );
				}


			public:

				/**
				 * @returns The minimum loop size before a parallel-for is recommended.
				 *
				 * This function can be called from a sequential or parallel context.
				 *
				 * Use this to guard OpenMP parallel sections within performance-critical
				 * code sections.
				 */
				static size_t minLoopSize() {
					assert( std::ceil(std::log2(config::CACHE_LINE_SIZE::value()))
						<= 4*sizeof(size_t) );
					const size_t cacheLineSize = config::CACHE_LINE_SIZE::value();
					return cacheLineSize * cacheLineSize;
				}

				/**
				 * @returns The number of threads reported by OpenMP.
				 *
				 * This function must be called from a sequential context.
				 *
				 * \warning Do not call from performance-critical sections.
				 */
				static size_t threads() {
					size_t ret;
					#pragma omp parallel
					{
						#pragma omp single
						{ ret = static_cast< size_t >( omp_get_num_threads() ); }
					}
#ifdef _DEBUG
					std::cout << "OMP::config::threads() returns " << ret << "\n";
#endif
					return ret;
				}

				/**
				 * @returns The number of threads in the current OpenMP parallel section.
				 *
				 * This function must be called from a parallel context.
				 */
				static inline size_t current_threads() {
					return static_cast< size_t >( omp_get_num_threads() );
				}

				/**
				 * @returns The thread ID in the current OpenMP parallel section.
				 *
				 * This function must be called from a parallel context.
				 */
				static inline size_t current_thread_ID() {
					return static_cast< size_t >( omp_get_thread_num() );
				}

				/**
				 * Partitions a range across all available #threads. Elements of the range
				 * are assigned in blocks of a given block size.
				 *
				 * This function must be called from a parallel context.
				 *
				 * @param[out] local_start Where this thread's local range starts
				 * @param[out] local_end   Where this thread's range ends (exclusive)
				 * @param[in]  start       The lowest index of the global range (inclusive)
				 * @param[in]  end         The lowest index that is out of the global range
				 *
				 * The caller must ensure that \a end >= \a start.
				 *
				 * \note This function may return an empty range, i.e., \a local_start >=
				 *       \a local_end.
				 *
				 * Optional arguments:
				 *
				 * @param[in] block_size Local ranges should be a multiple of this value
				 * @param[in] t          The thread ID
				 * @param[in] T          The total number of threads
				 *
				 * The parameters \a t and \a T are by default determined automatically.
				 *
				 * The parameter \a block_size by default equals
				 * #config::CACHE_LINE_SIZE::value().
				 *
				 * \note The number of elements in the returned local range may not be a
				 *       multiple of \a block_size if and only if the number of elements
				 *       in the global range is not a multiple of \a block_size. In this
				 *       case, only one thread may have a number of local elements that
				 *       is not a multiple of \a block_size.
				 */
				static inline void localRange(
					size_t &local_start, size_t &local_end,
					const size_t start, const size_t end,
					const size_t block_size = config::CACHE_LINE_SIZE::value(),
					const size_t t = static_cast< size_t >( omp_get_thread_num() ),
					const size_t T = current_threads()
				) {
					assert( start <= end );
					assert( block_size > 0 );
					assert( T > 0 );
					assert( t < T );
					const size_t blocks = nblocks( start, end, block_size );
					const size_t blocks_per_thread = blocks / T + ( blocks % T > 0 ? 1 : 0 );
					local_start = start + t * blocks_per_thread * block_size;
					local_end = local_start + blocks_per_thread * block_size;
#ifdef _DEBUG
					#pragma omp critical
					std::cout << "\t\tThread " << t << " gets range " << local_start << "--"
						<< local_end << " from global range " << start << "--" << end << ". "
						<< "The local range will be capped at " << end << ".\n";
#endif
					if( local_end > end ) {
						local_end = end;
					}
					if( local_start > local_end ) {
						local_start = local_end;
					}
					assert( local_start >= start );
					assert( local_end <= end );
					assert( local_start <= local_end );
				}

				/**
				 * @returns Given a range that is to be distributed across the available
				 *          threads, how many thread-local ranges will be non-empty.
				 *
				 * The following parameters are mandatory:
				 *
				 * @param[in] start      The lowest index of the global range (inclusive)
				 * @param[in] end        The lowest index that is out of the global range
				 *
				 * The following parameters are optional:
				 *
				 * @param[in] block_size Local ranges should be a multiple of this value
				 * @param[in] T          The total number of threads
				 *
				 * The default values are the same as for #localRange.
				 */
				static inline size_t nranges(
					const size_t start, const size_t end,
					const size_t block_size = config::CACHE_LINE_SIZE::value(),
					const size_t T = current_threads()
				) {
					assert( start <= end );
					assert( block_size > 0 );
					assert( T > 0 );
					const size_t blocks = nblocks( start, end, block_size );
					return std::min( blocks, T );
				}

		};

	} // namespace config

} // namespace grb

#endif

