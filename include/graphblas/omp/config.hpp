
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

namespace grb {
	namespace config {

		/** OpenMP defaults and utilities for internal use. */
		class OMP {

		public:
			/**
			 * @returns The number of threads reported by OpenMP.
			 *
			 * This function must be called from a sequential context.
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
			 * Partitions a range across all available #threads. Elements of the range
			 * are assigned in blocks of a given block size.
			 *
			 * This function must be called from a parallel context.
			 */
			static inline void localRange( size_t & local_start, size_t & local_end, const size_t start, const size_t end, const size_t block_size = config::CACHE_LINE_SIZE::value() ) {
				const size_t T = current_threads();
				const size_t t = static_cast< size_t >( omp_get_thread_num() );
				assert( start <= end );
				assert( block_size > 0 );
				assert( T > 0 );
				assert( t < T );
				const size_t n = end - start;
				const size_t blocks = n / block_size + ( n % block_size > 0 ? 1 : 0 );
				const size_t blocks_per_thread = blocks / T + ( blocks % T > 0 ? 1 : 0 );
				local_start = start + t * blocks_per_thread * block_size;
				local_end = local_start + blocks_per_thread * block_size;
#ifdef _DEBUG
				std::cout << "\t\tThread " << t << " gets range " << local_start << "--" << local_end << " from global range " << start << "--" << end << ". The local range will be capped at " << end
						  << ".\n";
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
		};
	} // namespace config
} // namespace grb

#endif
