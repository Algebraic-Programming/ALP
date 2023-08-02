
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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

#ifndef _H_TEST_UTILS_PREFIX_SUM
#define _H_TEST_UTILS_PREFIX_SUM

/**
 * @file prefix_sum.hpp
 * @author Benjamin Lozes
 * @brief Primitive computing an in-place prefix sum on a vector.
 * @date 02/08/2021
 */

#include <algorithm>
#include <chrono>
#include <utility>

#include <graphblas/reference/config.hpp>

#include <graphblas.hpp>

using namespace grb;

namespace {

	template< typename T >
	void sequential_prefix_sum( T * array, size_t array_size ) {
		for( size_t i = 1; i < array_size; ++i ) {
			array[ i ] += array[ i - 1 ];
		}
	}

	template< typename T >
	void parallel_prefix_sum( T * array, const size_t array_size ) {
#ifdef _GRB_WITH_OMP
		const size_t min_loop_size = config::OMP::minLoopSize();

		// If the parallelism is not worth it, compute the prefix sum sequentially
		if( min_loop_size >= array_size ) {
			sequential_prefix_sum( array, array_size );
			return;
		}

		T frontiers[ omp_get_max_threads() ];
#pragma omp parallel default( none ) shared( array, frontiers ) firstprivate( array_size )
		{
			size_t begin, end;
			config::OMP::localRange( begin, end, 0, array_size );
			const size_t nthreads = config::OMP::current_threads();
			const size_t tid = config::OMP::current_thread_ID();

			for( size_t i = begin; i < end - 1; ++i ) {
				array[ i + 1 ] += array[ i ];
			}
			frontiers[ tid ] = array[ end - 1 ];

#pragma omp barrier
#pragma omp single
			{
				for( size_t i = 1; i < nthreads; ++i ) {
					frontiers[ i ] += frontiers[ i - 1 ];
				}
			}
			// Implicit barrier after the single section

			if( tid > 0 ) {
				for( size_t i = begin; i < end; ++i ) {
					array[ i ] += frontiers[ tid - 1 ];
				}
			}
		}
#else
		std::cerr << "This section should never be executed" << std::endl;
		sequential_prefix_sum( array, array_size );
#endif
	}
} // anonymous namespace

template< typename T, Backend backend = grb::config::default_backend >
void prefix_sum( T * array, size_t array_size ) {
	if( backend == Backend::reference_omp ) {
		parallel_prefix_sum( array, array_size );
	} else {
		sequential_prefix_sum( array, array_size );
	}
}

#endif // _H_TEST_UTILS_PREFIX_SUM
