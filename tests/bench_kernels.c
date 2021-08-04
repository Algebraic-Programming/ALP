
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


#include "bench_kernels.h"

#ifdef BENCH_KERNELS_OPENMP

void bench_kernels_axpy( double * restrict a, const double alpha, const double * restrict x, const double * restrict y, const size_t n ) {
	assert( a != x );
	assert( a != y );
	assert( x != y );
	#pragma omp parallel for schedule(static,8)
	for( size_t i = 0; i < n; ++i ) {
		a[ i ] = alpha * x[ i ] + y[ i ];
	}
}

void bench_kernels_dot( double * restrict const alpha, const double * restrict xr, const double * restrict yr, const size_t n ) {
	assert( xr != yr );
	assert( alpha != xr );
	assert( alpha != yr );
	*alpha = xr[ n - 1 ] * yr[ n - 1];
	#pragma omp parallel
	{
		const size_t P = omp_get_num_threads();
		const size_t s = omp_get_thread_num();
		const size_t chunk = (n % P == 0) ? (n/P) : (n/P) + 1;
		size_t start = chunk * s;
		if( start > n - 1 ) {
			start = n - 1;
		}
		size_t end = start + chunk;
		if( end > n - 1 ) {
			end = n - 1;
		}
		assert( start <= end );
		if( start != end ) {
			double local_alpha = xr[ end - 1 ] * yr[ end - 1 ];
			for( size_t i = start; i < end - 1; ++i ) {
				local_alpha += xr[ i ] * yr[ i ];
			}
			#pragma omp critical
			{
				*alpha += local_alpha;
			}
		}
	}
}

void bench_kernels_reduce( double * restrict const alpha, const double * restrict xr, const size_t n ) {
	assert( alpha != xr );
	*alpha = xr[ n - 1 ];
	#pragma omp parallel
	{
		const size_t P = omp_get_num_threads();
		const size_t s = omp_get_thread_num();
		const size_t chunk = (n % P == 0) ? (n/P) : (n/P) + 1;
		size_t start = chunk * s;
		if( start > n - 1 ) {
			start = n - 1;
		}
		size_t end = start + chunk;
		if( end > n - 1 ) {
			end = n - 1;
		}
		assert( start <= end );
		if( start != end ) {
			double local_alpha = xr[ end - 1 ];
			for( size_t i = start; i < end - 1; ++i ) {
				local_alpha += xr[ i ];
			}
			#pragma omp critical
			{
				*alpha += local_alpha;
			}
		}
	}
}

#else

void bench_kernels_axpy( double * restrict a, const double alpha, const double * restrict x, const double * restrict y, const size_t n ) {
	assert( a != x );
	assert( a != y );
	assert( x != y );
	for( size_t i = 0; i < n; ++i ) {
		(*a++) = alpha * (*x++) + (*y++);
	}
}

void bench_kernels_dot( double * restrict const alpha, const double * restrict xr, const double * restrict yr, const size_t n ) {
	assert( xr != yr );
	assert( alpha != xr );
	assert( alpha != yr );
	*alpha = xr[ n - 1 ] * yr[ n - 1 ];
	for( size_t i = 0; i < n - 1; ++i ) {
		*alpha += (*xr++) * (*yr++);
	}
}

void bench_kernels_reduce( double * restrict const alpha, const double * restrict xr, const size_t n ) {
	assert( alpha != xr );
	*alpha = xr[ n - 1 ];
	for( size_t i = 0; i < n - 1; ++i ) {
		*alpha += (*xr++);
	}
}

#endif

