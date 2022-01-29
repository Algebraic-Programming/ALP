
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


#include <omp.h>
#include <assert.h>
#include <stddef.h> //for size_t

#ifdef __cplusplus

extern "C" {
	void bench_kernels_axpy  ( double * __restrict__, const double, const double * __restrict__, const double * __restrict__, const size_t );
	void bench_kernels_dot   ( double * __restrict__ const, const double * __restrict__, const double * __restrict__, const size_t );
	void bench_kernels_reduce( double * __restrict__ const, const double * __restrict__, const size_t );
}

#else

/** \todo add documentation */
void bench_kernels_axpy( double * restrict a, const double alpha, const double * restrict x, const double * restrict y, const size_t n );

/** \todo add documentation */
void bench_kernels_dot ( double * restrict const alpha, const double * restrict xr, const double * restrict yr, const size_t n );

/** \todo add documentation */
void bench_kernels_reduce( double * restrict const alpha, const double * restrict xr, const size_t n );

#endif

