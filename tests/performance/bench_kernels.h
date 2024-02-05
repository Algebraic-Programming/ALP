
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

	void bench_kernels_axpy(
		double * __restrict__,
		const double, const double * __restrict__, const double * __restrict__,
		const size_t
	);

	void bench_kernels_dot(
		double * __restrict__ const,
		const double * __restrict__, const double * __restrict__,
		const size_t
	);

	void bench_kernels_reduce(
		double * __restrict__ const, const double * __restrict__, const size_t
	);

}

#else

/**
 * Executes \f$ a = \alpha x + y \f$ for \a a, \a x, and \a y vectors of
 * length \a n.
 *
 * @param[out] a     The output vector.
 * @param[in]  alpha The scalar with which to multiply \a x prior to addition.
 * @param[in]  x     The right-hand multiplicand vector.
 * @param[in]  y     The vector which will be added to the output.
 * @param[in]  n     The size of the vectors \a a, \a x, and \a y.
 */
void bench_kernels_axpy(
	double * restrict a,
	const double alpha, const double * restrict x,
	const double * restrict y,
	const size_t n
);

/** 
 * Executes the inner-product computation \f$ alpha = (x,y) \f$ with \a x and
 * \a y vectors of length \a n.
 *
 * @param[out] alpha The output scalar.
 * @param[in]  x     The left-side input vector.
 * @param[in]  y     The right-side input vector.
 * @param[in]  n     The size of the vectors \a x and \a y.
 */
void bench_kernels_dot (
	double * restrict const alpha,
	const double * restrict x, const double * restrict y,
	const size_t n
);

/**
 * Executes the reduction \f$ alpha = (x,e) \f$, where \f$ e \f$ is a vector of
 * length \a n consisting of ones, and \a x is an input vector of length \a n.
 *
 * @param[out] alpha The output scalar.
 * @param[in]  x     The input vector.
 * @param[in]  n     The size of \a x.
 */
void bench_kernels_reduce(
	double * restrict const alpha, const double * restrict x, const size_t n
);

#endif

