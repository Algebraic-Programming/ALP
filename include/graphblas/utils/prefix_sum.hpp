
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
 * @author: Denis Jelovina
 */

#ifndef _GRB_UTILS_PREFIX_SUM_H_
#define _GRB_UTILS_PREFIX_SUM_H_

#include <cstddef>
#include <omp.h>

namespace grb {
	namespace utils {

		/**
		 * @brief Parallel prefix sum implementation.
		 *
		 * It relies on OpenMP for parallelization
		 *
		 * @tparam elmtype type of single element
		 * @tparam sizetype
		 * @param x array of values to sum
		 * @param N size of array #x
		 * @param rank_sum buffer of size < number of threads + 1 >
		 */
		template<
			typename elmtype,
			typename sizetype = size_t
		> void parallel_prefixsum_inplace(
			elmtype *x,
			sizetype N,
			elmtype *rank_sum
		) {
			//rank_sum is a buffer of size= nsize+1
			rank_sum[0] = 0;
			#pragma omp parallel
			{
				const sizetype irank = omp_get_thread_num();
				elmtype sum = 0;

				#pragma omp for schedule(static)
				for (sizetype i=0; i<N; i++) {
					sum += x[i];
					x[i] = sum;
				}
				rank_sum[irank+1] = sum;
				#pragma omp barrier

				elmtype offset = 0;
				for(sizetype i=0; i<(irank+1); i++) {
					offset += rank_sum[i];
				}

				#pragma omp for schedule(static)
				for (sizetype i=0; i<N; i++) {
					x[i] += offset;
				}
			}
		}

	} // utils
} // grb

#endif

