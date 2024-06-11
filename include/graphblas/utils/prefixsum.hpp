
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
 * Sequential and parallel prefix-sum algorithms.
 *
 * @author A. N. Yzelman
 * @date 3rd of April, 2024
 */

#ifndef _H_GRB_UTILS_PREFIXSUM
#define _H_GRB_UTILS_PREFIXSUM

#include <graphblas/omp/config.hpp>
#include <graphblas/base/config.hpp>

#include <cstddef>
#include <algorithm>


namespace grb {

	namespace utils {

		/**
		 * Sequential in-place prefix-sum computation.
		 *
		 * @tparam copyEnd Whether the last entry of the prefix-sum array, after it
		 *                 has been computed, should be copied to <tt>array[ n ]</tt>.
		 *
		 * \note This is a common requirement for CRS/CCS based operations, and hence
		 *       explicitly supported by this utility.
		 *
		 * @tparam T The type of the array on which the prefix-sum computation is to
		 *           proceed.
		 *
		 * @param[in,out] array The array for which to compute the prefix-sum.
		 *
		 * Computation will happen in-place.
		 *
		 * @param[in] n The size of \a array (in number of elements).
		 * 
		 */
		template< bool copyEnd, typename T >
		void prefixSum_seq(
			T *__restrict__ array, const size_t n
		) {
			for( size_t i = 0; i < n - 1; ++i ) {
				array[ i + 1 ] += array[ i ];
			}
			if( copyEnd ) {
				array[ n ] = array[ n - 1 ];
			}
		}

#ifdef _GRB_WITH_OMP
		/**
		 * Phase 1/3 for OpenMP-based prefix sum.
		 *
		 * Should be called from within an OpenMP parallel section.
		 *
		 * Should be followed with an OpenMP barrier before a subsequent call to
		 * #prefixSum_ompPar_phase2.
		 *
		 * See #prefixSum_ompPar for full documentation.
		 */
		template< bool copyEnd, typename T >
		void prefixSum_ompPar_phase1(
			T *__restrict__ array, const size_t n,
			T &workspace
		) {
			(void) copyEnd;
			(void) workspace;
			size_t start, end;
			config::OMP::localRange( start, end, 0, n,
				config::CACHE_LINE_SIZE::value() );
			if( end > start ) {
				prefixSum_seq< false >( array + start, end - start );
			}
		}

		/**
		 * Phase 2/3 for OpenMP-based prefix sum.
		 *
		 * Should be called from within an OpenMP parallel section and after a call to
		 * #prefixSum_ompPar_phase1 \em and subsequent OpenMP barrier.
		 *
		 * Should be followed with an OpenMP barrier before a subsequent call to
		 * #prefixSum_ompPar_phase3.
		 *
		 * See #prefixSum_ompPar for full documentation.
		 */
		template< bool copyEnd, typename T >
		void prefixSum_ompPar_phase2(
			T *__restrict__ array, const size_t n,
			T &myOffset
		) {
#ifdef _DEBUG
			std::cout << "\t entering prefixSum_ompPar_phase2\n";
#endif
			(void) copyEnd;
			size_t dummy, offset_index;
			myOffset = 0;
			for( size_t k = 0; k < config::OMP::current_thread_ID(); ++k ) {
				config::OMP::localRange( dummy, offset_index, 0, n,
					config::CACHE_LINE_SIZE::value(), k );
				if( offset_index > dummy ) {
					assert( offset_index > 0 );
					myOffset += array[ offset_index - 1 ];
				}
			}
#ifdef _DEBUG
			#pragma omp critical
			std::cout << "\t\t thread " << config::OMP::current_thread_ID()
				<< " offset is " << myOffset << std::endl;
#endif
		}

		/**
		 * Phase 3/3 for OpenMP-based prefix sum.
		 *
		 * Should be called from within an OpenMP parallel section and after a call to
		 * #prefixSum_ompPar_phase2 \em and subsequent OpenMP barrier.
		 *
		 * See #prefixSum_ompPar for full documentation.
		 */
		template< bool copyEnd, typename T >
		void prefixSum_ompPar_phase3(
			T *__restrict__ array, const size_t n,
			T &myOffset
		) {
#ifdef _DEBUG
			std::cout << "\t entering prefixSum_ompPar_phase3\n"
				<< "\t\t computed offset is " << myOffset << "\n"
				<< "\t\t my thread ID is " << omp_get_thread_num() << "\n";
#endif
			size_t start, end;
			config::OMP::localRange( start, end, 0, n,
				config::CACHE_LINE_SIZE::value() );
			for( size_t i = start; i < end; ++i ) {
				array[ i ] += myOffset;
			}
			if( copyEnd && start < n && end == n ) {
				array[ n ] = array[ n - 1 ];
			}
		}

		/**
		 * Prefix-sum to be called from within an OpenMP parallel section.
		 *
		 * @tparam copyEnd Whether the last entry of the prefix-sum array, after it
		 *                 has been computed, should be copied to <tt>array[ n ]</tt>.
		 *
		 * \note This is a common requirement for CRS/CCS based operations, and hence
		 *       explicitly supported by this utility.
		 *
		 * @tparam T The type of the array on which the prefix-sum computation is to
		 *           proceed.
		 *
		 * @param[in,out] array The array for which to compute the prefix-sum.
		 *
		 * Computation will happen in-place.
		 *
		 * @param[in] n The size of \a array (in number of elements).
		 * 
		 * The algorithm requires the following workspace:
		 *
		 * @param[in,out] ws A single element of type \a T.
		 *
		 * The algorithm proceeds in three phases, separated by barriers. To compute
		 * multiple prefix-sums and to save from unnecessarily incurring barriers,
		 * each of the three phases can also be called manually; see
		 *  -# #prefixSum_ompPar_phase1,
		 *  -# #prefixSum_ompPar_phase2, and
		 *  -# #prefixSum_ompPar_phase3.
		 * When using these manual calls, barrier synchronisation in-between phases
		 * must be performed manually by the user.
		 */
		template< bool copyEnd, typename T >
		void prefixSum_ompPar(
			T *__restrict__ array, const size_t n,
			T &ws
		) {
			prefixSum_ompPar_phase1< copyEnd >( array, n, ws );
			#pragma omp barrier
			prefixSum_ompPar_phase2< copyEnd >( array, n, ws );
			#pragma omp barrier
			prefixSum_ompPar_phase3< copyEnd >( array, n, ws );
		}

		/**
		 * OpenMP-parallelised in-place prefix-sum computation.
		 *
		 * This variant starts its own OpenMP parallel section.
		 *
		 * @tparam copyEnd Whether the last entry of the prefix-sum array, after it
		 *                 has been computed, should be copied to <tt>array[ n ]</tt>.
		 *
		 * \note This is a common requirement for CRS/CCS based operations, and hence
		 *       explicitly supported by this utility.
		 *
		 * @tparam T The type of the array on which the prefix-sum computation is to
		 *           proceed.
		 *
		 * @param[in,out] array The array for which to compute the prefix-sum.
		 *
		 * Computation will happen in-place.
		 *
		 * @param[in] n The size of \a array (in number of elements).
		 *
		 * This function automatically reduces the number of threads when appropriate
		 * using a simple analytic model. If electing one thread, it will call the
		 * sequential prefix-sum algorithm.
		 *
		 * The analytic model depends on the following configuration settings:
		 *  - #grb::config::CACHE_LINE_SIZE, and
		 *  - #grb::config::OMP::minLoopSize().
		 */
		template< bool copyEnd, typename T >
		void prefixSum_omp( T *__restrict__ array, const size_t n ) {
			const size_t nthreads = std::min(
					config::OMP::threads(),
					std::max(
						static_cast< size_t >(1),
						n % grb::config::CACHE_LINE_SIZE::value() == 0
							? n / grb::config::CACHE_LINE_SIZE::value()
							: n / grb::config::CACHE_LINE_SIZE::value() + 1
					)
				);
			if( n < grb::config::OMP::minLoopSize() || nthreads == 1 ) {
				prefixSum_seq< copyEnd >( array, n );
			} else {
				assert( nthreads > 1 );
				#pragma omp parallel num_threads( nthreads )
				{
					T ws;
					prefixSum_ompPar< copyEnd >( array, n, ws );
				}
			}
		}
#endif

	} // end namespace grb::utils

} // end namespace grb

#endif // end _H_GRB_UTILS_PREFIXSUM

