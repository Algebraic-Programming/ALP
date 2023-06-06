
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
 * Contains the configuration parameters for the reference and reference_omp
 * backends.
 *
 * @author A. N. Yzelman
 * @date 13th of September 2017.
 */

#ifndef _H_GRB_REFERENCE_CONFIG
#define _H_GRB_REFERENCE_CONFIG

#include <graphblas/base/config.hpp>


namespace grb {

	namespace config {

		/**
		 * \defgroup referenceConfig Reference and reference_omp backend configuration
		 * \ingroup config
		 *
		 * All configuration parameters for the #grb::reference and the
		 * #grb::reference_omp backends.
		 *
		 * @{
		 */

		/**
		 * The memory allocation modes implemented in the #grb::reference and the
		 * #grb::reference_omp backends.
		 *
		 * \ingroup reference
		 */
		enum ALLOC_MODE {

			/** Allocation via <tt>posix_memalign</tt>. */
			ALIGNED,

			/** Allocation via <tt>numa_alloc_interleaved</tt>. */
			INTERLEAVED

		};

		/**
		 * Converts instances of #grb::config::ALLOC_MODE to a descriptive
		 * lower-case string.
		 *
		 * \ingroup reference
		 */
		std::string toString( const ALLOC_MODE mode );

		/**
		 * Default prefetching settings for reference and reference_omp backends.
		 *
		 * \note By default, prefetching is turned OFF as we found no setting that
		 *       will never result in a performance degradation across the dataset,
		 *       workloads, and architectures in our standard test set.
		 *
		 * \note The defaults may be overridden by specialisation, which additionally
		 *       makes it possible to choose different distances for different
		 *       backends.
		 *
		 * Prefetching presently only is implemented and evaluated for the SpMV and
		 * the SpMSpV multiplication kernels. Furthermore, it is only implemented for
		 * the gathering variant of either kernel. If you wish further support or
		 * evaluation, please feel free to create an issue or to contribute a merge
		 * request.
		 *
		 * \internal
		 * \warning This class should only be used by the reference or reference_omp
		 *          backends.
		 * \endinternal
		 *
		 * \ingroup reference
		 */
		template< Backend backend >
		class PREFETCHING {

			// guard against unintended use
			static_assert( backend == reference || backend == reference_omp || backend == tutorial,
				"Instantiating for non-reference backend" );

			public:

				/**
				 * Whether prefetching is enabled.
				 */
				static constexpr bool enabled() {
					return false;
				}

				/**
				 * The prefetch distance used during level-2 and level-3 operations.
				 *
				 * This value will be ignored if #enabled() returns <tt>false</tt>.
				 */
				static constexpr size_t distance() {
					return 128;
				}

		};

		/**
		 * This class collects configuration parameters that are specific to the
		 * #grb::reference backend. It details both configurations that could
		 * be modified by end users, as well as configurations that are sensible
		 * only to ALP developers; the full specification hence is only available
		 * within the developer documentation.
		 *
		 * \internal
		 * This class extends the base implementation API with some fields that
		 * facilitate composability between the #grb::reference and the
		 * #grb::reference_omp backends on the one hand, and the #grb::bsp1d and the
		 * #grb::hybrid backends on the other.
		 * \endinternal
		 *
		 * \ingroup reference
		 */
		template<>
		class IMPLEMENTATION< reference > {

			public:

				/** How to allocate private memory segments. */
				static constexpr ALLOC_MODE defaultAllocMode() {
					return ALLOC_MODE::ALIGNED;
				}

				/** How to allocate shared memory segments. */
				static constexpr ALLOC_MODE sharedAllocMode() {
					return ALLOC_MODE::ALIGNED;
				}

				/**
				 * \internal
				 * Whether the backend has vector capacities always fixed to their
				 * defaults.
				 * \endinternal
				 */
				static constexpr bool fixedVectorCapacities() {
					return true;
				}

				/**
				 * \internal
				 * The buffer size for allowing parallel updates to the sparsity of a
				 * vector of a given length. In the sequential reference implementation
				 * such a buffer is not required, hence this function will always return
				 * 0.
				 * \endinternal
				 */
				static constexpr size_t vectorBufferSize( const size_t, const size_t ) {
					return 0;
				}

				/**
				 * \internal
				 * By default, use the coordinates of the selected backend.
				 * \endinternal
				 */
				static constexpr Backend coordinatesBackend() {
					return reference;
				}

		};

		/**
		 * This class collects configuration parameters that are specific to the
		 * #grb::reference_omp backend. It details both configurations that could
		 * be modified by end users, as well as configurations that are sensible
		 * only to ALP developers; the full specification hence is only available
		 * within the developer documentation.
		 *
		 * \internal
		 * This class extends the base implementation API with some fields that
		 * facilitate composability between the #grb::reference and the
		 * #grb::reference_omp backends on the one hand, and the #grb::bsp1d and the
		 * #grb::hybrid backends on the other.
		 * \endinternal
		 *
		 * \ingroup reference
		 */
		template<>
		class IMPLEMENTATION< reference_omp > {

			private:

				/**
				 * \internal
				 * If \a N independent concurrent chunks are supported for parallel sparsity
				 * updates, then each chunk will have the returned minimum size (in bytes).
				 * \endinternal
				 */
				static constexpr size_t minVectorBufferChunksize() {
					return CACHE_LINE_SIZE::value();
				}

				/**
				 * \internal
				 * Vector-local buffer size for parallel sparsity updates (to vectors).
				 *
				 * The given buffer size is in the number of elements.
				 *
				 * This configuration parameter represents a space-time tradeoff; larger
				 * buffers will allow greater parallelism, smaller buffers obviously
				 * result in less memory use.
				 *
				 * Either this or relVectorBufferSize() must be set to a different value
				 * from 0.
				 * \endinternal
				 */
				static constexpr size_t absVectorBufferSize() {
					return 0;
				}

				/**
				 * \internal
				 * Vector-local buffer size for parallel sparsity updates (to vectors).
				 *
				 * The given buffer size is relative to the vector length.
				 *
				 * This configuration parameter represents a space-time tradeoff; larger
				 * buffers will allow greater parallelism, smaller buffers obviously
				 * result in less memory use.
				 *
				 * Values must be equal or larger to 0.
				 *
				 * Either this or absVectorBufferSize() must be set to a different value
				 * from 0.
				 * \endinternal
				 */
				static constexpr double relVectorBufferSize() {
					return 1;
				}


		public:

				/**
				 * A private memory segment shall never be accessed by threads other than
				 * the thread who allocates it. Therefore we choose aligned mode here.
				 */
				static constexpr ALLOC_MODE defaultAllocMode() {
					return ALLOC_MODE::ALIGNED;
				}

				/**
				 * For the reference_omp backend, a shared memory-segment should use
				 * interleaved alloc so that any thread has uniform access on average.
				 */
				static constexpr ALLOC_MODE sharedAllocMode() {
					// return ALLOC_MODE::ALIGNED; //DBG
					return ALLOC_MODE::INTERLEAVED;
				}

				/**
				 * \internal
				 * By default, use the coordinates of the selected backend.
				 * \endinternal
				 */
				static constexpr Backend coordinatesBackend() {
					return reference_omp;
				}

				/**
				 * \internal
				 * Whether the backend has vector capacities always fixed to their
				 * defaults.
				 * \endinternal
				 */
				static constexpr bool fixedVectorCapacities() {
					return true;
				}

				/**
				 * \internal
				 * Helper function that computes the effective buffer size for a vector
				 * of \a n elements using #absVectorBufferSize and #relVectorBufferSize
				 * and adds \a T elements to maintain local stack sizes.
				 *
				 * In case of a relative buffer size, the effective buffer size returned
				 * must perfectly divide \a T and will be rounded up.
				 *
				 * @param[in] n The size of the vector.
				 * @param[in] T The maximum number of threads that need be supported.
				 *
				 * @returns The buffer size given the vector size, maximum number of
				 *          threads, and the requested configuration.
				 * \endinternal
				 */
				static inline size_t vectorBufferSize( const size_t n, const size_t T ) {
					size_t ret;
					if( absVectorBufferSize() > 0 ) {
						(void)n;
						ret = absVectorBufferSize();
					} else {
						constexpr const double factor = relVectorBufferSize();
						static_assert( factor > 0, "Configuration error" );
						ret = factor * n;
					}
					ret = std::max( ret, T * minVectorBufferChunksize() );
					ret += T;
					if( ret % T > 0 ) {
						ret += T - ( ret % T );
					}
					ret = std::max( 2 * T, ret );
					assert( ret % T == 0 );
					return ret;
				}

		};

		/** @} */

	} // namespace config

} // namespace grb

#endif // end ``_H_GRB_REFERENCE_CONFIG''

