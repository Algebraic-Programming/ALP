
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
 * Configuration settings for the nonblocking backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_CONFIG
#define _H_GRB_NONBLOCKING_CONFIG

#include <graphblas/base/config.hpp>
#include <graphblas/reference/config.hpp>


namespace grb {

	/**
	 * \defgroup nonblockingConfiug Configuration of the nonblocking backend
	 * \ingroup config
	 *
	 * Groups all definitions and documentations corresponding to the
	 * #grb::nonblocking backend.
	 *
	 * @{
	 */

	namespace config {

		/**
		 * Configuration parameters relating to the pipeline data structure.
		 */
		class PIPELINE {

			public:

				/**
				 * How many independent pipelines any ALP algorithm may concurrently expose.
				 *
				 * The number of pipelines could exceed this maximum number. If this
				 * happens, and if #grb::config::PIPELINE::warn_if_exceeded_capacity is
				 * configured <tt>true</tt>, a warning will be output to the standard error
				 * stream.
				 */
				static constexpr const size_t max_pipelines = 4;

				/**
				 * Pipelines are constructed with default space for this many containers.
				 *
				 * The default is such that each underlying set used by the pipeline
				 * representation takes less than one kB space.
				 *
				 * Pipelines could exceed this maximum number of containers. If this
				 * happens, and if #grb::config::PIPELINE::warn_if_exceeded_capacity is
				 * configured <tt>true</tt>, a warning will be output to the standard error
				 * stream.
				 */
				static constexpr const size_t max_containers = 16;

				/**
				 * Pipelines are constructed with default space for this many stages.
				 *
				 * Pipelines could exceed this number of stages. If this happens, and if
				 * #grb::config::PIPELINE::warn_if_exceeded_capacity is configured
				 * <tt>true</tt>, a warning will be output to the standard error stream.
				 */
				static constexpr const size_t max_depth = 16;

				/**
				 * Pipelines are constructed with default space for this many tiles.
				 *
				 * Pipelines could exceed this number of tiles. If this happens, and if
				 *
				 * #grb::config::PIPELINE::warn_if_exceeded_capacity is configured
				 * <tt>true</tt>, a warning will be output to the standard error stream.
				 */
				static constexpr const size_t max_tiles = 1 << 16;

				/**
				 * Emit a warning to standard error stream if the default pipeline
				 * capacities are exceeded.
				 */
				static constexpr bool warn_if_exceeded = true;

				/**
				 * When <tt>true</tt>, calling a fake nonblocking primitive for a first time
				 * will emit a warning to the standard error stream.
				 */
				static constexpr bool warn_if_not_native = true;

		};

		/**
		 * Configuration parameters relating to the analytic model employed by the
		 * nonblocking backend.
		 */
		class ANALYTIC_MODEL {

			public:

				/**
				 * The minimum tile size that may be automatically selected by the analytic
				 * model.
				 *
				 * A tile size that is set manually may be smaller than MIN_TILE_SIZE.
				 */
				static constexpr size_t MIN_TILE_SIZE = 512;

				/**
				 * The L1 cache size is assumed to be a bit smaller than the actual size to
				 * take into account any data that may be stored in cache and are not
				 * considered by the analytic model, e.g., matrices for the current design.
				 */
				static constexpr double L1_CACHE_USAGE_PERCENTAGE = 0.98;

				/**
				 * Determines whether the tile size is automatically selected by the
				 * analytic model or whether it is manually selected by the user with the
				 * environment variable GRB_NONBLOCKING_TILE_SIZE.
				 */
				static bool manual_tile_size;

				/**
				 * The tile size that is manually selected by the user and is initialized in
				 * init.cpp. This variable is only set when the GRB_NONBLOCKING_TILE_SIZE
				 * environment variable is defined, and if so, this variable equal its
				 * content.
				 */
				static size_t manual_fixed_tile_size;

				/**
				 * The maximum number of threads available in the system that may be set
				 * with the environment variable OMP_NUM_THREADS.
				 */
				static size_t num_threads;

		};

		/**
		 * Implementation-dependent configuration parameters for the \a nonblocking
		 * backend
		 *
		 * @see grb::config::IMPLEMENTATION
		 */
		template<>
		class IMPLEMENTATION< nonblocking > {

			public:

				/**
				 * The selected backend performs nonblocking execution.
				 */
				static constexpr bool isNonblockingExecution() {
					return true;
				}

				/**
				 * The minimum tile size that may be used by the analytic model.
				 */
				static constexpr size_t analyticModelMinimumTileSize() {
					return ANALYTIC_MODEL::MIN_TILE_SIZE;
				}

				/**
				 * The percentage of the L1 cache size that is used by the analytic model.
				 */
				static constexpr double analyticModelL1CacheUsagePercentage() {
					return ANALYTIC_MODEL::L1_CACHE_USAGE_PERCENTAGE;
				}

				/**
				 * Whether the tile size is manually set by the user or not.
				 */
				static bool isManualTileSize() {
					return ANALYTIC_MODEL::manual_tile_size;
				}

				/**
				 * The tile size that is manually selected by the user.
				 */
				static size_t manualFixedTileSize() {
					return ANALYTIC_MODEL::manual_fixed_tile_size;
				}

				/**
				 * The maximum number of threads available in the system.
				 */
				static size_t numThreads() {
					return ANALYTIC_MODEL::num_threads;
				}

				/**
				 * A private memory segment shall never be accessed by threads other than
				 * the thread who allocates it. Therefore we choose aligned mode here.
				 */
				static constexpr ALLOC_MODE defaultAllocMode() {
					return ALLOC_MODE::ALIGNED;
				}

				/**
				 * For the nonblocking backend, a shared memory-segment should use
				 * interleaved alloc so that any thread has uniform access on average.
				 */
				static constexpr ALLOC_MODE sharedAllocMode() {
					return ALLOC_MODE::INTERLEAVED;
				}

				/**
				 * By default, use the coordinates of the selected backend.
				 */
				static constexpr Backend coordinatesBackend() {
					return nonblocking;
				}

				/**
				 * Whether the backend has vector capacities always fixed to their
				 * defaults.
				 */
				static constexpr bool fixedVectorCapacities() {
					return true;
				}

				/**
				 * The number of individual buffers that a vector should be able to
				 * concurrently maintain.
				 *
				 * @param[in] n The vector size.
				 *
				 * @returns The number of individual buffers that should be supported.
				 */
				static inline size_t maxBufferTiles( const size_t n ) {
					return n;
				}

				/**
				 * Helper function that computes the effective buffer size for a vector
				 * of \a n elements by taking into account the space required for storing
				 * the local stack size, the number of new nonzeroes, and the offset used
				 * for the prefix-sum algorithm
				 *
				 * @param[in] n The size of the vector.
				 * @param[in] T The maximum number of tiles that need be supported.
				 *
				 * @returns The buffer size given the vector size, maximum number of
				 *          tiles, and the requested configuration.
				 */
				static inline size_t vectorBufferSize( const size_t n ) {
					const size_t T = maxBufferTiles( n );
					size_t ret = n;

					// +1 for storing the local stack size
					// +1 for storing the number of new nonzeroes
					// +1 for storing the offset used for the prefix-sum algorithm
					ret += 3 * T;
					ret = std::max( 4 * T, ret );

					return ret;
				}

		};

	} // namespace config

	/** @} */

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_CONFIG''

