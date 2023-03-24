
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
	 * \defgroup nonblockingConfig Nonblocking backend configuration
	 *
	 * \ingroup config
	 *
	 * All configuration parameters for the #grb::nonblocking backend.
	 *
	 * @{
	 */

	namespace config {

		/**
		 * Configuration parameters relating to the pipeline data structure.
		 *
		 * \ingroup nonblockingConfig
		 */
		class PIPELINE {

			public:

				/**
				 * How many independent pipelines any ALP algorithm may concurrently expose.
				 *
				 * The number of pipelines could exceed this maximum number. If this
				 * happens, and if #grb::config::PIPELINE::warn_if_exceeded is configured
				 * <tt>true</tt>, a warning will be output to the standard error stream.
				 */
				static constexpr const size_t max_pipelines = 4;

				/**
				 * Pipelines are constructed with default space for this many containers.
				 *
				 * The default is such that each underlying set used by the pipeline
				 * representation takes less than one kB space.
				 *
				 * Pipelines could exceed this maximum number of containers. If this
				 * happens, and if #grb::config::PIPELINE::warn_if_exceeded is configured
				 * <tt>true</tt>, a warning will be output to the standard error stream.
				 */
				static constexpr const size_t max_containers = 16;

				/**
				 * Pipelines are constructed with default space for this many stages.
				 *
				 * Pipelines could exceed this number of stages. If this happens, and if
				 * #grb::config::PIPELINE::warn_if_exceeded is configured <tt>true</tt>, a
				 * warning will be output to the standard error stream.
				 */
				static constexpr const size_t max_depth = 16;

				/**
				 * Pipelines are constructed with default space for this many tiles.
				 *
				 * Pipelines could exceed this number of tiles. If this happens, and if
				 *
				 * #grb::config::PIPELINE::warn_if_exceeded is configured <tt>true</tt>, a
				 * warning will be output to the standard error stream.
				 */
				static constexpr const size_t max_tiles = 1 << 16;

				/**
				 * Emit a warning to standard error stream if the default pipeline
				 * capacities are exceeded.
				 */
				static constexpr const bool warn_if_exceeded = true;

				/**
				 * When <tt>true</tt>, calling a fake nonblocking primitive for a first time
				 * will emit a warning to the standard error stream.
				 */
				static constexpr const bool warn_if_not_native = true;

		};

		/**
		 * Configuration parameters relating to the analytic model employed by the
		 * nonblocking backend.
		 *
		 * \ingroup nonblockingConfig
		 */
		class ANALYTIC_MODEL {

			public:

				/**
				 * The minimum tile size that may be automatically selected by the analytic
				 * model.
				 *
				 * A tile size that is set manually may be smaller than MIN_TILE_SIZE.
				 */
				static constexpr const size_t MIN_TILE_SIZE = 512;

				/**
				 * The L1 cache size is assumed to be a bit smaller than the actual size to
				 * take into account any data that may be stored in cache and are not
				 * considered by the analytic model, e.g., matrices for the current design.
				 */
				static constexpr const double L1_CACHE_USAGE_PERCENTAGE = 0.98;

		};

		/**
		 * Implementation-dependent configuration parameters for the \a nonblocking
		 * backend.
		 *
		 * \note The user documentation only specifies the fields that under some
		 *       circumstances may benefit from a user adapting it. For viewing all
		 *       fields, please see the developer documentation.
		 *
		 * \note Adapting the fields should be done with care and may require
		 *       re-compilation and re-installation of the ALP framework.
		 *
		 * \ingroup nonblockingConfig
		 *
		 * @see grb::config::IMPLEMENTATION
		 */
		template<>
		class IMPLEMENTATION< nonblocking > {

			public:

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
				 * \internal
				 * By default, use the coordinates of the selected backend.
				 *
				 * \note This is an extension that may, at some later stage, be used for
				 *       composability with the #grb::bsp1d and #grb::hybrid backends.
				 * \endinternal
				 */
				static constexpr Backend coordinatesBackend() {
					return nonblocking;
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

		};

	} // namespace config

	/** @} */

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_CONFIG''

