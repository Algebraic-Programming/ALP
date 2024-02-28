
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
 * Provides the initialisation and finalisation routines for the nonblocking
 * backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_INIT
#define _H_GRB_NONBLOCKING_INIT

#include <graphblas/base/init.hpp>
#include <graphblas/utils/DMapper.hpp>


namespace grb {

	template<>
	RC init< nonblocking >( const size_t, const size_t, void * const );

	template<>
	RC finalize< nonblocking >();

	namespace internal {

		/** Internal state of the nonblocking backend. */
		class NONBLOCKING {

			friend RC init< nonblocking >( const size_t, const size_t, void * const );

			private:

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


			public:

				/**
				 * When <tt>true</tt>, calling a fake nonblocking primitive for a first time
				 * will emit a warning to the standard error stream.
				 */
				static bool warn_if_not_native;

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

				/**
				 * Whether the tile size is manually set by the user or not.
				 */
				static bool isManualTileSize() {
					return manual_tile_size;
				}

				/**
				 * The tile size that is manually selected by the user.
				 */
				static size_t manualFixedTileSize() {
					return manual_fixed_tile_size;
				}

				/**
				 * The maximum number of threads available in the system.
				 */
				static size_t numThreads() {
					return num_threads;
				}

		};

	}

} // namespace grb

#endif //``end _H_GRB_NONBLOCKING_INIT''

