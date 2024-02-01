
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
 * Implements some internals required to for the ALP native interface.
 *
 * This file was split from the preceding <tt>sparseblas.cpp</tt>, which
 * previously implemented the SpBLAS interface as well as the SparseBLAS
 * interface. These implementations required some native functionality of the
 * kind that should never be exposed to end-users.
 *
 * @author A. N. Yzelman
 * @date 1/2/2024
 */

#ifndef _H_NATIVE_INTERNAL
#define _H_NATIVE_INTERNAL

#include <graphblas.hpp>

#include <assert.h>

#include <cstdlib>
#include <algorithm>


namespace grb {

	namespace native {

		/**
		 * Provides a set of functions that are internally required to realise the
		 * native interface.
		 *
		 * These functions are shared not only with the implementation of the native
		 * interface itself, but also with various transition-path libraries.
		 *
		 * Note the anonymous namespace employed here-- this is to avoid clashes
		 * should an application employ multiple transition paths simultaneously.
		 */
		namespace {

			/**
			 * Internal global buffer.
			 *
			 * \note The buffer is kept separate from now as in principle it could occur
			 *       for a single executable to use both ALP directly while
			 *       \em additionally using transition path functionality.
			 *
			 * \note Therefore, similarly, each transition path library should include
			 *       this header so that it generates its own buffer logic in case a
			 *       single user program mixes multiple transition path libraries.
			 */
			char * buffer = nullptr;

			/**
			 * The size of #buffer.
			 */
			size_t buffer_size = 0;

			/**
			 * Provides a standard ALP-like global buffer management, specifically for
			 * retrieving sparse accumulators (SPA) from the global buffer.
			 *
			 * This function is tailored for retrieving a SPA for assisting with matrix
			 * operations, and provides the below three buffers.
			 *
			 * @param[out] bitmask Where the bitmask array is located.
			 * @param[out] stack   Where the stack is located.
			 * @param[out] valbuf  Where the value buffer is located.
			 *
			 * The buffers are guaranteed to hold enough space for the below given
			 * number of elements.
			 *
			 * @param[in] size The maximum number of elements \a bitmask, \a stack, and
			 *                 \a valbuf may hold.
			 *
			 * @returns <tt>false</tt> If and only if buffer retrieval failed.
			 * @returns <tt>true</tt> If and only if buffer retrieval succeeded.
			 *
			 * \warning This function may re-allocate if insufficient capacity was found.
			 */
			template< typename T >
			bool getSPA(
				char * &bitmask, char * &stack, T * &valbuf,
				const size_t size
			) {
				typedef typename grb::internal::Coordinates<
					grb::config::default_backend
				> Coors;
				constexpr const size_t b = grb::config::CACHE_LINE_SIZE::value();

				// catch trivial case
				if( size == 0 ) {
					bitmask = stack = nullptr;
					valbuf = nullptr;
					return true;
				}

				// compute required size
				size_t reqSize = Coors::arraySize( size ) + Coors::stackSize( size ) +
					(size * sizeof(T)) + 3 * b;

				// ensure buffer is at least the required size
				if( buffer == nullptr ) {
					assert( buffer_size == 0 );
					buffer_size = reqSize;
					buffer = static_cast< char * >( malloc( buffer_size ) );
					if( buffer == nullptr ) {
						buffer_size = 0;
						return false;
					}
				} else if( buffer_size < reqSize ) {
					free( buffer );
					buffer_size = std::max( reqSize, 2 * buffer_size );
					buffer = static_cast< char * >( malloc( buffer_size ) );
					if( buffer == nullptr ) {
						buffer_size = 0;
						return false;
					}
				}

				// set buffers and make sure they are aligned
				char * walk = buffer;
				uintptr_t cur_mod = reinterpret_cast< uintptr_t >(walk) % b;
				if( cur_mod > 0 ) {
					walk += (b - cur_mod);
				}
				bitmask = walk;
				walk += Coors::arraySize( size );
				cur_mod = reinterpret_cast< uintptr_t >(walk) % b;
				if( cur_mod > 0 ) {
					walk += (b - cur_mod);
				}
				stack = walk;
				walk += Coors::stackSize( size );
				cur_mod = reinterpret_cast< uintptr_t >(walk) % b;
				if( cur_mod > 0 ) {
					walk += (b - cur_mod);
				}
				valbuf = reinterpret_cast< T * >( walk);

				// done
				return true;
			}

			/**
			 * Frees the current global buffer, if any was allocated.
			 *
			 * Subsequent calls to primitives that return a valid buffer will, when
			 * made directly after a call to this function, simply re-allocate a new
			 * global buffer.
			 */
			void destroyGlobalBuffer() {
				if( buffer != nullptr || buffer_size > 0 ) {
					assert( buffer != nullptr );
					assert( buffer_size > 0 );
					free( buffer );
					buffer_size = 0;
				}
			}

		} // end namespace grb::native::<anon>

	} // end namespace grb::native

} // end namespace grb

#endif // end ifndef _H_NATIVE_INTERNAL

