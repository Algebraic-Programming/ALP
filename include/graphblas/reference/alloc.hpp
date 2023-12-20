
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
 * @author A. N. Yzelman
 * @date 22nd of January, 2021
 */

#ifndef _H_GRB_ALLOC_REFERENCE
#define _H_GRB_ALLOC_REFERENCE

#include <iostream>

#include <graphblas/base/alloc.hpp>

#include "config.hpp"


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * Provides standard allocation mechanisms using the POSIX and libnuma
			 *   -# posix_memalign() and
			 *   -# numa_alloc_interleaved()
			 * system calls.
			 *
			 * When one of these functions are not available a different allocation
			 * mechanism must be selected.
			 */
			template<>
			class AllocatorFunctions< reference > {

				private:

					/** Disable instantiation. */
					AllocatorFunctions() {}

				public:

					/**
					 * Allocates a single chunk of memory.
					 *
					 * @param[in,out] allocd Running accumulation of memory that has been
					 *                       allocated.
					 */
					template< typename T >
					static RC mode_alloc(
						const size_t elements,
						const grb::config::ALLOC_MODE mode, utils::AutoDeleter< T > &deleter,
						size_t &allocd
					) {
						// catch trivial case
						if( elements == 0 ) {
							deleter.clear();
							return SUCCESS;
						}
						// non-trivial case, first compute size
						const size_t size = elements * sizeof( T );
						// check if the region is supposed to be shared or not
						if( mode == grb::config::ALLOC_MODE::INTERLEAVED ) {
#ifdef _GRB_NO_LIBNUMA
							return UNSUPPORTED;
#else
							// allocate
							T * pointer = static_cast< T * >( numa_alloc_interleaved( size ) );
							// check for error
							if( pointer == NULL ) {
								return OUTOFMEM;
							}
							// record appropriate deleter
							deleter = utils::AutoDeleter< T >( pointer, size,
								utils::AutoDeleter< T >::AllocationType::OPTIMIZED );
#endif
						} else if( mode == grb::config::ALLOC_MODE::ALIGNED ) {
							// allocate
							void * new_pointer = NULL;
							const int prc = posix_memalign(
								&new_pointer, grb::config::CACHE_LINE_SIZE::value(), size );
							// check for error
							if( prc == ENOMEM ) {
								return OUTOFMEM;
							}
							if( prc != 0 ) {
								return PANIC;
							}
							// record pointer
							// pointer = static_cast< T * >( new_pointer );
							// record appropriate deleter
							deleter = utils::AutoDeleter< T >( static_cast< T * >( new_pointer ), size,
								utils::AutoDeleter< T >::AllocationType::SIMPLE );
						} else {
							// we should never reach this code block
							assert( false );
							return PANIC;
						}
						// final sanity check
						// assert( pointer != NULL );
						// record memory taken
						allocd += size;
						// done
						return SUCCESS;
					}

					/**
					 * Allocates a single chunk of memory. Wrapper function that relies on the
					 * config parameters in #grb::config::MEMORY.
					 */
					template< typename T >
					static RC single_alloc(
						const size_t elements,
						const bool shared, utils::AutoDeleter< T > &deleter, size_t &allocd
					) {
						return mode_alloc( elements,
							shared ? grb::config::IMPLEMENTATION<>::sharedAllocMode() :
								grb::config::IMPLEMENTATION<>::defaultAllocMode(),
							deleter, allocd );
					}

					/** Base case for internal::alloc (variadic version). */
					template< typename T >
					static RC alloc(
						size_t &allocd, const size_t size,
						const bool shared, utils::AutoDeleter< T > &deleter
					) {
						// try new alloc
						// T * __restrict__ new_pointer = NULL;
						utils::AutoDeleter< T > new_deleter;
						RC recursive_error_code = single_alloc(
							size, shared, new_deleter, allocd );
						// if OK, set output pointer to newly allocated memory
						if( recursive_error_code == SUCCESS ) {
							// pointer = new_pointer;
							deleter = new_deleter;
						}
						// done
						return recursive_error_code;
					}

			};

			template<>
			class Allocator< reference_omp > {

				private:
					/** Prevent initialisation. */
					Allocator();

				public:
					/** Refer to the standard allocation mechanism. */
					typedef AllocatorFunctions< reference > functions;

			};

		} // namespace internal

	}     // namespace utils

} // namespace grb

#endif

