
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

#ifndef _H_GRB_DENSEREF_ALLOC
#define _H_GRB_DENSEREF_ALLOC

#include <iostream>

#include <graphblas/base/alloc.hpp>

#include "config.hpp"

namespace grb {
	namespace utils {
		namespace internal {

			/** 
			 * \internal AllocatorFunctions< reference_dense > is an exact copy of
			 * AllocatorFunctions< reference > as they provide exactly the same functionalities.
			 * TODO: Think if it makes sense to merge them together and, if so, how.
			 */

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
			class AllocatorFunctions< reference_dense > {
			private:
				/** Disable instantiation. */
				AllocatorFunctions() {}

			public:
				/**
				 * Allocates a single chunk of memory.
				 *
				 * @param[in,out] allocd Running accumulation of memory that has been allocated.
				 */
				template< typename T >
				static RC mode_alloc( T * __restrict__ &pointer, const size_t elements,
					const grb::config::ALLOC_MODE mode, utils::AutoDeleter< T > &deleter,
					size_t &allocd
				) {
					// catch trivial case
					if( elements == 0 ) {
						pointer = NULL;
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
						pointer = static_cast< T * >( numa_alloc_interleaved( size ) );
						// check for error
						if( pointer == NULL ) {
							return OUTOFMEM;
						}
						// record appropriate deleter
						deleter = utils::AutoDeleter< T >( pointer, size );
#endif
					} else if( mode == grb::config::ALLOC_MODE::ALIGNED ) {
						// allocate
						void * new_pointer = NULL;
						const int prc = posix_memalign( &new_pointer, grb::config::CACHE_LINE_SIZE::value(), size );
						// check for error
						if( prc == ENOMEM ) {
							return OUTOFMEM;
						}
						if( prc != 0 ) {
							return PANIC;
						}
						// record pointer
						pointer = static_cast< T * >( new_pointer );
						// record appropriate deleter
						deleter = utils::AutoDeleter< T >( pointer, 0 );
					} else {
						// we should never reach this code block
						assert( false );
						return PANIC;
					}
					// final sanity check
					assert( pointer != NULL );
					// record memory taken
					allocd += size;
					// done
					return SUCCESS;
				}

				/** Allocates a single chunk of memory. Wrapper function that relies on the config parameters in #grb::config::MEMORY. */
				template< typename T >
				static RC single_alloc( T * __restrict__ & pointer, const size_t elements, const bool shared, utils::AutoDeleter< T > & deleter, size_t & allocd ) {
					if( shared ) {
						return mode_alloc( pointer, elements, grb::config::IMPLEMENTATION<>::sharedAllocMode(), deleter, allocd );
					} else {
						return mode_alloc( pointer, elements, grb::config::IMPLEMENTATION<>::defaultAllocMode(), deleter, allocd );
					}
				}

				/** Base case for internal::alloc (variadic version). */
				template< typename T >
				static RC alloc( size_t & allocd, T * __restrict__ & pointer, const size_t size, const bool shared, utils::AutoDeleter< T > & deleter ) {
					// try new alloc
					T * __restrict__ new_pointer = NULL;
					utils::AutoDeleter< T > new_deleter;
					RC recursive_error_code = single_alloc( new_pointer, size, shared, new_deleter, allocd );
					// if OK, set output pointer to newly allocated memory
					if( recursive_error_code == SUCCESS ) {
						pointer = new_pointer;
						deleter = new_deleter;
					}
					// done
					return recursive_error_code;
				}

				/** Allocates multiple memory segments in a safe way. */
				template< typename T, typename... Targs >
				static RC alloc( size_t & allocd, T * __restrict__ & pointer, const size_t size, const bool shared, utils::AutoDeleter< T > & deleter, Targs &&... args ) {
					// set local deleter
					utils::AutoDeleter< T > new_deleter;
					// try new alloc
					T * __restrict__ new_pointer = NULL;
					RC recursive_error_code = single_alloc( new_pointer, size, shared, new_deleter, allocd );
					// check for success
					if( recursive_error_code != SUCCESS ) {
						// fail, so propagate
						return recursive_error_code;
					}
					// recurse on remainder arguments
					recursive_error_code = alloc( allocd, std::forward< Targs >( args )... );
					// check for failure
					if( recursive_error_code != SUCCESS ) {
						// fail, so reset old pointer and propagate error code
						return recursive_error_code;
					}
					// all is OK, so finally 1) set pointer to newly allocated memory area and 2) propagate the deleter to user space
					pointer = new_pointer;
					deleter = new_deleter;
					// done
					return SUCCESS;
				}

				/** Helper function that prints allocation information to stdout. */
				static void postAlloc( const RC ret, const size_t allocd, const std::string prefix, const std::string postfix ) {
					if( ret == SUCCESS ) {
						if( config::MEMORY::report( prefix, "allocated", allocd, false ) ) {
							std::cout << postfix << ".\n";
						}
					} else {
						if( config::MEMORY::report( prefix, "failed to allocate", allocd, false ) ) {
							std::cout << postfix << ".\n";
						}
					}
				}
			};

		} // namespace internal
	}     // namespace utils
} // namespace grb

#endif
