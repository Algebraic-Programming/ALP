
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

#ifndef _H_GRB_ALLOC_BANSHEE
#define _H_GRB_ALLOC_BANSHEE

#include "graphblas/utils/autodeleter.hpp"

#include "snrt.h"

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
			static size_t * banshee_memory_offset = nullptr;
			const int padding = 4;

			template<>
			class AllocatorFunctions< banshee > {
			private:
				/** Disable instantiation. */
				AllocatorFunctions() {};

			public:
				/**
				 * Allocates a single chunk of memory.
				 *
				 * Wrapper function that relies on the config parameters in
				 * #grb::config::MEMORY.
				 */

				template< typename T >
				static RC single_alloc( T * __restrict__ & pointer, const size_t elements, const bool shared, utils::AutoDeleter< T > & deleter, size_t & allocd ) {

					(void)shared;
					// catch trivial case
					if( elements == 0 ) {
						pointer = NULL;
						return SUCCESS;
					}

					// non-trivial case, first compute size
					size_t size = elements * sizeof( T );

					// Padding to memory alignment
					int remainder = size % padding;
					if( remainder != 0 )
						size += padding - remainder;

					if( snrt_global_core_idx() == 0 ) {

						// Actual memory allocation
						pointer = (T *)( (size_t)snrt_cluster_memory().start + (size_t)banshee_memory_offset );
						banshee_memory_offset = (size_t *)( (size_t)banshee_memory_offset + (size_t)size );

						// check for error
						if( (size_t)banshee_memory_offset > (size_t)snrt_cluster_memory().end ) {
							printf( "No more cluster memory available\n" );
							return OUTOFMEM;
						}
#ifdef _DEBUG
						printf( "For size %d, allocated memory from %p to %p\n", size, pointer, banshee_memory_offset );
#endif

						// record appropriate deleter
						deleter = utils::AutoDeleter< T >( pointer, size );
						// record memory taken
						allocd += size;

						// Distribute the data descriptor to the other cores.
						snrt_bcast_send( pointer, size );
					} else {
						// Receive the data descriptor from the main core.
						snrt_bcast_recv( pointer, size );
					}

					// done
					return SUCCESS;
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
					// all is OK, so finally 1) set pointer to newly allocated
					// memory area and 2) propagate the deleter to user space
					pointer = new_pointer;
					deleter = new_deleter;
					// done
					return SUCCESS;
				}

				/** Helper function that prints allocation information to stdout. */
				static void postAlloc( const RC ret, const size_t allocd, const std::string prefix, const std::string postfix ) {
					(void)ret;
					(void)allocd;
					(void)prefix;
					(void)postfix;
				}
			};

		} // namespace internal
	}     // namespace utils
} // namespace grb

namespace grb {
	namespace utils {
		template< typename T, typename... Targs >
		RC alloc( T * __restrict__ & pointer, const size_t size, const bool shared, utils::AutoDeleter< T > & deleter, Targs &&... args ) {
			size_t allocd = 0;

			const RC ret = internal::Allocator<>::functions::alloc( allocd, pointer, size, shared, deleter, std::forward< Targs >( args )... );
			return ret;
		}

	} // namespace utils
} // namespace grb

// Workaround allocator used for vector and other template function
template< class T >
class banshee_allocator {
public:
	using value_type = T;

	banshee_allocator() noexcept {} // not required, unless used
	template< class U >
	banshee_allocator( banshee_allocator< U > const & ) noexcept {}

	value_type * // Use pointer if pointer is not a value_type*
	allocate( std::size_t size ) {
		value_type * pointer = nullptr;
		grb::utils::AutoDeleter< T > _raw_deleter;

		grb::utils::alloc( pointer, size, false, _raw_deleter );
		return pointer;
	}

	void deallocate( value_type * p,
		std::size_t ) noexcept // Use pointer if pointer is not a value_type*
	{
		(void)p;
	}
};

template< class T, class U >
bool operator==( banshee_allocator< T > const &, banshee_allocator< U > const & ) noexcept {
	return true;
}

template< class T, class U >
bool operator!=( banshee_allocator< T > const & x, banshee_allocator< U > const & y ) noexcept {
	return ! ( x == y );
}

#endif
