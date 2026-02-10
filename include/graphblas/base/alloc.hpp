
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
 * @date 13th of September, 2017
 */

#ifndef _H_GRB_BASE_ALLOC
#define _H_GRB_BASE_ALLOC

#include <stdlib.h> //posix_memalign

#include <utility> //std_forward

#include <assert.h>

#include <graphblas/backends.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils/autodeleter.hpp>

#ifndef _GRB_NO_LIBNUMA
 #include <numa.h> // numa_alloc_interleaved
#endif

#ifndef _GRB_NO_STDIO
 #include <iostream> // std::err
#endif

namespace grb {

	namespace utils {

		namespace internal {

			template< enum Backend implementation >
			class AllocatorFunctions {

			private:
				/** Disable instantiation. */
				AllocatorFunctions() {}

			public:

				template< typename T, typename... Targs >
				static RC alloc( size_t &, T * __restrict__ &, const size_t, const bool, utils::AutoDeleter< T > &, Targs &&... ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Error: backend " << implementation << " did not define an allocation mechanism!" << std::endl;
#endif
					return PANIC;
				}

				static void postAlloc( const RC, const size_t, const std::string, const std::string ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Error: backend " << implementation << " did not define an allocation mechanism!" << std::endl;
#endif
				}
			};

			template< enum Backend implementation >
			class Allocator {
			private:
				/** Disable instantiation. */
				Allocator() {}

			public:
				/** Forward to the given backend's allocation functions. */
				typedef internal::AllocatorFunctions< implementation > functions;
			};

		} // namespace internal

	}     // namespace utils

} // namespace grb

// define user API:
namespace grb {

	namespace utils {

		/**
		 * Allocates a single memory area.
		 *
		 * This is the trivial version of the other #grb::alloc function which is
		 * designed to allocate multiple memory areas simultaneously.
		 *
		 * If the requested memory is significant, output shall be printed to
		 * <tt>stdout</tt>. See #grb::config::MEMORY on what is deemed a significant
		 * amount of memory. This output is given both on success and failure of the
		 * allocations. The line is formatted as follows:
		 * <tt>Info: <prefix> allocated xxx bytes/kB/MB/GB/TB, <postfix>.</tt>
		 * or, in case of a failed call:
		 * <tt>Info: <prefix> failed to allocate xxx bytes/kB/MB/GB, <postfix>.</tt>
		 *
		 * On a successful call this function returns a #grb::utils::AutoDeleter in
		 * \a deleter. On failure, the \a deleter on input remains unchanged (if it
		 * was default-instantiated, it will not free anything when destructed).
		 *
		 * The strategy for allocation (and therefore how this function constructs the
		 * corresponding #grb::utils::AutoDeleter) depends on whether the memory
		 * segment will be shared by any underlying threads. If yes, the memory area
		 * \em may be allocated in an interleaved fashion, depending on the value of
		 * #grb::config::MEMORY::sharedAllocMode. Otherwise, memory is allocated
		 * according to the value of #grb::config::MEMORY::defaultAllocMode.
		 *
		 * @tparam T The type of elements that need to be stored at \a pointer.
		 * @tparam implementation Which backend the allocator is for.
		 *
		 * \note \a implementation becomes relevant if a backend targets a specific
		 *       accelerator or embedded architecture.
		 *
		 * @param[out] prefix  Used for reporting on large (attempted) memory
		 *                     allocation.
		 * @param[out] postfix Used for reporting on large (attempted) memory
		 *                     allocation.
		 * @param[out] pointer Where a pointer to the requested memory segment should
		 *                     be stored. The value will not be written to if this
		 *                     function did not return #grb::RC::SUCCESS.
		 * @param[in]  size    The size (in number of elements of type \a T) of the
		 *                     memory area to be allocated.
		 * @param[in]  shared  Whether the memory area is to be used by multiple
		 *                     threads.
		 * @param[out] deleter Where an #grb::utils::AutoDeleter corresponding to the
		 *                     allocated memory segment should be stored. The value
		 *                     will not be written to if this function did not return
		 *                     #grb::RC::SUCCESS.
		 *
		 * @returns SUCCESS    When the function has completed as expected. When any
		 *                     other code is returned, then the following fields shall
		 *                     remain untouched: \a pointer and \a deleter.
		 * @returns OUTOFMEM   When there was not sufficient memory available to
		 *                     allocate the requested memory areas.
		 * @returns PANIC      When any non-mitigable failure was encountered.
		 */
		template< typename T, enum Backend implementation >
		RC alloc(
			const std::string prefix, const std::string postfix,
			T * __restrict__ &pointer, const size_t size, const bool shared,
			utils::AutoDeleter< T, implementation > &deleter
		) {
			size_t allocd = 0;
			const RC ret = internal::Allocator< implementation >::functions::alloc( allocd, pointer, size, shared, deleter );
			internal::Allocator< implementation >::functions::postAlloc( ret, allocd, prefix, postfix );
			return ret;
		}

		/**
		 * Allocates several memory areas in one combined allocation primitive.
		 *
		 * If one of the allocations fails, all previously successful allocations
		 * are freed and the call returns an appropriate error code. Other than the
		 * returned code the state of the program shall be as though the call to this
		 * function was never made; in particular, it is guaranteed none of the output
		 * fields will have been written to.
		 *
		 * If the sum of all requested memory areas is significant, output shall be
		 * printed to <tt>stdout</tt>. See #grb::config::MEMORY on what is deemed a
		 * significant amount of memory. This output is given both on success and
		 * failure of the allocations. The line is formatted as follows:
		 * <tt>Info: <prefix> allocated xxx bytes/kB/MB/GB/TB, <postfix>.</tt>
		 * or, in case of a failed call:
		 * <tt>Info: <prefix> failed to allocate xxx bytes/kB/MB/GB, <postfix>.</tt>
		 *
		 * On a successful call, the function returns a #grb::utils::AutoDeleter for
		 * each requested memory segment. On failure, the given \a deleters remain
		 * unchanged (and hence, if they were default-instantiated, will not free
		 * anything on deletion).
		 *
		 * The strategy for allocation (and therefore how this function constructs the
		 * corresponding #grb::utils::AutoDeleter) depends on whether the memory
		 * segment will be shared by any underlying threads. If yes, the memory area
		 * \em may be allocated in an interleaved fashion, depending on the value of
		 * #grb::config::MEMORY::sharedAllocMode. Otherwise, memory is allocated
		 * according to the value of #grb::config::MEMORY::defaultAllocMode.
		 *
		 * @tparam T The type of elements that need to be stored at \a pointer.
		 * @tparam implementation Which backend the allocator is for.
		 *
		 * \note \a implementation becomes relevant if a backend targets a specific
		 *       accelerator or embedded architecture.
		 *
		 * The request for multiple memory areas in one call employs variadic
		 * arguments:
		 *
		 * @tparam Targs Any remainder typenames to any number of other memory regions
		 *               that need to be allocated.
		 *
		 * A single allocation consists of a group of six arguments:
		 *
		 * @param[out] prefix  Used for reporting on large (attempted) memory
		 *                     allocation.
		 * @param[out] postfix Used for reporting on large (attempted) memory
		 *                     allocation.
		 * @param[out] pointer Where a pointer to the requested memory segment should
		 *                     be stored. The value will not be written to if this
		 *                     function did not return #grb::RC::SUCCESS.
		 * @param[in]  size    The size (in number of elements of type \a T) of the
		 *                     memory area to be allocated.
		 * @param[in]  shared  Whether the memory area is to be used by multiple
		 *                     threads.
		 * @param[out] deleter Where an #grb::utils::AutoDeleter corresponding to the
		 *                     allocated memory segment should be stored. The value
		 *                     will not be written to if this function did not return
		 *                     #grb::RC::SUCCESS.
		 *
		 * If another allocation is to follow the present one, it should be provided
		 * as a separate group of the above six arguments:
		 *
		 * @param[in,out] args Any remainder arguments to any number of other memory
		 *                     regions that need to be allocated.
		 *
		 * @returns SUCCESS    When the function has completed as expected. When any
		 *                     other code is returned, then the following fields shall
		 *                     remain untouched: \a pointer, \a deleter, plus all
		 *                     instances of those two fields in \a args.
		 * @returns OUTOFMEM   When there was not sufficient memory available to
		 *                     allocate one of the requested memory areas.
		 * @returns PANIC      When any other non-mitigable failure encountered during
		 *                     allocation.
		 */
		template< typename T, enum Backend implementation, typename... Targs >
		RC alloc(
			const std::string prefix, const std::string postfix,
			T * __restrict__ &pointer, const size_t size,
			const bool shared,
			utils::AutoDeleter< T, implementation > &deleter,
			Targs &&... args
		) {
			size_t allocd = 0;
			const RC ret = internal::Allocator< implementation >::functions::alloc(
				allocd, pointer, size, shared,
				deleter, std::forward< Targs >( args )...
			);
			internal::Allocator< implementation >::functions::postAlloc(
				ret, allocd, prefix, postfix
			);
			return ret;
		}

	} // namespace utils

} // namespace grb

#endif // _H_GRB_BASE_ALLOC

