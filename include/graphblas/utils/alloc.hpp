
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

#ifndef _H_GRB_UTILS_ALLOC
#define _H_GRB_UTILS_ALLOC

#include <stdlib.h> //posix_memalign

#include <utility> //std_forward

#include <assert.h>

#include <graphblas/base/alloc.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils/autodeleter.hpp>

namespace grb {
	namespace utils {
		namespace internal {

			// template< enum Backend implementation = config::default_backend >
			// class Allocator;

		} // namespace internal
	}     // namespace utils
} // namespace grb

// include available allocator implementations:
#ifdef _GRB_WITH_REFERENCE
 #include "graphblas/reference/alloc.hpp"
#endif
#ifdef _GRB_WITH_DENSEREF
 #include "graphblas/denseref/alloc.hpp"
#endif
#ifdef _GRB_WITH_LPF
 #include "graphblas/bsp1d/alloc.hpp"
#endif
#ifdef _GRB_WITH_BANSHEE
 #include "graphblas/banshee/alloc.hpp"
#endif

// define user API:
namespace grb {
	namespace utils {

		/**
		 * Allocates a single memory area. This is the trivial version of the other
		 * #grb::utils::alloc function which is designed to allocate multiple memory
		 * areas in one go; please see that function for extended documentation.
		 *
		 * @see grb::utils::alloc.
		 */
		/*
		template< typename T >
		inline RC alloc( const std::string prefix, const std::string postfix, T * __restrict__ & pointer, const size_t size, const bool shared, utils::AutoDeleter< T > & deleter ) {
		    return alloc< T, config::default_backend >( prefix, postfix, pointer, size, shared, deleter );
		}
		*/

		/**
		 * Allocate a bunch of memory areas in one go.
		 *
		 * If one of the allocations fails, all previously successful allocations
		 * are rewound and the call returns an appropriate error code. Other than the
		 * returned code the state of the program shall be as though the call to this
		 * function was never made.
		 *
		 * If the sum of all requested memory areas is significant, output shall be
		 * printed to stdout. See #grb::config::MEMORY on what is deemed a
		 * significant amount of memory.
		 * In such a case, output is given both on success and failure of the
		 * aggregate allocations. The line is formatted as follows:
		 * <tt>Info: <prefix> allocated xxx bytes/kB/MB/GB/TB, <postfix>.</tt>
		 * or, in case of a failed call:
		 * <tt>Info: <prefix> failed to allocate xxx bytes/kB/MB/GB, <postfix>.</tt>
		 *
		 * On #grb::RC::SUCCESS, the function returns a #grb::utils::AutoDeleter for
		 * requested memory segment. On failure, the given autodeleters remain
		 * unchanged (and hence, if they were default-instantiated, will not free
		 * anything on deletion).
		 *
		 * The strategy for allocation (and therefore how the #grb::utils::AutoDeleter
		 * is constructed) depends on whether the memory segment will be shared by any
		 * underlying threads. If yes, the memory area \em may be allocated in an
		 * interleaved fashion, depending on the value of
		 * #grb::config::MEMORY::sharedAllocMode. Otherwise, memory is allocated
		 * according to the value of #grb::config::MEMORY::defaultAllocMode.
		 *
		 * @tparam T     The type of elements that need to be stored at \a pointer.
		 * @tparam Targs Any remainder typenames to any number of other memory regions
		 *               that need to be allocated.
		 *
		 * @param[out] pointer Where a pointer to the requested memory segment should
		 *                     be stored. The value will not be written to if this
		 *                     function did not return #grb::RC::SUCCESS.
		 * @param[out] prefix  Used for reporting on large (attempted) memory
		 *                     allocation.
		 * @param[out] postfix Used for reporting on large (attempted) memory
		 *                     allocation.
		 * @param[in]  size    The size (in number of elements of type \a T) of the
		 *                     memory area to be allocated.
		 * @param[in]  shared  Whether the memory area is to be used by multiple
		 *                     threads.
		 * @param[out] deleter Where an #grb::utils::AutoDeleter corresponding to the
		 *                     allocated memory segment should be stored. The value
		 *                     will not be written to if this function did not return
		 *                     #grb::RC::SUCCESS.
		 * @param[in,out] args Any remainder arguments to any number of other memory
		 *                     regions that need to be allocated.
		 *
		 * @returns SUCCESS    When the function has completed as expected. When any
		 *                     other code is returned, then the following fields shall
		 *                     remain untouched: \a pointer, \a deleter, plus all
		 *                     instances of those two fields in \a args.
		 * @returns OUTOFMEM   When there was not sufficient memory available to
		 *                     allocate the requested memory areas.
		 * @returns PANIC      When any other non-mitigable failure was found during
		 *                     allocation.
		 */

		/*
		template< typename T, typename... Targs >
		inline RC alloc( const std::string prefix, const std::string postfix, T * __restrict__ & pointer, const size_t size, const bool shared, utils::AutoDeleter< T > & deleter, Targs &&... args ) {
		    return alloc< T, config::default_backend, Targs... >( prefix, postfix, pointer, size, shared, deleter );
		}
		*/

	} // namespace utils
} // namespace grb

#endif

