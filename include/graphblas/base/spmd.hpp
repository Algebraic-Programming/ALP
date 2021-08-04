
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
 * @date 28th of April, 2017
 */

#ifndef _H_GRB_BASE_SPMD
#define _H_GRB_BASE_SPMD

#include <cstddef> //size_t

#include <stdint.h> // SIZE_MAX

#include <graphblas/backends.hpp>
#include <graphblas/rc.hpp>

#include "config.hpp"

namespace grb {

	/** \todo documentation */
	template< Backend implementation >
	class spmd {

	public:
#if 0
			typedef void * memslot_t;

			/**
			 * Registers a memory area for global communication. This is a collective
			 * call. A call to this function is always followed by a matching call to
			 * #deregister. A memory area is defined by a \a pointer and a \a size. Any
			 * two calls to this function will never have an argument where its pointer
			 * is within any other registered memory area, \em and will never make it so
			 * that a previously registered pointer becomes embedded in its own registered
			 * memory area.
			 *
			 * \warning If any of the above conditions are violated, undefined behaviour
			 *          will occur.
			 *
			 * @param[in] pointer The start of the memory area to be registered. Must not
			 *                    be within any previously registered (but not yet
			 *                    deregistered) memory area.
			 * @param[in] size    The size of the memory area to be registered. Must not
			 *                    be such that any previously registered (but not yet de-
			 *                    registered) memory area \a x becomes embedded in the
			 *                    memory area defined by \a pointer and this \a size.
			 *
			 * @return The memory registry.
			 *
			 * A call to this function never fails.
			 */
			memslot_t create_slot( void * const pointer, const size_t size );
	
			/**
			 * De-registers a memory area for global communication. This is a collective
			 * call. The argument must match across all user processes. A call to this
			 * function must follow exactly one corresponding call to #register.
			 *
			 * \warning Non-collective calls will result in undefined behaviour. A call
			 *          using an invalid \a registration will of course also result in
			 *          undefined behaviour.
			 *
			 * @param[in] registration The memory registration object obtained from
			 *                         \a push_reg.
			 *
			 * A call to this function never fails.
			 */
			void destroy_slot( const memslot_t registration );
#endif

		/** @return The number of user processes in this GraphBLAS run. */
		static inline size_t nprocs() noexcept {
			return 0;
		}

		/** @return The user process ID. */
		static inline size_t pid() noexcept {
			return SIZE_MAX;
		}

		/**
		 * Calls a PlatformBSP \a bsp_sync.
		 *
		 * @param[in] msgs_in  The maximum number of messages to be received across
		 *                     \em all user processes. Default is zero.
		 * @param[in] msgs_out The maximum number of messages to be sent across
		 *                     \em all user processes. Default is zero.
		 *
		 * If both \a msgs_in and \a msgs_out are zero, the values will be
		 * automatically inferred. This requires a second call to the PlatformBSP
		 * \a bsp_sync primitive, thus increasing the latency by at least \f$ l \f$.
		 *
		 * If the values for \a msgs_in or \a msgs_out are underestimated, undefined
		 * behaviour will occur. If this is not the case but one or more are instead
		 * \a over estimated, this call will succeed as normal.
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		static enum RC sync( const size_t msgs_in = 0, const size_t msgs_out = 0 ) noexcept {
			(void)msgs_in;
			(void)msgs_out;
			return PANIC;
		}

	}; // end class ``spmd''

} // namespace grb

#endif // end _H_GRB_BASE_SPMD
