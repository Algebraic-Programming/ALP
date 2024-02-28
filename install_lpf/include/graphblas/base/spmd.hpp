
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
 * Exposes facilities for direct SPMD programming
 *
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

	/**
	 * For backends that support multiple user processes this class defines some
	 * basic primitives to support SPMD programming.
	 *
	 * All backends must implement this interface, including backends that do not
	 * support multiple user processes. The interface herein defined hence ensures
	 * to allow for trivial implementations for single user process backends.
	 */
	template< Backend implementation >
	class spmd {

		public:

			/** @return The number of user processes in this ALP run. */
			static inline size_t nprocs() noexcept {
				return 0;
			}

			/** @return The ID of this user process. */
			static inline size_t pid() noexcept {
				return SIZE_MAX;
			}

			/**
			 * \internal
			 * Provides functionalities similar to the LPF primitive \a lpf_sync,
			 * enhanced with zero-cost synchronisation semantics.
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
			 *
			 * \todo If exposing this API, there should also be exposed a mechanism for
			 *       initiating messages.
			 * \endinternal
			 */
			static enum RC sync( const size_t msgs_in = 0, const size_t msgs_out = 0 ) noexcept {
				(void) msgs_in;
				(void) msgs_out;
				return PANIC;
			}

	}; // end class ``spmd''

} // namespace grb

#endif // end _H_GRB_BASE_SPMD

