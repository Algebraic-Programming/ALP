
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
 */

#ifndef _H_GRB_LPF_SPMD
#define _H_GRB_LPF_SPMD

#include <cstddef> //size_t

#include <lpf/core.h>

#include <graphblas/base/spmd.hpp>
#include <graphblas/bsp1d/init.hpp>

namespace grb {

	/** Superclass implementation for all LPF-backed implementations. */
	template<>
	class spmd< GENERIC_BSP > {

	public:

		/** @return The number of user processes in this GraphBLAS run. */
		static inline size_t nprocs() noexcept {
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			return data.P;
		}

		/** @return The user process ID. */
		static inline size_t pid() noexcept {
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			return data.s;
		}

		/**
		 * Ensures execution holds until all communication this process is involved
		 * with has completed.
		 *
		 * @param[in] msgs_in  The maximum number of messages to be received across
		 *                     \em all user processes. Default is zero.
		 * @param[in] msgs_out The maximum number of messages to be sent across
		 *                     \em all user processes. Default is zero.
		 *
		 * If both \a msgs_in and \a msgs_out are zero, the values will be
		 * automatically inferred. This requires a second call to the LPF
		 * \a lpf_sync primitive, thus increasing the latency by at least \f$ l \f$.
		 *
		 * If the values for \a msgs_in or \a msgs_out are underestimated, undefined
		 * behaviour will occur. If this is not the case but one or more are instead
		 * \a over estimated, this call will succeed as normal.
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		static RC sync( const size_t msgs_in = 0, const size_t msgs_out = 0 ) noexcept {
			(void)msgs_in;
			(void)msgs_out;
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			const lpf_err_t rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
			if( rc == SUCCESS ) {
				return SUCCESS;
			} else {
				return PANIC;
			}
		}

		/**
		 * Executes a barrer between this process and all its siblings.
		 *
		 * @return grb::SUCCESS When all queued communication is executed succesfully.
		 * @return grb::PANIC   When an unrecoverable error occurs. When this value is
		 *                      returned, the library enters an undefined state.
		 */
		static inline RC barrier() noexcept {
			return sync();
		}

	}; // end class ``spmd'' generic LPF implementation

} // namespace grb

#endif // end _H_GRB_LPF_SPMD
