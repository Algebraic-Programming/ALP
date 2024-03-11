
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
 * @file exec_broadcast_routines.hpp
 *
 * Routines used in the Launcher for broadcasting data.
 *
 * @author Alberto Scolari
 * @date August, 2023
 */

#ifndef _H_BSP1D_EXEC_BROADCAST_ROUTINES
#define _H_BSP1D_EXEC_BROADCAST_ROUTINES

#include <stddef.h>

#include <lpf/collectives.h>
#include <lpf/core.h>


namespace grb {

	namespace internal {

		/** Global internal singleton to track whether MPI was initialized. */
		extern bool grb_mpi_initialized;

		/**
		 * Initialize collective communication for broadcast.
		 *
		 * @param[in,out] ctx  Fresh(!) LPF context to work with.
		 * @param[in]     s    This user process ID.
		 * @param[in]     P    Total number of user processes.
		 * @param[in]     regs Total number of memory slot registrations to be made
		 *                     as part of preparing for the broadcast.
		 * @param[out]    coll New collectives context.
		 *
		 * \internal We follow here the LPF convention where output arguments are
		 *           ordered last.
		 */
		lpf_err_t lpf_init_collectives_for_broadcast(
			lpf_t &ctx,
			const lpf_pid_t s, const lpf_pid_t P,
			const size_t regs,
			lpf_coll_t &coll
		);

		/**
		 * Register a memory area as a global one and perform a broadcast.
		 *
		 * @param[in,out] ctx  The LPF context in which \a coll was initialised.
		 * @param[in]     coll The initialised collectives context.
		 * @param[in]     data Pointer to data to broadcast.
		 * @param[in[     size The size of the data (in bytes) to broadcast.
		 */
		lpf_err_t lpf_register_and_broadcast(
			lpf_t &ctx, lpf_coll_t &coll,
			void * const data, const size_t size
		);

	} // end internal

} // end grb

#endif // _H_BSP1D_EXEC_BROADCAST_ROUTINES

