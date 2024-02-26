
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
 * @date 17th of April, 2017
 */

#include "graphblas/bsp/exec_broadcast_routines.hpp"

#include <atomic>
#include <algorithm>

#include <assert.h>

#include <lpf/collectives.h>
#include <lpf/core.h>


bool grb::internal::grb_mpi_initialized = false;

lpf_err_t grb::internal::lpf_init_collectives_for_broadcast(
	lpf_t &ctx,
	const lpf_pid_t s, const lpf_pid_t P, const size_t max_regs,
	lpf_coll_t &coll
) {
	assert( max_regs >= 2 );
	lpf_err_t brc = lpf_resize_memory_register( ctx, max_regs );
	assert( brc == LPF_SUCCESS );
	// lpf_collectives_init needs at least one slot, and if this call is followed
	// by lpf_register_and_broadcast (as is intended), then at least one more slot
	// is needed.
	brc = lpf_collectives_init( ctx, s, P, 0, 0, 0, &coll );
	assert( brc == LPF_SUCCESS );
	// required messages follows LPF collectives user manual
	const size_t nmsgs = P > 1 ? std::max( P + 1, 2 * P - 3 ) : P + 1;
	brc = lpf_resize_message_queue( ctx, nmsgs );
	assert( brc == LPF_SUCCESS );
	brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
	assert( brc == LPF_SUCCESS );
	return brc;
}

lpf_err_t grb::internal::lpf_register_and_broadcast(
		lpf_t &ctx, lpf_coll_t &coll,
		void * data, size_t size
) {
	lpf_memslot_t global;
	lpf_err_t brc = lpf_register_global( ctx, data, size, &global );
	assert( brc == LPF_SUCCESS );
	// TODO FIXME: double sync for registrations on launcher::exec necessary?
	brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
	assert( brc == LPF_SUCCESS );
	brc = lpf_broadcast( coll, global, global, size, 0 );
	assert( brc == LPF_SUCCESS );
	brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
	assert( brc == LPF_SUCCESS );
	brc = lpf_deregister( ctx, global );
	assert( brc == LPF_SUCCESS );
	return brc;
}

