
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

#include <assert.h>
#include <atomic>

#include <lpf/collectives.h>
#include <lpf/core.h>

#include <graphblas/bsp1d/exec_broadcast_routines.hpp>


bool grb::internal::grb_mpi_initialized = false;

lpf_err_t grb::internal::lpf_init_collectives_for_bradocast( lpf_t & ctx, lpf_coll_t & coll,
	lpf_pid_t s, lpf_pid_t P, size_t max_regs ) {
	lpf_err_t brc = lpf_collectives_init( ctx, s, P, 0, 0, 0, &coll );
	assert( brc == LPF_SUCCESS );
	brc = lpf_resize_message_queue( ctx, (P-1) );
	assert( brc == LPF_SUCCESS );
	brc = lpf_resize_memory_register( ctx, max_regs );
	assert( brc == LPF_SUCCESS );
	brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
	assert( brc == LPF_SUCCESS );
	return brc;
}

lpf_err_t grb::internal::lpf_register_and_broadcast( lpf_t & ctx, lpf_coll_t & coll, void * data, size_t size ) {
	lpf_memslot_t global;
	lpf_err_t brc = lpf_register_global( ctx, data, size, &global );
	assert( brc == LPF_SUCCESS );
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


