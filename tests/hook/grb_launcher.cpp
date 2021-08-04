
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

#include <assert.h>

#include "graphblas.hpp"
#include "lpf/core.h"

// forward declaration of the GraphBLAS program
void grbProgram( const size_t s, const size_t P, int & exit_status );

constexpr size_t outer_loop = 5;
constexpr size_t inner_loop = 3;

void spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	int exit_status_dummy = 0;

	// set exit status to local memory location by default
	int & exit_status = exit_status_dummy;

	// sanity check on args: s == 0 must always have sizeof(bool) input in args
	assert( s > 0 || args.input_size == sizeof( bool ) );

	// check if started from a lpf_exec
	bool automatic;
	if( args.input_size == sizeof( bool ) ) {
		// copy automatic variable
		automatic = *static_cast< const bool * >( args.input );
		// if s > 0 and we are in this branch, then we must be started via hook
		assert( s == 0 || ! automatic );
	} else {
		// this means we have s > 0 and have started from exec
		automatic = true;
		assert( s > 0 );
	}

	// if we started from lpf_hook, we take exit status from args
	if( ! automatic ) {
		// yes, so take exit status from args
		exit_status = *static_cast< int * >( args.output );
	}

	// initialise GraphBLAS
	enum grb::RC rc = grb::init( s, P, ctx );
	if( rc != grb::SUCCESS ) {
		exit_status = 10;
	}

	// run GraphBLAS program
	grbProgram( s, P, exit_status );

	// finalize GraphBLAS
	rc = grb::finalize();
	if( rc != grb::SUCCESS ) {
		exit_status = 20;
	}

	// when doing a parallel run
#ifndef GRB_LAUNCH_SEQUENTIAL
	// if started automatically (using lpf_exec)
	if( automatic ) {
		// do allreduce on exit status
		lpf_err_t brc = lpf_resize_message_queue( ctx, P );
		if( brc != LPF_SUCCESS ) {
			exit_status = 30;
			return;
		}
		brc = lpf_resize_memory_register( ctx, 2 );
		if( brc != LPF_SUCCESS ) {
			exit_status = 35;
			return;
		}
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		if( brc != LPF_SUCCESS ) {
			exit_status = 40;
			return;
		}
		int * status = NULL;
		if( s == 0 ) {
			assert( P > 0 );
			status = new int[ P ];
			if( status == NULL ) {
				exit_status = 45;
				return;
			}
		}
		lpf_memslot_t destination = LPF_INVALID_MEMSLOT;
		lpf_memslot_t source = LPF_INVALID_MEMSLOT;
		if( s == 0 ) {
			brc = lpf_register_global( ctx, status, P * sizeof( int ), &destination );
		} else {
			brc = lpf_register_global( ctx, status, 0, &destination );
		}
		if( brc != LPF_SUCCESS ) {
			exit_status = 50;
			return;
		}
		if( lpf_sync( ctx, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
			exit_status = 52;
			return;
		}
		brc = lpf_register_local( ctx, &exit_status, sizeof( int ), &source );
		if( brc != LPF_SUCCESS ) {
			exit_status = 55;
			return;
		}
		brc = lpf_put( ctx, source, 0, 0, destination, s * sizeof( int ), sizeof( int ), LPF_MSG_DEFAULT );
		if( brc != LPF_SUCCESS ) {
			exit_status = 60;
			return;
		}
		brc = lpf_sync( ctx, LPF_SYNC_DEFAULT );
		if( brc != LPF_SUCCESS ) {
			exit_status = 70;
			return;
		}
		for( lpf_pid_t k = 0; exit_status == 0 && k < P && s == 0; ++k ) {
			exit_status = status[ k ];
		}
		brc = lpf_deregister( ctx, destination );
		if( brc != LPF_SUCCESS ) {
			exit_status = 80;
			return;
		}
		if( s == 0 ) {
			delete[] status;
			status = NULL;
		}
	}
#endif // end ``ifndef GRB_LAUNCH_SEQUENTIAL''

	// done
}
