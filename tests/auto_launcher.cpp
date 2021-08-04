
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

#include <stdio.h>
#include <stdlib.h>

#include <lpf/core.h>
#include <lpf/mpi.h>
#include <mpi.h>
#include <sys/wait.h>
#include <unistd.h>

#define USE1 (void)printf( "Usage: %s <#processes>\n", argv[ 0 ] );
#define USE2                                                                   \
	(void)printf( "  <#processes>  The integer value for #processes. May not " \
				  "be negative. This program must be called #processes times " \
				  "on any number of connected nodes.\n" );

void spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args );

int main( int argc, char ** argv ) {

	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	if( argc != 2 ) {
		USE1 USE2 return 0;
	}

	// read command-line args
	lpf_pid_t P = static_cast< lpf_pid_t >( atoi( argv[ 1 ] ) );

	// input sanity checks
	if( P <= 0 ) {
		(void)fprintf( stderr, "Invalid value for #processes (%s, parsed as %zd).\n", argv[ 1 ], (size_t)P );
		USE2 return 100;
	}

	// prepare args
	int exit_status = 0;
	bool automatic = true;
	lpf_args_t args = { &automatic, sizeof( bool ), &exit_status, sizeof( int ), NULL, 0 };

	// call SPMD
	const lpf_err_t spmdrc = lpf_exec( LPF_ROOT, P, &spmd, args );
	if( spmdrc != LPF_SUCCESS ) {
		(void)printf( "Error in call to lpf_exec.\n" );
		return 200;
	}

	// master process reports test success
	if( exit_status ) {
		(void)printf( "Test FAILED (exit code %d).\n\n", exit_status );
		(void)fflush( stdout );
		(void)fflush( stderr );
		return exit_status;
	}

	(void)printf( "Test OK.\n\n" );
	return fflush( stdout );
}

#undef USE1
#undef USE2
