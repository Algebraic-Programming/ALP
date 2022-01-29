
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
 * This is a template for testing a fork-based way of starting an initial parallel context for GraphBLAS to use.
 * However, the current version of the PlatformBSP layer does not support this, so this test is currently unused.
 */

#include <lpf/core.h>
#include <lpf/mpi.h>
#include <mpi.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>


const int LPF_MPI_AUTO_INITIALIZE = 0;

static constexpr size_t REQ_P = 3;
static pid_t ids[ REQ_P ];
static bool child;

void spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args ) {
	(void)ctx;
	const pid_t parent_unix_id = *static_cast< const pid_t * >( args.input );
	std::cout << "Hello world from PID " << getpid() << ", "
		<< "which has parent " << parent_unix_id << " "
		<< "and local ID " << static_cast< size_t >(s) << ". "
		<< "This process is part of an SPMD run with "
		<< static_cast< size_t >(P) << "processes.\n";
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	// standard return code: all OK
	int fail = 0;

	// the process ID
	lpf_pid_t s = 0;

	// fork until we have a total of P processes, keep track of process ID
	child = false;
	for( size_t i = 1; !child && i < REQ_P; ++i, ++s ) {
		// track UNIX process ID at the process with SPMD ID 0
		ids[ i ] = fork();
		// track whether we are a child (SPMD ID > 0) or the originating parent process
		child = ( ids[ i ] == 0 );
	}

	// if I am the originating process, my_id should be reset to zero
	if( !child ) {
		s = 0;
	}

	// get parent UNIX PID
	const pid_t parent_unix_id = child ? getppid() : getpid();

	// initialise MPI
	if( MPI_Init( NULL, NULL ) != MPI_SUCCESS ) {
		std::cerr << "MPI_Init returns non-SUCCESS error code." << std::endl;
		return 10;
	}

	// choose a port
	int port = parent_unix_id;
	if( port < 1024 ) {
		port += 1024;
	}

	// try and create a lpf_init_t
	lpf_init_t init;
	char str[ 12 ];
	if( sprintf( str, "%d", port ) < 0 ) {
		std::cout << "Warning: error while transforming the port number "
			<< port << " into a string; result is " << str << ".\n";
	}
	const lpf_err_t initrc = lpf_mpi_initialize_over_tcp( "localhost", str, 3000, s, REQ_P, &init );
	if( initrc != LPF_SUCCESS ) {
		std::cout << "Error in call to lpf_mpi_initialize_over_tcp." << std::endl;
		return 1;
	}

	// prepare args
	lpf_args_t args = { &parent_unix_id, sizeof( pid_t ), NULL, 0, NULL, 0 };

	// call SPMD
	const lpf_err_t spmdrc = lpf_hook( init, &spmd, args );
	if( spmdrc != LPF_SUCCESS ) {
		std::cout << "Error in call to lpf_hook." << std::endl;
		return 2;
	}

	// try and destroy the lpf_init_t
	const lpf_err_t finrc = lpf_mpi_finalize( init );
	if( finrc != LPF_SUCCESS ) {
		std::cout << "Error in call to lpf_mpi_finalize." << std::endl;
		return 3;
	}

	// finalise MPI
	if( MPI_Finalize() != MPI_SUCCESS ) {
		std::cerr << "MPI_Finalize returns non-SUCCESS error code." << std::endl;
		return 20;
	}

	// join process before master exits
	if( !child ) {
		int rc = 0;
		for( size_t i = 1; i < REQ_P; ++i ) {
			// keep waiting until process exits
			do {
				if( waitpid( ids[ i ], &rc, 0 ) == -1 ) {
					std::cerr << "Error on call to waitpid. This program "
						<< " may leave ghost processes, sorry."
						<< std::endl;
					return 30;
				}
			} while( !WIFEXITED( rc ) );
			// check exit code
			rc = WEXITSTATUS( rc );
			// report any nonzero exit code
			if( rc != 0 ) {
				fail = rc;
				std::cout << "Child process " << i << " exited with nonzero "
					<< "exit code " << static_cast< int >(rc) << ".\n";
			}
		}
	} else {
		// child process exits at this point
		return 0;
	}

	// master process reports test success
	if( fail ) {
		std::cout << "Test FAILED\n" << std::endl;
		return fail;
	}

	std::cout << "Test OK\n" << std::endl;
	return 0;
}

