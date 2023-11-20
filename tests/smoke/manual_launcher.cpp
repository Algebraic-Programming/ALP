
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

#include <stdlib.h>

#include <lpf/core.h>
#include <lpf/mpi.h>
#include <mpi.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>

#include <graphblas/utils/ranges.hpp> //is_geq_zero


const int LPF_MPI_AUTO_INITIALIZE = 0;

#define USE1 std::cout << "Usage: " << argv[ 0 ] << " <Host server> <PID> <#processes> <port>\n";
#define USE2 std::cout << "  <host server> This argument can be a string (e.g., "      \
                          "`localhost') or an IP address, just to give two examples. " \
                          "The host name must resolve, at all program calls with "     \
                          "<PID> larger than zero, to the node that calls this "       \
                          "program with <PID> equal to zero.\n";
#define USE3 std::cout << "     <PID>      The integer value for <PID> must be in "   \
                          "between 0 (inclusive) and #processes (exclusive). This "   \
                          "value must be unique amongst all of the #processes calls " \
                          "to this program.\n";
#define USE4 std::cout << "  <#processes>  The integer value for #processes. May not " \
                          "be negative. This program must be called #processes times " \
                          "on any number of connected nodes.\n";
#define USE5 std::cout << "     <port>     This argument must either be a service "    \
                          "name or a port number. This port will be opened by the "    \
                          "program with <PID> equal to zero. All other programs will " \
                          "attempt to connect to program by opening TCP connections "  \
                          "to host:port. The time-out for all connection requests to " \
                          "arrive is set to 30 seconds.\n";

void spmd( lpf_t ctx, lpf_pid_t s, lpf_pid_t P, lpf_args_t args );

int main( int argc, char ** argv ) {

	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	if( argc != 5 ) {
		USE1 USE2 USE3 USE4 USE5
		return 0;
	}

	// read command-line args
	char * const host = argv[ 1 ];
	lpf_pid_t s = static_cast< lpf_pid_t >( atoi( argv[ 2 ] ) );
	lpf_pid_t P = static_cast< lpf_pid_t >( atoi( argv[ 3 ] ) );
	char * const port = argv[ 4 ];

	// input sanity checks
	if( host == NULL || host[ 0 ] == '\0' ) {
		std::cerr << "Invalid hostname: " << argv[ 1 ] << std::endl;
		USE2
		return 100;
	}
	if( !grb::utils::is_geq_zero( P ) ) {
		std::cerr << "Invalid value for #processes: " << argv[ 3 ] << ", "
			"parsed as " << static_cast< size_t >(P) << "." << std::endl;
		USE3
		return 200;
	}
	if( !grb::utils::is_in_normalized_range( s, P ) ) {
		std::cerr << "Invalid value for PID: " << argv[ 2 ] << ", "
			<< "parsed as " << static_cast< size_t >(s) << "."
			<< std::endl;
		USE4
		return 300;
	}
	if( port == NULL || port[ 0 ] == '\0' ) {
		std::cerr << "Invalid value for port name or number: "
			<< argv[ 4 ] << "." << std::endl;
		USE5
		return 400;
	}

	// initialise MPI
	if( MPI_Init( NULL, NULL ) != MPI_SUCCESS ) {
		std::cerr << "MPI_Init returns with non-SUCCESS exit code." << std::endl;
		return 10;
	}

	// try and create a lpf_init_t
	lpf_init_t init;
	const lpf_err_t initrc = lpf_mpi_initialize_over_tcp( host, port, 30000, s, P, &init );
	if( initrc != LPF_SUCCESS ) {
		std::cerr << "Error in call to lpf_mpi_initialize_over_tcp." << std::endl;
		return 500;
	}

	// prepare args
	int exit_status = 0;
	bool automatic = false;
	lpf_args_t args = { &automatic, sizeof( bool ), &exit_status, sizeof( int ), NULL, 0 };

	// print debug message
	if( initrc == LPF_SUCCESS ) {
		std::cout << "Initialisation complete, calling hook...\n";
	} else {
		std::cout << "Test FAILED: failed to initialize for lpf_hook." << std::endl;
		return 550;
	}

	// call SPMD
	const lpf_err_t spmdrc = lpf_hook( init, &spmd, args );
	if( spmdrc != LPF_SUCCESS ) {
		std::cout << "Test FAILED: error in call to lpf_hook." << std::endl;
		return 600;
	}

	// try and destroy the lpf_init_t
	const lpf_err_t finrc = lpf_mpi_finalize( init );
	if( finrc != LPF_SUCCESS ) {
		std::cout << "Test FAILED: error in call to lpf_mpi_finalize." << std::endl;
		return 700;
	}

	// finalise MPI
	if( MPI_Finalize() != MPI_SUCCESS ) {
		std::cout << "Test FAILED: error in call to MPI_Finalize()." << std::endl;
		return 20;
	}

	// master process reports test success
	if( exit_status ) {
		std::cout << "Test FAILED with exit code " << exit_status << "\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return exit_status;
}

#undef USE1
#undef USE2
#undef USE3
#undef USE4
#undef USE5

