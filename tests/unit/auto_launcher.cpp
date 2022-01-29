
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

#include <sys/wait.h>
#include <unistd.h>

#include <iostream>

#include <graphblas.hpp>

#define USE1 std::cout << "Usage: " << argv[ 0 ] << " <#processes>\n";
#define USE2 std::cout << "  <#processes>  The integer value for #processes. May not " \
                          "be negative. This program must be called #processes times " \
			  "on any number of connected nodes.\n";

extern void grbProgram( const size_t &, int & );

int main( int argc, char ** argv ) {

	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	if( argc != 2 ) {
		USE1 USE2 return 0;
	}

	// read command-line args
	const size_t P = static_cast< size_t >( atoi( argv[ 1 ] ) );

	// input sanity checks
	if( P <= 0 ) {
		std::cerr << "Invalid value for #processes (" << argv[1] << ", "
			<< "parsed as " << static_cast< size_t >(P) << ".\n";
		USE2 return 100;
	}

	// prepare launcher and output field
	int exit_status = 0;
	grb::Launcher< grb::AUTOMATIC > launcher;

	// run
	if( launcher.exec( &grbProgram, P, exit_status ) != grb::SUCCESS ) {
		std::cout << "Test FAILED (launcher did not return SUCCESS).\n" << std::endl;
		return 200;
	}

	// master process reports test success
	if( exit_status ) {
		std::cout << "Test FAILED (exit code " << exit_status << ").\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}

	// done
	return exit_status;
}

#undef USE1
#undef USE2

