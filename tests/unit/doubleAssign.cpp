
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

#include <iostream>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	assert( n > 0 );
	{
		Vector< double > a( n ), b( n );
		rc = set( a, 1.2 );
		rc = rc ? rc : set( b, 1.5 );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Warning: first subtest initialision FAILED\n";
			return;
		}
		a = b;
		a = b;
	}
#if 0 // disable below only if operator= is defined for grb::Matrix
	if( n > 17 ) {
		Matrix< void > A( n, n, 1 ), B( n, n, 1 );
		size_t anInteger = 17;
		const size_t * const start = &anInteger;
		rc = buildMatrixUnique( A, start, start, 1, SEQUENTIAL );
		anInteger = 7;
		rc = rc ? rc : buildMatrixUnique( B, start, start, 1, SEQUENTIAL );
		A = B;
		A = B;
	} else {
		std::cerr << "Warning: part of the test is disabled-- "
			<< "please choose a larger size n\n";
	}
#endif
	return;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
