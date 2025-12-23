
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

#include <graphblas.hpp>

bool exec_tests(
	const grb::Matrix< void > &A, const grb::Matrix< char > &B,
	grb::Vector< bool > &out, grb::Vector< bool > &out2,
	const grb::Vector< char > &in, const grb::Vector< char > &in2,
	const size_t offset
) {
	grb::semirings::boolean ring;

	// test 1, exec
	grb::RC rc = grb::mxv< grb::descriptors::dense >( out, A, in2, ring );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t test " << offset << " FAILED\n";
		return false;
	}

	// test 2, exec
	rc = grb::vxm< grb::descriptors::dense >( out2, in, A, ring );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t test " << (1 + offset) << " FAILED\n";
		return false;
	}

	// test 3, exec
	rc = grb::mxv< grb::descriptors::dense >( out2, B, in, ring );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t test " << (2 + offset) << " FAILED\n";
		return false;
	}

	// test 4, exec
	rc = grb::vxm< grb::descriptors::dense >( out, in2, B, ring );
	if( rc != grb::ILLEGAL ) {
		std::cerr << "\t test " << (3 + offset) << " FAILED\n";
		return false;
	}

	return true;
}

void grb_program( const size_t &n, grb::RC &rc ) {

	// repeatedly used containers
	grb::Matrix< void > A( n, 2 * n );
	grb::Matrix< char > B( 2 * n, n );
	grb::Vector< bool > out( n ), out2( 2 * n );
	grb::Vector< char > in( n ), in2( 2 * n );

	// run tests 1-4
	if( !exec_tests( A, B, out, out2, in, in2, 1 ) ) {
		rc = grb::FAILED;
		return;
	}

	// init tests 5-8
	rc = grb::setElement( out, true, 0 );
	rc = rc ? rc : grb::setElement( out2, true, 0 );
	if( rc != grb::SUCCESS ) {
		std::cout << "Test batch 5-8: initialisation FAILED\n";
		return;
	}

	// run tests 5-8
	if( !exec_tests( A, B, out, out2, in, in2, 5 ) ) {
		rc = grb::FAILED;
		return;
	}

	// init tests 9-12
	rc = grb::set( in, 1 );
	rc = rc ? rc : grb::set( in2, 1 );
	rc = rc ? rc : grb::setElement( in, 0, n - 1 );
	rc = rc ? rc : grb::setElement( in2, 0, 2 * n - 1 );
	rc = rc ? rc : grb::set( out, in, false );
	rc = rc ? rc : grb::set( out2, in2, false );
	if( rc != grb::SUCCESS ) {
		std::cout << "Test batch 9-12: initialisation FAILED\n";
		return;
	}
	
	// run tests 9-12
	if( !exec_tests( A, B, out, out2, in, in2, 9 ) ) {
		rc = grb::FAILED;
		return;
	}

	// init tests 13-16
	rc = rc ? rc : grb::set( in, out, 2 );
	rc = rc ? rc : grb::set( in2, out2, 3 );
	rc = rc ? rc : grb::set( out, true );
	rc = rc ? rc : grb::set( out2, true );
	if( rc != grb::SUCCESS ) {
		std::cout << "Test batch 13-16: initialisation FAILED\n";
		return;
	}

	// run tests 13-16
	if( !exec_tests( A, B, out, out2, in, in2, 13 ) ) {
		rc = grb::FAILED;
		return;
	}
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
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
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
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

