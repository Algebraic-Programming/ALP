/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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

#include <graphblas/mlir/matrix.hpp>

#include <graphblas.hpp>

using namespace grb;

static bool failed( grb::RC rc ) {
	if( rc != SUCCESS )
		return true;
	return false;
}

/// Test chain mxm.
void grb_program_chain( const size_t & n, grb::RC & rc ) {
	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	std::cout << "\tStarting chain mxm test with size: " << n << "\n";

	// initialize test
	grb::Matrix< float > A1( 1000, 2000 );
	grb::Matrix< float > A2( 2000, 900 );
	grb::Matrix< float > O1( 1000, 900 );
	grb::Matrix< float > A3( 900, 1500 );
	grb::Matrix< float > O2( 1000, 1500 );
	grb::Matrix< float > A4( 1500, 600 );
	grb::Matrix< float > O3( 1000, 600 );
	grb::Matrix< float > A5( 600, 800 );
	grb::Matrix< float > O4( 1000, 800 );

	std::vector< float > vA1( 1000 * 2000, 1.0 ), vA2( 2000 * 900, 2.0 ), vA3( 900 * 1500, 3.0 ), vA4( 1500 * 600, 4.0 ), vA5( 600 * 800, 1.0 );

	if( failed( grb::buildMatrixUnique( A1, vA1.begin(), vA1.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation for A FAILED\n";
		return;
	}
	if( failed( grb::buildMatrixUnique( A2, vA2.begin(), vA2.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation for B FAILED\n";
		return;
	}
	if( failed( grb::buildMatrixUnique( A3, vA3.begin(), vA3.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation for D FAILED\n";
		return;
	}
	if( failed( grb::buildMatrixUnique( A4, vA4.begin(), vA4.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation for D FAILED\n";
		return;
	}
	if( failed( grb::buildMatrixUnique( A5, vA5.begin(), vA5.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation for D FAILED\n";
		return;
	}

	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	if( failed( grb::mxm( O1, A2, A1, ring ) ) ) {
		std::cerr << "Call to grb::mxm 1 FAILED\n";
		return;
	}

	if( failed( grb::mxm( O2, A3, O1, ring ) ) ) {
		std::cerr << "Call to grb::mxm 2 FAILED\n";
		return;
	}

	if( failed( grb::mxm( O3, A4, O2, ring ) ) ) {
		std::cerr << "Call to grb::mxm 3 FAILED\n";
		return;
	}

	if( failed( grb::mxm( O4, A5, O3, ring ) ) ) {
		std::cerr << "Call to grb::mxm 4 FAILED\n";
		return;
	}

	auto deepCopy = internal::getFull( O4 );
#ifdef _DEBUG 
//	for( size_t i = 0; i < 30; i++ ) {
//		for( size_t j = 0; j < 25; j++ ) {
//			std::cout << deepCopy[ i * 25 + j ] << " ";
//		}
//		std::cout << "\n";
//	}
#endif
	rc = SUCCESS;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 5;

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
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;

	if( launcher.exec( &grb_program_chain, in, out, true ) != SUCCESS ) {
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
