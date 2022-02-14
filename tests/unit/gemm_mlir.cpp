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

static bool failed(grb::RC rc) {
  if (rc != SUCCESS)
    return true;
  return false;
}

/// Test chain mxm.
void grb_program_chain( const size_t & n, grb::RC & rc ) {
	grb::Semiring< grb::operators::add< double >, 
  grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	std::cout << "\tStarting chain mxm test with size: " << n << "\n";

	// initialize test
	grb::Matrix< float > A1( 30, 35 );
	grb::Matrix< float > A2( 35, 15 );
  grb::Matrix< float > O1( 30, 15);
  grb::Matrix< float > A3( 15, 5 );
  grb::Matrix< float > O2( 30, 5 );
	grb::Matrix< float > A4( 5, 10 );
  grb::Matrix< float > O3( 30, 10 );
  grb::Matrix< float > A5( 10, 20 );
  grb::Matrix< float > O4( 30, 20 );
  grb::Matrix< float > A6( 20, 25 );
  grb::Matrix< float > O5( 30, 25 );

	std::vector< float > vA1( 30 * 35, 1.0 ), 
                       vA2( 35 * 15, 2.0 ), 
                       vA3( 15 * 5, 3.0 ),
                       vA4( 5 * 10, 4.0 ),
                       vA5( 10 * 20, 1.0 ),
                       vA6( 20 * 25, 1.0);

	if (failed(grb::buildMatrixUnique( A1, vA1.begin(), vA1.end(), SEQUENTIAL ))) {
    std::cerr << "\tinitialisation for A FAILED\n";
    return;
  }
  if (failed(grb::buildMatrixUnique( A2, vA2.begin(), vA2.end(), SEQUENTIAL ))) {
    std::cerr << "\tinitialisation for B FAILED\n";
    return;
  }
  if (failed(grb::buildMatrixUnique( A3, vA3.begin(), vA3.end(), SEQUENTIAL ))) {
    std::cerr << "\tinitialisation for D FAILED\n";
    return; 
  }
  if (failed(grb::buildMatrixUnique( A4, vA4.begin(), vA4.end(), SEQUENTIAL ))) {
    std::cerr << "\tinitialisation for D FAILED\n";
    return;
  }
  if (failed(grb::buildMatrixUnique( A5, vA5.begin(), vA5.end(), SEQUENTIAL ))) {
    std::cerr << "\tinitialisation for D FAILED\n";
    return;
  }
  if (failed(grb::buildMatrixUnique( A6, vA6.begin(), vA6.end(), SEQUENTIAL ))) {
    std::cerr << "\tinitialisation for D FAILED\n";
    return;
  }
  
	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	if (failed(grb::mxm( O1, A2, A1, ring ))) {
	  std::cerr << "Call to grb::mxm 1 FAILED\n";
		return;
	}

  if (failed(grb::mxm( O2, A3, O1, ring ))) {
    std::cerr << "Call to grb::mxm 2 FAILED\n";
    return;
  }

  if (failed(grb::mxm( O3, A4, O2, ring ))) {
    std::cerr << "Call to grb::mxm 3 FAILED\n";
    return;
  }

  if (failed(grb::mxm( O4, A5, O3, ring ))) {
    std::cerr << "Call to grb::mxm 4 FAILED\n";
    return;
  }

  if (failed(grb::mxm( O5, A6, O4, ring ))) {
    std::cerr << "Call to grb::mxm  5 FAILED\n";
    return;
  }

	auto deepCopy = internal::getFull( O5 );
	for( size_t i = 0; i < 30; i++ ) {
		for( size_t j = 0; j < 25; j++ ) {
			std::cout << deepCopy[ i * 25 + j ] << " ";
		}
		std::cout << "\n";
	}

	rc = SUCCESS;
}

/// Test call to mxm.
void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Semiring< grb::operators::add< double >, 
  grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	std::cout << "\tStarting mxm test with size: " << n << "\n";

	// initialize test
	grb::Matrix< float > A( n, n );
	grb::Matrix< float > B( n, n );
	grb::Matrix< float > C( n, n );

	std::vector< float > vA( n * n, 2.0 ), vB( n * n, 1.0 );

	rc = grb::buildMatrixUnique( A, vA.begin(), vA.end(), SEQUENTIAL );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrixUnique( B, vB.begin(), vB.end(), SEQUENTIAL );
	} else {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	for( int i = 0; i < 5; i++ ) {
		rc = grb::mxm( C, A, B, ring );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm FAILED\n";
			return;
		}
	}

	auto deepCopy = internal::getFull( C );
	for( size_t i = 0; i < n; i++ ) {
		for( size_t j = 0; j < n; j++ ) {
			std::cout << deepCopy[ i * C.n + j ] << " ";
		}
		std::cout << "\n";
	}

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

	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}

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
