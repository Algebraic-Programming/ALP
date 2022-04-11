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

/// Test call to mxm.
void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Semiring< grb::operators::add< double >, 
  grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	std::cout << "\tStarting mxm test with size: " << n << "\n";

	// initialize test
	grb::Matrix< float > A( n, n );
	grb::Matrix< float > B( n, n );
	grb::Matrix< float > C( n, n );
  grb::Matrix< float > D( n, n );
  grb::Matrix< float > E( n, n );
  grb::Matrix< float > F( n, n );

	std::vector< float > vA( n * n, 2.0 ), vB( n * n, 1.0 );

	if( failed( grb::buildMatrixUnique( A, vA.begin(), vA.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	if( failed( grb::buildMatrixUnique( B, vB.begin(), vB.end(), SEQUENTIAL ) ) ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}
  
  if( failed( grb::buildMatrixUnique( D, vA.begin(), vA.end(), SEQUENTIAL ) ) ) {
    std::cerr << "\tinitialization FAILED\n";
    return;
  }

  if( failed( grb::buildMatrixUnique( E, vA.begin(), vA.end(), SEQUENTIAL ) ) ) {
    std::cerr << "\tinitialization FAILED\n";
    return;
  }

  if( failed( grb::buildMatrixUnique( F, vA.begin(), vA.end(), SEQUENTIAL ) ) ) {
    std::cerr << "\tinitialization FAILED\n";
    return;
  }

	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	rc = grb::mxm( C, A, B, ring );
	if( rc != SUCCESS ) {
	  std::cerr << "Call to grb::mxm FAILED\n";
		return;
	}
  rc = grb::mxm( E, C, F, ring );
  

	auto deepCopy = internal::getFull( E );
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
	size_t in = 8;

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

	return 0;
}
