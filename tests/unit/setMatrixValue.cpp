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
#include <numeric>
#include <algorithm>
#include <vector>

#include <graphblas/utils/iterators/MatrixVectorIterator.hpp>

#include <graphblas.hpp>


using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Matrix< int > Identity( n, n, n );
	std::vector< size_t > Identity_coords( n );
	std::iota( Identity_coords.begin(), Identity_coords.end(), 0UL );
	std::vector< int > Identity_vals( n, 1 );
	if( SUCCESS !=
		buildMatrixUnique( Identity, Identity_coords.data(),
			Identity_coords.data(), Identity_vals.data(), n, SEQUENTIAL )
	) {
		std::cerr << "\t initialisation (buildMatrixUnique) FAILED: rc is "
			<< grb::toString(rc) << "\n";
		rc = FAILED;
		return;
	}

	// Check first if the matrix is correctly initialised with 1s
	rc = std::all_of( Identity.cbegin(), Identity.cend(),
		[]( const std::pair< std::pair< size_t, size_t >, int > &entry ) {
			return entry.second == 1 && entry.first.first == entry.first.second;
		} ) ? SUCCESS : FAILED;
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation (buildMatrixUnique check) FAILED: rc is "
			<< grb::toString(rc) << "\n";
		return;
	}

	// Try to set the matrix to 2s ( RESIZE )
	rc = grb::set( Identity, 2UL, Phase::RESIZE );
	if( rc != SUCCESS ) {
		std::cerr << "\t set matrix to 2s ( RESIZE ) FAILED: rc is "
			<< grb::toString(rc) << "\n";
		return;
	}
	// As the RESIZE phase is useless, the matrix should not be resized.
	if( nnz( Identity ) != n ) {
		std::cerr << "\t unexpected number of nonzeroes in matrix "
			<< "( " << nnz( Identity ) << " ), expected " << n << "\n";
		rc = FAILED;
		return;
	}

	// Try to set the matrix to 2s ( EXECUTE )
	rc = grb::set( Identity, 2UL, Phase::EXECUTE );
	if( rc != SUCCESS ) {
		std::cerr << "\t set matrix to 2s ( EXECUTE ) FAILED: rc is "
			<< grb::toString(rc) << "\n";
		return;
	}
	// Now all values should be 2s
	rc = std::all_of( Identity.cbegin(), Identity.cend(),
		[]( const std::pair< std::pair< size_t, size_t >, int > &entry ) {
			return entry.second == 2 && entry.first.first == entry.first.second;
		} ) ? SUCCESS : FAILED;
	if( rc != SUCCESS ) {
		std::cerr << "\t Check of set matrix to 2s ( EXECUTE ) FAILED";
		return;
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 1000;

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
		std::cerr << "  -n (optional, default is 1000): an integer test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

