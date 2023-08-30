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
#include <numeric>
#include <algorithm>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/MatrixVectorIterator.hpp>


using namespace grb;


void grb_program( const size_t &n, grb::RC &rc ) {
	Matrix< int > A_int( n, n, n );
	Matrix< void > A_void( n, n, n );

	{ // Build A_int and A_void: identity matrices
		std::vector< size_t > A_coords( n );
		std::iota( A_coords.begin(), A_coords.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( A_void, A_coords.data(), A_coords.data(), n, PARALLEL )
		) {
			std::cerr << "\t initialisation of A_void FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			rc = FAILED;
			return;
		}

		std::vector< int > A_vals( n );
		std::fill( A_vals.begin(), A_vals.end(), 1 );
		if( SUCCESS !=
			buildMatrixUnique( A_int, A_coords.data(), A_coords.data(), A_vals.data(), n, PARALLEL )
		) {
			std::cerr << "\t initialisation of A_int FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			rc = FAILED;
			return;
		}
	}



	{ // Try cast to ushort (should succeed)
		Matrix< ushort > M_short( n, n, 0UL );
		rc = grb::set( M_short, A_int, Phase::RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_short, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		rc = grb::set( M_short, A_int, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_short, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		// Cast back to compare
		Matrix< int > M_int( n, n, nnz( A_int ) );
		rc = grb::set( M_int, M_short, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_int, M_short ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		if( !std::equal( M_int.begin(), M_int.end(), A_int.begin() ) ) {
			std::cerr << "\t FAILED: M_int != A_int\n";
			rc = FAILED;
			return;
		}
	}
	{ // Try (fake-)cast to int (should succeed)
		Matrix< int > M_int( n, n, 0UL );
		rc = grb::set( M_int, A_int, Phase::RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_int, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		rc = grb::set( M_int, A_int, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_int, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		if( !std::equal( M_int.begin(), M_int.end(), A_int.begin() ) ) {
			std::cerr << "\t FAILED: M_int != A_int\n";
			rc = FAILED;
			return;
		}
	}
	{ // Try cast to bool (should succeed)
		Matrix< bool > M_bool( n, n, 0UL );
		rc = grb::set( M_bool, A_int, Phase::RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_bool, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		rc = grb::set( M_bool, A_int, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_bool, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		// Cast back to compare
		Matrix< int > M_int( n, n, nnz( A_int ) );
		rc = grb::set( M_int, M_bool, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_int, M_short ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		if( !std::equal( M_int.begin(), M_int.end(), A_int.begin() ) ) {
			std::cerr << "\t FAILED: M_int != A_int\n";
			rc = FAILED;
			return;
		}
	}
	{ // Try cast to void (should succeed)
		Matrix< void > M_void( n, n, 0UL );
		rc = grb::set( M_void, A_int, Phase::RESIZE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_void, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		rc = grb::set( M_void, A_int, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_void, A_int ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		// Cast back to compare
		Matrix< int > M_int( n, n, nnz( A_int ) );
		rc = grb::set( M_int, M_void, 1, Phase::EXECUTE );
		if( rc != SUCCESS ) {
			std::cerr << "\t set( M_int, M_void, 1 ) FAILED: rc is "
				<< grb::toString( rc ) << "\n";
			return;
		}
		if( !std::equal( M_int.begin(), M_int.end(), A_int.begin() ) ) {
			std::cerr << "\t FAILED: M_int != A_int\n";
			rc = FAILED;
			return;
		}
	}

	rc = SUCCESS;
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
		std::cerr << "  -n (optional, default is 100): an integer test size.\n";
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

