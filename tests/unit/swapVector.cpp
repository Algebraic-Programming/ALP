
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

using namespace grb;

void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Vector< double > fullVector( n ), emptyVector( n );
	rc = grb::set( fullVector, 1.5 );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}
	if( grb::nnz( fullVector ) != n ) {
		std::cerr << "\tinitialisation FAILED: vector has " << grb::nnz( fullVector ) << " entries, while expecting " << n << "\n";
		rc = FAILED;
		return;
	}
	if( grb::nnz( emptyVector ) != 0 ) {
		std::cerr << "\tinitialisation FAILED: vector has " << grb::nnz( emptyVector ) << " entries, while expecting 0\n";
		rc = FAILED;
		return;
	}
	std::swap( fullVector, emptyVector );
	if( grb::nnz( emptyVector ) != n ) {
		std::cerr << "\tunexpected number of nonzeroes " << grb::nnz( emptyVector ) << ", expected " << n << "\n";
		rc = FAILED;
	}
	if( grb::nnz( fullVector ) != 0 ) {
		std::cerr << "\tunexpected number of nonzeroes " << grb::nnz( fullVector ) << ", expected 0.\n";
		rc = FAILED;
	}
	for( const auto & pair : emptyVector ) {
		if( pair.second != 1.5 ) {
			std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second << " ), expected value 1.5\n";
			rc = FAILED;
		}
	}
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
