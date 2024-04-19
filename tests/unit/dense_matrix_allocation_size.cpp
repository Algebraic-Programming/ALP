
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
#include <string>
#include <type_traits>
#include <vector>

#include <graphblas.hpp>

void grb_program( const size_t & n, grb::RC & rc ) {
	// initialize test
	grb::Matrix< double, grb::reference_dense > M( n, n );

	size_t elems = grb::internal::DataElementsCalculator< double, grb::storage::Dense::full >::calculate( M );
	std::cout << "Matrix< Dense::full, structure::General> " << n << " x " << n << " can be stored with " << elems << " elements.\n";

	elems = grb::internal::DataElementsCalculator< double, grb::storage::Dense::full, grb::structures::Triangular >::calculate( M );
	std::cout << "Matrix< Dense::full, structure::Triangular> " << n << " x " << n << " can be stored with " << elems << " elements.\n";

	// The following call does not work because the template cannot figure out that UpperTriangular is also Triangular.
	// TODO: fix it
	// grb::internal::DataElementsCalculator< double, grb::storage::Dense::full, grb::structures::UpperTriangular >::calculate( M );
	// std::cout << "Matrix< Dense::full, structure::UpperTriangular> " << n << " x " << n << " can be stored with " << elems << " elements.\n";

	rc = grb::SUCCESS;
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
