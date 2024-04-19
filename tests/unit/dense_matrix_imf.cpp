
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

void buildUpperTriangularRawArray( std::vector< double > & v, int n ) {
	for (int i = 0; i < n; ++i ) {
		for (int j = 0; j < n; ++j ) {
			if( i >= j ) {
				v[ i * n + j ] = 1;
			} else {
				v[ i * n + j ] = 0;
			}
		}
	}
}

void grb_program( const size_t & n, grb::RC & rc ) {
	// initialize test
	// using General Views over General Structured Matrix
	grb::StructuredMatrix< double, grb::structures::General > A( n, n );
	std::cout << "General gather from a general StructuredMatrix (expect success)\n";
	try {
		auto Aview = grb::get_view< grb::structures::General >(
			A,
			grb::utils::range(1,3), grb::utils::range(1,5)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}


	// using Upper Triangular Structured Matrix
	grb::StructuredMatrix< double, grb::structures::UpperTriangular > U( n, n );

	// Initialize StructuredMatrix
	std::vector< double > Mdata ( n * n, 1 );
	buildUpperTriangularRawArray( Mdata, n );
	rc = grb::buildMatrix( U, Mdata.begin(), Mdata.end() );
	if( rc != grb::SUCCESS ) {
		return;
	}

	// Valid block
	std::cout << "Gather to UpperTriangular (expect success)\n"
	"|x  x  x  x  x  x|\n"
	"|.  A  A  x  x  x|\n"
	"|.  A  A  x  x  x|\n"
	"|.  .  .  x  x  x|\n"
	"|.  .  .  .  x  x|\n"
	"|.  .  .  .  .  x|\n";
	try {
		auto Uview1 = grb::get_view< grb::structures::UpperTriangular >(
			U,
			grb::utils::range(1,3), grb::utils::range(1,3)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}

	// Valid block -  because of "casting" to general structure
	std::cout << "Gather to General (expect success)\n"
	"|x  x  x  A  A  x|\n"
	"|.  x  x  A  A  x|\n"
	"|.  .  x  x  x  x|\n"
	"|.  .  .  x  x  x|\n"
	"|.  .  .  .  x  x|\n"
	"|.  .  .  .  .  x|\n";
	try {
		auto Uview2 = grb::get_view< grb::structures::General >(
			U,
			grb::utils::range(0,2), grb::utils::range(3,5)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}

	// Invalid block -  selecting a block that is not UpperTriangular
	std::cout << "Gather to UpperTriangular (expect failure)\n"
	"|x  x  x  x  x  x|\n"
	"|.  A  A  A  A  x|\n"
	"|.  A  A  A  A  x|\n"
	"|.  .  .  x  x  x|\n"
	"|.  .  .  .  x  x|\n"
	"|.  .  .  .  .  x|\n";
	try {
		auto Uview3 = grb::get_view< grb::structures::UpperTriangular >(
			U,
			grb::utils::range(1,3), grb::utils::range(1,5)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}

	// Invalid block -  currently no support for zero matrix
	std::cout << "Gather to General (expect failure)\n"
	"|x  x  x  x  x  x|\n"
	"|.  x  x  x  x  x|\n"
	"|.  .  x  x  x  x|\n"
	"|.  .  .  x  x  x|\n"
	"|A  A  .  .  x  x|\n"
	"|A  A  .  .  .  x|\n";
	try {
		auto Uview4 = grb::get_view< grb::structures::General >(
			U,
			grb::utils::range(4,n), grb::utils::range(0,2)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}

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
		} else if( read < 6 ) {
			std::cerr << "Given value for n is smaller than 6\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer >= 6, the "
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
