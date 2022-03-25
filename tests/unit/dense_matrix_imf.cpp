
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

#include <alp.hpp>

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

void alp_program( const size_t & n, alp::RC & rc ) {
	// initialize test
	// using General Views over General Structured Matrix
	alp::Matrix< double, alp::structures::General > A( n, n );
	std::cout << "General gather from a general Matrix (expect success)\n";
	try {
		auto Aview = alp::get_view< alp::structures::General >(
			A,
			alp::utils::range(1,3), alp::utils::range(1,5)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}


	// using Upper Triangular Structured Matrix
	alp::Matrix< double, alp::structures::UpperTriangular > U( n, n );

	// Initialize Matrix
	std::vector< double > Mdata ( n * n, 1 );
	buildUpperTriangularRawArray( Mdata, n );
	rc = alp::buildMatrix( U, Mdata.begin(), Mdata.end() );
	if( rc != alp::SUCCESS ) {
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
		auto Uview1 = alp::get_view< alp::structures::UpperTriangular >(
			U,
			alp::utils::range(1,3), alp::utils::range(1,3)
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
		auto Uview2 = alp::get_view< alp::structures::General >(
			U,
			alp::utils::range(0,2), alp::utils::range(3,5)
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
		auto Uview3 = alp::get_view< alp::structures::UpperTriangular >(
			U,
			alp::utils::range(1,3), alp::utils::range(1,5)
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
		auto Uview4 = alp::get_view< alp::structures::General >(
			U,
			alp::utils::range(4,n), alp::utils::range(0,2)
		);
		std::cout << "\tSUCCESS\n";
	} catch( const std::exception & e ) {
		std::cerr << e.what() << "\n";
	}

	rc = alp::SUCCESS;
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
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
