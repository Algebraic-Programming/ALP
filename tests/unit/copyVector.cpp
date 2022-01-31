
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
	grb::Vector< double > vector( n );
	rc = grb::set( vector, 1.5 ); // vector = 1.5 everywhere
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// test copy constructor
	try {
		grb::Vector< double > copy( vector );
		if( grb::nnz( copy ) != n ) {
			std::cerr << "\t unexpected number of nonzeroes after copy-construction: "
			       << grb::nnz( copy ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto &pair : copy ) {
			if( pair.second != 1.5 ) {
				std::cerr << "\t unexpected value at entry "
					<< "( " << pair.first << ", " << pair.second << " ), "
					<< "after copy-construction; expected 1.5\n";
				rc = FAILED;
			}
		}
	} catch( ... ) {
		std::cerr << "\t test copy constructor on vectors FAILED\n";
		rc = FAILED;
	}

	// test copy assignment
	grb::Vector< double > copy = vector;
	if( grb::nnz( copy ) != n ) {
		std::cerr << "\t unexpected number of nonzeroes after copy-assignment: "
			<< grb::nnz( copy ) << ", expected " << n << "\n";
		rc = FAILED;
	}

	for( const auto &pair : copy ) {
		if( pair.second != 1.5 ) {
			std::cerr << "\t unexpected value at entry "
				<< "( " << pair.first << ", " << pair.second << " ) "
				<< "after copy-assignment; expected 1.5\n";
			rc = FAILED;
		}
	}

	// test same thing for empty vectors
	{
		grb::Vector< char > empty( 0 );
		try {
			grb::Vector< char > emptyCopy( empty );
			if( grb::size( emptyCopy ) != 0 ) {
				std::cerr << "\t unexpected size after copy-constructing"
					<< "an empty vector: "
					<< grb::nnz( emptyCopy ) << "\n";
				rc = FAILED;
			}
			if( grb::nnz( emptyCopy ) != 0 ) {
				std::cerr << "\t unexpected number of nonzeroes after "
					<< "copy-constructing an empty vector: "
					<< grb::nnz( emptyCopy ) << "\n";
				rc = FAILED;
			}
		} catch( ... ) {
			std::cerr << "\t copy constructor on empty vectors FAILED\n";
			rc = FAILED;
		}
		grb::Vector< char > emptyCopy = empty;
		if( grb::size( emptyCopy ) != 0 ) {
			std::cerr << "\t unexpected size after copy-assignment"
				<< "an empty vector: "
				<< grb::nnz( emptyCopy ) << "\n";
			rc = FAILED;
		}
		if( grb::nnz( emptyCopy ) != 0 ) {
			std::cerr << "\t unexpected number of nonzeroes after "
				<< "copy-assigning an empty vector: "
				<< grb::nnz( emptyCopy ) << "\n";
			rc = FAILED;
		}
	}

	// done
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
