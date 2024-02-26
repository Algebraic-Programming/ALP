
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

#include <graphblas.hpp>


using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	std::vector< grb::Vector< unsigned char > > vectors;
	rc = SUCCESS;
	// We test a lot of ways to push a grb::Vector into an std::Vector.
	// vectors.push_back of a
	//   -# std::moved temporary
	//   -# temporary
	//   -# newly constructed grb::Vector
	//   -# std::moved from a newly constructed grb::Vector
	// We also test pushing grb::Vectors of different sizes
	//   (note that pushing different types is not allowed)
	for( size_t i = 0; rc == SUCCESS && i < 7; ++i ) {
		grb::Vector< unsigned char > temp( n );
		if( i % 2 == 0 ) {
			rc = grb::set( temp, i );
			vectors.push_back( std::move( temp ) );
		} else {
			vectors.push_back( temp );
		}
	}
	for( size_t i = 7; rc == SUCCESS && i < 9; ++i ) {
		grb::Vector< unsigned char > temp( n / 2 );
		if( i % 2 == 0 ) {
			rc = grb::set( temp, i );
		}
		vectors.push_back( std::move( temp ) );
	}
	for( size_t i = 9; rc == SUCCESS && i < 11; ++i ) {
		vectors.push_back( grb::Vector< unsigned char >( n / 2 ) );
		if( i % 2 == 0 ) {
			rc = grb::set( vectors[ i ], i );
		}
	}
	for( size_t i = 11; rc == SUCCESS && i < 13; ++i ) {
		// Commonly through copy elision will be equivalent to the above loop.
		// Neither this nor the former loop fail at present even if
		// there is no move constructor-- they will both call copy
		// constructors instead. Hopefully this variant may fail
		// though for future standards / compilers.
		vectors.push_back( std::move( grb::Vector< unsigned char >( n / 2 ) ) );
		if( i % 2 == 0 ) {
			rc = grb::set( vectors[ i ], i );
		}
	}
	for( size_t i = 1; rc == SUCCESS && i < 13; i += 2 ) {
		rc = grb::set( vectors[ i ], i );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}
	for( size_t i = 0; i < 7; ++i ) {
		if( grb::nnz( vectors[ i ] ) != n ) {
			std::cerr << "\tunexpected number of nonzeroes at vector " << i << ": " << grb::nnz( vectors[ i ] ) << ", expected " << n << "\n";
			rc = FAILED;
		}
	}
	for( size_t i = 7; i < 13; ++i ) {
		if( grb::nnz( vectors[ i ] ) != n / 2 ) {
			std::cerr << "\tunexpected number of nonzeroes at vector " << i << ": " << grb::nnz( vectors[ i ] ) << ", expected " << ( n / 2 ) << "\n";
			rc = FAILED;
		}
	}
	for( size_t i = 0; i < 13; ++i ) {
		for( const auto & pair : vectors[ i ] ) {
			if( pair.second != static_cast< unsigned char >( i ) ) {
				std::cerr << "\tunexpected value at entry ( " << pair.first << ", " << static_cast< size_t >( pair.second ) << " ) of vector " << i << "; expected " << i << " as value\n";
				rc = FAILED;
			}
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
