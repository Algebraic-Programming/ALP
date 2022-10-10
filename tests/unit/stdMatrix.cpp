
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


void grb_program( const size_t & n, grb::RC & rc ) {
	std::vector< grb::Matrix< unsigned char > > matrices;
	size_t * const I = new size_t[ n ];
	size_t * const J = new size_t[ n ];
	unsigned char * const V = new unsigned char[ n ];
	if( I == nullptr || J == nullptr || V == nullptr ) {
		std::cerr << "\t could not allocate matrix ingestion arrays\n";
		rc = FAILED;
		return;
	}
	for( size_t k = 0; k < n; ++k ) {
		I[ k ] = k;
		J[ k ] = k/2;
		V[ k ] = 2*k;
	}
	rc = SUCCESS;
	// We test a lot of ways to push a grb::Matrix into an std::Vector.
	// matrices.push_back of a
	//   -# std::moved temporary
	//   -# temporary
	//   -# newly constructed grb::Matrix
	//   -# std::moved from a newly constructed grb::Matrix
	// We also test pushing grb::Matrix instances of different sizes
	//   (note that pushing different types is not allowed)
	for( size_t i = 0; rc == SUCCESS && i < 7; ++i ) {
		grb::Matrix< unsigned char > temp( n, 2*n );
		if( i % 2 == 0 ) {
			rc = grb::buildMatrixUnique( temp, I, J, V, n, SEQUENTIAL );
			matrices.push_back( std::move( temp ) );
		} else {
			matrices.push_back( temp );
		}
	}
	for( size_t i = 7; rc == SUCCESS && i < 9; ++i ) {
		grb::Matrix< unsigned char > temp( n, n/2 );
		if( i % 2 == 0 ) {
			rc = grb::buildMatrixUnique( temp, I, J, V, n, SEQUENTIAL );
		}
		matrices.push_back( std::move( temp ) );
	}
	for( size_t i = 9; rc == SUCCESS && i < 11; ++i ) {
		matrices.push_back( grb::Matrix< unsigned char >( n, n ) );
		if( i % 2 == 0 ) {
			rc = grb::buildMatrixUnique( matrices[ i ], I, J, V, n, SEQUENTIAL );
		}
	}
	for( size_t i = 11; rc == SUCCESS && i < 13; ++i ) {
		// Commonly through copy elision will be equivalent to the above loop.
		// Neither this nor the former loop fail at present even if
		// there is no move constructor-- they will both call copy
		// constructors instead. Hopefully this variant may fail
		// though for future standards / compilers.
		matrices.push_back( std::move( grb::Matrix< unsigned char >( n, n/2 ) ) );
		if( i % 2 == 0 ) {
			rc = grb::buildMatrixUnique( matrices[ i ], I, J, V, n, SEQUENTIAL );
		}
	}
	for( size_t i = 1; rc == SUCCESS && i < 13; i += 2 ) {
		for( size_t k = 0; k < n; ++k ) {
			V[k] = 2 * k + i;
		}
		rc = grb::clear( matrices[ i ] );
		rc = rc ? rc : grb::buildMatrixUnique( matrices[ i ], I, J, V, n, SEQUENTIAL );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation FAILED\n";
		return;
	}
	for( size_t i = 0; i < 13; ++i ) {
		if( grb::nnz( matrices[ i ] ) != n ) {
			std::cerr << "\t unexpected number of nonzeroes at matrix " << i << ": "
				<< grb::nnz( matrices[ i ] ) << ", expected " << n << "\n";
			rc = FAILED;
		}
	}
	for( size_t i = 0; i < 13; ++i ) {
		for( const auto &nonzero : matrices[ i ] ) {
			if( nonzero.first.first / 2 != nonzero.first.second ) {
				std::cerr << "\t unexpected value at position ( " << nonzero.first.first
					<< ", " << nonzero.first.second << " )\n";
			}
			const unsigned char chk = i % 2 == 0 ?
				static_cast< unsigned char >( 2 * nonzero.first.first ) :
				static_cast< unsigned char >( 2 * nonzero.first.first + i );
			if( nonzero.second != chk ) {
				std::cerr << "\t unexpected value at entry ( " << nonzero.first.first
					<< ", " << nonzero.first.second << " ) = "
					<< static_cast< size_t >( nonzero.second ) << " ) of matrix " << i << "; "
					<< "expected " << (2*nonzero.first.first+i)
					<< " as value\n";
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

