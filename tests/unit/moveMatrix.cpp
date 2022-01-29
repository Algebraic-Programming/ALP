
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
	grb::Matrix< double > matrix( n, n );
	if( grb::nrows( matrix ) != n || grb::ncols( matrix ) != n || grb::nnz( matrix ) != 0 ) {
		std::cerr << "\tinitialisation FAILED: matrix has " << grb::nrows( matrix ) << " rows, " << grb::ncols( matrix ) << " columns, and " << grb::nnz( matrix ) << " entries, while expecting an n by n matrix with 0 entries.\n";
		rc = FAILED;
		return;
	}
	// initialise matrix
	rc = grb::resize( matrix, n );
	if( rc == SUCCESS ) {
		size_t * i = new size_t[ n ];
		size_t * j = new size_t[ n ];
		double * v = new double[ n ];
		for( size_t k = 0; k < n; ++k ) {
			i[ k ] = j[ k ] = k;
			v[ k ] = 1.5;
		}
		rc = buildMatrixUnique( matrix, i, j, v, n, SEQUENTIAL );
		if( n > 0 ) {
			delete [] i;
			delete [] j;
			delete [] v;
		}
		if( grb::nnz( matrix ) != n ) {
			std::cerr << "\t ingestion FAILED: matrix has " << grb::nnz( matrix ) << " entries, but should have " << n << "\n";
			rc = FAILED;
		}
	}
	if( rc == SUCCESS ) {
		grb::Matrix< double > tempMatrix( n, n );
		if( grb::nnz( tempMatrix ) != 0 ) {
			std::cerr << "\t initialisation of temporary FAILED: matrix has " << grb::nnz( tempMatrix ) << " entries, while expecting 0\n";
			rc = FAILED;
			return;
		}
		tempMatrix = std::move( matrix );
		if( grb::nnz( tempMatrix ) != n ) {
			std::cerr << "\t move FAILED: unexpected number of nonzeroes " << grb::nnz( tempMatrix ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & triple : tempMatrix ) {
			if( triple.second != 1.5 ) {
				std::cerr << "\t move FAILED: unexpected entry ( " << triple.first.first << ", " << triple.first.second << " ) = " << triple.second << ", expected value 1.5\n";
				rc = FAILED;
			}
		}
		matrix = std::move( tempMatrix );
	}
	if( rc == SUCCESS ) {
		if( grb::nnz( matrix ) != n ) {
			std::cerr << "\t second move FAILED: unexpected number of nonzeroes " << grb::nnz( matrix ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & triple : matrix ) {
			if( triple.second != 1.5 ) {
				std::cerr << "\t second move FAILED: unexpected entry ( " << triple.first.first << ", " << triple.first.second << " ) = " << triple.second << ", expected value 1.5\n";
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
