
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

#include <graphblas/utils/MatrixVectorIterator.hpp>

#include <graphblas.hpp>

using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Matrix< double > diag( n, n );
	grb::Vector< double > vector( n );
	rc = grb::set< grb::descriptors::use_index >( vector, 0 );
	if( rc == SUCCESS ) {
		auto converter = grb::utils::makeVectorToMatrixConverter< double >( vector, []( const size_t &ind, const double &val ) {
			return std::make_pair( std::make_pair( ind, ind ), val );
		} );
		auto start = converter.begin();
		auto end = converter.end();
		rc = grb::buildMatrixUnique( diag, start, end, PARALLEL );
	}
	if( rc != SUCCESS || grb::nnz( diag ) != n ) {
		std::cerr << "\t initialisation FAILED\n";
		if( rc == SUCCESS ) {
			rc = FAILED;
		}
		return;
	}

	rc = grb::clear( diag );
	if( rc != SUCCESS ) {
		std::cerr << "\t clear matrix FAILED\n";
		return;
	}

	if( grb::nnz( diag ) != 0 ) {
		std::cerr << "\t unexpected number of nonzeroes in matrix ( " << grb::nnz( diag ) << " ), expected 0\n";
		rc = FAILED;
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
