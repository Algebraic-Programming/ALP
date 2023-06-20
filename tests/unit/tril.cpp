
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

template< typename D >
RC is_lower_triangle( const grb::Matrix< D > & L ) {
	for( const auto & triple : L ) {
		const size_t & i = triple.first.first;
		const size_t & j = triple.first.second;
		const size_t & v = triple.second;
		if( i < j ) {
			std::cout << "Unexpected entry at position ( " << i << ", " << j << " ) "
					  << "-- only expected entries on the diagonal\n";
			return RC::FAILED;
		}
		if( v != 2 ) {
			std::cout << "Unexpected value at position ( " << i << ", " << j << " ) "
					  << "-- expected 2, found " << v << "\n";
			return RC::FAILED;
		}
	}
	return RC::SUCCESS;
}

void grb_program( const size_t & n, grb::RC & rc ) {
	rc = RC::SUCCESS;

	// Matrix initialisation
	grb::Matrix< int > A( n, n );
	grb::Matrix< size_t > L_A( n, n );  // L_A is the lower triangular matrix of A
	grb::Matrix< size_t > L_At( n, n ); // L_At is the lower triangular matrix of A^T
	size_t * I = new size_t[ n ];
	size_t * J = new size_t[ n ];
	double * V = new double[ n ];
	for( size_t k = 0; k < n; ++k ) {
		I[ k ] = k % 3 == 0 ? k : k - 1;
		J[ k ] = std::rand() % n;
		V[ k ] = 2;
	}
	assert( not grb::buildMatrixUnique( A, I, J, V, n, SEQUENTIAL ) );

	{ // Mixed-domain matrix, should be successful
		rc = grb::tril( L_A, A, Phase::RESIZE );
		rc = rc ? rc : grb::tril( L_A, A, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: mixed-domain matrix" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = is_lower_triangle( L_A );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: mixed-domain matrix" << std::endl;
			std::cerr << "Error on result, not a lower-triangle" << std::endl;
			return;
		}
	}
	{ // Transpose_matrix descriptor, should be successful
		rc = grb::tril< descriptors::transpose_matrix >( L_At, A, Phase::RESIZE );
		rc = rc ? rc : grb::tril< descriptors::transpose_matrix >( L_At, A, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Transpose_matrix descriptor" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = is_lower_triangle< size_t >( L_At );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Transpose_matrix descriptor" << std::endl;
			std::cerr << "Error on result, not a lower-triangle" << std::endl;
			return;
		}
	}
	
}

int main( int argc, char ** argv ) {
	// defaults
	size_t n = 1000000;

	// error checking
	if( argc == 2 ) {
		n = std::strtoul( argv[ 1 ], nullptr, 10 );
	}
	if( argc > 3 ) {
		std::cerr << "Usage: " << argv[ 0 ] << "[n = " << n << "]\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, n, out, false ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
