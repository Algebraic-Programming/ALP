
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

template< typename D, Descriptor descr = descriptors::no_operation, typename T = long >
D compute_value( T i, T j ) {
	return descr & descriptors::transpose_matrix ? i + 2 * j : 2 * i + j;
}

template< Descriptor descr = descriptors::no_operation, typename D >
RC check_obtained( const Matrix< D > &U, long k = 0 ) {
	for( const auto &triple : U ) {
		const long &i = static_cast<long>( triple.first.first );
		const long &j = static_cast<long>( triple.first.second );
		const auto &v = triple.second;
		if( i + k < j ) {
			std::cout << "Unexpected entry at position ( " << i << ", " << j << " ) "
					  << "-- only expected entries on the lower triangular part above "
					  << "the " << k << "-th diagonal\n";
			return FAILED;
		}
		const D expected_value = compute_value< D, descr >( i, j );
		if( v != expected_value ) {
			std::cout << "Unexpected value at position ( " << i << ", " << j << " ) "
					  << "-- expected " << expected_value << ", found " << v << "\n";
			return FAILED;
		}
	}
	return SUCCESS;
}

template< typename D >
RC is_identity( const Matrix< D > &U ) {
	size_t n = nrows( U );
	size_t counter = 0;
	for( const auto &triple : U ) {
		const auto &i = triple.first.first;
		const auto &j = triple.first.second;
		const auto &v = triple.second;
		if( i != j ) {
			std::cout << "Unexpected entry at position ( " << i << ", " << j << " ) = "
						<< v << "  --  only expected entries on the main diagonal\n";
			return FAILED;
		}
		counter++;
	}
	return counter == n ? SUCCESS : FAILED;
}

void grb_program( const long &n, grb::RC &rc ) {
	rc = SUCCESS;

	// Matrix initialisation
	Matrix< int > A( n, n );
	size_t * I = new size_t[ 2*n ];
	size_t * J = new size_t[ 2*n ];
	double * V = new double[ 2*n ];
	for( auto k = 0; k < 2*n; ++k ) {
		I[ k ] = k % 3 == 0 ? k : k - 1;
		J[ k ] = std::rand() % n;
		V[ k ] = compute_value< int >( I[ k ], J[ k ] );
	}
	if( SUCCESS !=
		grb::buildMatrixUnique( A, I, J, V, n, SEQUENTIAL )
	) {
		std::cerr << "Error on test: building matrix" << std::endl;
		rc = FAILED;
		return;
	}

	{ // Mixed-domain matrix, should be successful
		Matrix< size_t > U_A( n, n );
		rc = grb::tril( U_A, A, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A, A, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: mixed-domain matrix" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( U_A );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: mixed-domain matrix" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: mixed-domain matrix" << std::flush << std::endl;
	}
	{ // k = 10, should be successful
		Matrix< size_t > U_A( n, n );
		const long k = 10;
		rc = grb::tril( U_A, A, k, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A, A, k, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: k = " << k << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( U_A, k );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: k = " << k << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: k = " << k << std::flush << std::endl;
	}
	{ // k = -10, should be successful
		Matrix< size_t > U_A( n, n );
		const long k = -10;
		rc = grb::tril( U_A, A, k, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A, A, k, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: k = " << k << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( U_A, k );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: k = " << k << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: k = " << k << std::flush << std::endl;
	}
	{ // Transpose_matrix descriptor, should be successful
		Matrix< size_t > U_At( n, n );
		rc = grb::tril< descriptors::transpose_matrix >( U_At, A, Phase::RESIZE );
		rc = rc ? rc : grb::tril< descriptors::transpose_matrix >( U_At, A, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: transpose_matrix descriptor" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained< descriptors::transpose_matrix >( U_At );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: transpose_matrix descriptor" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: transpose_matrix descriptor" << std::flush << std::endl;
	}
	{ // Overlap is forbidden, should return RC::OVERLAP
		rc = grb::tril( A, A, Phase::RESIZE );

		if( rc != RC::OVERLAP ) {
			std::cerr << "Error on test: overlap, should return RC::OVERLAP" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << " instead of RC::OVERLAP" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: overlap, should return RC::OVERLAP" << std::flush << std::endl;
	}
	{ // Empty matrix, should be successful
		Matrix< int > A_empty( n, n );
		Matrix< size_t > U_A_empty( n, n );
		rc = grb::tril( U_A_empty, A_empty, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A_empty, A_empty, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: empty matrix" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( U_A_empty );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: empty matrix" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: empty matrix" << std::flush << std::endl;
	}
	{ // Out-of-bound <k> parameter, should be successful and return an empty matrix
		const long k = 2*n;
		Matrix< size_t > U_A( n, n );
		rc = grb::tril( U_A, A, k, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A, A, k, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Out-of-bound <k> parameter" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( U_A, k );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Out-of-bound <k> parameter" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: Out-of-bound <k> parameter" << std::flush << std::endl;
	}
	{ // Out-of-bound <-k> parameter, should be successful and return an empty matrix
		const long k = -2*n;
		Matrix< size_t > U_A( n, n );
		rc = grb::tril( U_A, A, k, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A, A, k, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Out-of-bound <-k> parameter" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( U_A, k );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Out-of-bound <-k> parameter" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: Out-of-bound <-k> parameter" << std::flush << std::endl;
	}
	{ // Identity isolation using triu( tril ( A, 1 ), 1 )
		const Matrix< int > dense( n, n, n*n );
		size_t * I = new size_t[ n*n ];
		size_t * J = new size_t[ n*n ];
		int * V = new int[ n*n ];
		std::fill( V, V + n*n, 1 );
		for( auto i=0; i<n; i++ ) {
			std::iota( I + i*n, I + (i+1)*n, 0 );
			std::fill( J + i*n, J + (i+1)*n, i );
		}
		if( SUCCESS !=
			buildMatrixUnique( A, I, J, V, n*n, SEQUENTIAL )
		) {
			std::cerr << "Error on test: building matrix in: "
						<< "identity isolation using triu( tril ( A, 1 ), 1 )" << std::endl;
			rc = FAILED;
			return;
		}
		const long k = 0;
		Matrix< size_t > U_A( n, n );
		rc = grb::tril( U_A, A, k, Phase::RESIZE );
		rc = rc ? rc : grb::tril( U_A, A, k, Phase::EXECUTE );
		Matrix< size_t > I_A( n, n );
		rc = grb::triu( I_A, U_A, k, Phase::RESIZE );
		rc = rc ? rc : grb::triu( I_A, U_A, k, Phase::EXECUTE );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Identity isolation using triu( tril ( A, 1 ), 1 )" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = is_identity( I_A );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: Identity isolation using triu( tril ( A, 1 ), 1 )" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: Identity isolation using triu( tril ( A, 1 ), 1 )" << std::flush << std::endl;
	}
}

int main( int argc, char ** argv ) {
	// defaults
	long n = 1000;

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
	RC out;
	if( launcher.exec( &grb_program, n, out, false ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}

	std::cerr << std::flush;
	if( out != SUCCESS ) {
		std::cout << std::flush << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << std::flush << "Test OK" << std::endl;
		return 0;
	}
}
