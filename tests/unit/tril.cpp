
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

template< class Iterator >
void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
	if( rows > 64 || cols > 64 ) {
		return;
	}
	std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
	// os.precision( 3 );
	for( size_t y = 0; y < rows; y++ ) {
		os << std::string( 3, ' ' );
		for( size_t x = 0; x < cols; x++ ) {
			auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
				return a.first.first == y && a.first.second == x;
			} );
			if( nnz_val != end )
				os << std::fixed << ( *nnz_val ).second;
			else
				os << '_';
			os << " ";
		}
		os << std::endl;
	}
	os << "]" << std::endl;
}

template< typename D >
void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
	grb::wait( mat );
	printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
}

template< typename D, Descriptor descr = descriptors::no_operation >
D compute_value( size_t i, size_t j ) {
	return descr & descriptors::transpose_matrix ? i + 2 * j : 2 * i + j;
}

template< Descriptor descr = descriptors::no_operation, typename D >
RC check_obtained( const grb::Matrix< D > & L ) {
	for( const auto & triple : L ) {
		const size_t & i = triple.first.first;
		const size_t & j = triple.first.second;
		const size_t & v = triple.second;
		if( i < j ) {
			std::cout << "Unexpected entry at position ( " << i << ", " << j << " ) "
					  << "-- only expected entries on the lower triangular part\n";
			return RC::FAILED;
		}
		const D expected_value = compute_value< D, descr >( i, j );
		if( v != expected_value ) {
			std::cout << "Unexpected value at position ( " << i << ", " << j << " ) "
					  << "-- expected " << expected_value << ", found " << v << "\n";
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
		V[ k ] = compute_value< int >( I[ k ], J[ k ] );
	}
	assert( not grb::buildMatrixUnique( A, I, J, V, n, SEQUENTIAL ) );

	{ // Mixed-domain matrix, should be successful
		printSparseMatrix( A, "A" );
		rc = grb::tril( L_A, A, Phase::RESIZE );
		rc = rc ? rc : grb::tril( L_A, A, Phase::EXECUTE );
		printSparseMatrix( L_A, "L_A" );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: mixed-domain matrix" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained( L_A );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: mixed-domain matrix" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: mixed-domain matrix" << std::flush << std::endl;
	}
	{ // Transpose_matrix descriptor, should be successful
		printSparseMatrix( A, "A" );
		rc = grb::tril< descriptors::transpose_matrix >( L_At, A, Phase::RESIZE );
		rc = rc ? rc : grb::tril< descriptors::transpose_matrix >( L_At, A, Phase::EXECUTE );
		printSparseMatrix( L_At, "L_At" );

		if( rc != SUCCESS ) {
			std::cerr << "Error on test: transpose_matrix descriptor" << std::endl;
			std::cerr << "Error on executing: " << grb::toString( rc ) << std::endl;
			return;
		}
		rc = check_obtained< descriptors::transpose_matrix >( L_At );
		if( rc != SUCCESS ) {
			std::cerr << "Error on test: transpose_matrix descriptor" << std::endl;
			std::cerr << "Error on result, incorrect result" << std::endl;
			return;
		}
		std::cout << std::flush << " -- Test passed: transpose_matrix descriptor" << std::flush << std::endl;
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
