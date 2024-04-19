
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
#include <memory>

#include <alp.hpp>

#include "../utils/print_alp_containers.hpp"

typedef float T;

template< typename T >
void init_matrix( std::vector< T > &A, const size_t rows, const size_t cols ) {

	size_t multiplier;
	for( multiplier = 1; multiplier < rows; multiplier *= 10 );

	for( size_t row = 0; row < rows; ++row ) {
		for( size_t col = 0; col < cols; ++col ) {
			A[ row * cols + col ] = multiplier * row + col;
		}
	}
}

void alp_program( const size_t & n, alp::RC & rc ) {

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	rc = alp::SUCCESS;

	// create the original matrix
	std::vector< T > M_data( n * n );
	init_matrix( M_data, n, n );

	alp::Matrix< T, alp::structures::General > M( n, n );
	alp::buildMatrix( M, M_data.begin(), M_data.end() );
	print_matrix( "M", M );

	T *M_ptr = alp::internal::getRawPointerToFirstElement( M );
	std::cout << M_ptr[0] << std::endl;
	std::cout << "Leading dimension = " << alp::internal::getLeadingDimension( M ) << "\n";

	// matrix view
	auto A = alp::get_view( M, alp::utils::range( 2, 4 ), alp::utils::range( 2, 4 ) );

	T *A_ptr = alp::internal::getRawPointerToFirstElement( A );
	std::cout << A_ptr[0] << std::endl;
	std::cout << "Leading dimension = " << alp::internal::getLeadingDimension( A ) << "\n";

	// vector view
	auto v = alp::get_view( M, 1, alp::utils::range( 2, 4 ) );
	T *v_ptr = alp::internal::getRawPointerToFirstElement( v );
	std::cout << v_ptr[0] << std::endl;
	std::cout << " INC = " << alp::internal::getIncrement( v ) << "\n";

	auto u = alp::get_view( M, alp::utils::range( 2, 4 ), 1 );
	T *u_ptr = alp::internal::getRawPointerToFirstElement( u );
	std::cout << u_ptr[0] << std::endl;
	std::cout << " INC = " << alp::internal::getIncrement( u ) << "\n";

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
