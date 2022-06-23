
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

#include <utility>
#include <iostream>

#include "alp.hpp"

template< typename MatrixType >
void print_matrix( std::string name, const MatrixType &A) {

	if( ! alp::internal::getInitialized( A ) ) {
		std::cout << "Matrix " << name << " uninitialized.\n";
		return;
	}

	std::cout << name << ":" << std::endl;
	for( size_t row = 0; row < alp::nrows( A ); ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < alp::ncols( A ); ++col ) {
			auto pos  = alp::internal::getStorageIndex( A, row, col );
			// std::cout << "(" << pos << "): ";
			std::cout << alp::internal::access( A, pos ) << "\t";
		}
		std::cout << "]" << std::endl;
	}
}


// alp program
void alpProgram( const size_t &n, alp::RC &rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T one  = ring.getOne< T >();
	T zero = ring.getZero< T >();

	// allocate
	std::vector< T > u_data( n, one );
	std::vector< T > v_data( n, one );
	std::vector< T > M_data( n, zero );

	alp::Vector< T > u( n );
	alp::Vector< T > v( n );
	alp::Matrix< T, alp::structures::General > M( n, n );

	alp::buildVector( u, u_data.begin(), u_data.end() );
	alp::buildVector( v, v_data.begin(), v_data.end() );

	rc = alp::outer( M, u, v, ring.getMultiplicativeOperator());

	// Example with matrix view on a lambda function.
	auto uvT = alp::outer( u, v, ring.getMultiplicativeOperator() );
	print_matrix( "uvT", uvT );

	// Example when outer product takes the same vector as both inputs.
	// This operation results in a symmetric positive definite matrix.
	auto vvT = alp::outer( v, ring.getMultiplicativeOperator() );
	print_matrix( "vvT", uvT );

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
		} else if( read == 0 ) {
			std::cerr << "n must be a positive number\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an integer, the "
					 "test size.\n";
		return 1;
	}
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alpProgram, in, out, true ) != alp::SUCCESS ) {
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

