
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

// alp program
void alpProgram( const size_t &n, alp::RC &rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T zero = ring.getZero< T >();
	T one = ring.getOne< T >();

	alp::Matrix< T, alp::structures::General > A( n, n );
	alp::Scalar< T > zero_scalar( zero );
	alp::Scalar< T > one_scalar( one );

	assert( !alp::internal::getInitialized( A ) );
	rc = alp::set( A, one_scalar );
	assert( rc == alp::SUCCESS );
	assert( alp::internal::getInitialized( A ) );

	alp::Matrix< T, alp::structures::General > B( n, n );
	// set with matching structures and sizes, but source is uninitialized
	rc = set( A, B );
	assert( rc == alp::SUCCESS );
	assert( !alp::internal::getInitialized( A ) );

	// re-initialize matrix A
	rc = set( A, one_scalar );
	assert( rc == alp::SUCCESS );

	// set a matrix to another matrix of the same structure but different size
	alp::Matrix< T, alp::structures::General > C( 2 * n, n );
	rc = set( C, A );
	assert( rc == alp::MISMATCH );

	// set a symmetric matrix to a scalar
	alp::Matrix< T, alp::structures::Symmetric > D( n, n );
	rc = set( D, one_scalar );
	assert( rc == alp::SUCCESS );
	assert( alp::internal::getInitialized( D ) );
	for( size_t i = 0; i < nrows( D ); ++i ) {
		for( size_t j = i; j < ncols( D ); ++j ) {
			assert( *one_scalar == alp::internal::access( D, alp::internal::getStorageIndex( D, i, j ) ) );
		}
	}

	// set a symmetric matrix to another symmetric matrix
	alp::Matrix< T, alp::structures::Symmetric > E( n, n );
	rc = set( E, D );
	assert( rc == alp::SUCCESS );
	assert( alp::internal::getInitialized( E ) );
	for( size_t i = 0; i < nrows( E ); ++i ) {
		for( size_t j = i; j < ncols( E ); ++j ) {
			assert(
				alp::internal::access( E, alp::internal::getStorageIndex( E, i, j ) ) ==
				alp::internal::access( D, alp::internal::getStorageIndex( D, i, j ) )
			);
		}
	}

	alp::Vector< T > v( n );
	assert( !alp::internal::getInitialized( v ) );

	// set vector to a scalar
	rc = set( v, one_scalar );
	assert( rc == alp::SUCCESS );
	assert( alp::internal::getInitialized( v ) );

	// set vector to another vector
	alp::Vector< T > u( n );
	rc = set( u, v );
	assert( rc == alp::SUCCESS );
	assert( v[ 0 ] == u[ 0 ] );
	assert( alp::internal::getInitialized( u ) );

	// set scalar to another scalar
	alp::Scalar< T > beta;
	assert( !alp::internal::getInitialized( beta ) );
	rc = set( beta, one_scalar );
	assert( rc == alp::SUCCESS );
	assert( alp::internal::getInitialized( beta ) );
	assert( *beta == *one_scalar );

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

