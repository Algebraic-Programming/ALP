
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
#include "../utils/print_alp_containers.hpp"

template< typename T >
void print_stdvec_as_matrix( std::string name, const std::vector< T > &vA, const size_t m, const size_t n, const size_t lda ) {

	std::cout << "Vec " << name << ":" << std::endl;
	for( size_t row = 0; row < m; ++row ) {
		std::cout << "[\t";
		for( size_t col = 0; col < n; ++col ) {
			std::cout << vA[ row * lda + col ] << "\t";
		}
		std::cout << "]" << std::endl;
	}
}

template< typename T, typename Operator >
void outer_stdvec_as_matrix(
	std::vector< T > &vC, const size_t ldc,
	const std::vector< T > &vA,
	const std::vector< T > &vB,
	const size_t m, const size_t n,
	const Operator oper
) {

	print_stdvec_as_matrix("vA", vA, m, 1, 1);
	print_stdvec_as_matrix("vB", vB, 1, n, n);
	print_stdvec_as_matrix("vC - PRE", vC, m, n, n);

	for( size_t i = 0; i < m; ++i ) {
		for( size_t j = 0; j < n; ++j ) {
			const T &a_val { vA[ i ] };
			const T &b_val { vB[ j ] };
			T &c_val { vC[ i * ldc + j ] };
			(void)alp::internal::apply( c_val, a_val, b_val, oper );
		}
	}

	print_stdvec_as_matrix("vC - POST", vC, m, n, n);

}

template< typename Structure, typename T >
void stdvec_build_matrix( std::vector< T > &vA, const size_t m, const size_t n, const size_t lda, const T zero, const T one ) {

	if( std::is_same< Structure, alp::structures::General >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	} else if( std::is_same< Structure, alp::structures::Symmetric >::value ) {
		std::fill( vA.begin(), vA.end(), one );
	}
}

template< typename MatType, typename T >
void diff_stdvec_matrix(
	const std::vector< T > &vA, const size_t m, const size_t n, const size_t lda,
	const MatType &mA,
	double threshold=1e-7
) {

	if( std::is_same< typename MatType::structure, alp::structures::General >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = 0; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( alp::internal::access( mA, alp::internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl;
				}
			}
		}
	} else if( std::is_same< typename MatType::structure, alp::structures::Symmetric >::value ) {
		for( size_t row = 0; row < m; ++row ) {
			for( size_t col = row; col < n; ++col ) {
				double va = ( double )( vA[ row * lda + col ] );
				double vm = ( double )( alp::internal::access( mA, alp::internal::getStorageIndex( mA, row, col ) ) );
				double re = std::abs( ( va - vm ) / va );
				if( re > threshold ) {
					std::cout << "Error ( " << row << ", " << col << " ): " << va << " v " << vm << std::endl;
				}
			}
		}
	}
}


// alp program
void alpProgram( const size_t &n, alp::RC &rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T zero = ring.getZero< T >();

	// allocate
	const size_t m = 2 * n;
	std::vector< T > u_data( m );
	for( size_t i = 0; i < u_data.size(); ++i ) {
		u_data[ i ] = i + 1;
	}
	std::vector< T > v_data( n );
	for( size_t i = 0; i < v_data.size(); ++i ) {
		v_data[ i ] = i + 1;
	}
	std::vector< T > M_data( n, zero );

	alp::Vector< T > u( m );
	alp::Vector< T > v( n );
	alp::Matrix< T, alp::structures::General > M( m, n );

	// Example with matrix view on a lambda function.
	// Create before building vector to test functor init status mechanism
	auto uvT = alp::outer( u, v, ring.getMultiplicativeOperator() );

	std::cout << "Is uvt initialized before initializing source containers? " << alp::internal::getInitialized( uvT ) << "\n";

	alp::buildVector( u, u_data.begin(), u_data.end() );
	alp::buildVector( v, v_data.begin(), v_data.end() );

	std::cout << "Is uvT initialized after initializing source containers? " << alp::internal::getInitialized( uvT ) << "\n";

	print_matrix( "uvT", uvT );

	std::vector< T > uvT_test( m * n, zero );
	outer_stdvec_as_matrix( uvT_test, n, u_data, v_data, m, n, ring.getMultiplicativeOperator() );
	diff_stdvec_matrix( uvT_test, m, n, n, uvT );

	// Example when outer product takes the same vector as both inputs.
	// This operation results in a symmetric positive definite matrix.
	auto vvT = alp::outer( v, ring.getMultiplicativeOperator() );
	print_matrix( "vvT", vvT );

	std::vector< T > vvT_test( n * n, zero );
	outer_stdvec_as_matrix( vvT_test, n, v_data, v_data, n, n, ring.getMultiplicativeOperator() );
	diff_stdvec_matrix( vvT_test, n, n, n, vvT );

	// Example with storage-based matrix
	rc = alp::outer( M, u, v, ring.getMultiplicativeOperator() );

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

