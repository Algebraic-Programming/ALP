
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

#include "../utils/print_alp_containers.hpp"

#include <alp.hpp>

typedef int T;
constexpr T ALPHA_VALUE = 3;
constexpr T BETA_VALUE = 2;

// Function used to initialize elements of matrix A
T f_A( const size_t i, const size_t j, const size_t n ){
	return static_cast< T >( n * i + j );
}

// Function used to initialize elements of matrix B
T f_B( const size_t i, const size_t j, const size_t n ){
	return static_cast< T >( i + n * j );
}

// Function used to initialize elements of matrix C
T f_C( const size_t i, const size_t j, const size_t n ){
	(void)n;
	return static_cast< T >( i ) - static_cast< T >( j );
}

// Function used to calculate C = alpha .* B
T f_alphaB( const size_t i, const size_t j, const size_t n ) {
	const T b_value = f_B( i, j, n );
	const T c_value = f_C( i, j, n );
	return c_value + ALPHA_VALUE * b_value;
}

// Function used to calculate C = A .* beta
T f_Abeta( const size_t i, const size_t j, const size_t n ) {
	const T a_value = f_A( i, j, n );
	const T c_value = f_C( i, j, n );
	return c_value + a_value * BETA_VALUE;
}

// Function used to calculate C = A .* B
T f_AB( const size_t i, const size_t j, const size_t n ) {
	const T a_value = f_A( i, j, n );
	const T b_value = f_B( i, j, n );
	const T c_value = f_C( i, j, n );
	return c_value + a_value * b_value;
}

template< typename MatrixType, typename Function >
alp::RC initialize( MatrixType &A, const Function &f ) {
	const size_t n = nrows( A );
	const alp::RC rc = alp::eWiseLambda(
		[ &f, n ]( const size_t i, const size_t j, T &val ) {
			val = f( i, j, n );
		},
		A
	);

	if( rc != alp::SUCCESS ){
		std::cerr << "\tFailed to initialize matrix. alp::eWiseLambda (matrix, no vectors) FAILED\n";
	}
	return rc;
}

// Checks if each element of provided matrix match the value calculated by the provided function
template< typename MatrixType, typename Function >
alp::RC check_correctness( const MatrixType &A, const Function &f ) {
	for( size_t i = 0; i < nrows( A ); ++i ) {
		for( size_t j = 0; j < ncols( A ); ++j ) {
			const T expected = f( i, j, nrows( A ) );
			const auto result = alp::internal::access( A, alp::internal::getStorageIndex( A, i, j ) );
			if( expected != result ) {
				return alp::FAILED;
			}
		}
	}
	return alp::SUCCESS;
}

bool verify_success( const alp::RC &rc, const std::string &message ) {
	if( rc != alp::SUCCESS ) {
		std::cerr << message << "\n";
		return false;
	} else {
		return true;
	}
}

void alp_program( const size_t &n, alp::RC &rc ) {

	// This test is designed to work with the ring below
	// because it assumes that operators::add and operators::mul are equivalent to
	// C++ operators + and *, respectively, for the value type T defined at the top.
	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	alp::Matrix< T, alp::structures::General > A( n, n );
	alp::Matrix< T, alp::structures::General > B( n, n );
	alp::Matrix< T, alp::structures::General > C( n, n );
	alp::Scalar< T > alpha( ALPHA_VALUE );
	alp::Scalar< T > beta( BETA_VALUE );

	alp::internal::setInitialized( A, true );
	alp::internal::setInitialized( B, true );
	alp::internal::setInitialized( C, true );

	rc = alp::SUCCESS;
	// Initialize matrices
	rc = rc ? rc : initialize( A, f_A );
	rc = rc ? rc : initialize( B, f_B );
	(void) verify_success( rc, "Input matrix initialization FAILED" );

	// Test C += alpha .* B
	rc = rc ? rc : initialize( C, f_C );
	(void) verify_success( rc, "Matrix C initialization FAILED" );

	rc = rc ? rc : alp::eWiseMul( C, alpha, B, ring );
	if( !verify_success( rc, "eWiseMul ( matrix += scalar .* matrix ) FAILED" ) ) {
		return;
	}

	rc = rc ? rc : check_correctness( C, f_alphaB );
	if( !verify_success( rc, "eWiseMul ( matrix += scalar .* matrix ) FAILED: numerically incorrect" ) ) {
#ifdef DEBUG
		print_matrix( "A", A );
		print_matrix( "B", B );
		print_matrix( "C", C );
#endif
		return;
	}

	// Test C += A .* beta
	rc = rc ? rc : initialize( C, f_C );
	(void) verify_success( rc, "Matrix C initialization FAILED" );

	rc = rc ? rc : alp::eWiseMul( C, A, beta, ring );
	if( !verify_success( rc, "eWiseMul ( matrix += matrix .* scalar ) FAILED" ) ) {
		return;
	}

	rc = rc ? rc : check_correctness( C, f_Abeta );
	if( !verify_success( rc, "eWiseMul ( matrix += matrix .* scalar ) FAILED: numerically incorrect" ) ) {
#ifdef DEBUG
		print_matrix( "A", A );
		print_matrix( "B", B );
		print_matrix( "C", C );
#endif
		return;
	}

	// Test C = A . B
	rc = rc ? rc : initialize( C, f_C );
	(void) verify_success( rc, "Matrix C initialization FAILED" );

	rc = rc ? rc : alp::eWiseMul( C, A, B, ring );
	if( !verify_success( rc, "eWiseMul ( matrix += matrix .* matrix ) FAILED" ) ) {
		return;
	}

	rc = rc ? rc : check_correctness( C, f_AB );
	if( !verify_success( rc, "eWiseMul ( matrix += matrix .* matrix ) FAILED: numerically incorrect" ) ) {
#ifdef DEBUG
		print_matrix( "A", A );
		print_matrix( "B", B );
		print_matrix( "C", C );
#endif
		return;
	}

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
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cout << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
