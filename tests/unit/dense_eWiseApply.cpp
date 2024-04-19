
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

#include <alp.hpp>

using namespace alp;

typedef int T;
constexpr T ALPHA_VALUE = 1;
constexpr T BETA_VALUE = 2;

// Function used to initialize elements of matrix A
T f_A( const size_t i, const size_t j, const size_t n ){
	return n * i + j;
}

// Function used to initialize elements of matrix B
T f_B( const size_t i, const size_t j, const size_t n ){
	return i + static_cast< T >( n ) * j;
}

// Important: The following functions must match the corresponding eWiseApply calls in this unit test
// Function used to calculate C = alpha .+ B
T f_alphaB( const size_t i, const size_t j, const size_t n ) {
	const T b_value = f_B( i, j, n );
	return ALPHA_VALUE + b_value;
}

// Function used to calculate C = A .* beta
T f_Abeta( const size_t i, const size_t j, const size_t n ) {
	const T a_value = f_A( i, j, n );
	return a_value * BETA_VALUE;
}

// Function used to calculate C = A .* B
T f_AB( const size_t i, const size_t j, const size_t n ) {
	const T a_value = f_A( i, j, n );
	const T b_value = f_B( i, j, n );
	return a_value * b_value;
}

// Checks if each element of provided matrix match the value calculated by the provided function
template< typename MatrixType, typename Function >
bool check_correctness( const MatrixType &A, const Function &f ) {
	for( size_t i = 0; i < nrows( A ); ++i ) {
		for( size_t j = 0; j < ncols( A ); ++j ) {
			const T expected = f( i, j, nrows( A ) );
			const auto result = alp::internal::access( A, alp::internal::getStorageIndex( A, i, j ) );
			if( expected != result ) {
				return false;
			}
		}
	}
	return true;
}

void alp_program( const size_t &n, alp::RC &rc ) {

	// This test is designed to work with the ring below
	// because it assumes that operators::add and operators::mul are equivalent to
	// C++ operators + and *, respectively, for the value type T defined at the top.
	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	alp::Matrix< T, structures::General > A( n, n );
	alp::Matrix< T, structures::General > B( n, n );
	alp::Matrix< T, structures::General > C( n, n );
	alp::Scalar< T > alpha( ALPHA_VALUE );
	alp::Scalar< T > beta( BETA_VALUE );

	internal::setInitialized( A, true );
	internal::setInitialized( B, true );
	internal::setInitialized( C, true );

	// Initialize matrices
	// A[i][j] = n * i + j
	rc = alp::eWiseLambda(
		[ n ]( const size_t i, const size_t j, T &val ) {
			val = f_A( i, j, n );
		},
		A
	);
	if( rc != alp::SUCCESS ){
		std::cerr << "\talp::eWiseLambda (matrix, no vectors) FAILED\n";
		return;
	}
	// B[i][j] = i + n * j
	rc = alp::eWiseLambda(
		[ n ]( const size_t i, const size_t j, T &val ) {
			val = f_B( i, j, n );
		},
		B
	);
	if( rc != alp::SUCCESS ){
		std::cerr << "\talp::eWiseLambda (matrix, no vectors) FAILED\n";
		return;
	}
	// Test C = alpha .+ B
	rc = alp::eWiseApply( C, alpha, B, ring.getAdditiveMonoid() );

	if( rc != alp::SUCCESS ){
		std::cerr << "\talp::eWiseApply ( matrix = scalar .* matrix ) FAILED\n";
	}
	if( ! check_correctness( C, f_alphaB ) ) {
		std::cerr << "\talp::eWiseApply ( matrix = scalar .* matrix ) FAILED: numerically incorrect\n";
	}

	// Test C = A .* beta
	rc = alp::eWiseApply( C, A, beta, ring.getMultiplicativeMonoid() );

	if( rc != alp::SUCCESS ){
		std::cerr << "\talp::eWiseApply ( matrix = matrix .* scalar ) FAILED\n";
	}
	if( ! check_correctness( C, f_Abeta ) ) {
		std::cerr << "\talp::eWiseApply ( matrix = matrix .* scalar ) FAILED: numerically incorrect\n";
	}

	// Test C = A . B
	rc = alp::eWiseApply( C, A, B, ring.getMultiplicativeMonoid() );

	if( rc != alp::SUCCESS ){
		std::cerr << "\talp::eWiseApply ( matrix = matrix .* matrix ) FAILED\n";
	}
	if( ! check_correctness( C, f_AB ) ) {
		std::cerr << "\talp::eWiseApply ( matrix = matrix .* matrix ) FAILED: numerically incorrect\n";
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
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
