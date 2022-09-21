
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
#include <vector>

#include <alp.hpp>
#include <alp/algorithms/gemm.hpp>

using namespace alp;

void alp_program( const size_t &unit, alp::RC &rc ) {

	rc = SUCCESS;

	alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

	std::cout << "\tTesting ALP gemm_like_example\n"
	             "\tC = alpha * A * B + beta * C\n";

	// dimensions of matrices A, B and C
	size_t M = 10 * unit;
	size_t N = 20 * unit;
	size_t K = 30 * unit;

	// dimensions of views over A, B and C
	size_t m = unit;
	size_t n = 2 * unit;
	size_t k = 3 * unit;

	alp::Matrix< double, structures::General > A( M, K );
	alp::Matrix< double, structures::General > B( K, N );
	alp::Matrix< double, structures::General > C( M, N );

	// Initialize containers A, B, C, alpha, beta
	constexpr double A_value = 2;
	constexpr double B_value = 3;
	constexpr double C_value = 4;
	rc = rc ? rc : alp::set( A, alp::Scalar< double >( A_value ) );
	rc = rc ? rc : alp::set( B, alp::Scalar< double >( B_value ) );
	rc = rc ? rc : alp::set( C, alp::Scalar< double >( C_value ) );

	constexpr double alpha_value = 0.5;
	constexpr double beta_value = 1.5;
	Scalar< double > alpha( alpha_value );
	Scalar< double > beta( beta_value );

#ifdef DEBUG
	if( rc != SUCCESS ) {
		std::cerr << "Initialization failed\n";
	}
#endif

	assert( rc == SUCCESS );

	// Set parameters to the gemm-like algorithm
	const size_t startAr = 1;
	const size_t startAc = 1;
	const size_t startBr = 2;
	const size_t startBc = 2;
	const size_t startCr = 3;
	const size_t startCc = 3;
	const size_t stride = 1;


	// Call gemm-like algorithm
	rc = rc ? rc : algorithms::gemm_like_example(
		m, n, k,
		alpha,
		A, startAr, stride, startAc, stride,
		B, startBr, stride, startBc, stride,
		beta,
		C, startCr, stride, startCc, stride,
		ring
	);
	assert( rc == SUCCESS );

	// Check correctness
	if( rc != SUCCESS ) {
		return;
	}

	// Check numerical correctness
	// Elements in the submatrix should be equal to
	//     alpha_value * A_value * B_value * k + beta_value * C_value,
	// while the elements outside the submatrix should be equal to
	//     C_value.
	for( size_t i = 0; i < alp::nrows( C ); ++i ) {
		for( size_t j = 0; j < alp::nrows( C ); ++j ) {

			const double expected_value = ( ( i >= startCr ) && ( i < startCr + m ) && ( j >= startCc ) && ( j < startCc + n ) ) ?
				alpha_value * A_value * B_value * k + beta_value * C_value :
				C_value;

			const auto calculated_value = alp::internal::access( C, alp::internal::getStorageIndex( C, i, j ) );

			if( expected_value != calculated_value ) {
				std::cerr << "Numerically incorrect: "
					"at (" << i << ", " << j << ") "
					"expected " << expected_value << ", but got " << calculated_value << "\n";
				rc = FAILED;
				return;
			}
		}
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
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

