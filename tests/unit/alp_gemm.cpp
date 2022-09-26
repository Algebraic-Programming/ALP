
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
#include <utility>
#include <vector>

#include <alp.hpp>
#include <alp/algorithms/gemm.hpp>

using namespace alp;

template<
	typename MatrixType,
	typename std::enable_if_t< alp::is_matrix< MatrixType >::value > * = nullptr
>
alp::RC initialize_random( MatrixType &A ) {
	alp::internal::setInitialized( A, true );
	for( size_t i = 0; i < alp::nrows( A ); ++i ) {
		for( size_t j = 0; j < alp::ncols( A ); ++j ) {
			alp::internal::access( A, alp::internal::getStorageIndex( A, i, j ) ) = static_cast< double >( rand() ) / RAND_MAX;
		}
	}

	return alp::SUCCESS;
}

template< typename... Args >
RC gemm_dispatch( bool transposeA, bool transposeB, Args&&... args ) {
	if( transposeA ) {
		if( transposeB ) {
			return algorithms::gemm_like_example< true, true >( std::forward< Args >( args )... );
		} else {
			return algorithms::gemm_like_example< true, false >( std::forward< Args >( args )... );
		}
	} else {
		if( transposeB ) {
			return algorithms::gemm_like_example< false, true >( std::forward< Args >( args )... );
		} else {
			return algorithms::gemm_like_example< false, false >( std::forward< Args >( args )... );
		}
	}
}

void alp_program( const size_t &unit, alp::RC &rc ) {

	rc = SUCCESS;

	alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

	std::cout << "\tTesting ALP gemm_like_example\n"
	             "\tC = alpha * A * B + beta * C\n";

	// dimensions of matrices A, B and C
	size_t M = 10 * unit;
	size_t N = 20 * unit;
	size_t K = 30 * unit;

	alp::Matrix< double, structures::General > A( M, K );
	alp::Matrix< double, structures::General > B( K, N );
	alp::Matrix< double, structures::General > C( M, N );
	alp::Matrix< double, structures::General > C_orig( M, N );

	// Initialize containers A, B, C, alpha, beta
	rc = rc ? rc : initialize_random( A );
	rc = rc ? rc : initialize_random( B );
	rc = rc ? rc : initialize_random( C_orig );

#ifdef DEBUG
	if( rc != SUCCESS ) {
		std::cerr << "Initialization failed\n";
	}
#endif

	assert( rc == SUCCESS );

	constexpr double alpha_value = 0.5;
	constexpr double beta_value = 1.5;
	Scalar< double > alpha( alpha_value );
	Scalar< double > beta( beta_value );

	const std::vector< std::pair< bool, bool > > transpose_AB_configs = {
		{ false, false }, { false, true }, { true, false }, { true, true }
	};

	for( auto config : transpose_AB_configs ){
		const bool transposeA = config.first;
		const bool transposeB = config.second;

		// dimensions of views over A, B and C
		size_t m = 1 * unit;
		size_t n = 2 * unit;
		size_t k = 3 * unit;

		// Set parameters to the gemm-like algorithm
		const size_t startAr = 1;
		const size_t startAc = 2;
		const size_t startBr = 3;
		const size_t startBc = 4;
		const size_t startCr = 5;
		const size_t startCc = 6;
		const size_t stride = 2;

		rc = rc ? rc : set( C, C_orig );
#ifndef NDEBUG
		if( rc != SUCCESS ) {
			std::cerr << "Initialization of C failed\n";
		}
#endif

		// Call gemm-like algorithm
#ifndef NDEBUG
		std::cout << "Calling gemm_like_example with "
			<< ( transposeA ? "" : "non-" ) << "transposed A and "
			<< ( transposeB ? "" : "non-" ) << "transposed B.\n";
#endif
		rc = rc ? rc : gemm_dispatch(
			transposeA, transposeB,
			m, n, k,
			alpha,
			A, startAr, stride, startAc, stride,
			B, startBr, stride, startBc, stride,
			beta,
			C, startCr, stride, startCc, stride,
			ring
		);

		// Check correctness
		if( rc != SUCCESS ) {
			return;
		}

		// Check numerical correctness
		for( size_t i = 0; i < alp::nrows( C ); ++i ) {
			for( size_t j = 0; j < alp::nrows( C ); ++j ) {

				// Calculate the expected value
				double expected_value;
				double C_orig_value = alp::internal::access( C_orig, alp::internal::getStorageIndex( C_orig, i, j ) );

				if( ( i >= startCr ) && ( i < startCr + m ) && ( j >= startCc ) && ( j < startCc + n ) ) {
					double mxm_res = 0;
					for( size_t kk = startAc; kk < startAc + k * stride; kk += stride ) {
						const size_t A_i = transposeA ? kk : i;
						const size_t A_j = transposeA ? i : kk;
						const auto A_val = alp::internal::access( A, alp::internal::getStorageIndex( A, A_i, A_j ) );
						const size_t B_i = transposeB ? j : kk;
						const size_t B_j = transposeB ? kk : j;
						const auto B_val = alp::internal::access( B, alp::internal::getStorageIndex( B, B_i, B_j ) );
						mxm_res += A_val * B_val;
					}
					expected_value = alpha_value * mxm_res + beta_value * C_orig_value;
				} else {
					expected_value = C_orig_value;
				}

				// Obtain the value calculated by the gemm-like algorithm
				const auto calculated_value = alp::internal::access( C, alp::internal::getStorageIndex( C, i, j ) );

				// Compare and report
				if( expected_value != calculated_value ) {
#ifndef NDEBUG
					std::cerr << "Numerically incorrect: "
						"at (" << i << ", " << j << ") "
						"expected " << expected_value << ", but got " << calculated_value << "\n";
#endif
					rc = FAILED;
					return;
				}
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

