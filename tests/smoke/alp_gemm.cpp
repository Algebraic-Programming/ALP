
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

#include <graphblas/utils/Timer.hpp>
#include <alp.hpp>
#include <alp/algorithms/gemm.hpp>
#include "../utils/print_alp_containers.hpp"

using namespace alp;

static double tol = 1.e-7;

struct inpdata {
	size_t N = 0;
	size_t repeat = 1;
};

/**
 * Initializes matrix elements to random values between 0 and 1.
 * Assumes the matrix uses full storage.
 * \todo Add support for any type of storage.
 */
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

void alp_program( const inpdata &unit, alp::RC &rc ) {
	rc = SUCCESS;

	std::cout << "\tTesting ALP gemm_like_example\n"
	             "\tC = alpha * A * B + beta * C\n";


	const std::vector< std::pair< bool, bool > > transpose_AB_configs = {
		{ false, false }, { false, true }, { true, false }, { true, true }
	};


	grb::utils::Timer timer;
	timer.reset();
	double times[] = { 0, 0, 0, 0 };

	for( size_t jrepeat = 0; jrepeat < unit.repeat; ++jrepeat ) {

		alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

		// dimensions of matrices A, B and C
		size_t M = 10 * unit.N;
		size_t N = 20 * unit.N;
		size_t K = 30 * unit.N;

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

#ifdef DEBUG
		print_matrix( "A", A );
		print_matrix( "B", B );
		print_matrix( "C_orig", C_orig );
#endif

		constexpr double alpha_value = 0.5;
		constexpr double beta_value = 1.5;
		Scalar< double > alpha( alpha_value );
		Scalar< double > beta( beta_value );

		size_t iconfig = 0;
		for( auto config : transpose_AB_configs ){
			const bool transposeA = config.first;
			const bool transposeB = config.second;

			// dimensions of views over A, B and C
			size_t m = 1 * unit.N;
			size_t n = 2 * unit.N;
			size_t k = 3 * unit.N;

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
			timer.reset();

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

			times[ iconfig ] += timer.time();
			++iconfig;

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

					// Check if coordinates (i, j) fall into the gather view over C
					if(
						( i >= startCr ) && ( i < startCr + m * stride ) &&
						( j >= startCc ) && ( j < startCc + n * stride ) &&
						( ( i - startCr ) % stride == 0 ) &&
						( ( j - startCc ) % stride == 0 )
					) {
						double mxm_res = 0;
						for( size_t kk = 0; kk < k; ++kk ) {
							// coordinates within the gather view over C
							const size_t sub_i = ( i - startCr ) / stride;
							const size_t sub_j = ( j - startCc ) / stride;

							// take into account the gather view on A and potential transposition
							const size_t A_i = startAr + stride * ( transposeA ? kk : sub_i );
							const size_t A_j = startAc + stride * ( transposeA ? sub_i : kk );
							const auto A_val = alp::internal::access( A, alp::internal::getStorageIndex( A, A_i, A_j ) );

							// take into account the gather view on B and potential transposition
							const size_t B_i = startBr + stride * ( transposeB ? sub_j : kk );
							const size_t B_j = startBc + stride * ( transposeB ? kk : sub_j );
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
					if( std::abs( expected_value - calculated_value ) > tol ) {
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

	for( size_t i = 0; i < 4; ++i ) {
		const bool transposeA = transpose_AB_configs[ i ].first;
		const bool transposeB = transpose_AB_configs[ i ].second;
		std::cout << " GEMM versions: transposeA = " << transposeA << ", transposeB = " << transposeB << "\n";
		std::cout << " time (ms, total) = " << times[ i ] << "\n";
		std::cout << " time (ms, per repeat) = " << times[ i ] / unit.repeat  << "\n";
	}
}

int main( int argc, char **argv ) {
	// defaults
	bool printUsage = false;
	inpdata in;

	// error checking
	if(
		( argc == 3 ) || ( argc == 5 )
	) {
		std::string readflag;
		std::istringstream ss1( argv[ 1 ] );
		std::istringstream ss2( argv[ 2 ] );
		if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
			std::cerr << "Error parsing\n";
			printUsage = true;
		} else if(
			readflag != std::string( "-n" )
		) {
			std::cerr << "Given first argument is unknown\n";
			printUsage = true;
		} else {
			if( ! ( ( ss2 >> in.N ) &&  ss2.eof() ) ) {
				std::cerr << "Error parsing\n";
				printUsage = true;
			}
		}

		if( argc == 5 ) {
			std::string readflag;
			std::istringstream ss1( argv[ 3 ] );
			std::istringstream ss2( argv[ 4 ] );
			if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
				std::cerr << "Error parsing\n";
				printUsage = true;
			} else if(
				readflag != std::string( "-repeat" )
			) {
				std::cerr << "Given third argument is unknown\n";
				printUsage = true;
			} else {
				if( ! ( ( ss2 >> in.repeat ) &&  ss2.eof() ) ) {
					std::cerr << "Error parsing\n";
					printUsage = true;
				}
			}
		}

	} else {
		std::cout << "Wrong number of arguments\n";
		printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: \n";
		std::cerr << "       " << argv[ 0 ] << " -n N \n";
		std::cerr << "      or  \n";
		std::cerr << "       " << argv[ 0 ] << " -n N   -repeat N \n";
		return 1;
	}

	alp::RC rc = alp::SUCCESS;
	alp_program( in, rc );
	if( rc == alp::SUCCESS ) {
		std::cout << "Test OK\n";
		return 0;
	} else {
		std::cout << "Test FAILED\n";
		return 1;
	}
}
