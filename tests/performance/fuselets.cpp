
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

#include <cstdio>
#include <cstdlib>

#include <assert.h>
#include <inttypes.h> //for strtoumax

#include <graphblas/utils/timer.hpp>

#include <graphblas.hpp>

#include <graphblas/algorithms/matrix_factory.hpp>

#include <transition/fuselets.h>


struct Output {
	grb::utils::TimerResults times;
	grb::RC error;
	size_t reps_used;
};

struct Input {
	size_t n;
	size_t rep;
};

template< typename T >
static grb::Matrix< T > setupUpperDiagonalMatrix( const size_t n, const T value ) {
	return grb::algorithms::matrices< double >::eye( n, n, value, 1 );
}

template< typename T >
static grb::RC setupVectors(
	grb::Vector< T > &x,
	grb::Vector< T > &y,
	grb::Vector< T > &z
) {
	grb::RC rc = grb::set< grb::descriptors::use_index >( x, 0.0 );
	rc = rc ? rc : grb::set( y, 0.0 );
	rc = rc ? rc : grb::set( z, 2.0 );
	return rc;
}

template< typename T >
static void getCRS(
	const grb::Matrix< T > &Am,
	const T * &av, const size_t * &ai, const unsigned int * &aj
) {
	const auto &A = grb::internal::getCRS( Am );
	av = A.values;
	ai = A.col_start;
	aj = A.row_index;
}

static grb::RC verifySpMV( const double * const y, const size_t n ) {
	if( y[ n - 1 ] != 1.0 ) {
		std::cerr << "\t\t error during SpMV output verification (I)\n"
			<< "\t\t\t expected: 1.0, got: " << y[ n - 1 ]
			<< " at position " << (n-1) << "\n";
		return grb::FAILED;
	}
	for( size_t i = 0; i < n - 1; ++i ) {
		if( y[ i ] != static_cast< double >( 2 * (i + 1) + 1 ) ) {
			std::cerr << "\t\t errpr during SpMV output verification (II)\n"
				<< "\t\t\t expected: " << ( 2 * ( i + 1 ) + 1 ) << ", got: " << y[ i ]
				<< " at position " << i << "\n";
			return grb::FAILED;
		}
	}
	return grb::SUCCESS;
}

void test_spmv_dot( const struct Input &in, struct Output &out ) {
	grb::utils::Timer timer;
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> reals;

	// set io time to 0
	out.times.io = 0;

	// start preamble
	timer.reset();
	grb::Vector< double > xv( in.n ), yv( in.n ), zv( in.n );
	grb::Matrix< double > Am = setupUpperDiagonalMatrix( in.n, 2.0 );
	grb::RC rc = setupVectors( xv, zv, yv );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test_spmv_dot: test initialisation FAILED (I)\n";
		out.error = grb::FAILED;
		return;
	}

	double * const x = xv.raw(), * const y = yv.raw(), * const z = zv.raw();
	const double * av = nullptr;
	const size_t * ai = nullptr; const unsigned int * aj = nullptr;
	getCRS( Am, av, ai, aj );
	double beta = 0.0;

	rc = rc ? rc : (initialize_fuselets() == 0 ? grb::SUCCESS : grb::FAILED);

	// verify
	if( rc == grb::SUCCESS ) {
		// A has values 2 above its diagonal
		// x is a dense vector with values (0, 1, ..., n-1)
		// y is a dense vector of twos
		// z is a dense vector of zeroes
		// therefore, calling the spmv_dot fuselet with alpha = 0.5 should result in:
		//  - beta = 0.0
		//  - y dense with values(3, 5, 7, ..., 2(n-1), 1)
		beta = 1.0;
		const int fuselet_rc = spmv_dot_dsu(
			y, &beta,
			ai, aj, av,
			x,
			0.5, z,
			in.n
		);
		if( fuselet_rc != 0 ) {
			std::cerr << "\t test_spmv_dot: verification FAILED (I)\n";
			out.error = grb::FAILED;
			return;
		}
		if( beta != 0.0 || beta != -0.0 ) {
			std::cerr << "\t test_spmv_dot: verification FAILED (II)\n";
			out.error = grb::FAILED;
			return;
		}
		out.error = verifySpMV( y, in.n );
		if( out.error != grb::SUCCESS ) {
			std::cerr << "\t test_spmv_dot: verification FAILED (III)\n";
			return;
		}
	}

	out.times.preamble = timer.time();
	// end preamble

	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test_spmv_dot: test initialisation FAILED (II)\n";
		out.error = rc;
		return;
	}

	// benchmark

	timer.reset();
	for( size_t i = 0; i < in.rep; ++i ) {
		beta = 0.0;
		(void) spmv_dot_dsu(
			y, &beta,
			ai, aj, av,
			x,
			0.5, z,
			in.n
		);
	}
	const double fast = timer.time();

	timer.reset();
	for( size_t i = 0; i < in.rep; ++i ) {
		(void) grb::foldr< grb::descriptors::dense >( 0.5, yv,
			grb::operators::mul< double >() );
		(void) grb::mxv< grb::descriptors::dense >( yv, Am, xv,
			grb::semirings::plusTimes< double >() );
		beta = 0.0;
		(void) grb::dot< grb::descriptors::dense >( beta, zv, yv,
			grb::semirings::plusTimes< double >() );
		(void) grb::wait();
	}
	const double slow = timer.time();

	// record speedup
	std::cout << "\t reference_omp: " << slow << " ms.\n\t fuselets: " << fast
		<< " ms.\n\t (for " << in.rep << " spmv_dots)\n";
	out.times.useful =
		static_cast< double >(slow - fast) / static_cast< double >(in.rep);

	// postamble, and done
	timer.reset();
	out.error = finalize_fuselets() == 0 ? grb::SUCCESS : grb::PANIC;
	out.times.postamble = timer.time();
}

void test_spmv_dot_norm2(
	const struct Input &in, struct Output &out
) {
	grb::utils::Timer timer;

	// I/O phase is empty
	out.times.io = 0;

	// start preamble
	timer.reset();
	grb::Vector< double > xv( in.n ), yv( in.n ), zv( in.n );
	grb::Matrix< double > Am = setupUpperDiagonalMatrix( in.n, 2.0 );
	grb::RC rc = setupVectors( xv, yv, zv );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test_spmv_dot_norm2: test initialisation FAILED (I)\n";
		out.error = grb::FAILED;
		return;
	}

	double * const x = xv.raw(), * const y = yv.raw(), * const z = zv.raw();
	const double * av = nullptr;
	const size_t * ai = nullptr; const unsigned int * aj = nullptr;
	getCRS( Am, av, ai, aj );
	double beta, gamma;
	beta = gamma = 0.0;

	rc = initialize_fuselets() == 0 ? grb::SUCCESS : grb::FAILED;

	// verify
	if( rc == grb::SUCCESS ) {
		// A and x are as in the above test
		// y is dense with entries 4.23 everywhere
		// z is dense with entries twos everywhere
		// therefore, as in the above, after spmv_dot_norm2, the output
		//  - z should be dense with values (3, 5, 7, ..., 2(n-1), 1)
		//  - beta should be zero
		// new to the above, gamma should read norm2-squared of z
		beta = 1.13;
		gamma = 2.17;
		const int fuselet_rc = spmv_dot_norm2_dsu(
			z, &beta, &gamma,
			ai, aj, av,
			x, 0.5, y,
			in.n
		);
		if( fuselet_rc != 0 ) {
			std::cerr << "\t test_spmv_dot_norm2: verification FAILED (I)\n";
			out.error = grb::FAILED;
			return;
		}
		if( beta != 0.0 || beta != -0.0 ) {
			std::cerr << "\t test_spmv_dot_norm2: verification FAILED (II)\n";
			out.error = grb::FAILED;
			return;
		}
		double check_gamma = z[ in.n - 1];
		check_gamma *= check_gamma;
		for( size_t i = 0; i < in.n - 1; ++i ) {
			check_gamma += z[ i ] * z[ i ];
		}
		if( !grb::utils::equals( gamma, check_gamma, 2 * in.n - 1) ) {
			std::cerr << "\t test_spmv_dot_norm2: verification FAILED (III)\n"
				<< "\t\t expected: " << check_gamma << ", got: " << gamma << "\n";
			out.error = grb::FAILED;
			return;
		}
		out.error = verifySpMV( z, in.n );
		if( out.error != grb::SUCCESS ) {
			std::cerr << "\t test_spmv_dot_norm2: verification FAILED (IV)\n";
			return;
		}
	}

	out.times.preamble = timer.time();
	// end preamble

	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test_spmv_dot_norm2: test initialisation FAILED (II)\n";
		out.error = grb::FAILED;
		return;
	}

	// benchmark

	timer.reset();
	for( size_t i = 0; i < in.rep; ++i ) {
		beta = gamma = 0.0;
		(void) spmv_dot_norm2_dsu(
			z, &beta, &gamma,
			ai, aj, av,
			x, 0.5, y,
			in.n
		);
	}
	const double fast = timer.time();

	timer.reset();
	for( size_t i = 0; i < in.rep; ++i ) {
		grb::semirings::plusTimes< double > plusTimes_FP64;
		(void) grb::foldr< grb::descriptors::dense >( 0.5, zv,
			grb::operators::mul< double >() );
		(void) grb::mxv< grb::descriptors::dense >( zv, Am, xv,
			grb::semirings::plusTimes< double >() );
		beta = gamma = 0.0;
		(void) grb::dot< grb::descriptors::dense >( beta, yv, zv, plusTimes_FP64 );
		(void) grb::dot< grb::descriptors::dense >( gamma, zv, zv, plusTimes_FP64 );
		(void) grb::wait();
	}
	const double slow = timer.time();

	// record speedup:
	std::cout << "\t test_spmv_dot_norm2 (" << in.rep << " repetitions):\n"
		<< "\t\t reference_omp: " << slow << " ms.\n"
		<< "\t\t fuselets: " << fast << " ms.\n";
	out.times.useful =
		static_cast< double >(slow - fast) / static_cast< double >(in.rep);

	// postamble, and done
	timer.reset();
	out.error = finalize_fuselets() == 0 ? grb::SUCCESS : grb::PANIC;
	out.times.postamble = timer.time();
}

void test_update_spmv_dot(
	const struct Input &in, struct Output &out
) {
	grb::utils::Timer timer;

	// I/O phase is empty
	out.times.io = 0;

	// start preamble
	timer.reset();
	grb::Vector< double > xv( in.n ), yv( in.n ), zv( in.n );
	grb::Matrix< double > Am = setupUpperDiagonalMatrix( in.n, 2.0 );
	grb::RC rc = setupVectors( xv, yv, zv );
	rc = rc ? rc : grb::set< grb::descriptors::dense >( yv, 4.23 );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test_spmv_dot_norm2: test initialisation FAILED (I)\n";
		out.error = grb::FAILED;
		return;
	}

	double * const x = xv.raw(), * const y = yv.raw(), * const z = zv.raw();
	const double * av = nullptr;
	const size_t * ai = nullptr; const unsigned int * aj = nullptr;
	getCRS( Am, av, ai, aj );
	double beta, gamma;
	beta = -1.615;
	gamma = 1.17;

	rc = initialize_fuselets() == 0 ? grb::SUCCESS : grb::FAILED;

	// verify
	if( rc == grb::SUCCESS ) {
		// A and x are as in the above test:
		//  - A has values 2 above its diagonal
		//  - x is a dense vector with values (0, 1, ..., n-1)
		// y is dense with entries 4.23 everywhere
		// z is dense with entries twos everywhere
		// therefore, after update_spmv_dot_dsu, the output
		//  - z should be dense with value one (1.0) everywhere
		//  - x should be dense with value two (2.0) everywhere
		//     - except at its last position, which should read zero (0)
		//  - gamma should read 2.0 * (n-1)
		const int fuselet_rc = update_spmv_dot_dsu(
			z, x, &gamma,
			y, beta,
			ai, aj, av,
			in.n
		);
		if( fuselet_rc != 0 ) {
			std::cerr << "\t update_spmv_dot: verification FAILED (I)\n";
			out.error = grb::FAILED;
			return;
		}
		{
			bool fail = false;
			for( size_t i = 0; i < in.n; ++i ) {
				if( !grb::utils::equals( z[ i ], 1.0, 3 ) ) {
					fail = true;
					std::cerr << "\t\t z[ " << i << " ], expected one, got "
						<< z[ i ] << "\n";
				}
			}
			if( fail ) {
				std::cerr << "\t update_spmv_dot: verification FAILED (II)\n";
				out.error = grb::FAILED;
				return;
			}
		}
		double check_gamma = z[ in.n - 1 ] * x[ in.n - 1 ];
		for( size_t i = 0; i < in.n - 1; ++i ) {
			check_gamma += z[ i ] * x[ i ];
		}
		if( !grb::utils::equals( gamma, check_gamma, 2 * in.n - 1 ) ) {
			std::cerr << "\t update_spmv_dot: verification FAILED (III)\n"
				<< "\t\t expected: " << check_gamma << ", got: " << gamma << "\n";
			out.error = grb::FAILED;
			return;
		}
		{
			bool fail = false;
			constexpr double zero = 0.0;
			constexpr double two = 2.0;
			if( !grb::utils::equals( x[ in.n - 1 ], zero, 4 ) ) {
				fail = true;
				std::cerr << "\t\t x[ " << (in.n-1) << " ] (last entry) "
				       << "expected zero, got " << x[ in.n - 1 ] << "\n";
			}
			for( size_t i = 0; i < in.n - 1; ++i ) {
				if( !grb::utils::equals( x[ i ], two, 4 ) ) {
					fail = true;
					std::cerr << "\t\t x[ " << i << " ] expected 2.0, got "
						<< x[ i ] << "\n";
				}
			}
			if( fail ) {
				std::cerr << "\t update_spmv_dot: verification FAILED (IV)\n";
				out.error = grb::FAILED;
				return;
			}
		}
	}

	out.times.preamble = timer.time();
	// end preamble

	if( rc != grb::SUCCESS ) {
		std::cerr << "\t update_spmv_dot: test initialisation FAILED (II)\n";
		out.error = grb::FAILED;
		return;
	}

	// benchmark

	timer.reset();
	for( size_t i = 0; i < in.rep; ++i ) {
		beta = -1.615;
		gamma = 1.17;
		(void) update_spmv_dot_dsu(
			z, x, &gamma,
			y, beta,
			ai, aj, av,
			in.n
		);
	}
	const double fast = timer.time();

	timer.reset();
	for( size_t i = 0; i < in.rep; ++i ) {
		beta = -1.615;
		gamma = 1.17;
		grb::semirings::plusTimes< double > plusTimes_FP64;
		(void) grb::foldr< grb::descriptors::dense >( beta, zv,
			grb::operators::mul< double >() );
		(void) grb::foldr< grb::descriptors::dense >( yv, zv,
			grb::operators::add< double >() );
		(void) grb::set< grb::descriptors::dense >( xv, 0.0 );
		(void) grb::mxv< grb::descriptors::dense >( xv, Am, zv, plusTimes_FP64 );
		gamma = 0.0;
		(void) grb::dot< grb::descriptors::dense >( gamma, zv, xv, plusTimes_FP64 );
		(void) grb::wait();
	}
	const double slow = timer.time();

	// record speedup:
	std::cout << "\t test_update_spmv_dot (" << in.rep << " repetitions):\n"
		<< "\t\t reference_omp: " << slow << " ms.\n"
		<< "\t\t fuselets: " << fast << " ms.\n";
	out.times.useful =
		static_cast< double >(slow - fast) / static_cast< double >(in.rep);

	// postamble, and done
	timer.reset();
	out.error = finalize_fuselets() == 0 ? grb::SUCCESS : grb::PANIC;
	out.times.postamble = timer.time();
}

int main( int argc, char ** argv ) {
	// sanity check on program args
	if( argc < 2 || argc > 4 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <vector length> (inner iterations) "
			<< "(outer iterations)" << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// prepare input, output structs
	struct Input in;
	struct Output out;

	// get vector length
	char * end = NULL;
	in.n = strtoumax( argv[ 1 ], &end, 10 );
	if( argv[ 1 ] == end ) {
		std::cerr << "Could not parse argument " << argv[ 1 ] << " "
			<< "for vector length." << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return 10;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	if( argc >= 3 ) {
		in.rep = strtoumax( argv[ 2 ], &end, 10 );
		if( argv[ 2 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 2 ] << " "
				<< "for number of inner experiment repititions." << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 4 ) {
		outer = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 30;
		}
	}

	// prepare benchmarker
	grb::Benchmarker< grb::AUTOMATIC > bench;

	// start tests
	std::cout << "\nBenchmark label: spmv_dot of size " << in.n
		<< std::endl;
	grb::RC rc = bench.exec( &(test_spmv_dot), in, out, 1, outer, true );

	if( rc == grb::SUCCESS ) {
		std::cout << "\nBenchmark label: spmv_dot_norm2 of size " << in.n
			<< std::endl;
		rc = bench.exec( &(test_spmv_dot_norm2), in, out, 1, outer, true );
	}

	if( rc == grb::SUCCESS ) {
		std::cout << "\nBenchmark label: update_spmv_dot of size " << in.n
			<< std::endl;
		rc = bench.exec( &(test_update_spmv_dot), in, out, 1, outer, true );
	}

	if( rc != grb::SUCCESS ) {
		std::cerr << "Test launch failed: " << grb::toString( rc ) << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return EXIT_FAILURE;
	}

	if( out.error != grb::SUCCESS ) {
		std::cerr << "Functional test exits with nonzero exit code. "
			<< "Reason: " << grb::toString( out.error ) << "." << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "NOTE: please check the above performance figures manually-- "
		<< "the useful timings should positive, indicating speedup for fuselets vs. "
		<< "regular blocking execution (for large enough vector lengths)\n";

	// done
	std::cout << "Test OK\n" << std::endl;
	return EXIT_SUCCESS;
}

