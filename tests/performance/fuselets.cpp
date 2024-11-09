
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
	grb::Matrix< double > Am =
		grb::algorithms::matrices< double >::eye( in.n, in.n, 2.0, 1 );
	grb::RC rc = grb::set< grb::descriptors::use_index >( xv, 0.0 );
	rc = rc ? rc : grb::set( yv, 2.0 );
	rc = rc ? rc : grb::set( zv, 0.0 );

	double * const x = xv.raw(), * const y = yv.raw(), * const z = zv.raw();
	const auto &A = grb::internal::getCRS( Am );
	const auto * const av = A.values;
	const auto * const ai = A.col_start;
	const auto * const aj = A.row_index;
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
			std::cerr << "Error during verification (I)\n";
			out.error = grb::FAILED;
			return;
		}
		if( beta != 0.0 || beta != -0.0 ) {
			std::cerr << "Error during verification (II)\n";
			out.error = grb::FAILED;
			return;
		}
		if( y[ in.n - 1 ] != 1.0 ) {
			std::cerr << "Error during verification (III)\n"
				<< "\t expected: 1.0, got: " << y[ in.n - 1 ] << "\n";
			out.error = grb::FAILED;
			return;
		}
		for( size_t i = 0; i < in.n - 1; ++i ) {
			if( y[ i ] != static_cast< double >( 2 * (i + 1) + 1 ) ) {
				std::cerr << "Error during verification (IV)\n"
					<< "\t expected: " << ( 2 * ( i + 1 ) + 1 ) << ", got: " << y[ i ]
					<< " at position " << i << "\n";
				out.error = grb::FAILED;
				return;
			}
		}
	}

	out.times.preamble = timer.time();
	// end preamble

	if( rc != grb::SUCCESS ) {
		std::cerr << "Error during test initialisation\n";
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

	// postamble
	timer.reset();
	rc = finalize_fuselets() == 0? grb::SUCCESS : grb::PANIC;
	out.times.postamble = timer.time();

	// done
	out.error = rc;
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
		<< "the useful timings should positive, indicating speedup.\n";

	// done
	std::cout << "Test OK\n" << std::endl;
	return EXIT_SUCCESS;
}

