
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
#include <inttypes.h>

#include "graphblas/utils/timer.hpp"

#include "bench_kernels.h" //for bench_kernels_dot

#include <graphblas.hpp>


using namespace grb;

struct test_output {
	int error_code;
	double check;
	double time;
};

struct bench_output {
	int error_code;
	grb::utils::TimerResults times;
};

struct test_input {
	size_t n;
};

struct bench_input {
	size_t n;
	size_t rep;
	double check;
};

void functional_test( const struct test_input & in, struct test_output & out ) {
	out.error_code = 0;

	// declare graphBLAS data structures
	typedef grb::Vector< double > vector;
	const size_t & n = in.n;
	vector xv( n ), yv( n );

	// declare raw data structures for comparison purposes
	double check = 0.0;
	double * __restrict__ xr = NULL;
	double * __restrict__ yr = NULL;
	int prc = posix_memalign( (void **)&xr, grb::config::CACHE_LINE_SIZE::value(), n * sizeof( double ) );
	assert( prc == 0 );
	if( prc != 0 ) {
		out.error_code = 98;
		return;
	}
	prc = posix_memalign( (void **)&yr, grb::config::CACHE_LINE_SIZE::value(), n * sizeof( double ) );
	assert( prc == 0 );
	if( prc != 0 ) {
		out.error_code = 99;
		return;
	}

	// set input and calculate result sequentially
	if( grb::set< grb::descriptors::no_operation >( yv, 0.5 ) != grb::SUCCESS ) {
		out.error_code = 100;
		free( yr );
		free( xr );
		return;
	}
	if( grb::set< grb::descriptors::use_index >( xv, yv ) != grb::SUCCESS ) {
		out.error_code = 101;
		free( yr );
		free( xr );
		return;
	}
	for( size_t i = 0; i < n; ++i ) {
		xr[ i ] = (double)i;
		yr[ i ] = 0.5;
		check += 0.5 * (double)i;
	}
	out.check = check;

	// call templated dot
	grb::utils::Timer timer;
	typename grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > reals;
	timer.reset();
	double alpha = 0.0;
	const RC rc = grb::dot( alpha, xv, yv, reals );
	out.time = timer.time();
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "Call to grb::dot failed with exit code %d.\n", (int)rc );
		out.error_code = 200;
	}

	// compiler result
	double beta = 0.0;
	for( size_t i = 0; i < n; ++i ) {
		beta += xr[ i ] * yr[ i ];
	}

	// now check result
	if( ! grb::utils::equals( check, alpha, 2 * n ) ) {
		(void)printf( "%lf (templated) does not equal %lf (sequential).\n", alpha, check );
		out.error_code = 300;
	}
	if( ! grb::utils::equals( check, beta, 2 * n ) ) {
		(void)printf( "%lf (compiler) does not equal %lf (sequential).\n", beta, check );
		out.error_code = 301;
	}
	if( ! grb::utils::equals( alpha, beta, 2 * n ) ) {
		(void)printf( "%lf (templated) does not equal %lf (compiler).\n", alpha, beta );
		out.error_code = 302;
	}

	// free memory
	free( xr );
	free( yr );

	// done
}

void bench_templated( const struct bench_input & in, struct bench_output & out ) {
	out.error_code = 0;
	grb::utils::Timer timer;
	timer.reset();

	// declare graphBLAS data structures
	typedef grb::Vector< double > vector;
	const size_t & n = in.n;
	vector xv( n ), yv( n );

	// set input
	if( grb::set< grb::descriptors::no_operation >( yv, 0.5 ) != grb::SUCCESS ) {
		out.error_code = 102;
	}
	if( grb::set< grb::descriptors::use_index >( xv, 0 ) != grb::SUCCESS ) {
		out.error_code = 103;
	}

	// first do a cold run
	typename grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > reals;
	double alpha = 0.0;
	const enum RC rc = grb::dot( alpha, xv, yv, reals );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "Call to grb::dot failed with exit code %d.\n", (int)rc );
		out.error_code = 201;
	}

	// done with preamble, start useful work
	out.times.preamble = timer.time();
	timer.reset();

	// benchmark hot runs
	double ttime = 0;
	for( size_t i = 0; i < in.rep; ++i ) {
		timer.reset();
		alpha = 0.0;
		const enum RC grc = grb::dot( alpha, xv, yv, reals );
		ttime += timer.time() / static_cast< double >( in.rep );

		// sanity checks
		if( ! grb::utils::equals( in.check, alpha, 2 * n ) ) {
			(void)printf( "%lf (templated, re-entrant) does not equal %lf "
						  "(sequential).\n",
				alpha, in.check );
			out.error_code = 304;
		}
		if( grc != SUCCESS ) {
			(void)printf( "Call to grb::dot failed (re-entrant) with exit code "
						  "%d.\n",
				(int)grc );
			out.error_code = 202;
		}
	}

	// done
	out.times.useful = ttime;
	out.times.io = out.times.postamble = 0;
}

void bench_lambda( const struct bench_input & in, struct bench_output & out ) {
	out.error_code = 0;
	grb::utils::Timer timer;
	timer.reset();

	// declare graphBLAS data structures
	typedef grb::Vector< double > vector;
	const size_t & n = in.n;
	vector xv( n ), yv( n );

	// set input
	if( grb::set< grb::descriptors::no_operation >( yv, 0.5 ) != grb::SUCCESS ) {
		out.error_code = 104;
	}
	if( grb::set< grb::descriptors::use_index >( xv, 0 ) != grb::SUCCESS ) {
		out.error_code = 105;
	}

	// first do a cold run
	typename grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > reals;
	double alpha = reals.template getZero< double >();
	const RC rc = grb::eWiseLambda(
		[ &xv, &yv, &alpha, &reals ]( const size_t i ) {
			double temp = 0.0;
			const auto mul_op = reals.getMultiplicativeOperator();
			const auto add_op = reals.getAdditiveOperator();
			(void)grb::apply( temp, xv[ i ], yv[ i ], mul_op );
			(void)grb::foldl( alpha, temp, add_op );
		},
		xv );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "Error in call to grb::eWiseLambda, non-SUCCESS return code %d.\n", (int)rc );
		out.error_code = 203;
	}

	// done with preamble, start useful work
	out.times.preamble = timer.time();
	timer.reset();

	// now do a hot run
	double ltime = 0.0;
	for( size_t k = 0; k < in.rep; ++k ) {
		timer.reset();
		alpha = reals.template getZero< double >();
		const enum RC grc = grb::eWiseLambda(
			[ &xv, &yv, &alpha, &reals ]( const size_t i ) {
				double temp = xv[ i ];
				const auto mul_op = reals.getMultiplicativeOperator();
				const auto add_op = reals.getAdditiveOperator();
				// grb::operators::mul< double >::foldl( temp, yv[ i ] );        //note: these two lines are about equally
			    // grb::operators::add< double >::foldl( alpha, temp );          //      fast as the below two lines
			    // grb::foldl< grb::operators::mul< double > >( temp, yv[ i ] ); //note: these two lines are about equally
			    // grb::foldl< grb::operators::add< double > >( alpha, temp );   //      fast as the below two lines
				(void)grb::foldl( temp, yv[ i ], mul_op ); // note: these two lines are about equally
				(void)grb::foldl( alpha, temp, add_op );   //      fast as the below one. This is the
			                                               //      recommended GraphBLAS call. The
			                                               //      performance difference vs grb::dot
			                                               //      is due to vectorisation.
			                                               // alpha += xv[ i ] * yv[ i ];
			},
			xv );
		ltime += timer.time() / static_cast< double >( in.rep );

		if( ! grb::utils::equals( in.check, alpha, 2 * n ) ) {
			(void)printf( "%lf (eWiseLambda, re-entrant) does not equal %lf "
						  "(sequential).\n",
				alpha, in.check );
			out.error_code = 305;
		}
		if( grc != SUCCESS ) {
			(void)printf( "Call to grb::eWiseLambda failed (re-entrant) with "
						  "exit code %d.\n",
				(int)grc );
			out.error_code = 204;
		}
	}

	// done
	out.times.useful = ltime;
	out.times.io = out.times.postamble = 0;
}

void bench_raw( const struct bench_input & in, struct bench_output & out ) {
	out.error_code = 0;
	grb::utils::Timer timer;
	timer.reset();

	// declare raw data structures
	const size_t & n = in.n;
	double * __restrict__ xr = NULL;
	double * __restrict__ yr = NULL;
	int prc = posix_memalign( (void **)&xr, grb::config::CACHE_LINE_SIZE::value(), n * sizeof( double ) );
	assert( prc == 0 );
	prc = posix_memalign( (void **)&yr, grb::config::CACHE_LINE_SIZE::value(), n * sizeof( double ) );
	assert( prc == 0 );
#ifdef NDEBUG
	(void)prc;
#endif

	// set input
	for( size_t i = 0; i < n; ++i ) {
		xr[ i ] = (double)i;
		yr[ i ] = 0.5;
	}

	// first do a cold run
	double alpha = 0.0;
	bench_kernels_dot( &alpha, xr, yr, n );

	// done with preamble, start useful work
	out.times.preamble = timer.time();
	timer.reset();

	// now do hot run
	double ctime = 0.0;
	for( size_t k = 0; k < in.rep; ++k ) {
		timer.reset();
		bench_kernels_dot( &alpha, xr, yr, n );
		ctime += timer.time() / static_cast< double >( in.rep );

		if( ! grb::utils::equals( in.check, alpha, 2 * n ) ) {
			(void)printf( "%lf (compiler, re-entrant) does not equal %lf "
						  "(sequential).\n",
				alpha, in.check );
			out.error_code = 306;
		}
	}

	// done with useful work
	out.times.useful = ctime;

	// free memory
	timer.reset();
	free( xr );
	free( yr );
	out.times.postamble = timer.time();

	// done
	out.times.io = 0;
}

int main( int argc, char ** argv ) {
	// sanity check on program args
	if( argc < 2 || argc > 4 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <vector length> (inner iterations) (outer iterations)" << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// prepare input, output structs
	struct test_input test_in;
	struct test_output test_out;
	struct bench_input in;
	struct bench_output out;

	// get vector length
	char * end = NULL;
	in.n = strtoumax( argv[ 1 ], &end, 10 );
	if( argv[ 1 ] == end ) {
		std::cerr << "Could not parse argument " << argv[ 1 ] << " for vector length.\n Test FAILED." << std::endl;
		return 10;
	}
	test_in.n = in.n;

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	if( argc >= 3 ) {
		in.rep = strtoumax( argv[ 2 ], &end, 10 );
		if( argv[ 2 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 2 ]
					  << " for number of inner experiment repititions.\n Test "
						 "FAILED."
					  << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 4 ) {
		outer = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ]
					  << " for number of outer experiment repititions.\n Test "
						 "FAILED."
					  << std::endl;
			return 30;
		}
	}

	// prepare launcher and benchmarker
	grb::Launcher< AUTOMATIC > launch;
	grb::Benchmarker< AUTOMATIC > bench;

	// start functional test
	if( launch.exec( &functional_test, test_in, test_out, true ) != SUCCESS ) {
		std::cerr << "Error launching functional test.\n Test FAILED." << std::endl;
		return 30;
	}
	if( test_out.error_code != 0 ) {
		std::cerr << "Functional test exits with nonzero exit code " << out.error_code << "\nTest FAILED." << std::endl;
		return out.error_code;
	}

	// set remaining input fields for benchmarks
	in.check = test_out.check; // pass through checksum value to benchmarks
	if( in.rep == 0 ) {
		in.rep = static_cast< size_t >( 1000.0 / test_out.time ) + 1;
		std::cout << "Auto-selected number of inner repetitions is " << in.rep << " (at an estimated time of " << test_out.time << " ms. of useful work per benchmark).\n";
	}

	// start benchmark test 1
	std::cout << "\nBenchmark label: compiler-optimised dot product on raw "
				 "arrays of size "
			  << in.n << std::endl;
	if( bench.exec( &bench_raw, in, out, 1, outer, true ) != SUCCESS ) {
		std::cerr << "Error launching raw benchmark test.\nTest FAILED." << std::endl;
		return 60;
	}
	if( out.error_code != 0 ) {
		std::cerr << "Raw benchmark test exits with nonzero exit code " << out.error_code << "\nTest FAILED." << std::endl;
		return out.error_code;
	}

	// start benchmark test 2
	std::cout << "\nBenchmark label: grb::dot of size " << in.n << std::endl;
	if( bench.exec( &bench_templated, in, out, 1, outer, true ) != SUCCESS ) {
		std::cerr << "Error launching templated benchmark test.\n Test FAILED." << std::endl;
		return 40;
	}
	if( out.error_code != 0 ) {
		std::cerr << "Templated benchmark test exits with nonzero exit code " << out.error_code << "\nTest FAILED." << std::endl;
		return out.error_code;
	}

	// start benchmark test 3
	if( grb::Properties<>::writableCaptured ) {
		std::cout << "\nBenchmark label: grb::eWiseLambda (dot) of size " << in.n << std::endl;
		if( bench.exec( &bench_lambda, in, out, 1, outer, true ) != SUCCESS ) {
			std::cerr << "Error launching lambda benchmark test.\nTest FAILED." << std::endl;
			return 50;
		}
		if( out.error_code != 0 ) {
			std::cerr << "Lambda benchmark test exits with nonzero exit code " << out.error_code << "\nTest FAILED." << std::endl;
			return out.error_code;
		}
	} else {
		std::cout << "\nBackend does not support writing to captured scalars, "
					 "skipping benchmark of lambda-based dot product...\n\n";
	}

	// done
	std::cout << "Test OK.\n\n";
	return EXIT_SUCCESS;
}
