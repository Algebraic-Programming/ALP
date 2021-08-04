
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

#include "graphblas/utils/Timer.hpp"

#include "bench_kernels.h" //for bench_kernels_axpy

#include <graphblas.hpp>

using namespace grb;

struct Output {
	grb::utils::TimerResults times;
	grb::RC error;
	size_t reps_used;
};

struct Input {
	size_t n;
	size_t rep;
};

enum BenchMode { TEMPLATED, LAMBDA, RAW };

template< BenchMode mode >
void test( const struct Input & in, struct Output & out ) {
	grb::utils::Timer timer;
	grb::Monoid< grb::operators::add< double >, grb::identities::zero > realm;

	// set io time to 0
	out.times.io = 0;

	// start preamble
	timer.reset();
	grb::Vector< double > xv( in.n );
	{
		grb::Vector< int > dummy( in.n );
		out.error = grb::set( dummy, 0 );
		if( out.error == grb::SUCCESS ) {
			out.error = grb::set< grb::descriptors::use_index >( xv, dummy );
		}
	}
	if( out.error != grb::SUCCESS ) {
		return;
	}

	// set constant multiplicant to x
	double * x = xv.raw();
	double alpha = 0.0;
	const double expected = in.n * ( in.n - 1 ) / 2;

	if( mode == TEMPLATED ) {
		double ttime = timer.time();
		// get cache `hot'
		out.error = grb::foldl< grb::descriptors::dense >( alpha, xv, realm );
		if( out.error != SUCCESS ) {
			std::cerr << "grb::reduce returns non-SUCCESS exit code " << grb::toString( out.error ) << ".\n";
			return;
		}
		// use this to infer number of inner iterations, if requested to be computed
		ttime = timer.time() - ttime;
		if( in.rep == 0 ) {
			out.reps_used = static_cast< size_t >( 1000.0 / ttime ) + 1;
			std::cout << "Auto-selected " << out.reps_used << " inner repititions of approx. " << ttime
					  << " ms. each (to achieve around 1 second if inner loop "
						 "wall-clock time).\n";
		} else {
			out.reps_used = in.rep;
		}
		out.times.preamble = timer.time();
		timer.reset();
		// benchmark templated axpy
		for( size_t i = 0; i < out.reps_used; ++i ) {
			alpha = 0.0;
			(void)grb::foldl< grb::descriptors::dense >( alpha, xv, realm );
		}
		out.times.useful = timer.time() / static_cast< double >( out.reps_used );

		// postamble
		timer.reset();
		if( ! grb::utils::equals( expected, alpha, in.n - 1 ) ) {
			std::cout << expected << " (expected) does not equal " << alpha << " (template optimised).\n";
			out.error = FAILED;
			return;
		}
		out.times.postamble = timer.time();
	}

	if( mode == LAMBDA ) {
		if( ! grb::Properties<>::writableCaptured ) {
			std::cerr << "grb::eWiseLambda called to reduce while the backend "
						 "does not support writable captured instances.\n";
			return;
		}
		double ltime = timer.time();
		// get cache `hot'
		alpha = realm.template getIdentity< double >();
		out.error = grb::eWiseLambda(
			[ &alpha, &xv, &realm ]( const size_t i ) {
				(void)grb::foldl( alpha, xv[ i ], realm.getOperator() );
			},
			xv );
		if( out.error != SUCCESS ) {
			std::cerr << "grb::eWiseLambda returns non-SUCCESS exit code " << grb::toString( out.error ) << ".\n";
			return;
		}
		// use this to infer number of inner iterations, if requested to be computed
		ltime = timer.time() - ltime;
		if( in.rep == 0 ) {
			out.reps_used = static_cast< size_t >( 1000.0 / ltime ) + 1;
			std::cout << "Auto-selected " << out.reps_used << " inner repititions of approx. " << ltime
					  << " ms. each (to achieve around 1 second if inner loop "
						 "wall-clock time).\n";
		} else {
			out.reps_used = in.rep;
		}
		out.times.preamble = timer.time();
		timer.reset();
		// benchmark templated axpy
		for( size_t i = 0; i < out.reps_used; ++i ) {
			alpha = realm.template getIdentity< double >();
			(void)grb::eWiseLambda(
				[ &alpha, &xv, &realm ]( const size_t i ) {
					(void)grb::foldl( alpha, xv[ i ], realm.getOperator() );
				},
				xv );
		}
		out.times.useful = timer.time() / static_cast< double >( out.reps_used );

		// postamble
		timer.reset();
		for( size_t i = 0; i < in.n; ++i ) {
			if( ! grb::utils::equals( expected, alpha, in.n - 1 ) ) {
				std::cout << expected << " (expected) does not equal " << alpha << " (eWiseLambda).\n";
				out.error = FAILED;
				return;
			}
		}
		out.times.postamble = timer.time();
	}

	if( mode == RAW ) {
		double ctime = timer.time();
		double alpha;
		// get cache `hot'
		bench_kernels_reduce( &alpha, x, in.n );
		// use this to infer number of inner iterations, if requested to be computed
		ctime = timer.time() - ctime;
		if( in.rep == 0 ) {
			out.reps_used = static_cast< size_t >( 1000.0 / ctime ) + 1;
			std::cout << "Auto-selected " << out.reps_used << " inner repititions of approx. " << ctime
					  << " ms. each (to achieve around 1 second if inner loop "
						 "wall-clock time).\n";
		} else {
			out.reps_used = in.rep;
		}
		out.times.preamble = timer.time();
		timer.reset();
		// benchmark raw axpy
		for( size_t k = 0; k < out.reps_used; ++k ) {
			bench_kernels_reduce( &alpha, x, in.n );
		}
		out.times.useful = timer.time() / static_cast< double >( out.reps_used );

		// postamble
		timer.reset();
		for( size_t i = 0; i < in.n; ++i ) {
			if( ! grb::utils::equals( alpha, expected, in.n - 1 ) ) {
				std::cout << alpha << " (compiler optimised) does not equal " << expected << " (expected) at position.\n";
				out.error = FAILED;
				return;
			}
		}
		out.times.postamble = timer.time();
	}

	// done
	out.error = SUCCESS;
}

int main( int argc, char ** argv ) {
	// sanity check on program args
	if( argc < 2 || argc > 4 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <vector length> (inner iterations) (outer iterations)" << std::endl;
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
		std::cerr << "Could not parse argument " << argv[ 1 ] << " for vector length.\n Test FAILED." << std::endl;
		return 10;
	}

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

	// prepare benchmarker
	grb::Benchmarker< AUTOMATIC > bench;

	// start functional test
	std::cout << "\nBenchmark label: grb::reduce of size " << in.n << std::endl;
	grb::RC rc = bench.exec( &(test< TEMPLATED >), in, out, 1, outer, true );
	if( rc == SUCCESS && grb::Properties<>::writableCaptured ) {
		std::cout << "\nBenchmark label: grb::eWiseLambda (reduce) of size " << in.n << std::endl;
		rc = bench.exec( &(test< LAMBDA >), in, out, 1, outer, true );
	}
	if( rc == SUCCESS ) {
		std::cout << "\nBenchmark label: compiler-optimised reduce of size " << in.n << std::endl;
		rc = bench.exec( &(test< RAW >), in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "Error launching test; exec returns " << grb::toString( rc ) << ".\n Test FAILED." << std::endl;
		return EXIT_FAILURE;
	}
	if( out.error != SUCCESS ) {
		std::cerr << "Functional test exits with nonzero exit code. Reason: " << grb::toString( out.error ) << "." << std::endl;
		std::cout << "Test FAILED.\n" << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "NOTE: please check the above performance figures manually-- "
				 "the timings should approximately match.\n";

	// done
	std::cout << "Test OK.\n" << std::endl;
	return EXIT_SUCCESS;
}
