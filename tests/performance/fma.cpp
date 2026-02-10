
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
void test( const struct Input &in, struct Output &out ) {
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

	double *x = xv.raw(), *y = yv.raw(), *z = zv.raw();

	out.error = grb::set( yv, 1 );
	if( out.error != grb::SUCCESS ) {
		std::cerr << "Error during initialisation of yv "
			<< grb::toString( out.error ) << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return;
	}

	out.error = grb::set( zv, 0 );
	if( out.error != grb::SUCCESS ) {
		std::cerr << "Error during initialisation of zv "
			<< grb::toString( out.error ) << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return;
	}

	out.error = grb::set< grb::descriptors::use_index >( xv, zv );
	if( out.error != grb::SUCCESS ) {
		std::cerr << "Error during initialisation of xv "
			<< grb::toString( out.error ) << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return;
	}

	if( grb::nnz( yv ) != in.n || grb::nnz( zv ) != in.n ) {
		std::cerr << "Unexpected number of nonzeroes in yv or zv: expected " << in.n
			<< ", got " << grb::nnz(yv) << " and " << grb::nnz(zv) << "\n";
		std::cout << "Test FAILED\n" << std::endl;
		out.error = grb::FAILED;
		return;
	}

	if( grb::nnz( xv ) != in.n ) {
		std::cerr << "Unexpected number of nonzeroes in xv: expected " << in.n
			<< ", got " << grb::nnz(xv) << "\n";
		std::cout << "Test FAILED\n" << std::endl;
		out.error = grb::FAILED;
		return;
	}

	{
		size_t errors = 0;
		bool suppress = false;
		for( size_t i = 0; i < in.n; ++i ) {
			if( x[ i ] != static_cast< double >( i ) ) {
				if( !suppress ) {
					std::cerr << "Unexpected value x[ " << i << " ] = " << x[ i ] << ", "
						<< "expected " << i << ". Test initialisation FAILED\n";
				}
				(void) ++errors;
			}
			if( y[ i ] != 1.0 ) {
				if( !suppress ) {
					std::cerr << "Unexpected value y[ " << i << " ] = " << y[ i ] << ", "
						<< "expected " << i << ". Test initialisation FAILED\n";
				}
				(void) ++errors;
			}
			if( z[i ] != 0.0 || z[ i ] != -0.0 ) {
				if( !suppress ) {
					std::cerr << "Unexpected value z[ " << z << " ] = " << z[ i ] << ", "
						<< "expected " << i << ". Test initalisation FAILED\n";
				}
				(void) ++errors;
			}
			if( !suppress && errors > 50 ) {
				std::cerr << "More than 50 errors emitted; suppressing further output\n";
				suppress = true;
			}
		}
		if( errors ) {
			if( suppress && errors > 50 ) {
				std::cerr << "A total of " << errors << " verification errors found during "
					<< "initialisation.\n";
			}
			out.error = grb::FAILED;
			return;
		}
	}

	// set constant multiplicant to x
	const double alpha = 2.0;

	// WARNING: ALP incurs performance loss unless compiled using the nonblocking
	//          backend
	if( mode == TEMPLATED ) {
		// flush any pending ops
		out.error = grb::wait();
		// start timing using a cold run to get the cache `hot' and get an early
		// run-time estimate
		double ttime = timer.time();
		out.error = out.error ? out.error :
			grb::set< grb::descriptors::dense >( zv, yv );
		out.error = out.error ? out.error :
			grb::eWiseMul< grb::descriptors::dense >( zv, alpha, xv, reals );
		out.error = out.error ? out.error : grb::wait();
		if( out.error != SUCCESS ) {
			std::cerr << "grb::eWiseMul returns non-SUCCESS exit code "
				<< grb::toString( out.error ) << "." << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return;
		}
		// use this to infer number of inner iterations, if requested to be computed
		ttime = timer.time() - ttime;
		if( in.rep == 0 ) {
			out.reps_used = static_cast< size_t >( 100.0 / ttime ) + 1;
			std::cout << "Auto-selected " << out.reps_used << " inner repetitions "
				<< "of approximately " << ttime << " ms. each in order to achieve around "
				<< "100 ms. of inner-loop wall-clock time.\n";
		} else {
			out.reps_used = in.rep;
		}
		// verify
		double checksum = 0;
		for( size_t i = 0; i < in.n; ++i ) {
			checksum += z[ i ];
			const double expected = alpha * x[ i ] + y[ i ];
			if( !grb::utils::equals( expected, z[ i ], 2 ) ) {
				std::cout << expected << " (expected) does not equal " << z[ i ]
					<< " (template optimised) at position " << i << ".\n";
				out.error = FAILED;
				return;
			}
		}
		std::cout << "Checksum: " << checksum << std::endl;
		out.times.preamble = timer.time();

		// benchmark ALP axpy
		timer.reset();
		for( size_t i = 0; i < out.reps_used; ++i ) {
			// zv[ i ] = alpha * xv[ i ] + yv[ i ]
			(void) grb::set< grb::descriptors::dense >( zv, yv );
			(void) grb::eWiseMul< grb::descriptors::dense >( zv, alpha, xv, reals );
			(void) grb::wait();
		}
		out.times.useful = timer.time() / static_cast< double >( out.reps_used );

		// postamble
		out.times.postamble = 0;
	}

	if( mode == LAMBDA ) {
		// flush any pending ops
		out.error = grb::wait();
		// start timing using a cold run to get the cache `hot' and get an early
		// run-time estimate
		double ltime = timer.time();
		out.error = out.error ? out.error : grb::eWiseLambda(
			[ &zv, &alpha, &xv, &yv, &reals ]( const size_t i ) {
				// zv[ i ] = alpha * xv[ i ] + yv[ i ]
				(void) grb::apply( zv[ i ], alpha, xv[ i ],
					reals.getMultiplicativeOperator() );
				(void) grb::foldl( zv[ i ], yv[ i ], reals.getAdditiveOperator() );
			},
			zv, xv, yv );
		out.error = out.error ? out.error : grb::wait();
		if( out.error != SUCCESS ) {
			std::cerr << "grb::eWiseLambda returns non-SUCCESS exit code "
				<< grb::toString( out.error ) << ".\n";
			return;
		}
		// use this to infer number of inner iterations, if requested to be computed
		ltime = timer.time() - ltime;
		if( in.rep == 0 ) {
			out.reps_used = static_cast< size_t >( 100.0 / ltime ) + 1;
			std::cout << "Auto-selected " << out.reps_used << " inner repetitions "
				<< "of approx. " << ltime << " ms. each in order to achieve around "
				<< "100 ms. of inner loop wall-clock time.\n";
		} else {
			out.reps_used = in.rep;
		}
		// do verification
		double checksum = 0;
		for( size_t i = 0; i < in.n; ++i ) {
			checksum += z[ i ];
			const double expected = alpha * x[ i ] + y[ i ];
			if( !grb::utils::equals( expected, z[ i ], 2 ) ) {
				std::cout << expected << " (expected) does not equal " << z[ i ]
					<< " (eWiseLambda) at position " << i << ".\n";
				out.error = FAILED;
				return;
			}
		}
		std::cout << "Checksum: " << checksum << std::endl;
		out.times.preamble = timer.time();
		timer.reset();
		// benchmark templated axpy
		for( size_t i = 0; i < out.reps_used; ++i ) {
			(void) grb::eWiseLambda(
				[ &zv, &alpha, &xv, &yv, &reals ]( const size_t i ) {
					(void) grb::apply( zv[ i ], alpha, xv[ i ],
						reals.getMultiplicativeOperator() );
					(void) grb::foldl( zv[ i ], yv[ i ], reals.getAdditiveOperator() );
				}, zv, xv, yv
			);
			(void) grb::wait();
		}
		out.times.useful = timer.time() / static_cast< double >( out.reps_used );

		// postamble
		out.times.postamble = 0;
	}

	if( mode == RAW ) {
		double * a = nullptr;
		int prc = posix_memalign(
			(void **)&a,
			grb::config::CACHE_LINE_SIZE::value(),
			in.n * sizeof( double )
		);
		assert( prc == 0 );
		if( prc == ENOMEM ) {
			out.error = OUTOFMEM;
			return;
		}
		if( prc != 0 ) {
			out.error = PANIC;
			return;
		}
		for( size_t i = 0; i < in.n; ++i ) {
			a[ i ] = 0;
		}

		double ctime = timer.time();
		// get cache `hot'
		bench_kernels_axpy( a, alpha, x, y, in.n );
		// use this to infer number of inner iterations, if requested to be computed
		ctime = timer.time() - ctime;
		if( in.rep == 0 ) {
			out.reps_used = static_cast< size_t >( 100.0 / ctime ) + 1;
			std::cout << "Auto-selected " << out.reps_used << " inner repetitions "
				<< "of approx. " << ctime << " ms. each in order to achieve around "
				<< "100 ms. of inner-loop wall-clock time.\n";
		} else {
			out.reps_used = in.rep;
			out.times.preamble = timer.time();
		}
		// do verification
		double checksum = 0;
		for( size_t i = 0; i < in.n; ++i ) {
			const double expected = alpha * x[ i ] + y[ i ];
			checksum += a[ i ];
			if( !grb::utils::equals( a[ i ], expected, 2 ) ) {
				std::cout << a[ i ] << " (compiler optimised) does not equal " << expected
					<< " (expected) at position " << i << ".\n";
				out.error = FAILED;
				return;
			}
		}
		std::cout << "Checksum: " << checksum << std::endl;
		out.times.preamble = timer.time();
		timer.reset();

		// benchmark raw axpy
		for( size_t k = 0; k < out.reps_used; ++k ) {
			bench_kernels_axpy( a, alpha, x, y, in.n );
		}
		out.times.useful = timer.time() / static_cast< double >( out.reps_used );

		// postamble
		timer.reset();
		free( a );
		out.times.postamble = timer.time();
	}

	// done
	out.error = SUCCESS;
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
	char * end = nullptr;
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
				<< "for number of inner experiment repetitions." << std::endl;
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
				<< "for number of outer experiment repetitions." << std::endl;
			std::cout << "Test FAILED\n" << std::endl;
			return 30;
		}
	}

	// prepare benchmarker
	grb::Benchmarker< AUTOMATIC > bench;

	// start functional test
	std::cout << "\nBenchmark label: grb::set + grb::eWiseMul (axpy, "
		<< grb::toString( grb::config::default_backend ) << ") of size " << in.n
		<< std::endl;
	out.error = SUCCESS;
	grb::RC rc = bench.exec( &(test< TEMPLATED >), in, out, 1, outer, true );
	if( rc != SUCCESS || out.error != SUCCESS ) {
		std::cerr << "Functional test exits with nonzero exit code. "
			<< "Benchmarker reports: " << grb::toString( rc )
			<< "; test reports:"  << grb::toString( out.error ) << "." << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return 40;
	}
	std::cout << "\nBenchmark label: grb::eWiseLambda (axpy, "
		<< grb::toString( grb::config::default_backend ) << ") of size " << in.n
		<< std::endl;
	rc = bench.exec( &(test< LAMBDA >), in, out, 1, outer, true );
	if( rc != SUCCESS || out.error != SUCCESS ) {
		std::cerr << "Functional test exits with nonzero exit code. "
			<< "Benchmarker reports: " << grb::toString( rc )
			<< "; test reports:"  << grb::toString( out.error ) << "." << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return 50;
	}

	std::cout << "\nBenchmark label: ";
	if( bench_kernels_parallel() ) {
		std::cout << "parallel (OpenMP) ";
	} else {
		std::cout << "sequential (C) ";
	}
	std::cout << "compiler-optimised axpy of size " << in.n << std::endl;
	rc = bench.exec( &(test< RAW >), in, out, 1, outer, true );
	if( rc != SUCCESS || out.error != SUCCESS ) {
		std::cerr << "Functional test exits with nonzero exit code. "
			<< "Benchmarker reports: " << grb::toString( rc )
			<< "; test reports:"  << grb::toString( out.error ) << "." << std::endl;
		std::cout << "Test FAILED\n" << std::endl;
		return 60;
	}

	std::cout << "\nNOTE: please check the above performance figures manually-- "
		<< "the eWiseLambda and compiler-optimised timings should approximately "
		<< "match while that of the grb::set + grb::eWiseMul should only approx. "
		<< "match when the nonblocking backend is employed\n\n";

	// done
	std::cout << "Test OK\n" << std::endl;
	return EXIT_SUCCESS;
}

