
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

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <inttypes.h>

#include "graphblas/utils/timer.hpp"

#include <graphblas.hpp>


using namespace grb;

struct input {
	size_t n;
	size_t test;
	size_t rep;
};

struct output {
	enum grb::RC error_code;
	grb::utils::TimerResults times;
};

static grb::RC setupSparseMatrix( grb::Matrix< double > &mx, const size_t n ) {
	// number of elements
	const size_t elems = n * 5;
	// reserve space in mx
	grb::RC rc = grb::resize( mx, elems );
	if( rc != SUCCESS ) {
		return rc;
	}

	// generate random data
	size_t *I, *J, *mxValues;
	I = new size_t[ elems ];
	J = new size_t[ elems ];
	mxValues = new size_t[ elems ];
	const size_t step = ( n - 1 ) / 5;
	for( size_t e = 0; e < elems; ) {
		for( size_t i = 0; e < elems && i < 5; ++i, ++e ) {
			I[ e ] = e / 5;
			J[ e ] = ( e / 5 + i * step ) % n;
			mxValues[ e ] = rand() % 1000;
			assert( I[ e ] < n );
		}
	}

	// load into GraphBLAS
	rc = grb::buildMatrixUnique( mx, &( I[ 0 ] ), &( J[ 0 ] ), mxValues, elems,
		SEQUENTIAL );
	if( rc == SUCCESS && elems != nnz( mx ) ) {
		rc = PANIC;
	}

	// free random data
	delete [] I;
	delete [] J;
	delete [] mxValues;

	// done
	return rc;
}

// main benchmark
void grbProgram( const struct input &data_in, struct output &out ) {

	grb::utils::Timer timer;

	const size_t s = spmd<>::pid();
#ifndef NDEBUG
	assert( s < spmd<>::nprocs() );
#else
	(void) s;
#endif

	// get input n and test case
	const size_t n = data_in.n;
	const size_t test = data_in.test;
	out.error_code = SUCCESS;

	// setup
	grb::Vector< double > vx( n ), vy( n );
	grb::Matrix< double > mx( n, n );
	Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	const Descriptor descr = descriptors::dense;

	switch( test ) {

		// Ax
		case 1: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vx, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix( mx, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = mxv< descr >( vy, mx, vx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		// A^Tx
		case 2: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vx, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix( mx, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = mxv< descr | descriptors::transpose_matrix >(
					vy, mx, vx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		// xA
		case 3: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vx, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix( mx, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = vxm< descr >( vy, vx, mx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		// xA^T
		case 4: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::set( vx, 1 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix( mx, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = vxm< descr | descriptors::transpose_matrix >(
					vy, vx, mx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// done
			out.times.postamble = 0;
			break;
		}

		default:
			std::cerr << "Unknown test case " << test << std::endl;
			break;
	}
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 5 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <problem size> <test case> "
			<< "(inner repititions) (outer repititions)" << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	in.n = atoi( argv[ 1 ] );
	in.test = atoi( argv[ 2 ] );
	in.rep = grb::config::BENCHMARKING::inner();
	size_t outer = grb::config::BENCHMARKING::outer();
	char * end = nullptr;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument for number of inner repetitions."
				<< std::endl;
			return 25;
		}
	}
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument for number of outer repetitions."
				<< std::endl;
			return 25;
		}
	}

	std::cout << "Executable called with parameters: problem size " << in.n
		<< " test case ";
	switch( in.test ) {
		case 1:
			std::cout << "Ax";
			break;
		case 2:
			std::cout << "A^Tx";
			break;
		case 3:
			std::cout << "xA";
			break;
		case 4:
			std::cout << "xA^T";
			break;
		default:
			std::cout << " UNRECOGNISED TEST CASE, ABORTING.\nTest FAILED.\n"
				<< std::endl;
			return 30;
	}
	std::cout << ", inner = " << in.rep << ", outer = " << outer << "."
		<< std::endl;

	// the output struct
	struct output out;

	// run the program one time to infer number of inner repititions
	if( in.rep == 0 ) {
		in.rep = 1;
		grb::Launcher< AUTOMATIC > launcher;
		const enum grb::RC rc = launcher.exec( &grbProgram, in, out, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< grb::toString(rc) << std::endl;
			return 40;
		}
		// set guesstimate for inner repititions: a single experiment should take at least a second
		in.rep = static_cast< double >( 1000.0 / out.times.useful ) + 1;
		std::cout << "Auto-selected number of inner repetitions is " << in.rep
			<< " (at an estimated time of " << out.times.useful
			<< " ms. of useful work per benchmark).\n";
	}

	// start benchmarks
	grb::Benchmarker< AUTOMATIC > benchmarker;
	const enum grb::RC rc = benchmarker.exec( &grbProgram, in, out, 1, outer,
		true );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code "
			<< grb::toString(rc) << std::endl;
		return 50;
	}

	// done
	if( out.error_code != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
		return out.error_code;
	}
	std::cout << "Test OK\n" << std::endl;
	return 0;
}

