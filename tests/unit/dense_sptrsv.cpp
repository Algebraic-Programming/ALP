
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

#include <map>
#include <set>
#include <cmath>
#include <array>
#include <cstdio>
#include <string>
#include <vector>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <inttypes.h>

#include "graphblas/algorithms/matrix_factory.hpp"
#include "graphblas/utils/timer.hpp"

#include "graphblas.hpp"


using namespace grb;

struct input {
	size_t n;
	size_t rep;
};

struct output {
	enum grb::RC error_code;
	grb::utils::TimerResults times;
};

template< typename T, class Semiring >
bool checkVectorDenseAndEqualTo(
	const grb::Vector< T > &x, const T val, struct output &out,
	const Semiring &semiring
) {
	size_t count = 0;
	bool error = false;
	std::set< size_t > localSet;
	const size_t n = grb::size( x );
	for( const auto &pair : x ) {
		(void) ++count;
		localSet.insert( pair.first );
		if( pair.second != val ) {
			std::cout << "unexpected entry ( " << pair.first << ", " << pair.second
				<< " ); expected value " << val << "\n";
			error = true;
		}
	}
	if( grb::collectives<>::allreduce( count, semiring.getAdditiveOperator() )
		!= grb::SUCCESS
	) {
		std::cerr << "error during computing checksum (I)\n";
		out.error_code = grb::PANIC;
		return false;
	}
	if( count != n ) {
		std::cout << "expected " << n << " nonzeroes, got " << count << "\n";
		error = true;
	}
	count = localSet.size();
	if( grb::collectives<>::allreduce( count, semiring.getAdditiveOperator() )
		!= grb::SUCCESS
	) {
		std::cerr << "error during computing checksum (II)\n";
		out.error_code = grb::PANIC;
		return false;
	}
	if( count != n ) {
		std::cout << "expected " << n << " indices, got " << count << "\n";
		error = true;
	}
	if( out.error_code == grb::SUCCESS ) {
		if( error ) {
			out.error_code = grb::FAILED;
			return false;
		}
	}
	return true;
}

// main label propagation algorithm
void grbProgram( const struct input &data_in, struct output &out ) {
	out.error_code = grb::PANIC;
	grb::utils::Timer timer;

	const size_t s = grb::spmd<>::pid();
#ifndef NDEBUG
	assert( s < grb::spmd<>::nprocs() );
#else
	(void) s;
#endif

	// get input n and test case
	const size_t n = data_in.n;

	// setup
	grb::Vector< double > x( n );
	grb::Matrix< double > A = grb::algorithms::matrices< double >::identity( n );
	Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> semiring;
	grb::operators::divide< double > division;
	grb::operators::subtract< double > subtraction;

	out.error_code = grb::set( x, 3.14 );
	if( out.error_code ) {
		std::cerr << "Error during test initialisation\n";
		return;
	}
	if( !checkVectorDenseAndEqualTo( x, 3.14, out, semiring ) ) {
		std::cerr << "NOTE: the aforementioned errors occurred during test "
			<< "initialisation\n";
		return;
	}
	out.times.preamble = timer.time();

	if( s == 0 ) {
		std::cout << "Test 1: ";
	}
	timer.reset();
	out.error_code = grb::sptrsv( x, A, false, semiring, subtraction, division );
	if( out.error_code ) {
		std::cerr << "test returned error\n";
	}
	if( !checkVectorDenseAndEqualTo( x, 3.14, out, semiring ) ) { return; }
	out.times.postamble += timer.time();
	timer.reset();
	for( size_t rep = 0; rep < data_in.rep; ++rep ) {
		(void) grb::sptrsv( x, A, false, semiring, subtraction, division );
	}
	out.times.useful += timer.time();

	assert( out.error_code == grb::SUCCESS );
	if( s == 0 ) {
		std::cout << "\b\b 2: ";
	}
	timer.reset();
	out.error_code = grb::set< grb::descriptors::dense >( x, 2.14 );
	if( out.error_code ) {
		std::cerr << "Error during test initialisation\n";
		return;
	}
	if( !checkVectorDenseAndEqualTo( x, 2.14, out, semiring ) ) {
		std::cerr << "NOTE: aforementioned errors occurred during initialisation\n";
		return;
	}
	out.times.preamble += timer.time();
	timer.reset();
	out.error_code = grb::sptrsv( x, A, true, semiring, subtraction, division );
	if( out.error_code ) {
		std::cerr << "test returned error\n";
	}
	if( !checkVectorDenseAndEqualTo( x, 2.14, out, semiring ) ) { return; }
	out.times.postamble += timer.time();
	timer.reset();
	for( size_t rep = 0; rep < data_in.rep; ++rep ) {
		(void) grb::sptrsv( x, A, true, semiring, subtraction, division );
	}
	out.times.useful += timer.time();

	assert( out.error_code == grb::SUCCESS );
	if( s == 0 ) {
		std::cout << "\b\b 3: ";
	}
	timer.reset();
	out.error_code = grb::set< grb::descriptors::dense >( x, 3.14 );
	if( out.error_code ) {
		std::cerr << "Error during test initialisation\n";
		return;
	}
	if( !checkVectorDenseAndEqualTo( x, 3.14, out, semiring ) ) {
		std::cerr << "NOTE: aforementioned errors occurred during initialisation\n";
		return;
	}
	out.times.preamble += timer.time();
	timer.reset();
	out.error_code = grb::sptrsv< grb::descriptors::dense >( x, A, false, semiring,
		subtraction, division );
	if( out.error_code ) {
		std::cerr << "test returned error\n";
	}
	if( !checkVectorDenseAndEqualTo( x, 3.14, out, semiring ) ) { return; }
	out.times.postamble += timer.time();
	timer.reset();
	for( size_t rep = 0; rep < data_in.rep; ++rep ) {
		(void) grb::sptrsv< grb::descriptors::dense >( x, A, false, semiring,
			subtraction, division );
	}
	out.times.useful += timer.time();

	assert( out.error_code == grb::SUCCESS );
	if( s == 0 ) {
		std::cout << "\b\b 4: ";
	}
	timer.reset();
	out.error_code = grb::set< grb::descriptors::dense >( x, 2.14 );
	if( out.error_code ) {
		std::cerr << "Error during test initialisation\n";
		return;
	}
	if( !checkVectorDenseAndEqualTo( x, 2.14, out, semiring ) ) {
		std::cerr << "NOTE: aforementioned errors occurred during initialisation\n";
		return;
	}
	out.times.preamble += timer.time();
	timer.reset();
	out.error_code = grb::sptrsv< grb::descriptors::dense >( x, A, true, semiring,
		subtraction, division );
	if( out.error_code ) {
		std::cerr << "test returned error\n";
	}
	if( !checkVectorDenseAndEqualTo( x, 2.14, out, semiring ) ) { return; }
	out.times.postamble += timer.time();
	timer.reset();
	for( size_t rep = 0; rep < data_in.rep; ++rep ) {
		(void) grb::sptrsv< grb::descriptors::dense >( x, A, true, semiring,
			subtraction, division );
	}
	out.times.useful += timer.time();

	std::cout << "OK\n";
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 5 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <problem size> "
			<< "(inner repititions) (outer repititions)" << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	in.n = atoi( argv[ 1 ] );
	in.rep = grb::config::BENCHMARKING::inner();
	size_t outer = grb::config::BENCHMARKING::outer();
	char * end = NULL;
	if( argc >= 3 ) {
		in.rep = strtoumax( argv[ 2 ], &end, 10 );
		if( argv[ 2 ] == end ) {
			std::cerr << "Could not parse argument for number of inner "
				<< "repetitions." << std::endl;
			return 25;
		}
	}
	if( argc >= 4 ) {
		outer = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument for number of outer "
				<< "reptitions." << std::endl;
			return 30;
		}
	}

	std::cout << "Executable called with parameters: problem size " << in.n
		<< ", inner = " << in.rep << ", outer = " << outer << "." << std::endl;

	// the output struct
	struct output out;

	// run the program one time to infer number of inner repititions
	if( in.rep == 0 ) {
		in.rep = 1;
		grb::Launcher< AUTOMATIC > launcher;
		const enum grb::RC rc = launcher.exec( &grbProgram, in, out, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< grb::toString( rc ) << std::endl;
			return 40;
		}
		// set guesstimate for inner repititions: a single experiment should take at least a second
		in.rep = static_cast< double >( 1000.0 / out.times.useful ) + 1;
		std::cout << "Auto-selected number of inner repetitions is "
			<< in.rep << " (at an estimated time of "
			<< out.times.useful << " ms. of useful work per benchmark).\n";
	}

	// start benchmarks
	grb::Benchmarker< AUTOMATIC > benchmarker;
	const enum grb::RC rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code "
			<< grb::toString( rc ) << std::endl;
		return 50;
	}

	// done
	if( out.error_code != SUCCESS ) {
		std::cout << "Test FAILED\n" << std::endl;
		std::cerr << std::flush;
		return out.error_code;
	}
	std::cout << "Test OK\n" << std::endl;
	return 0;
}

