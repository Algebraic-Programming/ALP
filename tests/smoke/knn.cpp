
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

#include <inttypes.h>

#include <graphblas/algorithms/knn.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>


using namespace grb;
using namespace algorithms;

struct input {
	char filename[ 1024 ];
	bool direct;
	size_t k;
};

struct output {
	RC error_code;
	PinnedVector< bool > neighbourhood;
	grb::utils::TimerResults times;
	size_t rep;
};

void grbProgram( const struct input &data_in, struct output &out ) {
	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	grb::utils::Timer timer;
	timer.reset();

	// parse file locally
	grb::utils::MatrixFileReader< void, unsigned int > reader(
		data_in.filename, data_in.direct
	);

	// retrieve number of vertices
	const size_t n = reader.n();
	if( n != reader.m() ) {
		std::cout << "The matrix loaded is not square\n";
		out.error_code = ILLEGAL;
		return;
	}

	// create output
	Vector< bool > neighbourhood( n );

	// create buffers
	Vector< bool > buf1( n );

	out.times.preamble = timer.time();
	timer.reset();

	// handle trivial case
	if( n == 0 ) {
		out.neighbourhood = PinnedVector< bool >( neighbourhood, SEQUENTIAL );
		out.error_code = SUCCESS;
		return;
	}

	// set source to approx. middle vertex
	const size_t source = n / 2;
	std::cout << s << ": starting " << data_in.k << "-hop from source vertex "
		<< source << "\n";

	// handle trivial case
	if( data_in.k == 0 ) {
		out.error_code = setElement( neighbourhood, true, source );
		out.neighbourhood = PinnedVector< bool >( neighbourhood, SEQUENTIAL );
		return;
	}
	// assume successful run
	out.error_code = SUCCESS;

	// load parsed file into GraphBLAS
	Matrix< void > A( n, n );
	RC rc = buildMatrixUnique( A,
		reader.begin( PARALLEL ), reader.end( PARALLEL ),
		PARALLEL
	);
	if( rc != SUCCESS ) {
		out.error_code = rc;
		return;
	}
	out.times.io = timer.time();
	timer.reset();

	// time the knn computation
	double time_taken;
#ifdef _DEBUG
	std::cout << s << ": starting knn with a " << grb::nrows( A ) << " by "
		<< grb::ncols( A ) << " matrix holding " << grb::nnz( A ) << " nonzeroes.\n";
#endif
	rc = knn< descriptors::no_operation >(
		neighbourhood, A, source, data_in.k,
		buf1
	);
	time_taken = timer.time();
	out.times.useful = time_taken;
	out.rep = static_cast< size_t >( 100.0 / time_taken ) + 1;
	timer.reset();

	// sanity check
	if( rc != SUCCESS ) {
		out.error_code = PANIC;
		return;
	}

#ifdef _DEBUG
	for( size_t k = 0; k < spmd<>::nprocs(); ++k ) {
		if( k == s ) {
			auto it = neighbourhood.cbegin();
			for( ; it != neighbourhood.cend(); ++it ) {
				if( ( *it ).second ) {
					std::cout << s << ": " << ( *it ).first << "\n";
				}
			}
		}
		spmd<>::sync();
	}
#endif

	// pin output
	out.neighbourhood = PinnedVector< bool >( neighbourhood, SEQUENTIAL );

	out.times.postamble = timer.time();

	// print test output at root process
	if( s == 0 ) {
#ifdef _DEBUG
		size_t count = 0;
		std::cout << "First 10 neighbours:\n";
		for( size_t k = 0; count < 10 && k < out.neighbourhood.nonzeroes(); ++k ) {
			const auto value = out.neighbourhood.getNonzeroValue( k, true );
			const auto index = out.neighbourhood.getNonzeroIndex( k );
			if( value ) {
				std::cout << index << "\n";
				++count;
			}
		}
#endif
	}

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 4 || argc > 6 ) {
		std::cout << "Usage: " << argv[ 0 ] << " "
			<< "<k> <dataset> <direct/indirect> "
			<< "(inner iterations) (outer iterations)\n";
		std::cout << "<k>, <dataset>, and <direct/indirect> "
			<< "are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::inner() << ". "
			<< "If set to zero, the program will select a number of iterations "
			<< "approximately required to take at least one second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::outer() << ". "
			<< "This value must be strictly larger than 0." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get k
	in.k = atoi( argv[ 1 ] );

	// get file name
	(void)strncpy( in.filename, argv[ 2 ], 1023 );
	in.filename[ 1023 ] = '\0';

	// get direct or indirect
	if( strncmp( argv[ 3 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	size_t inner = grb::config::BENCHMARKING::inner();
	char * end = NULL;
	if( argc >= 5 ) {
		inner = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ]
				<< " for number of inner experiment repititions." << std::endl;
			return 2;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 6 ) {
		outer = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 5 ]
				<< " for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}

	std::cout << "Executable called with parameters k = " << in.k << ", "
		<< in.filename << ", " << "direct = " << in.direct << ", "
		<< "inner repetitions = " << inner << ", "
		<< "outer repetitions = " << outer << std::endl;

	// the output struct
	struct output out;

	// launch estimater (if requested)
	grb::RC rc = SUCCESS;
	if( inner == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			inner = out.rep;
			std::cout << "Auto-selected " << inner << " repetitions to reach approx. "
				<< "1 second run-time." << std::endl;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< (int)rc << std::endl;
			return 6;
		}
	}

	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, inner, outer, true );
	}

	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code "
			<< (int)rc << std::endl;
		return 50;
	}

	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Output vector size is " << out.neighbourhood.size() << ".\n";
	const size_t count = out.neighbourhood.nonzeroes();
	std::cout << "Neighbourhood size is " << count << " "
		<< "(out of " << out.neighbourhood.size() << ").\n";
#if defined PRINT_FIRST_TEN || defined _DEBUG
	const size_t firstTen = std::min( count, static_cast< size_t >(10) );
	std::cout << "First " << firstTen << " neighbours:\n";
	for( size_t k = 0; k < firstTen; ++k ) {
		const auto index = out.neighbourhood.getNonzeroIndex( k );
		const auto value = out.neighbourhood.getNonzeroValue( k, true );
		if( value ) {
			std::cout << index << "\n";
		}
	}
#endif

	// done
	if( out.error_code != SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
		return 255;
	}
	std::cout << "Test OK\n" << std::endl;
	return 0;
}

