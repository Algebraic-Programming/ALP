
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

/*
 * @authors Anders Hansson, Aristeidis Mastoras, A. N. Yzelman
 * @date May, 2022
 */

#include <exception>
#include <iostream>
#include <vector>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>


using namespace grb;

/** Parser type */
typedef grb::utils::MatrixFileReader<
	double,
	std::conditional<
		(sizeof(grb::config::RowIndexType) > sizeof(grb::config::ColIndexType)),
		grb::config::RowIndexType,
		grb::config::ColIndexType
	>::type
> Parser;

/** Nonzero type */
typedef internal::NonzeroStorage<
	grb::config::RowIndexType,
	grb::config::ColIndexType,
	double
> NonzeroT;

/** In-memory storage type */
typedef grb::utils::Singleton<
	std::pair<
		// stores n and nz (according to parser)
		std::pair< size_t, size_t >,
		// stores the actual nonzeroes
		std::vector< NonzeroT >
	>
> Storage;

struct input {
	char filename[ 1024 ];
	bool direct;
	size_t rep;
};

struct output {
	int error_code;
	size_t rep;
	grb::utils::TimerResults times;
	PinnedVector< double > pinnedVector;
};

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;

	// sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << "Error: no input file given\n";
		return;
	}
	// Parse and store matrix in singleton class
	auto &data = Storage::getData().second;
	try {
		Parser parser( data_in.filename, data_in.direct );
		Storage::getData().first.first  = parser.m();
		Storage::getData().first.second = parser.n();
		size_t parser_nz;
		try {
			parser_nz = parser.nz();
		} catch( ... ) {
			parser_nz = parser.entries();
		}
		/* Once internal issue #342 is resolved this can be re-enabled
		for(
			auto it = parser.begin( PARALLEL );
			it != parser.end( PARALLEL );
			++it
		) {
			data.push_back( *it );
		}*/
		for(
			auto it = parser.begin( SEQUENTIAL );
			it != parser.end( SEQUENTIAL );
			++it
		) {
			data.push_back( NonzeroT( *it ) );
		}
		if( parser_nz != data.size() ) {
			std::cerr << "Warning: stored nnz (" << data.size() << ") does not equal "
				<< "parser nnz (" << parser_nz << "). " << "This could naturally occur "
				<< "if the input matrix file employs symmetric storage; in that case, the "
				<< "number of entries is roughly half of the number of nonzeroes.\n";
		}
	} catch( std::exception &e ) {
		std::cerr << "I/O program failed: " << e.what() << "\n";
		return;
	}
	success = true;
}

void grbProgram( const struct input &data_in, struct output &out ) {
	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// assume successful run
	out.error_code = 0;

	// load into GraphBLAS
	const size_t m = Storage::getData().first.first;
	const size_t n = Storage::getData().first.second;
	Matrix< double > A( m, n );
	{
		const auto &data = Storage::getData().second;
		const RC rc = buildMatrixUnique(
			A,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cend() ),
			SEQUENTIAL
		);
		/* Once internal issue #342 is resolved this can be re-enabled
		const RC rc = buildMatrixUnique(
			A,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, double
			>( data.cend() ),
			PARALLEL
		);*/
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed "
				<< "(" << toString( rc ) << ")." << std::endl;
			return;
		}
	}
	out.times.io = timer.time();
	timer.reset();

	RC rc = SUCCESS;

	Vector< double > y( m ), x( n );

	const Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	rc = rc ? rc : set( x, static_cast< double >( 1 ) );
	assert( rc == SUCCESS );

	// that was the preamble
	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;

	// time a single call
	{
		grb::utils::Timer subtimer;
		subtimer.reset();

		rc = rc ? rc : set( y, static_cast< double >( 0 ) );
		assert( rc == SUCCESS );

		rc = rc ? rc : mxv( y, A, x, ring );
		assert( rc == SUCCESS );

		double single_time = subtimer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to mxv did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 25;
		}
		out.times.useful = single_time;
		const size_t recommended_inner_repetitions =
			static_cast< size_t >( 100.0 / single_time ) + 1;
		if( rc == SUCCESS && out.rep == 0 ) {
			if( s == 0 ) {
				std::cout << "Info: cold mxv completed"
					<< ". Time taken was " << single_time << " ms. "
					<< "Deduced inner repetitions parameter of " << out.rep << " "
					<< "to take 100 ms. or more per inner benchmark.\n";
				out.rep = recommended_inner_repetitions;
			}
			return;
		}
	}

	// now do benchmark
	double time_taken;
	timer.reset();
	for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
#ifndef NDEBUG
		rc = rc ? rc : set( y, static_cast< double >( 0 ) );
		assert( rc == SUCCESS );
		rc = rc ? rc : mxv( y, A, x, ring );
		assert( rc == SUCCESS );
#else
		(void) set( y, static_cast< double >( 0 ) );
		(void) mxv( y, A, x, ring );
#endif
	}
	time_taken = timer.time();
	if( rc == SUCCESS ) {
		out.times.useful = time_taken / static_cast< double >( out.rep );
	}
	// print timing at root process
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "Time taken for a " << out.rep << " "
			<< "Mxv calls (hot start): " << out.times.useful << ". "
			<< "Error code is " << out.error_code << std::endl;
	}

	// start postamble
	timer.reset();

	// set error code
	if( rc == FAILED ) {
		out.error_code = 30;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}

	// output
	out.pinnedVector = PinnedVector< double >( y, SEQUENTIAL );

	// finish timing
	time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 7 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <dataset> <direct/indirect> "
			<< "(inner iterations) (outer iterations) (verification <truth-file>)\n";
		std::cout << "<dataset> and <direct/indirect> are mandatory arguments.\n";
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
#ifndef NDEBUG
	std::cerr << "Warning: this benchmark utility was **not** compiled with the "
		<< "NDEBUG macro defined(!)\n";
#endif

	// the input struct
	struct input in;

	// get file name
	(void) strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';

	// get direct or indirect addressing
	if( strncmp( argv[ 2 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		if( strncmp( argv[ 2 ], "indirect", 8 ) == 0 ) {
			in.direct = false;
		} else {
			std::cerr << "Error: could not parse 2nd argument \"" << argv[ 2 ] << "\", "
				<< "expected \"direct\" or \"indirect\"\n";
			return 10;
		}
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 2 ] << " "
				<< "for number of inner experiment repititions." << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			return 30;
		}
	}

	std::cout << "Executable called with parameters "
		<< "file name = " << in.filename << ", "
		<< "inner repititions = " << in.rep << ", and "
		<< "outer reptitions = " << outer
		<< std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// launch I/O
	{
		bool success;
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &ioProgram, in, success, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec(I/O) returns with non-SUCCESS error code \""
				<< grb::toString( rc ) << "\"\n";
			return 40;
		}
		if( !success ) {
			std::cerr << "I/O program caught an exception\n";
			return 50;
		}
	}

	// launch estimator (if requested)
	if( in.rep == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			in.rep = out.rep;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< (int)rc << std::endl;
			return 60;
		}
	}

	// launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code "
			<< grb::toString( rc ) << std::endl;
		return 70;
	}

	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Size of x is " << out.pinnedVector.size() << ".\n";
	if( out.error_code == 0 && out.pinnedVector.nonzeroes() > 0 ) {
		std::cerr << std::fixed;
		std::cerr << "Output vector: (";
		for( size_t k = 0; k < out.pinnedVector.nonzeroes(); ++k ) {
			const auto &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
			std::cerr << nonzeroValue << " ";
		}
		std::cerr << ")" << std::endl;
		std::cerr << std::defaultfloat;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cerr << "Test FAILED\n";
	} else {
		std::cout << "Test OK\n";
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}

