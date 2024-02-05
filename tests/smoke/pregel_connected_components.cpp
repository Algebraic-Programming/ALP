
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

#include <exception>
#include <iostream>
#include <vector>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/algorithms/pregel_connected_components.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>


using namespace grb;
using namespace algorithms;

/** Parser type */
typedef grb::utils::MatrixFileReader<
	void,
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
	void
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
	size_t iterations;
	grb::utils::TimerResults times;
	PinnedVector< size_t > pinnedVector;
};

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;

	// sanity check on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << "Error: no file name given as input.\n";
		return;
	}

	// Parse and store matrix in singleton class
	try {
		auto &data = Storage::getData().second;
		Parser parser( data_in.filename, data_in.direct );
		if( parser.m() != parser.n() ) {
			std::cerr << "Error: input matrix must be square.\n";
			return;
		}
		Storage::getData().first.first = parser.n();
		try {
			Storage::getData().first.second = parser.nz();
		} catch( ... ) {
			Storage::getData().first.second = parser.entries();
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

	// get data from storage and prepare Pregel interface
	const size_t n = Storage::getData().first.first;
	const auto &data = Storage::getData().second;
	/* Once internal issue #342 is resolved this can be re-enabled
        grb::interfaces::Pregel< void > pregel(
		n, n,
		utils::makeNonzeroIterator<
			grb::config::RowIndexType, grb::config::ColIndexType, void
		>( data.cbegin() ),
		utils::makeNonzeroIterator<
			grb::config::RowIndexType, grb::config::ColIndexType, void
		>( data.cend() ),
		PARALLEL
	);*/
        grb::interfaces::Pregel< void > pregel(
		n, n,
		utils::makeNonzeroIterator<
			grb::config::RowIndexType, grb::config::ColIndexType, void
		>( data.cbegin() ),
		utils::makeNonzeroIterator<
			grb::config::RowIndexType, grb::config::ColIndexType, void
		>( data.cend() ),
		SEQUENTIAL
	);
	{
		const size_t parser_nnz = Storage::getData().first.second;
		if( pregel.numEdges() != parser_nnz ) {
			std::cerr << "Warning: number of edges (" << pregel.numEdges() << ") does "
				<< "not equal parser nnz (" << parser_nnz << "). This could naturally "
				<< "occur if the input file employs symmetric storage, in which case only "
				<< "roughly one half of the input is stored (and visible to the parser).\n";
		}
	}

	// finish I/O, go to preamble
	out.times.io = timer.time();
	timer.reset();

	// 1. initalise connected components IDs and message buffers
        grb::Vector< size_t > cc( n );

	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;

	grb::RC rc = grb::SUCCESS;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
	        rc = grb::algorithms::pregel::ConnectedComponents< size_t >::execute(
			pregel, cc, pregel.numVertices() );
		double single_time = timer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to Pregel ConnectedAlgorithms did not succeed "
				<< "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 25;
		}
		out.times.useful = single_time;
		out.rep = static_cast< size_t >( 1000.0 / single_time ) + 1;
		if( rc == SUCCESS ) {
			if( s == 0 ) {
				std::cout << "Info: cold connected components completed within "
					<< out.iterations << " iterations. "
					<< "Time taken was " << single_time
					<< " ms. Deduced inner repetitions parameter of " << out.rep
					<< " to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();
		grb::Vector< size_t > in_msgs( n ), out_msgs( n );
		grb::Vector< size_t > out_buffer = interfaces::config::out_sparsify
			? grb::Vector< size_t >( n )
			: grb::Vector< size_t >( 0 );
		out.times.preamble += timer.time();
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			if( rc == SUCCESS ) {
				grb::set< grb::descriptors::use_index >( cc, 0 );
				rc = pregel.template execute<
					grb::operators::max< size_t >,
					grb::identities::negative_infinity
				>(
					&(grb::algorithms::pregel::ConnectedComponents< size_t >::program),
					cc,
					grb::algorithms::pregel::ConnectedComponents< size_t >::Data(),
					in_msgs, out_msgs,
					out.iterations,
					out_buffer
				);
			}
		}
		time_taken = timer.time();
		if( rc == SUCCESS ) {
			out.times.useful = time_taken / static_cast< double >( out.rep );
		}
		sleep( 1 );
#ifndef NDEBUG
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for a " << out.rep
				<< " Connected Components calls (hot start): " << out.times.useful << ". "
				<< "Error code is " << out.error_code << std::endl;
		}
#endif
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
	out.pinnedVector = PinnedVector< size_t >( cc, SEQUENTIAL );

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 7 ) {
		std::cout << "Usage: " << argv[ 0 ] << " "
			<< "<dataset> <direct/indirect> "
			<< "(inner iterations) (outer iterations) (verification <truth-file>)\n";
		std::cout << "<dataset> and <direct/indirect> are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::inner() << ". "
			<< "If set to zero, the program will select a number of iterations "
			<< "approximately required to take at least one second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::outer()
			<< ". This value must be strictly larger than 0.\n";
		std::cout << "(verification <truth-file>) is optional. "
			<< "The <truth-file> must point to a pre-computed solution that the "
			<< "computed solution will be verified against." << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

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
			std::cerr << "Could not parse argument \"" << argv[ 2 ] << "\"; expected "
				<< "\"direct\" or \"indirect\"\n";
			return 10;
		}
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ]
				<< " for number of inner experiment repetitions." << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ]
				<< " for number of outer experiment repetitions." << std::endl;
			return 30;
		}
	}

	// check for verification of the output
	bool verification = false;
	char truth_filename[ 1024 ];
	if( argc >= 6 ) {
		if( strncmp( argv[ 5 ], "verification", 12 ) == 0 ) {
			verification = true;
			if( argc >= 7 ) {
				(void) strncpy( truth_filename, argv[ 6 ], 1023 );
				truth_filename[ 1023 ] = '\0';
			} else {
				std::cerr << "The verification file was not provided as an argument."
					<< std::endl;
				return 40;
			}
		} else {
			std::cerr << "Could not parse argument \"" << argv[ 5 ] << "\", "
				<< "the optional \"verification\" argument was expected." << std::endl;
			return 50;
		}
	}

	std::cout << "Executable called with parameters " << in.filename << ", "
		<< "inner repititions = " << in.rep << ", "
		<< "and outer reptitions = " << outer << std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// run I/O program
	{
		bool success;
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &ioProgram, in, success, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec(I/O) returns with non-SUCCESS error code \""
				<< grb::toString( rc ) << "\"\n";
			return 60;
		}
		if( !success ) {
			std::cerr << "I/O program caught an exception\n";
			return 70;
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
				<< grb::toString( rc ) << std::endl;
			return 80;
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
		return 90;
	} else if( out.error_code == 0 ) {
		std::cout << "Benchmark completed successfully and took "
			<< out.iterations << " iterations to converge.\n";
			//<< "with residual " << out.residual << ".\n";
	}

	const size_t n = out.pinnedVector.size();
	std::cout << "Error code is " << out.error_code << ".\n";
	std::cout << "Size of pr is " << n << ".\n";
	if( out.error_code == 0 && n > 0 ) {
		std::cout << "First 10 nonzeroes of pr are: (\n";
		for( size_t k = 0; k < out.pinnedVector.nonzeroes() && k < 10; ++k ) {
			const auto &index = out.pinnedVector.getNonzeroIndex( k );
			const auto &value = out.pinnedVector.getNonzeroValue( k );
			std::cout << "\t " << index << ", " << value << "\n";
		}
		std::cout << ")" << std::endl;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		if( verification ) {
			std::cerr << "Verification is not yet implemented\n";
			out.error_code = 255;
			if( out.error_code == 0 ) {
				std::cout << "Verification OK\n";
				std::cout << "Test OK\n";
			} else {
				std::cerr << std::flush;
				std::cout << "Verification FAILED\n";
				std::cout << "Test FAILED\n";
			}
		} else {
			std::cout << "Test OK\n";
		}
	}
	std::cout << std::endl;

	// done
	if( out.error_code == 0 ) {
		return 0;
	} else {
		return (100 + out.error_code);
	}
}

