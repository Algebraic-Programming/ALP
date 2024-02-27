
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

/** In-memory storage type -- left matrix */
typedef grb::utils::Singleton<
	std::pair<
		// stores n and nz (according to parser)
		std::pair< size_t, size_t >,
		// stores the actual nonzeroes
		std::vector< NonzeroT >
	>, 0
> StorageL;

/** In-memory storage type -- right matrix */
typedef grb::utils::Singleton<
	std::pair<
		// stores n and nz (according to parser)
		std::pair< size_t, size_t >,
		// stores the actual nonzeroes
		std::vector< NonzeroT >
	>, 1
> StorageR;

struct input {
	char filenameL[ 1024 ];
	char filenameR[ 1024 ];
	bool direct;
	size_t rep;
};

struct output {
	int error_code;
	size_t rep;
	grb::utils::TimerResults times;
	PinnedVector< double > pinnedVector;
	size_t result_nnz;
};

void ioProgram( const struct input &data_in, bool &success ) {
	success = false;

	// sanity checks on input
	if( data_in.filenameL[ 0 ] == '\0' ) {
		std::cerr << "Error: no file name given as input for left matrix.\n";
		return;
	} else if( data_in.filenameR[ 0 ] == '\n' ) {
		std::cerr << "Error: no file name given as input for right matrix.\n";
		return;
	}

	// Parse and store matrix in singleton class
	try {
		Parser parserL( data_in.filenameL, data_in.direct );
		Parser parserR( data_in.filenameR, data_in.direct );
		if( parserL.n() != parserR.m() ) {
			std::cerr << "Error: matrix files do not match.\n";
			return;
		}
		StorageL::getData().first.first  = parserL.m();
		StorageL::getData().first.second = parserL.n();
		StorageR::getData().first.first  = parserR.m();
		StorageR::getData().first.second = parserR.n();
		{
			auto &data = StorageL::getData().second;
			/* Once internal issue #342 is resolved this can be re-enabled
			for(
				auto it = parserL.begin( PARALLEL );
				it != parserL.end( PARALLEL );
				++it
			) {
				data.push_back( *it );
			}*/
			for(
				auto it = parserL.begin( SEQUENTIAL );
				it != parserL.end( SEQUENTIAL );
				++it
			) {
				data.push_back( NonzeroT( *it ) );
			}
		}
		{
			auto &data = StorageR::getData().second;
			/* Once internal issue #342 is resolved this can be re-enabled
			for(
				auto it = parserR.begin( PARALLEL );
				it != parserR.end( PARALLEL );
				++it
			) {
				data.push_back( *it );
			}*/
			for(
				auto it = parserR.begin( SEQUENTIAL );
				it != parserR.end( SEQUENTIAL );
				++it
			) {
				data.push_back( NonzeroT( *it ) );
			}
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

	const size_t l = StorageL::getData().first.first;
	const size_t m = StorageL::getData().first.second;
	const size_t n = StorageR::getData().first.second;
	assert( m == StorageR::getData().first.first );

	out.times.io = timer.time();
	timer.reset();

	// load into GraphBLAS
	Matrix< double > A( l, m ), B( m, n );
	{
		const auto &data = StorageL::getData().second;
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
			std::cerr << "Failure: call to buildMatrixUnique did not succeed for the "
				<< "left-hand matrix " << "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 10;
			return;
		}
	}
	{
		const auto &data = StorageR::getData().second;
		const RC rc = buildMatrixUnique(
			B,
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
			std::cerr << "Failure: call to buildMatrixUnique did not succeed for the "
				<< "right-hand matrix " << "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 20;
			return;
		}
	}

	// check number of nonzeroes
	{
		// note that this check is a lighter form of the usual one (which compares
		// versus the parser nnz-- this one compares versus what was cached in
		// memory).
		const size_t global_nnzL = nnz( A );
		const size_t global_nnzR = nnz( B );
		const size_t storage_nnzL = StorageL::getData().second.size();
		const size_t storage_nnzR = StorageR::getData().second.size();
		if( global_nnzL != storage_nnzL ) {
			std::cerr << "Error: left matrix global nnz (" << global_nnzL << ") "
				<< "does not equal I/O program nnz (" << storage_nnzL << ")." << std::endl;
			out.error_code = 30;
			return;
		} else if( global_nnzR != storage_nnzR ) {
			std::cerr << "Error: right matrix global nnz (" << global_nnzR << ") "
				<< "does not equal I/O program nnz (" << storage_nnzR << ")." << std::endl;
			out.error_code = 40;
			return;
		}
	}

	RC rc = SUCCESS;

	// test default SpMSpM run
	const Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;

	// time a single call
	{
		Matrix< double > C( l, n );

		grb::utils::Timer subtimer;
		subtimer.reset();
		rc = rc ? rc : grb::mxm( C, A, B, ring, RESIZE );
		assert( rc == SUCCESS );
		rc = rc ? rc : grb::mxm( C, A, B, ring );
		assert( rc == SUCCESS );
		double single_time = subtimer.time();

		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to mxm did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 50;
			return;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 60;
			return;
		}
		out.times.useful = single_time;
		const size_t deduced_inner_reps =
			static_cast< size_t >( 100.0 / single_time ) + 1;
		if( rc == SUCCESS && out.rep == 0 ) {
			if( s == 0 ) {
				std::cout << "Info: cold mxm completed"
					<< ". Time taken was " << single_time << " ms. "
					<< "Deduced inner repetitions parameter of " << out.rep << " "
					<< "to take 1 second or more per inner benchmark.\n";
				out.rep = deduced_inner_reps;
			}
			return;
		}
	}

	if( out.rep > 1 ) {
		std::cerr << "Error: more than 1 inner repetitions are not supported due to "
			<< "having to time the symbolic phase while not timing the initial matrix "
			<< "allocation cost\n";
		out.error_code = 70;
		return;
	}

	// allocate output for benchmark
	Matrix< double > C( l, n );

	// that was the preamble
	out.times.preamble = timer.time();

	// do benchmark
	double time_taken;
	timer.reset();

#ifndef NDEBUG
	rc = rc ? rc : grb::mxm( C, A, B, ring, RESIZE );
	assert( rc == SUCCESS );
	rc = rc ? rc : grb::mxm( C, A, B, ring );
	assert( rc == SUCCESS );
#else
	(void) grb::mxm( C, A, B, ring, RESIZE );
	(void) grb::mxm( C, A, B, ring );
#endif

	time_taken = timer.time();
	if( rc == SUCCESS ) {
		out.times.useful = time_taken / static_cast< double >( out.rep );
	}
	// print timing at root process
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "Time taken for a " << out.rep << " "
			<< "mxm calls (hot start): " << out.times.useful << ". "
			<< "Error code is " << out.error_code << std::endl;
	}

	// start postamble
	timer.reset();

	// set error code
	if( rc == FAILED ) {
		out.error_code = 80;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 90;
		return;
	}

	// finish timing
	time_taken = timer.time();
	out.times.postamble = time_taken;

	int nnz = 0;
	auto it = C.begin();
	while( it != C.end() ) {
		if( (*it).second != 0.0 ){
			(void) ++nnz;
		}
		(void) ++it;
	}

	out.result_nnz = nnz;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 7 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <datasetL> <datasetR> "
			<< "<direct/indirect> (inner iterations) (outer iterations) "
			<< "(verification <truth-file>)\n";
		std::cout << "<datasetL>, <datasetR>, and <direct/indirect> are mandatory "
			<< "arguments.\n";
		std::cout << "<datasetL> is the left matrix of the multiplication and "
			<< "<datasetR> is the right matrix \n";
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
	std::cerr << "Warning: benchmark driver compiled without the NDEBUG macro "
		<< "defined(!)\n";
#endif

	// the input struct
	struct input in;

	// get file name Left
	(void) strncpy( in.filenameL, argv[ 1 ], 1023 );
	in.filenameL[ 1023 ] = '\0';

	// get file name Right
	(void) strncpy( in.filenameR, argv[ 2 ], 1023 );
	in.filenameL[ 1023 ] = '\0';

	// get direct or indirect addressing
	if( strncmp( argv[ 3 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		if( strncmp( argv[ 3 ], "indirect", 8 ) == 0 ) {
			in.direct = false;
		} else {
			std::cerr << "Error: could not parse third argument \"" << argv[ 3 ] << ", "
				<< "expected \"direct\" or \"indirect\"\n";
			return 10;
		}
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 5 ) {
		in.rep = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ] << " "
				<< "for number of inner experiment repetitions." << std::endl;
			return 20;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 6 ) {
		outer = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 5 ] << " "
				<< "for number of outer experiment repetitions." << std::endl;
			return 30;
		}
	}

	std::cout << "Executable called with parameters:  Left matrix A = "
		<< in.filenameL << ", right matrix B = " << in.filenameR << ", "
		<< "inner repititions = " << in.rep
		<< ", and outer reptitions = " << outer
		<< std::endl;

	// the output struct
	struct output out;

	// set standard exit code
	grb::RC rc = SUCCESS;

	// launch I/O program
	{
		bool success;
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &ioProgram, in, success, true );
		if( rc != SUCCESS ) {
			std::cerr << "Error: could not launch I/O subprogram\n";
			return 40;
		}
		if( !success ) {
			std::cerr << "Error: I/O subprogram failed\n";
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
				<< grb::toString(rc) << std::endl;
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

	std::cout << "Number of non-zeroes in output matrix: "
		<< out.result_nnz << "\n";

	if( out.error_code == 0 && out.pinnedVector.size() > 0 ) {
		std::cerr << std::fixed;
		std::cerr << "Output matrix: (";
		for( size_t k = 0; k < out.pinnedVector.nonzeroes(); k++ ) {
			const auto &nonZeroValue = out.pinnedVector.getNonzeroValue( k );
			std::cerr << nonZeroValue << ", ";
		}
		std::cerr << ")" << std::endl;
		std::cerr << std::defaultfloat;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		std::cout << "Test OK\n";
	}
	std::cout << std::endl;

	// done
	if( out.error_code == 0 ) {
		return 0;
	} else {
		return (80 + out.error_code);
	}
}

