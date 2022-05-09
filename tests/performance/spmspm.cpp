
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

#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>
//#include <utils/output_verification.hpp>

#define C1 0.0001
#define C2 0.0001


using namespace grb;

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

void grbProgram( const struct input &data_in, struct output &out ) {
	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	// sanity checks on input
	if( data_in.filenameL[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input for left matrix." << std::endl;
		out.error_code = ILLEGAL;
		return;
	} else if( data_in.filenameR[ 0 ] == '\n' ) {
		std::cerr << s << ": no file name given as input for right matrix." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	// assume successful run
	out.error_code = 0;

	// create local parser
	grb::utils::MatrixFileReader< double,
		std::conditional< (sizeof(grb::config::RowIndexType) >
				sizeof(grb::config::ColIndexType)),
			grb::config::RowIndexType,
			grb::config::ColIndexType
		>::type
	> parserL( data_in.filenameL, data_in.direct );

	grb::utils::MatrixFileReader< double,
		std::conditional< (sizeof(grb::config::RowIndexType) >
				sizeof(grb::config::ColIndexType)),
			grb::config::RowIndexType,
			grb::config::ColIndexType
		>::type
	> parserR( data_in.filenameR, data_in.direct );

	assert( parserL.n() == parserR.m() );

	const size_t l = parserL.m();
	const size_t m = parserL.n();
	const size_t n = parserR.n();

	out.times.io = timer.time();
	timer.reset();

	// load into GraphBLAS
	Matrix< double > A( l, m ), B( m, n );
	{
		RC rc = buildMatrixUnique(
			A,
			parserL.begin( SEQUENTIAL ), parserL.end( SEQUENTIAL ),
			SEQUENTIAL
		);
		/* Once internal issue #342 is resolved this can be re-enabled
		const RC rc = buildMatrixUnique( A,
			parser.begin( PARALLEL ), parser.end( PARALLEL),
			PARALLEL
		);*/
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed for the "
				<< "left-hand matrix " << "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 10;
			return;
		}

		rc = buildMatrixUnique(
			B,
			parserR.begin( SEQUENTIAL ), parserR.end( SEQUENTIAL ),
			SEQUENTIAL
		);

		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed for the "
				<< "right-hand matrix " << "(" << toString( rc ) << ")." << std::endl;
			out.error_code = 20;
			return;
		}
	}

	// check number of nonzeroes
	try {
		const size_t global_nnzL = nnz( A );
		const size_t global_nnzR = nnz( B );
		const size_t parser_nnzL = parserL.nz();
		const size_t parser_nnzR = parserR.nz();
		if( global_nnzL != parser_nnzL ) {
			std::cerr << "Left matrix Failure: global nnz (" << global_nnzL << ") "
				<< "does not equal parser nnz (" << parser_nnzL << ")." << std::endl;
			return;
		} else if( global_nnzR != parser_nnzR ) {
			std::cerr << "Right matrix Failure: global nnz (" << global_nnzR << ") "
				<< "does not equal parser nnz (" << parser_nnzR << ")." << std::endl;
			return;
		}

	} catch( const std::runtime_error & ) {
		std::cout << "Info: nonzero check skipped as the number of nonzeroes "
			<< "cannot be derived from the matrix file header. The "
			<< "grb::Matrix reports " << nnz( A ) << " nonzeroes in left "
			<< "and " << nnz( B ) << " n right \n";
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
			out.error_code = 70;
			return;
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
		}
		if( rc != SUCCESS ) {
			out.error_code = 80;
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
		}
	}

	if( out.rep > 1 ) {
		std::cerr << "Error: more than 1 inner repetitions are not supported due to "
			<< "having to time the symbolic phase while not timing the initial matrix "
			<< "allocation cost\n";
		out.error_code = 90;
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
		out.error_code = 30;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}

	// finish timing
	time_taken = timer.time();
	out.times.postamble = time_taken;

	// copy to pinned vector for printing result comparison
	//TODO: refactor to avoid out of memory error
	/*Vector< double > a( l * n);
	rc = clear(a);

	auto it = C.begin();
	while( it != C.end() ) {
		// col + (row * rowsize)
		const size_t i = ( *it ).first.first + ( ( *it ).first.second * n );

		rc = rc ? rc : setElement( a, ( *it ).second, i );
		it.operator++();

		if( rc != SUCCESS ) {
			std::cerr << "Error during copy/pinning of result matrix: " << rc << '\n';
			out.error_code = 40;
			return;
		}
	}

	out.pinnedVector = PinnedVector< double >( a, SEQUENTIAL );
	*/

	out.result_nnz = nnz(C);

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 7 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <datasetL> <datasetR> <direct/indirect> "
			<< "(inner iterations) (outer iterations) (verification <truth-file>)\n";
		std::cout << "<datasetL>, <datasetR>, and <direct/indirect> are mandatory arguments.\n";
		std::cout << "<datasetL> is the left matrix of the multiplication and "
			<< "<datasetR> is the right matrix \n";
		std::cout << "(inner iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::inner() << ". "
			<< "If set to zero, the program will select a number of iterations "
			<< "approximately required to take at least one second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is "
			<< grb::config::BENCHMARKING::outer() << ". "
			<< "This value must be strictly larger than 0.\n";
		// std::cout << "(verification <truth-file>) is optional." << std::endl;
		// TODO: Update verification to work with matrices
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get file name Left
	(void)strncpy( in.filenameL, argv[ 1 ], 1023 );
	in.filenameL[ 1023 ] = '\0';

	// get file name Right
	(void)strncpy( in.filenameR, argv[ 2 ], 1023 );
	in.filenameL[ 1023 ] = '\0';

	// get direct or indirect addressing
	if( strncmp( argv[ 3 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 5 ) {
		in.rep = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " "
				<< "for number of inner experiment repititions." << std::endl;
			return 2;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 6 ) {
		outer = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}

	// check for verification of the output
	bool verification = false;
	char truth_filename[ 1024 ];
	if( argc >= 7 ) {
		if( strncmp( argv[ 6 ], "verification", 12 ) == 0 ) {
			verification = true;
			if( argc >= 7 ) {
				(void)strncpy( truth_filename, argv[ 7 ], 1023 );
				truth_filename[ 1023 ] = '\0';
			} else {
				std::cerr << "The verification file was not provided as an argument." << std::endl;
				return 5;
			}
		} else {
			std::cerr << "Could not parse argument \"" << argv[ 6 ] << "\", "
				<< "the optional \"verification\" argument was expected." << std::endl;
			return 5;
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
			return 6;
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
		return 8;
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
		std::cerr << "Test FAILED\n";
	} else {
		// TODO: update to support matrices
		/*if( verification ) {
			out.error_code = vector_verification(
				out.pinnedVector, truth_filename,
				C1, C2
			);
			if( out.error_code == 0 ) {
				std::cout << "Output vector verificaton was successful!\n";
				std::cout << "Test OK\n";
			} else {
				std::cerr << std::flush;
				std::cout << "Verification FAILED\n";
				std::cout << "Test FAILED\n";
			}
		} else {
			std::cout << "Test OK\n";
		}*/
		std::cout << "Test OK\n";
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}

