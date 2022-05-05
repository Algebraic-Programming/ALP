
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

#define ERR( ret, fun )    \
	ret = ret ? ret : fun; \
	assert( ret == SUCCESS );

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
};

void grbProgram( const struct input & data_in, struct output & out ) {

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
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parserL( data_in.filenameL, data_in.direct );

	grb::utils::MatrixFileReader< double,
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parserR( data_in.filenameR, data_in.direct );

	assert( parserL.n() == parserR.m() );

	const size_t l = parserL.m();
	const size_t m = parserL.n();
	const size_t n = parserR.n();

	out.times.io = timer.time();
	timer.reset();

	// load into GraphBLAS
	Matrix< double > L( l, m ), R( m, n );
	{
		RC rc = buildMatrixUnique( L, parserL.begin( SEQUENTIAL ), parserL.end( SEQUENTIAL ), SEQUENTIAL );
		/* Once internal issue #342 is resolved this can be re-enabled
		const RC rc = buildMatrixUnique( A,
		    parser.begin( PARALLEL ), parser.end( PARALLEL),
		    PARALLEL
		);*/
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed "
					  << "(" << toString( rc ) << ")." << std::endl;
			return;
		}

		rc = buildMatrixUnique( R, parserR.begin( SEQUENTIAL ), parserR.end( SEQUENTIAL ), SEQUENTIAL );

		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed "
					  << "(" << toString( rc ) << ")." << std::endl;
			return;
		}
	}

	// TODO: add R numZeroes?
	// check number of nonzeroes
	try {
		const size_t global_nnz = nnz( L );
		const size_t parser_nnz = parserL.nz();
		if( global_nnz != parser_nnz ) {
			std::cerr << "Failure: global nnz (" << global_nnz << ") does not equal "
					  << "parser nnz (" << parser_nnz << ")." << std::endl;
			return;
		}
	} catch( const std::runtime_error & ) {
		std::cout << "Info: nonzero check skipped as the number of nonzeroes "
				  << "cannot be derived from the matrix file header. The "
				  << "grb::Matrix reports " << nnz( L ) << " nonzeroes.\n";
	}

	RC rc = SUCCESS;

	// test default pagerank run
	Matrix< double > C( l, n );
	const Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();

		ERR( rc, grb::mxm( C, L, R, ring, RESIZE ) );

		ERR( rc, mxm( C, L, R, ring ) );

		double single_time = timer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to mxm did not succeed (" << toString( rc ) << ")." << std::endl;
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
				std::cout << "Info: cold mxm completed"
						  << ". Time taken was " << single_time << " ms. "
						  << "Deduced inner repetitions parameter of " << out.rep << " "
						  << "to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();

		ERR( rc, grb::mxm( C, L, R, ring, RESIZE ) );

		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			ERR( rc, mxm( C, L, R, ring ) );
		}

		time_taken = timer.time();
		if( rc == SUCCESS ) {
			out.times.useful = time_taken / static_cast< double >( out.rep );
		}
		// print timing at root process
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "Time taken for a " << out.rep << " "
					  << "Mxm calls (hot start): " << out.times.useful << ". "
					  << "Error code is " << out.error_code << std::endl;
		}
		sleep( 1 );
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
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 8 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <datasetL> <datasetR> <direct/indirect> "
				  << "(inner iterations) (outer iterations) (verification <truth-file>)\n";
		std::cout << "<datasetL>, <datasetR>, and <direct/indirect> are mandatory arguments.\n";
		std::cout << "<datasetL> is the left matrix of the multiplication and <datasetR> is the right matrix \n";
		std::cout << "(inner iterations) is optional, the default is " << grb::config::BENCHMARKING::inner() << ". "
				  << "If set to zero, the program will select a number of iterations "
				  << "approximately required to take at least one second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is " << grb::config::BENCHMARKING::outer() << ". This value must be strictly larger than 0.\n";
		//std::cout << "(verification <truth-file>) is optional." << std::endl;
		//TODO: Update verification to work with matrices 
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

	std::cout << "Executable called with parameters " << in.filenameL << ", "
			  << "inner repititions = " << in.rep << ", and outer reptitions = " << outer << std::endl;

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
			std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
			return 6;
		}
	}

	// launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code " << grb::toString( rc ) << std::endl;
		return 8;
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		//TODO: update to support matrices
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
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}
