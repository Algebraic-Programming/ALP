
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
	char filename[ 1024 ];
	bool direct;
	size_t rep;
	int numelem;
	char ** elements;
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
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	// assume successful run
	out.error_code = 0;

	// create local parser
	grb::utils::MatrixFileReader< double,
		std::conditional< ( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ), grb::config::RowIndexType, grb::config::ColIndexType >::type >
		parser( data_in.filename, data_in.direct );

	const size_t n = parser.n();
	const size_t m = parser.m();

	out.times.io = timer.time();
	timer.reset();

	// load into GraphBLAS
	Matrix< double > A( m, n );
	{
		const RC rc = buildMatrixUnique( A, parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL ), SEQUENTIAL );
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
	}

	// check number of nonzeroes
	try {
		const size_t global_nnz = nnz( A );
		const size_t parser_nnz = parser.nz();
		if( global_nnz != parser_nnz ) {
			std::cerr << "Failure: global nnz (" << global_nnz << ") does not equal "
					  << "parser nnz (" << parser_nnz << ")." << std::endl;
			return;
		}
	} catch( const std::runtime_error & ) {
		std::cout << "Info: nonzero check skipped as the number of nonzeroes "
				  << "cannot be derived from the matrix file header. The "
				  << "grb::Matrix reports " << nnz( A ) << " nonzeroes.\n";
	}

	RC rc = SUCCESS;

	// test default pagerank run
	Vector< double > x( n ), y( m );
	ERR( rc, clear( x ) );

	const Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	// read in the elements of the input vector, if none use default
	if( data_in.numelem == 0 ) {
		int pos = n / 2;
		std::cout << "Setting default source value at position " << pos << "\n";
		rc = setElement( x, 1.0, pos );
		if( rc != 0 ) {
			std::cerr << "Failed to insert entry at position " << pos << "\n";
			out.error_code = 22;
			return;
		}
	} else {
		for( int k = 0; k < data_in.numelem; ++k ) {
			int pos = atoi( data_in.elements[ k ] );
			if( pos < 0 || pos >= n ) {

				std::cerr << "Requested source position " << pos << " is invalid (max is " << n << ")\n";
				out.error_code = 23;
				return;
			} else {
				std::cout << "Setting default source value at position " << pos << "\n";

				rc = setElement( x, 1.0, pos );
				if( rc != 0 ) {
					std::cerr << "Failed to insert entry at position " << pos << "\n";
					out.error_code = 24;
					return;
				}
			}
		}
	}

	out.times.preamble = timer.time();

	// by default, copy input requested repetitions to output repititions performed
	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();

		ERR( rc, clear( y ) ); // TODO: make sparse

		ERR( rc, mxv( y, A, x, ring ) ); // TODO: make sparse

		double single_time = timer.time();
		if( rc != SUCCESS ) {
			std::cerr << "Failure: call to mxv did not succeed (" << toString( rc ) << ")." << std::endl;
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
				std::cout << "Info: cold mxv completed"
						  << ". Time taken was " << single_time << " ms. "
						  << "Deduced inner repetitions parameter of " << out.rep << " "
						  << "to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		double time_taken;
		timer.reset();

		ERR( rc, clear( y ) );

		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {

			ERR( rc, mxv( y, A, x, ring ) );
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

	// output
	out.pinnedVector = PinnedVector< double >( y, SEQUENTIAL );

	// finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	// done
	return;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <dataset> <direct/indirect> "
				  << "(inner iterations) (outer iterations) (source vertex 1) (source vertex 2) ...\n";
		std::cout << "<dataset> and <direct/indirect> are mandatory arguments.\n";
		std::cout << "(inner iterations) is optional, the default is " << grb::config::BENCHMARKING::inner() << ". "
				  << "If set to zero, the program will select a number of iterations "
				  << "approximately required to take at least one second to complete.\n";
		std::cout << "(outer iterations) is optional, the default is " << grb::config::BENCHMARKING::outer() << ". This value must be strictly larger than 0.\n";
		std::cout << "(Source vertices 1, 2 ...) are optional and defines which elements in the vectors are non-zero."
				  << "The default value for this is element n/2 is non-zero \n";
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;

	// get file name
	(void)strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';

	// get direct or indirect addressing
	if( strncmp( argv[ 2 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}

	// get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 2 ] << " "
					  << "for number of inner experiment repititions." << std::endl;
			return 2;
		}
	}

	// get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " "
					  << "for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}

	// pass the rest of the elements to be read/constructed inside the function
	in.numelem = argc - 5;
	in.elements = argv + 5;

	std::cout << "Executable called with parameters " << in.filename << ", "
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
	std::cout << "Size of x is " << out.pinnedVector.size() << ".\n";
	std::cout << "Number of non-zeroes are: " << out.pinnedVector.nonzeroes() << ".\n";
	if( out.error_code == 0 && out.pinnedVector.size() > 0 ) {
		std::cerr << std::fixed;
		std::cerr << "Output vector: (";
		for (size_t k = 0; k < out.pinnedVector.size(); k++)
		{
			const auto & nonZeroValue = out.pinnedVector.getNonzeroValue( k );
			std::cerr << nonZeroValue << ", ";
		}
		std::cout << ")" << std::endl;
		std::cerr << std::defaultfloat;
	}

	if( out.error_code != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	}
	std::cout << std::endl;

	// done
	return out.error_code;
}
