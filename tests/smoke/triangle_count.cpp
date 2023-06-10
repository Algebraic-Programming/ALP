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

#ifdef _CG_COMPLEX
#include <complex>
#endif

#include <inttypes.h>

#include <graphblas/algorithms/triangle_count.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>
#include <utils/output_verification.hpp>

/** Must be an integer type (int, long, unsigned, etc.) */
using BaseScalarType = int;
#ifdef _CG_COMPLEX
using IntegerType = std::complex< BaseScalarType >;
#else
using IntegerType = BaseScalarType;
#endif

constexpr BaseScalarType TOL = 0;
constexpr size_t MAX_ITERS = 10000;

using namespace grb;
using namespace algorithms;

struct input {
	size_t inner_rep;
	size_t outer_rep;
	TriangleCountAlgorithm algorithm;
	size_t expectedTriangleCount;
	char filename[ 1024 ];
	bool direct;
};

struct output {
	RC rc = RC::SUCCESS;
	size_t inner_rep;
	size_t outer_rep;
	size_t iterations;
	size_t triangleCount;
	grb::utils::TimerResults times;
};

bool parse_arguments( int argc, char ** argv, input & in, int& err );

void grbProgram( const input & data_in, output & out ) {
	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );
	grb::utils::Timer timer;

	// Sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input." << std::endl;
		out.rc = ILLEGAL;
		return;
	}


	timer.reset();
	// Create a local parser
	grb::utils::MatrixFileReader< 
		void,
		std::conditional<
			( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
			grb::config::RowIndexType,
			grb::config::ColIndexType 
		>::type
	> parser( data_in.filename, data_in.direct );
	assert( parser.m() == parser.n() );
	const size_t n = parser.n();
	// Load the matrix, first as a pattern, then copy it into a matrix with integer values
	Matrix< IntegerType > A( n, n );
	{
		Matrix< void > A_pattern( n, n );
		{
			const RC rc = buildMatrixUnique( A_pattern,
				parser.begin( SEQUENTIAL ), parser.end( SEQUENTIAL),
				SEQUENTIAL
			);
			/* Once internal issue #342 is resolved this can be re-enabled
			const RC rc = buildMatrixUnique( A_pattern,
				parser.begin( PARALLEL ), parser.end( PARALLEL),
				PARALLEL
			);*/
			if( rc != SUCCESS ) {
				std::cerr << "Failure: call to buildMatrixUnique did not succeed "
					<< "(" << toString( rc ) << ")." << std::endl;
				return;
			}
		}
		// Check number of non-zero entries between the parser and the matrix A_pattern
		try {
			const size_t global_nnz = nnz( A_pattern );
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
		// Build A from A_pattern, filled with static_cast< IntegerType>( 1 )
		std::vector< size_t > rows, cols;
		std::vector< IntegerType > values( nnz( A_pattern ), static_cast< IntegerType >( 1 ) );
		rows.reserve( nnz( A_pattern ) );
		cols.reserve( nnz( A_pattern ) );
		for( const std::pair< size_t, size_t > p : A_pattern ) {
			rows.push_back( p.first );
			cols.push_back( p.second );
		}
		buildMatrixUnique( A, rows.data(), cols.data(), values.data(), values.size(), IOMode::SEQUENTIAL );
	}
	out.times.io = timer.time();

	// Check that the input matrix does not contains self-loops
	for( const auto & p : A ) {
		if( p.first.first == p.first.second ) {
			std::cerr << "Failure: input matrix contains self-loops." << std::endl;
			return;
		}
	}


	timer.reset();
	// Allocate the buffers
	Matrix< IntegerType > buffer( n, n );
	Matrix< IntegerType > buffer2( n, n );
	Matrix< IntegerType > L( n, n );
	Matrix< IntegerType > U( n, n );
	// Split A into L and U
	// TODO:
	out.times.preamble = timer.time();

	timer.reset();
	out.rc = triangle_count( data_in.algorithm, out.triangleCount, A, buffer, buffer2, L, U );
	out.times.useful = timer.time();
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;

	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// Input struct
	struct input in;
	int err;
	if( !parse_arguments( argc, argv, in, err ) ) {
		return err;
	}
	
	std::cout << "Executable called with parameters " << in.filename << ", "
		<< "inner repititions = " << in.inner_rep << ", and outer reptitions = " 
		<< in.outer_rep	<< std::endl;

	// Run the test for all algorithms
	RC all_algorithms_rc = RC::SUCCESS;
	for( const std::pair< TriangleCountAlgorithm, std::string > & algo : TriangleCountAlgorithmNames ) {
		in.algorithm = algo.first;
		std::cout << "  -- Running algorithm " << algo.second << std::endl;

		// Output struct
		struct output out;
		RC rc = RC::SUCCESS;

		// Launch the estimator (if requested)
		if( in.inner_rep == 0 ) {
			grb::Launcher< AUTOMATIC > launcher;
			rc = launcher.exec( &grbProgram, in, out, true );
			if( rc == RC::SUCCESS ) {
				in.inner_rep = out.inner_rep;
			}
			if( rc != RC::SUCCESS ) {
				std::cerr << "launcher.exec returns with non-SUCCESS error code "
					<< (int)rc << std::endl;
				return 6;
			}
		}

		// Launch the benchmarker
		grb::Benchmarker< EXEC_MODE::AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, in.outer_rep, true );
		if( rc != RC::SUCCESS ) {
			std::cerr << "benchmarker.exec returns with non-SUCCESS error code "
				<< grb::toString( rc ) << std::endl;
			return 8;
		}
		if( out.rc == RC::SUCCESS ) {
			std::cout << "Benchmark completed successfully.\n";
			std::cout << "** Obtained " << out.triangleCount << " triangles.\n";
			std::cout << "** Expected " << in.expectedTriangleCount << " triangles.\n";
			if( out.triangleCount != in.expectedTriangleCount ) {
				all_algorithms_rc = RC::FAILED;
			}
		} else {
			std::cerr << "Benchmark failed with error code "
				<< grb::toString( out.rc ) << std::endl;
			std::cerr << std::flush;
			all_algorithms_rc = RC::FAILED;
		}
		std::cout << std::endl;
	}

	if( all_algorithms_rc == RC::SUCCESS ) {
		std::cout << "Test OK" << std::endl;
	} else {
		std::cout << "Test FAILED" << std::endl;
	}

	return 0;
}

bool parse_arguments( int argc, char ** argv, input & in, int& err ) {
	// Check if we are testing on a file
	if( argc < 4 || argc > 6 ) {
		std::cerr << "Usages: \n\t" 
			<< argv[ 0 ] << " <graph_filepath> <direct/indirect> <expected_triangle_count> (inner iterations) (outer iterations)" 
			<< std::endl;
		err = 1;
		return false;
	}

	// Get file name
	(void)strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';

	// Get direct or indirect addressing
	in.direct = ( strncmp( argv[ 2 ], "direct", 6 ) == 0 );

	// Get the expected number of triangles
	in.expectedTriangleCount = std::stoul( argv[ 3 ] );

	// Get the inner number of iterations
	in.inner_rep = grb::config::BENCHMARKING::inner();
	char * end = nullptr;
	if( argc > 4 ) {
		in.inner_rep = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 4 ] << " "
				<< "for number of inner experiment repititions." << std::endl;
			err = 4;
			return false;
		}
	}

	// Get the outer number of iterations
	in.outer_rep = grb::config::BENCHMARKING::outer();
	if( argc > 5 ) {
		in.outer_rep = strtoumax( argv[ 5 ], &end, 10 );
		if( argv[ 5 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 5 ] << " "
				<< "for number of outer experiment repititions." << std::endl;
			err = 5;
			return false;
		}
	}
	return true;
}