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
#include <vector>

#include <graphblas/algorithms/bfs.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>

using namespace grb;

constexpr bool Verbose = false;

template< typename D >
void printSparseVector( const Vector< D > & v, const std::string & name ) {
	if( size( v ) > 50 ) {
		return;
	}
	wait( v );
	std::cout << "  [  ";
	if( nnz( v ) <= 0 ) {
		for( size_t i = 0; i < size( v ); i++ )
			std::cout << "_ ";
	} else {
		size_t nnz_idx = 0;
		auto it = v.cbegin();
		for( size_t i = 0; i < size( v ); i++ ) {
			if( nnz_idx < nnz( v ) && i == it->first ) {
				std::cout << it->second << " ";
				nnz_idx++;
				if( nnz_idx < nnz( v ) )
					++it;
			} else {
				std::cout << "_ ";
			}
		}
	}
	std::cout << " ]  -  "
			  << "Vector \"" << name << "\" (" << size( v ) << ")" << std::endl;
}

struct input_t {
	// Input file parameters
	std::string filename;
	bool direct;
	// Algorithm parameters
	algorithms::AlgorithmBFS algorithm;
	size_t root;
	bool expected_explored_all;
	long expected_max_level;
	bool verify = false;
	const Vector< long > & expected_values; // Levels or parents depending on the selected algorithm

	// Necessary for distributed backends
	input_t( const std::string & filename = "",
		bool direct = true,
		algorithms::AlgorithmBFS algorithm = algorithms::AlgorithmBFS::LEVELS,
		size_t root = 0,
		bool expected_explored_all = true,
		long expected_max_level = 0,
		const Vector< long > & expected_values = { 0 } ) :
		filename( filename ),
		direct( direct ), algorithm( algorithm ), root( root ), expected_explored_all( expected_explored_all ), expected_max_level( expected_max_level ), expected_values( expected_values ) {}
};

struct output_t {
	RC rc = RC::SUCCESS;
	utils::TimerResults times;
	size_t data_in_local;
};

void grbProgram( const struct input_t & input, struct output_t & output ) {
	utils::Timer timer;
	long max_level;
	bool explored_all;

	// Read matrix from file as a pattern matrix (i.e. no values)
	timer.reset();
	utils::MatrixFileReader< void > reader( input.filename, input.direct );
	size_t r = reader.n(), c = reader.m();
	assert( r == c );
	Matrix< void > A( r, c );
	output.rc = buildMatrixUnique( A, reader.cbegin( IOMode::SEQUENTIAL ), reader.cend( IOMode::SEQUENTIAL ), IOMode::SEQUENTIAL );
	if( output.rc != RC::SUCCESS ) {
		std::cerr << "ERROR during buildMatrixUnique of the pattern matrix: " << toString( output.rc ) << std::endl;
		return;
	}
	output.times.io = timer.time();

	// Allocate output vector
	timer.reset();
	Vector< long > values( nrows( A ) );
	output.times.preamble = timer.time();

	// Run the BFS algorithm
	timer.reset();
	output.rc = output.rc ? output.rc : algorithms::bfs( input.algorithm, A, input.root, explored_all, max_level, values );
	grb::wait();
	output.times.useful = timer.time();

	{ // Check the outputs
		if( explored_all == input.expected_explored_all ) {
			std::cout << "SUCCESS: explored_all = " << explored_all << " is correct" << std::endl;
		} else {
			std::cerr << "FAILED: expected explored_all = " << input.expected_explored_all << " but got " << explored_all << std::endl;
			output.rc = output.rc ? output.rc : RC::FAILED;
		}

		if( max_level > 0 && max_level <= input.expected_max_level ) {
			std::cout << "SUCCESS: max_level = " << max_level << " is correct" << std::endl;
		} else {
			std::cerr << "FAILED: expected max_level " << input.expected_max_level << " but got " << max_level << std::endl;
			output.rc = output.rc ? output.rc : RC::FAILED;
		}

		// Check levels by comparing it with the expected one
		if( input.verify && not std::equal( input.expected_values.cbegin(), input.expected_values.cend(), values.cbegin() ) ) {
			std::cerr << "FAILED: values are incorrect" << std::endl;
			std::cerr << "values != expected_values" << std::endl;
			printSparseVector( values, "values" );
			printSparseVector( input.expected_values, "expected_values" );
			output.rc = output.rc ? output.rc : RC::FAILED;
		}

		if( output.rc == RC::SUCCESS && Verbose ) {
			printSparseVector( values, "values" );
		}	
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;

	size_t inner_iterations = 1, outer_iterations = 1;
	Benchmarker< EXEC_MODE::AUTOMATIC > benchmarker;

	if( argc != 6 ) {
		std::cerr << "Usage: \n\t" << argv[ 0 ] << " <graph_path> <direct|indirect> <root> <expected_explored_all> <expected_max_level> [ outer_iters=1 inner_iters=1 ]" << std::endl;
		return 1;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	std::string file_to_test( argv[ 1 ] );
	bool direct = ( std::string( argv[ 2 ] ) == "direct" );
	size_t root = std::stoul( argv[ 3 ] );
	bool expected_explored_all = std::stol( argv[ 4 ] ) > 0;
	long expected_max_level = std::stol( argv[ 5 ] );
	if( argc > 6 )
		outer_iterations = std::stoul( argv[ 6 ] );
	if( argc > 7 )
		inner_iterations = std::stoul( argv[ 7 ] );

	{ // Run the test: AlgorithmBFS::LEVELS
		std::cout << std::endl << "-- Running AlgorithmBFS::LEVELS on file " << file_to_test << std::endl;
		input_t input( file_to_test, direct, algorithms::AlgorithmBFS::LEVELS, root, expected_explored_all, expected_max_level );
		output_t output;
		RC rc = benchmarker.exec( &grbProgram, input, output, inner_iterations, outer_iterations, true );
		if( rc ) {
			std::cerr << "ERROR during execution: rc = " << toString( rc ) << std::endl;
			return rc;
		}
		if( output.rc ) {
			std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
			return output.rc;
		}
	}
	{ // Run the test: AlgorithmBFS::PARENTS
		std::cout << std::endl << "-- Running AlgorithmBFS::PARENTS on file " << file_to_test << std::endl;
		input_t input( file_to_test, direct, algorithms::AlgorithmBFS::PARENTS, root, expected_explored_all, expected_max_level );
		output_t output;
		RC rc = benchmarker.exec( &grbProgram, input, output, inner_iterations, outer_iterations, true );
		if( rc ) {
			std::cerr << "ERROR during execution: rc = " << toString( rc ) << std::endl;
			return rc;
		}
		if( output.rc ) {
			std::cerr << "Test failed: rc = " << toString( output.rc ) << std::endl;
			return output.rc;
		}
	}

	std::cout << "Test OK" << std::endl;
	return 0;
}
