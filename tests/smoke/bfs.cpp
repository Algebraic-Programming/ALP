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

template< typename T >
bool verify_parents( const Matrix< void > & A, Vector< T > parents ) {
	for( const std::pair< size_t, T > & e : parents ) {
		if( e.second < 0 ) // Not found node
			continue;

		if( ( (size_t)e.second ) == e.first ) // Root ndoe
			continue;

		bool ok = std::any_of( A.cbegin(), A.cend(), [ e ]( const std::pair< size_t, size_t > position ) {
			return position.first == ( (size_t)e.second ) && position.second == e.first;
		} );

		if( not ok ) {
			std::cerr << "ERROR: parent " << e.second << " of node " << e.first << " is not a valid edge" << std::endl;
			return false;
		}
	}
	return true;
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

	switch( input.algorithm ) {
		case algorithms::AlgorithmBFS::LEVELS: {
			// AlgorithmBFS::LEVELS specific allocations
			timer.reset();
			Vector< bool > x( nrows( A ), 1UL );
			Vector< bool > y( nrows( A ), 0UL );
			Vector< bool > not_visited( nrows( A ) );
			output.times.preamble += timer.time();

			// Run the algorithm
			timer.reset();
			output.rc = output.rc ? output.rc : algorithms::bfs_levels( A, input.root, explored_all, max_level, values, x, y, not_visited );
			grb::wait();
			output.times.useful = timer.time();

			break;
		}
		case algorithms::AlgorithmBFS::PARENTS: {

			// AlgorithmBFS::PARENTS specific allocations
			timer.reset();
			Vector< long > x( nrows( A ), 1UL );
			Vector< long > y( nrows( A ), 0UL );
			output.times.preamble += timer.time();

			// Run the algorithm
			timer.reset();
			output.rc = output.rc ? output.rc : algorithms::bfs_parents( A, input.root, explored_all, max_level, values, x, y );
			grb::wait();
			output.times.useful = timer.time();

			break;
		}
		default: {
			std::cerr << "ERROR: unknown algorithm" << std::endl;
			output.rc = RC::ILLEGAL;
			return;
		}
	}

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
			output.rc = output.rc ? output.rc : RC::FAILED;
		}

		if( output.rc == RC::SUCCESS && input.algorithm == algorithms::AlgorithmBFS::PARENTS ) {
			bool correct = verify_parents( A, values );
			std::cout << "CHECK - parents are correct is: " << std::to_string( correct ) << std::endl;
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
		std::cout << "-- Running AlgorithmBFS::LEVELS on file " << file_to_test << std::endl;
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
		std::cout << "-- Running AlgorithmBFS::PARENTS on file " << file_to_test << std::endl;
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
