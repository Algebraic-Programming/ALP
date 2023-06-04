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

#include <graphblas/algorithms/bfs.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>
#include <utils/output_verification.hpp>

grb::Vector< size_t > stdVectorToGrbVector( const std::vector< size_t > & in ) {
	grb::Vector< size_t > out( in.size() );
	for( size_t i = 0; i < in.size(); i++ )
		grb::setElement( out, in[ i ], i );
	return out;
}

template< typename T = void >
struct input_t {
	grb::Matrix< T > A;
	size_t root;
	size_t expected_total_steps;
	bool test_steps_per_vertex;
	grb::Vector< size_t > expected_steps_per_vertex;
};

struct output_t {
	grb::RC rc = grb::RC::SUCCESS;
	grb::utils::TimerResults times;
	size_t data_in_local;
};

template< typename T >
void grbProgram( const input_t< T > & input, output_t & output ) {
	std::cout << std::endl << "Running BFS" << std::endl;
	grb::utils::Timer timer;
	size_t total_steps = 0;

	timer.reset();
	output.rc = output.rc ? output.rc : grb::algorithms::bfs_steps( total_steps, input.A, input.root );
	timer.reset();

	// TODO: Add check for steps_per_vertex
	if( total_steps <= input.expected_total_steps ) {
		std::cout << "SUCCESS: total_steps = " << total_steps << " is correct" << std::endl;
	} else {
		std::cerr << "FAILED: expected maximum " << input.expected_total_steps << " total_steps but got " << total_steps << std::endl;
		output.rc = grb::RC::FAILED;
	}

	timer.reset();
	grb::Vector< size_t > steps_per_vertex( grb::nrows( input.A ) );
	output.rc = output.rc ? output.rc : grb::algorithms::bfs_steps_per_vertex( total_steps, steps_per_vertex, input.A, input.root );
	timer.reset();

	// Check steps_per_vertex by comparing it with the expected one
	if( std::equal( input.expected_steps_per_vertex.cbegin(), input.expected_steps_per_vertex.cend(), steps_per_vertex.cbegin() ) ) {
		std::cout << "SUCCESS: steps_per_vertex is correct" << std::endl;
	} else {
		std::cerr << "FAILED: steps_per_vertex is incorrect" << std::endl;
		output.rc = grb::RC::FAILED;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;
	constexpr size_t niterations = 1;

	grb::Benchmarker< grb::EXEC_MODE::AUTOMATIC > benchmarker;
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// Check if we are testing on a file
	if( argc != 1 && argc != 3 ) {
		std::cerr << "Usage: \n\t" << argv[ 0 ] << " [ <graph_path> <expected_triangle_count> ]" << std::endl;
		return 1;
	}
	bool test_on_file = argc == 3;
	std::string file_to_test( test_on_file ? argv[ 1 ] : "" );
	size_t expected_file_triangles = test_on_file ? std::stoul( argv[ 2 ] ) : 0;

	/** Matrix A1:
	 *
	 * Schema:
	 *  0 ----- 1
	 *  | \
	 *  |   \
	 *  |     \
	 *  2       3
	 *
	 * => 1 step(s) to reach all nodes, root = 0
	 */
	{ // Directed version, pattern matrix
		std::cout << "-- Running test on A1 (directed, non-pattern)" << std::endl;
		size_t expected_total_steps = 1;
		size_t root = 0;
		grb::Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 0 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
		std::vector< size_t > expected_steps_per_vertex { 0, 1, 1, 1 };
		input_t< void > input { A, root, expected_total_steps, true, stdVectorToGrbVector( expected_steps_per_vertex ) };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}

	/** Matrix A2:
	 *
	 * Schema:
	 *  0 ----- 1
	 *  |
	 *  |
	 *  |
	 *  2 ----- 3
	 *
	 * => 2 step(s) to reach all nodes, root = 0
	 */
	{ // Directed version, pattern matrix
		std::cout << "-- Running test on A2 (directed, pattern)" << std::endl;
		size_t expected_total_steps = 2;
		size_t root = 0;
		grb::Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
		std::vector< size_t > expected_steps_per_vertex { 0, 1, 1, 2 };
		input_t< void > input { A, root, expected_total_steps, true, stdVectorToGrbVector( expected_steps_per_vertex ) };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}

	/** Matrix A3:
	 *
	 * Schema:
	 *  0 ----- 1
	 *          |
	 *          |
	 *          |
	 *  2 ----- 3
	 *
	 * => 3 step(s) to reach all nodes, root = 0
	 */
	{ // Directed version, non-pattern matrix
		std::cout << "-- Running test on A3 (directed, non-pattern: int)" << std::endl;
		size_t expected_total_steps = 3;
		size_t root = 0;
		grb::Matrix< int > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 3 } };
		std::vector< size_t > A_cols { { 1, 3, 2 } };
		std::vector< int > A_values( A_rows.size(), 1 );
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_values.size(), grb::IOMode::PARALLEL );
		std::vector< size_t > expected_steps_per_vertex { 0, 1, 3, 2 };
		input_t< int > input { A, root, expected_total_steps, true, stdVectorToGrbVector( expected_steps_per_vertex ) };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}
	{ // Directed version, pattern matrix
		std::cout << "-- Running test on A3 (directed, pattern)" << std::endl;
		size_t expected_total_steps = 3;
		size_t root = 0;
		grb::Matrix< void > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 3 } };
		std::vector< size_t > A_cols { { 1, 3, 2 } };
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
		std::vector< size_t > expected_steps_per_vertex { 0, 1, 3, 2 };
		input_t< void > input { A, root, expected_total_steps, true, stdVectorToGrbVector( expected_steps_per_vertex ) };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}


	std::cout << "Test OK" << std::endl;

	return 0;
}
