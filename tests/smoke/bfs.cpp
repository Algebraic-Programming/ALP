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

template< typename T >
grb::Vector< T > stdToGrbVector( const std::vector< T > & in ) {
	grb::Vector< T > out( in.size() );
	for( size_t i = 0; i < in.size(); i++ )
		grb::setElement( out, in[ i ], i );
	return out;
}

template< typename T >
struct input_t {
	grb::Matrix< T > A;
	size_t root;
	size_t expected_max_level;
	bool compute_levels;
	const grb::Vector< size_t > & expected_levels;
	bool compute_parents;
	const grb::Vector< long > & expected_parents;
};

struct output_t {
	grb::RC rc = grb::RC::SUCCESS;
	grb::utils::TimerResults times;
	size_t data_in_local;
};

template< typename T >
void grbProgram( const struct input_t< T > & input, struct output_t & output ) {
	std::cout << std::endl << "Running BFS" << std::endl;
	grb::utils::Timer timer;
	size_t max_level;

	if( input.compute_levels ) {
	    grb::Vector< size_t > levels( grb::nrows( input.A ) );

		timer.reset();
		output.rc = output.rc ? output.rc : grb::algorithms::bfs_levels( input.A, input.root, max_level, levels );
		timer.reset();

		if( max_level <= input.expected_max_level ) {
			std::cout << "SUCCESS: max_level = " << max_level << " is correct" << std::endl;
		} else {
			std::cerr << "FAILED: expected maximum " << input.expected_max_level << " max_level but got " << max_level << std::endl;
			output.rc = grb::RC::FAILED;
			return;
		}

		// Check levels by comparing it with the expected one
		if( std::equal( input.expected_levels.cbegin(), input.expected_levels.cend(), levels.cbegin() ) ) {
			std::cout << "SUCCESS: expected_levels is correct" << std::endl;
		} else {
			std::cerr << "FAILED: levels is incorrect" << std::endl;
			std::cerr << "levels != expected_levels" << std::endl;
			for( size_t i = 0; i < grb::nrows( input.A ); i++ )
				std::cerr << std::string( 3, ' ' ) << levels[ i ] << " | " << input.expected_levels[ i ] << std::endl;
			output.rc = grb::RC::FAILED;
			return;
		}
	}

	if( input.compute_parents ) {
		grb::Vector< long > parents( grb::nrows( input.A ) );

		timer.reset();
		output.rc = output.rc ? output.rc : grb::algorithms::bfs_parents( input.A, input.root, max_level, parents );
		timer.reset();

		if( max_level <= input.expected_max_level ) {
			std::cout << "SUCCESS: max_level = " << max_level << " is correct" << std::endl;
		} else {
			std::cerr << "FAILED: expected maximum " << input.expected_max_level << " max_level but got " << max_level << std::endl;
			output.rc = grb::RC::FAILED;
			return;
		}

		// Check levels by comparing it with the expected one
		if( std::equal( input.expected_parents.cbegin(), input.expected_parents.cend(), parents.cbegin() ) ) {
			std::cout << "SUCCESS: expected_parents is correct" << std::endl;
		} else {
			std::cerr << "FAILED: parents is incorrect" << std::endl;
			std::cerr << "parents != expected_parents" << std::endl;
			for( size_t i = 0; i < grb::nrows( input.A ); i++ )
				std::cerr << std::string( 3, ' ' ) << parents[ i ] << " | " << input.expected_parents[ i ] << std::endl;
			output.rc = grb::RC::FAILED;
			return;
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;
	constexpr size_t niterations = 1;

	grb::Benchmarker< grb::EXEC_MODE::AUTOMATIC > benchmarker;
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// Check if we are testing on a file
	if( argc != 1 && argc != 4 ) {
		std::cerr << "Usage: \n\t" << argv[ 0 ] << " [ <graph_path> <root> <expected_max_level> ]" << std::endl;
		return 1;
	}
	bool test_on_file = ( argc == 4 );

	if( test_on_file ) { // Test on a file
		std::string file_to_test( argv[ 1 ] );
		size_t root = std::stoul( argv[ 2 ] );
		size_t expected_max_level = std::stoul( argv[ 3 ] );

		std::cout << "-- Running test on file " << file_to_test << std::endl;

		// Read matrix from file as a pattern matrix (i.e. no values)
		grb::utils::MatrixFileReader< void > reader( file_to_test, false, true );
		size_t r = reader.n(), c = reader.m();
		grb::Matrix< void > A( r, c );
		grb::RC rc_build = buildMatrixUnique( A, reader.cbegin( grb::IOMode::SEQUENTIAL ), reader.cend( grb::IOMode::SEQUENTIAL ), grb::IOMode::PARALLEL );
		if( rc_build != grb::RC::SUCCESS ) {
			std::cerr << "ERROR during buildMatrixUnique of the pattern matrix: rc = " << rc_build << std::endl;
			return 1;
		}

		std::cout << "Matrix read successfully" << std::endl;
		input_t< void > input { A, root, expected_max_level, false, { 0 }, false, { 0 } };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
		if( bench_rc ) {
			std::cerr << "ERROR during execution of file " << file_to_test << ": rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
	} else {

		/** Matrix A1:
		 *
		 * Schema:
		 *  0 ----- 1
		 *  | \
		 *  |   \
		 *  |     \
		 *  2       3
		 *
		 * => 1 step(s) to reach all nodes
		 */
		{ // Directed version, pattern matrix, root = 0
			size_t root = 0;
			std::cout << "-- Running test on A1 (directed, non-pattern, root " + std::to_string(root) + ")" << std::endl;
			size_t expected_max_level = 1;
			grb::Matrix< void > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 0, 0 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
			std::vector< size_t > expected_levels { 0, 1, 1, 1 };
            std::vector< long > expected_parents { 0, 0, 0, 0 };
			input_t< void > input { A, root, expected_max_level, true, stdToGrbVector( expected_levels ), true, stdToGrbVector( expected_parents ) };
			output_t output;
			grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
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
		 *  0 ----- 2 ----- 3
		 *  |
		 *  |
		 *  |
		 *  1
		 *
		 */
		{ /*
		   * Directed version, pattern matrix, root = 0
		   * => 2 step(s) to reach all nodes
		   */
			size_t root = 0;
			std::cout << "-- Running test on A2 (directed, pattern, root " + std::to_string(root) + ")" << std::endl;
			size_t expected_max_level = 2;
			grb::Matrix< void > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 0, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
			std::vector< size_t > expected_levels { 0, 1, 1, 2 };
			std::vector< long > expected_parents { 0, 0, 0, 2 };
			input_t< void > input { A, root, expected_max_level, true, stdToGrbVector( expected_levels ), true, stdToGrbVector( expected_parents ) };
			output_t output;
			grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
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
		 *
		 *  0 ----- 1 ----- 2 ----- 3
		 *
		 */
		{ /*
		   * Directed version, non-pattern matrix, root = 0
		   * => 3 step(s) to reach all nodes
		   */
			size_t root = 0;
			std::cout << "-- Running test on A3 (directed, non-pattern: int, root " + std::to_string(root) + ")" << std::endl;
			size_t expected_max_level = 3;
			grb::Matrix< int > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 1, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
			std::vector< int > A_values( A_rows.size(), 1 );
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_values.size(), grb::IOMode::PARALLEL );
			std::vector< size_t > expected_levels { 0, 1, 2, 3 };
			std::vector< long > expected_parents { 0, 0, 1, 2 };
			input_t< int > input { A, root, expected_max_level, true, stdToGrbVector( expected_levels ), true, stdToGrbVector( expected_parents ) };
			output_t output;
			grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << output.rc << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
		{ /*
		   * Directed version, pattern matrix, root = 0
		   * => 3 step(s) to reach all nodes
		   */
			size_t root = 0;
			std::cout << "-- Running test on A3 (directed, pattern, root " + std::to_string(root) + ")" << std::endl;
			size_t expected_max_level = 3;
			grb::Matrix< void > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 1, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
			std::vector< size_t > expected_levels { 0, 1, 2, 3 };
			std::vector< long > expected_parents { 0, 0, 1, 2 };
			input_t< void > input { A, root, expected_max_level, true, stdToGrbVector( expected_levels ), true, stdToGrbVector( expected_parents ) };
			output_t output;
			grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << output.rc << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
		{ /*
		   * Directed version, pattern matrix, root = 3
		   * => impossible to reach all nodes
		   */
			size_t root = 3;
			std::cout << "-- Running test on A3 (directed, pattern, root " + std::to_string(root) + ")" << std::endl;
			size_t expected_max_level = ULONG_MAX;
			grb::Matrix< void > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 1, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
			std::vector< size_t > expected_levels { ULONG_MAX, ULONG_MAX, ULONG_MAX, 0 };
			std::vector< long > expected_parents { -1, -1, -1, 3 };
			input_t< void > input { A, root, expected_max_level, true, stdToGrbVector( expected_levels ), true, stdToGrbVector( expected_parents ) };
			output_t output;
			grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << output.rc << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}

		/** Matrix A4:
		 *
		 * Schema:
		 *  0 ----- 1
		 *        / |
		 *      /   |
		 *    /     |
		 *  2 ----- 3
		 *
		 * Note: Contains one cycle
		 */
		{ /*
		   * Directed version, pattern matrix, root = 0
		   * => 3 step(s) to reach all nodes
		   */
			size_t root = 0;
			std::cout << "-- Running test on A4 (directed, pattern, one cycle, root " + std::to_string(root) + ")" << std::endl;
			size_t expected_max_level = 3;
			grb::Matrix< void > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 1, 2, 3 } };
			std::vector< size_t > A_cols { { 1, 3, 1, 2 } };
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_rows.size(), grb::IOMode::PARALLEL );
			std::vector< size_t > expected_levels { 0, 1, 3, 2 };
			std::vector< long > expected_parents { 0, 0, 3, 1 };
			input_t< void > input { A, root, expected_max_level, true, stdToGrbVector( expected_levels ), true, stdToGrbVector( expected_parents ) };
			output_t output;
			grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
			if( bench_rc ) {
				std::cerr << "ERROR during execution: rc = " << bench_rc << std::endl;
				return bench_rc;
			} else if( output.rc ) {
				std::cerr << "Test failed: rc = " << output.rc << std::endl;
				return output.rc;
			}
			std::cout << std::endl;
		}
	}
	std::cout << "Test OK" << std::endl;

	return 0;
}
