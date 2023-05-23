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

#include <graphblas/algorithms/triangle_count.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>
#include <utils/output_verification.hpp>

/** Must be an integer type (int, long, unsigned, etc.) */
using nonzeroval_t = long;

typedef struct {
	grb::Matrix< nonzeroval_t > A;
	size_t expected_triangle_count;
} input_t;

typedef struct {
	grb::RC rc = grb::RC::SUCCESS;
	std::vector< size_t > triangleCounts; // Per algorithm
	grb::utils::TimerResults times;
	size_t data_in_local;
} output_t;

void grbProgram( const input_t & input, output_t & output ) {

	// get user process ID.
	const size_t s = grb::spmd< grb::config::default_backend >::pid();
	assert( s < grb::spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	std::cout << std::endl << "Running triangle counting with Burkhardt algorithm" << std::endl;
	output.triangleCounts.push_back( 0 );
	triangle_count( grb::algorithms::TriangleCountAlgorithm::Burkhardt, output.triangleCounts.back(), input.A );
	timer.reset();
	if( output.triangleCounts.back() != input.expected_triangle_count ) {
		std::cerr << "ERROR: Burkhardt triangle count failed: expected " << input.expected_triangle_count << " but got " << output.triangleCounts.back() << std::endl;
		output.rc = output.rc ? output.rc : grb::RC::FAILED;
	} else {
		std::cout << "Burkhardt triangle count succeeded: " << output.triangleCounts.back() << std::endl;
	}

	std::cout << std::endl << "Running triangle counting with Cohen algorithm" << std::endl;
	output.triangleCounts.push_back( 0 );
	output.rc = output.rc ? output.rc : output.rc ? output.rc : grb::algorithms::triangle_count( grb::algorithms::TriangleCountAlgorithm::Cohen, output.triangleCounts.back(), input.A );
	timer.reset();
	if( output.triangleCounts.back() != input.expected_triangle_count ) {
		std::cerr << "ERROR: Cohen triangle count failed: expected " << input.expected_triangle_count << " but got " << output.triangleCounts.back() << std::endl;
		output.rc = output.rc ? output.rc : grb::RC::FAILED;
	} else {
		std::cout << "Cohen triangle count succeeded: " << output.triangleCounts.back() << std::endl;
	}

	std::cout << std::endl << "Running triangle counting with Sandia_LL algorithm" << std::endl;
	output.triangleCounts.push_back( 0 );
	triangle_count( grb::algorithms::TriangleCountAlgorithm::Sandia_LL, output.triangleCounts.back(), input.A );
	timer.reset();
	if( output.triangleCounts.back() != input.expected_triangle_count ) {
		std::cerr << "ERROR: Sandia_LL triangle count failed: expected " << input.expected_triangle_count << " but got " << output.triangleCounts.back() << std::endl;
		output.rc = output.rc ? output.rc : grb::RC::FAILED;
	} else {
		std::cout << "Sandia_LL triangle count succeeded: " << output.triangleCounts.back() << std::endl;
	}

	std::cout << std::endl << "Running triangle counting with Sandia_LUT algorithm" << std::endl;
	output.triangleCounts.push_back( 0 );
	grb::algorithms::triangle_count( grb::algorithms::TriangleCountAlgorithm::Sandia_LUT, output.triangleCounts.back(), input.A );
	timer.reset();
	if( output.triangleCounts.back() != input.expected_triangle_count ) {
		std::cerr << "ERROR: Sandia_LUT triangle count failed: expected " << input.expected_triangle_count << " but got " << output.triangleCounts.back() << std::endl;
		output.rc = output.rc ? output.rc : grb::RC::FAILED;
	} else {
		std::cout << "Sandia_LUT triangle count succeeded: " << output.triangleCounts.back() << std::endl;
	}

	std::cout << std::endl << "Running triangle counting with Sandia_ULT algorithm" << std::endl;
	output.triangleCounts.push_back( 0 );
	triangle_count( grb::algorithms::TriangleCountAlgorithm::Sandia_ULT, output.triangleCounts.back(), input.A );
	timer.reset();
	if( output.triangleCounts.back() != input.expected_triangle_count ) {
		std::cerr << "ERROR: Sandia_ULT triangle count failed: expected " << input.expected_triangle_count << " but got " << output.triangleCounts.back() << std::endl;
		output.rc = output.rc ? output.rc : grb::RC::FAILED;
	} else {
		std::cout << "Sandia_ULT triangle count succeeded: " << output.triangleCounts.back() << std::endl;
	}

	std::cout << std::endl << "Running triangle counting with Sandia_UU algorithm" << std::endl;
	output.triangleCounts.push_back( 0 );
	triangle_count( grb::algorithms::TriangleCountAlgorithm::Sandia_UU, output.triangleCounts.back(), input.A );
	timer.reset();
	if( output.triangleCounts.back() != input.expected_triangle_count ) {
		std::cerr << "ERROR: Sandia_UU triangle count failed: expected " << input.expected_triangle_count << " but got " << output.triangleCounts.back() << std::endl;
		output.rc = output.rc ? output.rc : grb::RC::FAILED;
	} else {
		std::cout << "Sandia_UU triangle count succeeded: " << output.triangleCounts.back() << std::endl;
	}

	std::cout << std::endl;
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

	/** Matrix A0:
	 *    0  1  2  3
	 * 0  _  X  X  _
	 * 1  X  _  X  X
	 * 2  X  X  _  X
	 * 3  _  X  X  _
	 *
	 * Schema:
	 *  0 ------ 1
	 *  |      /
	 *  |    /
	 *  |  /
	 *  2        3
	 *
	 * => 1 triangle
	 */
	{ // Undirected version
		size_t expected_triangle_count = 1;
		grb::Matrix< nonzeroval_t > A0_undirected( 4, 4 );
		std::vector< size_t > A0_undirected_rows { { 0, 0, 1, 1, 2, 2 } };
		std::vector< size_t > A0_undirected_cols { { 1, 2, 0, 2, 0, 1 } };
		std::vector< nonzeroval_t > A0_undirected_values( A0_undirected_rows.size(), 1 );
		grb::buildMatrixUnique( A0_undirected, A0_undirected_rows.data(), A0_undirected_cols.data(), A0_undirected_values.data(), A0_undirected_values.size(), grb::IOMode::PARALLEL );
		input_t input_A0_undirected { A0_undirected, expected_triangle_count };
		output_t output_A0_undirected;
		std::cout << "-- Running test on A0_undirected" << std::endl;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input_A0_undirected, output_A0_undirected, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution of A0_undirected: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output_A0_undirected.rc ) {
			std::cerr << "Test failed: rc = " << output_A0_undirected.rc << std::endl;
			return output_A0_undirected.rc;
		}
		std::cout << std::endl;
	}

	/** Matrix A1:
	 *    0  1  2  3
	 * 0  _  X  X  _
	 * 1  X  _  X  X
	 * 2  X  X  _  X
	 * 3  _  X  X  _
	 *
	 * Schema:
	 *  0 ------ 1
	 *  |      / |
	 *  |    /   |
	 *  |  /     |
	 *  2 ------ 3
	 *
	 * => 2 triangles
	 */
	{ // Undirected version
		size_t expected_triangle_count = 2;
		grb::Matrix< nonzeroval_t > A1_undirected( 4, 4 );
		std::vector< size_t > A1_undirected_rows { { 0, 0, 1, 1, 1, 2, 2, 2, 3, 3 } };
		std::vector< size_t > A1_undirected_cols { { 1, 2, 0, 2, 3, 0, 1, 3, 1, 2 } };
		std::vector< nonzeroval_t > A1_undirected_values( A1_undirected_rows.size(), 1 );
		grb::buildMatrixUnique( A1_undirected, A1_undirected_rows.data(), A1_undirected_cols.data(), A1_undirected_values.data(), A1_undirected_values.size(), grb::IOMode::PARALLEL );
		input_t input_A1_undirected { A1_undirected, expected_triangle_count };
		output_t output_A1_undirected;
		std::cout << "-- Running test on A1_undirected" << std::endl;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input_A1_undirected, output_A1_undirected, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution of A1_undirected: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output_A1_undirected.rc ) {
			std::cerr << "Test failed: rc = " << output_A1_undirected.rc << std::endl;
			return output_A1_undirected.rc;
		}
		std::cout << std::endl;
	}

	/** Matrix A2:
	 *    0  1  2  3
	 * 0  _  X  X  X
	 * 1  X  _  X  X
	 * 2  X  X  _  X
	 * 3  X  X  X  _
	 *
	 * Schema:
	 *  0 ----- 1
	 *  |  \  / |
	 *  |   X   |
	 *  | /  \  |
	 *  2 ----- 3
	 *
	 * => 4 triangles
	 */
	{ // Undirected version
		size_t expected_triangle_count = 4;
		grb::Matrix< nonzeroval_t > A2_undirected( 4, 4 );
		std::vector< size_t > A2_undirected_rows { { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 } };
		std::vector< size_t > A2_undirected_cols { { 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2 } };
		std::vector< nonzeroval_t > A2_undirected_values( A2_undirected_rows.size(), 1 );
		grb::buildMatrixUnique( A2_undirected, A2_undirected_rows.data(), A2_undirected_cols.data(), A2_undirected_values.data(), A2_undirected_values.size(), grb::IOMode::PARALLEL );
		input_t input_A2_undirected { A2_undirected, expected_triangle_count };
		output_t output_A2_undirected;
		std::cout << "-- Running test on A2_undirected" << std::endl;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input_A2_undirected, output_A2_undirected, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution of A2_undirected: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output_A2_undirected.rc ) {
			std::cerr << "Test failed: rc = " << output_A2_undirected.rc << std::endl;
			return output_A2_undirected.rc;
		}
		std::cout << std::endl;
	}

	/** Matrix A3:
	 *
	 * Schema:
	 * 0 ----- 1 ----- 2
	 * |  \  / |  \  / |
	 * |   X   |   X   |
	 * | /  \  | /  \  |
	 * 3 ----- 4 ----- 5
	 * |  \  / |  \  / |
	 * |   X   |   X   |
	 * | /  \  | /  \  |
	 * 6 ----- 7 ----- 8
	 *
	 * note: 1-7, 3-5 are not connected
	 *
	 * => 24 triangles
	 */
	{ // Undirected version
		size_t expected_triangle_count = 24;
		grb::Matrix< nonzeroval_t > A3_undirected( 9, 9 );
		std::vector< size_t > A3_undirected_rows { { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8 } };
		std::vector< size_t > A3_undirected_cols { { 1, 2, 3, 4, 6, 0, 2, 3, 4, 5, 0, 1, 4, 5, 8, 0, 1, 4, 6, 7, 0, 1, 2, 3, 5, 6, 7, 8, 1, 2, 4, 7, 8, 0, 3, 4, 7, 8, 3, 4, 5, 6, 8, 2, 4, 5, 6, 7 } };
		std::vector< nonzeroval_t > A3_undirected_values( A3_undirected_rows.size(), 1 );
		grb::buildMatrixUnique( A3_undirected, A3_undirected_rows.data(), A3_undirected_cols.data(), A3_undirected_values.data(), A3_undirected_values.size(), grb::IOMode::PARALLEL );
		input_t input_A3_undirected { A3_undirected, expected_triangle_count };
		output_t output_A3_undirected;
		std::cout << "-- Running test on A3_undirected" << std::endl;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input_A3_undirected, output_A3_undirected, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution of A3_undirected: rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output_A3_undirected.rc ) {
			std::cerr << "Test failed: rc = " << output_A3_undirected.rc << std::endl;
			return output_A3_undirected.rc;
		}
		std::cout << std::endl;
	}

	/** Given matrix in input **/
	if( test_on_file ) {
		std::cout << "-- Running test on file " << file_to_test << std::endl;

		// Read matrix from file as a pattern matrix (i.e. no values), then convert it to a nonzeroval_t matrix
		grb::utils::MatrixFileReader< void > reader( file_to_test, false, true );
		size_t r = reader.n(), c = reader.m();
		if( r != c ) {
			std::cerr << "ERROR: matrix needs to be square" << std::endl;
			return 1;
		}
		grb::Matrix< void > A_pattern( r, r );
		grb::RC rc_build = buildMatrixUnique( A_pattern, reader.cbegin( grb::IOMode::PARALLEL ), reader.cend( grb::IOMode::PARALLEL ), grb::IOMode::PARALLEL );
		if( rc_build != grb::RC::SUCCESS ) {
			std::cerr << "ERROR during buildMatrixUnique of the pattern matrix: rc = " << rc_build << std::endl;
			return 1;
		}
		grb::Matrix< nonzeroval_t > A( r, r );
		std::vector< size_t > A_rows, A_cols;
		A_rows.reserve( grb::nnz( A_pattern ) );
		A_cols.reserve( grb::nnz( A_pattern ) );
		for( const std::pair< size_t, size_t > & p : A_pattern ) {
			A_rows.push_back( p.first );
			A_cols.push_back( p.second );
		}
		std::vector< nonzeroval_t > A_values( grb::nnz( A_pattern ), static_cast< nonzeroval_t >( 1 ) );
		rc_build = grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_values.size(), grb::IOMode::PARALLEL );
		if( rc_build != grb::RC::SUCCESS ) {
			std::cerr << "ERROR during buildMatrixUnique of the integer matrix: rc = " << rc_build << std::endl;
			return 1;
		}
		std::cout << "Matrix read successfully" << std::endl;
		input_t input { A, expected_file_triangles };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1 );
		if( bench_rc ) {
			std::cerr << "ERROR during execution of file " << file_to_test << ": rc = " << bench_rc << std::endl;
			return bench_rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
	}

	std::cout << "Test OK" << std::endl;

	return 0;
}
