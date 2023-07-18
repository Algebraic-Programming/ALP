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
#include <cinttypes>

#include <graphblas.hpp>
#include <graphblas/algorithms/sssp.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <utils/output_verification.hpp>

using namespace grb;
using weight_t = int;

template< typename T >
Vector< T > stdToGrbVector( const std::vector< T > & in ) {
	Vector< T > out( in.size() );
	for( size_t i = 0; i < in.size(); i++ )
		grb::setElement( out, in[ i ], i );
	return out;
}

template< typename T >
struct input_t {
	Matrix< T > A;
	size_t root;
	const Vector< T > & expected_distances;

	// Empty constructor necessary for distributed backends
	input_t( 
		const Matrix<T>& A = {0,0},
		size_t root = 0,
		const Vector< T > & expected_distances = {0} 
	) : A( A ), root( root ), expected_distances( expected_distances ) {}
};

struct output_t {
	RC rc = SUCCESS;
	utils::TimerResults times;
};

template< typename T >
void grbProgram( const struct input_t< T > & input, struct output_t & output ) {
	std::cout << std::endl << "Running SSSP" << std::endl;
	output.rc = SUCCESS;
	utils::Timer timer;

	timer.reset();
	bool explored_all = false;
	size_t max_level = 0;
	Vector< T > distances( grb::nrows( input.A ) );
	Vector< T > x( grb::nrows( input.A ) ), y( grb::nrows( input.A ) );
	output.times.preamble = timer.time();

	timer.reset();
	output.rc = algorithms::sssp( input.A, input.root, explored_all, max_level, distances, x, y );
	output.times.useful = timer.time();

	// Check distances by comparing it with the expected one
	if( std::equal( input.expected_distances.cbegin(), input.expected_distances.cend(), distances.cbegin() ) ) {
		std::cout << "SUCCESS: distances are correct" << std::endl;
	} else {
		std::cerr << "FAILED: distances are incorrect" << std::endl;
		std::cerr << "distances != expected_distances" << std::endl;
		for( size_t i = 0; i < grb::nrows( input.A ); i++ )
			std::cerr << std::string( 3, ' ' ) << distances[ i ] << " | " << input.expected_distances[ i ] << std::endl;
		output.rc = FAILED;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;

	Launcher< AUTOMATIC > launcher;
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	if( argc != 1 ) {
		std::cerr << "Usage: \n\t" << argv[ 0 ] << std::endl;
		return 1;
	}

	/** Matrix A0: Fully connected acyclic graph
	 *
	 * Schema:
	 *  0 ----- 1
	 *  | \   / |
	 *  |   X   |
	 *  | /   \ |
	 *  2 ----- 3
	 *
	 */
	{ // Directed version, root = 0, uniform weights = 1
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 1, 1, 1 };
		std::cout << "-- Running test on A0 (undirected, acyclic, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 } };
		std::vector< size_t > A_cols { { 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2 } };
		std::vector< weight_t > A_values( A_rows.size(), 1 );
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, stdToGrbVector( expected_distances ) };
		output_t output;
		RC rc = launcher.exec( &grbProgram, input, output, true );
		if( rc != SUCCESS ) {
			std::cerr << "ERROR during execution: rc = " << rc << std::endl;
			return rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}

	/** Matrix A1:
	 *
	 * Schema:
	 *  0 ----- 1
	 *  | \
	 *  |   \
	 *  |     \
	 *  2       3
	 *
	 */
	{ // Directed version, root = 0, uniform weights = 1
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 1, 1, 1 };
		std::cout << "-- Running test on A1 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 0 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 1 );
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, stdToGrbVector( expected_distances ) };
		output_t output;
		RC rc = launcher.exec( &grbProgram, input, output, true );
		if( rc != SUCCESS ) {
			std::cerr << "ERROR during execution: rc = " << rc << std::endl;
			return rc;
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
	{ // Directed version, root = 0, uniform weights = 1
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 1, 1, 2 };
		std::cout << "-- Running test on A2 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 1 );
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, stdToGrbVector( expected_distances ) };
		output_t output;
		RC rc = launcher.exec( &grbProgram, input, output, true );
		if( rc != SUCCESS ) {
			std::cerr << "ERROR during execution: rc = " << rc << std::endl;
			return rc;
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
	{ // Directed version, root = 0, uniform weights = 1
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 1, 2, 3 };
		std::cout << "-- Running test on A2.1 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 1 );
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, stdToGrbVector( expected_distances ) };
		output_t output;
		RC rc = launcher.exec( &grbProgram, input, output, true );
		if( rc != SUCCESS ) {
			std::cerr << "ERROR during execution: rc = " << rc << std::endl;
			return rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}
	{ // Directed version, root = 0, uniform weights = 10
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 10, 10, 30 };
		std::cout << "-- Running test on A2.2 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 10 );
		grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, stdToGrbVector( expected_distances ) };
		output_t output;
		RC rc = launcher.exec( &grbProgram, input, output, true );
		if( rc != SUCCESS ) {
			std::cerr << "ERROR during execution: rc = " << rc << std::endl;
			return rc;
		} else if( output.rc ) {
			std::cerr << "Test failed: rc = " << output.rc << std::endl;
			return output.rc;
		}
		std::cout << std::endl;
	}

	std::cout << "Test OK" << std::endl;

	return 0;
}
