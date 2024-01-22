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

#include <graphblas.hpp>
#include <graphblas/algorithms/sssp.hpp>
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
	const std::vector< T >& expected_distances;

	// Empty constructor necessary for distributed backends
	explicit input_t(
		const Matrix<T>& A = {0,0},
		size_t root = 0,
		const std::vector< T >& expected_distances = {}
	) : A( A ), root( root ), expected_distances( expected_distances ) {}
};

template< typename T >
RC test_case( const struct input_t< T > & input ) {
	std::cout << std::endl << "Running SSSP" << std::endl;
	RC rc = SUCCESS;

	bool explored_all = false;
	size_t max_level = 0;
	Vector< T > distances( grb::nrows( input.A ) );
	Vector< T > x( grb::nrows( input.A ) ), y( grb::nrows( input.A ) );

	rc = rc ? rc : algorithms::sssp( input.A, input.root, explored_all, max_level, distances, x, y );

	// Check distances by comparing it with the expected one
	bool equals_distances = true;
	for( size_t i = 0; i < grb::nrows( input.A ) && equals_distances; i++ ) {
		equals_distances &= (input.expected_distances[ i ] == distances[ i ]);
	}
	if( not equals_distances ) {
		std::cerr << "FAILED: distances are incorrect" << std::endl;
		std::cerr << "distances != expected_distances" << std::endl;
		for( size_t i = 0; i < grb::nrows( input.A ); i++ )
			std::cerr << std::string( 3, ' ' ) << distances[ i ] << " | " << input.expected_distances[ i ] << std::endl;
		rc = FAILED;
	}

	return rc;
}

void grb_test_suite( const void *, const size_t, RC& rc ) {

	/** Matrix A0: Fully connected graph
	 *
	 * [0]     [1]
	 *  ├───────┤
	 * [2]     [3]
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
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, expected_distances };
		rc = test_case(input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
	}

	/** Matrix A1: Node [0] connected to all other nodes
	 *
	 * Schema:
	 *  0 ----- 1
	 *  | \
	 *  |   \
	 *  |     \
	 *  2       3
	 *
	 *  [0] ──┬──▶ [1]
	 *   │    │
	 *   ▼    ▼
	 *  [2]  [3]
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
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, expected_distances };
		rc = test_case(input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
	}

	/** Matrix A2:
	 *
	 *  [0] ──▶ [2] ──▶ [3]
	 *   │
	 *   ▼
	 *  [1]
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
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, expected_distances };
		rc = test_case(input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
	}

	/** Matrix A3:
	 *
	 *  [0] ──▶ [1] ──▶ [2] ──▶ [3]
	 *
	 */
	{ // Directed version, root = 0, uniform weights = 1
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 1, 2, 3 };
		std::cout << "-- Running test on A3.1 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 1 );
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, expected_distances };
		rc = test_case(input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
	}
	{ // Directed version, root = 0, uniform weights = 10
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 10, 20, 30 };
		std::cout << "-- Running test on A3.2 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 1, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 10 );
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root,  expected_distances };
		rc = test_case(input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
	}

	/** Matrix A4:
	 *  Graph A3 with a shortcut from [0] to [2]
	 *
	 *  [0] ──▶ [1] ──▶ [2] ──▶ [3]
	 *   │               ▲
	 *   └───────────────┘
	 *
	 */
	{ // Directed version, root = 0, uniform weights = 1
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 1, 1, 2 };
		std::cout << "-- Running test on A4.1 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 1, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 1 );
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root, expected_distances };
		rc = test_case(input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
	}
	{ // Directed version, root = 0, uniform weights = 10
		size_t root = 0;
		std::vector< weight_t > expected_distances { 0, 10, 10, 20 };
		std::cout << "-- Running test on A4.2 (directed, root " + std::to_string( root ) + ")" << std::endl;
		Matrix< weight_t > A( 4, 4 );
		std::vector< size_t > A_rows { { 0, 0, 1, 2 } };
		std::vector< size_t > A_cols { { 1, 2, 2, 3 } };
		std::vector< weight_t > A_values( A_rows.size(), 10 );
		buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), PARALLEL );
		input_t< weight_t > input { A, root,  expected_distances };
		rc = test_case( input );
		if( rc != SUCCESS ) {
			std::cerr << "Test failed: rc = " << rc << std::endl;
			return;
		}
		std::cout << std::endl;
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

	RC success = SUCCESS;
	RC execution_rc = launcher.exec( &grb_test_suite, nullptr, 0, success, true );
	if( execution_rc != SUCCESS ) {
		std::cerr << "ERROR during execution: execution_rc is "
			<< toString(execution_rc) << std::endl;
		return execution_rc;
	}

	if( success != SUCCESS ) {
		std::cerr << "Test FAILED. Return code (RC) is " << grb::toString(success) << std::endl;
		return FAILED;
	}
	std::cout << "Test OK" << std::endl;

	return 0;
}
