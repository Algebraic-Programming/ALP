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

#include <graphblas/algorithms/sssp.hpp>
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
	const grb::Vector< T > & expected_distances;
};

struct output_t {
	grb::RC rc = grb::RC::SUCCESS;
	grb::utils::TimerResults times;
	size_t data_in_local;
};

template< typename T >
void grbProgram( const struct input_t< T > & input, struct output_t & output ) {
	std::cout << std::endl << "Running SSSP" << std::endl;
	grb::utils::Timer timer;

	grb::Vector< T > distances( grb::nrows( input.A ) );

	timer.reset();
	output.rc = output.rc ? output.rc : grb::algorithms::sssp( input.A, input.root, distances );
	timer.reset();

	// Check distances by comparing it with the expected one
	if( std::equal( input.expected_distances.cbegin(), input.expected_distances.cend(), distances.cbegin() ) ) {
		std::cout << "SUCCESS: distances are correct" << std::endl;
	} else {
		std::cerr << "FAILED: distances are incorrect" << std::endl;
		std::cerr << "distances != expected_distances" << std::endl;
		for( size_t i = 0; i < grb::nrows( input.A ); i++ )
			std::cerr << std::string( 3, ' ' ) << distances[ i ] << " | " << input.expected_distances[ i ] << std::endl;
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
	if( argc != 1 && argc != 4 ) {
		std::cerr << "Usage: \n\t" << argv[ 0 ] << " [ <graph_filepath> <root> <expected_distances_filepath> ]" << std::endl;
		return 1;
	}
	bool test_on_file = ( argc == 4 );

	if( test_on_file ) { // Test on a file
		std::string graph_filepath( argv[ 1 ] );
		size_t root = std::stoul( argv[ 2 ] );
		std::string expected_distances_filepath( argv[ 3 ] );

		std::cout << "-- Running test on file " << graph_filepath << std::endl;

		// Read matrix from file
		grb::utils::MatrixFileReader< double > reader( graph_filepath, false, true );
		size_t r = reader.n(), c = reader.m();
		assert( r == c );
		grb::Matrix< double > A( r, c );
		grb::RC rc_build = buildMatrixUnique( A, reader.cbegin( grb::IOMode::SEQUENTIAL ), reader.cend( grb::IOMode::SEQUENTIAL ), grb::IOMode::PARALLEL );
		if( rc_build != grb::RC::SUCCESS ) {
			std::cerr << "ERROR during buildMatrixUnique: rc = " << rc_build << std::endl;
			return 1;
		}
		std::cout << "Matrix read successfully" << std::endl;
		grb::Vector< double > expected_distances( r );
		// TODO: Read expected_distances vector from file
		// Run the algorithm
		input_t< double > input { A, root, expected_distances };
		output_t output;
		grb::RC bench_rc = benchmarker.exec( &grbProgram, input, output, niterations, 1, true );
		if( bench_rc ) {
			std::cerr << "ERROR during execution on file " << graph_filepath << ": rc = " << bench_rc << std::endl;
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
		 */
		{ // Directed version, root = 0, uniform weights = 1
			size_t root = 0;
			std::vector< double > expected_distances { 0, 1, 1, 1 };
			std::cout << "-- Running test on A1 (directed, root " + std::to_string( root ) + ")" << std::endl;
			grb::Matrix< double > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 0, 0 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
			std::vector< double > A_values( A_rows.size(), 1 );
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(),  A_rows.size(), grb::IOMode::PARALLEL );
			input_t< double > input { A, root, stdToGrbVector( expected_distances ) };
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
		{ // Directed version, root = 0, uniform weights = 1
			size_t root = 0;
            std::vector< double > expected_distances { 0, 1, 1, 2 };
			std::cout << "-- Running test on A2 (directed, root " + std::to_string( root ) + ")" << std::endl;
			grb::Matrix< double > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 0, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
            std::vector< double > A_values( A_rows.size(), 1 );
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), grb::IOMode::PARALLEL );
			input_t< double > input { A, root, stdToGrbVector( expected_distances ) };
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
		{ // Directed version, root = 0, uniform weights = 1
			size_t root = 0;
            std::vector< double > expected_distances { 0, 1, 2, 3 };
			std::cout << "-- Running test on A2.1 (directed, root " + std::to_string( root ) + ")" << std::endl;
			grb::Matrix< double > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 0, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
            std::vector< double > A_values( A_rows.size(), 1 );
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), grb::IOMode::PARALLEL );
			input_t< double > input { A, root, stdToGrbVector( expected_distances ) };
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
        { // Directed version, root = 0, uniform weights = 10
			size_t root = 0;
            std::vector< double > expected_distances { 0, 10, 10, 30 };
			std::cout << "-- Running test on A2.2 (directed, root " + std::to_string( root ) + ")" << std::endl;
			grb::Matrix< double > A( 4, 4 );
			std::vector< size_t > A_rows { { 0, 0, 2 } };
			std::vector< size_t > A_cols { { 1, 2, 3 } };
            std::vector< double > A_values( A_rows.size(), 10 );
			grb::buildMatrixUnique( A, A_rows.data(), A_cols.data(), A_values.data(), A_rows.size(), grb::IOMode::PARALLEL );
			input_t< double > input { A, root, stdToGrbVector( expected_distances ) };
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
