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

template< typename T >
Vector< T > stdToGrbVector( const std::vector< T > & in ) {
	Vector< T > out( in.size() );
	for( size_t i = 0; i < in.size(); i++ )
		setElement( out, in[ i ], i );
	return out;
}

template< typename T >
struct input_t {
	const Matrix< T >& A;
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
    bool expected_equals = std::equal( 
        input.expected_distances.cbegin(), input.expected_distances.cend(), distances.cbegin() 
    );
	if( expected_equals ) {
		std::cout << "SUCCESS: distances are correct" << std::endl;
	} else {
		std::cerr << "FAILED: distances are incorrect" << std::endl;
		std::cerr << "distances != expected_distances" << std::endl;
		for( size_t i = 0; i < grb::nrows( input.A ); i++ )
			std::cerr << std::string( 3, ' ' ) << distances[ i ] << " | " 
                        << input.expected_distances[ i ] << std::endl;
		output.rc = FAILED;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)argv;

	Benchmarker< AUTOMATIC > benchmarker;
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	if( argc < 4 ) {
		std::cerr << "Usage: \n\t" 
                    << argv[ 0 ]
                    << " <dataset>"
                    << " <direct|indirect>"
                    << " <root>"
                    << " <expected_distances_filepath>"
                    << " [ inner_iterations=1 ]"
                    << " [ outer_iterations=1 ]" 
                    << std::endl;
		return 1;
	}
    std::string dataset( argv[ 1 ] );
    bool direct = ( strcmp( argv[ 2 ], "direct" ) == 0 );
    size_t root = std::stoul( argv[ 3 ] );
    std::string expected_distances_filepath( argv[ 4 ] );
	size_t inner_iterations = ( argc >= 5 ) ? std::stoul( argv[ 4 ] ) : 1;
    size_t outer_iterations = ( argc >= 6 ) ? std::stoul( argv[ 5 ] ) : 1;

	std::cout << "-- Running test on file: " << dataset << std::endl;

	// Read matrix from file
	utils::MatrixFileReader< double > reader( dataset, direct, true );
	size_t r = reader.n(), c = reader.m();
	assert( r == c );
	Matrix< double > A( r, c );
	RC rc_build = buildMatrixUnique(
        A, reader.cbegin( SEQUENTIAL ), reader.cend( SEQUENTIAL ), SEQUENTIAL
    );
	if( rc_build != SUCCESS ) {
		std::cerr << "ERROR during buildMatrixUnique: rc = "
                    << grb::toString( rc_build ) << std::endl;
		return 1;
	}
	std::cout << "Matrix read successfully" << std::endl;

	Vector< double > expected_distances( r );
	// TODO: Read expected_distances vector from file

	// Run the algorithm
	input_t< double > input { A, root, expected_distances };
	output_t output;
	RC bench_rc = benchmarker.exec(
        &grbProgram, input, output, inner_iterations, outer_iterations, true
    );
	if( bench_rc ) {
		std::cerr << "ERROR during execution on file " << dataset
                    << ": rc = " << grb::toString(bench_rc) << std::endl;
		return bench_rc;
	} else if( output.rc ) {
		std::cerr << "Test failed: rc = " << grb::toString(output.rc) << std::endl;
		return output.rc;
	}

    std::cerr << std::flush;
	std::cout << "Test OK" << std::endl << std::flush;

	return 0;
}
