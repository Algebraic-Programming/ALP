
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

#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <inttypes.h>

#include <graphblas/algorithms/label.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>
#include <utils/print_vec_mat.hpp>

using namespace grb;

constexpr size_t MaxPrinting = 10;

// forward declaration of the graph dataset parser
bool readEdges( std::string filename, bool use_indirect, size_t * n, size_t * nz, size_t ** I, size_t ** J, double ** weights );

struct input {
	char filename[ 1024 ];
	bool direct;
	size_t n;
};

struct output {
	RC error_code;
	PinnedVector< double > f;
	grb::utils::TimerResults times;
};

using namespace grb;

// take a sparse matrix, convert to dense format and display with a message

void printMatrix( Matrix< double > & sparse, const size_t n, const char * message ) {

	// only print small matrices
	if( n > MaxPrinting )
		return;

#ifdef _GRB_WITH_LPF
	(void)sparse;
	(void)message;
#else
	// allocate and clear a dense matrix
	print_matrix< double >( sparse, 0, message );
#endif
}

// main label propagation algorithm
void grbProgram( const struct input & data_in, struct output & out ) {

	grb::utils::Timer timer;
	timer.reset();
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	const size_t n = data_in.n;
	out.error_code = SUCCESS;

	// sanity checks on input
	if( data_in.filename[ 0 ] == '\0' ) {
		std::cerr << s << ": no file name given as input." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	// n nodes with 20% labelled
	const size_t l = ( size_t )( (double)n * 0.2 );
	double * const labels = new double[ n ];
	for( size_t i = 0; i < n; i++ ) {
		labels[ i ] = ( i < l ) ? ( rand() % 2 ) : 0; // 0,1
	}

	// initialise problem set
	srand( 314159 );
	grb::utils::MatrixFileReader< void > reader( data_in.filename, data_in.direct );
	size_t nz = reader.nz();
	std::vector< size_t > I;
	std::vector< size_t > J;
	std::vector< double > weights;

	for( const auto &pair : reader ) {
		if( pair.first <= pair.second ) {
			(void)--nz;
		} else {
			I.push_back( pair.first );
			J.push_back( pair.second );
		}
	}
	assert( I.size() == J.size() );
	assert( I.size() == nz );
	nz = I.size();

	// make symmetric
	for( size_t k = 0; k < nz; ++k ) {
		if( I[ k ] != J[ k ] ) {
			I.push_back( J[ k ] );
			J.push_back( I[ k ] );
		}
	}
	assert( I.size() == J.size() );

	// construct symmetric weights matrix with nz non-zeros in total
	for( size_t i = 0; i < nz; i++ ) {
		const double random = // random between 0.01 .. 1.00 (inclusive)
			static_cast< double >((rand() % 100) + 1) / 100.0;
		weights.push_back( random );
	}

	// fill in the symmetric values based on the former half
	for( size_t k = 0; k < nz; k++ ) {
		if( I[ k ] != J[ k ] ) {
			weights.push_back( weights[ k ] );
		}
	}
	nz = I.size();
	assert( weights.size() == nz );

	out.times.io = timer.time();
	timer.reset();

	// create the intial set of l input labels in the vector y
	Vector< double > y( n );
	Vector< double > f( n );
	RC rc = buildVector( y, &(labels[ 0 ]), &(labels[ 0 ]) + n, SEQUENTIAL );

	// create the symmetric weight matrix W, representing the weighted graph
	Matrix< double > W( n, n );
	if( rc == SUCCESS ) {
		rc = resize( W, nz );
	}
	if( rc == SUCCESS ) {
		rc = buildMatrixUnique( W, &(I[ 0 ]), &(J[ 0 ]), &(weights[ 0 ]), nz, SEQUENTIAL );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		out.error_code = FAILED;
	} else {
		printMatrix( W, n, "Symmetric weight matrix W" );
		out.times.preamble = timer.time();

		timer.reset();
		algorithms::label( f, y, W, n, l );
		out.times.useful = timer.time();

		timer.reset();
		out.f = PinnedVector< double >( f, SEQUENTIAL );
		out.times.postamble = timer.time();
		timer.reset();
	}
	delete[] labels;
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	size_t outer = grb::config::BENCHMARKING::outer();
	size_t inner = grb::config::BENCHMARKING::inner();

	// sanity check
	if( argc < 3 || argc > 5 ) {
		std::cout << "Usage: " << argv[ 0 ]
				  << " <dataset> <direct/indirect> (number of inner "
					 "iterations) (number of outer iterations)"
				  << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	if( strlen( argv[ 1 ] ) > 1023 ) {
		std::cerr << "Could not parse filename: too long." << std::endl;
		return 10;
	}
	(void)strncpy( in.filename, argv[ 1 ], 1023 );
	in.filename[ 1023 ] = '\0';
	if( strncmp( argv[ 2 ], "direct", 6 ) == 0 ) {
		in.direct = true;
	} else {
		in.direct = false;
	}
	grb::utils::MatrixFileReader< void > reader( in.filename, in.direct );
	in.n = reader.n();
	if( in.n != reader.m() ) {
		std::cerr << "The given matrix is not square." << std::endl;
		return 20;
	}
	char * end = NULL;
	if( argc >= 4 ) {
		inner = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument for number of inner "
						 "repititions."
					  << std::endl;
			return 30;
		}
	}
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument for number of outer "
						 "repititions."
					  << std::endl;
			return 40;
		}
	}

	std::cout << "Executable called with parameters filename " << in.filename << ", direct = " << in.direct << ", and #vertices = " << in.n << std::endl;

	// the output struct
	struct output out;

	grb::Benchmarker< AUTOMATIC > launcher;

	enum grb::RC rc = launcher.exec( &grbProgram, in, out, inner, outer, true );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
		return 50;
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	// done
	if( out.error_code != SUCCESS ) {
		std::cout << "Test FAILED.\n\n";
		return 255;
	}
	std::cout << "Test OK.\n\n";
	return 0;
}
