
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

#include "graphblas/algorithms/label.hpp"
#include "graphblas/utils/Timer.hpp"

#include "graphblas.hpp"

using namespace grb;

static size_t n;  // total number of vertices
static size_t l;  // number of vertices with labels
static size_t nz; // number of connected graph vertices

static double * labels;
static double * weights;
static size_t * I;
static size_t * J;

struct input {
	size_t n;
};

struct output {
	RC error_code;
	PinnedVector< double > f;
	grb::utils::TimerResults times;
};

using namespace grb;

// initialise the problem set to be worked upon
// data_in.n gives the number of vertices at the top of a binary tree
bool init_input( const struct input & data_in ) {
	// a binary tree with data_in.n vertices at the leaves
	n = 2 * data_in.n - 1;
	// construct the input labels: the first l (leaves) are clamped to 1 and the rest are 0
	l = data_in.n;
	labels = new double[ n ];
	for( size_t i = 0; i < n; i++ )
		labels[ i ] = ( i < l ) ? 1 : 0;

	// there are n-1 edges in the tree and so 2(n-1) in the matrix, since it's a symmetric
	nz = 2 * ( n - 1 );
	weights = new double[ nz ];
	I = new size_t[ nz ];
	J = new size_t[ nz ];

	size_t level = 0,
		   levels = (size_t)log2( l ); // current tree level and total levels
	size_t edge = 0,
		   edges = l; // current edge at this level and total edges at this level
	for( size_t e = 0; e < ( nz / 2 ); e++ ) {
		size_t dst = ( e & ~0x01 ) + pow( 2.0, ( levels - level ) ) - floor( edge / 2 );
		I[ e ] = e;
		J[ e ] = dst;
		// std::cout << "e " << e << " level " << level << " edge " << edge << " edges " << edges << " I " << I[e] << " J " << J[e] << std::endl;
		weights[ e ] = 1.0;
		size_t e_other = e + ( nz / 2 );
		I[ e_other ] = dst;
		J[ e_other ] = e;
		weights[ e_other ] = 1.0;
		// std::cout << "e other " << e_other << " level " << level << " edge " << edge << " edges " << edges << " I " << I[e_other] << " J " << J[e_other] << std::endl;
		edge++;
		// update counters when we come to the end of the current tree level
		if( edge == edges ) {
			edge = 0;
			edges = edges / 2;
			level++;
		}
	}

	return true;
}

void free_input() {
	// clean up datasets
	delete[] labels;
	delete[] weights;
	delete[] I;
	delete[] J;
}

// main label propagation algorithm
void grbProgram( const struct input & data_in, struct output & out ) {

	grb::utils::Timer timer;
	timer.reset();
	const size_t s = spmd<>::pid();
	(void)s;
	assert( s < spmd<>::nprocs() );

	// get input n
	out.error_code = SUCCESS;

	// initialise problem set
	init_input( data_in );
	out.times.io = timer.time();
	timer.reset();

	// create the intial set of l input labels in the vector y
	Vector< double > y( n );
	Vector< double > f( n );
	buildVector( y, &( labels[ 0 ] ), &( labels[ 0 ] ) + n, SEQUENTIAL );

	// create the symmetric weight matrix W, representing the weighted graph
	Matrix< double > W( n, n );
	resize( W, nz );
	RC rc = buildMatrixUnique( W, &( I[ 0 ] ), &( J[ 0 ] ), &( weights[ 0 ] ), nz, SEQUENTIAL );
	if( rc != SUCCESS ) {
		out.error_code = ILLEGAL;
		free_input();
		return;
	}
	out.times.preamble = timer.time();
	timer.reset();

	algorithms::label( f, y, W, n, l );
	out.times.useful = timer.time();
	timer.reset();
	out.f = PinnedVector< double >( f, SEQUENTIAL );
	free_input();
	out.times.postamble = timer.time();
	timer.reset();
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 2 || argc > 4 ) {
		std::cout << "Usage: " << argv[ 0 ]
				  << " <number of vertices> (number of inner iterations) "
					 "(number of outer iterations)"
				  << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	in.n = atoi( argv[ 1 ] );
	std::cout << "Executable called with parameters #vertices = " << in.n << std::endl;

	// the output struct
	struct output out;

	grb::Launcher< AUTOMATIC > launcher;

	enum grb::RC rc = launcher.exec( &grbProgram, in, out );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
		return 50;
	}

	std::cout << "Error code is " << out.error_code << ".\n";

	// done
	if( out.error_code != SUCCESS ) {
		std::cout << "Test FAILED.\n\n";
		return 1;
	}
	std::cout << "Test OK.\n\n";
	return 0;
}
