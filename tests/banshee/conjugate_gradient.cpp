
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

#include <inttypes.h>

#include <graphblas/banshee/algorithms/conjugate_gradient.hpp>

#include <graphblas.hpp>

using namespace algorithms;

extern const uint32_t I[];
extern size_t I_size;

extern const uint32_t J[];
extern size_t J_size;

extern const double V[];
extern size_t V_size;

int main( int argc, char ** argv ) {

	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	size_t I_edges = I_size / sizeof( uint32_t );
	size_t J_edges = J_size / sizeof( uint32_t );
	size_t edges;

	if( I_edges != J_edges ) {
		(void)printf( "I and J arrays do not match.\n\n" );
		return 1;
	} else {
		edges = I_edges;
		(void)printf( "Total number of edges is %d.\n", edges );
	}
	size_t max_I = 0, max_J = 0;
	size_t n;

	for( size_t i = 0; i < edges; i++ ) {
		if( I[ i ] > max_I )
			max_I = I[ i ];
		if( J[ i ] > max_J )
			max_J = J[ i ];
	}
	if( max_I > max_J )
		n = max_I + 1;
	else
		n = max_J + 1;
#ifdef _DEBUG
	printf( "Matrix size is %d\n", n );
#endif

	// initialise
	int error = 0;
	enum grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from grb::init: %d.\n", (int)rc );
		error = 1;
	}

	// exit early if failure detected as this point
	if( error ) {
		(void)printf( "Test FAILED.\n\n" );
		return error;
	}

	// load into GraphBLAS
	grb::Matrix< double > L( n, n );

	// resize for lines in file elements
	rc = resize( L, edges );
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from Matrix constructor: %d.\n", (int)rc );
		error = 2;
		return error;
	}
	rc = grb::buildMatrixUnique( L, I, J, V, edges, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from Matrix buildMatrixUnique: "
					  "%d.\n",
			(int)rc );
		error = 3;
		return error;
	}

	// test default pagerank run
	Vector< double > x( n ), b( n ), r( n ), u( n ), temp( n );
	;
	set( x, static_cast< double >( 1 ) / static_cast< double >( n ) );

	size_t iterations;
	double residual;

	const size_t bn = size( b );
	for( size_t i = 0; i < bn; ++i ) {
		setElement( b, i + 1, i );
	}

	rc = conjugate_gradient( x, L, b, 1000, .001, iterations, residual, r, u, temp );

#ifdef _DEBUG
	// output
	PinnedVector< double > pinnedVector = PinnedVector< double >( x, SEQUENTIAL );

	printf( "Total number of iterations %zu\n", iterations );

	printf( "Size of pr is %d\n", pinnedVector.length() );
	if( pinnedVector.length() > 0 ) {
		printf( "First 10 elements of pr are: ( " );
		if( pinnedVector.mask( 0 ) ) {
			printf( "%d", (int)( pinnedVector[ 0 ] * 10000 ) );
		} else {
			printf( "0" );
		}
		for( size_t i = 1; i < pinnedVector.length() && i < 10; ++i ) {
			printf( ", " );
			if( pinnedVector.mask( i ) ) {
				printf( "%d", (int)( pinnedVector[ i ] * 10000 ) );
			} else {
				printf( "0" );
			}
		}
		printf( " )\n" );
		printf( "First 10 nonzeroes of pr are: ( " );
		size_t nnzs = 0;
		size_t i = 0;
		while( i < pinnedVector.length() && nnzs == 0 ) {
			if( pinnedVector.mask( i ) ) {
				printf( "%d", (int)( pinnedVector[ i ] * 10000 ) );
				++nnzs;
			}
			++i;
		}
		while( nnzs < 10 && i < pinnedVector.length() ) {
			if( pinnedVector.mask( i ) ) {
				printf( ", %d", (int)( pinnedVector[ i ] * 10000 ) );
				++nnzs;
			}
			++i;
		}
		printf( " )\n" );
	}
#endif

	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code of pagerank pr: %d.\n", (int)rc );
		error = 5;
	}

	// finalize
	if( error ) {
		(void)grb::finalize();
	} else {
		rc = grb::finalize();
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from grb::finalize: %d.\n", (int)rc );
			error = 6;
		}
	}

	if( ! error ) {
		(void)printf( "Test OK.\n\n" );
	} else {
		(void)printf( "Test FAILED.\n\n" );
	}

	// done
	return error;
}
