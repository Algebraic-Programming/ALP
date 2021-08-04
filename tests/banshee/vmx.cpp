
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

#include <cstdio>

#include "graphblas.hpp"

//#define VERTICES 15

// static const double    data1[ VERTICES ] = {  4,  7,  4,  6,  4};
// static const double    data1[ 15 ] = {  4,  7,  4,  6,  4,  7,  1,  7,  3,  6,  7,  5,  1,  8,  7 };
// static const double      chk[ VERTICES ] = { 32, 63, 32, 36, 32};

extern const uint32_t I[];
extern size_t I_size;

extern const uint32_t J[];
extern size_t J_size;

extern const double V[];
extern size_t V_size;

extern const double X[];
extern size_t X_size;

extern const double Y[];
extern size_t Y_size;

double const EPS = 10;

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	size_t vertices = X_size / sizeof( double );

	size_t edges = I_size / sizeof( uint32_t );

#ifdef _DEBUG
	printf( "\nNumber of vertices  %zu\n", vertices );
	printf( "\nNumber of edges is %zu\n", edges );

	for( unsigned int i = 0; i < edges; i++ ) {
		printf( "%ld ", I[ i ] );
	}
	printf( "\n" );

	for( unsigned int i = 0; i < edges; i++ ) {
		printf( "%ld ", J[ i ] );
	}

	printf( "\n" );

	for( unsigned int i = 0; i < edges; i++ ) {
		printf( "%ld ", (long)V[ i ] );
	}

	printf( "\n" );

	for( unsigned int i = 0; i < vertices; i++ ) {
		printf( "%ld ", (long)X[ i ] );
	}

	printf( "\n" );

	for( unsigned int i = 0; i < vertices; i++ ) {
		printf( "%ld ", (long)Y[ i ] );
	}

	printf( "\n" );
#endif

	// initialise
	int error = 0;
	enum grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from grb::init: %d.\n", (int)rc );
		error = 2;
	}

	// exit early if failure detected as this point
	if( error ) {
		(void)printf( "Test FAILED.\n\n" );
#ifndef _GRB_NO_STDIO
		(void)fflush( stderr );
		(void)fflush( stdout );
#endif
		return error;
	}

	// allocate
	grb::Vector< double > x( vertices );
	grb::Vector< double > y( vertices );
	grb::Matrix< double > A( vertices, vertices );

	// resize for lines in file elements
	rc = resize( A, edges );
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from Matrix constructor: %d.\n", (int)rc );
		error = 3;
	}

	// initialise x
	if( ! error ) {
		const double * iterator = &( X[ 0 ] );
		rc = grb::buildVector( x, iterator, iterator + vertices, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from Vector build (x): %d.\n", (int)rc );
			error = 4;
		}
	}

	// initialise y
	if( ! error ) {
		rc = grb::set( y, 0 );
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from Vector build (y): %d.\n", (int)rc );
			error = 5;
		}
	}

	// check contents of x
	const double * __restrict__ const xraw = x.raw();
	for( size_t i = 0; ! error && i < vertices; ++i ) {
		if( ! grb::utils::equals( (int)X[ i ], (int)xraw[ i ] ) ) {
			(void)printf( "Initialisation error: vector x element at position "
						  "%d: %d does not equal %d.\n",
				i, (int)xraw[ i ], (int)X[ i ] );
			error = 20;
		}
	}
	// check contents of y
	const double * __restrict__ const against = y.raw();
	for( size_t i = 0; ! error && i < vertices; ++i ) {
		if( ! grb::utils::equals( 0, (int)against[ i ] ) ) {
			(void)printf( "Initialisation error: vector y element at position "
						  "%d: %d does not equal %d.\n",
				i, 0, (int)against[ i ] );
			error = 6;
			return error;
		}
	}

	// initialise A
	rc = grb::buildMatrixUnique( A, I, J, V, edges, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from Matrix buildMatrixUnique: "
					  "%d.\n",
			(int)rc );
		error = 7;
	}

	// get a semiring where multiplication is addition, and addition is multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > integers;

	// execute what amounts to elementwise vector addition
	if( ! error ) {
		rc = grb::vxm( y, x, A, integers );
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from grb::vxm: %d.\n", (int)rc );
			error = 8;
		}
	}

	// check
	for( size_t i = 0; ! error && i < vertices; ++i ) {
		if( fabs( Y[ i ] - against[ i ] ) > EPS ) {
			(void)printf( "Output vector element mismatch at position %zd: %d "
						  "does not equal %d.\n",
				i, (int)Y[ i ], (int)against[ i ] );
			error = 9;
		}
		//        printf("%d %d\n", (int) Y[i], (int) against[i]);
	}

	// finalize
	if( error ) {
		(void)grb::finalize();
	} else {
		rc = grb::finalize();
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from grb::finalize: %d.\n", (int)rc );
			error = 10;
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
