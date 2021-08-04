
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

#include <assert.h>

#include "graphblas.hpp"

using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };
static const bool mask[ 15 ] = { true, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	// sanity check against metabugs
	int error = 0;
	for( size_t i = 0; i < 15; ++i ) {
		if( ! grb::utils::equals( data1[ i ] * data2[ i ], chk[ i ] ) ) {
			(void)printf( "Sanity check error at position %zd: %d + %d does "
						  "not equal %d.\n",
				i, data1[ i ], data2[ i ], chk[ i ] );
			error = 1;
		}
	}

	// initialise
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
	grb::Vector< int > x( 15 );
	grb::Matrix< int > A( 15, 15 );

	// resize for 15 elements
	rc = resize( A, 15 );
	if( rc != grb::SUCCESS ) {
		printf( "Unexpected return code from Matrix constructor: %d.\n", (int)rc );
		error = 3;
	}

	// initialise x
	if( ! error ) {
		const int * iterator = &( data1[ 0 ] );
		rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			printf( "Unexpected return code from Vector build (x): %d.\n", (int)rc );
			error = 4;
		}
	}

	// initialise A
	rc = grb::buildMatrixUnique( A, I, J, data2, 15, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		printf( "Unexpected return code from Matrix buildMatrixUnique: %d.\n", (int)rc );
		error = 5;
	}

	// get a semiring where multiplication is addition, and addition is multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > integers;

	// for each output element, try masked SpMV
	for( size_t i = 0; ! error && i < 15; ++i ) {
		// initialise output vector and mask vector
		grb::Vector< int > y( 15 );
		grb::Vector< bool > m( 15 );

		// check zero sizes
		assert( ! error );
		if( grb::nnz( y ) != 0 ) {
			printf( "Unexpected number of nonzeroes in y: %zd (expected 0).\n", grb::nnz( y ) );
			error = 6;
		}
		if( ! error && grb::nnz( m ) != 0 ) {
			printf( "Unexpected number of nonzeroes in m: %zd (expected 0).\n", grb::nnz( m ) );
			error = 7;
		}

		// initialise mask
		if( ! error ) {
			rc = grb::setElement( m, true, i );
			if( rc != grb::SUCCESS ) {
				printf( "Unexpected return code from vector set (m[%zd]): "
						"%d.\n",
					i, (int)rc );
				error = 8;
			}
		}
		if( ! error ) {
			if( grb::nnz( m ) != 1 ) {
				printf( "Unexpected number of nonzeroes in m: %zd (expected "
						"1).\n",
					grb::nnz( m ) );
				error = 9;
			}
		}

		// execute what amounts to elementwise vector addition
		if( ! error ) {
			rc = grb::vxm( y, m, x, A, integers );
			if( rc != grb::SUCCESS ) {
				printf( "Unexpected return code from grb::vxm: %d.\n", (int)rc );
				error = 10;
			}
		}

		// check sparsity
		if( ! error && grb::nnz( y ) != 1 ) {
			printf( "Unexpected number of nonzeroes in y: %zd (expected 1).\n", grb::nnz( y ) );
			error = 11;
		}

		// check value
		const int * against = y.raw();
		if( ! error && ! grb::utils::equals( chk[ i ], against[ i ] ) ) {
			printf( "Output vector element mismatch at position %zd: %d does "
					"not equal %d.\n",
				i, chk[ i ], against[ i ] );
			error = 12;
		}
	}

	// finalize
	if( error ) {
		(void)grb::finalize();
	} else {
		rc = grb::finalize();
		if( rc != grb::SUCCESS ) {
			printf( "Unexpected return code from grb::finalize: %d.\n", (int)rc );
			error = 13;
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
