
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

using namespace grb;

static const double data1[ 15 ] = { 4.32, 7.43, 4.32, 6.54, 4.21, 7.65, 7.43, 7.54, 5.32, 6.43, 7.43, 5.42, 1.84, 5.32, 7.43 };
static const double data2[ 15 ] = { 8.49, 7.84, 8.49, 6.58, 8.91, 7.65, 7.84, 7.58, 5.49, 6.84, 7.84, 5.89, 1.88, 5.49, 7.84 };
static const double chk[ 15 ] = { 12.81, 15.27, 12.81, 13.12, 13.12, 15.30, 15.27, 15.12, 10.81, 13.27, 15.27, 11.31, 3.72, 10.81, 15.27 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	// sanity check
	int error = 0;
	for( size_t i = 0; i < 15; ++i ) {
		if( ! grb::utils::equals( data1[ i ] + data2[ i ], chk[ i ], static_cast< double >( 1 ) ) ) {
			(void)fprintf( stderr,
				"Sanity check error at position %zd: %lf + %lf does not equal "
				"%lf.\n",
				i, data1[ i ], data2[ i ], chk[ i ] );
			error = 1;
		}
	}

	if( error )
		return error;

	// initialise
	enum grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Could not initialise default GraphBLAS backend.\n" );
		return 2;
	}

	// allocate
	grb::Vector< double > x( 15 );
	grb::Matrix< double > A( 15, 15 );

	rc = resize( A, 15 );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Matrix resize: %d.\n", (int)rc );
		return 3;
	}

	grb::Vector< double > y( 15 );

	// initialise
	const double * iterator = &( data1[ 0 ] );
	rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Vector build (x): %d.\n", (int)rc );
		return 4;
	}
	rc = grb::set( y, 1 );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Vector assign (y): %d.\n", (int)rc );
		return 5;
	}
	rc = grb::buildMatrixUnique( A, I, J, data2, 15, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Matrix build (A): %d.\n", (int)rc );
		return 6;
	}

	// get a semiring where multiplication is addition, and addition is multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring< grb::operators::mul< double >, grb::operators::add< double >, grb::identities::one, grb::identities::zero > switched;

	// execute what amounts to elementwise vector addition
	rc = grb::vxm( y, x, A, switched );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from grb::vmx (y=xA): %d.\n", (int)rc );
		return 7;
	}

	// check
	const double * __restrict__ const against = y.raw();
	for( size_t i = 0; i < 15; ++i ) {
		if( ! grb::utils::equals( chk[ i ], against[ i ], static_cast< double >( 1 ) ) ) {
			(void)fprintf( stderr,
				"Output vector element mismatch at position %zd: %lf does not "
				"equal %lf.\n",
				i, chk[ i ], against[ i ] );
			error = 8;
		}
	}

	// finalize
	rc = grb::finalize();
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Could not finalize default GraphBLAS backend.\n" );
		error = 9;
	}

	if( ! error ) {
		(void)printf( "Test OK.\n\n" );
	} else {
		(void)printf( "Test FAILED.\n\n" );
	}

	// done
	return error;
}
