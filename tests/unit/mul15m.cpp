
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

#include <stdbool.h>
#include <stdio.h>

#include <assert.h>
#include <string.h>

#include "graphblas.hpp"


using namespace grb;

const static float EPS = std::numeric_limits< float >::epsilon();

const static double data1[ 15 ] = { 4.32, 7.43, 4.32, 6.54, 4.21, 7.65, 7.43, 7.54, 5.32, 6.43, 7.43, 5.42, 1.84, 5.32, 7.43 };
const static int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
const static float chk[ 15 ] = { 34.56, 66.87, 34.56, 39.24, 33.68, 53.55, 59.44, 52.78, 26.60, 12.86, 22.29, 27.10, 1.84, 26.60, 37.15 };
const static float inval[ 15 ] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

// this function detects when something is definitely wrong. It checks for
// relatively large error bounds on the difference and makes use of the
// underlying multiplication in determining the error
static bool err( const float a, const float b, const size_t i ) {
	// note that grb::utils::equals cannot be used here
	const float difference = std::abs( a - b );
	float norm = std::abs( data1[ i ] ) * std::abs( data2[ i ] );
	float absmax = std::abs( data1[ i ] ) > std::abs( data2[ i ] ) ? std::abs( data1[ i ] ) : std::abs( data2[ i ] );
	if( absmax > norm ) { // in case the multiplication results are less or too close to the input values
		norm = 2 * absmax;
	}
	return difference > norm * EPS;
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	float out[ 15 ];
	int error = 0;

	for( size_t i = 0; i < 15; ++i ) {
		if( data1[ i ] * data2[ i ] - chk[ i ] > EPS * chk[ i ] ) {
			(void)fprintf( stderr, "Sanity check error at position %zd: %lf * %d does not equal %f.\n", i, data1[ i ], data2[ i ], chk[ i ] );
			(void)fprintf( stderr, "*** %f, %f, %e\n", data1[ i ] * data2[ i ], chk[ i ], data1[ i ] * data2[ i ] - chk[ i ] );
			error = 1;
		}
	}

	if( error )
		return error;

	typedef grb::operators::internal::mul< double, int, float > internal_op;

	(void)memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		internal_op::apply( &( data1[ i ] ), &( data2[ i ] ), &( out[ i ] ) );
		if( err( out[ i ], chk[ i ], i ) ) {
			(void)fprintf( stderr,
				"Internal operator check error at position %zd: %lf does not equal "
				"%lf.\n",
				i, chk[ i ], out[ i ] );
			error = 2;
		}
	}

	if( error )
		return error;

	typedef grb::operators::mul< double, int, float > public_op;

	(void)memcpy( out, inval, 15 * sizeof( float ) );
	public_op::eWiseApply( data1, data2, out, 15 );

	for( size_t i = 0; i < 15; ++i ) {
		if( err( out[ i ], chk[ i ], i ) ) {
			(void)fprintf( stderr,
				"Public operator (map) check error at position %zd: %lf does not "
				"equal %lf.\n",
				i, chk[ i ], out[ i ] );
			error = 2;
		}
	}

	if( error )
		return error;

	(void)memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; ! error && i < 15; ++i ) {
		const enum grb::RC rc = grb::apply< grb::descriptors::no_casting, public_op >( out[ i ], data1[ i ], data2[ i ] );
		if( rc != SUCCESS ) {
			(void)fprintf( stderr, "Public operator (apply) returns non-SUCCESS error code %d.\n", (int)rc );
			error = 4;
		}
		if( err( out[ i ], chk[ i ], i ) ) {
			(void)fprintf( stderr,
				"Public operator (apply) check error at position %zd: %lf does not "
				"equal %lf.\n",
				i, chk[ i ], out[ i ] );
			error = 3;
		}
	}

	if( error ) {
		(void)printf( "Test FAILED.\n\n" );
	} else {
		(void)printf( "Test OK.\n\n" );
	}

	return error;
}
