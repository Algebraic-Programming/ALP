
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

#include <stdio.h>

#include <string.h>

#include "graphblas.hpp"


using namespace grb;

int main( int argc, char ** argv ) {
	(void) argc;
	(void) printf( "Functional test executable: %s\n", argv[ 0 ] );

	const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
	const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
	const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };
	int out[ 15 ];
	int error = 0;

	for( size_t i = 0; i < 15; ++i ) {
		if( data1[ i ] * data2[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr, "Sanity check error at position %zd: %d * %d does not equal %d.\n", i, data1[ i ], data2[ i ], chk[ i ] );
			error = 1;
		}
	}

	if( error )
		return error;

	typedef grb::operators::internal::mul< int, int, int > internal_op;

	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		internal_op::foldr( &( data1[ i ] ), &( out[ i ] ) );
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Internal foldr check error at position %zd: %d does not equal "
				"%d.\n",
				i, chk[ i ], out[ i ] );
			error = 2;
		}
	}

	if( error )
		return error;
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		internal_op::foldl( &( out[ i ] ), &( data1[ i ] ) );
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Internal foldl check error at position %zd: %d does not equal "
				"%d.\n",
				i, chk[ i ], out[ i ] );
			error = 3;
		}
	}

	if( error )
		return error;

	for( size_t i = 0; i < 15; ++i ) {
		internal_op::apply( &( data1[ i ] ), &( data2[ i ] ), &( out[ i ] ) );
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Internal operator check error at position %zd: %d does not equal "
				"%d.\n",
				i, chk[ i ], out[ i ] );
			error = 4;
		}
	}

	if( error )
		return error;

	typedef grb::operators::mul< int, int, int > public_op;

	public_op::eWiseApply( data1, data2, out, 15 );

	for( size_t i = 0; i < 15; ++i ) {
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Public operator (map) check error at position %zd: %d does not "
				"equal %d.\n",
				i, chk[ i ], out[ i ] );
			error = 5;
		}
	}

	if( error )
		return error;

	(void)memcpy( out, data2, 15 * sizeof( int ) );
	public_op::eWiseFoldrAA( data1, out, 15 );

	for( size_t i = 0; i < 15; ++i ) {
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Public operator (mapInto) check error at position %zd: %d does "
				"not equal %d.\n",
				i, chk[ i ], out[ i ] );
			error = 6;
		}
	}

	if( error )
		return error;

	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::apply< grb::descriptors::no_casting, public_op >( out[ i ], data1[ i ], data2[ i ] );
		if( rc != SUCCESS ) {
			(void)fprintf( stderr, "Public operator (apply) returns non-SUCCESS exit code %d.\n", (int)rc );
		}
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Public operator (apply) check error at position %zd: %d does not "
				"equal %d.\n",
				i, chk[ i ], out[ i ] );
			error = 7;
		}
	}

	if( error )
		return error;

	(void)memcpy( out, data2, 15 * sizeof( int ) );
	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::foldr< grb::descriptors::no_casting, public_op >( data1[ i ], out[ i ] );
		if( rc != SUCCESS ) {
			(void)fprintf( stderr, "Public operator (foldr) returns non-SUCCESS exit code %d.\n", (int)rc );
		}
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Public operator (foldr) check error at position %zd: %d does not "
				"equal %d.\n",
				i, chk[ i ], out[ i ] );
			error = 8;
		}
	}
	if( error )
		return error;

	(void)memcpy( out, data2, 15 * sizeof( int ) );
	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::foldl< grb::descriptors::no_casting, public_op >( out[ i ], data1[ i ] );
		if( rc != SUCCESS ) {
			(void)fprintf( stderr, "Public operator (foldl) returns non-SUCCESS exit code %d.\n", (int)rc );
		}
		if( out[ i ] - chk[ i ] > 1e-14 ) {
			(void)fprintf( stderr,
				"Public operator (foldl) check error at position %zd: %d does not "
				"equal %d.\n",
				i, chk[ i ], out[ i ] );
			error = 9;
		}
	}

	if( ! error ) {
		(void)printf( "Test OK.\n\n" );
	}

	return error;
}
