
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

const static float EPS = std::numeric_limits< float >::epsilon();

static const double data1[ 15 ] = { 4.32, 7.43, 4.32, 6.54, 4.21, 7.65, 7.43, 7.54, 5.32, 6.43, 7.43, 5.42, 1.84, EPS / 2.0, 2 * EPS };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const float chk[ 15 ] = { 12.32, 16.43, 12.32, 12.54, 12.21, 14.65, 15.43, 14.54, 10.32, 8.43, 10.43, 10.42, 2.84, 5.0, static_cast< float >( 5.0 + 2 * EPS ) };
const static double inval[ 15 ] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static bool err( const float a, const float b ) {
	return !grb::utils::equals( a, b, 1 );
}

int main( int argc, char ** argv ) {
	(void) argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	float out[ 15 ];
	int error = 0;

	for( size_t i = 0; i < 15; ++i ) {
		if( err( data1[ i ] + data2[ i ], chk[ i ] ) ) {
			std::cerr << "Sanity check error at position " << i << ": " << data1[ i ]
				<< " + " << data2[ i ] << " does not equal " << chk[ i ] << ".\n";
			error = 1;
		}
	}

	if( error ) {
		return error;
	}

	typedef grb::operators::internal::add< double, int, float > internal_op;

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		internal_op::foldr( &( data1[ i ] ), &( out[ i ] ) );
		if( err( out[ i ], chk[ i ] ) ) {
			std::cerr << "Internal foldr check error at position " << i << ": "
				<< chk[ i ] << " does not equal " << out[ i ] << ".\n";
			error = 2;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data1[ i ];
		internal_op::foldl( &( out[ i ] ), &( data2[ i ] ) );
		if( err( out[ i ], chk[ i ] ) ) {
			std::cerr << "Internal foldl check error at position " << i << ": "
				<< chk[ i ] << " does not equal " << out[ i ] << ".\n";
			error = 3;
		}
	}

	if( error ) {
		return error;
	}

	typedef grb::operators::add< double, int, float > public_op;

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	public_op::eWiseApply( data1, data2, out, 15 );

	for( size_t i = 0; i < 15; ++i ) {
		if( err( out[ i ], chk[ i ] ) ) {
			std::cerr << "Public operator (apply) check error at position " << i << ": "
				<< chk[ i ] << " does not equal " << out[ i ] << ".\n";
			error = 4;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::apply< grb::descriptors::no_casting, public_op >(
			out[ i ], data1[ i ], data2[ i ] );
		if( rc != SUCCESS ) {
			std::cerr << "Public operator (element-by-element apply) does not return "
				<< "SUCCESS, but rather " << grb::toString( rc ) << "\n";
		}
		if( err( out[ i ], chk[ i ] ) ) {
			std::cerr << "Public operator (element-by-element apply) check error at "
				<< "position " << i << ": " << chk[ i ] << " does not equal " << out[ i ]
				<< ".\n";
			error = 5;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		// note that passing no_casting to the below should result in a compilation error
		const enum grb::RC rc = grb::foldr<
			grb::descriptors::no_operation, public_op
		>( data1[ i ], out[ i ] );
		if( rc != SUCCESS ) {
			std::cerr << "Public operator (element-by-element foldr) does not return "
				<< "SUCCESS, but rather " << grb::toString( rc ) << "\n";
		}
		if( err( out[ i ], chk[ i ] ) ) {
			std::cerr << "Public operator (element-by-element foldr) check error at "
				<< "position " << i << ": " << chk[ i ] << " does not equal " << out[ i ]
				<< ".\n";
			error = 6;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data1[ i ];
		// note that passing no_casting to the below should result in a compilation error
		const enum grb::RC rc = grb::foldl<
			grb::descriptors::no_operation, public_op
		>( out[ i ], data2[ i ] );
		if( rc != SUCCESS ) {
			std::cerr << "Public operator (element-by-element foldl) does not return "
				<< "SUCCESS, but rather " << grb::toString( rc ) << "\n";
		}
		if( err( out[ i ], chk[ i ] ) ) {
			std::cerr << "Public operator (element-by-element foldl) check error at "
				<< "position " << i << ": " << chk[ i ] << " does not equal " << out[ i ]
				<< ".\n";
			error = 7;
		}
	}

	// done
	if( !error ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}
	return error;
}

