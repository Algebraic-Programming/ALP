
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

static const double data1[ 15 ] = { 4.32, 7.43, 4.32, 6.54, 4.21, 7.65, 7.43, 7.54, 5.32, 6.43, 7.43, 5.42, 1.84, 5.32, 7.43 };
static const double data2[ 15 ] = { 8.49, 7.84, 8.49, 6.58, 8.91, 7.65, 7.84, 7.58, 5.49, 6.84, 7.84, 5.89, 1.88, 5.49, 7.84 };
static const double chk[ 15 ] = { 12.81, 15.27, 12.81, 13.12, 13.12, 15.30, 15.27, 15.12, 10.81, 13.27, 15.27, 11.31, 3.72, 10.81, 15.27 };
const static double inval[ 15 ] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

// for comments, see tests/add15m.cpp
static bool err( const double a, const double b ) {
	return !grb::utils::equals( a, b, 1 );
}

static inline void reportSanityCheckErrorPlus(
		const size_t i, const double x, const double y, const double z
) {
	std::cerr << "Sanity check error at position " << i << ": "
		<< x << " + " << y << " does not equal " << z << ".\n";
}

static inline void reportSanityCheckErrorEquals(
		const std::string str,
		const size_t i, const double x, const double y
) {
	std::cerr << str << " check error at position " << i << ": "
		<< x << " does not equal " << y << ".\n";
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	double out[ 15 ];
	int error = 0;

	for( size_t i = 0; i < 15; ++i ) {
		if( err( data1[ i ] + data2[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorPlus( i, data1[ i ], data2[ i ], chk[ i ] );
			error = 1;
		}
	}

	if( error ) {
		return error;
	}

	typedef grb::operators::internal::add< double, double, double > internal_op;

	(void)memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		internal_op::foldl( &( out[ i ] ), &( data1[ i ] ) );
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "::foldl", i, chk[ i ], out[ i ] );
			error = 2;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		out[ i ] = data2[ i ];
		internal_op::foldr( &( data1[ i ] ), &( out[ i ] ) );
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "::foldr", i, chk[ i ], out[ i ] );
			error = 3;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		internal_op::apply( &( data1[ i ] ), &( data2[ i ] ), &( out[ i ] ) );
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "::apply", i, chk[ i ], out[ i ] );
			error = 4;
		}
	}

	if( error ) {
		return error;
	}

	typedef grb::operators::add< double, double, double > PublicOp;
	PublicOp public_op;

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	PublicOp::eWiseApply( data1, data2, out, 15 );

	for( size_t i = 0; i < 15; ++i ) {
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "::eWiseApply", i, chk[ i ], out[ i ] );
			error = 5;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, data2, 15 * sizeof( double ) );
	PublicOp::eWiseFoldrAA( data1, out, 15 );

	for( size_t i = 0; i < 15; ++i ) {
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "::eWiseFoldrAA", i, chk[ i ], out[ i ] );
			error = 6;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, inval, 15 * sizeof( float ) );
	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::apply( out[ i ], data1[ i ], data2[ i ],
			public_op );
		if( rc != SUCCESS ) {
			std::cerr << "Public operator (out-of-place apply by argument) does not return "
				"SUCCESS, but rather " << grb::toString( rc ) << "\n";
		}
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "out-of-place apply by argument",
				i, chk[ i ], out[ i ] );
			error = 7;
		}
	}

	if( error ) {
		return error;
	}

	(void) memcpy( out, data2, 15 * sizeof( double ) );
	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::foldr( data1[ i ], out[ i ], public_op );
		if( rc != SUCCESS ) {
			std::cerr << "Public operator (in-place foldr) does not return SUCCESS, "
			       << "but rather " << grb::toString( rc ) << "\n";
		}
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "in-place foldr", i, chk[ i ], out[ i ] );
			error = 8;
		}
	}

	if( error ) {
		return error;
	}

	(void)memcpy( out, data2, 15 * sizeof( double ) );
	for( size_t i = 0; i < 15; ++i ) {
		const enum grb::RC rc = grb::foldl( out[ i ], data1[ i ], public_op );
		if( rc != SUCCESS ) {
			std::cerr << "Public operator (in-place foldl) does not return SUCCESS, "
			       << "but rather " << grb::toString( rc ) << "\n";
		}
		if( err( out[ i ], chk[ i ] ) ) {
			reportSanityCheckErrorEquals( "in-place foldl", i, chk[ i ], out[ i ] );
			error = 9;
		}
	}

	if( !error ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}

	return error;
}

