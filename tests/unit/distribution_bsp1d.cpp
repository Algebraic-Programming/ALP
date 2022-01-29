
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

#include "graphblas/bsp1d/distribution.hpp"
#include "graphblas/distribution.hpp"

using namespace grb;

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	int error = 0;

	if( internal::Distribution< BSP1D >::blocksize() != config::CACHE_LINE_SIZE::value() ) {
		(void)fprintf( stderr, "Unexpected value for blocksize (%zd, should be %zd).\n", internal::Distribution< BSP1D >::blocksize(), config::CACHE_LINE_SIZE::value() );
		error = 1;
	}

	constexpr size_t b = internal::Distribution< BSP1D >::blocksize();
	constexpr size_t n = 100000;
	constexpr size_t P = 4;

	if( internal::Distribution< BSP1D >::global_index_to_process_id( 0, n, P ) != 0 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_process_id at check 2.\n" );
		error = 2;
	}

	if( internal::Distribution< BSP1D >::global_index_to_process_id( b, n, P ) != 1 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_process_id at check 3.\n" );
		error = 3;
	}

	if( internal::Distribution< BSP1D >::global_index_to_process_id( 2 * b, n, P ) != 2 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_process_id at check 4.\n" );
		error = 4;
	}

	if( internal::Distribution< BSP1D >::global_index_to_process_id( 3 * b, n, P ) != 3 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_process_id at check 5.\n" );
		error = 5;
	}

	if( internal::Distribution< BSP1D >::global_index_to_process_id( n - 1, n, P ) != ( ( n / b ) % P ) ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_process_id at check 6.\n" );
		error = 6;
	}

	if( internal::Distribution< BSP1D >::global_index_to_local( 0, n, P ) != 0 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_local at check 7.\n" );
		error = 7;
	}

	if( internal::Distribution< BSP1D >::global_index_to_local( b, n, P ) != 0 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_local at check 8.\n" );
		error = 8;
	}

	if( internal::Distribution< BSP1D >::global_index_to_local( 2 * b, n, P ) != 0 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_local at check 9.\n" );
		error = 9;
	}

	if( internal::Distribution< BSP1D >::global_index_to_local( 3 * b, n, P ) != 0 ) {
		(void)fprintf( stderr, "Unexpected output from global_index_to_local at check 10.\n" );
		error = 10;
	}

	if( 2 * P * b + 17 < n && b > 17 && internal::Distribution< BSP1D >::global_index_to_local( P * b + 2 * b + 17, n, P ) != b + 17 ) {
		(void)fprintf( stderr,
			"Unexpected output from global_index_to_local at check 11. (Input: "
			"%zd, result: %zd, expected: %zd.)\n",
			P * b + 2 * b + 17, internal::Distribution< BSP1D >::global_index_to_local( P * b + 2 * b + 17, n, P ), b + 17 );
		error = 11;
	}

	for( size_t i = 0; i < n; ++i ) {
		const size_t lb_offset = internal::Distribution< BSP1D >::offset_to_pid( i, n, P );
		const size_t offset = internal::Distribution< BSP1D >::local_offset( n, lb_offset, P );
		if( offset > i ) {
			(void)fprintf( stderr,
				"Translating offset %zd to a PID yields %zd. The offset of PID "
				"%zd, however, is %zd.\n",
				i, lb_offset, lb_offset, offset );
			error = 12;
		}
		if( lb_offset + 1 < P ) {
			const size_t next_offset = internal::Distribution< BSP1D >::local_offset( n, lb_offset + 1, P );
			if( next_offset <= i ) {
				(void)fprintf( stderr,
					"Translating offset %zd to a PID yields %zd. The offset of PID "
					"%zd+1, however, is %zd.\n",
					i, lb_offset, lb_offset, next_offset );
				error = 13;
			}
		}
	}

	size_t localsize[ P ];
	for( size_t s = 0; s < P; ++s ) {
		localsize[ s ] = 0;
	}
	for( size_t i = 0; i < n; ++i ) {
		const size_t s = internal::Distribution< BSP1D >::global_index_to_process_id( i, n, P );
		++( localsize[ s ] );
	}
	for( size_t s = 0; s < P; ++s ) {
		if( localsize[ s ] != internal::Distribution< BSP1D >::global_length_to_local( n, s, P ) ) {
			error = static_cast< int >( 14 + s );
			(void)fprintf( stderr,
				"Unexpected output from global_length_to_local at check %d: for a "
				"vector of length %zd, PID %zd out of %zd has %zd elements "
				"(expected: %zd).\n",
				error, n, s, P, internal::Distribution< BSP1D >::global_length_to_local( n, s, P ), localsize[ s ] );
		}
	}

	localsize[ 0 ] = 0;
	for( size_t s = 0; s < P - 1; ++s ) {
		localsize[ s + 1 ] = internal::Distribution< BSP1D >::global_length_to_local( n, s, P );
	}
	for( size_t s = 1; s < P; ++s ) {
		localsize[ s ] += localsize[ s - 1 ];
	}
	for( size_t s = 0; s < P; ++s ) {
		if( localsize[ s ] != internal::Distribution< BSP1D >::local_offset( n, s, P ) ) {
			error = static_cast< int >( 14 + s + P );
			(void)fprintf( stderr,
				"Unexpected output from local_offset at check %d: for a vector of "
				"length %zd, PID %zd out of %zd has %zd preceding elements "
				"(expected: %zd)\n",
				error, n, s, P, internal::Distribution< BSP1D >::local_offset( n, s, P ), localsize[ s ] );
		}
	}

	if( ! error ) {
		(void)printf( "Test OK.\n\n" );
	}

	return error;
}
