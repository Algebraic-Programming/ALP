
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

#include "graphblas.hpp"

using namespace grb;

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );
	bool error = ! ( grb::internal::Distribution< BSP1D >::blocksize() > 0 );

	constexpr const size_t n = 10000000;
	for( size_t P = 1; ! error && P < 10; ++P ) {
		size_t offset = 0;
		for( size_t s = 0; ! error && s < P; ++s ) {
			error = ! ( offset == grb::internal::Distribution< BSP1D >::local_offset( n, s, P ) );
			if( error ) {
				(void)printf( "Error in grb::internal::Distribution< BSP1D "
							  ">::local_offset( n, s, P ) for n = %zd, s = "
							  "%zd, and P = %zd\n",
					n, s, P );
			}
			const size_t local_n = grb::internal::Distribution< BSP1D >::global_length_to_local( n, s, P );
			offset += local_n;
			if( ! error ) {
				error = ! ( local_n <= n );
				if( error ) {
					(void)printf( "Error in grb::internal::Distribution< BSP1D "
								  ">::global_length_to_local( n, s, P ) for n "
								  "= %zd, s = %zd, and P = %zd\n",
						n, s, P );
				}
			}
		}
		error = ! ( offset == n );
		if( error ) {
			(void)printf( "Sum of grb::internal::Distribution< BSP1D "
						  ">::local_offset calls (%zd) do not equal n (%zd)\n",
				offset, n );
		}
		for( size_t global_i = 0; ! error && global_i < n; ++global_i ) {
			const size_t dst_pid = grb::internal::Distribution< BSP1D >::global_index_to_process_id( global_i, n, P );
			const size_t dst_i = grb::internal::Distribution< BSP1D >::global_index_to_local( global_i, n, P );
			error = ! ( grb::internal::Distribution< BSP1D >::global_length_to_local( n, dst_pid, P ) > dst_i );
			if( error ) {
				(void)printf( "Local index %zd is larger or equal than local "
							  "length %zd\n",
					dst_i, grb::internal::Distribution< BSP1D >::global_length_to_local( n, dst_pid, P ) );
			}
			if( ! error ) {
				error = ! ( grb::internal::Distribution< BSP1D >::local_index_to_global( dst_i, n, dst_pid, P ) == global_i );
				if( error ) {
					(void)printf( "Local index %zd does not translate "
								  "correctly to global index: "
								  "grb::internal::Distribution< BSP1D "
								  ">::local_index_to_global( dst_i, n, "
								  "dst_pid, P ) = %zd for n = %zd, dst_pid = "
								  "%zd, and P = %zd\n",
						dst_i, grb::internal::Distribution< BSP1D >::local_index_to_global( dst_i, n, dst_pid, P ), n, dst_pid, P );
				}
			}
		}
		if( ! error ) {
			error = ! ( grb::internal::Distribution< BSP1D >::local_index_to_global( 0, n, 0, P ) == 0 );
			if( error ) {
				(void)printf( "0-th local index at PID 0 does not translate to "
							  "global index 0 for n = %zd and P = %zd\n",
					n, P );
			}
		}
		if( ! error ) {
			error = ! ( grb::internal::Distribution< BSP1D >::local_index_to_global( grb::internal::Distribution< BSP1D >::blocksize() - 1, n, 0, P ) ==
				grb::internal::Distribution< BSP1D >::blocksize() - 1 );
			if( error ) {
				(void)printf( "(b-1)-th local index at PID 0 does not "
							  "translate to global index (b-1) for b = %zd, n "
							  "= %zd, and P = %zd\n",
					grb::internal::Distribution< BSP1D >::blocksize(), n, P );
			}
		}
		if( ! error && P > 1 ) {
			error = ! (
				grb::internal::Distribution< BSP1D >::local_index_to_global( grb::internal::Distribution< BSP1D >::blocksize(), n, 0, P ) == P * grb::internal::Distribution< BSP1D >::blocksize() );
			if( error ) {
				(void)printf( "b-th local index at PID 0 does not translate to "
							  "P*b-th global index for b = %zd, n = %zd, P = "
							  "%zd\n",
					grb::internal::Distribution< BSP1D >::blocksize(), n, P );
			}
		}
	}

	if( ! error ) {
		(void)printf( "Test OK.\n\n" );
		return 0;
	} else {
		(void)printf( "Test FAILED.\n\n" );
		return 255;
	}
}
