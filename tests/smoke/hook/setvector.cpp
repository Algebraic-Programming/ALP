
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
#include <assert.h>

#include "graphblas/bsp1d/distribution.hpp"

#include "graphblas/utils/timer.hpp"

#include "graphblas.hpp"


using namespace grb;

constexpr size_t n = 100;

static bool check(
	const size_t s, const size_t P,
	const size_t global_index,
	const double * __restrict__ const parVec,
	const double equal_to,
	double &read,
	size_t &local_index
) {
	if( internal::Distribution<>::global_index_to_process_id( global_index, n, P ) == s ) {
		local_index = internal::Distribution<>::global_index_to_local( global_index, n, P );
		read = parVec[ local_index ];
		if( read != equal_to ) {
			return false;
		}
	}
	return true;
}

void grbProgram( const size_t &P, int &exit_status ) {
	const size_t s = spmd<>::pid();
	assert( P == spmd<>::nprocs() );
	assert( s < P );

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	grb::Vector< double > parVec( n ), test( n );
	enum RC return_code = SUCCESS;

	return_code = grb::set( parVec, 5 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set returns bad error code (%d).\n", (int)return_code );
		exit_status = 1;
		return;
	}

	const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
	for( size_t i = 0; i < internal::Distribution< BSP1D >::global_length_to_local( n, data.s, data.P ); ++i ) {
		if( parVec.raw()[ i ] != 5 ) {
			(void)fprintf( stderr, "raw vector component (%lf) is not equal to expected value (5).\n", parVec.raw()[ i ] );
			exit_status = 2;
			return;
		}
	}

	for( size_t i = 0; i < n; ++i ) {
		return_code = grb::setElement( parVec, i, i );
		if( return_code != SUCCESS ) {
			(void)fprintf( stderr, "grb::set (per element)  returns bad error code (%d).\n", (int)return_code );
			exit_status = 3;
			return;
		}
	}

	return_code = grb::setElement( parVec, 3.1415926535, 64 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (at element 64) returns bad error code (%d).\n", (int)return_code );
		exit_status = 4;
		return;
	}

	return_code = grb::setElement( parVec, 3.3, 11 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (at element 11) returns bad error code (%d).\n", (int)return_code );
		exit_status = 5;
		return;
	}

	return_code = grb::set( test, parVec );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (copy) returns bad error code (%d).\n", (int)return_code );
		exit_status = 6;
		return;
	}

	double read = 0;
	size_t local_index = 0;
	for( size_t i = 0; i < n; ++i ) {
		if( i == 64 && ! check( s, P, 64, parVec.raw(), 3.1415926535, read, local_index ) ) {
			(void)fprintf( stderr,
				"raw vector component (%lf) at index (%zd) is not equal to "
				"expected value (3.1415926535).\n",
				read, local_index );
			exit_status = 6;
			return;
		}
		if( i == 11 && ! check( s, P, 11, parVec.raw(), 3.3, read, local_index ) ) {
			(void)fprintf( stderr,
				"raw vector component (%lf) at index (%zd) is not equal to "
				"expected value (3.3).\n",
				read, local_index );
			exit_status = 7;
			return;
		}
		if( i != 64 && i != 11 && ! check( s, P, i, parVec.raw(), i, read, local_index ) ) {
			(void)fprintf( stderr,
				"raw vector component (%lf) at index (%zd) is not equal to "
				"expected value (%zd).\n",
				read, local_index, i );
			exit_status = 8;
			return;
		}
		if( i == 64 && ! check( s, P, 64, test.raw(), 3.1415926535, read, local_index ) ) {
			(void)fprintf( stderr,
				"copied vector component (%lf) at index (%zd) is not equal to "
				"expected value (3.1415926535).\n",
				read, local_index );
			exit_status = 6;
			return;
		}
		if( i == 11 && ! check( s, P, 11, test.raw(), 3.3, read, local_index ) ) {
			(void)fprintf( stderr,
				"copied vector component (%lf) at index (%zd) is not equal to "
				"expected value (3.3).\n",
				read, local_index );
			exit_status = 7;
			return;
		}
		if( i != 64 && i != 11 && ! check( s, P, i, test.raw(), i, read, local_index ) ) {
			(void)fprintf( stderr,
				"copied vector component (%lf) at index (%zd) is not equal to "
				"expected value (%zd).\n",
				read, local_index, i );
			exit_status = 8;
			return;
		}
	}

	/*for( size_t i = 0; i < ((n/b+1)/3+1)*config::CACHE_LINE_SIZE::value(); ++i ) {
	    printf( "%ld @ process %ld = %lf\n", i, s, parVec.raw()[ i ] );
	}*/

	exit_status = 0;
}
