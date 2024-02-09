
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

#include "graphblas/utils/timer.hpp"

#include "graphblas.hpp"


using namespace grb;

constexpr int n = 100000;

void grbProgram( const size_t &P, int &exit_status ) {
	const size_t s = spmd<>::pid();
	assert( P == spmd<>::nprocs() );
	assert( s < P );

	(void)s;
	(void)P;

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	grb::Vector< int > x( n ), y( n );
	enum RC return_code = SUCCESS;

	return_code = grb::set( x, 1 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (on x) returns bad error code (%d).\n", (int)return_code );
		exit_status = 1;
		return;
	}

	return_code = grb::set( y, 2 );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::set (on y) returns bad error code (%d).\n", (int)return_code );
		exit_status = 2;
		return;
	}

	int alpha = 0;
	return_code = grb::dot< grb::descriptors::no_operation, grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > >( alpha, x, y );
	if( return_code != SUCCESS ) {
		(void)fprintf( stderr, "grb::dot to calculate alpha = (x,y) returns bad error code (%d).\n", (int)return_code );
		exit_status = 3;
		return;
	}

	if( alpha != 2 * n ) {
		(void)fprintf( stderr, "Computed value by grb::dot (%d) does not equal expected value (%d).\n", alpha, 2 * n );
		exit_status = 4;
		return;
	}

	exit_status = 0;
}
