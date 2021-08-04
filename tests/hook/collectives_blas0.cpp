
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

#include "graphblas/utils/Timer.hpp"

#include "graphblas.hpp"

using namespace grb;

constexpr size_t n = 100000;
constexpr double pi = 3.14159;
constexpr size_t root = 0;

void grbProgram( const size_t s, const size_t P, int & exit_status ) {

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	enum RC rc = SUCCESS;
	const grb::operators::add< double, double, double > oper = grb::operators::add< double, double, double >();
	double d;

	grb::utils::Timer timer;
	timer.reset();

	// broadcast
	d = 0;
	if( s == root ) {
		d = pi;
	}
	rc = grb::collectives<>::broadcast( d, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::collectives::broadcast returns bad error code (%d).\n", (int)rc );
		goto fail;
	}
	if( d != pi ) {
		(void)fprintf( stderr, "grb::collectives::broadcast returns incorrect value (%lf).\n", d );
		goto fail;
	}

	// reduce
	d = pi;
	rc = grb::collectives<>::reduce( d, root, oper );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::collectives::reduce returns bad error code (%d).\n", (int)rc );
		goto fail;
	}
	if( s == root ) {
		if( d != pi * P ) {
			(void)fprintf( stderr, "grb::collectives::reduce returns incorrect value (%lf).\n", d );
			goto fail;
		}
	}

	// allreduce
	d = pi;
	rc = grb::collectives<>::allreduce( d, oper );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::collectives::allreduce returns bad error code (%d).\n", (int)rc );
		goto fail;
	}
	if( d != pi * P ) {
		(void)fprintf( stderr, "grb::collectives::allreduce returns incorrect value (%lf).\n", d );
		goto fail;
	}

	// all OK, return exit status zero
	exit_status = 0;
	return;

fail:

	exit_status = 1;
	(void)fflush( stderr );
}
