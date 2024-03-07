
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

#include <iostream>

#include "graphblas/utils.hpp"
#include "graphblas/utils/timer.hpp"

#include "graphblas.hpp"


using namespace grb;

constexpr double pi = 3.14159;
constexpr size_t root = 0;

void grbProgram( const size_t &P, int &exit_status ) {
	// assume all ok
	exit_status = 0;

	// preliminaries
	const size_t s = grb::spmd<>::pid();
	assert( P == grb::spmd<>::nprocs() );
	assert( s < P );
	grb::utils::Timer benchtimer;
	benchtimer.reset();
	enum RC rc = SUCCESS;
	const grb::operators::add< double, double, double > oper;
	double d = 0;

	grb::utils::Timer timer;
	timer.reset();

	// broadcast
	{
		d = 0;
		if( s == root ) {
			d = pi;
		}
		rc = grb::collectives<>::broadcast( d, root );
		if( rc != SUCCESS ) {
			std::cerr << "grb::collectives::broadcast returns bad error code: "
				<< grb::toString(rc) << "." << std::endl;
			exit_status = 10;
			return;
		}
		if( d != pi ) {
			std::cerr << "grb::collectives::broadcast returns incorrect value: " << d << ". "
				<< "Expected: " << pi << "." << std::endl;
			exit_status = 20;
			return;
		}
	}

	// reduce
	{
		d = pi;
		rc = grb::collectives<>::reduce( d, root, oper );
		if( rc != SUCCESS ) {
			std::cerr << "grb::collectives::reduce returns bad error code: "
				<< grb::toString(rc) << "." << std::endl;
			exit_status = 30;
			return;
		}
		if( s == root ) {
			// uses P instead of P-1 to survive the P=1 case
			if( !grb::utils::equals( d, pi * P, P ) ) {
				std::cerr << "grb::collectives::reduce returns incorrect value: "
					<< d << ". " << "Expected: " << (pi * P) << "." << std::endl;
				exit_status = 40;
				return;
			}
		}
	}

	// allreduce
	{
		d = pi;
		rc = grb::collectives<>::allreduce( d, oper );
		if( rc != SUCCESS ) {
			std::cerr << "grb::collectives::allreduce returns bad error code: "
				<< grb::toString(rc) << "." << std::endl;
			exit_status = 50;
			return;
		}
		if( !grb::utils::equals( d, pi * P, P ) ) { // used P instead of P-1 to survive P=1
			std::cerr << "grb::collectives::allreduce returns incorrect value: " << d
				<< ". Expected: " << (pi * P) << "." << std::endl;
			exit_status = 60;
			return;
		}
	}

	// all OK, return exit status zero
	return;
}

