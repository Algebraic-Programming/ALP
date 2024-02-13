
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

#include "graphblas/utils/timer.hpp"

#include "graphblas.hpp"

#include "graphblas/bsp/collectives_blas1_raw.hpp"


using namespace grb;

constexpr size_t n = 12;
constexpr double pi = 3.14159;
constexpr size_t root = 0;

static bool same( double a, double b, double epsilon = 0.00001 ) {
	return std::abs( a - b ) < epsilon;
}

void grbProgram( const size_t &P, int &exit_status ) {
	const size_t s = spmd<>::pid();
	assert( P == spmd<>::nprocs() );
	assert( s < P );

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	enum RC rc = SUCCESS;
	const grb::operators::add< double, double, double > oper;
	double d;
	double v[ P ];
	double vLarge[ n ];
	double vLarger[ n * P ];

	// prep buffer
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::initCollectivesBuffer returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 10;
		return;
	}

	// gather: small
	d = pi * s;
	rc = grb::internal::gather( d, v, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::gather (small) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 20;
		return;
	}
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			if( v[ i ] != pi * i ) {
				std::cerr << "grb::internal::gather (smal) returns incorrect value ("
					<< v[ i ] << ") at index " << i << ".\n";
				exit_status = 30;
				return;
			}
		}
	}

	// gather: large
	for( size_t i = 0; i < n; i++ ) {
		vLarge[ i ] = pi * s + i;
	}
	rc = grb::internal::gather( vLarge, n, vLarger, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::gather (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 40;
		return;
	}
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			for( size_t j = 0; j < n; j++ ) {
				if( vLarger[ i * n + j ] != pi * i + j ) {
					std::cerr << "grb::internal::gather (large) returns incorrect value ("
						<< vLarger[ i * n + j ] << ") at index " << i << "," << j << ".\n";
					exit_status = 50;
					return;
				}
			}
		}
	}

	// allgather
	d = pi * s;
	rc = grb::internal::allgather( d, v );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::allgather returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 60;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( v[ i ] != pi * i ) {
			std::cerr << "grb::internal::allgather returns incorrect value (" << v[ i ]
				<< ") at index " << i << ".\n";
			exit_status = 70;
			return;
		}
	}

	// scatter: small
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			v[ i ] = pi * i;
		}
	}
	rc = grb::internal::scatter( v, d, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::scatter (small) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 80;
		return;
	}
	if( d != pi * s ) {
		std::cerr << "grb::internal::scatter (small) returns incorrect value (" << d
			<< ".\n";
		exit_status = 90;
		return;
	}

	// scatter: large
	if( s == root ) {
		for( size_t i = 0; i < n * P; i++ ) {
			vLarger[ i ] = pi * i;
		}
	}

	rc = grb::internal::scatter( vLarger, n * P, vLarge, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::scatter (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 100;
		return;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( vLarge[ i ] != ( s * n + i ) * pi ) {
			std::cerr << "grb::internal::scatter (large) returns incorrect value ("
				<< vLarge[ i ] << " at index " << i << ".\n";
			exit_status = 110;
			return;
		}
	}

	// alltoall
	for( size_t i = 0; i < P; i++ ) {
		v[ i ] = pi * i;
	}
	double out[ P ];
	rc = grb::internal::alltoall( v, out );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::alltoall returns bad error code ("
			<< grb::toString(rc) << ".\n";
		exit_status = 120;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( out[ i ] != pi * s ) {
			std::cerr << "grb::internal::alltoall returns incorrect value (" << out[ i ]
				<< ") at index " << i << ".\n";
			exit_status = 130;
			return;
		}
	}

	// allcombine
	for( size_t i = 0; i < P; i++ ) {
		v[ i ] = pi * i;
	}
	rc = grb::internal::allcombine( v, P, oper );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::allcombine returns bad error code ("
			<< grb::toString(rc) << ".\n";
		exit_status = 140;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( v[ i ] != pi * P * i ) {
			std::cerr << "grb::internal::allcombine returns incorrect value (" << v[ i ]
				<< ") at index " << i << ".\n";
			exit_status = 150;
			return;
		}
	}

	// combine: large
	for( size_t i = 0; i < n; i++ ) {
		vLarge[ i ] = pi * s + i;
	}
	rc = grb::internal::combine( vLarge, n, oper, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::combine (large) returns bad error code ("
			<< grb::toString(rc) << ".\n";
		exit_status = 160;
		return;
	}
	if( s == root ) {
		size_t sum = ( P - 1 ) * ( P / 2 );
		for( size_t i = 0; i < n; i++ ) {
			double val = sum * pi + i * P;
			if( !same( vLarge[ i ], val ) ) {
				std::cerr << "grb::internal::combine (large) returns incorrect value ("
					<< vLarge[ i ] << " at index " << i << ".\n";
				exit_status = 170;
				return;
			}
		}
	}

#if 0 // TODO FIXME DBG checking whether we can remove this functionality
	// reduce: large
	for( size_t i = 0; i < n; i++ ) {
		vLarge[ i ] = pi * s + i;
	}
	d = 0;
	rc = grb::internal::reduce( vLarge, n, d, oper, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::reduce (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 180;
		return;
	}
	if( s == root ) {
		size_t sum = ( P - 1 ) * ( P / 2 );
		double val = 0;
		for( size_t i = 0; i < n; i++ ) {
			val += sum * pi + i * P;
		}
		if( !same( d, val ) ) {
			std::cerr << "grb::internal::reduce (large) returns incorrect value ("
				<< d << ".\n";
			exit_status = 190;
			return;
		}
	}

	// allreduce: large
	for( size_t i = 0; i < n; i++ ) {
		vLarge[ i ] = pi * s + i;
	}
	d = 0;
	rc = grb::internal::allreduce( vLarge, n, d, oper );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::allreduce (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 200;
		return;
	}
	size_t sum = ( P - 1 ) * ( P / 2 );
	double val = 0;
	for( size_t i = 0; i < n; i++ ) {
		val += sum * pi + i * P;
	}
	if( !same( d, val ) ) {
		std::cerr << "grb::internal::allreduce (large) returns incorrect value ("
			<< d << ".\n";
		exit_status = 210;
		return;
	}
#endif

	// broadcast: large
	if( s == root ) {
		for( size_t i = 0; i < n; i++ ) {
			vLarge[ i ] = pi * s + i;
		}
	}
	rc = grb::internal::broadcast( vLarge, n, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::broadcast (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 220;
		return;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( vLarge[ i ] != pi * root + i ) {
			std::cerr << "grb::internal::broadcast (large) returns incorrect value ("
				<< vLarge[ i ] << ") at index " << i << ".\n";
			exit_status = 230;
			return;
		}
	}
}

