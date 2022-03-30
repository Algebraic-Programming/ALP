
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

#include "graphblas/bsp/collectives_blas1_vec.hpp"
#include "graphblas/utils/Timer.hpp"

#include "graphblas.hpp"

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
	grb::Vector< double, reference > v( P );
	grb::Vector< double, reference > vLarge( n );
	grb::Vector< double, reference > vLarger( n * P );

	// prep buffer
	rc = grb::internal::template initCollectivesBuffer< double >( n * P );
	if( rc != SUCCESS ) {
		(void) fprintf( stderr, "grb::internal::initCollectivesBuffer returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}

	// gather: small
	d = pi * s;
	rc = grb::internal::gather( d, v, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::gather (small) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			if( v.raw()[ i ] != pi * i ) {
				(void)fprintf( stderr,
					"grb::internal::gather (small) returns incorrect value (%lf) at "
					"index %d.\n",
					v.raw()[ i ], (int)i );
				exit_status = 1;
				return;
			}
		}
	}

	// gather: large
	(void) grb::set< grb::descriptors::use_index >( vLarge, 0 );
	(void) grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	rc = grb::internal::gather( vLarge, vLarger, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::gather (large) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			for( size_t j = 0; j < n; j++ ) {
				if( vLarger.raw()[ i * n + j ] != pi * i + j ) {
					(void)fprintf( stderr,
						"grb::internal::gather (large) returns incorrect value "
						"(%lf) at index %d,%d.\n",
						vLarger.raw()[ i * n + j ], (int)i, (int)j );
					exit_status = 1;
					return;
				}
			}
		}
	}

	// allgather
	d = pi * s;
	rc = grb::internal::allgather( d, v );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::allgather returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( v.raw()[ i ] != pi * i ) {
			(void)fprintf( stderr,
				"grb::internal::allgather returns incorrect value (%lf) at index "
				"%d.\n",
				v.raw()[ i ], (int)i );
			exit_status = 1;
			return;
		}
	}

	// scatter: small
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			(void)grb::setElement( v, pi * i, i );
		}
	}
	rc = grb::internal::scatter( v, d, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::scatter (small) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	if( d != pi * s ) {
		(void)fprintf( stderr, "grb::internal::scatter (small) returns incorrect value (%lf).\n", d );
		exit_status = 1;
		return;
	}

	// scatter: large
	if( s == root ) {
		(void) grb::set< grb::descriptors::use_index >( vLarger, 0 );
		(void) grb::foldr( pi, vLarger, grb::operators::mul< double >() );
	}

	rc = grb::internal::scatter( vLarger, vLarge, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::scatter (large) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( vLarge.raw()[ i ] != ( s * n + i ) * pi ) {
			(void)fprintf( stderr,
				"grb::internal::scatter (large) returns incorrect value (%lf) at "
				"index %d.\n",
				vLarge.raw()[ i ], (int)i );
			exit_status = 1;
			return;
		}
	}

	// alltoall
	for( size_t i = 0; i < P; i++ ) {
		(void)grb::setElement( v, pi * i, i );
	}
	grb::Vector< double, reference > out( P );
	rc = grb::internal::alltoall( v, out );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::alltoall returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( out.raw()[ i ] != pi * s ) {
			(void)fprintf( stderr,
				"grb::internal::alltoall returns incorrect value (%lf) at index "
				"%d.\n",
				out.raw()[ i ], (int)i );
			exit_status = 1;
			return;
		}
	}

	// allcombine
	for( size_t i = 0; i < P; i++ ) {
		(void)grb::setElement( v, pi * i, i );
	}
	rc = grb::internal::allcombine( v, oper );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::allcombine returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( v.raw()[ i ] != pi * P * i ) {
			(void)fprintf( stderr,
				"grb::internal::allcombine returns incorrect value (%lf) at index "
				"%d.\n",
				v.raw()[ i ], (int)i );
			exit_status = 1;
			return;
		}
	}

	// combine: large
	(void) grb::set< grb::descriptors::use_index >( vLarge, 0 );
	(void) grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	rc = grb::internal::combine( vLarge, oper, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::combine (large) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	if( s == root ) {
		size_t sum = ( P - 1 ) * ( P / 2 );
		for( size_t i = 0; i < n; i++ ) {
			double val = sum * pi + i * P;
			if( ! same( vLarge.raw()[ i ], val ) ) {
				(void)fprintf( stderr,
					"grb::internal::combine (large) returns incorrect value (%lf) "
					"at index %d.\n",
					vLarge.raw()[ i ], (int)i );
				exit_status = 1;
				return;
			}
		}
	}

	// reduce: large
	(void) grb::set< grb::descriptors::use_index >( vLarge, 0 );
	(void) grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	d = 0;
	rc = grb::internal::reduce( vLarge, d, oper, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::reduce (large) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	if( s == root ) {
		size_t sum = ( P - 1 ) * ( P / 2 );
		double val = 0;
		for( size_t i = 0; i < n; i++ ) {
			val += sum * pi + i * P;
		}
		if( ! same( d, val ) ) {
			(void)fprintf( stderr, "grb::internal::reduce (large) returns incorrect value (%lf).\n", d );
			exit_status = 1;
			return;
		}
	}

	// allreduce: large
	(void) grb::set< grb::descriptors::use_index >( vLarge, 0 );
	(void) grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	d = 0;
	rc = grb::internal::allreduce( vLarge, d, oper );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::allreduce (large) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	size_t sum = ( P - 1 ) * ( P / 2 );
	double val = 0;
	for( size_t i = 0; i < n; i++ ) {
		val += sum * pi + i * P;
	}
	if( ! same( d, val ) ) {
		(void)fprintf( stderr, "grb::internal::allreduce (large) returns incorrect value (%lf).\n", d );
		exit_status = 1;
		return;
	}

	// broadcast: large
	if( s == root ) {
		(void) grb::set< grb::descriptors::use_index >( vLarge, 0 );
		(void) grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	}
	rc = grb::internal::broadcast( vLarge, root );
	if( rc != SUCCESS ) {
		(void)fprintf( stderr, "grb::internal::broadcast (large) returns bad error code (%d).\n", (int)rc );
		exit_status = 1;
		return;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( vLarge.raw()[ i ] != pi * root + i ) {
			(void)fprintf( stderr,
				"grb::internal::broadcast (large) returns incorrect value (%lf) at "
				"index %d.\n",
				vLarge.raw()[ i ], (int)i );
			exit_status = 1;
			return;
		}
	}
}

