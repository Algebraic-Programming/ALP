
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

#include "graphblas/utils/timer.hpp"

#include "graphblas.hpp"


using namespace grb;

constexpr size_t n = 12;
constexpr double pi = 3.14159;
constexpr size_t root = 0;

static bool same( double a, double b, double epsilon = 0.00001 ) {
	return std::abs( a - b ) < epsilon;
}

void grbProgram( const size_t &P, int &exit_status ) {
	grb::operators::any_or< int > reduceRC;
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

	// gather: small
	std::cout << "\ttesting gather (small)" << std::endl;
	d = pi * s;
	rc = grb::internal::gather( d, v, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::gather (small) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 10;
		return;
	}
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			if( v.raw()[ i ] != pi * i ) {
				std::cerr << "grb::internal::gather (small) returns incorrect value ("
					<< v.raw()[ i ] << ") at index " << i << ".\n";
				exit_status = 20;
			}
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 25;
	}
	if( exit_status != 0 ) { return; }

	// gather: large
	std::cout << "\ttesting gather (large)" << std::endl;
	assert( rc == SUCCESS );
	rc = grb::set< grb::descriptors::use_index >( vLarge, 0 );
	rc = rc ? rc : grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise gather (large) test\n";
		exit_status = 28;
		return;
	}
	rc = grb::internal::gather( vLarge, vLarger, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::gather (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 30;
		return;
	}
	if( s == root ) {
		for( size_t i = 0; i < P; i++ ) {
			for( size_t j = 0; j < n; j++ ) {
				if( vLarger.raw()[ i * n + j ] != pi * i + j ) {
					std::cerr << "grb::internal::gather (large) returns incorrect value ("
						<< vLarger.raw()[ i * n + j ] << ") at index " << i << "," << j << ".\n";
					exit_status = 40;
				}
			}
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 45;
	}
	if( exit_status != 0 ) { return; }

	// allgather
	std::cout << "\ttesting allgather" << std::endl;
	d = pi * s;
	rc = grb::internal::allgather( d, v );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::allgather returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 50;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( v.raw()[ i ] != pi * i ) {
			std::cerr << "grb::internal::allgather returns incorrect value ("
				<< v.raw()[ i ] << ") at index " << i << ".\n";
			exit_status = 60;
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 65;
	}
	if( exit_status != 0 ) { return; }

	// scatter: small
	std::cout << "\ttesting scatter (small)" << std::endl;
	if( s == root ) {
		for( size_t i = 0; rc == SUCCESS && i < P; i++ ) {
			rc = grb::setElement( v, pi * i, i );
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise scatter (small) test\n";
		exit_status = 68;
		return;
	}
	rc = grb::internal::scatter( v, d, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::scatter (small) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 70;
		return;
	}
	if( d != pi * s ) {
		std::cerr << "grb::internal::scatter (small) returns incorrect value (" << d
			<< ").\n";
		exit_status = 80;
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 85;
	}
	if( exit_status != 0 ) { return; }


	// scatter: large
	std::cout << "\ttesting scatter (large)" << std::endl;
	if( s == root && rc == SUCCESS ) {
		rc = grb::set< grb::descriptors::use_index >( vLarger, 0 );
		rc = rc ? rc : grb::foldr( pi, vLarger, grb::operators::mul< double >() );
	}
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise scatter (large) test\n";
		exit_status = 88;
		return;
	}
	rc = grb::internal::scatter( vLarger, vLarge, root );
	if( rc != SUCCESS ) {
		std::cerr <<  "grb::internal::scatter (large) returns bad error code ("
			<< grb::toString(rc) << ".\n";
		exit_status = 90;
		return;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( vLarge.raw()[ i ] != ( s * n + i ) * pi ) {
			std::cerr << "grb::internal::scatter (large) returns incorrect value ("
				<< vLarge.raw()[ i ] << ") at index " << i << ".\n";
			exit_status = 100;
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 105;
	}
	if( exit_status != 0 ) { return; }

	// alltoall
	std::cout << "\ttesting alltoall" << std::endl;
	grb::Vector< double, reference > out( P );
	for( size_t i = 0; rc == SUCCESS && i < P; i++ ) {
		rc = grb::setElement( v, pi * i, i );
	}
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise alltoall test\n";
		exit_status = 108;
		return;
	}
	rc = grb::internal::alltoall( v, out );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::alltoall returns bad error code ("
			<< grb::toString(rc) << ".\n";
		exit_status = 110;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( out.raw()[ i ] != pi * s ) {
			std::cerr << "grb::internal::alltoall returns incorrect value ("
				<< out.raw()[ i ] << ") at index " << i << ".\n";
			exit_status = 120;
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 125;
	}
	if( exit_status != 0 ) { return; }

	// allcombine
	std::cout << "\ttesting allcombine" << std::endl;
	for( size_t i = 0; rc == SUCCESS && i < P; i++ ) {
		rc = grb::setElement( v, pi * i, i );
	}
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise allcombine test\n";
		exit_status = 128;
		return;
	}
	rc = grb::internal::allcombine( v, oper );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::allcombine returns bad error code ("
			<< grb::toString(rc) << ".\n";
		exit_status = 130;
		return;
	}
	for( size_t i = 0; i < P; i++ ) {
		if( v.raw()[ i ] != pi * P * i ) {
			std::cerr << "grb::internal::allcombine returns incorrect value ("
				<< v.raw()[ i ] << ") at index " << i << ".\n";
			exit_status = 140;
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 145;
	}
	if( exit_status != 0 ) { return; }

	// combine: large
	std::cout << "\ttesting combine (large)" << std::endl;
	assert( rc == SUCCESS );
	rc = grb::set< grb::descriptors::use_index >( vLarge, 0 );
	rc = rc ? rc : grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise combine (large) test\n";
		exit_status = 148;
		return;
	}
	rc = grb::internal::combine( vLarge, oper, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::combine (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 150;
		return;
	}
	if( s == root ) {
		size_t sum = ( P - 1 ) * ( P / 2 );
		for( size_t i = 0; i < n; i++ ) {
			double val = sum * pi + i * P;
			if( !same( vLarge.raw()[ i ], val ) ) {
				std::cerr << "grb::internal::combine (large) returns incorrect value ("
					<< vLarge.raw()[ i ] << ") at index " << i << ".\n";
				exit_status = 160;
			}
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 165;
	}
	if( exit_status != 0 ) { return; }

	// broadcast: large
	std::cout << "\ttesting broadcast (large)" << std::endl;
	if( s == root ) {
		assert( rc == SUCCESS );
		rc = grb::set< grb::descriptors::use_index >( vLarge, 0 );
		rc = rc ? rc : grb::foldr( pi * s, vLarge, grb::operators::add< double >() );
	}
	if( rc != SUCCESS ) {
		std::cerr << "Could not initialise broadcast (large) test\n";
		exit_status = 168;
		return;
	}
	rc = grb::internal::broadcast( vLarge, root );
	if( rc != SUCCESS ) {
		std::cerr << "grb::internal::broadcast (large) returns bad error code ("
			<< grb::toString(rc) << ").\n";
		exit_status = 170;
		return;
	}
	for( size_t i = 0; i < n; i++ ) {
		if( vLarge.raw()[ i ] != pi * root + i ) {
			std::cerr << "grb::internal::broadcast (large) returns incorrect value ("
				<< vLarge.raw()[ i ] << ") at index " << i << ".\n";
			exit_status = 180;
		}
	}
	if( collectives<>::allreduce( exit_status, reduceRC ) != SUCCESS ) {
		exit_status = 185;
	}
	if( exit_status != 0 ) { return; }

}

