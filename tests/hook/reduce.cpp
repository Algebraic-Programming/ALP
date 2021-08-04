
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
constexpr size_t rep = 10;

void grbProgram( const size_t s, const size_t P, int & exit_status ) {

	(void)s;
	(void)P;

#ifdef _DEBUG
	(void)printf( "grbProgram (reduce) entered with parameters: %zd %zd %d\n", s, P, exit_status );
#endif

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	typename grb::Monoid< grb::operators::add< double >, grb::identities::zero > realm;
	typedef grb::Vector< double > vector;

	vector xv( n );
	double check = 0.0;
	double * __restrict__ xr = NULL;

	int rc = posix_memalign( (void **)&xr, grb::config::CACHE_LINE_SIZE::value(), n * sizeof( double ) );
	if( rc != 0 ) {
		assert( false );
		exit_status = 1;
		return;
	}
#ifdef NDEBUG
	(void)rc;
#endif

	for( size_t i = 0; i < n; ++i ) {
		if( grb::setElement< grb::descriptors::no_operation >( xv, (double)i, i ) != grb::SUCCESS ) {
			assert( false );
			exit_status = 2;
			return;
		}
		xr[ i ] = (double)i;
		check += (double)i;
	}

#ifdef _DEBUG
	(void)printf( "grbProgram (reduce) vector allocs of size %zd & "
				  "initialisations complete\n",
		n );
	fflush( stdout );
#endif

	bool error = false;
	double alpha = 0.0, beta = 0.0;
#ifdef __clang__
	// Otherwise the reference implementation hits this bug: https://llvm.org/bugs/show_bug.cgi?id=10856)
	if( grb::foldl< grb::descriptors::no_operation >( alpha, xv, NO_MASK, realm ) != grb::SUCCESS ) {
		assert( false );
		exit_status = 3;
		return;
	}
#else
	if( grb::foldl( alpha, xv, NO_MASK, realm ) != grb::SUCCESS ) {
		assert( false );
		exit_status = 3;
		return;
	}
#endif

#ifdef _DEBUG
	(void)printf( "grbProgram (reduce) post-foldl\n" );
	fflush( stdout );
#endif

	for( size_t i = 0; i < n; ++i ) {
		beta += xr[ i ];
	}
	if( ! grb::utils::equals( check, alpha, static_cast< double >( n - 1 ) ) ) {
		(void)printf( "%lf (templated) does not equal %lf (sequential).\n", alpha, check );
		error = true;
	}
	if( ! grb::utils::equals( check, beta, static_cast< double >( n ) ) ) {
		(void)printf( "%lf (compiler) does not equal %lf (sequential).\n", beta, check );
		error = true;
	}
	if( ! grb::utils::equals( alpha, beta, static_cast< double >( n ) ) ) {
		(void)printf( "%lf (templated) does not equal %lf (compiler).\n", alpha, beta );
		error = true;
	}

	if( ! error ) {
		(void)printf( "Functional test complete. Now starting benchmark run "
					  "1...\n" );
		fflush( stdout );
	}

	// first do a cold run
#ifdef __clang__
	// Otherwise the reference implementation hits this bug: https://llvm.org/bugs/show_bug.cgi?id=10856)
	if( grb::foldl< grb::descriptors::no_operation >( alpha, xv, NO_MASK, realm ) != grb::SUCCESS ) {
		assert( false );
		exit_status = 4;
		return;
	}
#else
	if( grb::foldl( alpha, xv, NO_MASK, realm ) != grb::SUCCESS ) {
		assert( false );
		exit_status = 4;
		return;
	}
#endif

	double ttime = 0.0;
	// now benchmark hot runs
	grb::utils::Timer timer;
	for( size_t i = 0; i < rep; ++i ) {
		alpha = realm.template getIdentity< double >();
		timer.reset();
#ifdef __clang__
		// Otherwise the reference implementation hits this bug: https://llvm.org/bugs/show_bug.cgi?id=10856)
		grb::foldl< grb::descriptors::no_operation >( alpha, xv, NO_MASK, realm );
#else
		grb::foldl( alpha, xv, NO_MASK, realm );
#endif
		ttime += timer.time() / static_cast< double >( rep );
		if( ! grb::utils::equals( check, alpha, static_cast< double >( n - 1 ) ) ) {
			(void)printf( "%lf (templated, re-entrant) does not equal %lf "
						  "(sequential).\n",
				alpha, check );
			error = true;
		}
	}
	(void)printf( "Average time taken for templated reduce: %lf.\n", ttime );
	(void)fflush( stdout );

	if( ! error ) {
		(void)printf( "Benchmark run 1 complete & verified. Now starting "
					  "benchmark run 2...\n" );
		fflush( stdout );
	}

	// first do a cold run
	alpha = xr[ 0 ];
	for( size_t i = 1; i < n; ++i ) {
		alpha += xr[ i ];
	}
	// now benchmark hot runs
	double ctime = 0.0;
	for( size_t k = 0; k < rep; ++k ) {
		timer.reset();
		alpha = xr[ 0 ];
		for( size_t i = 1; i < n; ++i ) {
			alpha += xr[ i ];
		}
		ctime += timer.time() / static_cast< double >( rep );
		if( ! grb::utils::equals( check, alpha, static_cast< double >( n - 1 ) ) ) {
			(void)printf( "%lf (compiler, re-entrant) does not equal %lf "
						  "(sequential).\n",
				alpha, check );
			error = true;
		}
	}
	(void)printf( "Average time taken for compiler-optimised reduce: %lf.\n", ctime );
	(void)fflush( stdout );

	free( xr );

	if( error ) {
		(void)printf( "Test FAILED.\n\n" );
		(void)fflush( stdout );
		(void)fflush( stderr );
		exit_status = 1;
	} else {
		(void)printf( "NOTE: please check the above performance figures "
					  "manually-- the last two timings\n      should "
					  "approximately match.\nTest OK.\n\n" );
		exit_status = 0;
	}
}
