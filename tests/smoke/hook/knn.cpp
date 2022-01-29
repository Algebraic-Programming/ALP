
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

#include <graphblas/algorithms/knn.hpp>
#include <graphblas/utils/Timer.hpp>

#include <graphblas.hpp>

#ifndef KNN_TEST_DIMENSION
 #define KNN_TEST_DIMENSION 10
#endif

#ifdef KNN_DATASET_FILE
 static constexpr size_t n = KNN_DATASET_N;
 static size_t nz;
 static double * weights;
 #define QUOTE( x ) #x
 #define STR( x ) QUOTE( x )
#else
 static constexpr size_t n = KNN_TEST_DIMENSION;
 static constexpr size_t nz = n + 1;
#endif

// forward declaration of the graph dataset parser
bool readEdges(
	const std::string filename, const bool use_indirect,
	const size_t n, size_t * const nz,
	size_t ** const I, size_t ** const J, double ** const weights
);

// helper function to clear dynamically allocated data
static void coda( size_t * const LI, size_t * const LJ ) {
	delete[] LI;
	delete[] LJ;
}

using namespace grb;
using namespace algorithms;

void grbProgram( const size_t &P, int &exit_status ) {
	const size_t s = spmd<>::pid();
	(void)P;

	grb::utils::Timer benchtimer;
	benchtimer.reset();

	// assume successful run
	exit_status = 0;

#ifdef KNN_DATASET_FILE
	// construct pattern matrix from a dataset file
	size_t *LI, *LJ;
	std::string type = STR( KNN_DATASET_TYPE );
	std::cout << "Loading from dataset " << std::to_string( STR( KNN_DATASET_FILE ) ) << "...\n";
	readEdges( STR( KNN_DATASET_FILE ), type == "indirect", KNN_DATASET_N, &nz, &LI, &LJ, &weights );
#else
	// construct example pattern matrix
	std::cout << "Loading example of " << n << " vertices and " << nz << " == " << ( n + 1 ) << " nonzeroes.\n";
	size_t * LI = new size_t[ nz ];
	size_t * LJ = new size_t[ nz ];
	for( size_t i = 0; i < n; ++i ) {
		LI[ i ] = i;
		LJ[ i ] = ( i + 1 ) % n;
	}
	LI[ n ] = n - 3;
	LJ[ n ] = n - 1;
#endif

	// load into GraphBLAS
	Matrix< void > L( n, n );
	RC rc = buildMatrixUnique( L, LI, LJ, nz, SEQUENTIAL );
	if( rc != SUCCESS ) {
		exit_status = 1;
		coda( LI, LJ );
		return;
	}

	// check number of nonzeroes
	if( nnz( L ) != nz ) {
		exit_status = 2;
		coda( LI, LJ );
		return;
	}

	// intialise timer instruments
	double time_taken;
	utils::Timer timer;

	// test default knn run on source n-4
	Vector< bool > neighbourhood( n );
	Vector< bool > buf1( n ), buf2( n );
	assert( nnz( neighbourhood ) == 0 );

	std::cout << "Now passing into grb::algorithms::knn with source = " << ( n - 4 ) << " for benchmark...\n";
	timer.reset();
	benchtimer.reset();
	rc = knn< descriptors::no_operation >( neighbourhood, L, n - 4, 1, buf1, buf2 );
	benchtimer.reset();
	time_taken = timer.time();

	// set error code
	if( rc != SUCCESS ) {
		exit_status = 4;
		coda( LI, LJ );
		return;
	}

	// print timing at root process
	if( s == 0 ) {
		std::cout << "Average time taken for call to knn (root user process): " << time_taken << std::endl;
	}

	rc = collectives<>::allreduce< descriptors::no_casting >( time_taken, operators::max< double >() );
	assert( rc == SUCCESS ); // do this via assert, we are not testing collectives here

	if( s == 0 ) {
		std::cout << "Average time taken for call to knn (max over all user "
					 "processes): "
				  << time_taken << std::endl;
	}

	// print check to screen if dimension is small
	if( nnz( neighbourhood ) <= 128 ) {
		for( size_t k = 0; k < P; ++k ) {
			if( s == k ) {
				std::cout << "Neighbourhood local to PID " << s << " on exit is ( ";
				auto it = neighbourhood.begin();
				const auto it_end = neighbourhood.end();
				for( ; it != it_end; ++it ) {
					const auto nonzero = *it;
					if( ! ( nonzero.second ) )
						continue;
					std::cout << nonzero.first << " ";
				}
				std::cout << ")\n";
			}
			const enum RC bsprc = spmd<>::sync();
			assert( bsprc == SUCCESS );
#ifdef NDEBUG
			(void) bsprc;
#endif
		}
	}

	// done
	coda( LI, LJ );
	return;
}

