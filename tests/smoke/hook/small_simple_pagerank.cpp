
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

#include <graphblas.hpp>

#include <graphblas/algorithms/simple_pagerank.hpp>

#include <iostream>


#ifndef PR_TEST_DIMENSION
 #define PR_TEST_DIMENSION 10
#endif

#ifdef PR_DATASET_FILE
 static constexpr size_t n = PR_DATASET_N;
 static size_t nz;
 static double * weights;
 #define QUOTE( x ) #x
 #define STR( x ) QUOTE( x )
#else
 static constexpr size_t n = PR_TEST_DIMENSION;
 static constexpr size_t nz = n + 1;
#endif

// forward declaration of the graph dataset parser
bool readEdges(
	const std::string filename, const bool use_indirect,
	const size_t n, size_t * const nz,
	size_t ** const I, size_t ** const J, double ** const weights
);

using namespace grb;
using namespace algorithms;

static void coda( size_t * const LI, size_t * const LJ, const size_t nz ) {
	if( nz == 0 ) { return; }
	delete [] LI;
	delete [] LJ;
}

void grbProgram( const size_t &P, int &exit_status ) {
	const size_t s = spmd<>::pid();
	(void)P;

	// assume successful run
	exit_status = 0;

#ifdef PR_DATASET_FILE
	// construct pattern matrix from a dataset file
	size_t *LI, *LJ;
	std::string type = STR( PR_DATASET_TYPE );
	readEdges( STR( PR_DATASET_FILE ), type == "indirect", PR_DATASET_N, &nz, &LI, &LJ, &weights );
#else
	// construct example pattern matrix
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
		coda( LI, LJ, nz );
		return;
	}

	// check number of nonzeroes
	if( nnz( L ) != nz ) {
		exit_status = 2;
		coda( LI, LJ, nz );
		return;
	}

	// test default pagerank run
	Vector< double > pr( n );
	Vector< double > buf1( n ), buf2( n ), buf3( n );

	rc = simple_pagerank<>( pr, L, buf1, buf2, buf3 );

	// set error code
	if( rc == FAILED ) {
		exit_status = 3;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		exit_status = 4;
		coda( LI, LJ, nz );
		return;
	}

	// print check to screen if dimension is small
	if( n / P <= 128 ) {
		for( size_t k = 0; k < P; ++k ) {
			if( s == k ) {
				std::cout << "Pagerank vector local to PID " << s << " on exit is ( ";
				for( auto nonzero : pr ) {
					std::cout << nonzero.second << " ";
				}
				std::cout << ")\n";
			}
			// note: if the backend uses a barrier for synchronisation, then the below
			//       synchronises the output to std::out which is useful for
			//       verification.
#ifndef NDEBUG
			const auto sync_rc = spmd<>::sync();
			assert( sync_rc == SUCCESS );
#else
			(void) spmd<>::sync();
#endif
		}
		if( s == 0 && rc == FAILED ) {
			std::cout << "The PageRank algorithm did not converge." << std::endl;
		}
	}

	// done
	coda( LI, LJ, nz );
	return;
}

