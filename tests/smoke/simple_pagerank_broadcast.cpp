
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
#include <vector>

#include <mpi.h>

#include <graphblas/algorithms/simple_pagerank.hpp>
#include <graphblas/utils/Timer.hpp>

#include <graphblas.hpp>

#define PR_TEST_DIMENSION 1000000

#ifdef MULTIPLE_ENTRY
#define LOOP_MAIN 3
#else
#define LOOP_MAIN 1
#endif

const int LPF_MPI_AUTO_INITIALIZE = 0;

using namespace grb;
using namespace algorithms;

constexpr size_t n = PR_TEST_DIMENSION;
constexpr size_t nz = n + 1;
constexpr size_t rep = 10;

struct input_matrix {
	size_t n, nz;
	size_t * rows;
	size_t * cols;
};

struct output_vector {
	int error_code;
#ifdef PINNED_OUTPUT
	PinnedVector< double > pinnedVector;
#else
	size_t local_size;
	std::vector< size_t > indices;
	std::vector< double > pr_values;
#endif
	grb::utils::TimerResults times;
};

void grbProgram( const void * data_in, const size_t in_size, struct output_vector & out ) {
	// sanity check
	assert( in_size >= sizeof( struct input_matrix ) );
#ifdef NDEBUG
	(void) in_size;
#endif
	// create more convenient view of in_size
	struct input_matrix A = *static_cast< const struct input_matrix * >( data_in );
	// correct input_matrix struct
	A.rows = reinterpret_cast< size_t * >( reinterpret_cast< char * >( const_cast< void * >( data_in ) ) + sizeof( struct input_matrix ) );
	A.cols = A.rows + A.nz;
	// sanity check
	assert( in_size == sizeof( struct input_matrix ) + nz * sizeof( size_t ) * 2 );

	// assume successful run
	out.error_code = 0;

	// load into GraphBLAS
	Matrix< void > L( A.n, A.n );
	RC rc = buildMatrixUnique( L, A.rows, A.cols, A.nz, SEQUENTIAL );
	if( rc != SUCCESS ) {
		out.error_code = 1;
		return;
	}

	// check number of nonzeroes
	if( nnz( L ) != A.nz ) {
		out.error_code = 2;
		return;
	}

	// test default pagerank run
	Vector< double > pr( A.n );
	Vector< double > buf1( A.n ), buf2( A.n ), buf3( A.n );

	double time_taken;
	grb::utils::Timer timer;
	constexpr double alpha = 0.85;
	constexpr double conv = 0.0000001;
	constexpr size_t max = 1000;
	size_t iterations;
	double quality;
	timer.reset();
	rc = simple_pagerank< descriptors::no_operation >( pr, L, buf1, buf2, buf3,
		alpha, conv, max, &iterations, &quality );
	time_taken = timer.time();
	if( conv <= quality ) {
		if( spmd<>::pid() == 0 ) {
			std::cerr << "Info: simple pagerank converged after " << iterations
				<< " iterations." << std::endl;
		}
	} else {
		if( spmd<>::pid() == 0 ) {
			std::cout << "Info: simple pagerank did not converge after "
				<< iterations << " iterations." << std::endl;
		}
		rc = grb::FAILED; // not converged
	}

	// print timing at root process
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "Time taken for a single PageRank call (cold start): " << time_taken << std::endl;
	}

	// set error code
	if( rc == FAILED ) {
		out.error_code = 3;
		// no convergence, but will print output
	} else if( rc != SUCCESS ) {
		out.error_code = 4;
		return;
	}

	// output
#ifdef PINNED_OUTPUT
	out.pinnedVector = PinnedVector< double >( pr, SEQUENTIAL );
#else
	for( auto nonzero : pr ) {
		out.indices.push_back( nonzero.first );
		out.pr_values.push_back( nonzero.second );
	}
#ifndef NDEBUG
	const size_t oinsize = out.indices.size();
	const size_t oprsize = out.pr_values.size();
	assert( oinsize == oprsize );
#endif
	out.local_size = out.indices.size();
#endif

	// done
	return;
}

int main( int argc, char ** argv ) {
	if( MPI_Init( &argc, &argv ) != MPI_SUCCESS ) {
		std::cerr << "MPI_Init returns with non-SUCCESS exit code." << std::endl;
		return 10;
	}

	int s = -1;
	(void)MPI_Comm_rank( MPI_COMM_WORLD, &s );
	assert( s != -1 );

	for( int loop = 0; loop < LOOP_MAIN; ++loop ) {

		// the input matrix as a single big chunk of memory
		const size_t in_size = s == 0 ? sizeof( struct input_matrix ) + nz * sizeof( size_t ) * 2 : 0;
		char * data_in = s == 0 ? new char[ in_size ] : NULL;

		// root process initialises
		if( s == 0 ) {
			// create more convenient view of in_size
			struct input_matrix & A = *reinterpret_cast< struct input_matrix * >( data_in );
			A.n = n;
			A.nz = nz;
			A.rows = reinterpret_cast< size_t * >( data_in + sizeof( struct input_matrix ) );
			A.cols = reinterpret_cast< size_t * >( A.rows + nz ); // note that A.rows is of type size_t so thepointer airthmetic is exact here (no need to add sizeof(size_t))
			// construct example pattern matrix
			for( size_t i = 0; i < A.n; ++i ) {
				A.rows[ i ] = i;
				A.cols[ i ] = ( i + 1 ) % A.n;
			}
			A.rows[ A.n ] = A.n - 3;
			A.cols[ A.n ] = A.n - 1;
		}

		// output vector
		struct output_vector pr;
		// set invalid defaults
		pr.error_code = -1;
#ifndef PINNED_OUTPUT
		pr.local_size = 0;
#endif

		grb::Launcher< FROM_MPI > launcher( MPI_COMM_WORLD );
		const enum grb::RC rc = launcher.exec( &grbProgram, data_in, in_size, pr, true );
		if( rc != SUCCESS ) {
			std::cerr << "grb::Launcher< FROM_MPI >::exec returns with "
						 "non-SUCCESS exit code "
					  << (int)rc << std::endl;
			return 16;
		}

		std::cout << "Error code is " << pr.error_code << ".\n";
#ifdef PINNED_OUTPUT
		assert( pr.pinnedVector.size() > 0 );
		std::cout << "Size of pr is " << pr.pinnedVector.size() << ".\n";
#else
		assert( pr.local_size > 0 );
		std::cout << "Size of pr is " << pr.local_size << ".\n";
#endif
		std::cout << "First 10 nonzeroes of pr are: ( ";
#ifdef PINNED_OUTPUT
		for( size_t k = 0; k < 10 && k < pr.pinnedVector.nonzeroes() && k < 10; ++k ) {
			const auto &nonzeroValue = pr.pinnedVector.getNonzeroValue( k );
			std::cout << nonzeroValue << " ";
		}
#else
		for( size_t i = 0; i < 10; ++i ) {
			std::cout << pr.pr_values[ i ] << " ";
		}
#endif
		std::cout << ")" << std::endl;

		// free all memory
		delete [] data_in;
	}

	if( MPI_Finalize() != MPI_SUCCESS ) {
		std::cerr << "MPI_Finalize returns with non-SUCCESS exit code." << std::endl;
		return 20;
	}

	// done
	return 0;
}
