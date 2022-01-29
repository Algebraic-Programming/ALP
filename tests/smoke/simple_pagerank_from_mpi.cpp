
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

#define PR_TEST_DIMENSION 129

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
	size_t local_size;
	std::vector< size_t > indices;
	std::vector< double > pr_values;
	grb::utils::TimerResults times;
};

void grbProgram( const input_matrix & A, struct output_vector & out ) {

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
	timer.reset();
	rc = simple_pagerank< descriptors::no_operation >( pr, L, buf1, buf2, buf3 );
	time_taken = timer.time();

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

	// done
	return;
}

int main( int argc, char ** argv ) {
	// init MPI
	if( MPI_Init( &argc, &argv ) != MPI_SUCCESS ) {
		std::cerr << "MPI_Init returns with non-SUCCESS exit code." << std::endl;
		return 10;
	}

	for( int loop = 0; loop < LOOP_MAIN; ++loop ) {

		// the input matrix as a single big chunk of memory
		const size_t in_size = sizeof( struct input_matrix ) + nz * sizeof( size_t ) * 2;
		char * data_in = new char[ in_size ];
		// initialise struct
		{
			struct input_matrix & A = *reinterpret_cast< struct input_matrix * >( data_in );
			A.n = n;
			A.nz = nz;
			A.rows = reinterpret_cast< size_t * >( data_in + sizeof( struct input_matrix ) );
			// note that A.rows is of type size_t so the pointer arithmetic
			// in the below is exact (no need to multiply with sizeof(size_t))
			A.cols = reinterpret_cast< size_t * >( A.rows + nz );
			// construct example pattern matrix
			for( size_t i = 0; i < A.n; ++i ) {
				A.rows[ i ] = i;
				A.cols[ i ] = ( i + 1 ) % A.n;
			}
			A.rows[ A.n ] = A.n - 3;
			A.cols[ A.n ] = A.n - 1;
		}

		// create more convenient view of in_size
		const struct input_matrix & A = *reinterpret_cast< struct input_matrix * >( data_in );

		// output vector
		struct output_vector pr;
		// set invalid defaults
		pr.error_code = -1;
		pr.local_size = 0;

		grb::Launcher< FROM_MPI > launcher( MPI_COMM_WORLD );

		const enum grb::RC rc = launcher.exec( &grbProgram, A, pr );
		if( rc != SUCCESS ) {
			std::cerr << "grb::Launcher< FROM_MPI >::exec returns with "
						 "non-SUCCESS exit code "
					  << (int)rc << std::endl;
		}

		std::cout << "Error code is " << pr.error_code << ".\n";
		std::cout << "Size of pr is " << pr.local_size << ".\n";
		size_t max = pr.local_size >= 10 ? 10 : pr.local_size;
		if( max > 0 ) {
			std::cout << "First " << max << " elements of pr are: ( " << pr.pr_values[ 0 ];
			for( size_t i = 1; i < max; ++i ) {
				std::cout << ", " << pr.pr_values[ i ];
			}
			std::cout << " )" << std::endl;
		}

		// free all memory
		delete[] data_in;
	}

	// finalise MPI
	if( MPI_Finalize() != MPI_SUCCESS ) {
		std::cerr << "MPI_Finalize returns with non-SUCCESS exit code." << std::endl;
		return 50;
	}

	// done
	return 0;
}
