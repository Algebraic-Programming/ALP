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

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <graphblas/utils/timer.hpp>

#include <alp.hpp>
#include <alp/algorithms/cholesky.hpp>

//#include "klapack.h"

typedef double ScalarType;
constexpr ScalarType tol = 1.e-9;
constexpr size_t RNDSEED = 1;

struct inpdata {
	size_t N = 0;
  	size_t repeat = 1;
};

void print( const char * name, const double* matrix, int N ) {
	printf( "\nMatrix %s size %d :\n", name, N );
	printf( " %s = array ( [", name );
	for( int i = 0; i < N; i++ ) {
		printf( "\n  [" );
		for( int j = 0; j < N; j++ ) {
			printf("%.10f, ", matrix[j*N + i]);
		}
		printf(" ],");
	}
	printf("\n])\n");
}

#ifdef _ALP_WITH_DISPATCH
alp::RC check_solution( const double* matrixH, const double* matrixL, size_t N ) {
	alp::RC rc = alp::SUCCESS;
	double fnorm1 = 0;
	double fnorm2 = 0;
	for( size_t i = 0; i < N; i++ ){
		for( size_t j = 0; j < N; j++ ){
			double val_mxm = 0;
			for( size_t k = 0; k <= std::min( i, j ); k++ ) {
				val_mxm += matrixL[ k * N + i ] * matrixL[ k * N + j ] ;
			}
			fnorm1 += std::abs( matrixH[ i * N + j] - val_mxm );
			fnorm2 += std::abs( matrixH[ i * N + j] );
		}
	}
	if( std::abs( fnorm1 ) / std::abs( fnorm2 ) > tol ) {
	  std::cout << " fnorm1 = " << std::abs( fnorm1 ) << "\n";
	  std::cout << " fnorm2 = " << std::abs( fnorm2 ) << "\n";
	  std::cout << " fnorm1 / fnorm1 = " << std::abs( fnorm1 ) / std::abs( fnorm2 ) << "\n";
	  rc = alp::FAILED;
	} 
	return rc;
}
#endif

//** gnerate upper/lower triangular part of a SPD matrix */
template< typename T >
void generate_spd_matrix_full( size_t N, std::vector<T> &data ) {
	if( data.size() != N * N ) {
		std::cout << "Error: generate_spd_matrix_full: Provided container does not have adequate size\n";
	}
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = 0; j < N; ++j ) {
			size_t k = i * N + j;
			if( i <= j ) {
				data[ k ] = static_cast< T >( std::rand() ) / static_cast< T >( RAND_MAX );
			}
			if( i == j ) {
				data[ k ] = data[ k ] + static_cast< T >( N );
			}
			if( i > j ) {
				data[ i * N + j ] = data[ j * N + i ];
			}

		}
	}
}



void alp_program( const inpdata &unit, alp::RC &rc ) {

  rc = alp::SUCCESS;

	size_t N = unit.N;
	grb::utils::Timer timer;
	timer.reset();
	double times;

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;
	
	std::vector< ScalarType > matrix_data( N * N );
	generate_spd_matrix_full( N, matrix_data );

	std::cout << "Testing Cholesky decomposition U^T * U = S, with S SPD of size ( " << N << " x " << N << " )\n";
	std::cout << "Test repeated " << unit.repeat << " times.\n";

	times = 0;
	alp::Matrix< ScalarType, alp::structures::Square, alp::Dense > LL_original( N );
	alp::Matrix< ScalarType, alp::structures::Square, alp::Dense > LL( N );
	rc = rc ? rc : alp::buildMatrix( LL_original, matrix_data.begin(), matrix_data.end() );
	constexpr size_t bs = 64;
	
	for( size_t j = 0; j < unit.repeat; ++j ) {
	  rc = rc ? rc : alp::set( LL, LL_original );
	  timer.reset();
	  rc = rc ? rc : alp::algorithms::cholesky_uptr_blk( LL, bs, ring );
	  times += timer.time();
	  if (rc != alp::SUCCESS) {
	    std::cout << "cholesky_uptr_blk retured error\n";
	    return;
	  } 
	}

	std::cout << " time (ms, total) = " << times << "\n";
	std::cout << " time (ms, per repeat) = " << times / unit.repeat  << "\n";
#ifdef _ALP_WITH_DISPATCH
	//print("matrix_data", &( matrix_data[ 0 ] ), N );
	auto LL_original_data = alp::internal::getRawPointerToFirstElement( LL_original );
	//print("LL_original_data", LL_original_data, N );
	auto LL_output_data = alp::internal::getRawPointerToFirstElement( LL );
	//print("LL_output_data", LL_output_data, N );
	rc = rc ? rc : check_solution(LL_original_data, LL_output_data, N);
#endif

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	inpdata in;

	// error checking
	if(
	   ( argc == 3 ) || ( argc == 5 )
	   ) {
	  std::string readflag;
	  std::istringstream ss1( argv[ 1 ] );
	  std::istringstream ss2( argv[ 2 ] );
	  if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
	    std::cerr << "Error parsing\n";
	    printUsage = true;
	  } else if(
		    readflag != std::string( "-n" )
		    ) {
	    std::cerr << "Given first argument is unknown\n";
	    printUsage = true;
	  } else {
	    if( ! ( ( ss2 >> in.N ) &&  ss2.eof() ) ) {
	      std::cerr << "Error parsing\n";
	      printUsage = true;
	    }
	  }

	  if( argc == 5 ) {
	    std::string readflag;
	    std::istringstream ss1( argv[ 3 ] );
	    std::istringstream ss2( argv[ 4 ] );
	    if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
	      std::cerr << "Error parsing\n";
	      printUsage = true;
	    } else if(
		      readflag != std::string( "-repeat" )
		      ) {
	      std::cerr << "Given third argument is unknown\n";
	      printUsage = true;
	    } else {
	      if( ! ( ( ss2 >> in.repeat ) &&  ss2.eof() ) ) {
		std::cerr << "Error parsing\n";
		printUsage = true;
	      }
	      
	    }

	    
	  }
	  
	} else {
	  std::cout << "Wrong number of arguments\n" ;
	  printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: \n";
		std::cerr << "      or  \n";
		std::cerr << "       " << argv[ 0 ] << " -n N \n";
		std::cerr << "       " << argv[ 0 ] << " -n N   -repeat N \n";
		return 1;
	}

	alp::RC rc = alp::SUCCESS;
	alp_program(in, rc);
	if (rc == alp::SUCCESS) {
	  std::cout << "Tests OK\n";
	} else {
	  std::cout << "Tests FAILED\n";
	}
	return 0;
}
