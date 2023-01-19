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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <graphblas/utils/Timer.hpp>

#include "lapacke.h"

typedef double ScalarType;
constexpr ScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

struct inpdata {
	size_t N=0;
  	size_t repeat=1;
};

void print(const char * name, const double* matrix, int N)
{
  printf("\nMatrix %s size %d :\n", name, N);
  printf(" %s = array ( [", name);
  for (int i = 0; i < N; i++){
  printf("\n  [");
    for (int j = 0; j < N; j++){
      printf("%.10f, ", matrix[j*N + i]);
    }
    printf(" ],");
  }
  printf("\n])\n");
}


//** generate vector or upper/lower triangular part of an SPD matrix */
template< typename T >
void generate_vec_or_spd_matrix_full( size_t N, std::vector<T> &data ) {
	if( data.size() == N ) {
		for( size_t i = 0; i < N; ++i ) {
			data[ i ] = static_cast< T >( std::rand() ) / static_cast< T >( RAND_MAX );
		}
	} else if( data.size() == N * N ) {
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
	} else {
		std::cout << "Error: generate_spd_matrix_full: Provided container does not have adequate size\n";
	}

}



void alp_program( const inpdata &unit, bool &rc ) {
	rc = true;

	int N = unit.N;
	grb::utils::Timer timer;
	timer.reset();
	double times;


	std::cout << "Testing dpotri_ ( " << N << " x " << N << " )\n";
	std::cout << "Test repeated " << unit.repeat << " times.\n";

	char uplo = 'U';
	std::vector< ScalarType > mat_a( N * N );
	generate_vec_or_spd_matrix_full( N, mat_a );
	int info;

	times = 0;

	for( size_t j = 0; j < unit.repeat; ++j ) {
	  std::vector< ScalarType > mat_a_work( mat_a );
	  timer.reset();
	  dpotri_( &uplo, &N, &( mat_a_work[0] ), &N, &info );
	  times += timer.time();
	  if( info != 0 ) {
	    std::cout << " info = " << info << "\n";
	    rc = false;
	    return;
	  }
	}


	std::cout << " time (ms, total) = " << times << "\n";
	std::cout << " time (ms, per repeat) = " << times / unit.repeat  << "\n";

	//print("matrix_data", &(matrix_data[0]), N);

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

	bool rc = true;
	alp_program(in, rc);
	if (rc) {
	  std::cout << "Tests OK\n";
	} else {
	  std::cout << "Tests FAILED\n";
	}
	return 0;
}
