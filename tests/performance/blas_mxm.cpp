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

#include <graphblas/utils/Timer.hpp>

#include <alp_blas.h> // for gemm

typedef double ScalarType;
constexpr ScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

struct inpdata {
	size_t N=0;
  	size_t repeat=1;
};

template< typename T >
void print(const char * name, const std::vector<T> &matrix, int M, int N )
{
  std::cout <<  name << " = array ( [\n";
  for (int i = 0; i < M; i++){
    std::cout << "  [";
    for (int j = 0; j < N; j++){
      std::cout << matrix[i * N + j ] << ", ";
    }
    std::cout << " ],\n";
  }
  std::cout << "\n])\n";
}



/** gnerate random rectangular matrix data */
template< typename T >
void generate_random_matrix_data( size_t n, std::vector<T> &data ) {
	if( data.size() != n ) {
	        std::cout << "Error: generate_random_matrix_data: Provided container does not have adequate size\n";
	}
        for( size_t i = 0; i < n; ++i ) {
                data[ i ] = static_cast< T >( std::rand() ) / static_cast< T >( RAND_MAX );
        }
}

void alp_program( const inpdata &unit, bool &rc ) {
  rc = true;

  const size_t N = unit.N;
  const size_t K = 1 * N;
  const size_t M = 1 * N;
  grb::utils::Timer timer;
  timer.reset();
  double times;

  std::vector< ScalarType > Amatrix_data( N * K );
  std::vector< ScalarType > Bmatrix_data( K * M );
  std::vector< ScalarType > Cmatrix_data( N * M );
  generate_random_matrix_data( N * K, Amatrix_data );
  generate_random_matrix_data( K * M, Bmatrix_data );

  // print("A ", Amatrix_data, N, K );
  // print("B ", Bmatrix_data, K, M );
  // print("C ", Cmatrix_data, N, M );


  std::cout << "Testing cblas_dgemm for C(" << N << " x " << M
	    << ") +=   A(" << N << " x " << K
	    << ") x B(" << K << " x " << M
	    << ")  "  << unit.repeat << " times.\n";

   times = 0;

	for( size_t j = 0; j < unit.repeat; ++j ) {
	  timer.reset();
	  cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		N,
		M,
		K,
		1,
		&(Amatrix_data[0]),
		K,
		&(Bmatrix_data[0]),
		M,
		1,
		&(Cmatrix_data[0]),
		M
	  );
	  times += timer.time();
	}

	std::cout << " time (ms, total) = " << times << "\n";
	std::cout << " time (ms, per repeat) = " << times / unit.repeat  << "\n";

	// print("C ", Cmatrix_data, N, M );
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
