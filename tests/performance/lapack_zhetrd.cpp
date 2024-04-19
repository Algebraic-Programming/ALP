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

#include <complex>
#include <cmath>
#include <iomanip>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/iscomplex.hpp>

#include "lapacke.h"

using BaseScalarType = double;
using ScalarType = std::complex< BaseScalarType >;
constexpr BaseScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

struct inpdata {
	size_t N=0;
  	size_t repeat=1;
};


//** generate vector or upper/lower triangular part of an SPD matrix */
template<
	typename T
>
void generate_symmherm_matrix_data(
	size_t N,
	std::vector< T > &data,
	const typename std::enable_if<
		grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	std::fill(data.begin(), data.end(), static_cast< T >( 0 ) );
	std::srand( RNDSEED );
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			T val( std::rand(), std::rand() );
			data[ i * N + j ] = val / std::abs( val );
			data[ j * N + i ] += grb::utils::is_complex< T >::conjugate( data[ i * N + j ] );
		}
	}
}



void alp_program( const inpdata &unit, bool &rc ) {
	rc = true;

	int N = unit.N;
	grb::utils::Timer timer;
	timer.reset();
	double times;


	std::cout << "Testing zhetrd_  ( " << N << " x " << N << " )\n";
	std::cout << "Test repeated " << unit.repeat << " times.\n";

	char uplo = 'U';
	std::vector< ScalarType > mat_a( N * N );
	generate_symmherm_matrix_data( N, mat_a );
	std::vector< BaseScalarType > vec_d( N );
	std::vector< BaseScalarType > vec_e( N - 1 );
	std::vector< ScalarType > vec_tau( N - 1 );
	ScalarType wopt;
	int lwork = -1;
	int info;
	
	zhetrd_(&uplo, &N, ( lapack_complex_double * )( &( mat_a[0] ) ), &N, 
		&( vec_d[0] ), &( vec_e[0] ), ( lapack_complex_double * )( &( vec_tau[0] ) ), ( lapack_complex_double * )( &( wopt ) ), &lwork, &info);
	lwork = (int)( wopt.real() );
	std::vector< ScalarType > work( lwork );
	
	times = 0;

	for( size_t j = 0; j < unit.repeat; ++j ) {
	  std::vector< ScalarType > mat_a_work( mat_a );
	  timer.reset();
	  zhetrd_(&uplo, &N, ( lapack_complex_double * )( &( mat_a_work[0] ) ), &N, 
	  	&( vec_d[0] ), &( vec_e[0] ), ( lapack_complex_double * )( &( vec_tau[0] ) ), ( lapack_complex_double * )( &( work[0] ) ), &lwork, &info);
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
