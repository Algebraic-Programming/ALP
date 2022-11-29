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

#include <alp.hpp>


typedef double ScalarType;
constexpr ScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

struct inpdata {
	size_t N=0;
  	size_t repeat=1;
};


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



void alp_program( const inpdata &unit, alp::RC &rc ) {

  rc = alp::SUCCESS;

	const size_t N = unit.N;
	const size_t K = 2 * N;
	const size_t M = 3 * N;
	grb::utils::Timer timer;
	timer.reset();
	double times;

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;



	times = 0;
	alp::Matrix< ScalarType, alp::structures::General, alp::Dense > A( N, K );
	alp::Matrix< ScalarType, alp::structures::General, alp::Dense > B( K, M );
	alp::Matrix< ScalarType, alp::structures::General, alp::Dense > C( N, M );

	{
		std::vector< ScalarType > Amatrix_data( N * K );
		generate_random_matrix_data( N * K, Amatrix_data );
		rc = rc ? rc : alp::buildMatrix( A, Amatrix_data.begin(), Amatrix_data.end() );

		std::vector< ScalarType > Bmatrix_data( K * M );
		generate_random_matrix_data( K * M, Bmatrix_data );
		rc = rc ? rc : alp::buildMatrix( A, Bmatrix_data.begin(), Bmatrix_data.end() );
	}

	std::cout << "Testing  C(" << nrows( C ) << " x " << ncols( C )
		  << ") +=   A(" << nrows( A ) << " x " << ncols( A )
		  << ") x B(" << nrows( B ) << " x " << ncols( B )
		  << ")  "  << unit.repeat << " times.\n";

	alp::Scalar< ScalarType > zero( ring.template getZero< ScalarType >() );

	for( size_t j = 0; j < unit.repeat; ++j ) {
		rc = rc ? rc : alp::set( C, zero );
		timer.reset();
		rc = rc ? rc : alp::mxm( C, A, B, ring );
		times += timer.time();
	}


	std::cout << " times(total) = " << times << "\n";
	std::cout << " times(per repeat) = " << times / unit.repeat  << "\n";

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
