
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
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <alp.hpp>

#include "../utils/print_alp_containers.hpp"

using namespace alp;

void alp_program( const size_t &n, alp::RC &rc ) {

	typedef double T;

    auto print_std_vector = [](std::vector< T > const vec) {
        for (auto val : vec) {
            std::cout << val << ' ';
        }
        std::cout << std::endl;
    };

	// Check with vector of length n randomly intitialized and shuffled
	alp::Vector< size_t > perm( n );
	alp::Vector< double > v( n );

	std::random_device rd;
	std::default_random_engine rng( rd() );

	std::vector< double > stdv( n );

	std::iota( std::begin( stdv ), std::end( stdv ), 0. );
	std::shuffle( std::begin( stdv ), std::end( stdv ), rng );

	alp::buildVector( v, std::begin( stdv ), std::end( stdv ) );

    std::cout << "Original content of the std::vector:" << std::endl;
	print_std_vector( stdv );
    std::cout << "Original content of the alp::Vector:" << std::endl;
	print_vector("v", v);

	alp::sort( perm, v, []( T a, T b) { 
		return a > b; 
	} );

	std::sort( std::begin( stdv ), std::end( stdv ) );

	for ( size_t i = 0; i < n; i++ ) {
		if ( stdv[i] != v[ perm[ i ] ] ) {
			std::cerr << "Error: ( std::v[ " << i << " ] = " << stdv[i] << " ) != " << " ( alp::v[ perm[ " << i << " ] ] = " << v[ perm[ i ] ] << " )" << std::endl;
		}
	}

	rc = alp::SUCCESS;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
