
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
#include <sstream>

#include <alp.hpp>

using namespace alp;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 7, 6 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 7, 5, 1 };
static const double data2[ 6 ] = { 1, 1, 1, 1, 1, 1 };
static const size_t I2[ 6 ] = { 0, 1, 0, 2, 1, 2 };
static const size_t J2[ 6 ] = { 1, 0, 2, 0, 2, 1 };
static const double testv[ 3 ] = { 0.1, 2.1, -2.3 };

void alp_program( const size_t &n, alp::RC &rc ) {
	// initialize test
	typedef int T;
	alp::Matrix< T, structures::General > A( n, n );
	alp::Vector< T > u( n );
	alp::Vector< T > v( n );

	internal::setInitialized( A, true );
	internal::setInitialized( u, true );
	internal::setInitialized( v, true );

	// test eWiseLambda on matrix
	rc = alp::eWiseLambda(
		[]( const size_t i, const size_t j, T &val ) {
			(void)i;
			(void)j;
			val = 1;
		},
		A
	);
	if( rc != alp::SUCCESS ){
		std::cerr << "\talp::eWiseLambda (matrix, no vectors) FAILED\n";
		return;
	}

	// test eWiseLambda on vector
	rc = alp::eWiseLambda(
		[]( const size_t i, T &val ) {
			(void)i;
			val = 2;
		},
		v
	);

	if( rc != SUCCESS ) {
		std::cerr << "\talp::eWiseLambda (vector) FAILED\n";
		return;
	}

	// test eWiseLambda on vector, consuming from another vector
	rc = alp::eWiseLambda(
		[ &v ]( const size_t i, T &val ) {
			val = v[ i ];
		},
		u, v
	);

	if( rc != SUCCESS ) {
		std::cerr << "\talp::eWiseLambda (vector, vectors...) FAILED\n";
		return;
	}

	// test eWiseLambda on Matrix, consuming two other vectors
	rc = alp::eWiseLambda(
		[ &u, &v ]( const size_t i, const size_t j, T &val ) {
			val = val + u[ i ] * v[ j ];
		},
		A, u, v
	);

	if( rc != SUCCESS ) {
		std::cerr << "\talp::eWiseLambda (matrix, vectors...) FAILED\n";
		return;
	}
	
	

	if( rc != SUCCESS ) {
		return;
	}
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
