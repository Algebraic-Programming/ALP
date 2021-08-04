
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

#include <graphblas.hpp>

using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 7, 6 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 7, 5, 1 };
static const double data2[ 6 ] = { 1, 1, 1, 1, 1, 1 };
static const size_t I2[ 6 ] = { 0, 1, 0, 2, 1, 2 };
static const size_t J2[ 6 ] = { 1, 0, 2, 0, 2, 1 };
static const double testv[ 3 ] = { 0.1, 2.1, -2.3 };

void grb_program( const size_t & n, grb::RC & rc ) {
	// initialize test
	unsigned int chk[ 10 ][ 10 ];
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			chk[ i ][ j ] = 0;
		}
	}
	for( size_t k = 0; k < 15; ++k ) {
		chk[ I[ k ] ][ J[ k ] ] = data1[ k ];
	}
	grb::Matrix< double > A( n, n );
	grb::Matrix< unsigned int > B( n, n );
	grb::Vector< unsigned int > u( n );
	grb::Vector< unsigned int > v( n );
	for( size_t i = 0; i < n; ++i ) {
		grb::setElement( u, i * n, i );
		grb::setElement( v, i * n * n, i );
	}
	rc = grb::resize( A, 15 );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrixUnique( A, I, J, data1, 15, SEQUENTIAL );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize( B, 15 );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( B, A );
	}
	if( rc != SUCCESS || grb::nnz( A ) != 15 || grb::nnz( B ) != 15 ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}
	rc = grb::eWiseLambda(
		[ &B ]( const size_t i, const size_t j, unsigned int & v ) {
			(void)i;
			(void)j;
			v -= 1;
		},
		B );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::eWiseLambda (matrix, no vectors) FAILED\n";
		return;
	}

	// checks
	if( grb::nnz( B ) != 15 ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( B ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & triplet : B ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ).\n";
			rc = FAILED;
		} else if( chk[ triplet.first.first ][ triplet.first.second ] - 1 != triplet.second ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value " << ( chk[ triplet.first.first ][ triplet.first.second ] - 1 ) << ".\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	rc = grb::eWiseLambda(
		[ &B, &u, &v ]( const size_t i, const size_t j, unsigned int & val ) {
			val = val + 1 + u[ i ] + v[ j ];
		},
		B, u, v );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::eWiseLambda (matrix and vectors) FAILED\n";
		return;
	}

	// checks
	if( grb::nnz( B ) != 15 ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( B ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & triplet : B ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ).\n";
			rc = FAILED;
		} else {
			const unsigned int expected = chk[ triplet.first.first ][ triplet.first.second ] + triplet.first.first * n + triplet.first.second * n * n;
			if( expected != triplet.second ) {
				std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
				if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
					std::cerr << ", expected no entry here.\n";
				} else {
					std::cerr << ", expected value " << expected << ".\n";
				}
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	grb::Matrix< double > W( 3, 3 );
	grb::Vector< double > vec( 3 );
	rc = grb::resize( W, 6 );
	rc = rc ? rc : grb::buildMatrixUnique( W, I2, J2, data2, 6, SEQUENTIAL );
	rc = rc ? rc : grb::buildVector( vec, testv, testv + 3, SEQUENTIAL );
	rc = rc ? rc :
              grb::eWiseLambda(
				  [ &W, &vec ]( const size_t i, const size_t j, double & v ) {
					  v *= vec[ i ] - vec[ j ];
				  },
				  W );
	if( grb::nnz( W ) != 6 ) {
		std::cout << "Unexpected number of nonzeroes in W: " << grb::nnz( W ) << ", expected 6.\n";
		rc = FAILED;
	}
	for( const std::pair< std::pair< size_t, size_t >, double > & triple : W ) {
		const size_t & i = triple.first.first;
		const size_t & j = triple.first.second;
		const double & v = triple.second;
		const double ex = testv[ i ] - testv[ j ];
		if( v != ex ) {
			std::cout << "Unexpected value at position ( " << i << ", " << j << " ) in W: " << v << ", expected " << ex << ".\n";
			rc = FAILED;
		}
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
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
