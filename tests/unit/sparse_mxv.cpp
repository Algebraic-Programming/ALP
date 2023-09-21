
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

#include <cstdio>

#include <assert.h>

#include "graphblas.hpp"


using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

void grbProgram( const int &, int &error ) {
	// allocate
	grb::Vector< int > x( 15 );
	grb::Vector< int > sparse_x( 15 );
	grb::Matrix< int > A( 15, 15 );

	// resize for 15 elements
	grb::RC rc = resize( A, 15 );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Matrix constructor: %d.\n", (int)rc );
		error = 3;
	}

	// initialise x
	if( !error ) {
		const int * iterator = &(data1[ 0 ]);
		rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr, "Unexpected return code from Vector build (x): %d.\n", (int)rc );
			error = 4;
		}
	}

	// initialise A
	rc = grb::buildMatrixUnique( A, I, J, data2, 15, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Matrix buildMatrixUnique: %d.\n", (int)rc );
		error = 5;
	}

	// get a semiring where multiplication is addition, and addition is multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring<
		grb::operators::add< int >, grb::operators::mul< int >,
		grb::identities::zero, grb::identities::one
	> integers;

	// for each output element, try masked SpMV
	for( size_t i = 0; !error && i < 15; ++i ) {
		// initialise output vector and mask vector
		grb::Vector< int > y( 15 );
		grb::Vector< bool > m( 15 );

		// check zero sizes
		if( grb::nnz( y ) != 0 ) {
			(void)fprintf( stderr, "Unexpected number of nonzeroes in y: %zd (expected 0).\n", grb::nnz( y ) );
			error = 6;
		}
		if( !error && grb::nnz( m ) != 0 ) {
			(void)fprintf( stderr, "Unexpected number of nonzeroes in m: %zd (expected 0).\n", grb::nnz( m ) );
			error = 7;
		}

		// initialise mask
		if( !error ) {
			rc = grb::setElement( m, true, i );
			if( rc != grb::SUCCESS ) {
				(void)fprintf( stderr, "Unexpected return code from vector set (m[%zd]): %d.\n", i, (int)rc );
				error = 8;
			}
		}
		if( !error ) {
			if( grb::nnz( m ) != 1 ) {
				(void)fprintf( stderr, "Unexpected number of nonzeroes in m: %zd (expected 1).\n", grb::nnz( m ) );
				error = 9;
			}
		}

		// execute what amounts to elementwise vector addition
		if( !error ) {
			rc = grb::mxv( y, m, A, x, integers );
			if( rc != grb::SUCCESS ) {
				(void)fprintf( stderr, "Unexpected return code from grb::vxm: %d.\n", (int)rc );
				error = 10;
			}
		}

		// check sparsity
		if( !error && grb::nnz( y ) != 1 ) {
			(void)fprintf( stderr, "Unexpected number of nonzeroes in y: %zd (expected 1).\n", grb::nnz( y ) );
			error = 11;
		}

		// check value
		for( const auto &pair : y ) {
			const size_t cur_index = pair.first;
			const int against = pair.second;
			if( !error && cur_index == i && !grb::utils::equals( chk[ i ], against ) ) {
				(void)fprintf( stderr,
					"Output vector element mismatch at position %zd: %d does not equal "
					"%d.\n",
					i, chk[ i ], against
				);
				error = 12;
			}
			if( !error && cur_index != i ) {
				(void)fprintf( stderr, "Expected no ouput vector element at position %zd;"
					"only expected an entry at position %zd.\n", cur_index, i );
				error = 13;
			}
		}

		// do it again, but now using in_place
		if( !error ) {
			rc = grb::clear( y );
			if( rc != grb::SUCCESS ) {
				(void)fprintf( stderr, "Unexpected return code from grb::clear (y): %d.\n", (int)rc );
				error = 14;
			}
		}
		if( !error ) {
			rc = grb::clear( sparse_x );
			if( rc != grb::SUCCESS ) {
				(void)fprintf( stderr, "Unexpected return code from grb::clear (sparse_x): %d.\n", (int)rc );
				error = 15;
			}
		}
		if( !error ) {
			rc = grb::setElement( sparse_x, data1[ i ], i );
			if( rc != grb::SUCCESS ) {
				(void)fprintf( stderr, "Unexpected return code from grb::set (sparse_x: %d.\n", (int)rc );
				error = 16;
			}
		}
		if( !error ) {
			rc = grb::mxv( y, m, A, sparse_x, integers );
			if( rc != grb::SUCCESS ) {
				(void)fprintf( stderr, "Unpexpected return code from grb::mxv (in_place): %d.\n", (int)rc );
				error = 17;
			}
		}

		// check sparsity
		if( !error && grb::nnz( y ) != 1 ) {
			(void)fprintf( stderr, "Unexpected number of nonzeroes in y: %zd (expected 1).\n", grb::nnz( y ) );
			error = 18;
		}

		// check value
		for( const auto &pair : y ) {
			const size_t cur_index = pair.first;
			const int against = pair.second;
			if( !error && cur_index == i && !grb::utils::equals( chk[ i ], against ) ) {
				(void)fprintf( stderr,
					"Output vector element mismatch at position %zd: %d does not equal "
					"%d.\n",
					i, chk[ i ], against
				);
				error = 19;
			}
			if( !error && cur_index != i ) {
				(void)fprintf( stderr, "Expected no ouput vector element at position %zd;"
					"only expected an entry at position %zd.\n", cur_index, i );
				error = 20;
			}
		}
	}
}

int main( int argc, char ** argv ) {
	(void) argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	// sanity check against metabugs
	int error = 0;
	for( size_t i = 0; i < 15; ++i ) {
		if( ! grb::utils::equals( data1[ i ] * data2[ i ], chk[ i ] ) ) {
			std::cerr << "Sanity check error at position " << i << ": " << data1[ i ]
				<< " + " << data2[ i ] << " does not equal " << chk[ i ] << ".\n";
			error = 1;
		}
	}

	if( !error ) {
		grb::Launcher< AUTOMATIC > launcher;
		if( launcher.exec( &grbProgram, error, error ) != grb::SUCCESS ) {
			std::cerr << "Fatal error: could not launch test.\n";
			error = 2;
		}
	}

	if( !error ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}

	// done
	return error;
}

