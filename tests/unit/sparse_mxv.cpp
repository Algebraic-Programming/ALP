
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

#include <graphblas.hpp>
#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };

void grbProgram( const int &, int &error ) {
	RC rc = SUCCESS;
	// allocate
	grb::Vector< int > x( 15 );
	grb::Vector< int > sparse_x( 15 );
	grb::Matrix< int > A = factory::diag< int >( 15, 15, data2, data2 + 15 );

	// initialise x
	if( !error ) {
		const int * iterator = &(data1[ 0 ]);
		rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (x): "
				<< grb::toString( rc ) << ".\n";
			error = 4;
		}
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
			std::cerr << "Unexpected number of nonzeroes in y: " << grb::nnz( y )
				<< " (expected 0).\n";
			error = 6;
		}
		if( !error && grb::nnz( m ) != 0 ) {
			std::cerr << "Unexpected number of nonzeroes in m: " << grb::nnz( m )
				<< " (expected 0).\n";
			error = 7;
		}

		// initialise mask
		if( !error ) {
			rc = grb::setElement( m, true, i );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from vector set (m[" << i << "]): "
					<< grb::toString( rc ) << ".\n";
				error = 8;
			}
		}
		if( !error ) {
			if( grb::nnz( m ) != 1 ) {
				std::cerr << "Unexpected number of nonzeroes in m: " << grb::nnz( m )
					<< "(expected 1).\n";
				error = 9;
			}
		}

		// execute what amounts to elementwise vector addition
		if( !error ) {
			rc = grb::mxv( y, m, A, x, integers );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from grb::vxm: "
				       << grb::toString( rc ) << ".\n";
				error = 10;
			}
		}

		// check sparsity
		if( !error && grb::nnz( y ) != 1 ) {
			std::cerr << "Unexpected number of nonzeroes in y: " << grb::nnz( y )
				<< " (expected 1).\n";
			error = 11;
		}

		// check value
		for( const auto &pair : y ) {
			const size_t cur_index = pair.first;
			const int against = pair.second;
			if( !error && cur_index == i && !grb::utils::equals( chk[ i ], against ) ) {
				std::cerr << "Output vector element mismatch at position " << i << ": "
					<< chk[ i ] << " does not equal " << against << ".\n";
				error = 12;
			}
			if( !error && cur_index != i ) {
				std::cerr << "Expected no ouput vector element at position " << cur_index
					<< ": only expected an entry at position " << i << ".\n";
				error = 13;
			}
		}

		// do it again, but now using in_place
		if( !error ) {
			rc = grb::clear( y );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from grb::clear (y): "
					<< grb::toString( rc ) << ".\n";
				error = 14;
			}
		}
		if( !error ) {
			rc = grb::clear( sparse_x );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from grb::clear (sparse_x): "
					<< grb::toString( rc ) << ".\n";
				error = 15;
			}
		}
		if( !error ) {
			rc = grb::setElement( sparse_x, data1[ i ], i );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from grb::set (sparse_x: "
					<< grb::toString( rc ) << ".\n";
				error = 16;
			}
		}
		if( !error ) {
			rc = grb::mxv( y, m, A, sparse_x, integers );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unpexpected return code from grb::mxv (in_place): "
					<< grb::toString( rc ) << ".\n";
				error = 17;
			}
		}

		// check sparsity
		if( !error && grb::nnz( y ) != 1 ) {
			std::cerr << "Unexpected number of nonzeroes in y: " << grb::nnz( y )
				<< " (expected 1).\n";
			error = 18;
		}

		// check value
		for( const auto &pair : y ) {
			const size_t cur_index = pair.first;
			const int against = pair.second;
			if( !error && cur_index == i && !grb::utils::equals( chk[ i ], against ) ) {
				std::cerr << "Output vector element mismatch at position " << i << ": "
					<< chk[ i ] << " does not equal " << against << ".\n";
				error = 19;
			}
			if( !error && cur_index != i ) {
				std::cerr << "Expected no ouput vector element at position " << cur_index
					<< ": only expected an entry at position " << i << ".\n";
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
		if( !grb::utils::equals( data1[ i ] * data2[ i ], chk[ i ] ) ) {
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

