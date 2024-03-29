
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

#include <graphblas.hpp>

#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;
using namespace grb::algorithms;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };

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

	// initialise
	enum grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from grb::init: "
			<< grb::toString( rc ) << ".\n";
		error = 2;
	}

	// exit early if failure detected as this point
	if( error ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
		return error;
	}

	// allocate
	grb::Vector< int > x( 15 );
	grb::Vector< int > y( 15 );
	grb::Matrix< int > A = matrices< int, grb::SEQUENTIAL >::diag(
		15, 15, data2, data2 + 15 );
	grb::Vector< bool > mask( 15 );

	// initialise x
	if( ! error ) {
		const int * iterator = &(data1[ 0 ]);
		rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (x): "
			       << grb::toString( rc );
			error = 4;
		}
	}

	// check contents of y
	const int * __restrict__ const against = y.raw();

	// Get a standard semiring over integers
	typename grb::Semiring<
		grb::operators::add< int >, grb::operators::mul< int >,
		grb::identities::zero, grb::identities::one
	> integers;

	// do masked vxm for 14 different mask combinations
	for( unsigned int i = 0; !error && i < 15; ++i ) {
		if( i == 3 ) {
			continue;
		}

		if( !error ) {
			rc = grb::clear( mask );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector clear (mask): "
					<< grb::toString( rc ) << ".\n";
				error = 10;
			}
		}

		if( !error ) {
			rc = grb::clear( y );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector clear (y): "
					<< grb::toString( rc ) << ".\n";
				error = 11;
			}
		}

		if( !error ) {
			rc = grb::setElement( mask, true, 3 );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector set (mask): "
					<< grb::toString( rc ) << ".\n";
				error = 12;
			}
		}

		if( !error ) {
			rc = grb::setElement( mask, true, i );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector set (mask, in-loop): "
					<< grb::toString( rc ) << ".\n";
				error = 13;
			}
		}

		// execute what amounts to elementwise vector addition
		if( !error ) {
			rc = grb::vxm( y, mask, x, A, integers );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from grb::vxm: "
				       << grb::toString( rc ) << ".\n";
				error = 14;
			}
		}

		// check
		if( !error && !(grb::nnz( y ) == 2) ) {
			std::cerr << "Output vector number of elements mismatch: " << grb::nnz( y )
			       << ", but expected 2.\n";
			error = 15;
		} else if( !error && !grb::utils::equals( chk[ 3 ], against[ 3 ] ) ) {
			std::cerr << "Output vector element mismatch at position 3: " << chk[ i ]
				<< " does not equal" << against[ i ] <<".\n";
			error = 16;
		} else if( !error && !grb::utils::equals( chk[ i ], against[ i ] ) ) {
			std::cerr << "Output vector element mismatch at position " << i << ": "
				<< chk[ i ] << " does not equal " << against[ i ] << ".\n";
			error = 17;
		}
		for( const auto &pair : y ) {
			if( !error && pair.second ) {
				if( pair.first != 3 && pair.first != i ) {
					std::cerr << "Output vector element " << pair.first << " is assigned; "
						<< "only " << i << " or 3 should be assigned.\n";
					error = 28;
				}
			}
		}
	}

	// do masked vxm for 14 different mask combinations
	for( unsigned int i = 0; !error && i < 15; ++i ) {
		if( i == 3 ) {
			continue;
		}

		if( !error ) {
			rc = grb::clear( mask );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector clear (mask): "
					<< grb::toString( rc ) << ".\n";
				error = 20;
			}
		}

		if( !error ) {
			rc = grb::clear( y );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector clear (y): "
					<< grb::toString( rc ) << ".\n";
				error = 21;
			}
		}

		if( !error ) {
			rc = grb::setElement( mask, true, 3 );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector set (mask): "
					<< grb::toString( rc ) << ".\n";
				error = 22;
			}
		}

		if( !error ) {
			rc = grb::setElement( mask, true, i );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from Vector set (mask, in-loop): "
					<< grb::toString( rc ) << ".\n";
				error = 23;
			}
		}

		// execute what amounts to elementwise vector addition
		if( !error ) {
			rc = grb::vxm( y, mask, x, A, integers );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Unexpected return code from grb::vxm: "
					<< grb::toString( rc ) << ".\n";
				error = 24;
			}
		}

		// check
		if( !error && !(grb::nnz( y ) == 2) ) {
			std::cerr << "Output vector number of elements mismatch: " << grb::nnz(y)
			       << ", but expected 2.\n";
			error = 25;
		} else if( !error && !grb::utils::equals( chk[ 3 ], against[ 3 ] ) ) {
			std::cerr << "Output vector element mismatch at position 3: " << chk[ i ]
				<< " does not equal " << against[ i ] << ".\n";
			error = 26;
		} else if( !error && !grb::utils::equals( chk[ i ], against[ i ] ) ) {
			std::cerr << "Output vector element mismatch at position " << i << ": "
				<< chk[ i ] << " does not equal " << against[ i ] << ".\n";
			error = 27;
		}
		for( const auto &pair : y ) {
			if( !error && pair.second ) {
				if( pair.first != 3 && pair.first != i ) {
					std::cerr << "Output vector element " << pair.first << " is assigned; "
						<< "only element " << i << " or 3 should be assigned.\n";
					error = 28;
				}
			}
		}
	}

	// finalize
	rc = grb::finalize();
	if( !error ) {
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from grb::finalize: "
				<< grb::toString( rc ) << ".\n";
			error = 6;
		}
	}

	if( !error ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::flush;
	}

	// done
	return error;
}

