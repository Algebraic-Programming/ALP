
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

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const int data2[ 15 ] = { 8, 9, 8, 6, 8, 7, 8, 7, 5, 2, 3, 5, 1, 5, 5 };
static const int chk[ 15 ] = { 32, 63, 32, 36, 32, 49, 8, 49, 15, 12, 21, 25, 1, 40, 35 };

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	// sanity check against metabugs
	int error = 0;
	for( size_t i = 0; i < 15; ++i ) {
		if( ! grb::utils::equals( data1[ i ] * data2[ i ], chk[ i ] ) ) {
			(void)printf( "Sanity check error at position %zd: %d + %d does "
						  "not equal %d.\n",
				i, data1[ i ], data2[ i ], chk[ i ] );
			error = 1;
		}
	}

	// initialise
	enum grb::RC rc = grb::init();
	if( rc != grb::SUCCESS ) {
		(void)printf( "Unexpected return code from grb::init: %d.\n", (int)rc );
		error = 2;
	}

	// exit early if failure detected as this point
	if( error ) {
		(void)printf( "Test FAILED.\n\n" );
		//		(void) fflush(stderr);
		//		(void) fflush(stdout);
		return error;
	}

	// allocate
	grb::Vector< int > x( 15 );
	grb::Vector< int > y( 15 );
	grb::Matrix< int > A = factory::identity< int >( 15, SEQUENTIAL, data2 );
	grb::Vector< bool > mask( 15 );

	// initialise x
	if( ! error ) {
		const int * iterator = &( data1[ 0 ] );
		rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from Vector build (x): %d.\n", (int)rc );
			error = 4;
		}
	}

	// check contents of y
	const int * __restrict__ const against = y.raw();

	// get a semiring where multiplication is addition, and addition is multiplication
	// this also tests if the proper identity is used
	typename grb::Semiring< grb::operators::add< int >, grb::operators::mul< int >, grb::identities::zero, grb::identities::one > integers;

	// do masked vxm for 14 different mask combinations
	for( unsigned int i = 0; ! error && i < 15; ++i ) {
		if( i == 3 )
			continue;

		if( ! error ) {
			rc = grb::clear( mask );
			if( rc != grb::SUCCESS ) {
				printf( "Unexpected return code from Vector clear (mask): "
						"%d.\n",
					(int)rc );
				error = 10;
			}
		}

		if( ! error ) {
			rc = grb::clear( y );
			if( rc != grb::SUCCESS ) {
				printf( "Unexpected return code from Vector clear (y): %d.\n", (int)rc );
				error = 11;
			}
		}

		if( ! error ) {
			rc = grb::setElement( mask, true, 3 );
			if( rc != grb::SUCCESS ) {
				printf( "Unexpected return code from Vector set (mask): %d.\n", (int)rc );
				error = 12;
			}
		}

		if( ! error ) {
			rc = grb::setElement( mask, true, i );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from Vector set (mask, "
							  "in-loop): %d.\n",
					(int)rc );
				error = 13;
			}
		}

		// execute what amounts to elementwise vector addition
		if( ! error ) {
			rc = grb::vxm( y, mask, x, A, integers );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from grb::vxm: %d.\n", (int)rc );
				error = 14;
			}
		}

		// check
		if( ! error && ! ( grb::nnz( y ) == 2 ) ) {
			(void)printf( "Output vector number of elements mismatch: %zd, but "
						  "expected 2.\n",
				grb::nnz( y ) );
			error = 15;
		} else if( ! error && ! grb::utils::equals( chk[ 3 ], against[ 3 ] ) ) {
			(void)printf( "Output vector element mismatch at position 3: %d "
						  "does not equal %d.\n",
				chk[ i ], against[ i ] );
			error = 16;
		} else if( ! error && ! grb::utils::equals( chk[ i ], against[ i ] ) ) {
			(void)printf( "Output vector element mismatch at position %d: %d "
						  "does not equal %d.\n",
				i, chk[ i ], against[ i ] );
			error = 17;
		}
		for( auto pair : y ) {
			if( ! error && pair.second ) {
				if( pair.first != 3 && pair.first != i ) {
					printf( "Output vector element %zd is assigned; only %d or "
							"3 should be assigned.\n",
						pair.first, i );
					error = 28;
				}
			}
		}
	}

	// do masked vxm for 14 different mask combinations
	for( unsigned int i = 0; ! error && i < 15; ++i ) {
		if( i == 3 )
			continue;

		if( ! error ) {
			rc = grb::clear( mask );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from Vector clear "
							  "(mask): %d.\n",
					(int)rc );
				error = 20;
			}
		}

		if( ! error ) {
			rc = grb::clear( y );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from Vector clear (y): "
							  "%d.\n",
					(int)rc );
				error = 21;
			}
		}

		if( ! error ) {
			rc = grb::setElement( mask, true, 3 );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from Vector set (mask): "
							  "%d.\n",
					(int)rc );
				error = 22;
			}
		}

		if( ! error ) {
			rc = grb::setElement( mask, true, i );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from Vector set (mask, "
							  "in-loop): %d.\n",
					(int)rc );
				error = 23;
			}
		}

		// execute what amounts to elementwise vector addition
		if( ! error ) {
			rc = grb::vxm( y, mask, x, A, integers );
			if( rc != grb::SUCCESS ) {
				(void)printf( "Unexpected return code from grb::vxm: %d.\n", (int)rc );
				error = 24;
			}
		}

		// check
		if( ! error && ! ( grb::nnz( y ) == 2 ) ) {
			(void)printf( "Output vector number of elements mismatch: %zd, but "
						  "expected 2.\n",
				grb::nnz( y ) );
			error = 25;
		} else if( ! error && ! grb::utils::equals( chk[ 3 ], against[ 3 ] ) ) {
			(void)printf( "Output vector element mismatch at position 3: %d "
						  "does not equal %d.\n",
				chk[ i ], against[ i ] );
			error = 26;
		} else if( ! error && ! grb::utils::equals( chk[ i ], against[ i ] ) ) {
			(void)printf( "Output vector element mismatch at position %d: %d "
						  "does not equal %d.\n",
				i, chk[ i ], against[ i ] );
			error = 27;
		}
		for( auto pair : y ) {
			if( ! error && pair.second ) {
				if( pair.first != 3 && pair.first != i ) {
					(void)printf( "Output vector element %zd is assigned; only "
								  "element %d or 3 should be assigned.\n",
						pair.first, i );
					error = 28;
				}
			}
		}
	}

	// finalize
	rc = grb::finalize();
	if( ! error ) {
		if( rc != grb::SUCCESS ) {
			(void)printf( "Unexpected return code from grb::finalize: %d.\n", (int)rc );
			error = 6;
		}
	}

	if( ! error ) {
		(void)printf( "Test OK.\n\n" );
	} else {
		(void)printf( "Test FAILED.\n\n" );
	}

	// done
	return error;
}
