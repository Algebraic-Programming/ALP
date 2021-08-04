
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

#include "graphblas.hpp"

using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static const size_t D[ 15 ] = { 4, 1, 4, 1, 9, 7, 7, 9, 0, 2, 14, 13, 13, 12, 12 };
static const int ddata[ 15 ] = { 3, 13, 6, 0, 8, 0, 0, 8, 0, 11, 0, 0, 15, 6, 7 };

void grbProgram( const void *, const size_t in_size, int & error ) {
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Vector< int > x( 15 );
	grb::Vector< int > y( 15 );
	grb::Vector< int > z( 15 );

	// initialise x
	const int * iterator = &( data1[ 0 ] );
	grb::RC rc = grb::buildVector( x, iterator, iterator + 15, SEQUENTIAL );
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from Vector build (x): %d.\n", (int)rc );
		error = 10;
	} else {
		if( grb::nnz( x ) != 15 ) {
			(void)fprintf( stderr, "Unexpected number of elements in x: %zd.\n", grb::nnz( x ) );
			error = 15;
		}
		for( const auto & pair : x ) {
			if( data1[ pair.first ] != pair.second ) {
				(void)fprintf( stderr, "Unexpected value %d at position %zd, expected %d\n", pair.second, pair.first, data1[ pair.first ] );
				error = 17;
			}
		}
	}

	// initialise y
	if( ! error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVector< grb::descriptors::no_duplicates >( y, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr, "Unexpected return code from Vector build (y): %d.\n", (int)rc );
			error = 20;
		} else {
			if( grb::nnz( y ) != 15 ) {
				(void)fprintf( stderr, "Unexpected number of elements in y: %zd.\n", grb::nnz( y ) );
				error = 22;
			}
			for( const auto & pair : y ) {
				if( data1[ pair.first ] != pair.second ) {
					(void)fprintf( stderr, "Unexpected value %d at position %zd", pair.second, pair.first );
					(void)fprintf( stderr, ", expected %d\n", data1[ pair.first ] );
					error = 25;
				}
			}
		}
	}
	// initialise z
	if( ! error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVectorUnique( z, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr, "Unexpected return code from Vector build (z): %d.\n", (int)rc );
			error = 30;
		} else {
			if( grb::nnz( z ) != 15 ) {
				(void)fprintf( stderr, "Unexpected number of elements in z: %zd.\n", grb::nnz( z ) );
				error = 32;
			}
			for( const auto & pair : z ) {
				if( data1[ pair.first ] != pair.second ) {
					(void)fprintf( stderr, "Unexpected value %d at position %zd", pair.second, pair.first );
					(void)fprintf( stderr, ", expected %d\n", data1[ pair.first ] );
					error = 35;
				}
			}
		}
	}

	// initialise x with possible duplicates (overwrite)
	if( ! error ) {
		rc = grb::set( x, 9 );
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVector( x, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL );
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr,
				"Unexpected return code from Vector build (x, with possible "
				"duplicates, overwrite): %d.\n",
				(int)rc );
			error = 40;
		} else {
			if( grb::nnz( x ) != 15 ) {
				(void)fprintf( stderr, "Unexpected number of elements in x: %zd (expected 15).\n", grb::nnz( x ) );
				error = 42;
			}
			for( const auto & pair : x ) {
				if( data1[ pair.first ] != pair.second ) {
					(void)fprintf( stderr, "Unexpected value %d at position %zd", pair.second, pair.first );
					(void)fprintf( stderr, ", expected %d\n", data1[ pair.first ] );
					error = 45;
				}
			}
		}
	}

	// initialise x with possible duplicates (add)
	if( ! error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVector( x, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL, grb::operators::add< int >() );
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr,
				"Unexpected return code from Vector build (x, with possible "
				"duplicates, add): %d.\n",
				(int)rc );
			error = 50;
		} else {
			if( grb::nnz( x ) != 15 ) {
				(void)fprintf( stderr, "Unexpected number of elements in x: %zd (expected 15).\n", grb::nnz( x ) );
				error = 52;
			}
			for( const auto & pair : x ) {
				if( 2 * data1[ pair.first ] != pair.second ) {
					(void)fprintf( stderr, "Unexpected value %d at position %zd", pair.second, pair.first );
					(void)fprintf( stderr, ", expected %d\n", data1[ pair.first ] );
					error = 55;
				}
			}
		}
	}
	// initialise x with possible duplicates (add into cleared)
	if( ! error ) {
		rc = grb::clear( x );
		if( rc != SUCCESS ) {
			(void)fprintf( stderr, "Unexpected return code from grb::clear: %d\n", (int)rc );
			error = 60;
		}
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( D[ 0 ] );
		if( rc == SUCCESS ) {
			rc = grb::buildVector( x, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL, grb::operators::add< int >() );
		}
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr,
				"Unexpected return code from Vector build (x, with possible "
				"duplicates, add into cleared): %d.\n",
				(int)rc );
			error = 61;
		} else {
			if( grb::nnz( x ) != 9 ) {
				(void)fprintf( stderr, "Unexpected number of elements in x: %zd (expected 9).\n", grb::nnz( x ) );
				error = 62;
			}
			for( const auto & pair : x ) {
				if( ddata[ pair.first ] == 0 ) {
					(void)fprintf( stderr, "Unexpected entry (%zd, %d); expected no entry here\n", pair.first, pair.second );
					error = 65;
				} else if( ddata[ pair.first ] != pair.second ) {
					(void)fprintf( stderr, "Unexpected entry (%zd, %d); expected (%zd, %d)\n", pair.first, pair.second, pair.first, ddata[ pair.first ] );
					error = 67;
				}
			}
		}
	}

	// check illegal duplicate input (1)
	if( ! error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVectorUnique( x, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL );
		if( rc != grb::ILLEGAL ) {
			(void)fprintf( stderr,
				"Unexpected return code from Vector build (x, with duplicates (1), "
				"while promising no duplicates exist): %d.\n",
				(int)rc );
			error = 70;
		} else {
			rc = SUCCESS;
		}
	}

	// check illegal duplicate input (2)
	if( ! error ) {
		rc = grb::clear( x );
		if( rc != SUCCESS ) {
			(void)fprintf( stderr,
				"Unexpected return code %d on grb::clear (check illegal duplicate "
				"input (2))\n",
				(int)rc );
			error = 80;
		}
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( D[ 0 ] );
		if( rc == SUCCESS ) {
			rc = grb::buildVectorUnique( x, iterator_ind, iterator_ind + 15, iterator_val, iterator_val + 15, SEQUENTIAL );
		}
		if( rc != grb::ILLEGAL ) {
			(void)fprintf( stderr,
				"Unexpected return code from Vector build (x, with duplicates (2), "
				"while promising no duplicates exist): %d.\n",
				(int)rc );
			error = 85;
		} else {
			rc = SUCCESS;
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	int error;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, error ) != SUCCESS ) {
		(void)fprintf( stderr, "Test failed to launch\n" );
		error = 255;
	}
	if( error == 0 ) {
		(void)printf( "Test OK.\n" );
	} else {
		fflush( stderr );
		(void)printf( "Test FAILED.\n" );
	}

	// done
	return error;
}
