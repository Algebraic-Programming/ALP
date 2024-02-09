
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

#include "graphblas.hpp"


using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static const size_t D[ 15 ] = { 4, 1, 4, 1, 9, 7, 7, 9, 0, 2, 14, 13, 13, 12, 12 };
static const int ddata[ 15 ] = { 3, 13, 6, 0, 8, 0, 0, 8, 0, 11, 0, 0, 15, 6, 7 };

void grbProgram( const void *, const size_t in_size, int &error ) {
	error = 0;

	if( in_size != 0 ) {
		std::cerr << "Unit tests called with unexpected input" << std::endl;
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
		std::cerr << "Unexpected return code from Vector build (x): "
			<< static_cast< int >(rc) << "." << std::endl;
		error = 10;
	} else {
		if( grb::nnz( x ) != 15 ) {
			std::cerr << "Unexpected number of elements in x: "
				<< grb::nnz( x ) << "." << std::endl;
			error = 15;
		}
		for( const auto &pair : x ) {
			if( data1[ pair.first ] != pair.second ) {
				std::cerr << "Unexpected value " << pair.second << " "
					<< "at position " << pair.first << ", "
					<< "expected " << data1[ pair.first ] << "."
					<< std::endl;
				error = 17;
			}
		}
	}

	// initialise y
	if( !error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVector< grb::descriptors::no_duplicates >( y,
			iterator_ind, iterator_ind + 15,
			iterator_val, iterator_val + 15,
			SEQUENTIAL
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (y): "
				<< static_cast< int >(rc) << std::endl;
			error = 20;
		} else {
			if( grb::nnz( y ) != 15 ) {
				std::cerr << "Unexpected number of elements in y: "
					<< grb::nnz( y ) << std::endl;
				error = 22;
			}
			for( const auto &pair : y ) {
				if( data1[ pair.first ] != pair.second ) {
					std::cerr << "Unexpected value " << pair.second << " "
						<< "at position  " << pair.first << ", "
						<< "expected " << data1[ pair.first ] << "."
						<< std::endl;
					error = 25;
				}
			}
		}
	}
	// initialise z
	if( ! error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVectorUnique( z,
			iterator_ind, iterator_ind + 15,
			iterator_val, iterator_val + 15,
			SEQUENTIAL
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (z): "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 30;
		} else {
			if( grb::nnz( z ) != 15 ) {
				std::cerr << "Unexpected number of elements in z: "
					<< grb::nnz( z ) << "." << std::endl;
				error = 32;
			}
			for( const auto &pair : z ) {
				if( data1[ pair.first ] != pair.second ) {
					std::cerr << "Unexpected value " << pair.second << " "
						<< "at position " << pair.first << ", "
						<< "expected " << data1[ pair.first ] << "."
						<< std::endl;
					error = 35;
				}
			}
		}
	}

	// initialise x with possible duplicates (overwrite)
	if( !error ) {
		rc = grb::set( x, 9 );
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVector( x,
			iterator_ind, iterator_ind + 15,
			iterator_val, iterator_val + 15,
			SEQUENTIAL
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (x, "
				<< "with possible duplicates, overwrite): "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 40;
		} else {
			if( grb::nnz( x ) != 15 ) {
				std::cerr << "Unexpected number of elements in x: "
				       << grb::nnz( x ) << ", expected 15." << std::endl;
				error = 42;
			}
			for( const auto &pair : x ) {
				if( data1[ pair.first ] != pair.second ) {
					std::cerr << "Unexpected value " << pair.second << " "
						<< "at position " << pair.first << ", "
						<< "expected " << data1[ pair.first ] << "."
						<< std::endl;
					error = 45;
				}
			}
		}
	}

	// initialise x with possible duplicates (add)
	if( !error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVector( x,
			iterator_ind, iterator_ind + 15,
			iterator_val, iterator_val + 15,
			SEQUENTIAL,
			grb::operators::add< int >()
		);
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (x, "
				<< "with possible duplicates, add): "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 50;
		} else {
			if( grb::nnz( x ) != 15 ) {
				std::cerr << "Unexpected number of elements in x: "
					<< grb::nnz( x ) << ", expected 15." << std::endl;
				error = 52;
			}
			for( const auto &pair : x ) {
				if( 2 * data1[ pair.first ] != pair.second ) {
					std::cerr << "Unexpected value " << pair.second
						<< " at position " << pair.first << ", "
						<< " expected " << data1[ pair.first ]
						<< "." << std::endl;
					error = 55;
				}
			}
		}
	}
	// initialise x with possible duplicates (add into cleared)
	if( !error ) {
		rc = grb::clear( x );
		if( rc != SUCCESS ) {
			std::cerr << "Unexpected return code from grb::clear: "
				<< static_cast< int >(rc) << std::endl;
			error = 60;
		}
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( D[ 0 ] );
		if( rc == SUCCESS ) {
			rc = grb::buildVector( x,
				iterator_ind, iterator_ind + 15,
				iterator_val, iterator_val + 15,
				SEQUENTIAL,
				grb::operators::add< int >()
			);
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from Vector build (x, "
				<< "with possible duplicates, add into cleared): "
				<< static_cast< int >(rc) << std::endl;
			error = 61;
		} else {
			if( grb::nnz( x ) != 9 ) {
				std::cerr << "Unexpected number of elements in x: "
					<< grb::nnz( x ) << ", expected 9." << std::endl;
				error = 62;
			}
			for( const auto &pair : x ) {
				if( ddata[ pair.first ] == 0 ) {
					std::cerr << "Unexpected entry (" << pair.first << ", "
						<< pair.second << "); expected no entry here"
						<< std::endl;
					error = 65;
				} else if( ddata[ pair.first ] != pair.second ) {
					std::cerr << "Unexpected entry (" << pair.first << ", "
						<< pair.second << "); expected (" << pair.first
						<< ", " << ddata[ pair.first ] << ")\n";
					error = 67;
				}
			}
		}
	}

	// check illegal duplicate input (1)
	if( !error ) {
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( I[ 0 ] );
		rc = grb::buildVectorUnique( x,
			iterator_ind, iterator_ind + 15,
			iterator_val, iterator_val + 15,
			SEQUENTIAL
		);
		if( rc != grb::ILLEGAL ) {
			std::cerr << "Unexpected return code from Vector build (x, with "
				<< "duplicates (1), while promising no duplicates exist): "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 70;
		} else {
			rc = SUCCESS;
		}
	}

	// check illegal duplicate input (2)
	if( ! error ) {
		rc = grb::clear( x );
		if( rc != SUCCESS ) {
			std::cerr << "Unexpected return code " << static_cast< int >(rc)
				<< " on grb::clear (check illegal duplicate input (2))"
				<< std::endl;
			error = 80;
		}
		const int * iterator_val = &( data1[ 0 ] );
		const size_t * iterator_ind = &( D[ 0 ] );
		if( rc == SUCCESS ) {
			rc = grb::buildVectorUnique( x,
				iterator_ind, iterator_ind + 15,
				iterator_val, iterator_val + 15,
				SEQUENTIAL
			);
		}
		if( rc != grb::ILLEGAL ) {
			std::cerr << "Unexpected return code from Vector build (x, with "
				<< "duplicates (2), while promising no duplicates exist): "
				<< static_cast< int >(rc) << "." << std::endl;
			error = 85;
		} else {
			rc = SUCCESS;
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << std::endl;

	int error;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, error ) != SUCCESS ) {
		std::cout << "Test FAILED (test failed to launch)" << std::endl;
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED" << std::endl;
	}

	// done
	return error;
}

