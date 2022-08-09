
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

#include <cstdlib>
#include <inttypes.h>

#include <graphblas/algorithms/knn.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>

#include <graphblas.hpp>


using namespace grb;
using namespace algorithms;

enum Test {
	EMPTY,
	UNPOPULATED,
	ZERO_CAP,
	DENSE,
	DENSE_CLEARED,
	/** \internal Most sparse, but not totally devoid of entries */
	MOST_SPARSE,
	MOST_SPARSE_CLEARED,
	SPARSE_RANDOM,
	/** \internal Least sparse, but not dense */
	LEAST_SPARSE
};

static const enum Test AllTests[] = {
	EMPTY,
	UNPOPULATED,
	ZERO_CAP,
	DENSE,
	DENSE_CLEARED,
	MOST_SPARSE,
	MOST_SPARSE_CLEARED,
	SPARSE_RANDOM,
	LEAST_SPARSE
};

constexpr const size_t n = 100009;

template< typename T >
struct input {
	enum Test test;
	T element;
};

template< typename T >
struct output {
	RC error_code;
	PinnedVector< T > vector;
};

template< typename T >
static inline bool checkDense( const size_t &i, const T &val, const T &chk ) {
	if( i >= n ) {
		std::cerr << "Nonzero with index " << i << ", while "
			<< "vector size is " << n << "\n";
		return false;
	}
	if( val != chk ) {
		std::cerr << "Nonzero has unexpected value\n";
		return false;
	}
	return true;
}

template< typename T >
static inline bool checkSparse(
	const size_t &i, const T &val,
	const T &chk, const enum Test &test
) {
	if( val != chk ) {
		std::cerr << "Nonzero has unexpected value\n";
		return false;
	}
	switch( test ) {
		case MOST_SPARSE:
			if( i != n/2 ) {
				std::cerr << "Nonzero at position " << i << ", expected " << n/2 << "\n";
				return false;		
			}
			break;
		case SPARSE_RANDOM:
			if( i > n ) {
				std::cerr << "Nonzero at invalid position " << i << "; "
					<< "vector size is " << n << "\n";
				return false;
			}
			break;
		case LEAST_SPARSE:
			if( i == n/2 ) {
				std::cerr << "Nonzero at position " << i << ", while none should be here\n";
				return false;
			}
			break;
		default:
			std::cerr << "This could should not be reached\n";
			assert( false );
			return false;
	}
	return true;
}

template< typename T >
void grbProgram( const struct input< T > &in, struct output< T > &out ) {
	// create container 
	constexpr const size_t zero = 0;
	Vector< T > empty( zero ), nonempty( n ), zero_cap( n, zero );
	srand( 15124 );
	RC rc = SUCCESS;

	// print progress
	switch( in.test ) {
		case EMPTY:
			std::cout << "\t Testing empty vectors...\n";
			break;
		case UNPOPULATED:
			std::cout << "\t Testing unpopulated vectors...\n";
			break;
		case ZERO_CAP:
			std::cout << "\t Testing zero-capacity vectors...\n";
			break;
		case DENSE:
			std::cout << "\t Testing dense vectors...\n";
			break;
		case DENSE_CLEARED:
			std::cout << "\t Testing cleared vectors...\n";
			break;
		case MOST_SPARSE:
			std::cout << "\t Testing sparse vector with one entry...\n";
			break;
		case MOST_SPARSE_CLEARED:
			std::cout << "\t Testing cleared vectors (from sparse)...\n";
			break;
		case SPARSE_RANDOM:
			std::cout << "\t Testing sparse vector with "
				<< "randomly positioned entries...\n";
			break;
		case LEAST_SPARSE:
			std::cout << "\t Testing sparse vector with only one unset entry...\n";
			break;
		default:
			assert( false );
	}

	// initialise
	switch( in.test ) {
		case DENSE:
		case DENSE_CLEARED:
			{
				rc = grb::set( nonempty, in.element );
				break;
			}
		case MOST_SPARSE:
		case MOST_SPARSE_CLEARED:
			{
				rc = grb::setElement( nonempty, in.element, n/2 );
				break;
			}
		case SPARSE_RANDOM:
			{
				for( size_t i = 0; i < n; ++i ) {
					if( rand() % 10 == 0 ) {
						rc = rc ? rc : grb::setElement( nonempty, in.element, i );
					}
				}
				break;
			}
		case LEAST_SPARSE:
			{
				Vector< bool > mask( n );
				rc = grb::setElement( mask, true, n/2 );
				rc = rc ? rc : grb::set<
						grb::descriptors::invert_mask
					>( nonempty, mask, in.element );
				break;
			}
		default:
			assert(
				in.test == EMPTY ||
				in.test == UNPOPULATED ||
				in.test == ZERO_CAP
			);
	}

	// clear if requested
	if(
		rc == SUCCESS && (
			in.test == DENSE_CLEARED ||
			in.test == MOST_SPARSE_CLEARED
		)
	) {
		rc = grb::clear( nonempty );
	}

	// return as a pinnedVector
	if( rc == SUCCESS ) {
		switch( in.test ) {
			case EMPTY:
				out.vector = PinnedVector< T >( empty, SEQUENTIAL );
				break;
			case UNPOPULATED:
			case DENSE:
			case DENSE_CLEARED:
			case MOST_SPARSE:
			case MOST_SPARSE_CLEARED:
			case SPARSE_RANDOM:
			case LEAST_SPARSE:
				out.vector = PinnedVector< T >( nonempty, SEQUENTIAL );
				break;
			case ZERO_CAP:
				out.vector = PinnedVector< T >( zero_cap, SEQUENTIAL );
			break;
			default:
				assert( false );
		}
	}

	// done
	out.error_code = rc;
	return;
}

template< typename T >
int runTests( struct input< T > &in ) {
	struct output< T > out;
	Launcher< AUTOMATIC > launcher;
	RC rc = SUCCESS;
	int offset = 0;

	// for every test
	for( const auto &test : AllTests ) {
		// run test
		in.test = test;
		rc = rc ? rc : launcher.exec( &grbProgram, in, out );
		if( out.error_code != SUCCESS ) {
			return offset + 10;
		}

		// check size of output vector
		switch( test ) {
			case EMPTY:
				if( out.vector.size() != 0 ) {
					std::cerr << "Empty pinned vector has nonzero size\n";
					rc = FAILED;
				}
				break;
			case UNPOPULATED:
			case ZERO_CAP:
			case DENSE:
			case DENSE_CLEARED:
			case MOST_SPARSE:
			case MOST_SPARSE_CLEARED:
			case SPARSE_RANDOM:
			case LEAST_SPARSE:
				if( out.vector.size() != n ) {
					std::cerr << "Vector does not have expected capacity\n";
					rc = FAILED;
				}
				break;
			default:
				assert( false );
		}
		if( rc != SUCCESS ) {
			return offset + 20;
		}

		// check number of nonzeroes
		switch( test ) {
			case EMPTY:
			case UNPOPULATED:
			case ZERO_CAP:
			case DENSE_CLEARED:
			case MOST_SPARSE_CLEARED:
				if( out.vector.nonzeroes() != 0 ) {
					std::cerr << "Pinned vector has nonzeroes ( " << out.vector.nonzeroes()
						<< " ), but none were expected\n";
					rc = FAILED;
				}
				break;
			case DENSE:
				if( out.vector.nonzeroes() != n ) {
					std::cerr << "Pinned vector has less than the expected number of "
						<< "nonzeroes ( " << out.vector.nonzeroes() << ", expected " << n
						<< " ).\n";
					rc = FAILED;
				}
				break;
			case MOST_SPARSE:
				if( out.vector.nonzeroes() != 1 ) {
					std::cerr << "Pinned vector has " << out.vector.nonzeroes()
						<< " nonzeroes, expected 1\n";
					rc = FAILED;
				}
				break;
			case SPARSE_RANDOM:
				if( out.vector.nonzeroes() > n ) {
					std::cerr << "Pinned vector has too many nonzeroes\n";
					rc = FAILED;
				}
				break;
			case LEAST_SPARSE:
				if( out.vector.nonzeroes() != n - 1 ) {
					std::cerr << "Pinned vector has " << out.vector.nonzeroes()
						<< ", but should have " << (n-1) << "\n";
					rc = FAILED;
				}
				break;
			default:
				assert( false );
		}
		if( rc != SUCCESS ) {
			return offset + 30;
		}

		// check nonzero contents via API
		for( size_t k = 0; rc == SUCCESS && k < out.vector.nonzeroes(); ++k ) {
			const size_t index = out.vector.getNonzeroIndex( k );
			const T value = out.vector.getNonzeroValue( k );
			switch( test ) {
				case EMPTY:
				case UNPOPULATED:
				case ZERO_CAP:
				case DENSE_CLEARED:
					std::cerr << "Iterating over nonzeroes, while none should exist (I)\n";
					rc = FAILED;
					break;
				case DENSE:
					if( !checkDense( index, value, in.element ) ) {
						rc = FAILED;
					}
					break;
				case MOST_SPARSE:
				case SPARSE_RANDOM:
				case LEAST_SPARSE:
					if( !checkSparse( index, value, in.element, test ) ) {
						rc = FAILED;
					}
					break;
				default:
					assert( false );
			}

		}
		if( rc != SUCCESS ) {
			return offset + 40;
		}

		// check nonzero contents via iterator
		// (TODO: this is not yet implemented in PinnedVector-- should we?)
		/*for( const auto &nonzero : out.vector ) {
			switch( test ) {
				case EMPTY:
				case UNPOPULATED:
				case ZERO_CAP:
				case DENSE_CLEARED:
					std::cerr << "Iterating over nonzeroes, while none should exist (II)\n";
					rc = FAILED;
					break;
				case DENSE:
					if( !checkDense( nonzero.first, nonzero.second, in.element ) ) {
						rc = FAILED;
					}
					break;
				case MOST_SPARSE:
				case SPARSE_RANDOM:
				case LEAST_SPARSE:
					if( !checkSparse( nonzero.first, nonzero.second, in.element, test ) ) {
						rc = FAILED;
					}
					break;
				default:
					assert( false );
			}
			if( rc != SUCCESS ) { break; }
		}
		if( rc != SUCCESS ) {
			return offset + 50;
		}*/

		offset += 60;
	}
	// done
	return 0;
}

int main( int argc, char ** argv ) {
	// sanity check
	if( argc != 1 ) {
		std::cout << "Usage: " << argv[ 0 ] << std::endl;
		return 0;
	}

	std::cout << "Test executable: " << argv[ 0 ] << "\n";

	// run some tests using a standard elementary type
	std::cout << "Running test with double vector entries...\n";
	struct input< double > in_double;
	in_double.element = 3.1415926535;
	int error = runTests( in_double );

	// run tests using a non-fundamental type
	if( error == 0 ) {
		std::cout << "Running test with std::pair vector entries...\n";
		struct input< std::pair< size_t, float > > in_pair;
		in_pair.element = std::make_pair< size_t, float >( 17, -2.7 );
		error = runTests( in_pair );
	}

	// done
	if( error ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
		return error;
	} else {
		std::cout << "Test OK\n" << std::endl;
		return 0;
	}
}

