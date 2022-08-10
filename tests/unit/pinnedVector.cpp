
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
	LEAST_SPARSE,
	LEAST_SPARSE_CLEARED
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
	LEAST_SPARSE,
	LEAST_SPARSE_CLEARED
};

constexpr const size_t n = 100009;

template< typename T >
struct input {
	enum Test test;
	T element;
	IOMode mode;
};

template< typename T >
struct output {
	RC error_code;
	PinnedVector< T > vector;
};

struct reducer_output {
	RC error_code;
	size_t reduced;
};

void reducer( const size_t &in, struct reducer_output &out ) {
	operators::add< size_t > addOp;
	out.reduced = in;
	out.error_code = collectives<>::allreduce( out.reduced, addOp );
}

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
			std::cout << "\t\t testing empty vectors...\n";
			break;
		case UNPOPULATED:
			std::cout << "\t\t testing unpopulated vectors...\n";
			break;
		case ZERO_CAP:
			std::cout << "\t\t testing zero-capacity vectors...\n";
			break;
		case DENSE:
			std::cout << "\t\t testing dense vectors...\n";
			break;
		case DENSE_CLEARED:
			std::cout << "\t\t testing cleared vectors...\n";
			break;
		case MOST_SPARSE:
			std::cout << "\t\t testing sparse vector with one entry...\n";
			break;
		case MOST_SPARSE_CLEARED:
			std::cout << "\t\t testing cleared vectors (from sparse)...\n";
			break;
		case SPARSE_RANDOM:
			std::cout << "\t\t testing sparse vector with "
				<< "randomly positioned entries...\n";
			break;
		case LEAST_SPARSE:
			std::cout << "\t\t testing sparse vector with only one unset entry...\n";
			break;
		case LEAST_SPARSE_CLEARED:
			std::cout << "\t\t testing cleared vector (from almost-dense)...\n";
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
		case LEAST_SPARSE_CLEARED:
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
			in.test == MOST_SPARSE_CLEARED ||
			in.test == LEAST_SPARSE_CLEARED
		)
	) {
		rc = grb::clear( nonempty );
	}

	// return as a pinnedVector
	if( rc == SUCCESS ) {
		switch( in.test ) {
			case EMPTY:
				out.vector = PinnedVector< T >( empty, in.mode );
				break;
			case UNPOPULATED:
			case DENSE:
			case DENSE_CLEARED:
			case MOST_SPARSE:
			case MOST_SPARSE_CLEARED:
			case SPARSE_RANDOM:
			case LEAST_SPARSE:
			case LEAST_SPARSE_CLEARED:
				out.vector = PinnedVector< T >( nonempty, in.mode );
				break;
			case ZERO_CAP:
				out.vector = PinnedVector< T >( zero_cap, in.mode );
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
			case LEAST_SPARSE_CLEARED:
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

		// get number of nonzeroes
		size_t nzs = out.vector.nonzeroes();
		if( in.mode == PARALLEL ) {
			struct reducer_output redout;
			const RC reducer_error =
				launcher.exec( &reducer, nzs, redout, false );
			if( reducer_error != SUCCESS || redout.error_code != SUCCESS ) {
				std::cerr << "Error recovering the global number of returned nonzeroes\n";
				return offset + 25;
			}
			nzs = redout.reduced;
		}

		// check number of nonzeroes
		switch( test ) {
			case EMPTY:
			case UNPOPULATED:
			case ZERO_CAP:
			case DENSE_CLEARED:
			case MOST_SPARSE_CLEARED:
			case LEAST_SPARSE_CLEARED:
				if( nzs != 0 ) {
					std::cerr << "Pinned vector has nonzeroes ( " << nzs
						<< " ), but none were expected\n";
					rc = FAILED;
				}
				break;
			case DENSE:
				if( nzs != n ) {
					std::cerr << "Pinned vector does not hold the expected number of "
						<< "nonzeroes ( " << nzs << ", expected " << n
						<< " ).\n";
					rc = FAILED;
				}
				break;
			case MOST_SPARSE:
				if( nzs != 1 ) {
					std::cerr << "Pinned vector has " << nzs
						<< " nonzeroes, expected 1\n";
					rc = FAILED;
				}
				break;
			case SPARSE_RANDOM:
				if( nzs > n ) {
					std::cerr << "Pinned vector has too many nonzeroes\n";
					rc = FAILED;
				}
				break;
			case LEAST_SPARSE:
				if( nzs != n - 1 ) {
					std::cerr << "Pinned vector has " << nzs
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
				case MOST_SPARSE_CLEARED:
				case LEAST_SPARSE_CLEARED:
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
				case MOST_SPARSE_CLEARED:
				case LEAST_SPARSE_CLEARED:
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

	int error = 0;
	const IOMode modes[] = { SEQUENTIAL, PARALLEL };
	for( const auto &mode : modes ) {
		std::cout << "Testing pinnedVector in ";
		if( mode == SEQUENTIAL ) {
			std::cout << "SEQUENTIAL ";
		} else if( mode == PARALLEL ) {
			std::cout << "PARALLEL ";
		} else {
			assert( false );
		}
		std::cout << "I/O mode\n";

		// run some tests using a standard elementary type
		std::cout << "\t running tests with double vector entries...\n";
		struct input< double > in_double;
		in_double.element = 3.1415926535;
		in_double.mode = mode;
		error = runTests( in_double );

		// run tests using a non-fundamental type
		if( error == 0 ) {
			std::cout << "\t running tests with std::pair vector entries...\n";
			struct input< std::pair< size_t, float > > in_pair;
			in_pair.element = std::make_pair< size_t, float >( 17, -2.7 );
			in_pair.mode = mode;
			error = runTests( in_pair );
		}
		if( error ) { break; }
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

