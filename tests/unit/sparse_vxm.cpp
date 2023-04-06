
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

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <inttypes.h>

#include "graphblas/utils/Timer.hpp"

#include "graphblas.hpp"


using namespace grb;

struct input {
	size_t n;
	size_t test;
	size_t rep;
};

struct output {
	enum grb::RC error_code;
	grb::utils::TimerResults times;
};

template< size_t mode >
static grb::RC setupSparseMatrix(
	grb::Matrix< double > &mx,
	grb::Vector< double > &c,
	const size_t n
) {
	std::vector< double > chk;
	// number of elements
	const size_t elems = n * 5;
	chk.resize( n );
	std::fill( chk.begin(), chk.end(), 0 );
	// reserve space in mx
	grb::RC rc = grb::resize( mx, elems );
	if( rc != SUCCESS ) {
		return rc;
	}

	// generate data
	size_t * I = nullptr, *J = nullptr;
	double * mxValues = nullptr;
	I = new size_t[ elems ];
	J = new size_t[ elems ];
	mxValues = new double[ elems ];
	const size_t step = ( n - 1 ) / 5;
	for( size_t e = 0; e < elems; ) {
		for( size_t i = 0; e < elems && i < 5; ++i, ++e ) {
			I[ e ] = e / 5;
			J[ e ] = ( e / 5 + i * step ) % n;
			mxValues[ e ] = static_cast< double >( e + 1 ) / static_cast< double >( elems );
			assert( I[ e ] < n );
			assert( J[ e ] < n );
			std::cout << I[ e ] << " " << J[ e ] << " " << mxValues[ e ] << "\n";
			if( ( mode == 1 || mode == 4 ) && J[ e ] == n / 2 ) {
				chk[ I[ e ] ] += mxValues[ e ];
			} else if( ( mode == 2 || mode == 3 ) && I[ e ] == n / 2 ) {
				chk[ J[ e ] ] += mxValues[ e ];
			}
		}
	}

	// load into GraphBLAS
	rc = grb::buildMatrixUnique( mx, &(I[ 0 ]), &(J[ 0 ]), mxValues, elems, SEQUENTIAL );
	if( rc == SUCCESS && elems != nnz( mx ) ) {
		rc = PANIC;
	}
	if( rc == SUCCESS ) {
		rc = grb::buildVector( c, chk.begin(), chk.end(), SEQUENTIAL );
	}
	if( rc == SUCCESS && nnz( c ) != n ) {
		rc = PANIC;
	}

	// free data
	delete[] I;
	delete[] J;
	delete[] mxValues;

	// done
	return rc;
}

static enum grb::RC checkResult( const grb::Vector< double > &left, const grb::Vector< double > &right ) {
	std::cout << "checkResult called on the following two vectors:\n";
	std::cout << "\tLeft vector (" << nnz( left ) << "/" << size( left ) << ") reads:\n";
	for( const std::pair< size_t, double > &pair : left ) {
		std::cout << "\t\t" << pair.first << " " << pair.second << "\n";
	}
	std::cout << "\tRight vector (" << nnz( right ) << "/" << size( right ) << ") reads:\n";
	for( const std::pair< size_t, double > &pair : right ) {
		std::cout << "\t\t" << pair.first << " " << pair.second << "\n";
	}
	enum grb::RC ret = SUCCESS;
	if( grb::size( left ) != grb::size( right ) ) {
		std::cout << "Left vector does not equal the size of the right vector.\n";
		return FAILED;
	}
	if( grb::nnz( left ) != grb::size( left ) ) {
		std::cout << "Left vector is not dense.\n";
		return FAILED;
	}
	grb::Vector< double > diff( grb::size( left ) );
	grb::Monoid< grb::operators::add< double >, grb::identities::zero > addMonoid;
	if( ret == SUCCESS ) {
		ret = grb::set( diff, left );
	}
	if( ret == SUCCESS ) {
		ret = grb::eWiseLambda( [ &diff, &right ]( const size_t i ) {
				diff[ i ] = std::abs( diff[ i ] - right[ i ] );
			},
			right, diff
		);
	}
	std::cout << "Difference vector (" << nnz( diff ) << "/" << size( diff ) << ") reads:\n";
	for( const std::pair< size_t, double > &pair : diff ) {
		std::cout << "\t" << pair.first << " " << pair.second << "\n";
	}
	if( ret == SUCCESS ) {
		double equal = 0;
		ret = grb::foldl( equal, diff, NO_MASK, addMonoid );
		if( ret == SUCCESS && std::abs( equal ) > std::numeric_limits< double >::epsilon() ) {
			std::cout << "The difference vector has 1-norm " << equal << "!\n";
			ret = FAILED;
		}
	}
	return ret;
}

// main label propagation algorithm
void grbProgram( const struct input &data_in, struct output &out ) {
	grb::utils::Timer timer;

	const size_t s = spmd<>::pid();
#ifndef NDEBUG
	assert( s < spmd<>::nprocs() );
#else
	(void)s;
#endif

	// get input n and test case
	const size_t n = data_in.n;
	const size_t test = data_in.test;
	out.error_code = SUCCESS;

	// setup
	grb::Vector< double > vx( n ), vy( n ), chk( n );
	grb::Matrix< double > mx( n, n );
	Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	switch( test ) {

		// Ax
		case 1: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::setElement( vx, 1, n / 2 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix< 1 >( mx, chk, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = mxv( vy, mx, vx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// check result
			if( out.error_code == SUCCESS ) {
				out.error_code = checkResult( chk, vy );
			}
			// done
			out.times.postamble = 0;
			break;
		}

		// A^Tx
		case 2: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::setElement( vx, 1, n / 2 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix< 2 >( mx, chk, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = mxv< descriptors::transpose_matrix >( vy, mx, vx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// check result
			if( out.error_code == SUCCESS ) {
				out.error_code = checkResult( chk, vy );
			}
			// done
			out.times.postamble = 0;
			break;
		}

		// xA
		case 3: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::setElement( vx, 1, n / 2 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix< 3 >( mx, chk, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = vxm( vy, vx, mx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// check result
			if( out.error_code == SUCCESS ) {
				out.error_code = checkResult( chk, vy );
			}
			// done
			out.times.postamble = 0;
			break;
		}

		// xA^T
		case 4: {
			// do experiment
			out.times.io = 0;
			timer.reset();
			if( out.error_code == SUCCESS ) {
				out.error_code = grb::setElement( vx, 1, n / 2 );
			}
			if( out.error_code == SUCCESS ) {
				out.error_code = setupSparseMatrix< 4 >( mx, chk, n );
			}
			out.times.preamble = timer.time();
			timer.reset();
			for( size_t i = 0; out.error_code == SUCCESS && i < data_in.rep; ++i ) {
				out.error_code = vxm< descriptors::transpose_matrix >( vy, vx, mx, ring );
			}
			out.times.useful = timer.time() / static_cast< double >( data_in.rep );
			// check result
			if( out.error_code == SUCCESS ) {
				out.error_code = checkResult( chk, vy );
			}
			// done
			out.times.postamble = 0;
			break;
		}

		default:
			std::cerr << "Unknown test case " << test << std::endl;
			break;
	}
}

// main function will execute in serial or as SPMD
int main( int argc, char ** argv ) {
	// sanity check
	if( argc < 3 || argc > 5 ) {
		std::cout << "Usage: " << argv[ 0 ] << " <problem size> <test case> "
			<< "(inner repititions) (outer repititions)" << std::endl;
		return 0;
	}
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	// the input struct
	struct input in;
	in.n = atoi( argv[ 1 ] );
	in.test = atoi( argv[ 2 ] );
	in.rep = grb::config::BENCHMARKING::inner();
	size_t outer = grb::config::BENCHMARKING::outer();
	char * end = NULL;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument for number of inner "
				<< "repititions." << std::endl;
			return 25;
		}
	}
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument for number of outer "
				<< "reptitions." << std::endl;
			return 25;
		}
	}

	std::cout << "Executable called with parameters: problem size " << in.n << " test case ";
	switch( in.test ) {
		case 1:
			std::cout << "Ax";
			break;
		case 2:
			std::cout << "A^Tx";
			break;
		case 3:
			std::cout << "xA";
			break;
		case 4:
			std::cout << "xA^T";
			break;
		default:
			std::cout << " UNRECOGNISED TEST CASE, ABORTING.\nTest FAILED.\n" << std::endl;
			return 30;
	}
	std::cout << ", inner = " << in.rep << ", outer = " << outer << "." << std::endl;

	// the output struct
	struct output out;

	// run the program one time to infer number of inner repititions
	if( in.rep == 0 ) {
		in.rep = 1;
		grb::Launcher< AUTOMATIC > launcher;
		const enum grb::RC rc = launcher.exec( &grbProgram, in, out, true );
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< (int)rc << std::endl;
			return 40;
		}
		// set guesstimate for inner repititions: a single experiment should take at least a second
		in.rep = static_cast< double >( 1000.0 / out.times.useful ) + 1;
		std::cout << "Auto-selected number of inner repetitions is "
			<< in.rep << " (at an estimated time of "
			<< out.times.useful << " ms. of useful work per benchmark).\n";
	}

	// start benchmarks
	grb::Benchmarker< AUTOMATIC > benchmarker;
	const enum grb::RC rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	if( rc != SUCCESS ) {
		std::cerr << "launcher.exec returns with non-SUCCESS error code "
			<< (int)rc << std::endl;
		return 50;
	}

	// done
	if( out.error_code != SUCCESS ) {
		std::cout << "Test FAILED.\n" << std::endl;
		std::cerr << std::flush;
		return out.error_code;
	}
	std::cout << "Test OK.\n" << std::endl;
	return 0;
}

