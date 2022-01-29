
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

#include <graphblas/utils.hpp> // grb::equals

#include <graphblas.hpp>

using namespace grb;

// nonzero values
static const int data[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };

// diagonal matrix
static size_t I1[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
static size_t J1[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

// matrix with empty rows and columns
static size_t I2[ 15 ] = { 1, 1, 3, 3, 6, 6, 6, 7, 7, 12, 12, 12, 13, 13, 13 };
static size_t J2[ 15 ] = { 0, 1, 4, 5, 8, 10, 11, 11, 12, 9, 11, 14, 2, 10, 14 };
// empty rows: 0, 2, 4, 5, 8, 9, 10, 11, 14
// empty cols: 2, 3, 6, 7, 9, 13

void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	// initialize test
	grb::Matrix< double > A( 15, 15 );
	grb::Matrix< double > B( 15, 15 );
	grb::Matrix< double > C( n, n );
	grb::Matrix< void > D( n, n );
	rc = grb::resize( A, 15 );
	rc = rc ? rc : grb::resize( B, 15 );
	rc = rc ? rc : grb::resize( C, 15 );
	rc = rc ? rc : grb::resize( D, 15 );
	rc = rc ? rc : grb::buildMatrixUnique( A, I1, J1, data, 15, SEQUENTIAL );
	rc = rc ? rc : grb::buildMatrixUnique( B, I2, J2, data, 15, SEQUENTIAL );
	const size_t offset = n - 15;
	for( size_t i = 0; i < 15; ++i ) {
		I1[ i ] += offset;
		I2[ i ] += offset;
		J1[ i ] += offset;
		J2[ i ] += offset;
	}
	rc = rc ? rc : grb::buildMatrixUnique( C, I2, J2, data, 15, SEQUENTIAL );
	rc = rc ? rc : grb::buildMatrixUnique( D, I1, J1, 15, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// test output iteration for A
	size_t count = 0;
	for( const std::pair< std::pair< size_t, size_t >, double > & pair : A ) {
		if( pair.first.first != pair.first.second ) {
			std::cerr << "\tunexpected entry at ( " << pair.first.first << ", " << pair.first.second << " ), value " << pair.second << ": expected diagonal values only\n";
			rc = FAILED;
		} else if( ! grb::utils::template equals< int >( data[ pair.first.first ], pair.second ) ) {
			std::cerr << "\tunexpected entry at ( " << pair.first.first << ", " << pair.first.second << " ), value " << pair.second << ": expected value " << data[ pair.first.first ] << "\n";
			rc = FAILED;
		}
		(void)++count;
	}
	rc = rc ? rc : collectives<>::allreduce( count, grb::operators::add< size_t >() );
	if( count != 15 ) {
		std::cerr << "\tunexpected number of entries ( " << count << " ), expected 15.\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 1 (diagonal 15 x 15 matrix) FAILED\n";
		return;
	}

	// test output iteration for B
	std::vector< size_t > rowCount( 15, 0 ), colCount( 15, 0 );
	for( size_t i = 0; i < 15; ++i ) {
		const size_t row_index = I2[ i ] - offset;
		if( grb::internal::Distribution<>::global_index_to_process_id( row_index, 15, grb::spmd<>::nprocs() ) == spmd<>::pid() ) {
			(void)++( rowCount[ row_index ] );
			(void)++( colCount[ J2[ i ] - offset ] );
		}
	}
	count = 0;
	// serialise output of B to stdout across potentially multiple user processes
	for( size_t k = 0; k < spmd<>::nprocs(); ++k ) {
		if( k == spmd<>::pid() ) {
			for( const std::pair< std::pair< size_t, size_t >, double > & triple : B ) {
				(void)--( rowCount[ triple.first.first ] );
				(void)--( colCount[ triple.first.second ] );
				(void)++count;
				std::cout << "( " << triple.first.first << ", " << triple.first.second << " ): " << triple.second << "\n";
			}
		}
#ifndef NDEBUG
		const auto sync_rc = spmd<>::sync();
		assert( sync_rc == SUCCESS );
#else
		(void) spmd<>::sync();
#endif
	}
	rc = rc ? rc : collectives<>::allreduce( count, grb::operators::add< size_t >() );
	for( size_t i = 0; i < 15; ++i ) {
		if( rowCount[ i ] != 0 ) {
			std::cerr << "\tunexpected row checksum " << rowCount[ i ] << " (expected 0) at row " << i << "\n";
			rc = FAILED;
		}
		if( colCount[ i ] != 0 ) {
			std::cerr << "\tunexpected column checksum " << colCount[ i ] << " (expected 0) at column " << i << "\n";
			rc = FAILED;
		}
	}
	if( count != 15 ) {
		std::cerr << "\tunexpected number of entries " << count << "; expected 15.\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 2 (general 15 x 15 matrix) FAILED\n";
		return;
	}

	// test output iteration for C
	for( size_t i = 0; i < 15; ++i ) {
		assert( I2[ i ] >= offset );
		assert( J2[ i ] >= offset );
		const size_t row_index = I2[ i ];
		if( grb::internal::Distribution<>::global_index_to_process_id( row_index, n, grb::spmd<>::nprocs() ) == spmd<>::pid() ) {
			(void)++( rowCount[ row_index - offset ] );
			(void)++( colCount[ J2[ i ] - offset ] );
		}
	}
	count = 0;
	for( const std::pair< std::pair< size_t, size_t >, double > & triple : C ) {
		assert( triple.first.first >= offset );
		assert( triple.first.second >= offset );
		(void)--( rowCount[ triple.first.first - offset ] );
		(void)--( colCount[ triple.first.second - offset ] );
		(void)++count;
		std::cout << "( " << ( triple.first.first - offset ) << ", " << ( triple.first.second - offset ) << " ): " << triple.second << "\n";
	}
	rc = rc ? rc : collectives<>::allreduce( count, grb::operators::add< size_t >() );
	for( size_t i = 0; i < 15; ++i ) {
		if( rowCount[ i ] != 0 ) {
			std::cerr << "\tunexpected row checksum " << rowCount[ i ] << " (expected 0) at row " << i << "\n";
			rc = FAILED;
		}
		if( colCount[ i ] != 0 ) {
			std::cerr << "\tunexpected column checksum " << colCount[ i ] << " (expected 0) at column " << i << "\n";
			rc = FAILED;
		}
	}
	if( count != 15 ) {
		std::cerr << "\tunexpected number of entries " << count << "; expected 15.\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 3 (general " << n << " x " << n << " matrix) FAILED\n";
		return;
	}

	// test output iteration for D
	count = 0;
	for( const std::pair< size_t, size_t > & pair : D ) {
		if( pair.first != pair.second ) {
			std::cerr << "\tunexpected entry ( " << pair.first << ", " << pair.second << " ), expected diagonal values only\n";
			rc = FAILED;
		}
		(void)++count;
	}
	rc = rc ? rc : collectives<>::allreduce( count, grb::operators::add< size_t >() );
	if( count != 15 ) {
		std::cerr << "\tunexpected number of entries ( " << count << " ), expected 15.\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tsubtest 4 (diagonal pattern " << n << " x " << n << " matrix) FAILED\n";
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
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return 255;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
