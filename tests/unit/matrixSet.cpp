
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

#include <graphblas.hpp>


using namespace grb;

static const int data1[ 15 ] = { 4, 7, 4, 6, 4, 7, 1, 7, 3, 6, 7, 5, 1, 8, 7 };
static const size_t I[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 7, 6 };
static const size_t J[ 15 ] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 5, 7, 5, 1 };

void grb_program( const size_t &n, grb::RC &rc ) {
	// initialize test
	int chk[ 10 ][ 10 ];
	for( size_t i = 0; i < 10; ++i ) {
		for( size_t j = 0; j < 10; ++j ) {
			chk[ i ][ j ] = 0;
		}
	}
	for( size_t k = 0; k < 15; ++k ) {
		chk[ I[ k ] ][ J[ k ] ] = data1[ k ];
	}
	grb::Matrix< double > A( n, n );
	grb::Matrix< double > B( n, n );
	grb::Matrix< void > C( n, n );
	grb::Matrix< void > D( n, n );
	grb::Matrix< unsigned int > E( n, n );
	grb::Matrix< int > Mask( n, n );
	grb::Matrix< int > Output( n, n );
	grb::Matrix< int > Input( n, n );
	

	
	rc = grb::resize( A, 15 );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrixUnique( A, I, J, data1, 15, SEQUENTIAL );
		for( const auto & triplet : A ) {
			if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
				std::cerr << "\tunexpected entry at A( " << triplet.first.first << ", " << triplet.first.second << " ).\n";
				rc = FAILED;
			} else if( chk[ triplet.first.first ][ triplet.first.second ] != triplet.second ) {
				std::cerr << "\tunexpected entry at A( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
				if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
					std::cerr << ", expected no entry here.\n";
				} else {
					std::cerr << ", expected value " << chk[ triplet.first.first ][ triplet.first.second ] << ".\n";
				}
				rc = FAILED;
			}
		}
	}

	size_t I_mask[ 2 * n - 1 ], J_mask[ 2 * n - 1 ];
	int mask_vals [ 2 * n - 1 ];
	int input_vals [ 2 * n - 1 ];

	for( size_t k = 0; k < n; ++k ) {
		I_mask[ k ] = J_mask[ k ] = k;
		mask_vals[ k ] = 1;
		input_vals[ k ] = k;
		if( k < n - 1 ) {
			I_mask[ n + k ] = k;
			J_mask[ n + k ] = k + 1;
			mask_vals [ n + k ] = 0;
			input_vals[ n + k ] = k;
		}
	}

	rc = grb::buildMatrixUnique( Mask, I_mask, J_mask, mask_vals, 2 * n - 1, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildMatrixUnique of mask matrix FAILED\n";
		return;
	}

	rc = grb::buildMatrixUnique( Input, I_mask, J_mask, input_vals, 2 * n - 1, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildMatrixUnique of input matrix FAILED\n";
		return;
	}

	if( rc == SUCCESS ) {
		rc = grb::resize( B, 15 );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize( C, 15 );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize( D, 15 );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize( E, 15 );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize(Output, 2 * n - 1 );
	}
	if( rc != SUCCESS || grb::nnz( A ) != 15 ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// check grb::set for non-voids
	rc = grb::set( B, A );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set FAILED\n";
		return;
	}
	if( grb::nnz( B ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in B ( " << grb::nnz( B ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & triplet : B ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\tunexpected entry at B( " << triplet.first.first << ", " << triplet.first.second << " ).\n";
			rc = FAILED;
		} else if( chk[ triplet.first.first ][ triplet.first.second ] != triplet.second ) {
			std::cerr << "\tunexpected entry at B( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value " << chk[ triplet.first.first ][ triplet.first.second ] << ".\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check grb::set for non-void to void
	rc = grb::set( C, B );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (non-void to void) FAILED\n";
		return;
	}
	if( grb::nnz( C ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in C ( " << grb::nnz( C ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & pair : C ) {
		if( pair.first >= 10 || pair.second >= 10 || chk[ pair.first ][ pair.second ] == 0 ) {
			std::cerr << "\t unexpected entry at C( " << pair.first << ", " << pair.second << " ).\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check grb::set for void-to-void
	rc = grb::set( D, C );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (void to void) FAILED\n";
		return;
	}
	if( grb::nnz( D ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in D ( " << grb::nnz( D ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & pair : D ) {
		if( pair.first >= 10 || pair.second >= 10 || chk[ pair.first ][ pair.second ] == 0 ) {
			std::cerr << "\t unexpected entry at D( " << pair.first << ", " << pair.second << " ).\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check casting grb::set
	rc = grb::set( E, A );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (cast from double to int) FAILED\n";
		return;
	}
	if( grb::nnz( E ) != 15 ) {
		std::cerr << "\t unexpected number of output elements in E ( " << grb::nnz( E ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & triplet : E ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\tunexpected entry at E( " << triplet.first.first << ", " << triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} else if( static_cast< unsigned int >( chk[ triplet.first.first ][ triplet.first.second ] ) != triplet.second ) {
			std::cerr << "\tunexpected entry at E( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value " << chk[ triplet.first.first ][ triplet.first.second ] << ".\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	rc = grb::clear( E );
	rc = rc ? rc : grb::set( E, A, 117.175 );
	if( rc != SUCCESS ) {
		std::cerr << "\tgrb::set (masked set-to-value-while-casting) FAILED\n";
		return;
	}
	if( grb::nnz( E ) != 15 ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( E ) << " ), expected 15.\n";
		rc = FAILED;
	}
	for( const auto & triplet : E ) {
		if( triplet.first.first >= 10 || triplet.first.second >= 10 ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} else if( 117 != triplet.second ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			if( chk[ triplet.first.first ][ triplet.first.second ] == 0 ) {
				std::cerr << ", expected no entry here.\n";
			} else {
				std::cerr << ", expected value 117.\n";
			}
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	//check masked matrix set

	rc = grb::set< descriptors::structural >( Output, Mask, Input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set structural (matrix to matrix masked) FAILED\n";
		return;
	}

	if( grb::nnz( Output ) != 2 * n - 1 ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( Output ) << " ), expected " << 2 * n - 1 <<".\n";
		rc = FAILED;
	}

	for( const auto & triplet : Output ) {
		if( triplet.first.first != triplet.first.second && triplet.first.first != triplet.first.second - 1 ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} if( (int) triplet.first.first != triplet.second ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			std::cerr << ", expected value "<< triplet.first.first <<".\n";
			rc = FAILED;
		}
	}

	if( rc != SUCCESS ) {
		return;
	}

	

	rc = grb::set( Output, Mask, Input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set (matrix to matrix masked) FAILED\n";
		return;
	}

	if( grb::nnz( Output ) != n ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( Output ) << " ), expected " << n <<".\n";
		rc = FAILED;
	}

	for( const auto & triplet : Output ) {
		if( triplet.first.first != triplet.first.second ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} if( (int) triplet.first.first != triplet.second ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			std::cerr << ", expected value "<< triplet.first.first <<".\n";
			rc = FAILED;
		}
	}

	if( rc != SUCCESS ) {
		return;
	}



	rc = grb::set< descriptors::invert_mask >( Output, Mask, Input );
	if( rc != SUCCESS ) {
		std::cerr << "\t grb::set invert mask (matrix to matrix masked) FAILED\n";
		return;
	}

	if( grb::nnz( Output ) != n - 1 ) {
		std::cerr << "\t unexpected number of output elements ( " << grb::nnz( Output ) << " ), expected " << n - 1 <<".\n";
		rc = FAILED;
	}

	for( const auto & triplet : Output ) {
		if( triplet.first.first != triplet.first.second - 1 ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ), value " << triplet.second << ".\n";
			rc = FAILED;
		} if( (int) triplet.first.first != triplet.second ) {
			std::cerr << "\tunexpected entry at ( " << triplet.first.first << ", " << triplet.first.second << " ) with value " << triplet.second;
			std::cerr << ", expected value "<< triplet.first.first <<".\n";
			rc = FAILED;
		}
	}

	if( rc != SUCCESS ) {
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
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
