
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
#include <string>
#include <type_traits>
#include <vector>
#include <memory>

#include <alp.hpp>

template< typename MatrixType >
void setElements( MatrixType &M, const typename MatrixType::value_type value ) {
	alp::internal::setInitialized( M, true );
	const size_t height = alp::ncols( M );
	const size_t width = alp::nrows( M );
	for( size_t r = 0; r < height; ++r ) {
		for( size_t c = 0; c < width; ++c ) {
			alp::internal::access( M, alp::internal::getStorageIndex( M, r, c ) ) = value;
		}
	}
}

void alp_program( const size_t & n, alp::RC & rc ) {

	const size_t width = 2 * n;
	const size_t height = n;
	std::cout << "\tStarting structured matrices test with size ( H x W ): " << height << " x " << width <<  "\n";
	rc = alp::SUCCESS;

	// create the original matrix
	alp::Matrix< float, alp::structures::General > M( n, n );
	// set matrix elements using the internal interface
	setElements( M, 1 );
	// create transposed view over M
	auto Mt = alp::get_view< alp::view::transpose >( M );
	// create a square view over M and treat it as a matrix view with a square structure
	const size_t block_size = 4;
	auto Mview = alp::get_view( M, alp::utils::range( 0, block_size ), alp::utils::range( 0, block_size ) );
	auto Sq_Mref = alp::get_view< alp::structures::Square > ( Mview );

	// verify that accessing corner elements succeeds
	// original matrix
	alp::internal::access( M, alp::internal::getStorageIndex( M, 0, 0 ) );
	alp::internal::access( M, alp::internal::getStorageIndex( M, height - 1, 0 ) );
	alp::internal::access( M, alp::internal::getStorageIndex( M, 0, width - 1 ) );
	alp::internal::access( M, alp::internal::getStorageIndex( M, height - 1, width - 1 ) );

	// transposed view
	alp::internal::access( Mt, alp::internal::getStorageIndex( Mt, 0, 0 ) );
	alp::internal::access( Mt, alp::internal::getStorageIndex( Mt, width - 1, 0 ) );
	alp::internal::access( Mt, alp::internal::getStorageIndex( Mt, 0, height - 1 ) );
	alp::internal::access( Mt, alp::internal::getStorageIndex( Mt, width - 1, height - 1 ) );

	// square block view
	alp::internal::access( Sq_Mref, alp::internal::getStorageIndex( Sq_Mref, 0, 0 ) );
	alp::internal::access( Sq_Mref, alp::internal::getStorageIndex( Sq_Mref, block_size - 1, 0 ) );
	alp::internal::access( Sq_Mref, alp::internal::getStorageIndex( Sq_Mref, 0, block_size - 1 ) );
	alp::internal::access( Sq_Mref, alp::internal::getStorageIndex( Sq_Mref, block_size - 1, block_size - 1 ) );

	rc = alp::SUCCESS;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 5;

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
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
