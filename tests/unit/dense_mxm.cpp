
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
#include <vector>

#include <alp.hpp>

using namespace alp;

template< typename Structure >
void print_matrix( std::string name, const alp::Matrix< double, Structure > & A) {

	if( ! alp::internal::getInitialized( A ) ) {
		std::cout << "Matrix " << name << " uninitialized.\n";
		return;
	}
	
	for( size_t row = 0; row < alp::nrows( A ); ++row ) {
		for( size_t col = 0; col < alp::ncols( A ); ++col ) {
			std::cout << name << "(" << row << ", " << col << ") ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void alp_program( const size_t & n, alp::RC & rc ) {
	alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

	std::vector< double > A_data( n * n, 1 );
	std::vector< double > B_data( n * n, 1 );
	std::vector< double > C_data( n * n, 0 );

	std::cout << "\tTesting dense General mxm " << n << "\n";
	// initialize test
	alp::Matrix< double, structures::General > A( n, n );
	alp::Matrix< double, structures::General > B( n, n );
	alp::Matrix< double, structures::General > C( n, n );

	// Initialize input matrices
	rc = alp::buildMatrix( A, A_data.begin(), A_data.end() );
	rc = alp::buildMatrix( B, B_data.begin(), B_data.end() );
	rc = alp::buildMatrix( C, C_data.begin(), C_data.end() );

	rc = alp::mxm( C, A, B, ring );

	std::cout << "\n\n=========== Testing Uppertriangular ============\n\n";

	alp::Matrix< double, structures::UpperTriangular > UA( n );
	alp::Matrix< double, structures::UpperTriangular > UB( n );
	alp::Matrix< double, structures::UpperTriangular > UC( n );

	rc = alp::buildMatrix( UA, A_data.begin(), A_data.end() );
	rc = alp::buildMatrix( UB, B_data.begin(), B_data.end() );
	rc = alp::buildMatrix( UC, C_data.begin(), C_data.end() );

	rc = alp::mxm( UC, UA, UB, ring );

	std::cout << "\n\n=========== Testing Symmetric Output ============\n\n";

	alp::Matrix< double, structures::Symmetric > SC( n );

	rc = alp::buildMatrix( SC, C_data.begin(), C_data.end() );

	rc = alp::mxm( SC, A, alp::get_view< alp::view::transpose >( A ), ring );

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 6;

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

	std::cout << "This is functional test " << argv[ 0 ] << " " << in << "\n";
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

