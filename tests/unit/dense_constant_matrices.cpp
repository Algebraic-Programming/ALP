
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

void alp_program( const size_t & n, alp::RC & rc ) {
	alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

	std::cout << "\tTesting dense Identity and Zero matrices\n";
	// initialize test
	rc = SUCCESS;
	//alp::Matrix< double, structures::Square > A( n );
	//alp::Matrix< double, structures::Square > C( n );
	const auto I = alp::structures::constant::I< double >( n );
	std::cout << "I(0, 0) = " << internal::access( I, internal::getStorageIndex( I, 0, 0 ) ) << "\n";
	std::cout << "I(1, 0) = " << internal::access( I, internal::getStorageIndex( I, 1, 0 ) ) << "\n";
	auto Zero = alp::structures::constant::Zero< double >( n, n );
	std::cout << "Zero(0, 0) = " << internal::access( Zero, internal::getStorageIndex( Zero, 0, 0 ) ) << "\n";
	std::cout << "Zero(1, 0) = " << internal::access( Zero, internal::getStorageIndex( Zero, 1, 0 ) ) << "\n";

	// Initialize input matrix
	//std::vector< double > A_data( n * n, 1 );
	//rc = alp::buildMatrix( A, A_data.begin(), A_data.end() );

	//TODO: These should forward to alp::set
	// if( rc == SUCCESS ) {
	// 	alp::mxm( C, A, I, ring );
	// 	// C should be equal to A
	// }

	// if (rc == SUCCESS ) {
	// 	alp::mxm( C, A, Zero, ring );
	// 	// C should be a zero
	// }
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
