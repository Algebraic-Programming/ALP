
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
#include <alp/algorithms/gemm.hpp>

using namespace alp;

void alp_program( const size_t & unit, alp::RC & rc ) {
	alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

	std::cout << "\tTesting ALP gemm_like_example\n"
	             "\tC = alpha * A * B + beta * C\n";

	// dimensions of matrices A, B and C
	size_t M = 10 * unit;
	size_t N = 20 * unit;
	size_t K = 30 * unit;

	// dimensions of views over A, B and C
	size_t m = unit;
	size_t n = 2 * unit;
	size_t k = 3 * unit;

	alp::Matrix< double, structures::General > A( M, K );
	alp::Matrix< double, structures::General > B( K, N );
	alp::Matrix< double, structures::General > C( M, N );

	Scalar< double > alpha( 0.5 );
	Scalar< double > beta( 1.5 );

	rc = algorithms::gemm_like_example(
		m, n, k,
		alpha,
		A, 1, 1, 1, 1,
		B, 2, 1, 2, 4,
		beta,
		C, 0, 0, 1, 1,
		ring );
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

