
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
#include <graphblas/algorithms/matrix_factory.hpp>
#include <utils/matrix_values_check.hpp>



using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// initialize test
	const grb::Matrix< double > A = factory::eye< double >( n, n, SEQUENTIAL, 1, 1 );
	const grb::Matrix< double > B = factory::identity< double >( n, SEQUENTIAL, 2 );
	grb::Matrix< double > C( n, n );
	grb::Matrix< double > C_expected = factory::eye< double >( n, n, SEQUENTIAL, 2, 1 );

	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	rc = grb::mxm( C, A, B, ring, RESIZE );
	if( rc == SUCCESS ) {
		rc = grb::mxm( C, A, B, ring );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::resize FAILED\n";
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check CRS output
	if( utils::compare_crs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error: unexpected CRS output\n";
	}

	// check CCS output
	if( utils::compare_ccs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error: unexpected CCS output\n";
	}

	// compute with the operator-monoid mxm
	std::cout << "\tVerifying the operator-monoid version of mxm\n";

	rc = grb::mxm(
		C, A, B,
		ring.getAdditiveMonoid(),
		ring.getMultiplicativeOperator(),
		RESIZE
	);
	if( rc == SUCCESS ) {
		rc = grb::mxm(
			C, A, B,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::resize FAILED\n";
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check CRS output
	if( utils::compare_crs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error: unexpected CRS output\n";
	}

	// check CCS output
	if( utils::compare_ccs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error: unexpected CCS output\n";
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
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

