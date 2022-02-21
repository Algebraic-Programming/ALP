
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

#include <graphblas.hpp>

using namespace grb;

template< typename Structure >
void print_matrix( const grb::StructuredMatrix< double, Structure > & A) {
	(void)A;
	// if( ! grb::internal::getInitialized< double >( A ) ) {
	// 	std::cout << "Matrix is uninitialized, nothing to print.\n";
	// 	return;
	// }
	// const double * Araw = grb::getRaw( internal::getContainer( A ) );
	// for( size_t row = 0; row < grb::nrows( A ); ++row ) {
	// 	for( size_t col = 0; col < grb::ncols( A ); ++col ) {
	// 		std::cout << Araw[row * grb::ncols( A ) + col] << " ";
	// 	}
	// 	std::cout << "\n";
	// }
}

void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	std::cout << "\tTesting dense mxm\n";
	// initialize test
	grb::StructuredMatrix< double, structures::General > A( n, n );
	grb::StructuredMatrix< double, structures::General > B( n, n );
	grb::StructuredMatrix< double, structures::General > C( n, n );
	std::vector< double > A_data( n * n, 1 );
	std::vector< double > B_data( n * n, 2 );

	double alpha = 10;
	double beta = 20;

	// Initialize input matrices
	rc = grb::buildMatrix( A, A_data.begin(), A_data.end() );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrix( B, B_data.begin(), B_data.end() );
	}
	

	std::cout << "Output matrix nrows = " << nrows( C ) << ", ncols = " << ncols( C ) << "\n";

	// C = A + B
	if( rc == SUCCESS ) {
		rc = grb::eWiseApply( C, A, B, ring.getAdditiveMonoid());
	}

	// C = alpha . B
	if ( rc == SUCCESS ) {
		rc = grb::eWiseApply( C, alpha, B, ring.getMultiplicativeMonoid());
	}

	// C = A . beta
	if( rc == SUCCESS ) {
		rc = grb::eWiseApply( C, A, beta, ring.getMultiplicativeMonoid());
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

