
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

#include "graphblas.hpp"


// test strategy: construct a sparse vector, then use it for various operations
// together with the dense descriptor. This should result in an ILLEGAL return
// code, which may either be found when executing the primitive (for blocking
// backends), or when executing grb::wait (for nonblocking backends).
void grbProgram( const int &n, grb::RC &rc ) {
	// prepare test
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	grb::Monoid<
		grb::operators::add< double >,
		grb::identities::zero
	> addMon;
	grb::Vector< double > x( n );
	double alpha = 0.0;
	rc = grb::setElement( x, 3.14, 0 );

	// test one: grb::wait without arguments
	{
		rc = rc ? rc : grb::foldl< grb::descriptors::dense >( alpha, x, addMon );
		const grb::RC wait_rc = grb::wait();
		if( rc == grb::SUCCESS ) {
			if( wait_rc != grb::ILLEGAL ) {
				std::cerr << "Test FAILED: an ILLEGAL operation was requested that was not "
					<< "caught by grb::foldl nor by a following grb::wait()\n";
				rc = grb::FAILED;
			} else {
				std::cout << "\t Test INFO: ILLEGAL detected by grb::wait()\n";
			}
		} else {
			if( rc == grb::ILLEGAL ) {
				std::cout << "\t Test INFO: ILLEGAL detected by grb::foldl\n";
				rc = grb::SUCCESS;
			} else {
				std::cerr << "Test FAILED: call to grb::foldl (vector to scalar) returned "
					<< grb::toString( rc ) << ", expected ILLEGAL or SUCCESS\n";
			}
		}
	}

	// test two: grb::wait with vector container
	if( rc == grb::SUCCESS ) {
		grb::Vector< double > y( n );
		rc = rc ? rc : grb::set< grb::descriptors::dense >( y, x );
		const grb::RC wait_rc = grb::wait( y );
		if( rc == grb::SUCCESS ) {
			if( wait_rc != grb::ILLEGAL ) {
				std::cerr << "Test FAILED: an ILLEGAL operation was requested that was not "
					<< "caught by grb::set nor by a following grb::wait()\n";
				rc = grb::FAILED;
			} else {
				std::cout << "\t Test INFO: ILLEGAL detected by grb::wait( vector )\n";
			}
		} else {
			if( rc == grb::ILLEGAL ) {
				std::cout << "\t Test INFO: ILLEGAL detected by grb::set\n";
				rc = grb::SUCCESS;
			} else {
				std::cerr << "Test FAILED: call to grb::set (vector to vector) returned "
					<< grb::toString( rc ) << ", expected ILLEGAL or SUCCESS\n";
			}
		}
	}

	// test three: grb::wait with multiple containers, including a matrix
	if( rc == grb::SUCCESS ) {
		size_t zero = 0;
		const auto zero_p = &zero;
		const auto alpha_p = &alpha;
		grb::Matrix< double > A( n, n );
		grb::Vector< double > y( n );
		rc = rc ? rc : grb::set( y, 1.0 );
		rc = rc ? rc : grb::buildMatrixUnique(
				A,
				zero_p, zero_p + 1,
				zero_p, zero_p + 1,
				alpha_p, alpha_p + 1,
				grb::SEQUENTIAL
			);
		rc = rc ? rc : grb::mxv< grb::descriptors::dense >( y, A, x, ring );
		const grb::RC wait_rc = grb::wait( y, A );
		if( rc == grb::SUCCESS ) {
			if( wait_rc != grb::ILLEGAL ) {
				std::cerr << "Test FAILED: an ILLEGAL operation was requested that was not "
					<< "caught by the operator sequence nor by a following grb::wait()\n";
				rc = grb::FAILED;
			} else {
				std::cout << "\t Test INFO: ILLEGAL detected by multi-variate mixed vector "
					<< " and matrix container grb::wait\n";
			}
		} else {
			if( rc == grb::ILLEGAL ) {
				std::cout << "\t Test INFO: ILLEGAL detected by operator sequence\n";
				rc = grb::SUCCESS;
			} else {
				std::cerr << "Test FAILED: operation sequence returned "
					<< grb::toString( rc ) << ", expected ILLEGAL or SUCCESS\n";
			}
		}
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	int input = 100; // unused

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		int read;
		std::istringstream ss( argv[ 1 ] );
		if( !(ss >> read) || !ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read <= 0 ) {
			std::cerr << "Given value for n is smaller than one\n";
			printUsage = true;
		} else {
			input = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " (n)\n";
		std::cerr << "\tn is an optional integer with value 1 or higher. "
			<< "Default is 100.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grbProgram, input, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}

