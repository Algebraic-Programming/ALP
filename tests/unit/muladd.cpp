
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

void grb_program( const size_t &n, grb::RC &rc ) {
	const double alpha = 1.5;
	const double beta = 3.14;
	const double gamma = 2.718;
	grb::Vector< double > a( n );
	grb::Vector< double > x( n );
	grb::Vector< double > y( n );
	grb::Vector< double > z( n );
	rc = grb::set( a, alpha );
	if( rc == SUCCESS ) {
		rc = grb::set( x, beta );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( y, gamma );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	// Test 1: vector-vector-vector-vector
	rc = grb::eWiseMulAdd( z, a, x, y, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test I, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != alpha * beta + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( alpha * beta + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test I failed.\n";
		return;
	}
	grb::clear( z );

	// Test 2: vector-scalar-vector-vector
	rc = grb::eWiseMulAdd( z, alpha, x, y, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test II, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != alpha * beta + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( alpha * beta + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test II failed.\n";
		return;
	}
	grb::clear( z );

	// Test 3: vector-vector-scalar-vector
	rc = grb::eWiseMulAdd( z, a, beta, y, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test III, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != beta * alpha + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( beta * alpha + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test III failed.\n";
		return;
	}
	grb::clear( z );

	// Test 4: vector-vector-vector-scalar
	rc = grb::eWiseMulAdd( z, a, x, gamma, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test IV, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != beta * alpha + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( beta * alpha + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test IV failed.\n";
		return;
	}
	grb::clear( z );

	// Test 5: vector-vector-scalar-scalar
	rc = grb::eWiseMulAdd( z, a, beta, gamma, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test V, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != beta * alpha + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( beta * alpha + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test V failed.\n";
		return;
	}
	grb::clear( z );

	// Test 6: vector-scalar-vector-scalar
	rc = grb::eWiseMulAdd( z, alpha, x, gamma, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test VI, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != beta * alpha + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( beta * alpha + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test VI failed.\n";
		return;
	}
	grb::clear( z );

	// Test 7: vector-scalar-scalar-vector
	rc = grb::eWiseMulAdd( z, alpha, beta, y, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test VII, failed: " << grb::toString( rc ) << "\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != alpha * beta + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( alpha * beta + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test VII failed.\n";
		return;
	}
	grb::clear( z );

	// Test 8: vector-scalar-scalar-scalar
	rc = grb::eWiseMulAdd( z, alpha, beta, gamma, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::eWiseMulAdd, test VIII, failed\n";
		rc = FAILED;
	} else {
		if( grb::nnz( z ) != n ) {
			std::cerr << "Unexpected number of nonzeroes: " << grb::nnz( z ) << ", expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto & pair : z ) {
			if( pair.second != alpha * beta + gamma ) {
				std::cerr << "Unexpected entry ( " << pair.first << ", " << pair.second << " ), expected value " << ( alpha * beta + gamma ) << "\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test VIII failed.\n";
		return;
	}
	grb::clear( z );

	return;
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
