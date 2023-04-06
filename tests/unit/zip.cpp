
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

void grb_program( const size_t & n, grb::RC & rc ) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	grb::Vector< double > left( n ), chk1( n );
	grb::Vector< int > right( n ), chk2( n );
	grb::Vector< std::pair< double, int > > out( n );
	rc = grb::set( left, 1.5 ); // left = 1.5 everywhere
	if( rc == SUCCESS ) {
		rc = grb::set( right, 2 );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	rc = grb::zip( out, left, right );
	if( rc != SUCCESS ) {
		std::cerr << "\t zip FAILED\n";
		return;
	}
	if( nnz( out ) != n ) {
		std::cerr << "\t unexpected number of nonzeroes ( " << nnz( out )
			<< ", expected " << n << " )\n";
		rc = FAILED;
	}
	for( const auto &pair : out ) {
		const std::pair< double, int > & out = pair.second;
		if( out.first != 1.5 || out.second != 2 ) {
			std::cerr << "\t unexpected output "
				<< "( " << pair.first << ", < " << out.first << ", "
				<< out.second << " > ), expected " << pair.first << ", "
				<< "< 1.5, 2 > )\n";
			rc = FAILED;
		}
	}
	if( rc == FAILED ) {
		return;
	}

	rc = grb::unzip( chk1, chk2, out );
	if( rc != SUCCESS ) {
		std::cerr << "\t unzip FAILED\n";
		return;
	}
	if( nnz( chk1 ) != n ) {
		std::cerr << "\t unexpected number of nonzeroes ( " << nnz( chk1 ) << ", "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	if( nnz( chk2 ) != n ) {
		std::cerr << "\t unexpected number of nonzeroes ( " << nnz( chk2 ) << ", "
			<< "expected " << n << "\n";
		rc = FAILED;
	}
	for( const auto &pair : chk1 ) {
		if( pair.second != 1.5 ) {
			std::cerr << "\t unexpected output ( " << pair.first << ", " << pair.second
				<< " ), expected " << pair.first << ", 1.5 )\n";
			rc = FAILED;
		}
	}
	for( const auto &pair : chk2 ) {
		if( pair.second != 2 ) {
			std::cerr << "\t unexpected output ( " << pair.first << ", " << pair.second
				<< " ), expected " << pair.first << ", 2 )\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

// TODO issue #66
#ifndef _GRB_WITH_LPF
	// matrix tests
	grb::Vector< size_t > I( n );
	grb::Vector< size_t > J( n );
	grb::Vector< double > V( n );
	grb::Matrix< double > A( n, n );
	rc = grb::set< grb::descriptors::use_index >( I, 0 );
	if( rc == SUCCESS ) {
		rc = grb::set( J, 1 );
	}
	if( rc == SUCCESS ) {
		rc = grb::setElement( J, n / 2, n / 2 );
	}
	if( rc == SUCCESS ) {
		rc = grb::set( V, 1 );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize( A, n );
	}
	if( rc != SUCCESS ) {
		std::cout << "grb::zip to matrix: initialisation FAILED\n";
	} else {
		rc = grb::zip( A, I, J, V );
	}
	if( rc != SUCCESS ) {
		std::cout << "grb::zip to matrix (non-void) FAILED with error "
			<< grb::toString( rc ) << "\n";
	} else {
		if( grb::nnz( A ) != n ) {
			std::cout << "\t got " << grb::nnz( A ) << " matrix nonzeroes, "
				<< "expected " << n << "\n";
			rc = FAILED;
		}
		// check via grb::mxv
		(void) grb::set( right, 1 );
		(void) grb::clear( left );
		(void) grb::vxm( left, right, A, ring );
		if( grb::nnz( left ) != 2 ) {
			std::cout << "\t got " << grb::nnz( left ) << " nonzeroes in output vector, "
				<< "expected 2\n";
			rc = FAILED;
		}
		for( const auto &pair : left ) {
			if( pair.first == 1 ) {
				const double expect = n - 1 + ( 1 == n / 2 ? 1 : 0 );
				if( pair.second != expect ) {
					std::cout << "\t got " << pair.second << " nonzeroes in column "
						<< pair.first << ", expected " << expect << "\n";
					rc = FAILED;
				}
			} else if( pair.first == n / 2 && n / 2 != 1 ) {
				if( pair.second != 1 ) {
					std::cout << "\t got " << pair.second << " nonzeroes in column "
						<< pair.first << ", expected 1\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != 0 ) {
					std::cout << "\t got " << pair.second << " nonzeroes in column "
						<< pair.first << ", expected none\n";
					rc = FAILED;
				}
			}
		}
		(void) grb::clear( left );
		(void) grb::mxv( left, A, right, ring );
		if( grb::nnz( left ) != n ) {
			std::cout << "\t got " << grb::nnz( left ) << " nonzeroes in output vector, "
				<< "expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto &pair : left ) {
			if( pair.second != 1 ) {
				std::cout << "\t got unexpected entry ( " << pair.first << ", "
					<< pair.second << " ), expected value 1.\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}

	grb::Matrix< void > A_void( n, n );
	rc = grb::resize( A_void, n );
	if( rc == SUCCESS ) {
		rc = grb::zip( A_void, I, J );
	}
	if( rc != SUCCESS ) {
		std::cout << "grb::zip to matrix (void) FAILED with error " << grb::toString( rc )
			<< "\n";
	} else {
		if( grb::nnz( A_void ) != n ) {
			std::cout << "\t got " << grb::nnz( A_void ) << " matrix nonzeroes, "
				<< "expected " << n << "\n";
			rc = FAILED;
		}
		// check via grb::mxv
		(void) grb::set( right, 1 );
		(void) grb::clear( left );
		(void) grb::vxm( left, right, A_void, ring );
		if( grb::nnz( left ) != 2 ) {
			std::cout << "\t got " << grb::nnz( left ) << " nonzeroes in output vector, "
				<< "expected 2\n";
			rc = FAILED;
		}
		for( const auto &pair : left ) {
			if( pair.first == 1 ) {
				const double expect = n - 1 + ( 1 == n / 2 ? 1 : 0 );
				if( pair.second != expect ) {
					std::cout << "\t got " << pair.second << " nonzeroes in column "
						<< pair.first << ", expected " << expect << "\n";
					rc = FAILED;
				}
			} else if( pair.first == n / 2 && n / 2 != 1 ) {
				if( pair.second != 1 ) {
					std::cout << "\t got " << pair.second << " nonzeroes in column "
						<< pair.first << ", expected 1\n";
					rc = FAILED;
				}
			} else {
				if( pair.second != 0 ) {
					std::cout << "\t got " << pair.second << " nonzeroes in column "
						<< pair.first << ", expected none\n";
					rc = FAILED;
				}
			}
		}
		(void) grb::clear( left );
		(void) grb::mxv( left, A_void, right, ring );
		if( grb::nnz( left ) != n ) {
			std::cout << "\t got " << grb::nnz( left ) << " nonzeroes in output vector, "
				<< "expected " << n << "\n";
			rc = FAILED;
		}
		for( const auto &pair : left ) {
			if( pair.second != 1 ) {
				std::cout << "\t got unexpected entry ( " << pair.first << ", "
					<< pair.second << " ): expected value 1.\n";
				rc = FAILED;
			}
		}
	}
	if( rc != SUCCESS ) {
		return;
	}
#endif

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
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")\n"
			<< std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

