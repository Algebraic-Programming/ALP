
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
#include <complex> // for testing using non-primitive value types

#include <graphblas.hpp>


void grb_program( const size_t &n, grb::RC &rc ) {
	// test default capacities
	{
		if( grb::spmd<>::pid() == 0 ) {
			std::cerr << "\t Testing default capacities on non-empty containers...\n";
		}

		grb::Vector< double > vec( n );
		grb::Matrix< std::pair< size_t, float > > mat( n, 2*n );
		if( grb::capacity( vec ) < n ) {
			std::cerr << "\t vector default capacity is " << grb::capacity(vec) << ", "
				<< "expected " << n << " (or higher)\n";
			rc = grb::FAILED;
		}
		if( grb::capacity( mat ) < 2 * n ) {
			std::cerr << "\t matrix default capacity is " << grb::capacity(mat) << ", "
				<< "expected " << (2*n) << " (or higher)\n";
			rc = grb::FAILED;
		}

		// test capacity after resize
		rc = grb::resize( mat, 3 * n );
		if( rc != grb::SUCCESS ) {
			std::cerr << "\t error during matrix resize (I): "
				<< grb::toString( rc ) << "\n";
		} else {
			if( grb::capacity( mat ) < 3 * n ) {
				std::cerr << "\t matrix capacity after resize to " << (3*n) << " is "
					<< grb::capacity( mat ) << "; expected it to be equal to or higher than "
					<< (3*n) << "\n";
				rc = grb::FAILED;
			}
		}
		if( rc != grb::SUCCESS ) { return; }

		// test illegal resizes
		if( grb::spmd<>::pid() == 0 ) {
			std::cerr << "\t Testing resize to illegal capacities on non-empty containers...\n";
		}

		rc = grb::resize( vec, 2 * n );
		if( rc != grb::ILLEGAL ) {
			std::cerr << "Received " << grb::toString( rc ) << " instead of "
				<< "ILLEGAL (I)\n";
			rc = grb::FAILED;
			return;
		}

		rc = grb::resize( vec, 2 * n * n + 1 );
		if( rc != grb::ILLEGAL ) {
			std::cerr << "Received " << grb::toString( rc ) << " instead of "
				<< "ILLEGAL (I)\n";
			rc = grb::FAILED;
			return;
		}
	}

	// test default capacities for empty containers
	rc = grb::SUCCESS;
	{
		if( grb::spmd<>::pid() == 0 ) {
			std::cerr << "\t Testing default capacities for empty containers...\n";
		}

		grb::Vector< unsigned char > vec( 0 );
		grb::Matrix< void > mat( 0, 0 );
		if( grb::capacity( vec ) != 0 ) {
			std::cerr << "\t vector default capacity is " << grb::capacity(vec) << ", "
				<< "expected 0.\n";
			rc = grb::FAILED;
		}
		if( grb::capacity( mat ) != 0 ) {
			std::cerr << "\t matrix default capacity is " << grb::capacity(mat) << ", "
				<< "expected 0.\n";
			rc = grb::FAILED;
		}
		if( rc != grb::SUCCESS ) { return; }
	}

	// test illegal explcit capacities during container construction
	if( grb::spmd<>::pid() == 0 ) {
		std::cerr << "\t Testing illegal explicit capacities "
			<< "during container construction...\n";
	}
	bool exception_caught = false;
	try {
		grb::Vector< bool > vec( n, 2 * n );
	} catch( const std::runtime_error &e ) {
		const std::string err_str( e.what() );
		if( err_str.compare( toString( grb::ILLEGAL ) ) == 0 ) {
			exception_caught = true;
		} else {
			throw;
		}
	}
	if( !exception_caught ) {
		std::cerr << "\t did not catch grb::ILLEGAL by exception during vector "
			<< "construction with illegal requested capacity\n";
		rc = grb::FAILED;
	}
	exception_caught = false;
	try {
		grb::Matrix< std::complex< float > > mat( 2 * n, n, 2 * n * n + 1 );
	} catch( const std::runtime_error &e ) {
		const std::string err_str( e.what() );
		if( err_str.compare( toString( grb::ILLEGAL ) ) == 0 ) {
			exception_caught = true;
		} else {
			throw;
		}
	}
	if( !exception_caught ) {
		std::cerr << "\t did not catch grb::ILLEGAL by exception during matrix "
			<< "construction with illegal requested capacity\n";
		rc = grb::FAILED;
	}
	if( rc != grb::SUCCESS ) { return; }

	// test explicit capacities during construction
	if( grb::spmd<>::pid() == 0 ) {
		std::cerr << "\t Testing explicit capacities on non-empty containers...\n";
	}

	grb::Vector< double > vec( n, 1 );
	grb::Matrix< std::pair< size_t, float > > mat( n, 2*n, 3*n );
	if( grb::capacity( vec ) == 0 ) {
		std::cerr << "\t vector capacity is " << grb::capacity(vec) << ", "
			<< "expected " << 1 << " (or higher)\n";
		rc = grb::FAILED;
	}
	if( grb::capacity( mat ) < 3 * n ) {
		std::cerr << "\t matrix capacity is " << grb::capacity(mat) << ", "
			<< "expected " << (3*n) << " (or higher)\n";
		rc = grb::FAILED;
	}
	if( rc != grb::SUCCESS ) { return; }

	// prepare for testing clear semantics while resizing to max capacity
	if( grb::spmd<>::pid() == 0 ) {
		std::cerr << "\t Testing resize to max capacity...\n";
	}
	rc = setElement( vec, 3.14, n / 2 );
	if( rc == grb::SUCCESS ) {
		const std::pair< size_t, float > pair = { 7, 3.14 };
		const std::pair< size_t, float > * const start_val = &pair;
		const std::pair< size_t, float > * const end_val = start_val + 1;
		const size_t one = 1;
		const size_t * const start_ind = &one;
		const size_t * const end_ind = start_ind + 1;
		rc = buildMatrixUnique(
			mat,
			start_ind, end_ind,
			start_ind, end_ind,
			start_val, end_val,
			grb::SEQUENTIAL
		);
	}
	if( rc != grb::SUCCESS ||
		grb::nnz( vec ) != 1 ||
		grb::nnz( mat ) != 1
	  ) {
		std::cerr << "\t error during intitialisation of clear-semantics test:\n"
			<< "\t  - rc is " << grb::toString( rc ) << ", expected SUCCESS\n"
			<< "\t  - grb::nnz( vec ) is " << grb::nnz( vec ) << ", expected 1\n"
			<< "\t  - grb::nnz( mat ) is " << grb::nnz( mat ) << ", expected 1\n";
		if( rc != grb::SUCCESS ) { rc = grb::FAILED; }
		return;
	}
	rc = grb::resize( vec, n );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t error during vector resize (I): "
			<< grb::toString( rc ) << "\n";
		rc = grb::FAILED;
	}
	if( grb::capacity( vec ) < n ) {
		std::cerr << "\t vector capacity after resize to " << n << " is "
			<< grb::capacity( vec ) << "; expected it to be equal to or higher than "
			<< n << "\n";
		rc = grb::FAILED;
	}
	if( rc != grb::SUCCESS ) { return; }

	rc = grb::resize( mat, 2 * n * n );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t error during matrix resize (II): "
			<< grb::toString( rc ) << "\n";
		rc = grb::FAILED;
	}
	if( grb::capacity( mat ) < n ) {
		std::cerr << "\t matrix capacity after resize to " << (2*n*n) << " is "
			<< grb::capacity( mat ) << "; expected it to be equal to or higher than "
			<< (2*n*n) << "\n";
		rc = grb::FAILED;
	}
	if( grb::nnz( vec ) != 1 ) {
		std::cerr << "\t vector contains " << grb::nnz( vec ) << " nonzeroes, "
			<< "expected one\n";
		rc = grb::FAILED;
	}
	if( grb::nnz( mat ) != 1 ) {
		std::cerr << "\t matrix contains " << grb::nnz( mat ) << " nonzeroes, "
			<< "expected one\n";
		rc = grb::FAILED;
	}
	if( rc != grb::SUCCESS ) { return; }

	// test resize to zero
	if( grb::spmd<>::pid() == 0 ) {
		std::cerr << "\t Testing resize to zero...\n";
	}

	rc = grb::resize( vec, 0 );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t error during vector resize (II): "
			<< grb::toString( rc ) << "\n";
		rc = grb::FAILED;
	}
	// implementations and backends may or may not resize to smaller capacities, so
	// we only test here if the call doesn't somehow fail. Any returned value is OK
	(void) grb::capacity( vec );
	if( rc != grb::SUCCESS ) { return; }
	rc = grb::resize( mat, 0 );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t error during matrix resize (III): "
			<< grb::toString( rc ) << "\n";
		rc = grb::FAILED;
	}
	// implementations and backends may or may not resize to smaller capacities, so
	// we only test here if the call doesn't somehow fail. Any returned value is OK
	(void) grb::capacity( mat );
	if( rc != grb::SUCCESS ) { return; }

	// done
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
		if( !(ss >> read) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( !ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( (read % 2) != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the test "
			<< "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	const grb::RC launch_rc = launcher.exec( &grb_program, in, out, true );
	if( launch_rc != grb::SUCCESS ) {
		std::cerr << "Launch test failed\n";
		out = launch_rc;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

