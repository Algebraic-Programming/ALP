
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
	grb::Vector< double > left( n );
	grb::Vector< double > right( n );
	rc = grb::set( left, 1.5 ); // left = 1.5 everywhere
	if( rc == SUCCESS ) {
		rc = grb::set( right, -1.0 );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	double out = 2.55;
	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;
	rc = grb::dot( out, left, right, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t dot FAILED\n";
		return;
	}
	const double expected = -static_cast< double >( n + n / 2 );
	if( out != expected ) {
		std::cerr << "\t unexpected output ( " << out << ", expected " << ( -( n + n / 2 ) ) << " )\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) {
		return;
	}

	out = 2.55;
	grb::Semiring< grb::operators::add< double >, grb::operators::left_assign_if< double, bool, double >, grb::identities::zero, grb::identities::logical_true > pattern_sum_if;
	rc = grb::clear( left );
	rc = rc ? rc : grb::clear( right );
	for( size_t i = 0; 2 * i < n; ++i ) {
		rc = rc ? rc : grb::setElement( left, 2.0, 2 * i );
		rc = rc ? rc : grb::setElement( right, 1.0, 2 * i );
	}
	rc = rc ? rc : grb::dot( out, left, right, pattern_sum_if );
	if( rc != SUCCESS ) {
		std::cerr << "\t test (sparse, non-standard semiring) FAILED\n";
	} else {
		if( out != static_cast< double >( n ) ) {
			std::cerr << "\t unexpected output (sparse, non-standard semiring): " << out << ", expected " << n << "\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		return;
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
