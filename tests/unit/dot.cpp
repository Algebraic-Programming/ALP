
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

	// repeatedly used containers
	grb::Vector< bool > even_mask( n );
	grb::Vector< size_t > temp( n );
	grb::Vector< double > left( n );
	grb::Vector< double > right( n );

	// create even mask
	rc = grb::set< grb::descriptors::use_index >( temp, 0 );
	rc = rc ? rc : grb::eWiseLambda( [&temp] (const size_t i) {
			if( temp[ i ] % 2 == 0 ) {
				temp[ i ] = 1;
			} else {
				temp[ i ] = 0;
			}
		}, temp );
	rc = rc ? rc : grb::set( even_mask, temp, true );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t initialisation of mask FAILED\n";
		return;
	}

	// test 1, init
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	rc = grb::set( left, 1.5 ); // left = 1.5 everywhere
	rc = rc ? rc : grb::set( right, -1.0 );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 1 (dense, regular semiring): initialisation FAILED\n";
		return;
	}
	double out = 2.55;

	// test 1, exec
	rc = grb::dot( out, left, right, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 1 (dense, regular semiring): dot FAILED\n";
		return;
	}

	// test 1, check
	const double expected = 2.55 - static_cast< double >( n + n / 2 );
	if( !utils::equals( out, expected, 2 * n + 1 ) ) {
		std::cerr << "\t test 1 (dense, regular semiring): unexpected output "
			<< "( " << out << ", expected "
			<< ( 2.55 - static_cast< double >(n + n/2) )
			<< " )\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) {
		return;
	}

	// test 2, init
	grb::Semiring<
		grb::operators::add< double >,
		grb::operators::left_assign_if< double, bool, double >,
		grb::identities::zero, grb::identities::logical_true
	> pattern_sum_if;
	rc = grb::clear( left );
	rc = rc ? rc : grb::clear( right );
	rc = rc ? rc : grb::set( left, even_mask, 2.0 );
	rc = rc ? rc : grb::set( right, even_mask, 1.0 );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 2 (sparse, non-standard semiring) "
			<< "initialisation FAILED\n";
		return;
	}
	out = 0;

	// test 2, exec
	rc = grb::dot( out, left, right, pattern_sum_if );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 2 (sparse, non-standard semiring) dot FAILED\n";
		return;
	}

	// test 2, check
	if( !utils::equals( out, static_cast< double >( n ), 2 * n ) ) {
		std::cerr << "\t test 2 (sparse, non-standard semiring), "
			<< "unexpected output: " << out << ", expected " << n
			<< ".\n";
		rc = FAILED;
		return;
	}

	// test 3, init
	grb::Semiring<
		grb::operators::add< int >, grb::operators::mul< int >,
		grb::identities::zero, grb::identities::one
	> intRing;
	grb::Vector< int > x( n ), y( n );
	rc = grb::set( x, 1 );
	rc = rc ? rc : grb::set( y, 2 );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test 3 (dense integer vectors) initialisation FAILED\n";
		return;
	}
	int alpha = 0;

	// test 3, exec
	rc = grb::dot( alpha, x, y, intRing );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test 3 (dense integer vectors) dot FAILED\n";
		return;
	}

	// test 3, check
	if( alpha != 2 * static_cast< int >(n) ) {
		std::cerr << "\t test 3 (dense integer vectors) unexpected value "
			<< alpha << ", expected 2 * n = " << (2*n) << ".\n";
		rc = FAILED;
		return;
	}

	// test 4, init
	grb::Vector< int > empty_left( 0 ), empty_right( 0 );
	// retain old value of alpha

	// test 4, exec
	rc = grb::dot( alpha, empty_left, empty_right, intRing );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 4 (empty vectors) dot FAILED\n";
		return;
	}

	// test 4, check
	if( alpha != 2 * static_cast< int >(n) ) {
		std::cerr << "\t test 4 (empty vectors) unexpected value "
			<< alpha << ", expected 2 * n = " << (2*n) << ".\n";
		rc = FAILED;
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
